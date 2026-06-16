"""SpecDVFS experiment orchestrator.

Runs the energy/latency measurement matrix for SpecDVFS and writes one JSON
result file per run. Use the same script for the small pilot and the full sweep:

    python experiments/run_experiment.py --mode pilot     # 12-run validation
    python experiments/run_experiment.py --mode full      # the full matrix
    python experiments/run_experiment.py --mode pilot --mock   # laptop dry-run, no GPU

Design notes
------------
* Models are LOADED ONCE per (model_pair, strategy) and all DVFS conditions /
  datasets are swept under that single load — model loading is the expensive part.
* CRASH-SAFE + RESUMABLE: each run writes results/<mode>/<key>.json immediately;
  a run whose file already exists is skipped, so you can restart after an interruption.
* GPU clocks are ALWAYS reset on exit (and on crash) via the finally block.
* --mock stubs out vLLM and NVML so the entire control loop, energy-API handling,
  JSON schema, and resume logic can be exercised on a laptop before spending GPU credit.

THREE SEAMS depend on the vLLM source study + your friend's monkey-patch. They are
marked  >>> TODO (PATCH SEAM)  and are the only things you fill after the patch lands:
  1. build_llm()            — the vLLM 0.6.6 speculative_config for each SD strategy.
  2. apply_patch_to()       — getting the SpecDecode worker + applying the patch.
  3. read_alpha_and_tokens()— reading the acceptance rate (blank [H]) + token counts.
Everything else is final.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import time
from collections import Counter
from itertools import groupby
from pathlib import Path

# Make the project root importable when run as `python experiments/run_experiment.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the SAME templates/prefix the datasets were prepared with (DRY).
from data.prepare_datasets import PROMPT_TEMPLATES, CNN_DM_PREFIX  # noqa: E402

# =============================================================================
# CONFIG
# =============================================================================

# Confirmed RTX 3090 clocks (a chosen low level + the sustainable max under lock).
F_HIGH = 1935
F_LOW = 735

SEED = 42

# Experiment controls (non-negotiable — see handoff §12).
TEMP_LIMIT_C = 75      # do not start a run until GPU temp is below this
WARMUP_PROMPTS = 5     # warm-up generations before CodeCarbon starts (not measured)
COOLDOWN_S = 30        # idle gap between measured runs
MONITOR_HZ = 100       # NVML background sampler: 100 Hz = every 10 ms
MAX_NEW_TOKENS = 256   # generation cap per prompt (keeps pilot bounded)

SAMPLED_DIR = PROJECT_ROOT / "data" / "sampled_indices"
RESULTS_DIR = PROJECT_ROOT / "results"

# model_pair -> target/draft repos + prompt-template family.
# RE-SCOPED: the project goal is energy savings in speculative decoding via the
# draft/verify phase asymmetry (with alpha + entropy as DVFS signals), NOT
# reproducing Dutta et al. So the headline pairs are modern, well-supported, cleanly
# matched draft/target families that clearly exhibit the asymmetry. Vicuna-13B/68M is
# kept only as an OPTIONAL legacy/architecture-diversity point (it was the old
# Dutta-replication pair), not a primary target.
MODEL_PAIRS = {
    # --- primary (modern, used in the full sweep by default) ---
    "qwen3_8b_0p6b": {            # ungated; clean Qwen3 draft/target match
        "target": "Qwen/Qwen3-8B",
        "draft": "Qwen/Qwen3-0.6B",
        "family": "qwen3",
    },
    "llama_8b_1b": {              # gated (needs HF auth); the standard modern SD pair
        "target": "meta-llama/Llama-3.1-8B-Instruct",
        "draft": "meta-llama/Llama-3.2-1B-Instruct",
        "family": "llama3",
    },
    # --- optional legacy (NOT in the default full matrix) ---
    "vicuna13b_68m": {
        "target": "lmsys/vicuna-13b-v1.3",
        "draft": "double7/vicuna-68m",
        "family": "vicuna",
    },
}

# Pairs actually swept in --mode full. Add "vicuna13b_68m" here for a legacy datapoint.
PRIMARY_PAIRS = ["qwen3_8b_0p6b", "llama_8b_1b"]

# dataset_key -> (sampled JSON file, prefix prepended before templating)
DATASETS = {
    "gsm8k": ("gsm8k_256.json", ""),
    "humaneval": ("humaneval_164.json", ""),
    "cnndm": ("cnndm_256.json", CNN_DM_PREFIX),
}

# SD strategies. RE-SCOPED around the phase-asymmetry insight rather than Dutta's
# COGA/DYGA-20. Each entry is a config the build_llm() seam turns into a vLLM
# speculative_config:
#   vanilla       -> no speculation (the energy/latency reference point).
#   spec_g{K}     -> standard separate-draft SD with a CONSTANT draft length K.
#                    K is swept because it sets the draft/verify TIME RATIO: larger K =
#                    bigger memory-bound draft fraction = more room for per-phase DVFS to
#                    save energy. Sweeping it tests how the savings scale with the
#                    asymmetry — the core generality claim. {3,5,7} spans the realistic
#                    production range; add 10/20 for a high-gamma stress test.
#   spec_dyn      -> dynamic draft length (cap given), a realism check that the method
#                    still works when the draft length varies per step.
#   eagle3        -> OPTIONAL: EAGLE self-draft (different draft profile). Needs an EAGLE
#                    head per TARGET model (not the small draft above) — source separately.
STRATEGIES = {
    "vanilla": {"kind": "vanilla"},
    "spec_g3": {"kind": "constant", "gamma": 3},
    "spec_g5": {"kind": "constant", "gamma": 5},
    "spec_g7": {"kind": "constant", "gamma": 7},
    "spec_dyn": {"kind": "dynamic", "gamma_max": 7},
    "eagle3": {"kind": "eagle"},   # optional; not in the default full sweep
}
# Strategies actually swept in --mode full (eagle3 excluded by default — head dependency).
SD_STRATEGIES_FULL = ["vanilla", "spec_g3", "spec_g5", "spec_g7", "spec_dyn"]

# DVFS conditions. How each drives the controller is the contract the patch honors
# (it branches on `controller.mode`):
#   off             -> controller.enabled=False; GPU at default clock. (SD-without-DVFS reference)
#   fixed_low       -> global lock to f_low for the whole run; patch no-ops. (naive "just downclock" baseline)
#   two_level       -> on_draft_start() [f_low]; on_verify_start(alpha=1.0) [forces f_high]. (static phase-aware)
#   adaptive_alpha  -> on_draft_start(); on_verify_start(alpha=tracker.estimate); record α. (alpha signal)
#   adaptive_entropy-> on_draft_start(); on_verify_start_entropy(H, a, b); record α.          (entropy signal)
#                      a,b come from configure_dvfs (per-pair calibration if present, else GELATO baseline).
#   coarse          -> one fixed frequency per time window (~100 ms) from recent average behavior.
#                      GRANULARITY ABLATION: a window spans several SD iterations, so it CANNOT see the
#                      within-iteration draft/verify asymmetry. Comparing per-phase control against it
#                      isolates whether the asymmetry — not "DVFS in general" — produces the savings.
#                      (Coarse-window DVFS is the general class; PELM is one instance, but a different setup.)
# Both signal conditions (alpha + entropy) are FIRST-CLASS goals, so both are core here.
FULL_CONDITIONS = ["off", "fixed_low", "two_level", "adaptive_alpha", "adaptive_entropy", "coarse"]

# Entropy-signal coefficients for α = a·exp(b·H). Provisional GELATO baseline until
# per-pair calibration exists (Phase 3); the full study loads a calibrated fit if present.
GELATO_BASELINE_AB = (1.0, -0.35)
CALIBRATION_DIR = PROJECT_ROOT / "calibration"


def entropy_coeffs_for(model_pair):
    """(a, b) for the entropy->alpha map: per-pair calibrated fit if present, else GELATO baseline."""
    f = CALIBRATION_DIR / f"fitted_{model_pair}.json"
    if f.exists():
        try:
            from calibration.entropy_calibration import CalibrationResult
            r = CalibrationResult.from_json(str(f))
            return r.a, r.b
        except Exception:
            pass
    return GELATO_BASELINE_AB


# ---- PILOT definition -------------------------------------------------------
# RE-SCOPED: validate the pipeline + BOTH signals at one realistic operating point on a
# modern, ungated pair. Qwen3 avoids HF gating; gamma=5 is a typical production setting;
# GSM-8K gives clear SD dynamics. The point is "does per-phase DVFS save energy within SD
# at minimal latency loss, and do both signals behave sanely + toggle the clock" — NOT
# reproducing any prior anomaly number.
PILOT_MODEL = "qwen3_8b_0p6b"
PILOT_STRATEGY = "spec_g5"        # one representative constant-gamma operating point
PILOT_DATASET = "gsm8k"
PILOT_N_PROMPTS = 64              # subset of the 256 GSM-8K prompts
PILOT_REPS = 3
FULL_REPS = 5

# Pilot matrix as explicit (model_pair, strategy, condition, dataset) tuples.
PILOT_MATRIX = [
    (PILOT_MODEL, "vanilla",        "off",              PILOT_DATASET),  # reference
    (PILOT_MODEL, PILOT_STRATEGY,   "off",              PILOT_DATASET),  # SD without DVFS (the thing we save against)
    (PILOT_MODEL, PILOT_STRATEGY,   "two_level",        PILOT_DATASET),  # static phase-aware DVFS
    (PILOT_MODEL, PILOT_STRATEGY,   "adaptive_alpha",   PILOT_DATASET),  # alpha signal
    (PILOT_MODEL, PILOT_STRATEGY,   "adaptive_entropy", PILOT_DATASET),  # entropy signal (provisional GELATO coeffs)
    # ^ Drop the adaptive_entropy row for a first pilot if the patch doesn't yet support
    #   entropy mode (it needs on_verify_start_entropy + the EntropyCollector wired in).
]


def build_full_matrix():
    """All (model_pair, strategy, condition, dataset) combos for the full sweep.

    Rule: vanilla has no draft/verify split, so it only runs under `off` (default clock).
    SD strategies run under every DVFS condition.
    """
    combos = []
    for mp in PRIMARY_PAIRS:
        for strat in SD_STRATEGIES_FULL:
            conditions = ["off"] if strat == "vanilla" else FULL_CONDITIONS
            for cond in conditions:
                for ds in DATASETS:
                    combos.append((mp, strat, cond, ds))
    return combos


# =============================================================================
# PROMPTS
# =============================================================================

def load_prompts(dataset_key, family, n=None):
    fname, prefix = DATASETS[dataset_key]
    data = json.loads((SAMPLED_DIR / fname).read_text(encoding="utf-8"))
    samples = data["samples"]
    if n is not None:
        samples = samples[:n]
    tmpl = PROMPT_TEMPLATES[family]
    return [tmpl.format(text=prefix + s["text"]) for s in samples]


# =============================================================================
# GPU MONITOR  (10 ms NVML sampler — the evidence that DVFS actually toggles clocks)
# =============================================================================

class GpuMonitor(threading.Thread):
    def __init__(self, hz=MONITOR_HZ, mock=False):
        super().__init__(daemon=True)
        self.interval = 1.0 / hz
        self.mock = mock
        self._stop = threading.Event()
        self.samples = []  # (ts, gpu_util%, mem_util%, power_mw, sm_clock_mhz, temp_c)

    def run(self):
        if self.mock:
            self._stop.wait()
            return
        import pynvml as N
        N.nvmlInit()
        h = N.nvmlDeviceGetHandleByIndex(0)
        while not self._stop.is_set():
            try:
                u = N.nvmlDeviceGetUtilizationRates(h)
                self.samples.append((
                    time.time(), u.gpu, u.memory,
                    N.nvmlDeviceGetPowerUsage(h),
                    N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_GRAPHICS),
                    N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU),
                ))
            except N.NVMLError:
                pass
            self._stop.wait(self.interval)
        N.nvmlShutdown()

    def stop_and_summary(self):
        self._stop.set()
        self.join(timeout=3)
        if not self.samples:
            return {}
        clocks = [s[4] for s in self.samples]
        powers = [s[3] for s in self.samples]
        temps = [s[5] for s in self.samples]
        return {
            "n_samples": len(self.samples),
            "sm_clock_min_mhz": min(clocks),
            "sm_clock_max_mhz": max(clocks),
            "sm_clock_unique_mhz": sorted(set(clocks)),
            "sm_clock_hist": dict(Counter(clocks).most_common(8)),
            "power_mw_mean": sum(powers) / len(powers),
            "temp_c_max": max(temps),
        }


# =============================================================================
# ENERGY METER  (CodeCarbon wrapper — handles the 3.x stop()-returns-float change)
# =============================================================================

class EnergyMeter:
    def __init__(self, mock=False):
        self.mock = mock
        self._t = None

    def start(self):
        if self.mock:
            self._t0 = time.time()
            return
        from codecarbon import EmissionsTracker
        self._t = EmissionsTracker(
            measure_power_secs=1, save_to_file=False,
            log_level="error", tracking_mode="machine",
        )
        self._t.start()

    def stop(self):
        if self.mock:
            return {"total_kwh": 1e-4, "gpu_kwh": 8e-5, "cpu_kwh": 1e-5, "ram_kwh": 1e-5}
        try:
            self._t.stop()  # in CodeCarbon 3.x this returns a float, not a data object
        except Exception:
            pass

        def kwh(attr):
            try:
                return float(getattr(self._t, attr).kWh)
            except Exception:
                return None

        return {
            "total_kwh": kwh("_total_energy"),
            "gpu_kwh": kwh("_total_gpu_energy"),
            "cpu_kwh": kwh("_total_cpu_energy"),
            "ram_kwh": kwh("_total_ram_energy"),
        }


# =============================================================================
# TEMP GATE + CLOCK RESET
# =============================================================================

def wait_for_temp(limit=TEMP_LIMIT_C, mock=False, timeout=300):
    if mock:
        return
    import pynvml as N
    N.nvmlInit()
    h = N.nvmlDeviceGetHandleByIndex(0)
    t0 = time.time()
    while True:
        t = N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU)
        if t < limit or (time.time() - t0) > timeout:
            break
        time.sleep(5)
    N.nvmlShutdown()


def reset_clocks(mock=False):
    if mock:
        return
    try:
        import pynvml as N
        N.nvmlInit()
        N.nvmlDeviceResetGpuLockedClocks(N.nvmlDeviceGetHandleByIndex(0))
        N.nvmlShutdown()
    except Exception as e:
        print("    WARN: could not reset clocks:", e)


def resolve_clocks(mock=False, allow_mismatch=False):
    """Validate the hardcoded F_HIGH/F_LOW against the GPU actually present (DOUBT 5).

    F_HIGH/F_LOW are NOT auto-derivable as the GPU's min/max: f_low (735) is a chosen
    low level, and f_high (1935) is the *sustainable lock ceiling* that had to be
    discovered (a 2100 request read back as 1935). They are also kept identical across
    every run so energy numbers stay comparable. So we hardcode the confirmed values
    and, on a real GPU, only VALIDATE them here:
      * snap each anchor to the nearest actually-supported clock level (the GPU has 127);
      * if the anchors sit far outside this GPU's range (i.e. you booted a DIFFERENT
        card), ABORT — running with the wrong clocks would silently produce energy
        numbers that can't be compared to the rest of the study. Pass
        --allow-clock-mismatch to auto-pick (max + ~38%) instead, then RE-CONFIRM the
        sustainable f_high by hand before trusting the numbers.
    Returns the (possibly snapped) (f_high, f_low) to use.
    """
    if mock:
        return F_HIGH, F_LOW
    import pynvml as N
    N.nvmlInit()
    h = N.nvmlDeviceGetHandleByIndex(0)
    mem = N.nvmlDeviceGetSupportedMemoryClocks(h)
    gfx = sorted(N.nvmlDeviceGetSupportedGraphicsClocks(h, mem[0]))
    name = N.nvmlDeviceGetName(h)
    name = name.decode() if isinstance(name, bytes) else name
    N.nvmlShutdown()

    lo, hi = gfx[0], gfx[-1]
    snap = lambda f: min(gfx, key=lambda g: abs(g - f))
    fh, fl = snap(F_HIGH), snap(F_LOW)
    far = (abs(fh - F_HIGH) > 50) or (abs(fl - F_LOW) > 50)

    print(f"    GPU: {name}  | supported {len(gfx)} levels, {lo}-{hi} MHz")
    print(f"    anchors: f_high {F_HIGH}->{fh}  f_low {F_LOW}->{fl} (snapped to supported)")

    if far and not allow_mismatch:
        raise SystemExit(
            f"\nABORT (DOUBT 5): this GPU's clocks ({lo}-{hi} MHz) do not match the "
            f"confirmed RTX 3090 anchors (f_high={F_HIGH}, f_low={F_LOW}).\n"
            "You likely booted a different card. Energy numbers would not be comparable.\n"
            "Re-confirm sustainable f_high/f_low for THIS GPU, update the constants, or "
            "re-run with --allow-clock-mismatch to auto-pick (and then re-confirm by hand)."
        )
    if far and allow_mismatch:
        fh, fl = hi, snap(int(round(0.38 * hi)))
        print(f"    --allow-clock-mismatch: auto-picked f_high={fh}, f_low={fl} "
              "(RE-CONFIRM sustainable f_high by hand!)")
    return fh, fl


# =============================================================================
# CONTROLLER + PATCH  (SEAMS 1-3)
# =============================================================================

def make_controller(mock=False):
    if mock:
        from controller.core import SimulatedDVFSController
        return SimulatedDVFSController(F_HIGH, F_LOW, enabled=True)
    # Server: your friend's NVML subclass that overrides set_frequency_mhz().
    from vllm_hooks.dvfs_controller import DVFSController
    return DVFSController(F_HIGH, F_LOW, enabled=True)


class _MockLLM:
    """Stand-in for vLLM.LLM in --mock runs."""
    def generate(self, prompts, *_a, **_k):
        class _Out:
            class _O:
                token_ids = list(range(50))
            outputs = [_O()]
        return [_Out() for _ in prompts]


def build_llm(model_pair, strategy, mock=False):
    if mock:
        return _MockLLM()

    from vllm import LLM
    target = MODEL_PAIRS[model_pair]["target"]
    draft = MODEL_PAIRS[model_pair]["draft"]
    cfg = STRATEGIES[strategy]

    # NF4 4-bit target via bitsandbytes; greedy decoding is set per-request below.
    common = dict(
        model=target,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        dtype="auto",
        gpu_memory_utilization=0.90,
        seed=SEED,
    )
    if cfg["kind"] == "vanilla":
        return LLM(**common)

    # >>> TODO (PATCH SEAM 1): confirm the exact vLLM 0.6.6 speculative_config.
    #   constant -> standard separate-draft SD; the clear case:
    #               speculative_model=draft, num_speculative_tokens=cfg["gamma"]
    #               (confirm those EngineArgs names exist in 0.6.6)
    #   dynamic  -> OPEN: 0.6.6 may not expose adaptive draft length natively
    #               (speculative_disable_by_batch_size is NOT dynamic gamma). The source
    #               study decides how/if this is realized; until then it falls back to a
    #               constant length of cfg["gamma_max"].
    #   eagle    -> needs an EAGLE head trained for THIS target (not `draft`), plus the
    #               EAGLE method/type string. Source the head per target before enabling.
    if cfg["kind"] == "constant":
        spec = dict(speculative_model=draft, num_speculative_tokens=cfg["gamma"])
    elif cfg["kind"] == "dynamic":
        spec = dict(speculative_model=draft, num_speculative_tokens=cfg["gamma_max"])  # + dynamic knob: TODO
    elif cfg["kind"] == "eagle":
        raise NotImplementedError(
            "eagle3 needs an EAGLE head per target model + the EAGLE method string (SEAM 1)."
        )
    else:
        raise NotImplementedError(f"unknown strategy kind: {cfg['kind']}")
    return LLM(**common, **spec)


def apply_patch_to(llm, controller, mock=False):
    if mock:
        return
    from vllm_hooks.patch_spec_decode import apply_patch
    # >>> TODO (PATCH SEAM 2): get the SpecDecode worker and apply the patch.
    #     The patch must run INSIDE each GPU worker process (handoff §8.4 — NVML is
    #     not fork-safe). With the LLM() API the worker is reachable via something
    #     like llm.llm_engine.model_executor.driver_worker — confirm for 0.6.6, and
    #     confirm whether NVML init / apply_patch must be injected at worker init
    #     rather than here for the multi-worker case.
    worker = llm.llm_engine.model_executor.driver_worker  # confirm this path
    apply_patch(worker, controller)


def read_alpha_and_tokens(outputs, controller=None, mock=False):
    if mock:
        return 0.30, sum(len(o.outputs[0].token_ids) for o in outputs)

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    # >>> TODO (PATCH SEAM 3 / blank [H]): obtain the mean acceptance rate.
    #   Option A: vLLM SpecDecodeMetrics via vllm_hooks/metrics_reader.py.
    #   Option B: average the alphas the patch recorded on the controller log.
    alpha_mean = None
    if controller is not None and getattr(controller, "transition_log", None):
        alphas = [r.alpha for r in controller.transition_log if r.alpha is not None]
        alpha_mean = sum(alphas) / len(alphas) if alphas else None
    return alpha_mean, total_tokens


def configure_dvfs(controller, condition, model_pair, mock=False):
    """Set the controller up for one DVFS condition. The patch branches on .mode."""
    if hasattr(controller, "reset_log"):
        controller.reset_log()
    controller.mode = condition  # contract the patch honors (see FULL_CONDITIONS comment)

    if condition == "off":
        controller.enabled = False
        reset_clocks(mock)                      # default clock
    elif condition == "fixed_low":
        controller.enabled = False              # patch no-ops; we hold a global low lock
        controller.set_frequency_mhz(F_LOW)
    else:                                        # two_level / adaptive_alpha / adaptive_entropy / coarse
        controller.enabled = True
        if condition == "adaptive_entropy":
            # Supply (a, b) for α = a·exp(b·H); the patch reads these when it calls
            # on_verify_start_entropy(H, controller.entropy_a, controller.entropy_b).
            controller.entropy_a, controller.entropy_b = entropy_coeffs_for(model_pair)
        reset_clocks(mock)                      # start from default; patch drives per-phase


# =============================================================================
# ONE RUN
# =============================================================================

def run_single(llm, controller, model_pair, strategy, condition, dataset, rep,
               prompts, out_path, mock=False):
    print(f"  RUN  {out_path.name}")
    wait_for_temp(mock=mock)

    # Warm-up (not measured).
    if WARMUP_PROMPTS and not mock:
        _ = llm.generate(prompts[:WARMUP_PROMPTS], _sampling_params())

    configure_dvfs(controller, condition, model_pair, mock=mock)

    monitor = GpuMonitor(mock=mock)
    meter = EnergyMeter(mock=mock)
    monitor.start()
    meter.start()
    t0 = time.time()

    outputs = llm.generate(prompts, _sampling_params())

    wall_s = time.time() - t0
    energy = meter.stop()
    mon = monitor.stop_and_summary()

    alpha_mean, total_tokens = read_alpha_and_tokens(outputs, controller, mock=mock)

    result = {
        "model_pair": model_pair, "strategy": strategy,
        "dvfs_condition": condition, "dataset": dataset, "rep": rep,
        "seed": SEED, "n_prompts": len(prompts), "max_new_tokens": MAX_NEW_TOKENS,
        "f_high": F_HIGH, "f_low": F_LOW,
        "wall_time_s": wall_s,
        "total_tokens": total_tokens,
        "alpha_mean": alpha_mean,
        "energy_kwh": energy,
        "gpu_monitor": mon,                         # sm_clock_unique_mhz here = DVFS-toggle evidence
        "transition_log_summary": _summarize_transitions(controller),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if COOLDOWN_S and not mock:
        time.sleep(COOLDOWN_S)


def _sampling_params():
    # Greedy decoding (do_sample=False <=> temperature=0).
    from vllm import SamplingParams
    return SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, seed=SEED)


def _summarize_transitions(controller):
    log = getattr(controller, "transition_log", None)
    if not log:
        return {"n_transitions": 0}
    by_phase = {}
    for r in log:
        by_phase.setdefault(r.phase.value, []).append(r.freq_mhz)
    return {
        "n_transitions": len(log),
        "freqs_by_phase": {p: sorted(set(f)) for p, f in by_phase.items()},
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="SpecDVFS experiment orchestrator")
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--mock", action="store_true",
                    help="laptop dry-run: stub vLLM + NVML to exercise the harness")
    ap.add_argument("--n-prompts", type=int, default=None,
                    help="override prompt count (pilot default 64; full uses all)")
    ap.add_argument("--allow-clock-mismatch", action="store_true",
                    help="if the GPU isn't the confirmed 3090, auto-pick clocks instead of aborting")
    args = ap.parse_args()

    # DOUBT 5: validate the hardcoded anchors against the GPU actually present.
    global F_HIGH, F_LOW
    F_HIGH, F_LOW = resolve_clocks(args.mock, args.allow_clock_mismatch)

    matrix = PILOT_MATRIX if args.mode == "pilot" else build_full_matrix()
    reps = PILOT_REPS if args.mode == "pilot" else FULL_REPS
    n_prompts = args.n_prompts if args.n_prompts is not None else (
        PILOT_N_PROMPTS if args.mode == "pilot" else None
    )
    out_dir = RESULTS_DIR / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"mode={args.mode}  combos={len(matrix)}  reps={reps}  "
          f"total_runs={len(matrix) * reps}  mock={args.mock}  "
          f"f_high={F_HIGH} f_low={F_LOW}")

    completed = False
    # Group by (model_pair, strategy) so each model loads only once.
    matrix.sort(key=lambda x: (x[0], x[1]))
    try:
        for (mp, strat), group in groupby(matrix, key=lambda x: (x[0], x[1])):
            entries = list(group)  # each: (mp, strat, condition, dataset)
            family = MODEL_PAIRS[mp]["family"]

            # Resume: skip the whole group if every run file already exists.
            all_paths = [
                out_dir / f"{mp}__{strat}__{c}__{ds}__rep{r}.json"
                for (_, _, c, ds) in entries for r in range(reps)
            ]
            if all(p.exists() for p in all_paths):
                print(f"== skip group {mp}/{strat} (all {len(all_paths)} runs done)")
                continue

            print(f"== group {mp}/{strat}  ({len(entries)} conditions x {reps} reps)")
            llm = None
            controller = None
            prompt_cache = {}

            for (_, _, cond, ds) in entries:
                if ds not in prompt_cache:
                    prompt_cache[ds] = load_prompts(ds, family, n_prompts)
                prompts = prompt_cache[ds]

                for rep in range(reps):
                    out_path = out_dir / f"{mp}__{strat}__{cond}__{ds}__rep{rep}.json"
                    if out_path.exists():
                        print(f"  skip (done): {out_path.name}")
                        continue
                    if llm is None:  # lazy load on first needed run
                        print(f"  loading model: {mp} / {strat} ...")
                        llm = build_llm(mp, strat, args.mock)
                        controller = make_controller(args.mock)
                        apply_patch_to(llm, controller, args.mock)
                    run_single(llm, controller, mp, strat, cond, ds, rep,
                               prompts, out_path, args.mock)

            # Free the model before the next (model_pair, strategy).
            del llm, controller
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        completed = True
    finally:
        reset_clocks(args.mock)
        bar = "#" * 74
        if not completed:
            print("\n" + bar)
            print("##  RUN INTERRUPTED  —  GPU clocks reset.")
            print("##  Re-run the SAME command to resume (finished runs are skipped).")
            print(bar)
        elif args.mock:
            print("\n" + bar)
            print("##  MOCK DRY-RUN COMPLETE  —  harness OK. No GPU was used.")
            print(f"##  Stub results in: {out_dir}")
            print(bar)
        else:
            print("\n\n" + bar)
            print(bar)
            print("##" + " " * 70 + "##")
            print("##      ALL RUNS COMPLETE      —      GPU clocks reset.".ljust(72) + "##")
            print("##" + " " * 70 + "##")
            print(f"##   results on this VM: {str(out_dir):<44}##")
            print("##" + " " * 70 + "##")
            print("##   NEXT — run these on your LOCAL machine (NOT here):".ljust(72) + "##")
            print("##     bash collect_and_destroy.sh".ljust(72) + "##")
            print("##   (pulls results down, verifies them, THEN destroys the VM)".ljust(72) + "##")
            print("##" + " " * 70 + "##")
            print("##   Do NOT destroy the instance from inside the VM —".ljust(72) + "##")
            print("##   you would delete the only copy of the data.".ljust(72) + "##")
            print("##" + " " * 70 + "##")
            print(bar)
            print(bar)


if __name__ == "__main__":
    main()

# =============================================================================
# USAGE
# =============================================================================
# 0. Laptop dry-run (no GPU; validates the whole harness, JSON, resume, energy API):
#       python experiments/run_experiment.py --mode pilot --mock
#
# 1. On the VM, after setup_vm.sh and after the friend's patch + dvfs_controller.py
#    are in place and the three TODO SEAMS above are filled:
#       python experiments/run_experiment.py --mode pilot
#    PILOT SUCCESS = (a) the stack loads + SD runs; (b) gpu_monitor.sm_clock_unique_mhz
#    shows MORE THAN ONE clock for the DVFS-on runs (clock toggles) and a single/default
#    clock for `off`; (c) energy_kwh is sane and non-zero; (d) alpha_mean is plausible;
#    (e) energy(SD + two_level/adaptive_*) < energy(SD + off) at ~the same wall time
#    (energy saved within SD at minimal latency loss) — and BOTH signal conditions behave.
#
# 2. Full sweep (resumable — safe to restart after an interruption):
#       python experiments/run_experiment.py --mode full
#
# 3. Post-process with evaluation/compute_metrics.py: energy saving vs SD-without-DVFS and
#    vs vanilla, Wh/1K tokens, EDP, latency delta. (Avoid Dutta's "gamma" energy notation;
#    here gamma means draft length only.)
