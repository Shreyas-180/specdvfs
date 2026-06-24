"""SpecDVFS experiment orchestrator (final — wired to the delivered patch).

    python experiments/run_experiment.py --mode pilot          # core validation on the VM
    python experiments/run_experiment.py --mode full           # the full sweep
    python experiments/run_experiment.py --mode pilot --mock   # laptop dry-run, no GPU

WHAT THE DELIVERED PATCH SUPPORTS (verified against patch_spec_decode.py):
  The patch wraps draft/verify/_verify_tokens/_run_no_spec and ALWAYS calls
  on_verify_start(alpha=tracker.estimate). So mechanism-wise it implements exactly:
      off            -> controller.enabled = False  (hooks no-op; default clock)
      adaptive_alpha -> controller.enabled = True   (draft f_low; verify = verify_freq(lagging α))
  These two are wired and run now. The other conditions (fixed_low, two_level, coarse,
  adaptive_entropy) are kept in the matrix machinery but GUARDED — they each need a small
  patch extension (see PENDING_CONDITIONS and the patch-extension spec in the handoff).

CONTROL FLOW (the delivered patch's install() flow, fork-safe):
  install_dvfs() patches SpecDecodeWorker.init_device at the class level ONCE, with a
  factory that builds a DVFSController INSIDE each worker process (so nvmlInit runs there).
  Each LLM build then constructs+wraps a controller; retrieve_controller() fetches it from
  worker._dvfs_controller (in-process; TP=1 single-GPU). Per run we toggle controller.enabled
  (off vs adaptive_alpha). Vanilla has no SpecDecodeWorker, so its controller is None.

THREE SEAMS were resolved by the delivered files:
  build_llm()             — the vLLM 0.6.6 speculative_config (standard SD knobs; confirm).
  install/retrieve        — replaces the old apply_patch_to seam (patch_spec_decode.install).
  read_alpha_and_tokens() — metrics_reader.run_mean_acceptance (blank [H]).
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.prepare_datasets import PROMPT_TEMPLATES, CNN_DM_PREFIX  # noqa: E402

# =============================================================================
# CONFIG
# =============================================================================

F_HIGH = 1935          # confirmed RTX 3090 sustainable-lock ceiling
F_LOW = 735            # confirmed chosen low level
SEED = 42

# bitsandbytes is OFF by default: on this stack (vLLM 0.6.6 + Llama-3.2-1B draft),
# loading the draft model under `load_format="bitsandbytes"` crashes during weight
# loading (RuntimeError in linear.py's weight_loader narrow() — confirmed by tracing
# vLLM 0.6.6's own BitsAndBytesModelLoader: load_format is one engine-wide setting,
# so it's NOT possible to bnb-quantize the target while loading the draft in plain
# bf16 within a single LLM(...) call; on-the-fly bnb quantization unconditionally
# runs for every model the engine loads, target AND draft, and the 1B model's
# checkpoint layout breaks it). Loading both models native bf16 avoids the bug
# entirely; MAX_MODEL_LEN below keeps KV cache memory small enough to still fit a
# 24GB GPU. If a later vLLM build fixes the bnb+draft interaction, flip this back on.
USE_BNB_QUANTIZATION = False
MAX_MODEL_LEN = 2048   # plenty for GSM8K/HumanEval/CNN-DM prompts + MAX_NEW_TOKENS;
                       # keeps the bf16 KV cache footprint small on a 24GB GPU.
MAX_NUM_SEQS = 8       # caps concurrent-batch size; bounds the rejection sampler's
                       # per-step tensor (scales with batch_size x gamma x vocab_size)
                       # that caused the OOM seen with the default (256) concurrency.

TEMP_LIMIT_C = 75
WARMUP_PROMPTS = 5
COOLDOWN_S = 30
MONITOR_HZ = 100       # 10 ms NVML sampling
MAX_NEW_TOKENS = 256

SAMPLED_DIR = PROJECT_ROOT / "data" / "sampled_indices"
RESULTS_DIR = PROJECT_ROOT / "results"
CALIBRATION_DIR = PROJECT_ROOT / "calibration"

MODEL_PAIRS = {
    "qwen3_8b_0p6b": {"target": "Qwen/Qwen3-8B", "draft": "Qwen/Qwen3-0.6B",
                      "family": "qwen3", "eagle_head": None},
    "llama_8b_1b":   {"target": "meta-llama/Llama-3.1-8B-Instruct",
                      "draft": "meta-llama/Llama-3.2-1B-Instruct",
                      "family": "llama3", "eagle_head": None},
    "vicuna13b_68m": {"target": "lmsys/vicuna-13b-v1.3", "draft": "double7/vicuna-68m",
                      "family": "vicuna", "eagle_head": None},   # optional legacy
}
PRIMARY_PAIRS = ["qwen3_8b_0p6b", "llama_8b_1b"]

DATASETS = {
    "gsm8k": ("gsm8k_256.json", ""),
    "humaneval": ("humaneval_164.json", ""),
    "cnndm": ("cnndm_256.json", CNN_DM_PREFIX),
}

STRATEGIES = {
    "vanilla": {"kind": "vanilla"},
    "spec_g3": {"kind": "constant", "gamma": 3},
    "spec_g5": {"kind": "constant", "gamma": 5},
    "spec_g7": {"kind": "constant", "gamma": 7},
    "spec_dyn": {"kind": "dynamic", "gamma_max": 7},
    "eagle3": {"kind": "eagle"},   # runs only for pairs with eagle_head set
}
SD_STRATEGIES_FULL = ["vanilla", "spec_g3", "spec_g5", "spec_g7", "spec_dyn", "eagle3"]

# Conditions the patch supports out of the box (lagging-α mechanism in patch_spec_decode.py):
SUPPORTED_CONDITIONS = ["off", "adaptive_alpha"]
# Conditions enabled by the patch extension (entropy lm_head hook + 'collect' mode) together
# with the mode-aware on_verify_start in controller/core.py:
#   fixed_low        -> core.py pins f_low on verify too (draft already f_low -> global low)
#   two_level        -> core.py forces f_high on verify (static phase-aware), α ignored
#   coarse           -> core.py holds one clock per ~100 ms window across BOTH phases
#   adaptive_entropy -> core.py maps draft entropy H -> α via a·exp(b·H); the patch's hook
#                       feeds controller.last_entropy and 'collect' mode fills entropy_pairs
PENDING_CONDITIONS = ["fixed_low", "two_level", "coarse", "adaptive_entropy"]

# Active conditions in --mode full. The patch extension is in place, so the previously-pending
# conditions are now live alongside the originally-supported ones.
FULL_CONDITIONS = SUPPORTED_CONDITIONS + PENDING_CONDITIONS

# Conditions that map 1:1 onto a controller mode string (everything except the DVFS-off
# baseline). For these, the condition name IS the controller.mode value dispatched in core.py.
DVFS_MODE_CONDITIONS = [c for c in (SUPPORTED_CONDITIONS + PENDING_CONDITIONS) if c != "off"]

# ---- PILOT (exercise the code that actually differs across DVFS conditions) ----
# NOTE: pilot uses the Llama pair, not Qwen3 — vLLM 0.6.6 (pinned for the patch's
# spec_decode hooks) predates the Qwen3 architecture, so Qwen3 will not load on it.
# Llama-3.1-8B / 3.2-1B are supported by 0.6.6 (they are GATED — needs HF auth).
#
# The pilot's job is to surface bugs/mismatches before the full sweep, so it holds
# the model AND the SD strategy fixed and sweeps ALL SIX DVFS conditions — that is
# where the new, least-tested code lives (the mode-aware on_verify_start dispatch,
# and for adaptive_entropy the entropy lm_head hook + calibration chain). A single
# vanilla (no-SD) reference is included so energy/latency have a baseline.
PILOT_MODEL = "llama_8b_1b"
PILOT_STRATEGY = "spec_g5"
PILOT_DATASET = "gsm8k"
PILOT_N_PROMPTS = 64
PILOT_REPS = 3
FULL_REPS = 5

# One row per DVFS condition (all of FULL_CONDITIONS) on the fixed model+strategy,
# plus a vanilla reference. Built from FULL_CONDITIONS so it can never drift out of
# sync with the set of conditions the controller actually supports.
PILOT_MATRIX = (
    [(PILOT_MODEL, "vanilla", "off", PILOT_DATASET)]   # reference (no SD)
    + [(PILOT_MODEL, PILOT_STRATEGY, cond, PILOT_DATASET) for cond in FULL_CONDITIONS]
)


def build_full_matrix():
    combos = []
    for mp in PRIMARY_PAIRS:
        for strat in SD_STRATEGIES_FULL:
            if strat == "eagle3" and not MODEL_PAIRS[mp].get("eagle_head"):
                continue
            conditions = ["off"] if strat == "vanilla" else FULL_CONDITIONS
            for cond in conditions:
                for ds in DATASETS:
                    combos.append((mp, strat, cond, ds))
    return combos


# =============================================================================
# ENTROPY CALIBRATION — fits α = a·exp(b·H) per model pair for adaptive_entropy.
# Driven by calibrate_pairs_if_needed() (called from main() after install_dvfs):
# it runs a short DVFS-off 'collect' pass via the patch's entropy hook, fits the
# coefficients, and caches them to calibration/fitted_<pair>.json. Falls back to
# the GELATO baseline if a pair yields no usable (entropy, alpha) pairs. Fully
# exercised by --mock (synthetic pairs); inactive for matrices without an
# adaptive_entropy condition (e.g. the pilot), where it is a no-op.
# =============================================================================

GELATO_BASELINE_AB = (1.0, -0.35)
CALIB_GAMMA = 5
# Calibrate across ALL dataset types, not just one — a single dataset risks narrow
# entropy coverage (e.g. GSM8K's fairly formulaic math steps), which would leave the
# fitted curve poorly constrained outside that range. The total prompt budget is
# spread across types instead of concentrated in one.
CALIB_DATASETS = list(DATASETS)
CALIB_N_PROMPTS = 64        # TOTAL across CALIB_DATASETS, split as evenly as possible
                            # (64 over 3 types -> 22/21/21). Each prompt yields many
                            # (entropy, alpha) pairs (one per decode step), so the fit
                            # still sees >> CALIB_MIN_PAIRS samples.
CALIB_MIN_PAIRS = 200
CALIB_MIN_R2 = 0.5


def calibration_path(model_pair):
    return CALIBRATION_DIR / f"fitted_{model_pair}.json"


def entropy_coeffs_for(model_pair):
    f = calibration_path(model_pair)
    if f.exists():
        try:
            from calibration.entropy_calibration import CalibrationResult
            r = CalibrationResult.from_json(str(f))
            return r.a, r.b
        except Exception:
            pass
    return GELATO_BASELINE_AB


def collect_entropy_pairs(llm, controller, prompts, mock=False):
    """Needs the patch's 'collect' mode (controller.entropy_pairs). INACTIVE for now."""
    if mock:
        import math, random
        return [{"entropy": (H := random.uniform(0, 4)),
                 "alpha": max(0.0, min(1.0, math.exp(-0.35 * H) + random.uniform(-0.05, 0.05)))}
                for _ in range(256)]
    controller.entropy_pairs = []
    controller.mode = "collect"
    controller.enabled = False
    reset_clocks(mock)
    _ = llm.generate(prompts, _sampling_params())
    return list(getattr(controller, "entropy_pairs", []))


def fit_and_save_calibration(model_pair, pairs):
    import numpy as np
    from calibration.entropy_calibration import calibrate, plot_calibration
    H = np.asarray([p["entropy"] for p in pairs], float)
    A = np.asarray([p["alpha"] for p in pairs], float)
    result = calibrate(H, A)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    result.to_json(str(calibration_path(model_pair)))
    try:
        plot_calibration(H, A, result, str(CALIBRATION_DIR / f"fit_{model_pair}.png"))
    except Exception:
        pass
    print(f"    fitted {model_pair}: a={result.a:.3f} b={result.b:.3f} R^2={result.r_squared:.4f}")
    if result.r_squared < CALIB_MIN_R2:
        print(f"    WARNING: low R^2 for {model_pair}; entropy is a weak predictor.")
    return result


def calibration_prompts(family, mock=False):
    """CALIB_N_PROMPTS prompts total, split as evenly as possible across CALIB_DATASETS
    (e.g. 128 over 3 types -> 43/43/42). Combined into one list — one build_llm/generate
    pass per model pair, not one per dataset, so calibration cost doesn't multiply."""
    n = len(CALIB_DATASETS)
    base, extra = divmod(CALIB_N_PROMPTS, n)   # extra goes to the first `extra` datasets
    prompts = []
    for i, ds in enumerate(CALIB_DATASETS):
        k = base + (1 if i < extra else 0)
        prompts += load_prompts(ds, family, k, mock=mock)
    return prompts


def calibrate_pairs_if_needed(matrix, mock=False, recalibrate=False):
    pairs = [mp for mp in dict.fromkeys(m[0] for m in matrix)
             if any(x[0] == mp and x[2] == "adaptive_entropy" for x in matrix)
             and (recalibrate or not calibration_path(mp).exists())]
    if not pairs:
        return
    print(f"== AUTO-CALIBRATION for {pairs} ==")
    for mp in pairs:
        prompts = calibration_prompts(MODEL_PAIRS[mp]["family"], mock=mock)
        llm = build_llm(mp, f"spec_g{CALIB_GAMMA}", mock)
        controller = retrieve_controller(llm, mock)
        got = collect_entropy_pairs(llm, controller, prompts, mock)
        if got:
            fit_and_save_calibration(mp, got)
        else:
            print(f"    WARNING: no pairs for {mp} (patch lacks 'collect' mode) — GELATO fallback.")
        del llm, controller
        gc.collect()
    reset_clocks(mock)


# =============================================================================
# PROMPTS
# =============================================================================

def load_prompts(dataset_key, family, n=None, mock=False):
    fname, prefix = DATASETS[dataset_key]
    path = SAMPLED_DIR / fname
    if mock and not path.exists():
        # Laptop dry-run before datasets are prepared: synthesize prompts so the
        # harness/JSON/resume logic can be validated with no data files present.
        tmpl = PROMPT_TEMPLATES[family]
        k = n if n is not None else 8
        return [tmpl.format(text=prefix + f"Mock prompt {i} for harness validation.")
                for i in range(k)]
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data["samples"][:n] if n is not None else data["samples"]
    tmpl = PROMPT_TEMPLATES[family]
    return [tmpl.format(text=prefix + s["text"]) for s in samples]


# =============================================================================
# GPU MONITOR (10 ms NVML sampler — clock-toggle evidence)
# =============================================================================

class GpuMonitor(threading.Thread):
    def __init__(self, hz=MONITOR_HZ, mock=False):
        super().__init__(daemon=True)
        self.interval = 1.0 / hz
        self.mock = mock
        self._stopper = threading.Event()   # NB: not '_stop' — collides with Thread._stop()
        self.samples = []

    def run(self):
        if self.mock:
            self._stopper.wait()
            return
        import pynvml as N
        N.nvmlInit()
        h = N.nvmlDeviceGetHandleByIndex(0)
        while not self._stopper.is_set():
            try:
                u = N.nvmlDeviceGetUtilizationRates(h)
                self.samples.append((time.time(), u.gpu, u.memory,
                                     N.nvmlDeviceGetPowerUsage(h),
                                     N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_GRAPHICS),
                                     N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU)))
            except N.NVMLError:
                pass
            self._stopper.wait(self.interval)
        N.nvmlShutdown()

    def stop_and_summary(self):
        self._stopper.set()
        self.join(timeout=3)
        if not self.samples:
            return {}
        clocks = [s[4] for s in self.samples]
        powers = [s[3] for s in self.samples]
        temps = [s[5] for s in self.samples]
        return {
            "n_samples": len(self.samples),
            "sm_clock_min_mhz": min(clocks), "sm_clock_max_mhz": max(clocks),
            "sm_clock_unique_mhz": sorted(set(clocks)),
            "sm_clock_hist": dict(Counter(clocks).most_common(8)),
            "power_mw_mean": sum(powers) / len(powers), "temp_c_max": max(temps),
        }


# =============================================================================
# ENERGY METER (CodeCarbon; handles the 3.x stop()-returns-float change)
# =============================================================================

class EnergyMeter:
    def __init__(self, mock=False):
        self.mock = mock
        self._t = None

    def start(self):
        if self.mock:
            return
        from codecarbon import EmissionsTracker
        self._t = EmissionsTracker(measure_power_secs=1, save_to_file=False,
                                   log_level="error", tracking_mode="machine")
        self._t.start()

    def stop(self):
        if self.mock:
            return {"total_kwh": 1e-4, "gpu_kwh": 8e-5, "cpu_kwh": 1e-5, "ram_kwh": 1e-5}
        try:
            self._t.stop()
        except Exception:
            pass

        def kwh(attr):
            try:
                return float(getattr(self._t, attr).kWh)
            except Exception:
                return None

        return {"total_kwh": kwh("_total_energy"), "gpu_kwh": kwh("_total_gpu_energy"),
                "cpu_kwh": kwh("_total_cpu_energy"), "ram_kwh": kwh("_total_ram_energy")}


# =============================================================================
# TEMP GATE + CLOCKS
# =============================================================================

def wait_for_temp(limit=TEMP_LIMIT_C, mock=False, timeout=300):
    if mock:
        return
    import pynvml as N
    N.nvmlInit()
    h = N.nvmlDeviceGetHandleByIndex(0)
    t0 = time.time()
    while True:
        if N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU) < limit or (time.time() - t0) > timeout:
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
    """Validate F_HIGH/F_LOW against the live GPU (snap; abort on a clearly different card)."""
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
    snap = lambda f: min(gfx, key=lambda g: abs(g - f))
    fh, fl = snap(F_HIGH), snap(F_LOW)
    far = (abs(fh - F_HIGH) > 50) or (abs(fl - F_LOW) > 50)
    print(f"    GPU: {name} | {len(gfx)} levels {gfx[0]}-{gfx[-1]} MHz | anchors {F_HIGH}->{fh} {F_LOW}->{fl}")
    if far and not allow_mismatch:
        raise SystemExit(
            f"ABORT: this GPU ({gfx[0]}-{gfx[-1]} MHz) is not the confirmed RTX 3090 "
            f"(f_high={F_HIGH}, f_low={F_LOW}); energy numbers would not be comparable. "
            "Re-confirm/update the constants or pass --allow-clock-mismatch.")
    if far and allow_mismatch:
        fh, fl = gfx[-1], snap(int(round(0.38 * gfx[-1])))
        print(f"    --allow-clock-mismatch: f_high={fh}, f_low={fl} (RE-CONFIRM sustainable f_high!)")
    return fh, fl


# =============================================================================
# DVFS PATCH WIRING (delivered patch's install() flow)
# =============================================================================

_DVFS_INSTALLED = False


def install_dvfs(mock=False):
    """Patch SpecDecodeWorker.init_device ONCE with a per-worker DVFSController factory.

    enabled=True so NVML is initialised in the worker; the orchestrator toggles
    controller.enabled per run (off vs adaptive_alpha). Call before building any LLM.
    """
    global _DVFS_INSTALLED
    if mock or _DVFS_INSTALLED:
        return
    from vllm_hooks.patch_spec_decode import install
    from vllm_hooks.dvfs_controller import DVFSController
    install(lambda: DVFSController(f_high=F_HIGH, f_low=F_LOW, enabled=True))
    _DVFS_INSTALLED = True
    print("    installed DVFS patch (per-worker DVFSController factory)")


_WORKER_PATHS = (("llm_engine", "model_executor", "driver_worker"),
                 ("model_executor", "driver_worker"), ("driver_worker",))


def retrieve_controller(llm, mock=False):
    """Fetch the controller the patch created in the worker (None for vanilla / unreachable)."""
    if mock:
        from controller.core import SimulatedDVFSController
        return SimulatedDVFSController(F_HIGH, F_LOW, enabled=True)
    for path in _WORKER_PATHS:
        obj = llm
        ok = True
        for a in path:
            obj = getattr(obj, a, None)
            if obj is None:
                ok = False
                break
        if not ok or obj is None:
            continue
        ctrl = getattr(obj, "_dvfs_controller", None)
        if ctrl is None:
            w = getattr(obj, "worker", None)
            ctrl = getattr(w, "_dvfs_controller", None) if w is not None else None
        if ctrl is not None:
            return ctrl
    return None


class _MockLLM:
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
    if USE_BNB_QUANTIZATION:
        common = dict(model=target, quantization="bitsandbytes", load_format="bitsandbytes",
                      dtype="auto", gpu_memory_utilization=0.90, seed=SEED)
    else:
        # Native bf16 for both target and draft (see USE_BNB_QUANTIZATION comment above
        # for why bnb is off).
        #
        # Confirmed from a real OOM trace: vLLM's memory profiler sizes the KV cache
        # BEFORE CUDA-graph capture, but graph capture then adds its own ~1.2-1.3GB on
        # top (captured for many batch-size shapes) — on this 24GB card that pushed
        # actual usage to ~23.5/23.7GB with zero slack, and the next allocation (inside
        # the rejection sampler, whose tensor size scales with
        # batch_size x num_speculative_tokens x vocab_size) had nowhere to go. Fix,
        # directly addressing each contributor rather than just shrinking one knob:
        #   enforce_eager=True   -> no CUDA-graph capture, reclaims that ~1.2-1.3GB
        #                           and removes the riskiest overshoot path (this is
        #                           vLLM's own suggested remedy in the cudagraph
        #                           warning text). Applied uniformly to every
        #                           condition, so the DVFS comparison stays fair.
        #   max_num_seqs=8        -> directly bounds the rejection-sampler tensor that
        #                           actually crashed (it scales with concurrent batch
        #                           size); the default (256) let far too many of a
        #                           128-prompt calibration batch be scheduled at once.
        #   max_model_len=2048    -> still generous for GSM8K/HumanEval/CNN-DM prompts
        #                           + MAX_NEW_TOKENS=256; halves KV-cache-per-sequence
        #                           footprint vs the previous 4096, giving the
        #                           scheduler more room before hitting preemption.
        common = dict(model=target, dtype="auto", gpu_memory_utilization=0.90,
                      max_model_len=MAX_MODEL_LEN, max_num_seqs=MAX_NUM_SEQS,
                      enforce_eager=True, seed=SEED)
    if cfg["kind"] == "vanilla":
        return LLM(**common)
    # >>> SEAM: confirm vLLM 0.6.6 speculative_config keys (these are the standard ones).
    if cfg["kind"] == "constant":
        spec = dict(speculative_model=draft, num_speculative_tokens=cfg["gamma"])
    elif cfg["kind"] == "dynamic":
        spec = dict(speculative_model=draft, num_speculative_tokens=cfg["gamma_max"])  # + dynamic knob: TODO
    elif cfg["kind"] == "eagle":
        head = MODEL_PAIRS[model_pair].get("eagle_head")
        if not head:
            raise NotImplementedError(f"set MODEL_PAIRS['{model_pair}']['eagle_head'] for eagle3.")
        spec = dict(speculative_model=head, num_speculative_tokens=5)  # + EAGLE method: TODO
    else:
        raise NotImplementedError(cfg["kind"])
    return LLM(**common, **spec)


def configure_dvfs(controller, condition, mock=False):
    """off -> enabled=False (default clock). Every other condition -> enabled=True with
    controller.mode set to the condition name (core.py dispatches the verify policy)."""
    if controller is None:        # vanilla / no SpecDecodeWorker
        reset_clocks(mock)        # ensure default clock — clear any lock left by a prior group
        return
    if hasattr(controller, "reset_log"):
        controller.reset_log()
    if condition == "off":
        controller.enabled = False
        if hasattr(controller, "reset_clocks"):
            controller.reset_clocks()     # release any prior lock -> true default-clock baseline
        else:
            reset_clocks(mock)
    elif condition in DVFS_MODE_CONDITIONS:
        # The controller is built once per (model, strategy) group and REUSED across
        # conditions, so set BOTH flags on every run — mode in particular must be refreshed
        # or it would leak from the previous condition. The condition name is exactly the
        # controller.mode string (adaptive_alpha / fixed_low / two_level / coarse /
        # adaptive_entropy) that core.py's on_verify_start dispatches on.
        controller.enabled = True
        controller.mode = condition
    else:
        raise NotImplementedError(
            f"unknown condition '{condition}' (expected one of {['off'] + DVFS_MODE_CONDITIONS}).")


def read_alpha_and_tokens(outputs, llm=None, controller=None, mock=False):
    if mock:
        return 0.30, sum(len(o.outputs[0].token_ids) for o in outputs)
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    alpha_mean = None
    try:
        from vllm_hooks import metrics_reader
        alpha_mean = metrics_reader.run_mean_acceptance(llm)
    except Exception:
        alpha_mean = None
    if alpha_mean is None and controller is not None and getattr(controller, "transition_log", None):
        alphas = [r.alpha for r in controller.transition_log if r.alpha is not None]
        alpha_mean = sum(alphas) / len(alphas) if alphas else None
    return alpha_mean, total_tokens


# =============================================================================
# ONE RUN
# =============================================================================

def run_single(llm, controller, model_pair, strategy, condition, dataset, rep,
               prompts, out_path, mock=False):
    print(f"  RUN  {out_path.name}")
    wait_for_temp(mock=mock)
    if WARMUP_PROMPTS and not mock:
        _ = llm.generate(prompts[:WARMUP_PROMPTS], _sampling_params())

    configure_dvfs(controller, condition, mock=mock)

    monitor = GpuMonitor(mock=mock)
    meter = EnergyMeter(mock=mock)
    monitor.start()
    meter.start()
    sp = None if mock else _sampling_params()   # mock LLM ignores it; avoids importing vLLM
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    wall_s = time.time() - t0
    energy = meter.stop()
    mon = monitor.stop_and_summary()

    alpha_mean, total_tokens = read_alpha_and_tokens(outputs, llm, controller, mock=mock)

    result = {
        "model_pair": model_pair, "strategy": strategy, "dvfs_condition": condition,
        "dataset": dataset, "rep": rep, "seed": SEED, "n_prompts": len(prompts),
        "max_new_tokens": MAX_NEW_TOKENS, "f_high": F_HIGH, "f_low": F_LOW,
        "wall_time_s": wall_s, "total_tokens": total_tokens, "alpha_mean": alpha_mean,
        "energy_kwh": energy, "gpu_monitor": mon,
        "transition_log_summary": _summarize_transitions(controller),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if COOLDOWN_S and not mock:
        time.sleep(COOLDOWN_S)


def _sampling_params():
    from vllm import SamplingParams
    return SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, seed=SEED)


def _summarize_transitions(controller):
    log = getattr(controller, "transition_log", None) if controller is not None else None
    if not log:
        return {"n_transitions": 0}
    by_phase = {}
    for r in log:
        by_phase.setdefault(r.phase.value, []).append(r.freq_mhz)
    return {"n_transitions": len(log),
            "freqs_by_phase": {p: sorted(set(f)) for p, f in by_phase.items()}}


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="SpecDVFS experiment orchestrator")
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--mock", action="store_true", help="laptop dry-run: stub vLLM + NVML")
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--allow-clock-mismatch", action="store_true")
    ap.add_argument("--recalibrate", action="store_true",
                    help="refit entropy calibration even if calibration/fitted_<pair>.json exists")
    ap.add_argument("--f-low", type=int, default=None,
                    help="override F_LOW (the draft-phase clock, MHz) for this run — used to "
                         "sweep the draft clock and find the energy/latency knee. Snapped to a "
                         "valid GPU clock level by resolve_clocks.")
    ap.add_argument("--f-high", type=int, default=None,
                    help="override F_HIGH (the verify-phase clock, MHz) for this run.")
    args = ap.parse_args()

    global F_HIGH, F_LOW
    # CLI overrides take effect BEFORE resolve_clocks so they get snapped/validated
    # against the live GPU's clock table exactly like the defaults would be.
    if args.f_high is not None:
        F_HIGH = args.f_high
    if args.f_low is not None:
        F_LOW = args.f_low
    F_HIGH, F_LOW = resolve_clocks(args.mock, args.allow_clock_mismatch)
    reset_clocks(args.mock)   # clean start: clear any lock left by a prior/crashed run

    matrix = PILOT_MATRIX if args.mode == "pilot" else build_full_matrix()
    reps = PILOT_REPS if args.mode == "pilot" else FULL_REPS
    n_prompts = args.n_prompts if args.n_prompts is not None else (
        PILOT_N_PROMPTS if args.mode == "pilot" else None)
    out_dir = RESULTS_DIR / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"mode={args.mode}  combos={len(matrix)}  reps={reps}  "
          f"total_runs={len(matrix) * reps}  mock={args.mock}  f_high={F_HIGH} f_low={F_LOW}")
    print(f"  active conditions: {FULL_CONDITIONS}")

    install_dvfs(args.mock)   # patch SpecDecodeWorker.init_device once, before any build
    # Entropy auto-calibration: the patch now supports 'collect' mode, so fit α = a·exp(b·H)
    # for any model pair that has an adaptive_entropy condition and no cached fit. This runs
    # DVFS-off and writes calibration/fitted_<pair>.json (GELATO baseline if it yields none).
    calibrate_pairs_if_needed(matrix, args.mock, recalibrate=args.recalibrate)

    completed = False
    matrix.sort(key=lambda x: (x[0], x[1]))
    try:
        for (mp, strat), group in groupby(matrix, key=lambda x: (x[0], x[1])):
            entries = list(group)
            family = MODEL_PAIRS[mp]["family"]
            all_paths = [out_dir / f"{mp}__{strat}__{c}__{ds}__rep{r}.json"
                         for (_, _, c, ds) in entries for r in range(reps)]
            if all(p.exists() for p in all_paths):
                print(f"== skip group {mp}/{strat} (all {len(all_paths)} runs done)")
                continue

            print(f"== group {mp}/{strat}  ({len(entries)} conditions x {reps} reps)")
            llm = None
            controller = None
            prompt_cache = {}
            for (_, _, cond, ds) in entries:
                if ds not in prompt_cache:
                    prompt_cache[ds] = load_prompts(ds, family, n_prompts, mock=args.mock)
                prompts = prompt_cache[ds]
                for rep in range(reps):
                    out_path = out_dir / f"{mp}__{strat}__{cond}__{ds}__rep{rep}.json"
                    if out_path.exists():
                        print(f"  skip (done): {out_path.name}")
                        continue
                    if llm is None:
                        print(f"  loading model: {mp} / {strat} ...")
                        llm = build_llm(mp, strat, args.mock)
                        controller = retrieve_controller(llm, args.mock)
                        if controller is not None:
                            # Per-pair entropy coefficients for adaptive_entropy (constant
                            # across this group): fitted file if present, else GELATO baseline.
                            controller.entropy_a, controller.entropy_b = entropy_coeffs_for(mp)
                        if strat != "vanilla" and controller is None and not args.mock:
                            print("  WARN: no controller on the worker (DVFS will be inactive). "
                                  "Check install()/in-process worker (handoff §8.4).")
                    run_single(llm, controller, mp, strat, cond, ds, rep,
                               prompts, out_path, args.mock)
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
            print("##   NEXT — run on your LOCAL machine (NOT here):".ljust(72) + "##")
            print("##     bash collect_and_destroy.sh".ljust(72) + "##")
            print("##   (pulls results down, verifies them, THEN destroys the VM)".ljust(72) + "##")
            print("##   Do NOT destroy the instance from inside the VM.".ljust(72) + "##")
            print(bar)
            print(bar)


if __name__ == "__main__":
    main()

# ── USAGE ────────────────────────────────────────────────────────────────────
# 0. Laptop dry-run (no GPU; validates harness/JSON/resume; install_dvfs is a no-op):
#       python experiments/run_experiment.py --mode pilot --mock
# 1. VM, after setup_vm.sh + the friend's vllm_hooks/ files (patch + dvfs_controller +
#    metrics_reader) are in place:
#       python experiments/run_experiment.py --mode pilot
#    Checks: gpu_monitor.sm_clock_unique_mhz > 1 for adaptive_alpha (clock toggles),
#    == 1 for off; alpha_mean populated for the SD runs; energy(adaptive) < energy(off)
#    at ~the same wall time.
# 2. Full sweep (resumable):  python experiments/run_experiment.py --mode full
# 3. Post-process on the laptop: evaluation/compute_metrics.py -> aggregate -> figures.
