"""SpecDVFS experiment orchestrator (final — wired to the delivered patch).

    python experiments/run_experiment.py --mode pilot          # core validation on the VM
    python experiments/run_experiment.py --mode full           # the full sweep
    python experiments/run_experiment.py --mode pilot --mock   # laptop dry-run, no GPU

WHAT THE DELIVERED PATCH SUPPORTS (verified against patch_spec_decode.py):
  The patch wraps draft/verify/_verify_tokens/_run_no_spec and ALWAYS calls
  on_verify_start(alpha=tracker.estimate). The per-condition verify policy lives in
  core.py (mode-aware on_verify_start), so the patch stays condition-agnostic and the
  controller.mode string selects the behaviour:
      off              -> controller.enabled = False (hooks no-op; default clock)
      two_level        -> f_low draft, f_high verify (alpha ignored)
      adaptive_alpha   -> verify = verify_freq(lagging EMA alpha)
      adaptive_entropy -> verify = verify_freq(a*exp(b*H)) on the draft's entropy H
      fixed_low,coarse -> ablations (full-run only)
  adaptive_entropy additionally requires the draft lm_head entropy hook to fire and a
  per-pair (a,b) calibration (auto-fit pre-pass); it is the least-tested path, so the
  pilot checks that its runs actually differ from adaptive_alpha (see the footer USAGE).

CONTROL FLOW (the delivered patch's install() flow, fork-safe):
  install_dvfs() patches SpecDecodeWorker.init_device at the class level ONCE, with a
  factory that builds a DVFSController INSIDE each worker process (so nvmlInit runs there).
  Each LLM build then constructs+wraps a controller; retrieve_controller() fetches it from
  worker._dvfs_controller (in-process; TP=1 single-GPU). Per run we set controller.enabled
  and controller.mode (configure_dvfs). Vanilla has no SpecDecodeWorker, so controller is None.

THREE SEAMS were resolved by the delivered files:
  build_llm()             — the vLLM 0.6.6 speculative_config (standard SD knobs; confirm).
  install/retrieve        — replaces the old apply_patch_to seam (patch_spec_decode.install).
  read_alpha_and_tokens() — metrics_reader.run_mean_acceptance (blank [H]).
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess          # run_sm_sweep: one child process per SM level (MPS env is
                           # read at CUDA-context creation, so it can't change in-process)
import sys
import threading
import time
from collections import Counter
from itertools import groupby
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.prepare_datasets import PROMPT_TEMPLATES, CNN_DM_PREFIX  # noqa: E402
import sm_partition  # noqa: E402  (repo-root module; Approach 2 — SM restriction)

# =============================================================================
# CONFIG
# =============================================================================

F_HIGH = 1935          # confirmed RTX 3090 sustainable-lock ceiling
F_LOW = 735            # confirmed chosen low level
SEED = 42

TEMP_LIMIT_C = 75
WARMUP_PROMPTS = 5
COOLDOWN_S = 30
MONITOR_HZ = 100       # 10 ms NVML sampling
MAX_NEW_TOKENS = 256

# --- Engine memory fit (bf16, quantization OFF) -------------------------------
# These are the settings the *successful* pilot actually ran under (handoff:
# "Both models load in bf16 (quantization OFF)", weights 14.99GiB + KV 3.75GiB +
# ~2.6 overhead < 23.69GiB usable on the 3090). They are restored here because
# build_llm() in this file was still on the abandoned bitsandbytes path (see the
# FAILED-approaches list) and set none of them — i.e. this file as delivered would
# NOT reproduce the pilot: bnb crashes the 0.6.6 spec-decode draft path, and with
# no max_model_len / enforce_eager the engine OOMs (CUDA-graph capture after KV
# sizing; 4096 context). Quantization is NOT required for any pair the study runs
# (llama_8b_1b pilot; qwen3_8b_0p6b + llama_8b_1b full) — all fit in bf16 at 24GB.
MAX_MODEL_LEN = 2048   # validated fit; > gsm8k/humaneval prompt + 256 gen. cnndm
                       # articles can exceed this and are truncated at load (below).
MAX_NUM_SEQS  = 8      # bounds the spec rejection-sampler tensor (batch x gamma x
                       # vocab) that OOM'd in _get_recovered_probs at the default.
# NOTE: both of the above are DEFAULTS only. They are overridable per-run (--max-num-seqs /
# --max-model-len, and per-cell by the batch sweep) because MAX_NUM_SEQS is the instrument
# for Approach 1 — see the PILOT_BATCH_SWEEP block. MAX_NUM_SEQS is a SCHEDULER cap: raising
# it allocates no extra VRAM, it only allows more sequences to be resident at once. The real
# constraint is total KV slots (below), not this number.

# --- KV budget (from the pilot's own vLLM log; used to bound the batch sweep) --------------
# vLLM reported: 1517 GPU blocks x block_size 16 = 24,272 token slots, and "Maximum
# concurrency for 2048 tokens per request: 11.85x" — i.e. the engine itself states only ~11.8
# full-2048-context sequences fit at once. That number is an INDEPENDENT confirmation of the
# batch ceiling the sweep is built around, and it is why the sweep drops max_model_len to
# 1024: at 1024 ctx the ceiling rises to 23.7x, so batch 22 fits even in the worst case
# (every sequence simultaneously at full context).
KV_TOTAL_TOKEN_SLOTS = 24272

SAMPLED_DIR = PROJECT_ROOT / "data" / "sampled_indices"
RESULTS_DIR = PROJECT_ROOT / "results"
CALIBRATION_DIR = PROJECT_ROOT / "calibration"

MODEL_PAIRS = {
    "qwen3_8b_0p6b": {"target": "Qwen/Qwen3-8B", "draft": "Qwen/Qwen3-0.6B",
                      "family": "qwen3", "eagle_head": None},
    "llama_8b_1b":   {"target": "meta-llama/Llama-3.1-8B-Instruct",
                      "draft": "meta-llama/Llama-3.2-1B-Instruct",
                      "family": "llama3", "eagle_head": None},
    # Same 8B-Instruct target, but the BASE (non-instruct) 1B as the draft. Valid SD
    # config: 3.1/3.2 share the 128k tokenizer, so draft/target tokens align; only the
    # draft's predictions differ. A base draft is less aligned with the instruct target,
    # so expect LOWER alpha than llama_8b_1b — useful contrast (more of the alpha range
    # exercised, and a second entropy->alpha calibration). family stays llama3 because the
    # prompts are formatted for the (instruct) TARGET; the draft just consumes those tokens.
    "llama_8b_1b_base": {"target": "meta-llama/Llama-3.1-8B-Instruct",
                         "draft": "meta-llama/Llama-3.2-1B",
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
    "spec_g12": {"kind": "constant", "gamma": 12},   # added: longer draft -> lower mean alpha
    "spec_g18": {"kind": "constant", "gamma": 18},   # added: pushes alpha into the responsive band
    "spec_dyn": {"kind": "dynamic", "gamma_max": 7},
    "eagle3": {"kind": "eagle"},   # runs only for pairs with eagle_head set
}
SD_STRATEGIES_FULL = ["vanilla", "spec_g3", "spec_g5", "spec_g7", "spec_dyn", "eagle3"]
# NB: spec_g12/spec_g18 are intentionally NOT in SD_STRATEGIES_FULL — they are a
# pilot instrument for generating alpha variance, not (yet) part of the full sweep.

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

# ---- PILOT (alpha variance + all four phase-DVFS policies, incl. the entropy one) ----
# NOTE: pilot uses the Llama pairs, not Qwen3 — vLLM 0.6.6 (pinned for the patch's
# spec_decode hooks) predates the Qwen3 architecture, so Qwen3 will not load on it.
# Two draft choices on the SAME Llama-3.1-8B-Instruct target: the 3.2-1B-Instruct draft
# (llama_8b_1b) and the 3.2-1B BASE draft (llama_8b_1b_base). All GATED — needs HF auth.
#
# WHY THE OLD PILOT FAILED TO BE USEFUL.  It fixed (model, strategy, dataset) and swept
# DVFS conditions only.  But alpha (= num_accepted / num_draft; see metrics_reader) is
# decided by the accept/reject sequence, which at temp=0/seed=42 is bit-identical across
# DVFS conditions — DVFS changes clock timing, not computation.  So alpha was a CONSTANT
# 0.9208 in every cell, >> the mapper's alpha_high=0.7: verify_freq() always returned
# f_high and every adaptive mode collapsed onto two_level. Nothing to validate.
#
# WHAT MOVES ALPHA — and why that matters for BOTH adaptive policies. Not the DVFS
# condition (it can't) but the accept/reject sequence itself, via:
#   * draft length gamma — longer drafts propose tokens further out where draft/target
#     diverge, so mean acceptance falls as gamma grows;
#   * dataset — code/summarization diverge from the draft more than formulaic math.
# Sweeping gamma in {5,12,18} x {gsm8k, humaneval, cnndm} spreads alpha from a
# saturated-high corner (gsm8k, g5) down toward the responsive band (alpha_low=0.3 ..
# alpha_high=0.7). That spread is what lets adaptive_alpha pull apart from two_level AND
# what makes adaptive_entropy meaningful (its entropy->alpha map only matters where alpha
# actually varies). gamma 3 and 7 are dropped vs the prior pilot to keep the 4-condition
# matrix at a sane size; the three gammas kept still span the alpha range.
#
# CONDITIONS — all four phase-DVFS policies, because the entropy policy is a primary
# thing to validate (per request):
#   off            — per-cell SD-without-DVFS baseline (what savings_vs_off compares to).
#   two_level      — static phase-aware: f_low draft, f_high verify (alpha ignored). The
#                    reference every adaptive policy is judged against.
#   adaptive_alpha — lagging signal: verify clock = verify_freq(EMA of last iter's alpha).
#   adaptive_entropy — LEADING signal: verify clock from alpha_hat = a*exp(b*H) on the
#                    draft's output entropy H, set within the same iteration. Needs the
#                    per-pair (a,b) fit; calibrate_pairs_if_needed() auto-runs a 'collect'
#                    pre-pass (DVFS-off) PER PAIR and writes calibration/fitted_<pair>.json
#                    before the matrix (GELATO baseline a=1,b=-0.35 if a fit yields none).
#                    RISK: this is the least-tested path — it depends on the draft lm_head
#                    entropy hook firing. If last_entropy stays None, core.py degrades it to
#                    the lagging-alpha path (so it would look like adaptive_alpha). Sanity-
#                    check the fit JSON exists and that entropy runs differ from alpha runs.
# Deferred to the full run: fixed_low and coarse (pure granularity/− ablations; they bear
# on neither "does alpha vary" nor "does the leading signal help").
#
# TWO PAIRS (per request): the same matrix is run for BOTH draft choices, so the base vs
# instruct draft can be compared on identical conditions (and each gets its own entropy
# calibration). The base draft should sit at lower alpha, widening the range tested.
PILOT_PAIRS = ["llama_8b_1b", "llama_8b_1b_base"]      # 1B-instruct draft, then 1B-base draft
PILOT_MODEL = PILOT_PAIRS[0]                           # carries the shared vanilla + the f_low sweep
PILOT_GAMMAS = [5, 12, 18]                             # draft sizes -> the alpha axis
PILOT_STRATEGIES = [f"spec_g{g}" for g in PILOT_GAMMAS]
PILOT_DATASETS = ["gsm8k", "humaneval", "cnndm"]       # math / code / summarization
PILOT_CONDITIONS = ["off", "two_level", "adaptive_alpha", "adaptive_entropy"]
PILOT_N_PROMPTS = 64
PILOT_REPS = 1                                         # unchanged (per request)
FULL_REPS = 5

# 2 pairs x 4 conditions x 3 gammas x 3 datasets, plus ONE vanilla (no-SD) reference per
# dataset. Vanilla loads the TARGET only (no draft) and both pairs share the same 8B
# target, so the vanilla baseline is physically identical for both — run it ONCE under
# PILOT_PAIRS[0] (3 runs), NOT per pair. = 3 + (2 * 4 * 3 * 3) = 3 + 72 = 75 runs at REPS=1
# (matches the requested 36 + 36 + 3); the f_low sweep below adds 5 -> 80 total.
# RESUME NOTE: filenames for the llama_8b_1b portion are UNCHANGED from the single-pair
# matrix this replaces, so an existing results/pilot/ with those 39 files already done is
# matched as-is by the skip logic below (both the per-file check and the per-group
# groupby skip) — only the new llama_8b_1b_base rows (36 files) will actually run.
# NOTE for analysis: savings_vs_off is per-cell and works for both pairs. savings_vs_vanilla
# for llama_8b_1b_base must reuse llama_8b_1b's vanilla rows (same target) — the base pair
# has no vanilla rows of its own by design.
PILOT_MATRIX = (
    [(PILOT_MODEL, "vanilla", "off", ds) for ds in PILOT_DATASETS]
    + [(pair, strat, cond, ds)
       for pair in PILOT_PAIRS
       for strat in PILOT_STRATEGIES
       for ds in PILOT_DATASETS
       for cond in PILOT_CONDITIONS]
)

# ---- f_low (draft-clock) sweep: is the draft phase saving no energy because f_low is a
# bad value? ----------------------------------------------------------------------------
# The first pilot saw the draft phase give back ~14-16% GPU power but +30-58% wall time ->
# net-negative energy. A prime suspect is f_low itself: too HIGH and the memory-bound draft
# barely drops power; too LOW and even a memory-bound kernel becomes core-clock-limited and
# the draft stretches out, so the time penalty erases the power saving. There is a sweet
# spot, and 735 MHz may not be it. This sweep finds it empirically.
#
# Design (clean isolation of the DRAFT clock): hold ONE cell fixed and run two_level
# (draft=f_low, verify=f_high) at several f_low values, mutating only mapper.f_low between
# runs. f_floor (the verify floor = 0.6*f_high) is independent of f_low, and verify runs at
# f_high throughout, so verify work/energy is ~constant across the sweep — therefore the
# variation in TOTAL energy is dominated by the draft phase, and the f_low that minimizes
# total energy is the draft-optimal clock. temp=0/seed=42 => identical tokens => identical
# draft WORK across the sweep, so only the clock differs. The cell is the highest gamma
# (largest draft fraction => most sensitive) on the fastest dataset.
#   * Results go to results/flow_sweep/ (a SIBLING of results/pilot/), so they never enter
#     the main aggregation (compute_metrics globs results/<mode>/*.json non-recursively).
#   * 735 is included as the reference point, so the sweep is self-contained.
# This is a DIAGNOSTIC that informs the f_low for the full run; it does not feed back into
# the matrix above (those run at the current F_LOW so the four policies stay comparable).
# One sweep covers BOTH pairs: the 1B-base and 1B-instruct drafts have identical architecture
# (same kernels), so the draft-clock power/time curve — hence the optimal f_low — is the same.
PILOT_FLOW_SWEEP_STRATEGY = "spec_g18"
PILOT_FLOW_SWEEP_DATASET = "gsm8k"
PILOT_FLOW_SWEEP_MHZ = [480, 600, 735, 870, 990]       # snapped to supported levels at runtime
PILOT_FLOW_SWEEP_N_PROMPTS = 32                        # fewer than the matrix — keep 5 runs quick;
                                                       # still ample tokens for a stable kWh reading

# =============================================================================
# THE TWO RIDGE-CROSSING EXPERIMENTS (both answer the SAME question, two ways)
# =============================================================================
# The pilot's roofline was NO-GO: ridge I*=75.9, but measured draft I=41.1 and verify I=58.5
# — both memory-bound, so per-phase DVFS had no clean hardware justification. There are
# exactly two ways to make  I_draft < I* <= I_verify  true, and this file supports both from
# one codebase, selected by --sweep:
#
#   --sweep batch : move I_verify UP past a fixed ridge, by raising the batch size.
#                   I_verify scales with tokens per verify pass (batch x (gamma+1)), because
#                   more sequences amortise the same weight-matrix reads over more FLOPs.
#                   DEPLOYMENT-REALISTIC: batch size is a knob real serving stacks turn.
#
#   --sweep sm    : move the ridge DOWN past a fixed I_verify, by restricting SMs.
#                   I*(N) = (peak_TFLOPS x N/82) / BW. Phase intensities are unchanged (same
#                   kernels, same FLOPs, same bytes); only machine balance moves.
#                   NOT deployment-realistic — it is a controlled instrument for asking "IF
#                   the premise held, would DVFS help more?" Report it as such.
#
# Both write to their own results/<name>_sweep/ dir (siblings of results/pilot/, so they never
# enter the main aggregation, exactly like flow_sweep) and both reuse run_single() unchanged.

# ---- Approach 1: BATCH-SIZE sweep -------------------------------------------------------
# gamma is FIXED at 18 because that is the only gamma whose ridge-crossing batch fits in
# 24GB: at gamma=18 verify scores 19 tokens/seq, so ~11 seqs crosses I*=75.9. At gamma=5 it
# scores 6, needing ~33 seqs — reachable only at short context (see RUNBOOK caveat).
# max_model_len is held FIXED across all batch levels ON PURPOSE: letting it vary with batch
# would change cnndm's truncation point per level, confounding batch against workload. Fixed
# 1024 gives a 720-token prompt budget (1024 - 256 gen - 48 margin), which covers gsm8k and
# humaneval untruncated; cnndm is excluded from the default sweep for that reason.
PILOT_BATCH_SWEEP_STRATEGY = "spec_g18"
PILOT_BATCH_SWEEP_DATASETS = ["gsm8k", "humaneval"]
PILOT_BATCH_SWEEP_CONDITIONS = ["off", "two_level", "adaptive_alpha", "adaptive_entropy"]
PILOT_BATCH_SWEEP_SIZES = [4, 8, 11, 16, 22]   # 2 below the ridge, 3 above (see table below)
PILOT_BATCH_SWEEP_MAX_MODEL_LEN = 1024         # fixed; 22 x 1024 = 22,528 <= 24,272 KV slots
PILOT_BATCH_SWEEP_N_PROMPTS = 32
#   predicted I_verify = 58.5 x (batch x 19) / 152 :
#     batch= 4 -> I= 29.3  memory-bound        batch=16 -> I=117.1  COMPUTE-bound
#     batch= 8 -> I= 58.5  memory-bound (=pilot)  batch=22 -> I=161.0  COMPUTE-bound
#     batch=11 -> I= 80.5  COMPUTE-bound  <- predicted crossing, between 8 and 11

# ---- Approach 2: SM-RESTRICTION sweep ---------------------------------------------------
# The premise window is BOUNDED ON BOTH SIDES — this is the key design fact. Too few SMs and
# the DRAFT also becomes compute-bound, breaking the premise the other way:
#     N=82 (100%) I*=75.9  draft mem  / verify mem      <- reproduces the pilot's NO-GO
#     N=64  (78%) I*=59.2  draft mem  / verify mem      (marginal, just outside)
#     N=56  (68%) I*=51.8  draft mem  / verify COMPUTE  <- PREMISE HOLDS
#     N=48  (59%) I*=44.4  draft mem  / verify COMPUTE  <- PREMISE HOLDS
#     N=40  (49%) I*=37.0  draft COMPUTE / verify COMPUTE  <- broken the OTHER way
# so the ladder deliberately brackets the window rather than sitting inside it.
# Each level needs its OWN PROCESS: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is read once, when the
# CUDA context is created, so it cannot be changed in-process. run_sm_sweep() therefore
# re-invokes this script per level (see --_sm-child).
PILOT_SM_SWEEP_STRATEGY = "spec_g18"
PILOT_SM_SWEEP_DATASETS = ["gsm8k"]
PILOT_SM_SWEEP_CONDITIONS = ["off", "two_level", "adaptive_alpha", "adaptive_entropy"]
PILOT_SM_SWEEP_COUNTS = [82, 64, 56, 48, 40]   # full, marginal, IN-window, IN-window, past it
PILOT_SM_SWEEP_N_PROMPTS = 32
SM_TOTAL = sm_partition.SM_TOTAL_RTX3090       # 82 on the RTX 3090 (GA102)


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
CALIB_N_PROMPTS = 128       # TOTAL across CALIB_DATASETS, split as evenly as possible
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


def calibration_prompts(model_pair, mock=False):
    """CALIB_N_PROMPTS prompts total, split as evenly as possible across CALIB_DATASETS
    (e.g. 128 over 3 types -> 43/43/42). Combined into one list — one build_llm/generate
    pass per model pair, not one per dataset, so calibration cost doesn't multiply."""
    n = len(CALIB_DATASETS)
    base, extra = divmod(CALIB_N_PROMPTS, n)   # extra goes to the first `extra` datasets
    prompts = []
    for i, ds in enumerate(CALIB_DATASETS):
        k = base + (1 if i < extra else 0)
        prompts += load_prompts(ds, model_pair, k, mock=mock)
    return prompts


def calibrate_pairs_if_needed(matrix, mock=False, recalibrate=False):
    pairs = [mp for mp in dict.fromkeys(m[0] for m in matrix)
             if any(x[0] == mp and x[2] == "adaptive_entropy" for x in matrix)
             and (recalibrate or not calibration_path(mp).exists())]
    if not pairs:
        return
    print(f"== AUTO-CALIBRATION for {pairs} ==")
    for mp in pairs:
        prompts = calibration_prompts(mp, mock=mock)
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

# Leave room for generation inside max_model_len. cnndm articles routinely exceed
# the prompt budget (~1792 tokens after the 256-token reservation at
# max_model_len=2048), and vLLM 0.6.6 raises "prompt too long" and ABORTS the run
# rather than truncating — so without this guard the cnndm cells would crash. We
# truncate the *payload text* (instruction prefix + article), keeping the HEAD,
# where CNN/DM summary content concentrates; the template's special tokens are
# preserved. Truncation is deterministic (tokenizer + fixed budget), so the
# temp=0/seed=42 reproducibility contract is intact. Only ever fires for cnndm;
# gsm8k/humaneval prompts sit far under budget and pass through unchanged.
PROMPT_BUDGET_MARGIN = 48   # slack for decode/re-encode drift + chat-template tokens

_TOKENIZER_CACHE = {}


def _get_tokenizer(model_pair):
    tok = _TOKENIZER_CACHE.get(model_pair)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_PAIRS[model_pair]["target"])
        _TOKENIZER_CACHE[model_pair] = tok
    return tok


def _fit_prompt_text(text, tmpl, model_pair, max_model_len=None):
    """Truncate `text` (kept head) so tmpl.format(text=...) fits the prompt budget.

    max_model_len is a parameter (not the global) because the batch sweep runs at a SHORTER
    context (1024) than the main matrix (2048); using the global there would build prompts too
    long for the engine and vLLM 0.6.6 ABORTS on an over-length prompt rather than truncating.
    """
    tok = _get_tokenizer(model_pair)
    mml = MAX_MODEL_LEN if max_model_len is None else int(max_model_len)
    budget = mml - MAX_NEW_TOKENS - PROMPT_BUDGET_MARGIN
    if budget <= 0:
        raise ValueError(
            f"max_model_len={mml} leaves no prompt budget: {mml} - {MAX_NEW_TOKENS} (generation) "
            f"- {PROMPT_BUDGET_MARGIN} (margin) = {budget}. Raise max_model_len above "
            f"{MAX_NEW_TOKENS + PROMPT_BUDGET_MARGIN}.")
    if len(tok(tmpl.format(text=text), add_special_tokens=False)["input_ids"]) <= budget:
        return text
    overhead = len(tok(tmpl.format(text=""), add_special_tokens=False)["input_ids"])
    text_ids = tok(text, add_special_tokens=False)["input_ids"][:max(0, budget - overhead)]
    return tok.decode(text_ids)


def load_prompts(dataset_key, model_pair, n=None, mock=False, max_model_len=None):
    fname, prefix = DATASETS[dataset_key]
    family = MODEL_PAIRS[model_pair]["family"]
    tmpl = PROMPT_TEMPLATES[family]
    path = SAMPLED_DIR / fname
    if mock and not path.exists():
        # Laptop dry-run before datasets are prepared: synthesize prompts so the
        # harness/JSON/resume logic can be validated with no data files present.
        k = n if n is not None else 8
        return [tmpl.format(text=prefix + f"Mock prompt {i} for harness validation.")
                for i in range(k)]
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data["samples"][:n] if n is not None else data["samples"]
    texts = [prefix + s["text"] for s in samples]
    if not mock:
        texts = [_fit_prompt_text(t, tmpl, model_pair, max_model_len) for t in texts]
    return [tmpl.format(text=t) for t in texts]


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


def build_llm(model_pair, strategy, mock=False, max_num_seqs=None, max_model_len=None):
    """Build the engine. max_num_seqs / max_model_len default to the module constants but are
    overridable per-build: the batch sweep (Approach 1) varies max_num_seqs, and lowers
    max_model_len so the larger batch still fits the fixed KV pool."""
    if mock:
        return _MockLLM()
    from vllm import LLM
    max_num_seqs = MAX_NUM_SEQS if max_num_seqs is None else int(max_num_seqs)
    max_model_len = MAX_MODEL_LEN if max_model_len is None else int(max_model_len)
    target = MODEL_PAIRS[model_pair]["target"]
    draft = MODEL_PAIRS[model_pair]["draft"]
    cfg = STRATEGIES[strategy]
    # bf16, quantization OFF (see the FAILED-approaches note in the handoff):
    #   * bitsandbytes crashes the 0.6.6 spec-decode draft path (load_format is
    #     engine-wide, so the draft can't be bnb-loaded while the target is bf16, and
    #     on-the-fly bnb of Llama-3.2-1B's fused gate_up_proj breaks);
    #   * bf16 fits both models in 24GB, avoids NF4 dequant traffic (cleaner energy +
    #     roofline), and yields higher alpha than NF4 would.
    # max_model_len/max_num_seqs/enforce_eager are the knobs that made the pilot fit;
    # without them the engine OOMs (4096 context; CUDA-graph capture after KV sizing).
    common = dict(model=target, dtype="bfloat16",
                  max_model_len=max_model_len, max_num_seqs=max_num_seqs,
                  enforce_eager=True, gpu_memory_utilization=0.90, seed=SEED)
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

def _set_draft_clock(controller, f_low):
    """Pin the draft-phase clock (mapper.f_low) for this run. f_low is the only thing the
    sweep varies; mutating it does NOT touch f_floor (verify floor = 0.6*f_high), and the
    controller is built once per group, so we set this every run (idempotent) to avoid a
    swept value leaking into a later default-f_low run."""
    if controller is None:
        return
    mapper = getattr(controller, "mapper", None)
    if mapper is None:
        return
    if f_low >= mapper.f_high:
        raise ValueError(f"f_low {f_low} must be < f_high {mapper.f_high}")
    mapper.f_low = int(f_low)


def run_single(llm, controller, model_pair, strategy, condition, dataset, rep,
               prompts, out_path, mock=False, f_low=None,
               batch_size=None, max_model_len=None):
    print(f"  RUN  {out_path.name}")
    wait_for_temp(mock=mock)
    if WARMUP_PROMPTS and not mock:
        _ = llm.generate(prompts[:WARMUP_PROMPTS], _sampling_params())

    configure_dvfs(controller, condition, mock=mock)
    f_low_used = int(f_low) if f_low is not None else F_LOW
    _set_draft_clock(controller, f_low_used)   # default F_LOW, or the swept draft clock

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
        "max_new_tokens": MAX_NEW_TOKENS, "f_high": F_HIGH, "f_low": f_low_used,
        # --- ridge-crossing experiment axes (recorded on EVERY run, sweep or not, so the
        # main pilot rows are directly comparable against sweep rows) ---
        # sm_count is read from the live environment rather than passed in, so it reflects
        # what this PROCESS actually got (MPS is set at process launch), not an intention.
        "batch_size": int(batch_size) if batch_size is not None else MAX_NUM_SEQS,
        "max_model_len": int(max_model_len) if max_model_len is not None else MAX_MODEL_LEN,
        "sm_count": sm_partition.active_sm_count(SM_TOTAL) or SM_TOTAL,
        "sm_total": SM_TOTAL,
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


def _snap_flow_values(values, mock=False):
    """Snap requested f_low candidates to the GPU's supported graphics clocks (same rule
    resolve_clocks uses for the anchors) and de-dupe. In mock there is no GPU, so pass
    through. Returns ints in the original order."""
    if mock:
        return [int(v) for v in values]
    import pynvml as N
    N.nvmlInit()
    h = N.nvmlDeviceGetHandleByIndex(0)
    mem = N.nvmlDeviceGetSupportedMemoryClocks(h)
    gfx = sorted(N.nvmlDeviceGetSupportedGraphicsClocks(h, mem[0]))
    N.nvmlShutdown()
    snap = lambda f: min(gfx, key=lambda g: abs(g - f))
    out, seen = [], set()
    for v in values:
        s = snap(int(v))
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def run_flow_sweep(n_prompts, mock=False):
    """Draft-clock (f_low) sweep — find the f_low that minimises draft-phase energy.

    Fixed cell (PILOT_FLOW_SWEEP_{STRATEGY,DATASET}), two_level policy, only mapper.f_low
    varies. Verify stays at f_high, so total-energy variation across the sweep is dominated
    by the draft phase and the minimum-energy f_low is the draft-optimal clock. Writes to
    results/flow_sweep/ (a sibling of results/<mode>/, kept out of the main aggregation) and
    prints a summary sorted by energy. Builds its own LLM (the matrix already tore its down).
    """
    mp, strat, ds = PILOT_MODEL, PILOT_FLOW_SWEEP_STRATEGY, PILOT_FLOW_SWEEP_DATASET
    flows = _snap_flow_values(PILOT_FLOW_SWEEP_MHZ, mock=mock)
    out_dir = RESULTS_DIR / "flow_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {f: out_dir / f"{mp}__{strat}__two_level__{ds}__flow{f}__rep0.json" for f in flows}

    print(f"\n== f_low SWEEP  {mp}/{strat}/{ds}  two_level  f_low(snapped)={flows} ==")
    if all(p.exists() for p in paths.values()):
        print(f"   skip (all {len(flows)} sweep runs done)")
    else:
        prompts = load_prompts(ds, mp, n_prompts, mock=mock)
        llm = build_llm(mp, strat, mock)
        controller = retrieve_controller(llm, mock)
        if controller is None and not mock:
            print("   WARN: no controller on the worker — f_low sweep will not actually move "
                  "the clock (check install()/in-process worker).")
        try:
            for f in flows:
                if paths[f].exists():
                    print(f"   skip (done): {paths[f].name}")
                    continue
                run_single(llm, controller, mp, strat, "two_level", ds, 0,
                           prompts, paths[f], mock=mock, f_low=f)
        finally:
            del llm, controller
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            reset_clocks(mock)

    # Summary: read back the sweep JSONs and rank by energy (the optimum is the min).
    # energy_kwh is a dict ({"total_kwh","gpu_kwh","cpu_kwh","ram_kwh"} — see EnergyMeter.stop()),
    # not a scalar; total_kwh is the field to rank by here (same "total is a valid proxy for
    # draft energy since verify is held at f_high" reasoning as the rest of this function).
    rows = []
    for f in flows:
        try:
            d = json.loads(paths[f].read_text())
            energy_total = (d.get("energy_kwh") or {}).get("total_kwh")
            rows.append((f, energy_total, d.get("wall_time_s")))
        except Exception:
            continue
    have = [r for r in rows if r[1] is not None]
    if have:
        best = min(have, key=lambda r: r[1])
        print("   f_low(MHz)  energy_kWh      wall_s   (sorted by energy; * = min)")
        for f, e, w in sorted(have, key=lambda r: r[1]):
            star = " *" if f == best[0] else "  "
            ws = f"{w:8.1f}" if isinstance(w, (int, float)) else f"{str(w):>8}"
            print(f"  {star}{f:8d}  {e:12.6f}  {ws}")
        print(f"   -> draft-optimal f_low this cell: {best[0]} MHz "
              f"(vs current F_LOW={F_LOW}). Re-confirm on a higher-gamma/low-alpha cell "
              f"before adopting for the full run.")


# =============================================================================
# RIDGE-CROSSING SWEEPS (Approach 1: batch size / Approach 2: SM restriction)
# =============================================================================

def _sweep_summary(rows, axis_name, extra_note=""):
    """Shared energy-ranked summary for both sweeps (same shape as run_flow_sweep's).

    Reads energy_kwh as a DICT ({"total_kwh",...}) — the same schema fix run_flow_sweep needed;
    comparing the dict itself raises TypeError on the min()/sorted() below.
    """
    have = [r for r in rows if r[1] is not None]
    if not have:
        print("   (no readable results to summarise)")
        return
    print(f"   {axis_name:>10}  {'cond':<17}{'gpu_kWh':>11}  {'wall_s':>8}  {'tok/s':>8}")
    for a, e, w, cond, tok in sorted(have, key=lambda r: (r[0], r[3])):
        ws = f"{w:8.1f}" if isinstance(w, (int, float)) else f"{str(w):>8}"
        ts = f"{tok/w:8.1f}" if isinstance(w, (int, float)) and w and tok else f"{'-':>8}"
        print(f"   {a:>10}  {cond:<17}{e:11.6f}  {ws}  {ts}")
    if extra_note:
        print(extra_note)


def _read_sweep_rows(paths_by_axis):
    rows = []
    for a, p in paths_by_axis.items():
        try:
            d = json.loads(p.read_text())
            rows.append((a, (d.get("energy_kwh") or {}).get("gpu_kwh"),
                         d.get("wall_time_s"), d.get("dvfs_condition"),
                         d.get("total_tokens")))
        except Exception:
            continue
    return rows


def run_batch_sweep(mock=False):
    """APPROACH 1 — sweep MAX_NUM_SEQS to push verify's arithmetic intensity past the ridge.

    Fixed cell (gamma=18, per-dataset), fixed max_model_len, all four DVFS conditions at each
    batch level. Raising batch raises tokens-per-verify-pass, which raises I_verify (the same
    weight bytes are amortised over more FLOPs) while I_draft barely moves — so the phases
    separate across the ridge WITHOUT touching the hardware. That is what makes this the
    deployment-realistic of the two approaches: batch size is a knob real servers turn.

    One engine build per batch level (max_num_seqs is a construction-time argument), then all
    conditions x datasets run inside that build. Resumable per file, like the matrix.
    """
    strat = PILOT_BATCH_SWEEP_STRATEGY
    mml = PILOT_BATCH_SWEEP_MAX_MODEL_LEN
    mp = PILOT_MODEL
    out_dir = RESULTS_DIR / "batch_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n== BATCH SWEEP  {mp}/{strat}  max_model_len={mml}  "
          f"batches={PILOT_BATCH_SWEEP_SIZES}  datasets={PILOT_BATCH_SWEEP_DATASETS} ==")
    print(f"   KV budget: {KV_TOTAL_TOKEN_SLOTS:,} token slots; worst case at batch "
          f"{max(PILOT_BATCH_SWEEP_SIZES)} x {mml} = "
          f"{max(PILOT_BATCH_SWEEP_SIZES)*mml:,} -> "
          f"{'fits' if max(PILOT_BATCH_SWEEP_SIZES)*mml <= KV_TOTAL_TOKEN_SLOTS else 'OVER BUDGET'}")

    def path_for(bs, cond, ds):
        return out_dir / f"{mp}__{strat}__{cond}__{ds}__bs{bs}__rep0.json"

    for bs in PILOT_BATCH_SWEEP_SIZES:
        cells = [(cond, ds) for ds in PILOT_BATCH_SWEEP_DATASETS
                 for cond in PILOT_BATCH_SWEEP_CONDITIONS]
        if all(path_for(bs, c, d).exists() for c, d in cells):
            print(f"   skip batch={bs} (all {len(cells)} runs done)")
            continue
        print(f"   -- batch={bs}  (predicted I_verify ~ "
              f"{58.53 * (bs * 19) / 152.0:.0f} FLOPs/byte vs ridge 75.9) --")
        llm = build_llm(mp, strat, mock, max_num_seqs=bs, max_model_len=mml)
        controller = retrieve_controller(llm, mock)
        if controller is not None:
            controller.entropy_a, controller.entropy_b = entropy_coeffs_for(mp)
        if controller is None and not mock:
            print("   WARN: no controller on the worker — DVFS conditions will be inactive.")
        try:
            prompt_cache = {}
            for cond, ds in cells:
                p = path_for(bs, cond, ds)
                if p.exists():
                    print(f"   skip (done): {p.name}")
                    continue
                if ds not in prompt_cache:
                    prompt_cache[ds] = load_prompts(ds, mp, PILOT_BATCH_SWEEP_N_PROMPTS,
                                                    mock=mock, max_model_len=mml)
                run_single(llm, controller, mp, strat, cond, ds, 0, prompt_cache[ds], p,
                           mock=mock, batch_size=bs, max_model_len=mml)
        finally:
            del llm, controller
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            reset_clocks(mock)

    paths = {}
    for bs in PILOT_BATCH_SWEEP_SIZES:
        for ds in PILOT_BATCH_SWEEP_DATASETS:
            for cond in PILOT_BATCH_SWEEP_CONDITIONS:
                p = path_for(bs, cond, ds)
                if p.exists():
                    paths[(bs, cond, ds)] = p
    rows = []
    for (bs, cond, ds), p in paths.items():
        try:
            d = json.loads(p.read_text())
            rows.append((bs, (d.get("energy_kwh") or {}).get("gpu_kwh"),
                         d.get("wall_time_s"), f"{cond}/{ds}", d.get("total_tokens")))
        except Exception:
            continue
    print("\n   === batch sweep summary (GPU energy; compare conditions WITHIN a batch) ===")
    _sweep_summary(rows, "batch",
                   "   -> Next: re-capture the roofline at each batch level and check WHERE\n"
                   "      verify actually crosses (predicted between 8 and 11):\n"
                   "         python profiling/prof_roofline.py --gamma 18 --max-num-seqs <B> \\\n"
                   "             --max-model-len 1024 --selftest      (then the ncu commands)")


def run_sm_sweep(mock=False, only_sm=None):
    """APPROACH 2 — restrict SMs so the RIDGE falls through the fixed phase intensities.

    Two-level design, forced by how MPS works:
      * PARENT (only_sm=None): loops the SM ladder and re-invokes THIS SCRIPT once per level
        as a child process, with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE set in the child's env.
        A subprocess per level is REQUIRED, not a stylistic choice: the driver reads that
        variable when the CUDA context is created, so it cannot be changed in-process.
      * CHILD (only_sm=N): runs the cells for exactly one SM level in its own CUDA context.

    The parent refuses to run if the MPS daemon is down, because without it the percentage is
    silently ignored and all five "levels" would be identical full-GPU runs that merely look
    like a sweep.
    """
    strat, mp = PILOT_SM_SWEEP_STRATEGY, PILOT_MODEL
    out_dir = RESULTS_DIR / "sm_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    def path_for(sm, cond, ds):
        return out_dir / f"{mp}__{strat}__{cond}__{ds}__sm{sm}__rep0.json"

    # ---------------- CHILD: one SM level, inside its own CUDA context ----------------
    if only_sm is not None:
        sm = int(only_sm)
        eff = sm_partition.active_sm_count(SM_TOTAL) or SM_TOTAL
        print(f"   [sm={sm}] child process; env-reported effective SMs={eff}; "
              f"predicted ridge I*={sm_partition.ridge_point(sm, sm_total=SM_TOTAL):.1f}")
        cells = [(cond, ds) for ds in PILOT_SM_SWEEP_DATASETS
                 for cond in PILOT_SM_SWEEP_CONDITIONS]
        if all(path_for(sm, c, d).exists() for c, d in cells):
            print(f"   [sm={sm}] skip (all {len(cells)} runs done)")
            return
        install_dvfs(mock)
        llm = build_llm(mp, strat, mock)
        controller = retrieve_controller(llm, mock)
        if controller is not None:
            controller.entropy_a, controller.entropy_b = entropy_coeffs_for(mp)
        try:
            prompt_cache = {}
            for cond, ds in cells:
                p = path_for(sm, cond, ds)
                if p.exists():
                    print(f"   [sm={sm}] skip (done): {p.name}")
                    continue
                if ds not in prompt_cache:
                    prompt_cache[ds] = load_prompts(ds, mp, PILOT_SM_SWEEP_N_PROMPTS, mock=mock)
                run_single(llm, controller, mp, strat, cond, ds, 0, prompt_cache[ds], p,
                           mock=mock)
        finally:
            del llm, controller
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            reset_clocks(mock)
        return

    # ---------------- PARENT: drive one child per SM level ----------------
    lo, hi = sm_partition.premise_window(41.12, 58.53, sm_total=SM_TOTAL)
    print(f"\n== SM SWEEP  {mp}/{strat}  counts={PILOT_SM_SWEEP_COUNTS} of {SM_TOTAL} ==")
    print(f"   premise window (measured I_draft=41.1 < I* <= I_verify=58.5): "
          f"SMs in [{int(lo)+1}..{int(hi)}]  ({(int(lo)+1)/SM_TOTAL*100:.0f}%-{int(hi)/SM_TOTAL*100:.0f}% of GPU)")
    for n in PILOT_SM_SWEEP_COUNTS:
        r = sm_partition.ridge_point(n, sm_total=SM_TOTAL)
        holds = 41.12 < r <= 58.53
        print(f"     {n:>3} SMs ({n/SM_TOTAL*100:>3.0f}%)  predicted I*={r:5.1f}  "
              f"{'PREMISE HOLDS' if holds else ('both compute-bound' if r <= 41.12 else 'both memory-bound')}")

    if not mock and not sm_partition.preflight(min(PILOT_SM_SWEEP_COUNTS), SM_TOTAL, mock):
        print("   ABORT: refusing to run an SM sweep that cannot actually restrict SMs.")
        print("          Start MPS then re-run:  nvidia-cuda-mps-control -d")
        return

    for sm in PILOT_SM_SWEEP_COUNTS:
        cells = [(c, d) for d in PILOT_SM_SWEEP_DATASETS for c in PILOT_SM_SWEEP_CONDITIONS]
        if all(path_for(sm, c, d).exists() for c, d in cells):
            print(f"   skip sm={sm} (all {len(cells)} runs done)")
            continue
        env = sm_partition.env_for_sm_limit(sm, SM_TOTAL)
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--mode", "pilot", "--sweep", "sm", "--_sm-child", str(sm)]
        if mock:
            cmd.append("--mock")
        print(f"   -> launching child for sm={sm} "
              f"({sm_partition.sm_to_percent(sm, SM_TOTAL)}% MPS thread cap)")
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0:
            print(f"   !! child for sm={sm} exited {rc} — continuing to next level "
                  f"(re-run to retry; finished cells are skipped)")

    paths = {}
    for sm in PILOT_SM_SWEEP_COUNTS:
        for ds in PILOT_SM_SWEEP_DATASETS:
            for cond in PILOT_SM_SWEEP_CONDITIONS:
                p = path_for(sm, cond, ds)
                if p.exists():
                    paths[(sm, cond, ds)] = p
    rows = []
    for (sm, cond, ds), p in paths.items():
        try:
            d = json.loads(p.read_text())
            rows.append((sm, (d.get("energy_kwh") or {}).get("gpu_kwh"),
                         d.get("wall_time_s"), f"{cond}/{ds}", d.get("total_tokens")))
        except Exception:
            continue
    print("\n   === SM sweep summary (GPU energy; compare conditions WITHIN an SM level) ===")
    _sweep_summary(rows, "SMs",
                   "   -> Next: re-capture the roofline AT EACH SM LEVEL and pass --sm-count so\n"
                   "      analyze_roofline.py scales the peak and moves the ridge:\n"
                   "         python profiling/analyze_roofline.py --sm-count <N> \\\n"
                   "             --draft-csv ... --verify-csv ...\n"
                   "      The MODEL picks the sweep points; the MEASUREMENT decides the verdict.")


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
    # ---- ridge-crossing experiment selector (both approaches, one codebase) ----
    ap.add_argument("--sweep", choices=["none", "batch", "sm"], default="none",
                    help="none = the standard 75-run pilot matrix + f_low sweep (default, "
                         "unchanged); batch = Approach 1 (raise MAX_NUM_SEQS to lift I_verify "
                         "past the ridge); sm = Approach 2 (restrict SMs to lower the ridge "
                         "through the phase intensities)")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="override MAX_NUM_SEQS for a one-off run (the batch sweep sets this "
                         "per level automatically)")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="override MAX_MODEL_LEN for a one-off run")
    ap.add_argument("--_sm-child", type=int, default=None,
                    help=argparse.SUPPRESS)   # internal: one SM level, set by run_sm_sweep()
    args = ap.parse_args()

    global MAX_NUM_SEQS, MAX_MODEL_LEN
    if args.max_num_seqs is not None:
        MAX_NUM_SEQS = args.max_num_seqs
    if args.max_model_len is not None:
        MAX_MODEL_LEN = args.max_model_len

    global F_HIGH, F_LOW
    F_HIGH, F_LOW = resolve_clocks(args.mock, args.allow_clock_mismatch)
    reset_clocks(args.mock)   # clean start: clear any lock left by a prior/crashed run

    # ---- SWEEP DISPATCH ---------------------------------------------------------------
    # Both ridge-crossing experiments short-circuit the standard matrix: they are separate,
    # self-contained studies writing to their own results/<name>_sweep/ dir, so they neither
    # read nor write results/pilot/ and cannot disturb the 75 runs already collected.
    # --sweep none (the default) leaves the original behaviour byte-for-byte unchanged.
    if args._sm_child is not None:
        install_dvfs(args.mock)
        try:
            run_sm_sweep(mock=args.mock, only_sm=args._sm_child)
        finally:
            reset_clocks(args.mock)
        return
    if args.sweep in ("batch", "sm"):
        install_dvfs(args.mock)
        # Entropy coefficients are reused from the main pilot's fit; the sweeps deliberately
        # do NOT recalibrate (a per-level refit would confound the axis under test with a
        # changing controller). Warn if no fit exists rather than silently using the fallback.
        if not calibration_path(PILOT_MODEL).exists():
            print(f"    NOTE: no calibration/fitted_{PILOT_MODEL}.json — adaptive_entropy will "
                  f"use the GELATO baseline (a=1,b=-0.35). Run the main pilot first for a fit.")
        completed_sweep = False
        try:
            if args.sweep == "batch":
                run_batch_sweep(mock=args.mock)
            else:
                run_sm_sweep(mock=args.mock)
            completed_sweep = True
        finally:
            reset_clocks(args.mock)
            bar = "#" * 74
            print("\n" + bar)
            print(f"##  {args.sweep.upper()} SWEEP {'COMPLETE' if completed_sweep else 'INTERRUPTED'}"
                  f"  —  GPU clocks reset.".ljust(70) + "##")
            print(f"##  results: {str(RESULTS_DIR / (args.sweep + '_sweep')):<57}##")
            print("##  Re-run the SAME command to resume (finished runs are skipped).".ljust(72) + "##")
            print(bar)
        return
    # -----------------------------------------------------------------------------------

    matrix = PILOT_MATRIX if args.mode == "pilot" else build_full_matrix()
    reps = PILOT_REPS if args.mode == "pilot" else FULL_REPS
    n_prompts = args.n_prompts if args.n_prompts is not None else (
        PILOT_N_PROMPTS if args.mode == "pilot" else None)
    out_dir = RESULTS_DIR / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    n_extra = len(PILOT_FLOW_SWEEP_MHZ) if args.mode == "pilot" else 0
    print(f"mode={args.mode}  combos={len(matrix)}  reps={reps}  "
          f"matrix_runs={len(matrix) * reps}"
          + (f"  + {n_extra} f_low-sweep runs = {len(matrix) * reps + n_extra} total" if n_extra else "")
          + f"  mock={args.mock}  f_high={F_HIGH} f_low={F_LOW}")
    print(f"  conditions: {PILOT_CONDITIONS if args.mode == 'pilot' else FULL_CONDITIONS}")

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
                    prompt_cache[ds] = load_prompts(ds, mp, n_prompts, mock=args.mock)
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
        # After the matrix: the draft-clock (f_low) sweep — pilot only. Its own resume/skip
        # logic and a sibling output dir keep it independent of the matrix above.
        if args.mode == "pilot":
            run_flow_sweep(PILOT_FLOW_SWEEP_N_PROMPTS, mock=args.mock)
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
#       python experiments/run_experiment.py --mode pilot   # 39 matrix runs + 5 f_low-sweep
#    On first run, an entropy calibration pre-pass auto-fits a=exp(b*H) for llama_8b_1b
#    (adaptive_entropy is in the matrix) and writes calibration/fitted_llama_8b_1b.json.
#    Primary check #1 (alpha varies): tabulate alpha_mean by (strategy, dataset); it must
#    fall as gamma grows and reach the responsive band (< 0.7) toward the (cnndm, spec_g18)
#    corner. If alpha is pinned high everywhere, the adaptive modes are still open-loop.
#    Primary check #2 (entropy path is live, not degraded): confirm calibration/fitted_*.json
#    exists, and that adaptive_entropy runs differ from adaptive_alpha runs (clock trace
#    and/or energy). If they are identical, last_entropy never populated and core.py fell
#    back to the lagging-alpha path — the entropy lm_head hook isn't firing.
#    Secondary checks: gpu_monitor.sm_clock_unique_mhz > 1 for the adaptive modes, == 1 for
#    off; in HIGH-alpha cells the adaptive policies should sit near f_high and in LOW-alpha
#    cells drop below it (and below two_level) — that divergence is the adaptive logic working.
#    f_low SWEEP: results/flow_sweep/ holds two_level @ spec_g18/gsm8k across f_low; the run
#    prints an energy-ranked table and the draft-optimal f_low. If the min is well below the
#    current F_LOW=735, the draft phase was losing energy to a too-high low-clock — adopt the
#    swept optimum for the full run (re-confirm on a low-alpha cell first). This is a separate
#    question from the roofline (which still needs the clean ncu capture).
# 2. Full sweep (resumable):  python experiments/run_experiment.py --mode full
# 3. Post-process on the laptop: evaluation/compute_metrics.py -> aggregate -> figures
#    (gamma_trend.csv + savings_vs_off.csv are the per-cell outputs this pilot feeds;
#    the flow_sweep/ dir is analyzed separately — it is intentionally out of that pipeline).