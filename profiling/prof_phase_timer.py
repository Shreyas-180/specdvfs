#!/usr/bin/env python
"""Phase-1 driver — per-phase (draft vs verify) GPU timing via CUDA events.

Runs a short speculative-decoding job with a PhaseTimer attached at the SAME hook
points the DVFS patch uses (so they can never drift), then dumps draft_ms /
verify_ms and the verify/draft ratio. Run it across the pilot gammas to watch the
draft-time fraction grow with the draft length — the empirical basis for
"savings scale with gamma".

This is NOT a DVFS run: install_timing() and the DVFS install() are mutually
exclusive. By default the GPU clock is locked to F_HIGH for the duration so the
absolute timings are at one known, stable frequency (the verify/draft *ratio* is
fairly clock-robust, but locking removes the governor as a confound). Pass
--no-lock to profile at the default/live clock instead.

Model/build/prompt config is imported from experiments.run_experiment so a
profiling run uses byte-for-byte the same model, dtype (bf16), seed and prompts
as the measured runs.

USAGE (on the VM, after setup_vm.sh + the clock-lock GO and HF login):
    python profiling/prof_phase_timer.py --gamma 5
    python profiling/prof_phase_timer.py --gamma 3
    python profiling/prof_phase_timer.py --gamma 7
Outputs: profiling/out/phase_times_<pair>_g<gamma>.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the experiment's config + builders (single source of truth for build args).
from experiments.run_experiment import (        # noqa: E402
    MODEL_PAIRS, STRATEGIES, F_HIGH, SEED, build_llm, load_prompts,
)
from profiling.phase_timer import install_timing  # noqa: E402

# Constant-gamma SD strategies available (tracks spec_g12/g18 added for the pilot).
_GAMMA_STRATS = sorted(int(s[len("spec_g"):]) for s in STRATEGIES
                       if s.startswith("spec_g") and s[len("spec_g"):].isdigit())

OUT_DIR = PROJECT_ROOT / "profiling" / "out"

# Worker layouts to search for the in-process timer (mirrors run_experiment).
_WORKER_PATHS = (("llm_engine", "model_executor", "driver_worker"),
                 ("model_executor", "driver_worker"),
                 ("driver_worker",))


def _find_attr(llm, attr):
    """Return obj.<attr> for the first reachable worker that has it, else None."""
    for path in _WORKER_PATHS:
        obj = llm
        for a in path:
            obj = getattr(obj, a, None)
            if obj is None:
                break
        if obj is None:
            continue
        if getattr(obj, attr, None) is not None:
            return getattr(obj, attr)
        w = getattr(obj, "worker", None)
        if w is not None and getattr(w, attr, None) is not None:
            return getattr(w, attr)
    return None


def _lock_clock(freq_mhz):
    import pynvml as N
    N.nvmlInit()
    N.nvmlDeviceSetGpuLockedClocks(N.nvmlDeviceGetHandleByIndex(0), freq_mhz, freq_mhz)
    N.nvmlShutdown()
    print(f"  locked GPU clock to {freq_mhz} MHz for the timing run")


def _reset_clock():
    try:
        import pynvml as N
        N.nvmlInit()
        N.nvmlDeviceResetGpuLockedClocks(N.nvmlDeviceGetHandleByIndex(0))
        N.nvmlShutdown()
        print("  reset GPU clock")
    except Exception as e:
        print("  WARN: could not reset clock:", e)


def main():
    ap = argparse.ArgumentParser(description="Phase-1 per-phase CUDA-event timing")
    ap.add_argument("--model-pair", default="llama_8b_1b", choices=list(MODEL_PAIRS))
    ap.add_argument("--gamma", type=int, default=5,
                    help=f"num_speculative_tokens; must map to a spec_g{{N}} strategy "
                         f"(one of {_GAMMA_STRATS})")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--lock-mhz", type=int, default=F_HIGH,
                    help="clock to lock during timing (default F_HIGH)")
    ap.add_argument("--no-lock", action="store_true", help="do not lock the clock")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    strat = f"spec_g{args.gamma}"
    if strat not in STRATEGIES:
        sys.exit(f"--gamma {args.gamma} has no strategy '{strat}'. "
                 f"Phase timing expects gamma in {_GAMMA_STRATS} (the constant-gamma strategies).")

    out = Path(args.out) if args.out else (
        OUT_DIR / f"phase_times_{args.model_pair}_g{args.gamma}.json")

    if not args.no_lock:
        _lock_clock(args.lock_mhz)
    try:
        install_timing()                          # MUST be before building the LLM
        print(f"  building {args.model_pair} / {strat} ...")
        llm = build_llm(args.model_pair, strat, mock=False)
        prompts = load_prompts(args.dataset, args.model_pair, args.n_prompts)

        from vllm import SamplingParams
        print(f"  timing {len(prompts)} prompts x {args.max_tokens} tokens ...")
        llm.generate(prompts, SamplingParams(temperature=0.0,
                                             max_tokens=args.max_tokens, seed=SEED))

        timer = _find_attr(llm, "_phase_timer")
        if timer is None:
            sys.exit("FAIL: no _phase_timer on the worker — the timing patch did not "
                     "attach in-process. Re-check the worker path / vLLM build.")
        timer.save(str(out))
    finally:
        if not args.no_lock:
            _reset_clock()


if __name__ == "__main__":
    main()