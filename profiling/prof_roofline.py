#!/usr/bin/env python
"""Phase-1 driver — place NVTX ranges on draft/verify and run a short SD generate.

Run this UNDER Nsight Compute (ncu), scoped to each NVTX range, to read
authoritative per-phase FLOPs + DRAM bytes; then feed those numbers to
analyze_roofline.py for the GO/NO-GO premise verdict (draft memory-bound, verify
compute-bound). install_roofline() and the DVFS install() are mutually exclusive —
this is a fixed-clock profiling run, not a DVFS run.

ncu controls the GPU clock during profiling (locks to base by default; intensity is a
clock-independent ratio, so this does not affect the verdict). This driver builds the
LLM EAGER (build_llm sets enforce_eager) because ncu cannot replay CUDA-graph kernels,
and brackets only the generate in the CUDA profiler region so model load is not profiled.
Keep the workload TINY: ncu replays kernels, so a full-length generate is the main reason
a prior run appeared to hang.

Model/build/prompt config is imported from experiments.run_experiment so the
profiled phases match the real runs.

USAGE (on the VM, after setup + HF login):
    # 0) PRE-FLIGHT (no ncu): confirm the NVTX hooks fire in-process.
    python profiling/prof_roofline.py --gamma 5 --selftest

    # 1) DRAFT then VERIFY, bounded so ncu can't hang (see profiling/roofline.py USAGE
    #    for the full $M metric list incl. the tensor-core op metric):
    ncu --profile-from-start off --nvtx --nvtx-include "specdvfs_draft/"  \
        --metrics "$M" --target-processes all --csv \
        python profiling/prof_roofline.py --gamma 5 --n-prompts 1 --max-tokens 8 \
        > profiling/out/draft.csv
    ncu --profile-from-start off --nvtx --nvtx-include "specdvfs_verify/" \
        --metrics "$M" --target-processes all --csv \
        python profiling/prof_roofline.py --gamma 5 --n-prompts 1 --max-tokens 8 \
        > profiling/out/verify.csv
    # add --replay-mode application if a custom kernel errors under kernel-replay.

    # 2) verdict + plot:
    python profiling/analyze_roofline.py \
        --draft-csv profiling/out/draft.csv --verify-csv profiling/out/verify.csv

Quick whole-run FLOP sanity WITHOUT ncu (order of magnitude, non-tensor only):
    python profiling/prof_roofline.py --flops-sanity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_experiment import (        # noqa: E402
    MODEL_PAIRS, STRATEGIES, SEED, build_llm, load_prompts,
)
from profiling.roofline import install_roofline, range_call_counts  # noqa: E402

# Strategies that are a constant-gamma SD config (the only kind that produces a
# draft/verify phase split). Derived from STRATEGIES so it tracks spec_g12/g18.
_GAMMA_STRATS = sorted(int(s[len("spec_g"):]) for s in STRATEGIES
                       if s.startswith("spec_g") and s[len("spec_g"):].isdigit())


def main():
    ap = argparse.ArgumentParser(description="Phase-1 roofline NVTX driver (run under ncu)")
    ap.add_argument("--model-pair", default="llama_8b_1b", choices=list(MODEL_PAIRS))
    ap.add_argument("--gamma", type=int, default=5,
                    help=f"num_speculative_tokens; must map to spec_g{{N}} (one of {_GAMMA_STRATS})")
    ap.add_argument("--dataset", default="gsm8k")
    # ncu replays kernels, so the profiled workload must be TINY. One prompt and a
    # handful of new tokens is enough to enter the draft/verify NVTX ranges for one
    # or two SD iterations; intensity is a per-kernel ratio, so a couple of
    # iterations summed is all that is needed. Bigger values are the main reason a
    # prior ncu run looked like it "hung" (it was replaying every kernel across a
    # full-length generate).
    ap.add_argument("--n-prompts", type=int, default=1, help="keep tiny — ncu replays kernels")
    ap.add_argument("--max-tokens", type=int, default=8, help="keep tiny — ncu replays kernels")
    ap.add_argument("--flops-sanity", action="store_true",
                    help="skip ncu path; run torch.profiler whole-run FLOP estimate instead")
    ap.add_argument("--selftest", action="store_true",
                    help="NO ncu: run the tiny generate and report how many times the draft/"
                         "verify NVTX ranges fired. Use this FIRST to confirm the hooks attach "
                         "in-process before paying for an ncu capture (0 counts => a wiring "
                         "problem, not an ncu problem).")
    args = ap.parse_args()

    strat = f"spec_g{args.gamma}"
    if strat not in STRATEGIES:
        sys.exit(f"--gamma {args.gamma} has no strategy '{strat}'. Expected one of {_GAMMA_STRATS}.")

    install_roofline()                            # MUST be before building the LLM
    print(f"  building {args.model_pair} / {strat} ...", file=sys.stderr)
    llm = build_llm(args.model_pair, strat, mock=False)
    # NB build_llm sets enforce_eager=True (no CUDA graphs) — required for ncu, which
    # cannot replay graph-captured kernels (a graph build is a classic ncu hang).
    prompts = load_prompts(args.dataset, args.model_pair, args.n_prompts)

    if args.flops_sanity:
        from profiling.roofline import quick_total_flops
        quick_total_flops(llm, prompts, max_tokens=args.max_tokens)
        return

    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=SEED)

    if args.selftest:
        llm.generate(prompts, sp)
        d, v = range_call_counts()
        print(f"  SELFTEST: draft range fired {d}x, verify range fired {v}x "
              f"(n_prompts={args.n_prompts}, max_tokens={args.max_tokens})", file=sys.stderr)
        if d == 0 or v == 0:
            sys.exit("  -> a range never fired: the NVTX hooks did not attach in-process. "
                     "Fix the worker wiring (PROPOSER_ATTR/SCORER_ATTR/method names) BEFORE ncu; "
                     "ncu would also capture nothing.")
        print("  -> hooks OK; safe to run this same script under ncu (see USAGE).", file=sys.stderr)
        return

    # Bracket ONLY the generate in the CUDA profiler region. Run ncu with
    # `--profile-from-start off` so model load is NOT profiled (loading an 8B model
    # under ncu instrumentation is slow and was a likely cause of the apparent hang);
    # profiling switches on here and off right after, scoping the capture to decode.
    # Without ncu these calls are harmless no-ops.
    import torch
    torch.cuda.profiler.start()
    try:
        llm.generate(prompts, sp)
    finally:
        torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()