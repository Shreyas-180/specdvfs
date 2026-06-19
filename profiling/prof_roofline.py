#!/usr/bin/env python
"""Phase-1 driver — place NVTX ranges on draft/verify and run a short SD generate.

Run this UNDER Nsight Compute (ncu), scoped to each NVTX range, to read
authoritative per-phase FLOPs + DRAM bytes; then feed those numbers to
analyze_roofline.py for the GO/NO-GO premise verdict (draft memory-bound, verify
compute-bound). install_roofline() and the DVFS install() are mutually exclusive —
this is a fixed-clock profiling run, not a DVFS run.

ncu controls the GPU clock during profiling (it locks to base by default; add
--clock-control none to profile at the live clock), so this driver does NOT lock
clocks itself. Keep the workload SMALL: ncu replays each kernel many times, so a
full-length generate under ncu can take a very long time.

Model/build/prompt config is imported from experiments.run_experiment so the
profiled phases match the real runs.

USAGE (on the VM, after setup + HF login):
    # 1) DRAFT range -> per-phase FLOPs + DRAM bytes:
    ncu --nvtx --nvtx-include "specdvfs_draft/"  \
        --section MemoryWorkloadAnalysis --section SpeedOfLight \
        --csv --target-processes all \
        python profiling/prof_roofline.py > profiling/out/draft.csv

    # 2) VERIFY range:
    ncu --nvtx --nvtx-include "specdvfs_verify/" \
        --section MemoryWorkloadAnalysis --section SpeedOfLight \
        --csv --target-processes all \
        python profiling/prof_roofline.py > profiling/out/verify.csv

    # 3) verdict + plot:
    python profiling/analyze_roofline.py \
        --draft-csv profiling/out/draft.csv --verify-csv profiling/out/verify.csv

Quick whole-run FLOP sanity WITHOUT ncu (order of magnitude only):
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
from profiling.roofline import install_roofline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Phase-1 roofline NVTX driver (run under ncu)")
    ap.add_argument("--model-pair", default="llama_8b_1b", choices=list(MODEL_PAIRS))
    ap.add_argument("--gamma", type=int, default=5,
                    help="num_speculative_tokens; must map to spec_g{N} (3, 5 or 7)")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--n-prompts", type=int, default=4,
                    help="keep small — ncu replay is slow")
    ap.add_argument("--max-tokens", type=int, default=32, help="keep small for ncu")
    ap.add_argument("--flops-sanity", action="store_true",
                    help="skip ncu path; run torch.profiler whole-run FLOP estimate instead")
    args = ap.parse_args()

    strat = f"spec_g{args.gamma}"
    if strat not in STRATEGIES:
        sys.exit(f"--gamma {args.gamma} has no strategy '{strat}'. Expected gamma in {{3,5,7}}.")
    family = MODEL_PAIRS[args.model_pair]["family"]

    install_roofline()                            # MUST be before building the LLM
    print(f"  building {args.model_pair} / {strat} ...", file=sys.stderr)
    llm = build_llm(args.model_pair, strat, mock=False)
    prompts = load_prompts(args.dataset, family, args.n_prompts)

    if args.flops_sanity:
        from profiling.roofline import quick_total_flops
        quick_total_flops(llm, prompts, max_tokens=args.max_tokens)
        return

    from vllm import SamplingParams
    # The NVTX ranges installed by install_roofline() bracket each draft/verify call;
    # ncu (scoped via --nvtx-include) reads FLOPs + DRAM bytes for that range.
    llm.generate(prompts, SamplingParams(temperature=0.0,
                                         max_tokens=args.max_tokens, seed=SEED))


if __name__ == "__main__":
    main()
