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

    # 1) DRAFT then VERIFY (see profiling/roofline.py USAGE for the full $M metric list
    #    incl. the tensor-core op metric). NOTE --replay-mode application is REQUIRED, not
    #    optional: ncu's default kernel replay rewinds individual kernels in isolation, but
    #    vLLM speculative decoding reads the rejection sampler's acceptance count back to the
    #    CPU mid-iteration to choose control flow — so the CPU blocks on a device->host copy
    #    of a value produced by the very kernel ncu is mid-replay on. That circular wait
    #    parks the process at 0% GPU (the "ncu hang"). Application replay re-runs the whole
    #    tiny deterministic (temp=0/seed=42) program once per pass, so nothing is rewound and
    #    it cannot deadlock. Cost: the model is (re)loaded each pass — seconds, not the
    #    indefinite hang. The driver warms up once before the profiled region so the capture
    #    is steady-state decode only (that warmup is excluded by --profile-from-start off).
    ncu --replay-mode application --profile-from-start off \
        --nvtx --nvtx-include "specdvfs_draft/" \
        --metrics "$M" --target-processes all --csv \
        python profiling/prof_roofline.py --gamma 5 --n-prompts 1 --max-tokens 8 \
        > profiling/out/draft.csv
    ncu --replay-mode application --profile-from-start off \
        --nvtx --nvtx-include "specdvfs_verify/" \
        --metrics "$M" --target-processes all --csv \
        python profiling/prof_roofline.py --gamma 5 --n-prompts 1 --max-tokens 8 \
        > profiling/out/verify.csv
    # If perf counters are admin-locked (ERR_NVGPUCTRPERM, common on cloud VMs):
    #   prefix with  sudo env "PATH=$PATH"  . Optionally add --launch-count N to profile
    #   only the first N kernels of the phase (the intensity is a ratio, so a representative
    #   sample suffices) if a pass is slower than you like.

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


def resolve_n_prompts(n_prompts, max_num_seqs):
    """Return the prompt count that actually makes `max_num_seqs` bind.

    vLLM schedules up to `max_num_seqs` sequences CONCURRENTLY, but it can only do so if at
    least that many prompts have been submitted. Supplying fewer silently caps real
    concurrency at the prompt count, so a capture nominally at "batch 11" is really a capture
    at "batch n_prompts". Arithmetic intensity depends on the REAL concurrency (more
    sequences amortise the same weight-matrix reads over more FLOPs), so an under-supplied
    capture measures the wrong configuration while looking perfectly healthy.

    Raising to exactly max_num_seqs fills the batch on the first scheduling step. Note the
    batch still DECAYS as individual sequences finish and are not replaced; with the tiny
    --max-tokens used for profiling that is only a couple of decode steps, so the effect is
    small, but pass a larger --n-prompts explicitly if you want the batch held full longer.
    """
    if max_num_seqs is None or n_prompts >= max_num_seqs:
        return n_prompts
    print(f"  NOTE: raising --n-prompts {n_prompts} -> {max_num_seqs} to match --max-num-seqs. "
          f"With only {n_prompts} prompt(s) the scheduler cap would never bind and this "
          f"capture would measure a batch of {n_prompts}, not {max_num_seqs}.", file=sys.stderr)
    return max_num_seqs


def _active_sm():
    """SM count this capture is actually running under (None = unrestricted).

    Read from the environment, not from a flag, so it reports what the PROCESS really got.
    """
    try:
        import sm_partition
        return sm_partition.active_sm_count()
    except Exception:
        return None


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
    #
    # *** BUT: n_prompts MUST be >= max_num_seqs, or an Approach-1 batch capture is a NO-OP. ***
    # max_num_seqs is a SCHEDULER ADMISSION CAP -- it limits how many sequences may run
    # CONCURRENTLY. It only binds when that many sequences are actually in flight. With
    # n_prompts=1 there is exactly one sequence, so the cap never engages and every batch
    # level executes the IDENTICAL computation. That is precisely what happened on
    # 2026-07-27: captures at --max-num-seqs 8 and 11 returned verify intensities of 61.24
    # and 61.25 (a 0.016% difference, i.e. noise) and byte-identical CSV sizes, because
    # both were really running a batch of ONE. The energy sweep did not have this bug --
    # it submits 32 prompts, and its throughput scales 238 -> 591 tok/s from batch 4 to 22,
    # proving the cap works correctly when enough prompts are supplied.
    # resolve_n_prompts() below enforces this automatically; see its docstring.
    ap.add_argument("--n-prompts", type=int, default=1,
                    help="keep tiny — ncu replays kernels. AUTO-RAISED to --max-num-seqs when "
                         "that is larger, since the batch cap does not bind otherwise.")
    ap.add_argument("--max-tokens", type=int, default=8, help="keep tiny — ncu replays kernels")
    # ---- ridge-crossing experiment knobs (must MATCH the sweep cell being profiled) ----
    # The measured arithmetic intensity is only meaningful for the configuration it was
    # captured under, so the roofline capture has to be reproducible at each sweep point.
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="APPROACH 1: batch size for this capture. I_verify scales with "
                         "batch x (gamma+1) tokens per verify pass, so this is THE knob that "
                         "moves verify across the ridge. Match the batch-sweep level.")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="context length; the batch sweep runs at 1024, the main pilot at 2048")
    ap.add_argument("--sm-count", type=int, default=None,
                    help="APPROACH 2: informational only here — SM restriction is applied by "
                         "the MPS env var on the PROCESS (run under the same env as the sweep "
                         "child), not by this flag. Pass the same N to analyze_roofline.py, "
                         "which uses it to scale peak FLOPS and move the ridge.")
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
    _bs = args.max_num_seqs
    _mml = args.max_model_len
    # Enforce n_prompts >= max_num_seqs BEFORE loading prompts, or the batch cap is a no-op.
    args.n_prompts = resolve_n_prompts(args.n_prompts, _bs)
    sm_env = _active_sm()
    _conc = min(args.n_prompts, _bs) if _bs else args.n_prompts
    print(f"  building {args.model_pair} / {strat} "
          f"(max_num_seqs={_bs if _bs else 'default'}, max_model_len={_mml if _mml else 'default'}, "
          f"SMs={sm_env if sm_env else 'unrestricted'}) ...", file=sys.stderr)
    print(f"  effective concurrency: {_conc} sequence(s)  "
          f"(n_prompts={args.n_prompts}, max_num_seqs={_bs if _bs else 'default 8'})  "
          f"<- THIS is the batch size the intensity will reflect", file=sys.stderr)
    if args.sm_count is not None and sm_env is not None and int(args.sm_count) != int(sm_env):
        # Catch the easy mistake: asking for an SM level but forgetting to launch under the
        # MPS-capped environment. The capture would then silently profile the FULL GPU.
        print(f"  !! --sm-count {args.sm_count} but the process is actually running with "
              f"SMs={sm_env}. Launch this capture under the same env as the sweep child "
              f"(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE), or the numbers are for the full GPU.",
              file=sys.stderr)
    llm = build_llm(args.model_pair, strat, mock=False,
                    max_num_seqs=_bs, max_model_len=_mml)
    # NB build_llm sets enforce_eager=True (no CUDA graphs) — required for ncu, which
    # cannot replay graph-captured kernels (a graph build is a classic ncu hang).
    prompts = load_prompts(args.dataset, args.model_pair, args.n_prompts,
                           max_model_len=_mml)

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

    import torch
    # WARMUP (not profiled): run the SAME NVTX-ranged generate once with the profiler still
    # OFF (--profile-from-start off + we have not called profiler.start() yet), so first-
    # iteration lazy CUDA allocations / autotuning are completed here and the profiled
    # region below contains only steady-state decode kernels. The warmup enters the draft/
    # verify NVTX ranges too, but is excluded from the capture because profiling is off.
    llm.generate(prompts, sp)
    torch.cuda.synchronize()

    # Profiled region: bracket ONLY the steady-state generate. Two scoping mechanisms combine
    # so the capture is exactly the decode phase: --nvtx-include restricts WHICH kernels count
    # (draft xor verify), and --profile-from-start off + this start()/stop() pair restrict WHEN
    # (model load and the warmup above are outside it). Run ncu with --replay-mode application
    # (NOT the default kernel replay): vLLM spec decode reads acceptance counts back to the CPU
    # mid-iteration to pick control flow, and kernel replay rewinds individual kernels while the
    # CPU is blocked on exactly those results — a circular wait that parks the process at 0% GPU.
    # Application replay re-runs the whole tiny deterministic program per pass, so it can't
    # deadlock. Without ncu these profiler calls are harmless no-ops.
    torch.cuda.profiler.start()
    try:
        llm.generate(prompts, sp)
    finally:
        torch.cuda.profiler.stop()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()