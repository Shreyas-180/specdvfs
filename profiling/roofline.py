"""Phase 1 profiling — Roofline: arithmetic intensity of draft vs verify.

This is the GO/NO-GO premise check for the whole project. SpecDVFS assumes the DRAFT
phase is memory-bound (raising the clock doesn't speed it up -> downclock it for free) and
the VERIFY phase is compute-bound (the clock helps -> keep it high). The Roofline model
makes that testable:

  arithmetic intensity I = FLOPs / bytes_of_DRAM_traffic        (x-axis, FLOPs/byte)
  achievable perf is min(peak_FLOPS, I * peak_bandwidth)        (the "roofline")
  ridge point  I* = peak_FLOPS / peak_bandwidth                 (memory-bound  <-> compute-bound)

Expected, premise-confirming result:
  draft  -> I < I*  (LEFT of ridge,  memory-bound)   => downclocking draft is ~free
  verify -> I >= I* (AT/RIGHT of ridge, compute-bound) => verify needs f_high
If verify also lands left of the ridge, the "f_high for verify" strategy is wrong and you
rethink — which is exactly why this runs BEFORE the full sweep.

Why two measurements per phase, and why ncu:
  * FLOPs can be estimated by torch.profiler(with_flops=True), but it CANNOT give DRAM
    bytes, and per-phase attribution from the profiler event tree is fragile here.
  * Nsight Compute (ncu) gives BOTH FLOPs and DRAM bytes per region authoritatively. So the
    headline numbers come from ncu scoped to NVTX ranges this module installs; torch.profiler
    is only a quick order-of-magnitude sanity check.

WHEN: Phase 1, after the patch works and the pilot passes, before the full sweep. One-time.

Fork-safety / in-process: install_roofline() patches SpecDecodeWorker.init_device at the
class level so the NVTX ranges are placed inside the worker (TP=1, single-GPU, in-process).
Run profiling SEPARATELY from the DVFS sweep, at a fixed clock, so intensity isn't confounded.
"""

from __future__ import annotations

import functools
import importlib
import json
import logging
from pathlib import Path

import torch  # GPU-only tool

from vllm_hooks.patch_spec_decode import (
    WORKER_MODULE, WORKER_CLASS_NAME, PROPOSER_ATTR, SCORER_ATTR,
    DRAFT_METHOD, VERIFY_METHOD,
)

log = logging.getLogger(__name__)
_ROOFLINE_FLAG = "_specdvfs_roofline"

# RTX 3090 peak specs (override with your measured/clock-specific values).
# FP16 non-tensor-core is ~35.6 TFLOP/s; the bitsandbytes NF4 dequant path is mixed, so
# treat this as a documented default and confirm against ncu's achieved FLOP/s if in doubt.
RTX3090_PEAK_TFLOPS = 35.6
RTX3090_PEAK_BW_GBS = 936.0

DRAFT_RANGE = "specdvfs_draft"
VERIFY_RANGE = "specdvfs_verify"


# ── instrumentation: NVTX ranges on the same hook points the DVFS patch uses ──────────

def _nvtx_wrap(orig, name):
    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        torch.cuda.nvtx.range_push(name)
        try:
            return orig(*args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()
    return wrapper


def attach_ranges(worker):
    proposer = getattr(worker, PROPOSER_ATTR)
    scorer = getattr(worker, SCORER_ATTR)
    setattr(proposer, DRAFT_METHOD, _nvtx_wrap(getattr(proposer, DRAFT_METHOD), DRAFT_RANGE))
    setattr(scorer, VERIFY_METHOD, _nvtx_wrap(getattr(scorer, VERIFY_METHOD), VERIFY_RANGE))
    log.info("specdvfs.roofline: NVTX ranges attached to %s", type(worker).__name__)


def install_roofline():
    """Class-level init_device patch to place NVTX ranges in the worker. MAIN process,
    before building the LLM. Run under ncu (see USAGE) to read per-range FLOPs + bytes."""
    mod = importlib.import_module(WORKER_MODULE)
    cls = getattr(mod, WORKER_CLASS_NAME)
    if getattr(cls.init_device, _ROOFLINE_FLAG, False):
        return cls
    orig = cls.init_device

    @functools.wraps(orig)
    def wrapper(self, *a, **k):
        orig(self, *a, **k)
        try:
            attach_ranges(self)
        except Exception as e:
            log.error("specdvfs.roofline: attach failed: %s", e, exc_info=True)

    wrapper._specdvfs_roofline = True
    cls.init_device = wrapper
    return cls


# ── quick FLOP sanity check (order of magnitude only; ncu is authoritative) ────────────

def quick_total_flops(llm, prompts, max_tokens=64):
    """Total model FLOPs over a short generate via torch.profiler(with_flops=True). This is a
    whole-run number (NOT per-phase) — a sanity check only. Per-phase FLOPs come from ncu."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=42)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_flops=True,
    ) as prof:
        llm.generate(prompts, sp)
    total = sum(getattr(e, "flops", 0) or 0 for e in prof.key_averages())
    print(f"  torch.profiler total FLOPs (sanity, whole run): {total:.3e}")
    return total


# ── analysis: intensity, roofline plot, GO/NO-GO ──────────────────────────────────────

def ridge_point(peak_tflops=RTX3090_PEAK_TFLOPS, peak_bw_gbs=RTX3090_PEAK_BW_GBS):
    """Ridge intensity I* = peak_FLOPS / peak_bandwidth, in FLOPs/byte."""
    return (peak_tflops * 1e12) / (peak_bw_gbs * 1e9)


def analyze(phases, peak_tflops=RTX3090_PEAK_TFLOPS, peak_bw_gbs=RTX3090_PEAK_BW_GBS,
            out_png="profiling/out/roofline.png", out_json="profiling/out/roofline.json"):
    """phases: list of {'name', 'flops', 'bytes', 'seconds'(optional)} (from ncu --nvtx).

    Computes arithmetic intensity per phase, classifies vs the ridge, writes a JSON + a
    Roofline PNG, and prints the GO/NO-GO verdict for the project premise.
    """
    I_star = ridge_point(peak_tflops, peak_bw_gbs)
    rows = []
    for p in phases:
        flops, byts = float(p["flops"]), float(p["bytes"])
        I = flops / byts if byts > 0 else float("nan")
        row = {"name": p["name"], "flops": flops, "bytes": byts, "intensity_flops_per_byte": I,
               "regime": "compute-bound" if I >= I_star else "memory-bound"}
        if p.get("seconds"):
            row["achieved_tflops"] = flops / float(p["seconds"]) / 1e12
        rows.append(row)

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(
        {"ridge_flops_per_byte": I_star, "peak_tflops": peak_tflops,
         "peak_bw_gbs": peak_bw_gbs, "phases": rows}, indent=2))

    # verdict
    draft = next((r for r in rows if "draft" in r["name"].lower()), None)
    verify = next((r for r in rows if "verify" in r["name"].lower()), None)
    print(f"  ridge I* = {I_star:.1f} FLOPs/byte")
    for r in rows:
        print(f"    {r['name']:<18} I={r['intensity_flops_per_byte']:.2f} FLOPs/byte  -> {r['regime']}")
    if draft and verify:
        premise_ok = (draft["intensity_flops_per_byte"] < I_star <=
                      verify["intensity_flops_per_byte"])
        print("  PREMISE:", "GO — draft memory-bound, verify compute-bound (per-phase DVFS justified)"
              if premise_ok else
              "NO-GO — phases are NOT split across the ridge; revisit the frequency strategy.")

    _plot(rows, I_star, peak_tflops, peak_bw_gbs, out_png)
    print(f"  roofline -> {out_png}")
    return rows


def _plot(rows, I_star, peak_tflops, peak_bw_gbs, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"  (plot skipped: {e})")
        return
    xs = np.logspace(-1, 3, 200)
    roof = np.minimum(peak_tflops, xs * peak_bw_gbs / 1e3)  # TFLOP/s ceiling vs intensity
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(xs, roof, "k-", lw=2, label="roofline")
    ax.axvline(I_star, ls="--", color="gray", label=f"ridge {I_star:.0f}")
    for r in rows:
        y = r.get("achieved_tflops", peak_tflops if r["regime"] == "compute-bound"
                  else r["intensity_flops_per_byte"] * peak_bw_gbs / 1e3)
        ax.scatter(r["intensity_flops_per_byte"], y, s=90, zorder=5)
        ax.annotate(r["name"], (r["intensity_flops_per_byte"], y),
                    textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("arithmetic intensity (FLOPs/byte)")
    ax.set_ylabel("performance (TFLOP/s)")
    ax.set_title("SpecDVFS Roofline: draft vs verify")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ── USAGE ─────────────────────────────────────────────────────────────────────
# 1) Place NVTX ranges, then profile with Nsight Compute (authoritative FLOPs + bytes):
#
#    # tiny driver script prof_roofline.py:
#    from profiling.roofline import install_roofline
#    install_roofline()                         # MAIN process, before building the LLM
#    from vllm import LLM, SamplingParams
#    llm = LLM(model=..., speculative_model=..., num_speculative_tokens=5, ...)
#    llm.generate(prompts[:8], SamplingParams(temperature=0, max_tokens=64))
#
#    # run it under ncu, scoping to each NVTX range (one section gives both flops + bytes):
#    ncu --nvtx --nvtx-include "specdvfs_draft/"  --section MemoryWorkloadAnalysis \
#        --section SpeedOfLight --csv --target-processes all python prof_roofline.py > draft.csv
#    ncu --nvtx --nvtx-include "specdvfs_verify/" --section MemoryWorkloadAnalysis \
#        --section SpeedOfLight --csv --target-processes all python prof_roofline.py > verify.csv
#    # from the CSVs: FLOPs ~ sm__sass_thread_inst_executed_op_*_pred_on.sum (×2 for FMA),
#    #                bytes ~ dram__bytes.sum  (read+write DRAM traffic).
#
# 2) Plot + GO/NO-GO verdict from the measured numbers:
#    from profiling.roofline import analyze
#    analyze([
#        {"name": "draft",  "flops": <draft_flops>,  "bytes": <draft_bytes>},
#        {"name": "verify", "flops": <verify_flops>, "bytes": <verify_bytes>},
#    ])  # -> profiling/out/roofline.{png,json} + prints GO / NO-GO
#
# (quick_total_flops(llm, prompts) gives a whole-run FLOP sanity check without ncu.)
