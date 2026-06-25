#!/usr/bin/env python
"""Phase-1 — roofline verdict from measured per-phase FLOPs + DRAM bytes.

Computes arithmetic intensity I = FLOPs / bytes for the draft and verify phases,
classifies each against the ridge point I* = peak_FLOPS / peak_bandwidth, writes a
JSON + a roofline PNG, and prints the project's GO/NO-GO premise verdict
(draft LEFT of the ridge = memory-bound; verify AT/RIGHT = compute-bound).

This is the *analysis* half of profiling/roofline.py, kept dependency-light (no
torch) so it runs on the VM right after profiling OR on your laptop after pulling
the ncu CSVs down. It only needs numpy + matplotlib for the plot (the verdict
itself needs neither).

Two ways to supply the numbers:

  (A) Directly — bulletproof. Read FLOPs + DRAM bytes off the ncu output yourself
      (ncu prints the requested metrics per range) and pass them in:
        python profiling/analyze_roofline.py \
            --draft-flops 1.2e10 --draft-bytes 9.0e9 \
            --verify-flops 4.5e11 --verify-bytes 6.0e9

  (B) From ncu --csv files — best-effort. Sums dram__bytes.sum for bytes and the FLOP
      counters for FLOPs: CUDA-core FP (fadd + fmul + 2*ffma) PLUS the tensor-core
      ops_path_tensor_* metric (bf16 GEMMs live here and dominate — omitting them, as the
      first attempt did, makes everything look memory-bound). Prefix sm__/smsp__ is matched
      tolerantly. Request exactly these metrics via ncu --metrics (see prof_roofline.py) so
      the names in the CSV match what this parser looks for. If your ncu lacks the tensor
      metric name, the parser says what it found — confirm with `ncu --query-metrics` and
      either pass the right name or read achieved FLOP/s from SpeedOfLight and use (A).
        python profiling/analyze_roofline.py \
            --draft-csv profiling/out/draft.csv --verify-csv profiling/out/verify.csv

Override the GPU peaks with --peak-tflops / --peak-bw-gbs (defaults: RTX 3090).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# RTX 3090 defaults (same as profiling/roofline.py — keep them in sync).
# 71.0 = bf16 tensor-core peak, FP32 accumulate (matches counting tensor-core FLOPs below).
RTX3090_PEAK_TFLOPS = 71.0
RTX3090_PEAK_BW_GBS = 936.0

# ncu metric names. bf16 LLM math is dominated by tensor-core GEMMs, which the FP32 thread-
# instruction counters (fadd/fmul/ffma) DO NOT count — counting only those makes every phase
# look memory-bound (this is exactly why the earlier crashed capture read I~0.08). So
# FLOPs = CUDA-core FP ops + tensor-core ops. Matching is tolerant of the sm__ vs smsp__
# aggregation prefix, which differs across ncu versions / how the metric was requested.
_BYTES_METRIC = "dram__bytes.sum"
_PREFIXES = ("smsp__", "sm__")
# suffix (prefix-stripped) -> FLOPs-per-count multiplier.
_FLOP_SUFFIXES = {
    "sass_thread_inst_executed_op_fadd_pred_on.sum": 1.0,   # CUDA-core FP add
    "sass_thread_inst_executed_op_fmul_pred_on.sum": 1.0,   # CUDA-core FP mul
    "sass_thread_inst_executed_op_ffma_pred_on.sum": 2.0,   # CUDA-core FMA = 2 FLOPs
    # tensor-core op metrics are ALREADY FLOP counts (not instructions) -> multiplier 1.0.
    # The exact src/dst dtype suffix is version-dependent; accept the common bf16/fp16 forms
    # (we run bf16). Confirm with: ncu --query-metrics | grep ops_path_tensor
    "ops_path_tensor_src_bf16_dst_fp32.sum": 1.0,
    "ops_path_tensor_src_fp16_dst_fp32.sum": 1.0,
    "ops_path_tensor_src_bf16_dst_fp16.sum": 1.0,
    "ops_path_tensor_src_fp16_dst_fp16.sum": 1.0,
}


def _strip_prefix(name):
    for p in _PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def ridge_point(peak_tflops, peak_bw_gbs):
    """I* = peak_FLOPS / peak_bandwidth, FLOPs/byte."""
    return (peak_tflops * 1e12) / (peak_bw_gbs * 1e9)


# ── best-effort ncu --csv parsing ─────────────────────────────────────────────

def _parse_ncu_csv(path: Path):
    """Return (flops, bytes, metrics_seen) summed across all kernels in the CSV.

    Tolerant of column-name variation: finds the 'Metric Name' / 'Metric Value'
    columns case-insensitively. Returns flops/bytes as None when a metric family
    is entirely absent (so the caller can fall back to manual numbers).
    """
    rows = list(csv.reader(path.open(encoding="utf-8", errors="ignore")))
    if not rows:
        return None, None, set()
    # Find the header row (ncu prepends banner lines before the CSV header).
    header_idx = next((i for i, r in enumerate(rows)
                       if any(c.strip().lower() == "metric name" for c in r)), None)
    if header_idx is None:
        return None, None, set()
    header = [c.strip().lower() for c in rows[header_idx]]
    try:
        name_col = header.index("metric name")
        val_col = header.index("metric value")
    except ValueError:
        return None, None, set()

    flops = 0.0
    byts = 0.0
    seen = set()
    saw_flop_metric = False
    saw_bytes_metric = False
    for r in rows[header_idx + 1:]:
        if len(r) <= max(name_col, val_col):
            continue
        name = r[name_col].strip()
        if not name:
            continue
        seen.add(name)
        raw = r[val_col].strip().replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            continue
        if name == _BYTES_METRIC:
            byts += val
            saw_bytes_metric = True
        else:
            suffix = _strip_prefix(name)
            if suffix in _FLOP_SUFFIXES:
                flops += val * _FLOP_SUFFIXES[suffix]
                saw_flop_metric = True
    return (flops if saw_flop_metric else None,
            byts if saw_bytes_metric else None,
            seen)


def _phase_from_csv(label, path_str):
    path = Path(path_str)
    if not path.exists():
        sys.exit(f"FAIL: {label} CSV not found: {path}")
    flops, byts, seen = _parse_ncu_csv(path)
    if flops is None or byts is None:
        missing = []
        if flops is None:
            missing.append("FLOP counters (CUDA-core fadd/fmul/ffma + a tensor-core "
                           "ops_path_tensor_* metric)")
        if byts is None:
            missing.append(f"DRAM bytes ({_BYTES_METRIC})")
        print(f"  WARN: could not extract {' and '.join(missing)} from {path.name}.")
        print(f"        metrics present: {sorted(seen) if seen else '(none parsed)'}")
        sys.exit("  -> re-run ncu with those --metrics, or pass the numbers manually (mode A).")
    return {"name": label, "flops": flops, "bytes": byts}


# ── verdict + plot ────────────────────────────────────────────────────────────

def analyze(phases, peak_tflops, peak_bw_gbs, out_json, out_png):
    I_star = ridge_point(peak_tflops, peak_bw_gbs)
    rows = []
    for p in phases:
        flops, byts = float(p["flops"]), float(p["bytes"])
        I = flops / byts if byts > 0 else float("nan")
        rows.append({"name": p["name"], "flops": flops, "bytes": byts,
                     "intensity_flops_per_byte": I,
                     "regime": "compute-bound" if I >= I_star else "memory-bound"})

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(
        {"ridge_flops_per_byte": I_star, "peak_tflops": peak_tflops,
         "peak_bw_gbs": peak_bw_gbs, "phases": rows}, indent=2))

    print(f"  ridge I* = {I_star:.1f} FLOPs/byte")
    for r in rows:
        print(f"    {r['name']:<8} I={r['intensity_flops_per_byte']:.2f} FLOPs/byte -> {r['regime']}")
    draft = next((r for r in rows if "draft" in r["name"].lower()), None)
    verify = next((r for r in rows if "verify" in r["name"].lower()), None)
    if draft and verify:
        ok = draft["intensity_flops_per_byte"] < I_star <= verify["intensity_flops_per_byte"]
        print("  PREMISE:", "GO — draft memory-bound, verify compute-bound (per-phase DVFS justified)"
              if ok else
              "NO-GO — phases are NOT split across the ridge; revisit the frequency strategy.")
    _plot(rows, I_star, peak_tflops, peak_bw_gbs, out_png)
    print(f"  wrote {out_json} and {out_png}")
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
    roof = np.minimum(peak_tflops, xs * peak_bw_gbs / 1e3)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(xs, roof, "k-", lw=2, label="roofline")
    ax.axvline(I_star, ls="--", color="gray", label=f"ridge {I_star:.0f}")
    for r in rows:
        y = peak_tflops if r["regime"] == "compute-bound" else \
            r["intensity_flops_per_byte"] * peak_bw_gbs / 1e3
        ax.scatter(r["intensity_flops_per_byte"], y, s=90, zorder=5)
        ax.annotate(r["name"], (r["intensity_flops_per_byte"], y),
                    textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("arithmetic intensity (FLOPs/byte)")
    ax.set_ylabel("performance (TFLOP/s)")
    ax.set_title("SpecDVFS Roofline: draft vs verify")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Roofline GO/NO-GO from per-phase FLOPs + bytes")
    ap.add_argument("--draft-flops", type=float)
    ap.add_argument("--draft-bytes", type=float)
    ap.add_argument("--verify-flops", type=float)
    ap.add_argument("--verify-bytes", type=float)
    ap.add_argument("--draft-csv")
    ap.add_argument("--verify-csv")
    ap.add_argument("--peak-tflops", type=float, default=RTX3090_PEAK_TFLOPS)
    ap.add_argument("--peak-bw-gbs", type=float, default=RTX3090_PEAK_BW_GBS)
    ap.add_argument("--out-json", default=str(PROJECT_ROOT / "profiling" / "out" / "roofline.json"))
    ap.add_argument("--out-png", default=str(PROJECT_ROOT / "profiling" / "out" / "roofline.png"))
    args = ap.parse_args()

    manual = all(v is not None for v in
                 (args.draft_flops, args.draft_bytes, args.verify_flops, args.verify_bytes))
    if manual:
        phases = [{"name": "draft", "flops": args.draft_flops, "bytes": args.draft_bytes},
                  {"name": "verify", "flops": args.verify_flops, "bytes": args.verify_bytes}]
    elif args.draft_csv and args.verify_csv:
        phases = [_phase_from_csv("draft", args.draft_csv),
                  _phase_from_csv("verify", args.verify_csv)]
    else:
        ap.error("supply either all four --{draft,verify}-{flops,bytes}, "
                 "or both --draft-csv and --verify-csv.")
    analyze(phases, args.peak_tflops, args.peak_bw_gbs, args.out_json, args.out_png)


if __name__ == "__main__":
    main()