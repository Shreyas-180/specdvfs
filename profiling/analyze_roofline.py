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

GPU peaks are chosen automatically from the ncu CSV's Device column (RTX 3090 vs 4090);
override with --gpu 3090|4090 or --peak-tflops/--peak-bw-gbs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))   # so gpu_profiles (repo root) imports on the laptop too

from gpu_profiles import GPU_PROFILES, DEFAULT_GPU, profile_for_name  # noqa: E402

# Roofline ceilings are GPU-specific (3090 vs 4090). This script runs on the LAPTOP (no GPU),
# so it cannot detect the card directly — it reads the GPU name from the ncu CSV's 'Device'
# column (the CSV was produced on the VM, so the name is in it) and looks the peaks up in
# gpu_profiles.py. --gpu / --peak-* override. These two are only the LAST-RESORT fallback.
RTX3090_PEAK_TFLOPS = GPU_PROFILES["rtx_3090"]["peak_tflops_bf16"]
RTX3090_PEAK_BW_GBS = GPU_PROFILES["rtx_3090"]["peak_bw_gbs"]

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
    """Return (flops, bytes, metrics_seen, device_name) summed across all kernels in the CSV.

    Tolerant of column-name variation: finds the 'Metric Name' / 'Metric Value' columns
    case-insensitively, and the 'Device' column if present (ncu records the GPU name there,
    which is how this laptop-side script learns which card produced the CSV). Returns
    flops/bytes as None when a metric family is entirely absent (caller falls back).
    """
    rows = list(csv.reader(path.open(encoding="utf-8", errors="ignore")))
    if not rows:
        return None, None, set(), None
    # Find the header row (ncu prepends banner lines before the CSV header).
    header_idx = next((i for i, r in enumerate(rows)
                       if any(c.strip().lower() == "metric name" for c in r)), None)
    if header_idx is None:
        return None, None, set(), None
    header = [c.strip().lower() for c in rows[header_idx]]
    try:
        name_col = header.index("metric name")
        val_col = header.index("metric value")
    except ValueError:
        return None, None, set(), None
    dev_col = header.index("device") if "device" in header else None

    flops = 0.0
    byts = 0.0
    seen = set()
    device = None
    saw_flop_metric = False
    saw_bytes_metric = False
    for r in rows[header_idx + 1:]:
        if len(r) <= max(name_col, val_col):
            continue
        if device is None and dev_col is not None and len(r) > dev_col and r[dev_col].strip():
            device = r[dev_col].strip()
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
            seen, device)


def _phase_from_csv(label, path_str):
    path = Path(path_str)
    if not path.exists():
        sys.exit(f"FAIL: {label} CSV not found: {path}")
    flops, byts, seen, device = _parse_ncu_csv(path)
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
    return {"name": label, "flops": flops, "bytes": byts}, device


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


def _apply_sm_scaling(peak_tflops, peak_bw_gbs, sm_count, sm_total):
    """APPROACH 2: scale the peak for a process restricted to `sm_count` of `sm_total` SMs.

    Compute scales ~linearly with the SM fraction. BANDWIDTH IS LEFT UNCHANGED: the memory
    controllers and L2 hang off the crossbar, not off individual SMs, so masking SMs does not
    remove memory channels. Net effect: the ridge I* = peak_FLOPS/peak_BW moves DOWN, sliding
    through the (fixed, measured) phase intensities. Phase intensities themselves do not move
    — same kernels, same FLOPs, same bytes — which is what makes this a clean instrument.

    CAVEAT worth carrying into the writeup: with fewer SMs issuing memory requests, ACHIEVED
    bandwidth can also fall (fewer outstanding requests to saturate DRAM). If it does, the true
    ridge is somewhat LOWER than this linear model says. Treat the model as the thing that
    picks sweep points and the measurement as the thing that decides the verdict.
    """
    if sm_count is None or int(sm_count) >= int(sm_total):
        return peak_tflops, peak_bw_gbs
    frac = int(sm_count) / float(sm_total)
    scaled = peak_tflops * frac
    print(f"  SM restriction: {sm_count}/{sm_total} SMs ({frac*100:.0f}%) -> "
          f"peak {peak_tflops:.1f} -> {scaled:.1f} TFLOP/s (bandwidth unchanged at "
          f"{peak_bw_gbs:.0f} GB/s)")
    return scaled, peak_bw_gbs


def _resolve_peaks(gpu_arg, peak_t, peak_bw, device_name):
    """Pick (peak_tflops, peak_bw_gbs). Explicit --peak-* win; else --gpu 3090/4090; else
    'auto' reads the GPU name parsed from the ncu CSV's Device column. Falls back to
    DEFAULT_GPU with a warning if nothing identifies the card."""
    if gpu_arg in ("3090", "4090"):
        prof = GPU_PROFILES[f"rtx_{gpu_arg}"]
        print(f"  GPU peaks: --gpu {gpu_arg} -> {prof['peak_tflops_bf16']} TFLOP/s, {prof['peak_bw_gbs']} GB/s")
    else:  # auto
        key, prof = profile_for_name(device_name)
        if prof is None:
            prof = GPU_PROFILES[DEFAULT_GPU]
            print(f"  WARN: GPU not identifiable from the CSV (device={device_name!r}); using "
                  f"{DEFAULT_GPU} peaks. Pass --gpu 3090|4090 or --peak-tflops/--peak-bw-gbs to be sure.")
        else:
            print(f"  GPU from ncu CSV: {device_name} [{key}] -> "
                  f"{prof['peak_tflops_bf16']} TFLOP/s, {prof['peak_bw_gbs']} GB/s")
    t = peak_t if peak_t is not None else prof["peak_tflops_bf16"]
    bw = peak_bw if peak_bw is not None else prof["peak_bw_gbs"]
    return t, bw


def main():
    ap = argparse.ArgumentParser(description="Roofline GO/NO-GO from per-phase FLOPs + bytes")
    ap.add_argument("--draft-flops", type=float)
    ap.add_argument("--draft-bytes", type=float)
    ap.add_argument("--verify-flops", type=float)
    ap.add_argument("--verify-bytes", type=float)
    ap.add_argument("--draft-csv")
    ap.add_argument("--verify-csv")
    ap.add_argument("--gpu", choices=["auto", "3090", "4090"], default="auto",
                    help="hardware peaks: auto = read the GPU from the ncu CSV's Device column")
    ap.add_argument("--peak-tflops", type=float, default=None, help="override bf16 tensor peak")
    ap.add_argument("--peak-bw-gbs", type=float, default=None, help="override DRAM bandwidth (GB/s)")
    ap.add_argument("--sm-count", type=int, default=None,
                    help="APPROACH 2: number of SMs the profiled process was restricted to. "
                         "Scales peak FLOPS by sm_count/sm_total (bandwidth unchanged), which "
                         "moves the ridge DOWN through the measured phase intensities. Pass the "
                         "same N used for the sm_sweep level that produced these CSVs.")
    ap.add_argument("--sm-total", type=int, default=82,
                    help="total SMs on the card (82 = RTX 3090 / GA102)")
    ap.add_argument("--out-json", default=str(PROJECT_ROOT / "profiling" / "out" / "roofline.json"))
    ap.add_argument("--out-png", default=str(PROJECT_ROOT / "profiling" / "out" / "roofline.png"))
    args = ap.parse_args()

    manual = all(v is not None for v in
                 (args.draft_flops, args.draft_bytes, args.verify_flops, args.verify_bytes))
    device = None
    if manual:
        phases = [{"name": "draft", "flops": args.draft_flops, "bytes": args.draft_bytes},
                  {"name": "verify", "flops": args.verify_flops, "bytes": args.verify_bytes}]
    elif args.draft_csv and args.verify_csv:
        d_phase, d_dev = _phase_from_csv("draft", args.draft_csv)
        v_phase, v_dev = _phase_from_csv("verify", args.verify_csv)
        phases = [d_phase, v_phase]
        device = d_dev or v_dev
    else:
        ap.error("supply either all four --{draft,verify}-{flops,bytes}, "
                 "or both --draft-csv and --verify-csv.")

    peak_t, peak_bw = _resolve_peaks(args.gpu, args.peak_tflops, args.peak_bw_gbs, device)
    peak_t, peak_bw = _apply_sm_scaling(peak_t, peak_bw, args.sm_count, args.sm_total)
    analyze(phases, peak_t, peak_bw, args.out_json, args.out_png)


if __name__ == "__main__":
    main()