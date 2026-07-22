"""Evaluation — Stage 1: per-run metrics.

Reads every run JSON written by experiments/run_experiment.py and produces one tidy
CSV row per run with derived energy/latency/efficiency metrics. Pure post-processing:
no GPU, no vLLM. Runs on --mock output too (identical schema; mock energy is fake, so
savings come out ~0 — that only validates the plumbing).

Pipeline:
    compute_metrics.py   results/<mode>/*.json     -> evaluation/out/<mode>/runs.csv
    aggregate_results.py runs.csv                  -> summary + savings tables
    generate_figures.py  runs.csv (+ savings csvs) -> figures/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "pilot"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "evaluation" / "out"


def _g(d, *keys, default=None):
    """Safe nested dict get: _g(r, 'energy_kwh', 'gpu_kwh')."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def load_runs(results_dir: Path) -> pd.DataFrame:
    """Load all run JSONs in results_dir into a per-run DataFrame with derived metrics."""
    rows = []
    for f in sorted(Path(results_dir).glob("*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  WARN: skipping unreadable {f.name}: {e}")
            continue

        total_kwh = _g(r, "energy_kwh", "total_kwh")
        gpu_kwh = _g(r, "energy_kwh", "gpu_kwh")
        tokens = r.get("total_tokens") or 0
        wall = r.get("wall_time_s") or 0.0
        total_Wh = total_kwh * 1000 if total_kwh is not None else None
        gpu_Wh = gpu_kwh * 1000 if gpu_kwh is not None else None
        uniq = _g(r, "gpu_monitor", "sm_clock_unique_mhz", default=[]) or []

        rows.append({
            "file": f.name,
            "model_pair": r.get("model_pair"),
            "strategy": r.get("strategy"),
            "dvfs_condition": r.get("dvfs_condition"),
            "dataset": r.get("dataset"),
            "rep": r.get("rep"),
            # raw
            "wall_time_s": wall,
            "total_tokens": tokens,
            "alpha_mean": r.get("alpha_mean"),
            # energy (Wh)
            "gpu_energy_Wh": gpu_Wh,
            "total_energy_Wh": total_Wh,
            # efficiency
            "tokens_per_s": (tokens / wall) if wall else None,
            "gpu_wh_per_1k": (gpu_Wh * 1000.0 / tokens) if (gpu_Wh is not None and tokens) else None,
            "total_wh_per_1k": (total_Wh * 1000.0 / tokens) if (total_Wh is not None and tokens) else None,
            # energy-delay product (lower is better)
            "edp_gpu": (gpu_Wh * wall) if gpu_Wh is not None else None,
            "edp_total": (total_Wh * wall) if total_Wh is not None else None,
            # DVFS-toggle evidence (from the 10 ms monitor summary)
            "sm_clock_levels": len(uniq),
            "clock_toggled": len(uniq) > 1,
            "power_mw_mean": _g(r, "gpu_monitor", "power_mw_mean"),
            "temp_c_max": _g(r, "gpu_monitor", "temp_c_max"),
            "f_high": r.get("f_high"),
            "f_low": r.get("f_low"),
            # --- ridge-crossing experiment axes (see run_experiment.py) ---
            # Defaults keep OLD result JSONs (written before these fields existed) loadable:
            # the 75 pilot runs already on disk ran at batch 8 / 2048 ctx / full 82 SMs, which
            # is exactly what these fallbacks encode, so old and new rows stay comparable.
            "batch_size": r.get("batch_size", 8),
            "max_model_len": r.get("max_model_len", 2048),
            "sm_count": r.get("sm_count", 82),
            "sm_total": r.get("sm_total", 82),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Evaluation stage 1: per-run metrics")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS),
                    help="dir of run JSONs (e.g. results/pilot or results/full)")
    ap.add_argument("--out", default=None,
                    help="output CSV path (default: evaluation/out/<mode>/runs.csv)")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    df = load_runs(rd)
    if df.empty:
        print(f"No run JSONs found in {rd}. Run the experiment first.")
        return

    mode = rd.name  # 'pilot' or 'full'
    out = Path(args.out) if args.out else (DEFAULT_OUT_ROOT / mode / "runs.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} runs -> {out}")

    # Quick sanity: did DVFS actually toggle the clock where it should have?
    print("\nclock-toggle rate by (strategy, condition)  [1.0 = always toggled]:")
    chk = df.groupby(["strategy", "dvfs_condition"])["clock_toggled"].mean()
    print(chk.to_string())


if __name__ == "__main__":
    main()

# =============================================================================
# USAGE
# =============================================================================
#   python evaluation/compute_metrics.py --results-dir results/pilot
#   python evaluation/compute_metrics.py --results-dir results/full
# Output: evaluation/out/<mode>/runs.csv  (one row per run, with derived metrics)