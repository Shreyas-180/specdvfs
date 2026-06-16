"""Evaluation — Stage 2: aggregate + comparative metrics.

Reads runs.csv from compute_metrics.py and produces, in the same directory:

  summary_by_group.csv   mean/std/count of every metric per (model, strategy, condition, dataset)
  savings_vs_off.csv      HEADLINE: energy saved by each DVFS condition vs SD-WITHOUT-DVFS
                          (`off`), plus latency overhead, EDP improvement, and a Welch
                          t-test p-value (small-n: interpret with care).
  savings_vs_vanilla.csv  SD+DVFS vs plain vanilla decoding (energy ratio + speedup).
  gamma_trend.csv         savings vs draft length (gamma) — does the per-phase benefit
                          grow with the draft/verify time ratio? (the generality result)

The central claim is "DVFS makes SD more energy-efficient at minimal latency loss", so
`savings_vs_off.csv` is the table to read first.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "evaluation" / "out" / "pilot" / "runs.csv"

GROUP = ["model_pair", "strategy", "dvfs_condition", "dataset"]
METRICS = ["gpu_energy_Wh", "total_energy_Wh", "wall_time_s", "tokens_per_s",
           "gpu_wh_per_1k", "total_wh_per_1k", "edp_gpu", "edp_total", "alpha_mean"]


def parse_gamma(strategy: str):
    """'spec_g5' -> 5; vanilla/spec_dyn/eagle3 -> None."""
    m = re.fullmatch(r"spec_g(\d+)", str(strategy))
    return int(m.group(1)) if m else None


def _vals(df, model, strat, cond, ds, col):
    sub = df[(df.model_pair == model) & (df.strategy == strat)
             & (df.dvfs_condition == cond) & (df.dataset == ds)]
    return sub[col].dropna()


def _ttest_p(a, b):
    if len(a) > 1 and len(b) > 1:
        try:
            return float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        except Exception:
            return np.nan
    return np.nan


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    present = [m for m in METRICS if m in df.columns]
    agg = df.groupby(GROUP)[present].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def savings_vs_off(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sd_strats = sorted(s for s in df.strategy.unique() if s != "vanilla")
    conditions = sorted(c for c in df.dvfs_condition.unique() if c != "off")
    for model in sorted(df.model_pair.unique()):
        for strat in sd_strats:
            for ds in sorted(df.dataset.unique()):
                off_gpu = _vals(df, model, strat, "off", ds, "gpu_energy_Wh")
                if off_gpu.empty:
                    continue
                off_tot = _vals(df, model, strat, "off", ds, "total_energy_Wh")
                off_time = _vals(df, model, strat, "off", ds, "wall_time_s")
                off_edp = _vals(df, model, strat, "off", ds, "edp_gpu")
                for cond in conditions:
                    c_gpu = _vals(df, model, strat, cond, ds, "gpu_energy_Wh")
                    if c_gpu.empty:
                        continue
                    c_tot = _vals(df, model, strat, cond, ds, "total_energy_Wh")
                    c_time = _vals(df, model, strat, cond, ds, "wall_time_s")
                    c_edp = _vals(df, model, strat, cond, ds, "edp_gpu")
                    rows.append({
                        "model_pair": model, "strategy": strat, "gamma": parse_gamma(strat),
                        "dataset": ds, "dvfs_condition": cond,
                        "gpu_energy_saving_pct": 100 * (off_gpu.mean() - c_gpu.mean()) / off_gpu.mean(),
                        "total_energy_saving_pct": (100 * (off_tot.mean() - c_tot.mean()) / off_tot.mean()
                                                    if not off_tot.empty and not c_tot.empty else np.nan),
                        "latency_overhead_pct": (100 * (c_time.mean() - off_time.mean()) / off_time.mean()
                                                 if not off_time.empty and not c_time.empty else np.nan),
                        "edp_improvement": (off_edp.mean() / c_edp.mean()
                                            if not off_edp.empty and not c_edp.empty and c_edp.mean() else np.nan),
                        "p_value_gpu_energy": _ttest_p(c_gpu, off_gpu),
                        "n_cond": len(c_gpu), "n_off": len(off_gpu),
                    })
    return pd.DataFrame(rows)


def savings_vs_vanilla(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sd_strats = sorted(s for s in df.strategy.unique() if s != "vanilla")
    conditions = sorted(df.dvfs_condition.unique())
    for model in sorted(df.model_pair.unique()):
        for ds in sorted(df.dataset.unique()):
            # vanilla only runs under 'off'
            van_gpu = _vals(df, model, "vanilla", "off", ds, "gpu_energy_Wh")
            van_time = _vals(df, model, "vanilla", "off", ds, "wall_time_s")
            if van_gpu.empty:
                continue
            for strat in sd_strats:
                for cond in conditions:
                    c_gpu = _vals(df, model, strat, cond, ds, "gpu_energy_Wh")
                    if c_gpu.empty:
                        continue
                    c_time = _vals(df, model, strat, cond, ds, "wall_time_s")
                    rows.append({
                        "model_pair": model, "strategy": strat, "gamma": parse_gamma(strat),
                        "dataset": ds, "dvfs_condition": cond,
                        "gpu_energy_ratio_vs_vanilla": (van_gpu.mean() / c_gpu.mean()
                                                        if c_gpu.mean() else np.nan),
                        "speedup_vs_vanilla": (van_time.mean() / c_time.mean()
                                               if not van_time.empty and not c_time.empty and c_time.mean() else np.nan),
                    })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Evaluation stage 2: aggregate + comparisons")
    ap.add_argument("--runs-csv", default=str(DEFAULT_CSV),
                    help="runs.csv from compute_metrics.py")
    args = ap.parse_args()

    csv = Path(args.runs_csv)
    if not csv.exists():
        print(f"{csv} not found — run compute_metrics.py first.")
        return
    df = pd.read_csv(csv)
    out_dir = csv.parent

    summary = summarize(df)
    s_off = savings_vs_off(df)
    s_van = savings_vs_vanilla(df)
    g_trend = s_off[s_off["gamma"].notna()].copy() if not s_off.empty else s_off

    summary.to_csv(out_dir / "summary_by_group.csv", index=False)
    s_off.to_csv(out_dir / "savings_vs_off.csv", index=False)
    s_van.to_csv(out_dir / "savings_vs_vanilla.csv", index=False)
    g_trend.to_csv(out_dir / "gamma_trend.csv", index=False)
    print(f"wrote 4 tables -> {out_dir}")

    # Headline to stdout.
    if not s_off.empty:
        print("\n=== energy saving vs SD-without-DVFS (mean %, GPU energy) ===")
        piv = s_off.pivot_table(index=["model_pair", "dataset", "strategy"],
                                columns="dvfs_condition",
                                values="gpu_energy_saving_pct", aggfunc="mean")
        print(piv.round(1).to_string())
        best = s_off.loc[s_off["gpu_energy_saving_pct"].idxmax()]
        print(f"\nbest single result: {best['gpu_energy_saving_pct']:.1f}% GPU energy saved "
              f"({best['model_pair']} / {best['strategy']} / {best['dvfs_condition']} / {best['dataset']}), "
              f"latency overhead {best['latency_overhead_pct']:.1f}%")
        print(f"mean latency overhead across DVFS conditions: "
              f"{s_off['latency_overhead_pct'].mean():.1f}%")
    else:
        print("\n(no SD+DVFS-vs-off comparisons available — need both `off` and a DVFS "
              "condition for the same model/strategy/dataset)")


if __name__ == "__main__":
    main()

# =============================================================================
# USAGE
# =============================================================================
#   python evaluation/aggregate_results.py --runs-csv evaluation/out/pilot/runs.csv
#   python evaluation/aggregate_results.py --runs-csv evaluation/out/full/runs.csv
# Reads first column of savings_vs_off.csv for the headline energy-savings result.
