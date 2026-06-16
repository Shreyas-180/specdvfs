"""Evaluation — Stage 3: figures.

Reads runs.csv (from compute_metrics.py) and the savings tables (from
aggregate_results.py) in the same directory, and writes PNGs to <dir>/figures/.

Figures:
  energy_savings_by_condition.png  GPU energy saved (%) vs SD-without-DVFS, per condition,
                                    faceted by dataset.  [the headline]
  gamma_trend.png                  energy saved (%) vs draft length (gamma), per condition.
                                    [does the per-phase benefit grow with the asymmetry?]
  energy_vs_latency.png            scatter: latency overhead (%) vs energy saved (%).
                                    [the "minimal latency loss" tradeoff — want top-left]
  wh_per_1k_by_condition.png       absolute efficiency (Wh per 1k tokens) per condition.
  clock_levels_by_condition.png    distinct SM-clock levels used per condition.
                                    [DVFS conditions should use >1 level; off/fixed_low = 1]

Headless (Agg backend). matplotlib + pandas only (no seaborn dependency).
NOTE: a time-resolved draft/verify clock TRACE is a Phase-1 profiling artifact, not
available from the per-run summary here; this stage shows the clock-LEVEL evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "evaluation" / "out" / "pilot" / "runs.csv"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_savings_by_condition(savings_off: pd.DataFrame, fig_dir: Path):
    if savings_off is None or savings_off.empty:
        print("  skip energy_savings_by_condition (no savings_vs_off.csv)")
        return
    datasets = sorted(savings_off["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.2),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = savings_off[savings_off["dataset"] == ds]
        piv = sub.pivot_table(index="strategy", columns="dvfs_condition",
                              values="gpu_energy_saving_pct", aggfunc="mean")
        piv.plot(kind="bar", ax=ax, legend=(ds == datasets[-1]))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{ds}")
        ax.set_ylabel("GPU energy saved vs no-DVFS (%)")
        ax.set_xlabel("SD strategy")
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Energy saving from per-phase DVFS, by condition")
    _save(fig, fig_dir / "energy_savings_by_condition.png")


def fig_gamma_trend(gamma_trend: pd.DataFrame, fig_dir: Path):
    if gamma_trend is None or gamma_trend.empty:
        print("  skip gamma_trend (no gamma_trend.csv / no constant-gamma strategies)")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    # average over models/datasets to show the trend per condition
    g = (gamma_trend.groupby(["dvfs_condition", "gamma"])["gpu_energy_saving_pct"]
         .mean().reset_index())
    for cond, sub in g.groupby("dvfs_condition"):
        sub = sub.sort_values("gamma")
        ax.plot(sub["gamma"], sub["gpu_energy_saving_pct"], marker="o", label=cond)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("draft length gamma (num speculative tokens)")
    ax.set_ylabel("GPU energy saved vs no-DVFS (%)")
    ax.set_title("Does the DVFS benefit grow with the draft/verify ratio?")
    ax.legend(title="condition", fontsize=8)
    _save(fig, fig_dir / "gamma_trend.png")


def fig_energy_vs_latency(savings_off: pd.DataFrame, fig_dir: Path):
    if savings_off is None or savings_off.empty:
        print("  skip energy_vs_latency (no savings_vs_off.csv)")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for cond, sub in savings_off.groupby("dvfs_condition"):
        ax.scatter(sub["latency_overhead_pct"], sub["gpu_energy_saving_pct"],
                   label=cond, s=40, alpha=0.8)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("latency overhead vs no-DVFS (%)  ->  worse")
    ax.set_ylabel("GPU energy saved (%)  ->  better")
    ax.set_title("Energy saving vs latency cost (want top-left)")
    ax.legend(title="condition", fontsize=8)
    _save(fig, fig_dir / "energy_vs_latency.png")


def fig_wh_per_1k(runs: pd.DataFrame, fig_dir: Path):
    if "gpu_wh_per_1k" not in runs.columns or runs["gpu_wh_per_1k"].dropna().empty:
        print("  skip wh_per_1k_by_condition (no energy/token data)")
        return
    datasets = sorted(runs["dataset"].dropna().unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.2),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = runs[runs["dataset"] == ds]
        piv = sub.pivot_table(index="strategy", columns="dvfs_condition",
                              values="gpu_wh_per_1k", aggfunc="mean")
        piv.plot(kind="bar", ax=ax, legend=(ds == datasets[-1]))
        ax.set_title(f"{ds}")
        ax.set_ylabel("GPU Wh per 1k tokens (lower = better)")
        ax.set_xlabel("SD strategy")
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Absolute GPU energy efficiency")
    _save(fig, fig_dir / "wh_per_1k_by_condition.png")


def fig_clock_levels(runs: pd.DataFrame, fig_dir: Path):
    if "sm_clock_levels" not in runs.columns:
        print("  skip clock_levels_by_condition (no monitor data)")
        return
    g = runs.groupby("dvfs_condition")["sm_clock_levels"].mean().sort_values()
    if g.dropna().empty:
        print("  skip clock_levels_by_condition (monitor empty — e.g. --mock run)")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    g.plot(kind="bar", ax=ax, color="steelblue")
    ax.axhline(1, color="red", linewidth=1, linestyle="--",
               label="single fixed clock")
    ax.set_ylabel("distinct SM-clock levels used (mean)")
    ax.set_xlabel("DVFS condition")
    ax.set_title("DVFS-toggle evidence: per-phase conditions use multiple clocks")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=8)
    _save(fig, fig_dir / "clock_levels_by_condition.png")


def _maybe(path: Path):
    return pd.read_csv(path) if path.exists() else None


def main():
    ap = argparse.ArgumentParser(description="Evaluation stage 3: figures")
    ap.add_argument("--runs-csv", default=str(DEFAULT_CSV),
                    help="runs.csv from compute_metrics.py")
    args = ap.parse_args()

    csv = Path(args.runs_csv)
    if not csv.exists():
        print(f"{csv} not found — run compute_metrics.py first.")
        return
    runs = pd.read_csv(csv)
    out_dir = csv.parent
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    savings_off = _maybe(out_dir / "savings_vs_off.csv")
    gamma_trend = _maybe(out_dir / "gamma_trend.csv")
    if savings_off is None:
        print("  note: run aggregate_results.py first for the savings-based figures.")

    fig_savings_by_condition(savings_off, fig_dir)
    fig_gamma_trend(gamma_trend, fig_dir)
    fig_energy_vs_latency(savings_off, fig_dir)
    fig_wh_per_1k(runs, fig_dir)
    fig_clock_levels(runs, fig_dir)
    print(f"figures in {fig_dir}")


if __name__ == "__main__":
    main()

# =============================================================================
# USAGE
# =============================================================================
#   python evaluation/compute_metrics.py   --results-dir results/pilot
#   python evaluation/aggregate_results.py --runs-csv evaluation/out/pilot/runs.csv
#   python evaluation/generate_figures.py  --runs-csv evaluation/out/pilot/runs.csv
# Figures land in evaluation/out/pilot/figures/.
