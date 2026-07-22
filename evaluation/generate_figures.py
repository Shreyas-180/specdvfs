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
    # Facet by model_pair too: with >1 pair in the data, a plot indexed by strategy alone
    # would silently average two different drafts' results into one misleading bar.
    models = sorted(savings_off["model_pair"].unique())
    datasets = sorted(savings_off["dataset"].unique())
    fig, axes = plt.subplots(len(models), len(datasets),
                             figsize=(5 * len(datasets), 4.2 * len(models)), squeeze=False)
    for i, model in enumerate(models):
        for j, ds in enumerate(datasets):
            ax = axes[i][j]
            sub = savings_off[(savings_off["model_pair"] == model) & (savings_off["dataset"] == ds)]
            piv = sub.pivot_table(index="strategy", columns="dvfs_condition",
                                  values="gpu_energy_saving_pct", aggfunc="mean")
            piv.plot(kind="bar", ax=ax, legend=(i == 0 and j == len(datasets) - 1))
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"{model} / {ds}")
            ax.set_ylabel("GPU energy saved vs no-DVFS (%)" if j == 0 else "")
            ax.set_xlabel("SD strategy")
            ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Energy saving from per-phase DVFS, by condition")
    _save(fig, fig_dir / "energy_savings_by_condition.png")


def fig_gamma_trend(gamma_trend: pd.DataFrame, fig_dir: Path):
    if gamma_trend is None or gamma_trend.empty:
        print("  skip gamma_trend (no gamma_trend.csv / no constant-gamma strategies)")
        return
    models = sorted(gamma_trend["model_pair"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 4.2), squeeze=False)
    for ax, model in zip(axes[0], models):
        sub_model = gamma_trend[gamma_trend["model_pair"] == model]
        g = (sub_model.groupby(["dvfs_condition", "gamma"])["gpu_energy_saving_pct"]
             .mean().reset_index())
        for cond, sub in g.groupby("dvfs_condition"):
            sub = sub.sort_values("gamma")
            ax.plot(sub["gamma"], sub["gpu_energy_saving_pct"], marker="o", label=cond)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("draft length gamma (num speculative tokens)")
        ax.set_ylabel("GPU energy saved vs no-DVFS (%)")
        ax.set_title(model)
        ax.legend(title="condition", fontsize=8)
    fig.suptitle("Does the DVFS benefit grow with the draft/verify ratio?")
    _save(fig, fig_dir / "gamma_trend.png")


def fig_energy_vs_latency(savings_off: pd.DataFrame, fig_dir: Path):
    if savings_off is None or savings_off.empty:
        print("  skip energy_vs_latency (no savings_vs_off.csv)")
        return
    models = sorted(savings_off["model_pair"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 4.6), squeeze=False)
    for ax, model in zip(axes[0], models):
        sub_model = savings_off[savings_off["model_pair"] == model]
        for cond, sub in sub_model.groupby("dvfs_condition"):
            ax.scatter(sub["latency_overhead_pct"], sub["gpu_energy_saving_pct"],
                       label=cond, s=40, alpha=0.8)
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.axvline(0, color="grey", linewidth=0.6)
        ax.set_xlabel("latency overhead vs no-DVFS (%)  ->  worse")
        ax.set_ylabel("GPU energy saved (%)  ->  better")
        ax.set_title(model)
        ax.legend(title="condition", fontsize=8)
    fig.suptitle("Energy saving vs latency cost (want top-left)")
    _save(fig, fig_dir / "energy_vs_latency.png")


def fig_wh_per_1k(runs: pd.DataFrame, fig_dir: Path):
    if "gpu_wh_per_1k" not in runs.columns or runs["gpu_wh_per_1k"].dropna().empty:
        print("  skip wh_per_1k_by_condition (no energy/token data)")
        return
    models = sorted(runs["model_pair"].dropna().unique())
    datasets = sorted(runs["dataset"].dropna().unique())
    fig, axes = plt.subplots(len(models), len(datasets),
                             figsize=(5 * len(datasets), 4.2 * len(models)), squeeze=False)
    for i, model in enumerate(models):
        for j, ds in enumerate(datasets):
            ax = axes[i][j]
            sub = runs[(runs["model_pair"] == model) & (runs["dataset"] == ds)]
            piv = sub.pivot_table(index="strategy", columns="dvfs_condition",
                                  values="gpu_wh_per_1k", aggfunc="mean")
            piv.plot(kind="bar", ax=ax, legend=(i == 0 and j == len(datasets) - 1))
            ax.set_title(f"{model} / {ds}")
            ax.set_ylabel("GPU Wh per 1k tokens (lower = better)" if j == 0 else "")
            ax.set_xlabel("SD strategy")
            ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Absolute GPU energy efficiency")
    _save(fig, fig_dir / "wh_per_1k_by_condition.png")


def fig_clock_levels(runs: pd.DataFrame, fig_dir: Path):
    if "sm_clock_levels" not in runs.columns:
        print("  skip clock_levels_by_condition (no monitor data)")
        return
    models = sorted(runs["model_pair"].dropna().unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 4.0), squeeze=False)
    any_plotted = False
    for ax, model in zip(axes[0], models):
        g = (runs[runs["model_pair"] == model]
             .groupby("dvfs_condition")["sm_clock_levels"].mean().sort_values())
        if g.dropna().empty:
            ax.set_visible(False)
            continue
        any_plotted = True
        g.plot(kind="bar", ax=ax, color="steelblue")
        ax.axhline(1, color="red", linewidth=1, linestyle="--",
                   label="single fixed clock")
        ax.set_ylabel("distinct SM-clock levels used (mean)")
        ax.set_xlabel("DVFS condition")
        ax.set_title(model)
        ax.tick_params(axis="x", rotation=0)
        ax.legend(fontsize=8)
    if not any_plotted:
        print("  skip clock_levels_by_condition (monitor empty — e.g. --mock run)")
        plt.close(fig)
        return
    fig.suptitle("DVFS-toggle evidence: per-phase conditions use multiple clocks")
    _save(fig, fig_dir / "clock_levels_by_condition.png")


def fig_ridge_sweep(runs: pd.DataFrame, fig_dir: Path, axis: str):
    """Energy saving vs the ridge-crossing axis (batch_size or sm_count).

    THE headline figure for both approaches. For each axis level it computes, per DVFS
    condition, the GPU-energy saving relative to that SAME level's 'off' run — so every point
    is an apples-to-apples within-level comparison and the axis is the only thing varying.
    A vertical line marks where verify is predicted to cross the ridge; the question the
    figure answers is whether the DVFS advantage visibly changes across that line.
    """
    col = "batch_size" if axis == "batch" else "sm_count"
    if col not in runs.columns or runs[col].dropna().nunique() < 2:
        print(f"  skip ridge_sweep_{axis} (need >=2 distinct {col} values)")
        return
    if "gpu_energy_Wh" not in runs.columns:
        print(f"  skip ridge_sweep_{axis} (no gpu_energy_Wh)")
        return

    rows = []
    for (lvl, ds), sub in runs.groupby([col, "dataset"]):
        off = sub[sub["dvfs_condition"] == "off"]["gpu_energy_Wh"].mean()
        if not off or pd.isna(off):
            continue
        for cond, s2 in sub.groupby("dvfs_condition"):
            if cond == "off":
                continue
            e = s2["gpu_energy_Wh"].mean()
            if pd.isna(e):
                continue
            rows.append({col: lvl, "dataset": ds, "dvfs_condition": cond,
                         "saving_pct": (off - e) / off * 100.0})
    if not rows:
        print(f"  skip ridge_sweep_{axis} (no per-level 'off' baseline to compare against)")
        return
    d = pd.DataFrame(rows)

    datasets = sorted(d["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.0 * len(datasets), 4.4), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = d[d["dataset"] == ds]
        for cond, s in sub.groupby("dvfs_condition"):
            s = s.sort_values(col)
            ax.plot(s[col], s["saving_pct"], marker="o", label=cond)
        ax.axhline(0, color="black", linewidth=0.8)
        if axis == "batch":
            # I_verify = 58.5 x (batch x 19)/152 crosses I*=75.9 between batch 8 and 11
            ax.axvline(11, ls="--", color="crimson", linewidth=1.2,
                       label="predicted ridge crossing")
            ax.set_xlabel("MAX_NUM_SEQS (batch size)  ->  I_verify rises")
        else:
            # premise window from measured intensities: 45 <= SMs <= 63
            ax.axvspan(45, 63, color="mediumseagreen", alpha=0.15,
                       label="predicted premise window")
            ax.set_xlabel("SMs available  ->  ridge I* falls")
            ax.invert_xaxis()
        ax.set_ylabel("GPU energy saved vs 'off' at the SAME level (%)")
        ax.set_title(ds)
        ax.legend(fontsize=8)
    fig.suptitle("Approach 1: does DVFS help more once verify goes compute-bound?"
                 if axis == "batch" else
                 "Approach 2: does DVFS help more inside the premise window?")
    _save(fig, fig_dir / f"ridge_sweep_{axis}.png")


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
    # Ridge-crossing sweeps: both no-op unless the corresponding axis actually varies in the
    # data, so this is safe to call for the ordinary pilot too.
    fig_ridge_sweep(runs, fig_dir, "batch")
    fig_ridge_sweep(runs, fig_dir, "sm")
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