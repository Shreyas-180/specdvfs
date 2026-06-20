#!/usr/bin/env bash
# =============================================================================
# run_analysis_local.sh — run this ON YOUR LAPTOP, after collect_and_destroy.sh
# has pulled the raw data down from the VM.
#
# The VM only ever produces RAW data (pilot run JSONs + raw ncu CSVs + phase
# timing JSONs). ALL interpretation happens here, locally, so you don't burn GPU
# time on pandas/matplotlib and you keep the analysis reproducible off-VM.
#
# It reads from the folder collect_and_destroy.sh wrote to (default
# ./results_from_vm) and writes the tables/figures into the repo's normal output
# dirs (evaluation/out/<mode>/ and profiling/out/). Each half is skipped cleanly
# if that raw data wasn't pulled (e.g. you ran the pilot but not roofline yet).
#
# Needs the analysis deps locally: pandas, numpy, scipy, matplotlib (+ seaborn).
#
# USAGE:
#   bash run_analysis_local.sh                 # uses ./results_from_vm, mode 'pilot'
#   bash run_analysis_local.sh ./results_from_vm full
# =============================================================================

set -uo pipefail

PULLED="${1:-./results_from_vm}"   # where collect_and_destroy.sh put the raw data
MODE="${2:-pilot}"                 # 'pilot' or 'full' (matches the run JSON subfolder)

echo "==> analysis source: ${PULLED}   mode: ${MODE}"

# ---------- pilot/full: run JSONs -> metrics CSV -> tables -> figures ----------
RUN_JSON_DIR="${PULLED}/results/${MODE}"
if [ -d "$RUN_JSON_DIR" ] && ls "$RUN_JSON_DIR"/*.json >/dev/null 2>&1; then
  echo "==> [1/2] ${MODE} metrics + tables + figures  (from ${RUN_JSON_DIR})"
  python evaluation/compute_metrics.py   --results-dir "$RUN_JSON_DIR"
  python evaluation/aggregate_results.py --runs-csv "evaluation/out/${MODE}/runs.csv"
  python evaluation/generate_figures.py  --runs-csv "evaluation/out/${MODE}/runs.csv"
  echo "    -> evaluation/out/${MODE}/  (runs.csv, savings_*.csv, summary_*.csv, figures/)"
else
  echo "==> [1/2] no run JSONs under ${RUN_JSON_DIR} — skipping ${MODE} analysis."
fi

# ---------- roofline: raw ncu CSVs -> intensity + GO/NO-GO verdict + plot -------
DRAFT="${PULLED}/profiling/out/draft.csv"
VERIFY="${PULLED}/profiling/out/verify.csv"
if [ -f "$DRAFT" ] && [ -f "$VERIFY" ]; then
  echo "==> [2/2] roofline verdict  (from ${DRAFT} + ${VERIFY})"
  python profiling/analyze_roofline.py --draft-csv "$DRAFT" --verify-csv "$VERIFY"
  echo "    -> profiling/out/roofline.json + roofline.png"
else
  echo "==> [2/2] no roofline CSVs under ${PULLED}/profiling/out — skipping roofline analysis."
  echo "          (phase timing JSONs, if any, are already in ${PULLED}/profiling/out/ — raw, no analysis step.)"
fi

echo "==> done."
