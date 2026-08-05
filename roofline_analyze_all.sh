#!/bin/bash
# SpecDVFS roofline ANALYSIS — run on your LAPTOP after pulling the CSVs home.
#
#   bash roofline_analyze_all.sh [PULLED_DIR]      (default ./results_from_vm)
#
# Produces, under profiling/out/:
#   roofline_sm{82,56,48,40}.{json,png}  — Approach 2: same measured intensities,
#        ridge scaled by SM fraction (see roofline_capture.sh for why one capture suffices)
#   roofline_bs{8,11}.{json,png}         — Approach 1: intensity genuinely changes with batch
#   plus a printed summary table of every verdict.

PULLED="${1:-./results_from_vm}"
OUT="$PULLED/profiling/out"
PY="$(command -v python3 || command -v python)"
[ -z "$PY" ] && { echo "FAIL: no python3/python on PATH"; exit 1; }

mkdir -p profiling/out
missing=0
BATCH_LEVELS="${BATCH_LEVELS:-4 8 11 22}"   # must match roofline_capture.sh
REQ="draft_sm82 verify_sm82"
for B in $BATCH_LEVELS; do REQ="$REQ draft_bs$B verify_bs$B"; done
for f in $REQ; do
  [ -f "$OUT/$f.csv" ] || { echo "MISSING: $OUT/$f.csv"; missing=1; }
done
[ "$missing" -eq 1 ] && { echo "-> pull the CSVs home first (rsync), then re-run."; exit 1; }
echo "all $(echo $REQ | wc -w) CSVs present in $OUT"

echo ""
echo "=== Approach 2: SM restriction (one capture, ridge scaled per SM level) ==="
for SM in 82 56 48 40; do
  echo ""
  echo "--- $SM SMs ---"
  "$PY" profiling/analyze_roofline.py --gpu 3090 --sm-count "$SM" \
      --draft-csv  "$OUT/draft_sm82.csv" \
      --verify-csv "$OUT/verify_sm82.csv" \
      --out-json "profiling/out/roofline_sm${SM}.json" \
      --out-png  "profiling/out/roofline_sm${SM}.png"
done

echo ""
echo "=== Approach 1: batch size (intensity genuinely changes; full-GPU ridge) ==="
for BS in $BATCH_LEVELS; do
  echo ""
  echo "--- batch $BS ---"
  "$PY" profiling/analyze_roofline.py --gpu 3090 \
      --draft-csv  "$OUT/draft_bs${BS}.csv" \
      --verify-csv "$OUT/verify_bs${BS}.csv" \
      --out-json "profiling/out/roofline_bs${BS}.json" \
      --out-png  "profiling/out/roofline_bs${BS}.png"
done

echo ""
echo "================== SUMMARY =================="
"$PY" - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob("profiling/out/roofline_*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    tag = os.path.basename(f).replace("roofline_", "").replace(".json", "")
    star = d.get("ridge_flops_per_byte")
    ph = {p["name"]: p for p in d.get("phases", [])}
    dr = ph.get("draft", {}).get("intensity_flops_per_byte")
    ve = ph.get("verify", {}).get("intensity_flops_per_byte")
    # The JSON stores no explicit verdict flag, so derive it the same way
    # analyze_roofline.py does: the premise holds only when the ridge falls
    # BETWEEN the two phases (draft memory-bound, verify compute-bound).
    if None in (star, dr, ve):
        go = None
    else:
        go = dr < star <= ve
    rows.append((tag, star, dr, ve, go))
if rows:
    print(f"{'config':>8} {'ridge I*':>9} {'draft I':>9} {'verify I':>9}  verdict")
    for tag, star, dr, ve, go in rows:
        f2 = lambda x: f"{x:9.2f}" if isinstance(x, (int, float)) else f"{str(x):>9}"
        v = "?" if go is None else ("GO" if go else "NO-GO")
        print(f"{tag:>8} {f2(star)} {f2(dr)} {f2(ve)}  {v}")
else:
    print("(no roofline_*.json found -- check the per-config output above)")
PY
echo "============================================"
echo "plots: profiling/out/roofline_*.png"