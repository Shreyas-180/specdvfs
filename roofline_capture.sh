#!/bin/bash
# SpecDVFS roofline CAPTURE — run AFTER roofline_preflight.sh passes.
#
#   bash roofline_capture.sh
#
# WHY THERE ARE ONLY 3 CONFIGS (6 captures), NOT 5 SM LEVELS:
#
#   Arithmetic intensity I = FLOPs/bytes is a property of the COMPUTATION, not of
#   how many SMs are available to run it. Restricting SMs changes how FAST kernels
#   execute, not how many FLOPs they perform or how many bytes they move. So I is
#   IDENTICAL at 82, 56, 48 and 40 SMs.
#
#   What DOES change with SM count is the machine's ridge point:
#       I*(N) = (peak_TFLOPS x N/82) / peak_BW
#   and analyze_roofline.py --sm-count N already computes exactly that from a
#   single measured intensity pair. Verified end-to-end: one unrestricted capture
#   fed to --sm-count {82,56,48,40} reproduces all four verdicts correctly.
#
#   This matters because ncu CANNOT profile MPS clients with the flags this
#   project requires: --mps client forbids --replay-mode application (needed to
#   avoid vLLM's CPU-readback deadlock under the default kernel-replay), and also
#   forbids --csv and --export. Since SM-restricted captures are unnecessary,
#   that entire incompatibility is simply sidestepped -- no MPS is used here.
#
#   Batch size is different: it genuinely DOES change intensity (more sequences
#   amortise the same weight-matrix reads over more FLOPs), so bs8 and bs11 need
#   their own captures. Neither needs MPS either.
#
# FLAGS: --replay-mode application (required -- avoids the vLLM spec-decode
# deadlock that the default kernel-replay hits) and --csv. NOT --launch-count:
# it was never confirmed to produce a working capture and coincided with two
# distinct failure modes; every capture that succeeded did so without it.

# Disable torch.compile()/Dynamo/Inductor -- see roofline_preflight.sh for the
# full explanation. Without this, a fresh VM's cold compile cache can add
# 10-15+ minutes of CPU-only compilation (zero GPU activity, looks like a
# hang) before the FIRST capture even starts. Confirmed fix on 2026-07-27.
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

cd ~/specdvfs || { echo "FAIL: ~/specdvfs not found"; exit 1; }
mkdir -p profiling/out

# Hard stop if MPS is up -- this single condition broke every capture last session.
if pgrep -f nvidia-cuda-mps-control >/dev/null; then
  echo "FAIL: MPS daemon is running. ncu cannot profile MPS clients with our flags."
  echo "      Stop it first:  echo quit | nvidia-cuda-mps-control"
  exit 1
fi
echo "confirmed: MPS off, output dir ready"

M="dram__bytes.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__ops_path_tensor_src_bf16_dst_fp32.sum"

DONE=0; BAD=0

# Validate a CSV the moment it is written, rather than discovering at the end that
# 12 files are all header-only (which is exactly what happened last session).
validate () {
  local f="$1"
  if [ ! -s "$f" ]; then echo "   !! EMPTY: $f"; return 1; fi
  grep -qi "metric name" "$f" || { echo "   !! no CSV header in $f (see $f.log)"; return 1; }
  local nz
  nz=$(grep "sm__ops_path_tensor_src_bf16_dst_fp32.sum" "$f" | grep -vc ',"0"$')
  if [ "${nz:-0}" -lt 1 ]; then echo "   !! no non-zero tensor-op rows in $f"; return 1; fi
  echo "   OK: $(basename "$f") -- $(wc -l < "$f") lines, $nz non-zero tensor-op rows"
  return 0
}

capture () {
  local OUT="$1"; shift
  echo ""
  echo "-- $(date +%H:%M:%S)  capturing $(basename "$OUT")  (timeout 3000s / 50min) --"
  # 3000s, not 1500 -- this is the FIRST time n_prompts=8 (and n_prompts=BS for the
  # batch levels) has ever been captured under ncu in this project. Every previously
  # validated duration (the ~13-14 min draft-phase ceiling that 1500s was based on)
  # came from batch=1 captures. 8 concurrent sequences genuinely add scheduling/
  # batch-management overhead per kernel-launch cycle, which compounds across the
  # multiple full-application relaunches --replay-mode application needs per metric
  # group. A real batch=8/11/22 capture is expected to run substantially longer than
  # the batch=1 baseline this timeout was originally calibrated against -- confirmed
  # 2026-07-28: a batch=8 draft capture exceeded 30 real minutes while nvidia-smi
  # showed sustained ~108W (well above the ~21-37W true-idle floor), consistent with
  # genuine work, not a hang.
  if timeout 3000 env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE -u SPECDVFS_SM_COUNT \
       "$@" > "$OUT" 2>"${OUT}.log"; then
    if validate "$OUT"; then DONE=$((DONE+1)); return; fi
  else
    echo "   !! ncu exited non-zero / timed out (rc=$?) -- see ${OUT}.log"
    tail -3 "$OUT" 2>/dev/null | sed 's/^/      /'
  fi
  echo "   retrying once..."
  pkill -9 -f prof_roofline.py 2>/dev/null; pkill -9 -f ncu 2>/dev/null; sleep 3
  if timeout 3000 env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE -u SPECDVFS_SM_COUNT \
       "$@" > "$OUT" 2>"${OUT}.log" && validate "$OUT"; then
    DONE=$((DONE+1))
  else
    echo "   !! FAILED TWICE: $OUT"
    tail -5 "$OUT" 2>/dev/null | sed 's/^/      /'
    BAD=$((BAD+1))
  fi
}

# =============================================================================
# CONCURRENCY MUST MATCH THE ENERGY RUNS  (fixed 2026-07-28)
# =============================================================================
# The previous version of this script passed --n-prompts 1 to EVERY capture. That
# made max_num_seqs a no-op (it is a concurrency ADMISSION CAP -- it only binds
# when that many sequences are actually in flight), so:
#   * bs8 and bs11 measured the identical computation (verify I = 61.24 vs 61.25);
#   * sm82 measured a batch of ONE, while the SM-sweep energy runs used batch 8.
# Arithmetic intensity scales ~linearly with concurrency in the weight-dominated
# regime, so a batch-1 capture materially understates the real intensity and the
# resulting roofline verdicts describe a configuration that was never benchmarked.
# prof_roofline.py now auto-raises --n-prompts to --max-num-seqs, but we pass it
# explicitly here so the intent is visible in the command line and in the logs.
#
# Each capture below mirrors the engine settings of the energy runs it explains:
#   SM sweep   -> max_num_seqs 8,  max_model_len 2048
#   batch sweep-> max_num_seqs BS, max_model_len 1024
# =============================================================================

# ---- config A: sm82 baseline, at the SM sweep's REAL concurrency (8). ----
#      Feeds every --sm-count analysis, so this one must be right.
for PH in draft verify; do
  capture "profiling/out/${PH}_sm82.csv" \
    ncu --replay-mode application --profile-from-start off --nvtx \
        --nvtx-include "specdvfs_${PH}/" --metrics "$M" --target-processes all --csv \
        python profiling/prof_roofline.py --gamma 18 \
            --max-num-seqs 8 --max-model-len 2048 --n-prompts 8 --max-tokens 24
done

# ---- config B: batch levels, matching the batch sweep's cells. ----
#      4 and 22 are the U-curve extremes (highest savings); 8 is the dip; 11 was
#      the originally predicted crossing. Four levels are enough to establish
#      whether I_verify rises with concurrency and where it crosses the ridge.
#      Add 16 to BATCH_LEVELS for a complete 1:1 match with the energy sweep.
BATCH_LEVELS="4 8 11 22"
for BS in $BATCH_LEVELS; do
  for PH in draft verify; do
    capture "profiling/out/${PH}_bs${BS}.csv" \
      ncu --replay-mode application --profile-from-start off --nvtx \
          --nvtx-include "specdvfs_${PH}/" --metrics "$M" --target-processes all --csv \
          python profiling/prof_roofline.py --gamma 18 \
              --max-num-seqs "$BS" --max-model-len 1024 --n-prompts "$BS" --max-tokens 24
  done
done

# Expected capture count: 2 (sm82) + 2 x number of BATCH_LEVELS
EXPECTED=$(( 2 + 2 * $(echo $BATCH_LEVELS | wc -w) ))

echo ""
echo "================================================================"
echo "CAPTURES: $DONE ok, $BAD failed  (expected $EXPECTED ok)"
ls -la profiling/out/*.csv 2>/dev/null | awk '{print "  "$9"  "$5" bytes"}'
if [ "$BAD" -eq 0 ] && [ "$DONE" -eq "$EXPECTED" ]; then
  echo ""
  echo "ALL GOOD. Pull these home, then on your LAPTOP run:"
  echo "  bash roofline_analyze_all.sh"
else
  echo ""
  echo "Some captures failed -- check the matching .log files before pulling home."
fi
echo "================================================================"