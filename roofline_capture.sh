#!/bin/bash
# SpecDVFS roofline CAPTURE — run AFTER roofline_preflight.sh passes.
#
#   bash roofline_capture.sh
#
# CAPTURE PLAN: one sm82 baseline + four batch levels {4,8,11,22}, draft and verify
# each = 10 captures. The reason there is only ONE SM capture rather than one per SM
# level:
#
#   Arithmetic intensity I = FLOPs/bytes is a property of the computation, not of how
#   many SMs run it. Restricting SMs changes how fast kernels execute, not how many
#   FLOPs they perform or how many bytes they move, so I is identical across SM counts.
#   What changes with SM count is the machine's ridge point, I*(N) = (peak_TFLOPS x
#   N/82) / peak_BW, and analyze_roofline.py --sm-count N computes that from a single
#   measured intensity pair -- so all SM-level verdicts derive from the one sm82 capture.
#
#   This also sidesteps a hard tooling constraint: ncu cannot profile MPS clients with
#   the flags this project needs. --mps client forbids --replay-mode application (which
#   is required to avoid vLLM's CPU-readback deadlock under the default kernel-replay),
#   and also forbids --csv and --export. Since SM-restricted captures are unnecessary,
#   no MPS is used here at all.
#
#   Batch size, unlike SM count, genuinely changes intensity (more concurrent sequences
#   amortise the same weight-matrix reads over more FLOPs), so each batch level needs
#   its own capture.
#
# FLAGS: --replay-mode application avoids the vLLM spec-decode deadlock that the default
# kernel-replay hits. --launch-count is intentionally NOT used (it interacts poorly with
# application replay here).

# Disable torch.compile()/Dynamo/Inductor -- see roofline_preflight.sh for the rationale.
# Without this, a cold compile cache on a fresh VM can add 10-15 min of CPU-only
# compilation (zero GPU activity, resembling a hang) before the first capture starts.
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

cd ~/specdvfs || { echo "FAIL: ~/specdvfs not found"; exit 1; }
mkdir -p profiling/out

# Hard stop if MPS is up -- ncu cannot profile MPS clients with the flags used below.
if pgrep -f nvidia-cuda-mps-control >/dev/null; then
  echo "FAIL: MPS daemon is running. ncu cannot profile MPS clients with our flags."
  echo "      Stop it first:  echo quit | nvidia-cuda-mps-control"
  exit 1
fi
echo "confirmed: MPS off, output dir ready"

M="dram__bytes.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__ops_path_tensor_src_bf16_dst_fp32.sum"

DONE=0; BAD=0

# Validate each CSV as it is written, so a header-only/empty capture is caught
# immediately rather than at the end of the whole run.
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
  # 3000s timeout: batch captures (n_prompts = batch size) run substantially longer than a
  # batch=1 capture. Concurrent sequences add scheduling/batch-management overhead per
  # kernel-launch cycle, which compounds across the multiple full-application relaunches that
  # --replay-mode application performs per metric group; a draft-phase batch capture can take
  # well over 30 minutes. Sustained elevated GPU power (well above the ~20-40 W idle floor)
  # during a long capture indicates genuine work rather than a hang.
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
# CONCURRENCY MUST MATCH THE ENERGY RUNS
# =============================================================================
# Each roofline capture must run at the same concurrency as the energy run it is meant
# to explain. max_num_seqs is a concurrency admission cap: it only binds when that many
# sequences are actually in flight, so a capture with n_prompts=1 runs a batch of one
# regardless of max_num_seqs. Because arithmetic intensity scales ~linearly with
# concurrency in the weight-dominated regime, a batch-1 capture understates the real
# intensity and would describe a configuration that was never benchmarked.
# prof_roofline.py auto-raises --n-prompts to --max-num-seqs; it is also passed
# explicitly here so the intent is visible in the command line and logs.
#
# Each capture below mirrors the engine settings of the energy runs it explains:
#   SM sweep    -> max_num_seqs 8,  max_model_len 2048
#   batch sweep -> max_num_seqs BS, max_model_len 1024
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
