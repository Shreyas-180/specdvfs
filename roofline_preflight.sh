#!/bin/bash
# SpecDVFS roofline PREFLIGHT — run this FIRST, before roofline_capture.sh.
#
# Every check here corresponds to a failure that actually cost GPU time in a
# previous session. Total cost ~5 min, versus ~45 min for the real captures.
# If this passes, the real captures are very likely to succeed.
#
#   bash roofline_preflight.sh

# Disable torch.compile()/Dynamo/Inductor. On a FRESH VM with no compile cache,
# something in the stack (not our own enforce_eager=True model-forward path --
# that specifically avoids CUDA-graph capture, not this) triggers cold-cache
# JIT kernel compilation: 18 parallel `torch._inductor.compile_worker` processes
# spawn, all CPU-bound (a C++ compiler per worker), producing ZERO GPU activity
# for 10-15+ minutes -- looks exactly like a hang in nvidia-smi but isn't one.
# Confirmed on 2026-07-27: with these two vars set, an identical selftest that
# had exceeded a 900s timeout completed in ~15s (matching every prior working
# run). We don't want compiled kernels anyway -- ncu can't profile them cleanly,
# which is the whole reason enforce_eager=True was chosen in the first place.
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

cd ~/specdvfs || { echo "FAIL: ~/specdvfs not found"; exit 1; }

PASS=0; FAIL=0
ok ()   { echo "  [ok]   $1"; PASS=$((PASS+1)); }
bad ()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
sect () { echo ""; echo "=== $1 ==="; }

sect "1. MPS must be OFF"
# THE critical one. ncu CANNOT profile MPS clients with the flags this project
# needs: --mps client forbids --replay-mode application (required to avoid the
# vLLM CPU-readback deadlock), and also forbids --csv and --export. If MPS is
# running, every capture dies with "Profiling is not supported with MPS enabled".
if pgrep -f nvidia-cuda-mps-control >/dev/null; then
  bad "MPS daemon IS RUNNING -- captures will fail. Stop it:  echo quit | nvidia-cuda-mps-control"
else
  ok "MPS is off (required -- ncu cannot profile MPS clients with our flags)"
fi
if env | grep -q CUDA_MPS_ACTIVE_THREAD_PERCENTAGE; then
  bad "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is set in this shell -- unset it"
else
  ok "no stale MPS env vars in this shell"
fi

sect "2. Output directory"
mkdir -p profiling/out
[ -d profiling/out ] && ok "profiling/out exists (shell '>' redirect fails silently without it)" \
                     || bad "could not create profiling/out"

sect "3. HuggingFace auth (gated repos)"
if [ -z "${HF_TOKEN:-}" ]; then
  bad "HF_TOKEN not set in this shell -- build_llm() will die with GatedRepoError 401"
else
  ok "HF_TOKEN present"
  python3 - <<'PY' 2>/dev/null && echo "  [ok]   gated repo access confirmed" || echo "  [FAIL] cannot access gated repo -- check token + license acceptance"
import os
from huggingface_hub import model_info
model_info('meta-llama/Llama-3.1-8B-Instruct', token=os.environ['HF_TOKEN'])
PY
fi

sect "4. ncu present + metric names valid on THIS driver"
if command -v ncu >/dev/null; then
  ok "ncu found: $(ncu --version 2>/dev/null | grep -i version | head -1)"
  for MET in dram__bytes sm__ops_path_tensor_src_bf16_dst_fp32; do
    if ncu --query-metrics 2>/dev/null | grep -q "^${MET}\b"; then
      ok "metric available: $MET"
    else
      bad "metric NOT available on this driver: $MET"
    fi
  done
else
  bad "ncu not on PATH"
fi

sect "5. NVTX ranges actually fire (no ncu, fast)"
# If the draft/verify ranges don't fire, every capture produces an empty CSV
# even though ncu exits 0 -- a silent failure mode.
if timeout 120 python profiling/prof_roofline.py --gamma 18 --selftest 2>&1 | tee /tmp/pf_selftest.log | grep -q "hooks OK"; then
  ok "NVTX draft+verify ranges fire at gamma=18"
else
  bad "selftest did not report 'hooks OK' -- see /tmp/pf_selftest.log"
fi

sect "6. SMOKE TEST: one real tiny ncu capture (gamma=5, ~4 min)"
# Uses the EXACT configuration that is known to have worked before. This proves
# the whole chain (ncu attach -> NVTX filter -> metric collection -> CSV write)
# end to end, cheaply, before committing ~45 min to the gamma=18 captures.
M="dram__bytes.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__ops_path_tensor_src_bf16_dst_fp32.sum"
echo "  (running -- expect ~3-5 min; ~110W+ in nvidia-smi means it is working)"
if timeout 600 env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE -u SPECDVFS_SM_COUNT \
     ncu --replay-mode application --profile-from-start off --nvtx \
         --nvtx-include "specdvfs_draft/" --metrics "$M" --target-processes all --csv \
         python profiling/prof_roofline.py --gamma 5 --n-prompts 1 --max-tokens 8 \
         > /tmp/pf_smoke.csv 2>/tmp/pf_smoke.log; then
  if grep -qi "metric name" /tmp/pf_smoke.csv; then
    NZ=$(grep "sm__ops_path_tensor_src_bf16_dst_fp32.sum" /tmp/pf_smoke.csv | grep -vc ',"0"$')
    if [ "${NZ:-0}" -gt 0 ]; then
      ok "smoke capture produced $NZ non-zero tensor-op rows (real GEMM kernels)"
    else
      bad "smoke CSV has a header but NO non-zero tensor-op rows -- NVTX filter matched nothing"
    fi
  else
    bad "smoke CSV has no 'Metric Name' header -- see /tmp/pf_smoke.csv and /tmp/pf_smoke.log"
  fi
else
  bad "smoke capture failed/timed out (rc=$?) -- see /tmp/pf_smoke.log"
fi

echo ""
echo "================================================================"
if [ "$FAIL" -eq 0 ]; then
  echo "PREFLIGHT PASSED ($PASS checks). Safe to run: bash roofline_capture.sh"
else
  echo "PREFLIGHT FAILED: $FAIL problem(s), $PASS ok."
  echo "Fix the [FAIL] lines above BEFORE running the real captures."
fi
echo "================================================================"