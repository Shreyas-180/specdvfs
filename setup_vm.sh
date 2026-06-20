#!/usr/bin/env bash
# =============================================================================
# setup_vm.sh — one-shot setup + verification for the SpecDVFS experiment VM.
#
# TARGET: a Vast.ai *VM* instance (NOT a Docker container) with an NVIDIA
#         RTX 3090 and a CUDA 12.9+ driver already provided by the VM image.
#         Run ONCE, right after you get SSH access, before any experiment.
#
# WHY A VM, NOT DOCKER: nvmlDeviceSetGpuLockedClocks needs SYS_ADMIN, which
#         Vast.ai never grants to renters' containers. VM passthrough gives the
#         guest driver full clock control. Step [6] proves this works.
#
# WHAT IT DOES:
#   [1] GPU + driver sanity check.
#   [2] Enable persistence mode.
#   [3] Create the `specdvfs` conda env + install the pinned stack.
#   [4] Authenticate to Hugging Face (gated Llama models need this).
#   [5] Confirm the project code is present (you copy it over first via push_to_vm.sh).
#   [6] VERIFY NVML CLOCK-LOCKING WORKS   <-- GO/NO-GO for the whole project.
#   [7] Prepare + verify datasets, print MD5 checksums (record them!).
#
# If step [6] FAILS, STOP: this instance cannot run the DVFS experiment.
# Get a different (non-Docker) VM with a lockable GPU.
# =============================================================================

set -euo pipefail

# Some Vast.ai VMs run as root (no sudo) and some don't. Detect once and use $SUDO.
if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

# ----------------------------- EDIT THESE ------------------------------------
# Only HF_TOKEN needs filling in. Either edit it here, OR export it before running:
#     export HF_TOKEN=hf_AbCd1234EXAMPLE5678
#     bash setup_vm.sh
#
# The project code is NOT cloned here — you copy it onto the VM first with
# push_to_vm.sh (run on your laptop), then run this script from inside that folder.
# CONDA_ENV has a sensible default; only change it if you want to.
HF_TOKEN="${HF_TOKEN:-}"          # <-- hf_... token (needed for gated meta-llama models), or leave blank + log in interactively
CONDA_ENV="${CONDA_ENV:-specdvfs}"
# -----------------------------------------------------------------------------

echo "============================================================"
echo "  SpecDVFS VM setup"
echo "============================================================"

echo "==> [0/7] System prerequisites (curl, git, wget; pip comes from conda)"
# DOUBT 1 (setup): you don't need a system `pip` — Miniconda + the conda env below
# provide pip. That is why the script never runs `apt install pip`. We DO make sure
# the few system tools the script relies on exist (the previous instance was missing
# some of these). All guarded so the script still works if apt/sudo are unavailable.
if command -v apt-get >/dev/null 2>&1; then
  $SUDO apt-get update -y || echo "    WARN: apt-get update failed (continuing)"
  $SUDO apt-get install -y curl git wget ca-certificates || echo "    WARN: apt-get install failed (continuing)"
  # If you ever insist on using the SYSTEM python instead of conda, also install:
  #   $SUDO apt-get install -y python3-pip
fi

echo "==> [1/7] GPU + driver sanity"
command -v nvidia-smi >/dev/null 2>&1 || { echo "FATAL: nvidia-smi not found — no GPU/driver"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total,temperature.gpu --format=csv,noheader

echo "==> [2/7] Enable persistence mode"
$SUDO nvidia-smi -pm 1 || echo "    WARN: could not enable persistence mode (continuing)"

echo "==> [3/7] Conda env + pinned stack"
if ! command -v conda >/dev/null 2>&1; then
  echo "    installing Miniconda..."
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
  conda init bash
else
  eval "$(conda shell.bash hook)"
fi

conda create -y -n "$CONDA_ENV" python=3.10
conda activate "$CONDA_ENV"
# DOUBT 1 (setup): the conda env ships pip by default, so no system pip is needed.
# ensurepip is a belt-and-suspenders fallback for the rare env that lacks it.
python -m pip --version >/dev/null 2>&1 || python -m ensurepip --upgrade
python -m pip install --upgrade pip

# vLLM 0.6.6 pulls a compatible torch (2.5.1). Let it resolve, then verify the
# CUDA build below. NOTE: do NOT add the WSL PATH fix here — that is laptop-only.
pip install "vllm==0.6.6"
pip install "transformers>=4.47" accelerate bitsandbytes datasets \
            nvidia-ml-py codecarbon scipy numpy pandas matplotlib seaborn huggingface_hub

# If torch's bundled CUDA build mismatches the 12.9 driver, force-reinstall, e.g.:
#   pip install --force-reinstall torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python - <<'PY'
import torch
print(f"    torch {torch.__version__} | CUDA build {torch.version.cuda} | sees GPU: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "torch cannot see the GPU"
print(f"    device: {torch.cuda.get_device_name(0)} | capability sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
PY

echo "    Nsight Compute (ncu) — required for profiling/prof_roofline.py (Phase 3)."
echo "    Installed into the conda env so no sudo/apt-repo wrangling is needed."
if command -v ncu >/dev/null 2>&1; then
  echo "    ncu already on PATH: $(ncu --version | head -1)"
else
  conda install -y -c nvidia nsight-compute || \
    echo "    WARN: 'conda install -c nvidia nsight-compute' failed. Profiling (Phase 3)" \
         "will be unavailable until ncu is installed — see" \
         "https://developer.nvidia.com/tools-overview/nsight-compute/get-started" \
         "for the apt/cuda-toolkit-repo alternative."
  if command -v ncu >/dev/null 2>&1; then
    echo "    installed: $(ncu --version | head -1)"
  fi
fi
# Common cloud-GPU gotcha: GPU performance counters are often locked to admin-only
# (NVreg_RestrictProfilingToAdminUsers). If profiling/prof_roofline.py later fails
# under ncu with "ERR_NVGPUCTRPERM", prefix the ncu command with sudo, e.g.:
#   sudo env "PATH=$PATH" ncu --nvtx ...
# (sudo drops the conda env's PATH by default, hence the explicit "env PATH=$PATH").

echo "==> [4/7] Hugging Face auth"
# The PILOT now uses the Llama pair (Llama-3.1-8B / 3.2-1B), which is GATED — vLLM
# 0.6.6 (pinned for the patch's spec_decode hooks) predates Qwen3, so Qwen3 will
# not load on it. You therefore NEED HF auth (and access granted to the Llama repos)
# before the pilot, not just the full study.
if [ -n "$HF_TOKEN" ]; then
  python -c "from huggingface_hub import login; login('$HF_TOKEN')"
  echo "    logged in via HF_TOKEN."
else
  echo "    WARN: HF_TOKEN not set. The Llama pilot pair is GATED and WILL FAIL to"
  echo "    download without auth. Run:  huggingface-cli login   (and accept the"
  echo "    Llama-3.1-8B-Instruct / Llama-3.2-1B-Instruct licenses on huggingface.co)."
fi

echo "==> [5/7] Project code (expected to be already here, copied via push_to_vm.sh)"
# We run from inside the project folder, so just confirm the key files arrived.
if [ ! -f experiments/run_experiment.py ] || [ ! -f controller/core.py ]; then
  echo "    FATAL: this doesn't look like the specdvfs folder — run setup_vm.sh from"
  echo "    inside the directory you pushed (e.g.  cd ~/specdvfs && bash setup_vm.sh )."
  echo "    If you haven't copied the code yet, run push_to_vm.sh on your laptop first."
  exit 1
fi
echo "    code present: $(pwd)"

echo "==> [6/7] *** CLOCK-LOCK VERIFICATION (GO/NO-GO) ***"
python - <<'PY'
import sys, time
import pynvml as N

N.nvmlInit()
h = N.nvmlDeviceGetHandleByIndex(0)
name = N.nvmlDeviceGetName(h); name = name.decode() if isinstance(name, bytes) else name

mem_clocks = N.nvmlDeviceGetSupportedMemoryClocks(h)
gfx = sorted(N.nvmlDeviceGetSupportedGraphicsClocks(h, mem_clocks[0]))
print(f"    GPU: {name}")
print(f"    supported graphics clocks: {len(gfx)} levels, range {gfx[0]}-{gfx[-1]} MHz")

try:
    target = gfx[len(gfx) // 3]                 # a low-ish level for the test
    N.nvmlDeviceSetGpuLockedClocks(h, target, target)
    time.sleep(1.0)
    cur = N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_GRAPHICS)
    N.nvmlDeviceResetGpuLockedClocks(h)          # ALWAYS reset
    print(f"    locked to {target} MHz, read back {cur} MHz  ->  CLOCK LOCKING WORKS")
except N.NVMLError as e:
    print(f"    CLOCK LOCKING FAILED: {e}")
    print("    This VM cannot run the DVFS experiment (Docker container or non-lockable GPU).")
    print("    Get a non-Docker VM with a lockable GPU.")
    sys.exit(2)

N.nvmlShutdown()
PY

echo "==> [7/7] Datasets + integrity + MD5"
python data/prepare_datasets.py
# The verifier prints MD5s — RECORD THESE in your lab notebook. Tolerate either
# filename (the repo may name it verify_dataset.py or verify_datasets.py).
if [ -f data/verify_datasets.py ]; then
  python data/verify_datasets.py
elif [ -f data/verify_dataset.py ]; then
  python data/verify_dataset.py
else
  echo "    WARN: no data/verify_dataset(s).py found — skipping integrity/MD5 check."
fi

echo
echo "============================================================"
echo "  SETUP COMPLETE."
echo "  - Record the MD5 checksums and the clock range printed above."
echo "  - Project uses f_high=1935, f_low=735 for the RTX 3090"
echo "    (a chosen low level + the sustainable max — NOT the absolute min/max)."
echo "  - Pilot pair is GATED Llama-3.1-8B / 3.2-1B: make sure HF auth succeeded above."
echo "  - ncu (Nsight Compute) was installed for Phase 3 roofline profiling — if it's"
echo "    missing, see step [3]'s WARN above; if it errs with ERR_NVGPUCTRPERM at"
echo "    profiling time, re-run that ncu command with sudo (see step [3]'s note)."
echo "  - Next:  python experiments/run_experiment.py --mode pilot"
echo "============================================================"
