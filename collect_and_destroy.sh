#!/usr/bin/env bash
# =============================================================================
# collect_and_destroy.sh  —  run on your LOCAL machine AFTER the experiment ends.
#
#   1) rsync the results down from the VM,
#   2) verify they actually arrived,
#   3) ONLY THEN destroy the Vast.ai instance (with a typed confirmation).
#
# WHY THIS IS A SEPARATE LOCAL SCRIPT (not auto-run by the orchestrator):
#   The orchestrator runs ON the VM. If it destroyed the instance itself, it
#   would be deleting the only copy of the data — and the machine it is running
#   on — before the copy to your laptop is verified. The safe order is always
#   "pull, verify locally, then destroy", and that has to be driven from the
#   machine that survives (your laptop), where the Vast CLI + API key live.
# =============================================================================

set -euo pipefail

# ------------------------------- EDIT THESE ----------------------------------
# From the Vast.ai "Instances" page (the SSH command it shows you):
VM_HOST="root@1.2.3.4"                       # <-- user@host, e.g. root@84.12.55.9
VM_PORT="12345"                              # <-- SSH port, e.g. 41022
VM_RESULTS="/root/specdvfs/results/"         # <-- remote results dir (trailing slash matters)
LOCAL_DEST="./results_from_vm/"              # <-- where to put them locally
VAST_INSTANCE_ID="123456"                    # <-- from `vastai show instances`
# -----------------------------------------------------------------------------

mkdir -p "$LOCAL_DEST"

echo "==> [1/3] Pulling results from ${VM_HOST}:${VM_PORT} ..."
rsync -avz --progress -e "ssh -p ${VM_PORT}" "${VM_HOST}:${VM_RESULTS}" "${LOCAL_DEST}"

echo "==> [2/3] Verifying the copy ..."
N=$(find "$LOCAL_DEST" -name '*.json' | wc -l | tr -d ' ')
echo "    ${N} result JSON files now local in ${LOCAL_DEST}"
if [ "$N" -eq 0 ]; then
  echo "    FATAL: nothing copied — NOT destroying the instance. Check VM_HOST/VM_PORT/VM_RESULTS."
  exit 1
fi

echo "==> [3/3] Destroy Vast instance ${VAST_INSTANCE_ID}?"
echo "    This is irreversible. Make sure the ${N} files above are what you expect."
read -r -p "    Type 'destroy' to confirm: " ans
if [ "$ans" != "destroy" ]; then
  echo "    Aborted. Instance left running. Local copy is in ${LOCAL_DEST}."
  exit 0
fi

if ! command -v vastai >/dev/null 2>&1; then
  echo "    Vast CLI not found. Install + authenticate first:"
  echo "        pip install vastai"
  echo "        vastai set api-key <YOUR_VAST_API_KEY>"
  echo "    Then re-run, or destroy from the Vast.ai web UI."
  exit 1
fi

vastai destroy instance "$VAST_INSTANCE_ID"
echo "    Instance ${VAST_INSTANCE_ID} destroyed. Local copy: ${LOCAL_DEST}"
