#!/usr/bin/env bash
# =============================================================================
# collect_and_destroy.sh — run this ON YOUR LAPTOP after the VM work is done.
#
#   1) pull the results / profiling / calibration outputs down from the VM,
#   2) verify something actually arrived,
#   3) OPTIONALLY destroy the Vast.ai instance (only if you set VAST_INSTANCE_ID
#      AND type 'destroy' to confirm).
#
# Like push_to_vm.sh this uses `tar | ssh`, not rsync, so it needs nothing on the
# VM beyond ssh + tar. Safe to run as many times as you like — it only copies.
# Leave VAST_INSTANCE_ID blank to use it purely as a "bring my results home"
# button without any risk of tearing the instance down.
#
# WHY IT RUNS ON THE LAPTOP: the orchestrator runs ON the VM; a script can't pull
# data to, or safely destroy, the machine it is itself running on. The safe order
# is "pull, verify locally, then (maybe) destroy", driven from the laptop.
# =============================================================================

set -uo pipefail

# ----- EDIT THESE (same VM as push_to_vm.sh) -----
VM_USER="root"                 # USER in  ssh -p PORT USER@HOST
VM_HOST="1.2.3.4"              # HOST/IP
VM_PORT="12345"                # PORT (-p)
REMOTE_DIR="specdvfs"          # the folder you pushed to (matches push_to_vm.sh)
LOCAL_DEST="./results_from_vm" # where to drop everything on your laptop
VAST_INSTANCE_ID=""            # OPTIONAL: from `vastai show instances`. BLANK = never destroy.
# -------------------------------------------------

mkdir -p "$LOCAL_DEST"

echo "==> [1/3] checking SSH to ${VM_USER}@${VM_HOST} port ${VM_PORT} ..."
ssh -p "$VM_PORT" "${VM_USER}@${VM_HOST}" 'echo "    connected to $(hostname)"' || {
  echo "    FATAL: cannot SSH in. Re-check VM_USER / VM_HOST / VM_PORT."
  exit 1
}

echo "==> [2/3] pulling results + profiling/out + calibration -> ${LOCAL_DEST}/ ..."
# tar whatever of these dirs exists on the VM (missing ones are skipped, not fatal),
# stream it back, and unpack locally. The '|| true' keeps a missing dir from
# aborting the script; the file-count check below is the real gate.
ssh -p "$VM_PORT" "${VM_USER}@${VM_HOST}" \
    "cd ${REMOTE_DIR} 2>/dev/null && tar czf - results profiling/out calibration 2>/dev/null" \
  | tar xzf - -C "$LOCAL_DEST" 2>/dev/null || true

N=$(find "$LOCAL_DEST" -name '*.json' | wc -l | tr -d ' ')
echo "    ${N} JSON file(s) now under ${LOCAL_DEST}/"
if [ "$N" -eq 0 ]; then
  echo "    FATAL: nothing came back. Did the pilot run? Is REMOTE_DIR correct (${REMOTE_DIR})?"
  exit 1
fi
echo "    pulled into:"
find "$LOCAL_DEST" -maxdepth 2 -type d | sed 's/^/        /'

if [ -z "$VAST_INSTANCE_ID" ]; then
  echo "==> [3/3] VAST_INSTANCE_ID is blank -> leaving the instance running (pull-only)."
  echo "    Your results are safe in ${LOCAL_DEST}/. To tear the VM down later, set"
  echo "    VAST_INSTANCE_ID (from 'vastai show instances') and re-run."
  exit 0
fi

echo "==> [3/3] Destroy Vast instance ${VAST_INSTANCE_ID}? This is IRREVERSIBLE."
echo "    The ${N} file(s) above are already on your laptop."
read -r -p "    Type 'destroy' to confirm (anything else aborts): " ans
if [ "$ans" != "destroy" ]; then
  echo "    Aborted. Instance left running. Local copy is in ${LOCAL_DEST}/."
  exit 0
fi
if ! command -v vastai >/dev/null 2>&1; then
  echo "    Vast CLI not found. Install + authenticate, then re-run (or destroy in the web UI):"
  echo "        pip install vastai && vastai set api-key <YOUR_VAST_API_KEY>"
  exit 1
fi
vastai destroy instance "$VAST_INSTANCE_ID"
echo "    Instance ${VAST_INSTANCE_ID} destroyed. Local copy: ${LOCAL_DEST}/."
