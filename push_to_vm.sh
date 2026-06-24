#!/usr/bin/env bash
# =============================================================================
# push_to_vm.sh — run this ON YOUR LAPTOP to copy the project onto the VM.
#
# WHY THIS EXISTS: the VM starts empty and we're not using GitHub, so the code
# has to be pushed from the machine that has it (your laptop) over the SSH access
# you already have. It uses `tar | ssh` instead of rsync on purpose — rsync must
# be installed on BOTH ends, and a fresh VM image usually doesn't have it yet
# (setup hasn't run). tar + ssh are present on essentially every machine, so this
# works on a bare VM. Re-run it any time you change the code locally; it just
# overwrites the files on the VM.
#
# PLACEMENT: keep this script INSIDE your specdvfs folder (next to setup_vm.sh).
# =============================================================================

set -euo pipefail

# ----- EDIT THESE (read them off the Vast.ai SSH command: ssh -p PORT USER@HOST) -----
VM_USER="root"            # the USER in  ssh -p PORT USER@HOST
VM_HOST="1.2.3.4"         # the HOST/IP
VM_PORT="12345"           # the PORT  (-p)
REMOTE_DIR="specdvfs"     # folder to create in the VM's home dir; leave as-is
# -------------------------------------------------------------------------------------

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"   # this repo = the folder this script lives in

echo "==> [1/2] checking SSH to ${VM_USER}@${VM_HOST} port ${VM_PORT} ..."
if ! ssh -p "$VM_PORT" "${VM_USER}@${VM_HOST}" 'echo "    connected to $(hostname)"'; then
  # If the host is already in known_hosts, the TCP connection + host-key step
  # succeeded earlier (this run or the last) — so host/port are fine, and ANY
  # failure here is purely an auth/key problem, not a wrong-address problem.
  if ssh-keygen -F "[${VM_HOST}]:${VM_PORT}" -f ~/.ssh/known_hosts >/dev/null 2>&1; then
    echo "    FATAL: reached the host fine (host/port are correct) but your key was"
    echo "    rejected — 'Permission denied (publickey)'. Vast.ai instances accept SSH"
    echo "    keys only, and adding a key to your ACCOUNT only affects instances created"
    echo "    AFTER that — it does NOT retroactively reach this already-running one."
    echo "    Fix (unambiguous, confirmed via Vast.ai's own CLI docs):"
    echo "        pip install vastai"
    echo "        vastai set api-key <YOUR_VAST_API_KEY>      # from https://cloud.vast.ai/cli/"
    echo "        vastai show instances                        # find this instance's ID"
    echo "        vastai attach ssh <instance_id> \"\$(cat ~/.ssh/id_ed25519.pub)\""
    echo "    No local key yet? generate one first:  ssh-keygen -t ed25519"
    echo "    Still rejected after attach? it may need the instance restarted to pick up"
    echo "    the new authorized_keys — restart it from the Vast.ai console, then retry."
    echo "    No SSH-key field on that instance? Open its Jupyter Terminal (web-based,"
    echo "    no key needed) instead and run there:"
    echo "        echo \"\$(cat ~/.ssh/id_ed25519.pub)\" >> ~/.ssh/authorized_keys"
    echo "    More detail (shows exactly which key was offered/rejected):"
    echo "        ssh -vv -p ${VM_PORT} ${VM_USER}@${VM_HOST}"
  else
    echo "    FATAL: cannot reach the host at all. Re-check VM_USER / VM_HOST / VM_PORT"
    echo "    against the exact 'ssh -p PORT USER@HOST' line on the Vast.ai Instances page."
  fi
  exit 1
fi

echo "==> [2/2] copying code from ${LOCAL_DIR} -> ${VM_USER}@${VM_HOST}:~/${REMOTE_DIR} ..."
# Pack the repo (minus junk + generated artifacts) and unpack it on the VM in one pipe.
tar czf - -C "$LOCAL_DIR" \
    --exclude='.git' \
    --exclude='.pytest_cache' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='results' \
    --exclude='calibration/*.json' \
    --exclude='calibration/*.png' \
    --exclude='profiling/out' \
    --exclude='data/sampled_indices' \
    --exclude='results_from_vm' \
    . \
  | ssh -p "$VM_PORT" "${VM_USER}@${VM_HOST}" "mkdir -p ${REMOTE_DIR} && tar xzf - -C ${REMOTE_DIR}"

echo
echo "    Done. Code is in ~/${REMOTE_DIR} on the VM."
echo "    Next, SSH in and run setup:"
echo "        ssh -p ${VM_PORT} ${VM_USER}@${VM_HOST}"
echo "        cd ${REMOTE_DIR}"
echo "        export HF_TOKEN=hf_xxx        # paste your token here, in the live shell"
echo "        bash setup_vm.sh"
