#!/usr/bin/env bash
set -Eeuo pipefail

HUB_USER="${HUB_USER:-root}"
HUB_HOST="${HUB_HOST:-70.30.158.46}"
HUB_PORT="${HUB_PORT:-56013}"
APPLY="${APPLY:-0}"
if [[ -z "${SSH_BIN:-}" && -x /mnt/c/Windows/System32/OpenSSH/ssh.exe ]]; then
  SSH_BIN="/mnt/c/Windows/System32/OpenSSH/ssh.exe"
fi
if [[ -z "${SCP_BIN:-}" && -x /mnt/c/Windows/System32/OpenSSH/scp.exe ]]; then
  SCP_BIN="/mnt/c/Windows/System32/OpenSSH/scp.exe"
fi
SSH_BIN="${SSH_BIN:-ssh}"
SCP_BIN="${SCP_BIN:-scp}"

remote="${HUB_USER}@${HUB_HOST}"
stamp="$(date +%Y%m%d_%H%M%S)"
remote_archive="/tmp/f51_code_${stamp}.tar.gz"
remote_stage="/workspace/f51_code_stage_${stamp}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
mkdir -p "${repo_root}/tmp"
archive="${repo_root}/tmp/f51_code_${stamp}.tar.gz"
scp_archive="${archive}"
if [[ "${SCP_BIN}" == /mnt/c/* ]] && command -v wslpath >/dev/null 2>&1; then
  scp_archive="$(wslpath -w "${archive}")"
fi

echo "[plan] Sync local code to NUCLEO without data/checkpoints/runs."
echo "[plan] Remote: ${remote}:${HUB_PORT}"
echo "[plan] APPLY=${APPLY}"

if [[ "${APPLY}" != "1" ]]; then
  echo "[dry-run] Would package: f51_darwin configs scripts unified_train.py requirements*.txt pyproject.toml"
  echo "[dry-run] Would validate remote import before copying into /workspace."
  echo "[dry-run] Run with APPLY=1 to execute."
  exit 0
fi

tar -czf "${archive}" \
  --exclude='scripts/__pycache__' \
  --exclude='**/__pycache__' \
  f51_darwin configs scripts unified_train.py requirements.txt requirements_cloud.txt pyproject.toml

"${SCP_BIN}" -o StrictHostKeyChecking=no -P "${HUB_PORT}" "${scp_archive}" "${remote}:${remote_archive}"

"${SSH_BIN}" -o StrictHostKeyChecking=no -p "${HUB_PORT}" "${remote}" "bash -se" <<EOF
set -Eeuo pipefail
rm -rf "${remote_stage}"
mkdir -p "${remote_stage}"
tar -xzf "${remote_archive}" -C "${remote_stage}"
cd "${remote_stage}"
python3 - <<'PY'
import importlib
importlib.import_module("f51_darwin")
print("OK: staged f51_darwin import")
PY
mkdir -p /workspace/f51_code_backups
for item in f51_darwin configs scripts unified_train.py requirements.txt requirements_cloud.txt; do
  if [ -e "/workspace/\${item}" ]; then
    rm -rf "/workspace/f51_code_backups/\${item}_${stamp}"
    cp -a "/workspace/\${item}" "/workspace/f51_code_backups/\${item}_${stamp}"
  fi
done
rm -rf /workspace/f51_darwin
cp -a "${remote_stage}/f51_darwin" /workspace/f51_darwin
rm -rf /workspace/configs /workspace/scripts
cp -a "${remote_stage}/configs" /workspace/configs
cp -a "${remote_stage}/scripts" /workspace/scripts
cp -a "${remote_stage}/unified_train.py" /workspace/unified_train.py
cp -a "${remote_stage}/requirements.txt" /workspace/requirements.txt
cp -a "${remote_stage}/requirements_cloud.txt" /workspace/requirements_cloud.txt
cd /workspace
python3 - <<'PY'
import importlib
importlib.import_module("f51_darwin")
print("OK: live /workspace import")
PY
EOF

rm -f "${archive}"
echo "[done] Code synced to NUCLEO."
