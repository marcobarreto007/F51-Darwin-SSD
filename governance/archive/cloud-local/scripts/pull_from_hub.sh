#!/usr/bin/env bash
set -Eeuo pipefail

HUB_USER="${HUB_USER:-root}"
HUB_HOST="${HUB_HOST:-70.30.158.46}"
HUB_PORT="${HUB_PORT:-56013}"
WORKSPACE="${WORKSPACE:-/workspace}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-step_0007900_moe.pt}"
RUN_TRAIN="${RUN_TRAIN:-0}"
SKIP_PIP="${SKIP_PIP:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE:-0}"
MIN_CHECKPOINT_BYTES="${MIN_CHECKPOINT_BYTES:-21185819745}"
SSH_KEY="${SSH_KEY:-}"
if [[ -z "${SSH_BIN:-}" && -x /mnt/c/Windows/System32/OpenSSH/ssh.exe ]]; then
  SSH_BIN="/mnt/c/Windows/System32/OpenSSH/ssh.exe"
fi
if [[ -z "${SCP_BIN:-}" && -x /mnt/c/Windows/System32/OpenSSH/scp.exe ]]; then
  SCP_BIN="/mnt/c/Windows/System32/OpenSSH/scp.exe"
fi
SSH_BIN="${SSH_BIN:-ssh}"
SCP_BIN="${SCP_BIN:-scp}"

remote="${HUB_USER}@${HUB_HOST}"
ssh_base=("${SSH_BIN}" -o StrictHostKeyChecking=no)
scp_base=("${SCP_BIN}" -o StrictHostKeyChecking=no)
if [[ -n "${SSH_KEY}" ]]; then
  ssh_base+=(-i "${SSH_KEY}" -o IdentitiesOnly=yes)
  scp_base+=(-i "${SSH_KEY}" -o IdentitiesOnly=yes)
fi
ssh_base+=(-p "${HUB_PORT}" "${remote}")
scp_base+=(-P "${HUB_PORT}")

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

copy_from_hub() {
  local src="$1"
  local dest="$2"
  echo "[pull] ${src} -> ${dest}"
  "${scp_base[@]}" -r "${remote}:${src}" "${dest}"
}

need_cmd "${SSH_BIN}"
need_cmd "${SCP_BIN}"
need_cmd python3

echo "[probe] NUCLEO ${remote}:${HUB_PORT}"
remote_probe_script=$(cat <<EOF
set -Eeuo pipefail
require_file() {
  if [ ! -s "\$1" ]; then
    echo "missing required file: \$1" >&2
    exit 2
  fi
}
require_dir() {
  if [ ! -d "\$1" ]; then
    echo "missing required dir: \$1" >&2
    exit 2
  fi
}
require_file /workspace/tokens/tokens_full.bin
require_file /workspace/checkpoints/${CHECKPOINT_NAME}
require_dir /workspace/tokenizer/f51_bpe
require_dir /workspace/f51_darwin
require_file /workspace/unified_train.py
ckpt_bytes=\$(stat -c%s /workspace/checkpoints/${CHECKPOINT_NAME})
if [ "${ALLOW_INCOMPLETE}" != "1" ] && [ "\${ckpt_bytes}" -lt "${MIN_CHECKPOINT_BYTES}" ]; then
  echo "checkpoint incomplete: \${ckpt_bytes} < ${MIN_CHECKPOINT_BYTES}" >&2
  exit 2
fi
active_tokenizers="\$( (ps -eo pid,ppid,cmd | grep '[t]okenize_v2.py' | grep -v 'Wait for tokenization' || true) | wc -l )"
if [ "${ALLOW_INCOMPLETE}" != "1" ] && [ "\${active_tokenizers}" -gt 0 ]; then
  echo "tokenization still running; set ALLOW_INCOMPLETE=1 only for diagnostics" >&2
  exit 2
fi
EOF
)
local_probe="$(mktemp)"
remote_probe="/tmp/nucleo_pull_probe_${RANDOM}_$$.sh"
printf '%s\n' "${remote_probe_script}" > "${local_probe}"
"${scp_base[@]}" "${local_probe}" "${remote}:${remote_probe}" >/dev/null
set +e
"${ssh_base[@]}" "bash ${remote_probe}"
probe_rc=$?
"${ssh_base[@]}" "rm -f ${remote_probe}" >/dev/null 2>&1
rm -f "${local_probe}"
set -e
if [[ "${probe_rc}" -ne 0 ]]; then
  exit "${probe_rc}"
fi

if [[ "${CHECK_ONLY}" == "1" ]]; then
  if [[ "${ALLOW_INCOMPLETE}" == "1" ]]; then
    echo "[check-only] NUCLEO artifacts exist, but incomplete state was allowed."
  else
    echo "[check-only] NUCLEO artifacts are ready to pull."
  fi
  exit 0
fi

mkdir -p "${WORKSPACE}/tokens" "${WORKSPACE}/tokenizer" "${WORKSPACE}/checkpoints" "${WORKSPACE}/data"

copy_from_hub "/workspace/tokens/tokens_full.bin" "${WORKSPACE}/tokens/"
copy_from_hub "/workspace/tokenizer/f51_bpe" "${WORKSPACE}/tokenizer/"
copy_from_hub "/workspace/checkpoints/${CHECKPOINT_NAME}" "${WORKSPACE}/checkpoints/"
copy_from_hub "/workspace/f51_darwin" "${WORKSPACE}/"
copy_from_hub "/workspace/unified_train.py" "${WORKSPACE}/"

ln -sfn ../tokens/tokens_full.bin "${WORKSPACE}/data/tokens_full.bin"

if [[ "${SKIP_PIP}" != "1" ]]; then
  python3 -m pip install torch pyyaml tqdm psutil numpy
fi

cd "${WORKSPACE}"
python3 - <<'PY'
from pathlib import Path
import importlib

required = [
    Path("tokens/tokens_full.bin"),
    Path("data/tokens_full.bin"),
    Path("tokenizer/f51_bpe"),
    Path("checkpoints/step_0007900_moe.pt"),
    Path("f51_darwin"),
    Path("unified_train.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing after pull: " + ", ".join(missing))
importlib.import_module("f51_darwin")
print("OK: NUCLEO pull verified")
PY

train_cmd=(python3 unified_train.py --resume "checkpoints/${CHECKPOINT_NAME}" --corpus data/corpus --device cuda --precision bf16 --optimizer adafactor)

if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo "[train] ${train_cmd[*]}"
  exec "${train_cmd[@]}"
fi

echo "[ready] To start training:"
printf '  %q' "${train_cmd[@]}"
echo
