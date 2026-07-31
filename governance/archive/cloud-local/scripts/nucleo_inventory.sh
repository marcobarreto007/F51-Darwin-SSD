#!/usr/bin/env bash
set -Eeuo pipefail

HUB_USER="${HUB_USER:-root}"
HUB_HOST="${HUB_HOST:-70.30.158.46}"
HUB_PORT="${HUB_PORT:-56013}"
if [[ -z "${SSH_BIN:-}" && -x /mnt/c/Windows/System32/OpenSSH/ssh.exe ]]; then
  SSH_BIN="/mnt/c/Windows/System32/OpenSSH/ssh.exe"
fi
SSH_BIN="${SSH_BIN:-ssh}"

remote="${HUB_USER}@${HUB_HOST}"

"${SSH_BIN}" -o StrictHostKeyChecking=no -p "${HUB_PORT}" "${remote}" "bash -se" <<'EOF'
set -Eeuo pipefail

echo "== NUCLEO inventory =="
date -u +"utc=%Y-%m-%dT%H:%M:%SZ"
echo

echo "== disk =="
df -h /
echo

echo "== tokenization =="
if [ -f /workspace/tokens/tokenize_v2.log ]; then
  tr '\r' '\n' < /workspace/tokens/tokenize_v2.log | grep -E '[0-9]+/[0-9]+ files|DONE|done' | tail -5
else
  echo "missing /workspace/tokens/tokenize_v2.log"
fi
echo

echo "== core artifacts =="
for path in \
  /workspace/tokens/tokens_full.bin \
  /workspace/tokens/tokens_v1_backup.bin \
  /workspace/tokenizer/f51_bpe \
  /workspace/checkpoints/step_0007900_moe.pt \
  /workspace/f51_darwin \
  /workspace/unified_train.py \
  /workspace/corpus_merged
do
  if [ -e "$path" ]; then
    du -sh "$path" 2>/dev/null || ls -lh "$path"
  else
    echo "MISSING $path"
  fi
done
echo

echo "== cleanup candidates, no delete =="
for path in /workspace/qwen_math_1.5b.pt /workspace/qwen_math.pt /workspace/darwin_x_vampire.pt; do
  if [ -e "$path" ]; then
    du -sh "$path"
  else
    echo "absent $path"
  fi
done
echo

echo "== matching processes =="
pgrep -af 'tokenize_v2|scp|rsync|unified_train|train_darwin_x' || true
EOF
