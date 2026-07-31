#!/bin/bash
# ============================================
# pull_cloud_checkpoint.sh
# Baixa o checkpoint mais recente da nuvem.
# NAO MEXE em nada rodando. So baixa.
# ============================================
# Uso: bash pull_cloud_checkpoint.sh
# ============================================
set -e

# ── Config (ajuste se precisar) ──
CLOUD_HOST="root@192.234.50.251"
CLOUD_PORT="1585"
CLOUD_CHECKPOINT_DIR="/workspace/F51-Darwin-SSD/checkpoints/cloud"
LOCAL_DIR="checkpoints/cloud"

# ── Cria dir local ──
mkdir -p "$LOCAL_DIR"

# ── Pega o latest da nuvem ──
echo "Buscando checkpoint mais recente na nuvem..."
LATEST_JSON=$(ssh -o StrictHostKeyChecking=no -p "$CLOUD_PORT" "$CLOUD_HOST" \
  "cat $CLOUD_CHECKPOINT_DIR/cloud_latest.json 2>/dev/null" 2>/dev/null)

if [ -z "$LATEST_JSON" ]; then
  echo "ERRO: Nao foi possivel ler cloud_latest.json"
  echo "A instancia esta no ar?"
  exit 1
fi

CLOUD_PATH=$(echo "$LATEST_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['path'])")
CLOUD_STEP=$(echo "$LATEST_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['step'])")
CLOUD_LOSS=$(echo "$LATEST_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('loss','?'))")

echo "Nuvem: step=$CLOUD_STEP loss=$CLOUD_LOSS"
echo "Arquivo: $CLOUD_PATH"

# ── Verifica se ja temos esse checkpoint localmente ──
LOCAL_FILE="$LOCAL_DIR/cloud_step_$(printf '%07d' $CLOUD_STEP).pt"
if [ -f "$LOCAL_FILE" ]; then
  echo "Este checkpoint ja existe localmente: $LOCAL_FILE"
  echo "Nada a fazer."
  exit 0
fi

# ── Baixa ──
echo "Baixando..."
scp -o StrictHostKeyChecking=no -P "$CLOUD_PORT" "$CLOUD_HOST:$CLOUD_PATH" "$LOCAL_FILE"

if [ -f "$LOCAL_FILE" ]; then
  SIZE=$(ls -lh "$LOCAL_FILE" | awk '{print $5}')
  echo ""
  echo "OK! Checkpoint baixado:"
  echo "  Arquivo: $LOCAL_FILE"
  echo "  Tamanho: $SIZE"
  echo "  Step:    $CLOUD_STEP"
  echo "  Loss:    $CLOUD_LOSS"
  echo ""
  echo "Para usar no servidor, aponte serve_f51.py para este arquivo:"
  echo "  python scripts/serve_f51.py --checkpoint $LOCAL_FILE"
else
  echo "ERRO: Download falhou."
  exit 1
fi
