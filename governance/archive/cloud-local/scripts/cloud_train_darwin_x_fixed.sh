#!/bin/bash
# F51 Darwin-X Cloud Training - VERSÃO CORRIGIDA
# Usa arquivos que REALMENTE existem: tokens_chunk_aa, f51_darwin, configs

set -e

HUB_HOST="ssh6.vast.ai"
HUB_PORT="19862"
HUB_KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIN+nGjJq2xJyNL37XylBXbTo5GTuV7L7eEibCcKRJHwO"

echo "⏰ $(date): Aguardando 10 min..."
sleep 600

echo "🔍 $(date): Buscando melhor GPU..."
# Try A100 first, then 4090, then H200
for GPU in "A100" "4090" "H200"; do
  OFFER=$(vastai search offers "rentable=true verified=true num_gpus=1 gpu_name=~$GPU gpu_ram>=40000 reliability>=0.95" --order price --limit 1 --raw 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['id'])")
  [ -n "$OFFER" ] && break
done

if [ -z "$OFFER" ]; then
  echo "❌ Nenhuma GPU encontrada!"; exit 1
fi

echo "💰 Alugando GPU $OFFER..."
vastai create instance "$OFFER" --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel --disk 200 --ssh --direct

echo "⏳ Aguardando boot..."
sleep 120

# Get GPU SSH info
INFO=$(vastai show instance "$OFFER" --raw 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"ssh_host\"]}:{d[\"ssh_port\"]}')")
GPU_HOST=$(echo $INFO | cut -d: -f1)
GPU_PORT=$(echo $INFO | cut -d: -f2)

echo "🔑 Configurando SSH hub→GPU..."
ssh -o StrictHostKeyChecking=no -p $GPU_PORT root@$GPU_HOST "mkdir -p /root/.ssh && echo '$HUB_KEY' >> /root/.ssh/authorized_keys"

echo "📦 Transferindo tokens (11GB)..."
# CORRIGIDO: usa tokens_chunk_aa que existe
ssh -o StrictHostKeyChecking=no -p $HUB_PORT root@$HUB_HOST "scp -o StrictHostKeyChecking=no -P $GPU_PORT /workspace/tokens_chunk_aa root@$GPU_HOST:/workspace/tokens_darwin_x.bin"

echo "📦 Transferindo código..."
ssh -o StrictHostKeyChecking=no -p $HUB_PORT root@$HUB_HOST "scp -o StrictHostKeyChecking=no -r -P $GPU_PORT /workspace/f51_darwin root@$GPU_HOST:/workspace/"
ssh -o StrictHostKeyChecking=no -p $HUB_PORT root@$HUB_HOST "scp -o StrictHostKeyChecking=no -r -P $GPU_PORT /workspace/tokenizer root@$GPU_HOST:/workspace/"
ssh -o StrictHostKeyChecking=no -p $HUB_PORT root@$HUB_HOST "scp -o StrictHostKeyChecking=no -r -P $GPU_PORT /workspace/configs root@$GPU_HOST:/workspace/"

echo "🚀 Lançando treino Darwin-X 4B do zero..."
ssh -o StrictHostKeyChecking=no -p $GPU_PORT root@$GPU_HOST "
cd /workspace && PYTHONPATH=/workspace PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 -u train_darwin_x.py --config configs/darwin_x_4b.yaml \
  --token-bin tokens_darwin_x.bin \
  --tokenizer tokenizer/f51_bpe \
  --steps 10000 \
  --batch-size 4 \
  --block-size 512 \
  --save-every 500 \
  --device cuda > train.log 2>&1 &
echo PID=\$!
"

echo "✅ $(date): TREINO LANÇADO!"
echo "Monitor: ssh -p $GPU_PORT root@$GPU_HOST 'tail -f /workspace/train.log'"
