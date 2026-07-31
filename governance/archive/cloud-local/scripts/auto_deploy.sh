#!/bin/bash
# F51 Auto Deploy — Comprime, upload e inicia treino
set -e

GPU_HOST="ssh1.vast.ai"
GPU_PORT="33348"
PROJECT_NAME="F51-Darwin-SSD"
PROJECT_ROOT="$(pwd)"
CORPUS_DIR="data/corpus"
ARCHIVE="/tmp/corpus.tar.gz"

echo "================================"
echo "F51 AUTO DEPLOY GPU"
echo "================================"
echo ""

# 1. Verificar conexao
echo "[1/6] Testando conexao GPU..."
ssh -p $GPU_PORT -o StrictHostKeyChecking=no root@$GPU_HOST "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
echo "GPU conectada!"
echo ""

# 2. Verificar saldo
echo "[2/6] Verificando saldo Vast.ai..."
if command -v vastai &> /dev/null; then
    vastai show user | grep -i credit
fi
echo ""

# 3. Deploy projeto
echo "[3/6] Deploy projeto F51..."
PROJECT_EXISTS=$(ssh -p $GPU_PORT -o StrictHostKeyChecking=no root@$GPU_HOST "test -d /workspace/$PROJECT_NAME && echo OK || echo NOK")
if [ "$PROJECT_EXISTS" = "OK" ]; then
    echo "Projeto ja existe na GPU"
else
    echo "Empacotando projeto (sem data/)..."
    tar -cf /tmp/f51_project.tar --exclude='data' --exclude='.git' --exclude='checkpoints' --exclude='runs' f51_darwin scripts tokenizer configs
    echo "Enviando para GPU..."
    scp -P $GPU_PORT -o StrictHostKeyChecking=no /tmp/f51_project.tar root@$GPU_HOST:/workspace/
    ssh -p $GPU_PORT -o StrictHostKeyChecking=no root@$GPU_HOST "cd /workspace && tar -xf f51_project.tar && rm f51_project.tar"
    rm /tmp/f51_project.tar
    echo "Projeto deployado!"
fi
echo ""

# 4. Comprimir corpus
echo "[4/6] Comprimindo corpus..."
if [ -f "$ARCHIVE" ]; then
    echo "Arquivo ja existe: $ARCHIVE"
    SIZE=$(du -h "$ARCHIVE" | cut -f1)
    echo "Tamanho: $SIZE"
else
    echo "Iniciando compressao (68GB -> ~10-15GB)..."
    echo "Tempo estimado: 20-40 minutos"
    tar -czf "$ARCHIVE" "$CORPUS_DIR"
    FINAL_SIZE=$(du -h "$ARCHIVE" | cut -f1)
    echo "Compressao concluida! Tamanho: $FINAL_SIZE"
fi
echo ""

# 5. Upload corpus
echo "[5/6] Enviando corpus para GPU..."
echo "Tempo estimado: 30-60 minutos"
scp -P $GPU_PORT -o StrictHostKeyChecking=no "$ARCHIVE" root@$GPU_HOST:/workspace/
echo "Upload concluido!"
echo ""

# 6. Descomprimir na GPU
echo "[6/6] Descomprimindo corpus na GPU..."
ssh -p $GPU_PORT -o StrictHostKeyChecking=no root@$GPU_HOST "cd /workspace/$PROJECT_NAME && tar -xzf /workspace/corpus.tar.gz && ls data/corpus | wc -l"
echo "Corpus pronto!"
echo ""

# 7. Iniciar treino
echo "================================"
echo "INICIANDO TREINAMENTO"
echo "================================"
TRAIN_CMD="cd /workspace/$PROJECT_NAME && nohup python scripts/train_cloud.py --device cuda --batch-size 16 --block-size 512 --steps 100000 --save-every 1000 --eval-every 500 --checkpoint-dir checkpoints/cloud --corpus data/corpus --tokenizer tokenizer/f51_bpe > /workspace/train.log 2>&1 &"
ssh -p $GPU_PORT -o StrictHostKeyChecking=no root@$GPU_HOST "$TRAIN_CMD"
sleep 3
PID=$(ssh -p $GPU_PORT -o StrictHostKeyChecking=no root@$GPU_HOST "pgrep -f train_cloud.py | head -1")
echo "Treino iniciado! PID: $PID"
echo ""

echo "================================"
echo "DEPLOY CONCLUIDO!"
echo "================================"
echo ""
echo "Comandos uteis:"
echo "  Ver log:   ssh -p $GPU_PORT root@$GPU_HOST 'tail -f /workspace/train.log'"
echo "  Ver GPU:   ssh -p $GPU_PORT root@$GPU_HOST 'nvidia-smi'"
echo "  Ver checkpoints: ssh -p $GPU_PORT root@$GPU_HOST 'ls -lh /workspace/$PROJECT_NAME/checkpoints/cloud/'"
echo ""
