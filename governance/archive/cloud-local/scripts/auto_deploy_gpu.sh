#!/bin/bash
# F51 Darwin-SSD — Auto Deploy GPU
# Comprime, upload, descomprime e inicia treino automaticamente
# Uso: bash scripts/auto_deploy_gpu.sh

set -e

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
GPU_HOST="${GPU_HOST:-ssh1.vast.ai}"
GPU_PORT="${GPU_PORT:-33348}"
CORPUS_PATH="${CORPUS_PATH:-data/corpus}"
PROJECT_NAME="${PROJECT_NAME:-F51-Darwin-SSD}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
SKIP_COMPRESS="${SKIP_COMPRESS:-false}"
SKIP_UPLOAD="${SKIP_UPLOAD:-false}"

ARCHIVE_FILE="/tmp/corpus_backup.tar.gz"

# ═══════════════════════════════════════════════════════════════
# FUNCOES
# ═══════════════════════════════════════════════════════════════

log() {
    local color="$1"
    shift
    local msg="[$(date '+%H:%M:%S')] $*"
    case "$color" in
        red)    echo -e "\033[91m$msg\033[0m" ;;
        green)  echo -e "\033[92m$msg\033[0m" ;;
        yellow) echo -e "\033[93m$msg\033[0m" ;;
        cyan)   echo -e "\033[96m$msg\033[0m" ;;
        gray)   echo -e "\033[90m$msg\033[0m" ;;
        *)      echo "$msg" ;;
    esac
}

ssh_exec() {
    ssh -p "$GPU_PORT" -o StrictHostKeyChecking=no "root@$GPU_HOST" "$1" 2>&1 | grep -v "^Welcome" | grep -v "^If" | grep -v "^Have"
}

test_gpu_connection() {
    log cyan "Testando conexão GPU..."
    local result
    result=$(ssh_exec "echo 'GPU OK' && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if echo "$result" | grep -q "GPU OK"; then
        log green "GPU conectada: $result"
        return 0
    fi
    log red "FALHA: Não conectou à GPU"
    return 1
}

get_gpu_info() {
    log cyan "Coletando info da GPU..."
    local disk uptime
    disk=$(ssh_exec "df -h /workspace | grep -E '^/|overlay' | awk '{print \$4}' | head -1")
    uptime=$(ssh_exec "cat /proc/uptime | awk '{print \$1 / 60}'")
    log green "GPU: Disco livre=${disk:-N/A}, Uptime=${uptime:-N/A} min"
}

compress_corpus() {
    if [ "$SKIP_COMPRESS" = "true" ]; then
        log yellow "SKIP: Compressão desabilitada"
        return 0
    fi

    log cyan "Iniciando compressão do corpus..."
    log gray "Source: $CORPUS_PATH"
    log gray "Output: $ARCHIVE_FILE"

    # Tamanho do corpus
    local corpus_size
    corpus_size=$(du -sb "$CORPUS_PATH" | awk '{print $1 / 1e9}')
    log cyan "Tamanho do corpus: $(printf "%.2f" "$corpus_size") GB"

    # Comprimir
    log yellow "Comprimindo (pode demorar 20-60 min)..."
    local start_time=$(date +%s)

    tar -czf "$ARCHIVE_FILE" -C "$(dirname "$CORPUS_PATH")" "$(basename "$CORPUS_PATH")"

    local elapsed=$(($(date +%s) - start_time))
    local final_size
    final_size=$(stat -f%z "$ARCHIVE_FILE" 2>/dev/null || stat -c%s "$ARCHIVE_FILE" 2>/dev/null)
    local final_gb=$(echo "$final_size / 1e9" | bc -l 2>/dev/null || echo "N/A")
    local ratio
    ratio=$(echo "scale=1; (1 - $final_size / ($corpus_size * 1e9)) * 100" | bc 2>/dev/null || echo "N/A")

    log green "Compressão concluída!"
    log gray "Tempo: $(printf "%.1f" "$(echo "$elapsed / 60" | bc -l)) min"
    log green "Tamanho final: $(printf "%.2f" "$final_gb") GB (${ratio}% compressão)"
}

upload_to_gpu() {
    local local_file="$1"
    local remote_path="$2"

    if [ "$SKIP_UPLOAD" = "true" ]; then
        log yellow "SKIP: Upload desabilitado"
        return 0
    fi

    log cyan "Iniciando upload para GPU..."
    log gray "Local: $local_file"
    log gray "Remote: $remote_path"

    local file_size
    file_size=$(stat -f%z "$local_file" 2>/dev/null || stat -c%s "$local_file" 2>/dev/null)
    local file_gb=$(echo "$file_size / 1e9" | bc -l 2>/dev/null || echo "N/A")
    log cyan "Tamanho: $(printf "%.2f" "$file_gb") GB"

    # Criar diretório remoto
    local remote_dir
    remote_dir=$(dirname "$remote_path")
    ssh_exec "mkdir -p $remote_dir"

    # Upload via SCP
    log yellow "Uploading (tempo estimado: $(printf "%.0f" "$(echo "$file_gb * 2" | bc -l)) min)..."
    local start_time=$(date +%s)

    scp -P "$GPU_PORT" -o StrictHostKeyChecking=no "$local_file" "root@$GPU_HOST:$remote_path"

    local elapsed=$(($(date +%s) - start_time))
    log green "Upload concluído!"
    log gray "Tempo: $(printf "%.1f" "$(echo "$elapsed / 60" | bc -l)) min"
}

deploy_project() {
    log cyan "Fazendo deploy do projeto F51..."

    # Verificar se projeto existe na GPU
    local project_exists
    project_exists=$(ssh_exec "test -d /workspace/$PROJECT_NAME && echo 'EXISTS'")

    if [ "$project_exists" = "EXISTS" ]; then
        log yellow "Projeto já existe na GPU"
        return 0
    fi

    log yellow "Projeto não encontrado, fazendo upload..."

    # Criar tar do projeto (sem data/)
    local project_tar="/tmp/f51_project.tar"
    cd "$PROJECT_ROOT"
    tar -cf "$project_tar" --exclude='data' --exclude='.git' --exclude='checkpoints' --exclude='runs' --exclude='__pycache__' \
        f51_darwin scripts tokenizer configs 2>/dev/null

    log gray "Projeto empacotado: $project_tar"

    upload_to_gpu "$project_tar" "/workspace/$PROJECT_NAME.tar"
    ssh_exec "cd /workspace && tar -xf $PROJECT_NAME.tar && rm $PROJECT_NAME.tar"

    log green "Projeto deployado!"
    rm -f "$project_tar"
}

extract_corpus_on_gpu() {
    local remote_archive="$1"
    local target_dir="$2"

    log cyan "Descomprimindo corpus na GPU..."

    ssh_exec "mkdir -p $target_dir"

    log yellow "Extraindo (pode demorar)..."

    local result
    result=$(ssh_exec "cd $target_dir && tar -xzf $remote_archive && ls -lh | wc -l")

    log green "Arquivos extraídos: $result"
}

start_training() {
    log cyan "Iniciando treinamento..."

    local train_cmd="cd /workspace/$PROJECT_NAME && nohup python scripts/train_cloud.py --device cuda --batch-size $BATCH_SIZE --block-size $BLOCK_SIZE --steps $STEPS --save-every 1000 --eval-every 500 --checkpoint-dir checkpoints/cloud --corpus data/corpus --tokenizer tokenizer/f51_bpe > /workspace/train.log 2>&1 &"

    ssh_exec "$train_cmd"

    sleep 3
    local pid
    pid=$(ssh_exec "pgrep -f train_cloud.py | head -1")

    log green "Treinamento iniciado!"
    log gray "PID: ${pid:-N/A}"
    log cyan "Log: ssh -p $GPU_PORT root@${GPU_HOST} 'tail -f /workspace/train.log'"
}

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     F51 DARWIN-SSD — AUTO DEPLOY GPU                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Verificações
log cyan "Verificando pré-requisitos..."
if [ ! -d "$CORPUS_PATH" ]; then
    log red "ERRO: Corpus não encontrado em $CORPUS_PATH"
    exit 1
fi

# Verificar saldo Vast.ai
if command -v vastai &> /dev/null; then
    local balance
    balance=$(vastai show user 2>&1 | grep "Credit" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    if [ -n "$balance" ]; then
        log gray "Saldo Vast.ai: \$$balance"
        local hours
        hours=$(echo "$balance / 0.501" | bc)
        log gray "Tempo disponível: ~${hours} horas"
    fi
fi

# 1. Testar conexão GPU
test_gpu_connection || exit 1

# 2. Info GPU
get_gpu_info

# 3. Deploy projeto
deploy_project

# 4. Comprimir corpus (se necessário)
if [ "$SKIP_COMPRESS" != "true" ] && [ ! -f "$ARCHIVE_FILE" ]; then
    compress_corpus
elif [ -f "$ARCHIVE_FILE" ]; then
    log yellow "Arquivo comprimido já existe: $ARCHIVE_FILE"
    local existing_size
    existing_size=$(stat -f%z "$ARCHIVE_FILE" 2>/dev/null || stat -c%s "$ARCHIVE_FILE" 2>/dev/null)
    local existing_gb=$(echo "$existing_size / 1e9" | bc -l 2>/dev/null || echo "N/A")
    log gray "Tamanho: $(printf "%.2f" "$existing_gb") GB"
fi

# 5. Upload corpus
upload_to_gpu "$ARCHIVE_FILE" "/workspace/corpus_backup.tar.gz"

# 6. Descomprimir na GPU
extract_corpus_on_gpu "/workspace/corpus_backup.tar.gz" "/workspace/$PROJECT_NAME/data"

# 7. Limpar arquivo local
log cyan "Remover arquivo comprimido local? (y/N)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    rm -f "$ARCHIVE_FILE"
    log gray "Arquivo local removido"
fi

# 8. Iniciar treino
start_training

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     DEPLOY CONCLUÍDO! TREINAMENTO RODANDO                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
log cyan "Comandos úteis:"
echo "  Ver log:     ssh -p $GPU_PORT root@${GPU_HOST} 'tail -f /workspace/train.log'"
echo "  Ver GPU:     ssh -p $GPU_PORT root@${GPU_HOST} 'nvidia-smi'"
echo "  Ver checkpoints: ssh -p $GPU_PORT root@${GPU_HOST} 'ls -lh /workspace/$PROJECT_NAME/checkpoints/cloud/'"
echo ""
