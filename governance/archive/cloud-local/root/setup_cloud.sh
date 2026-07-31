#!/bin/bash
# ============================================================
# F51 Darwin-SSD — Cloud Setup Script (RunPod / Lambda / Vast)
# ============================================================
# Roda uma vez ao iniciar o pod.
# Uso:  bash setup_cloud.sh
# ============================================================
set -e

echo "=============================="
echo " F51 Darwin-SSD Cloud Setup"
echo "=============================="

# ── 1. Sistema ──
echo "[1/6] Atualizando sistema..."
apt-get update -qq && apt-get install -y -qq tmux git wget 2>/dev/null || true

# ── 2. Python venv limpo (evita conflito com conda) ──
echo "[2/6] Criando venv Python..."
python3 -m venv f51_env
source f51_env/bin/activate
pip install --upgrade pip -q

# ── 3. PyTorch com CUDA 12.1 ──
echo "[3/6] Instalando PyTorch..."
# Detecta CUDA disponivel
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
    echo "  PyTorch ja instalado com CUDA."
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# ── 4. Dependencias ──
echo "[4/6] Instalando dependencias..."
pip install PyYAML pytest numpy

# Confirma GPU
echo ""
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA disponivel: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# ── 5. Verifica estrutura do projeto ──
echo "[5/6] Verificando projeto..."
if [ ! -f "scripts/train_cloud.py" ]; then
    echo "  ERRO: scripts/train_cloud.py nao encontrado."
    echo "  Certifique-se de fazer upload do projeto completo."
    exit 1
fi
if [ ! -d "data/corpus" ]; then
    echo "  ERRO: data/corpus/ nao encontrado."
    exit 1
fi
echo "  OK: projeto encontrado."

# ── 6. Sobe tmux e inicia treino ──
echo "[6/6] Iniciando treino em tmux (sessao: f51_train)..."
tmux new-session -d -s f51_train "source f51_env/bin/activate && python scripts/train_cloud.py --steps 100000 --batch-size 32 --block-size 512 --save-every 500 2>&1 | tee train_log.txt"

echo ""
echo "=============================="
echo " SETUP CONCLUIDO!"
echo "=============================="
echo ""
echo "Comandos uteis:"
echo "  tmux attach -t f51_train   # ver o treino ao vivo"
echo "  tmux detach                # sair sem matar (Ctrl+B, D)"
echo "  tail -f train_log.txt      # ver log"
echo "  nvidia-smi                 # status da GPU"
echo "  ls checkpoints/cloud/      # checkpoints salvos"
echo ""
echo "Para baixar os checkpoints pra maquina local:"
echo "  scp -r user@ip:~/checkpoints/cloud/ ./checkpoints/cloud/"
echo ""
