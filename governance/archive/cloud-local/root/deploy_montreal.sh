#!/bin/bash
# ================================================================
# F51 DARWIN-X 600M — Deploy Montreal A100 SXM4
# 
# Máquina: #40749926 | Quebec, CA | 1x A100 SXM4 40GB
#          1321 GB/s | 48 CPU | 129 GB RAM | $0.669/hr
# ================================================================
set -euo pipefail

echo "========================================"
echo " F51 DARWIN-X 600M — Montreal A100 SXM4"
echo " $(date)"
echo "========================================"

# ── 1. Sistema ──
echo "[1/6] Setup sistema..."
apt-get update -qq && apt-get install -y -qq python3-pip git curl 2>&1 | tail -1
pip install --upgrade pip -q

# ── 2. Dependências ──
echo "[2/6] Instalando PyTorch + dependências..."
pip install -q torch --index-url https://download.pytorch.org/whl/cu121
pip install -q PyYAML datasets transformers tokenizers

# ── 3. Clone do repo ──
echo "[3/6] Clonando F51-Darwin-SSD..."
cd /workspace
if [ -d "F51-Darwin-SSD" ]; then
    cd F51-Darwin-SSD && git pull
else
    git clone https://github.com/marcobarreto007/F51-Darwin-SSD.git
    cd F51-Darwin-SSD
fi
git checkout cursor/bilingual-readme-olavo-sources 2>/dev/null || true

# ── 4. Tokenizer ──
echo "[4/6] Verificando tokenizer..."
if [ ! -f "tokenizer/f51_bpe_80k/tokenizer.json" ]; then
    echo "   Tokenizer não encontrado. Precisa subir do local:"
    echo "   scp -P <port> tokenizer/f51_bpe_80k/* root@<ip>:/workspace/F51-Darwin-SSD/tokenizer/f51_bpe_80k/"
fi

# ── 5. Tokens (precisa subir do local) ──
echo "[5/6] Verificando tokens..."
if [ ! -f "data/tokens_mixed.bin" ]; then
    echo "   tokens_mixed.bin não encontrado. Suba do local:"
    echo "   scp -P <port> data/tokens_mixed.bin root@<ip>:/workspace/F51-Darwin-SSD/data/"
fi

# ── 6. Lançar treino ──
echo "[6/6] Lançando treino..."
echo ""
echo "   ╔══════════════════════════════════════════════════╗"
echo "   ║     🚀 F51 DARWIN-X 600M — A100 SXM4           ║"
echo "   ║     6 experts | aux adaptativo | anti-colapso    ║"
echo "   ║     Flash Attention | bf16 | 4096 block          ║"
echo "   ╚══════════════════════════════════════════════════╝"
echo ""

python -u scripts/train_darwin_x.py \
    --config configs/darwin_x_600m.yaml \
    --token-bin data/tokens_mixed.bin \
    --tokenizer tokenizer/f51_bpe_80k \
    --device cuda \
    --steps 100000 \
    --batch-size 2 \
    --block-size 4096 \
    --lr 1.5e-4 \
    --save-every 1000 \
    --precision bf16 \
    2>&1 | tee train_montreal.log

echo ""
echo "========================================"
echo " Treino finalizado ou interrompido."
echo " Checkpoints em: checkpoints/darwin_x/"
echo "========================================"
