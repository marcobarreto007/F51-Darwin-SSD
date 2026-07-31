#!/usr/bin/env python3
"""
F51 CLOUD TRAIN AUTOMATOR — One-click deploy
==============================================

1. Empacota código
2. Upload via SCP
3. Instala dependências
4. Inicia treino por 12h
"""

import os
import subprocess
import sys
import time
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "ssh_host": "ssh6.vast.ai",
    "ssh_port": 17824,
    "instance_id": 45047825,
    "workspace": "/workspace/F51-Darwin-SSD",
}


def ssh(cmd: str):
    """Executa comando via SSH."""
    return subprocess.run(
        ["ssh", "-p", str(CONFIG["ssh_port"]), "-o", "StrictHostKeyChecking=no",
         f"root@{CONFIG['ssh_host']}", cmd],
        capture_output=True, text=True, check=False
    )


def scp(local: Path, remote: str):
    """Upload via SCP."""
    subprocess.run(
        ["scp", "-P", str(CONFIG["ssh_port"]), "-o", "StrictHostKeyChecking=no",
         str(local), f"root@{CONFIG['ssh_host']}:{remote}"],
        check=True
    )


def pack_code():
    """Empacota código essencial."""
    print("📦 Empacotando código...")
    pack = ROOT / "_cloud_pack.tar"
    files = [
        "f51_darwin",
        "scripts",
        "tokenizer/f51_bpe_80k",
        "data",
        "configs",
    ]
    existing = [f for f in files if (ROOT / f).exists()]
    subprocess.run(["tar", "-cf", str(pack)] + existing, cwd=str(ROOT), check=True)
    size_mb = pack.stat().st_size / 1e6
    print(f"   Pack: {size_mb:.0f} MB")
    return pack


def upload_and_setup():
    """Upload e configura ambiente."""
    pack = pack_code()

    # Criar workspace
    print("📤 Criando workspace...")
    ssh(f"mkdir -p {CONFIG['workspace']}")

    # Upload
    print(f"📤 Enviando {pack.stat().st_size/1e6:.0f} MB...")
    t0 = time.time()
    scp(pack, f"{CONFIG['workspace']}/")
    elapsed = time.time() - t0
    print(f"   Upload: {elapsed:.0f}s ({pack.stat().st_size/1e6/elapsed:.1f} MB/s)")

    # Extrair
    print("📂 Extraindo...")
    ssh(f"cd {CONFIG['workspace']} && tar -xf _cloud_pack.tar")

    # Instalar dependências
    print("🔧 Instalando PyTorch...")
    ssh(f"cd {CONFIG['workspace']} && "
        f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q "
        f"&& pip install numpy pyyaml tqdm -q")

    # Verificar GPU
    res = ssh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    print(f"   GPU: {res.stdout.strip()}")

    pack.unlink(missing_ok=True)
    print("✅ Ambiente pronto!")


def launch_training():
    """Inicia treino por 12h."""
    print("🧮 Iniciando treino (12h)...")

    cmd = (
        f"cd {CONFIG['workspace']} && "
        f"nohup python scripts/darwin_organism.py "
        f"--config configs/organism.yaml "
        f"--device cuda "
        f"--max-steps 50000 "
        f"--batch-size 4 "
        f"--block-size 1024 "
        f"--checkpoint-dir checkpoints/cloud_12h "
        f"> /workspace/train.log 2>&1 &"
    )

    ssh(cmd)
    time.sleep(5)

    # Ver log
    res = ssh("tail -20 /workspace/train.log")
    print("📋 Log inicial:")
    print(res.stdout[:500])

    print(f"\n✅ Treino iniciado!")
    print(f"   Monitor: ssh -p {CONFIG['ssh_port']} root@{CONFIG['ssh_host']} 'tail -f /workspace/train.log'")
    print(f"   Checkpoints: {CONFIG['workspace']}/checkpoints/cloud_12h/")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════╗")
    print("║   F51 CLOUD TRAIN — AUTOMATED DEPLOY          ║")
    print("╚══════════════════════════════════════════════╝\n")

    try:
        upload_and_setup()
        launch_training()
    except Exception as e:
        print(f"❌ Erro: {e}")
        sys.exit(1)
