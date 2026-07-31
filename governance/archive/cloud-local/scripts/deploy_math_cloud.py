#!/usr/bin/env python
"""
F51 Math-Only Cloud Training — Deploy & Run
Aluga GPU barata (RTX 3060 12GB a $0.43/h), sobe o código,
roda treino com 91% matemática, sincroniza checkpoints.

Uso:
    python scripts/deploy_math_cloud.py
"""
import subprocess
import sys
import time
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════
GPU_MIN_VRAM = 10          # GB
GPU_MAX_PRICE = 0.50        # $/hr
SEARCH_QUERY = f"gpu_ram>={GPU_MIN_VRAM} cuda_vers>=12.0 rentable=true price<={GPU_MAX_PRICE}"
INSTANCE_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
INSTANCE_DISK = 30          # GB

# ═══════════════════════════════════════════
# STEP 1: Search & Rent
# ═══════════════════════════════════════════
def search_gpus():
    print("🔍 Buscando GPUs baratas...")
    res = subprocess.run(
        ["vastai", "search", "offers", SEARCH_QUERY, "--order", "price", "--limit", "5", "--raw"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print("Erro:", res.stderr)
        sys.exit(1)
    return json.loads(res.stdout)

def rent_gpu(offer_id):
    print(f"💰 Alugando GPU {offer_id}...")
    res = subprocess.run(
        ["vastai", "create", "instance", str(offer_id),
         "--image", INSTANCE_IMAGE,
         "--disk", str(INSTANCE_DISK),
         "--raw"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print("Erro ao alugar:", res.stderr)
        sys.exit(1)
    data = json.loads(res.stdout)
    instance_id = data.get("new_contract") or data.get("instance_id") or data.get("id")
    print(f"✅ Instância criada: {instance_id}")
    return instance_id

def wait_online(instance_id, timeout=300):
    print(f"⏳ Aguardando instância {instance_id} ficar online...")
    start = time.time()
    while time.time() - start < timeout:
        res = subprocess.run(
            ["vastai", "show", "instance", str(instance_id), "--raw"],
            capture_output=True, text=True,
        )
        data = json.loads(res.stdout)
        status = data.get("actual_status")
        print(f"   Status: {status} ({int(time.time()-start)}s)")
        if status == "running":
            ssh_host = data.get("ssh_host", "ssh2.vast.ai")
            ssh_port = data.get("ssh_port", "22")
            print(f"✅ Online! SSH: root@{ssh_host} -p {ssh_port}")
            return ssh_host, ssh_port
        time.sleep(10)
    print("❌ Timeout aguardando instância")
    sys.exit(1)

# ═══════════════════════════════════════════
# STEP 2: Upload & Setup
# ═══════════════════════════════════════════
def ssh_cmd(ssh_host, ssh_port, command):
    return [
        "ssh", "-p", str(ssh_port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        f"root@{ssh_host}", command,
    ]

def upload_project(ssh_host, ssh_port):
    print("📦 Empacotando projeto...")
    tar_file = ROOT / "project_math.tar"
    subprocess.run(
        ["tar", "-cf", str(tar_file),
         "f51_darwin", "scripts", "tokenizer",
         "requirements_cloud.txt"],
        cwd=str(ROOT), check=True,
    )

    print("📤 Enviando para GPU...")
    subprocess.run(["ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no",
                     f"root@{ssh_host}", "mkdir -p /workspace/F51-Darwin-SSD"], check=True)

    subprocess.run(["scp", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no",
                     str(tar_file),
                     f"root@{ssh_host}:/workspace/F51-Darwin-SSD/"], check=True)

    subprocess.run(ssh_cmd(ssh_host, ssh_port,
        "cd /workspace/F51-Darwin-SSD && tar -xf project_math.tar"), check=True)

    tar_file.unlink(missing_ok=True)
    print("✅ Upload concluído!")

def setup_env(ssh_host, ssh_port):
    print("🔧 Instalando dependências...")
    subprocess.run(ssh_cmd(ssh_host, ssh_port,
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && "
        "pip install tqdm pyyaml --break-system-packages"
    ), check=True)
    print("✅ Ambiente pronto!")

# ═══════════════════════════════════════════
# STEP 3: Run Math Training
# ═══════════════════════════════════════════
def run_training(ssh_host, ssh_port, instance_id):
    print("🧮 Iniciando treino matemático...")
    cmd = (
        "cd /workspace/F51-Darwin-SSD && "
        "nohup python scripts/train_cloud.py "
        "--device cuda "
        "--batch-size 8 "
        "--block-size 512 "
        "--steps 50000 "
        "--save-every 2500 "
        "--lr 3e-4 "
        "--corpus data/corpus "
        "--tokenizer tokenizer/f51_bpe "
        "--checkpoint-dir checkpoints/math_cloud "
        "> /workspace/train_math.log 2>&1 &"
    )
    subprocess.run(ssh_cmd(ssh_host, ssh_port, cmd), check=True)
    print(f"✅ Treino iniciado em background!")
    print(f"   Instance ID: {instance_id}")
    print(f"   SSH: ssh -p {ssh_port} root@{ssh_host}")
    print(f"   Log: tail -f /workspace/train_math.log")
    print(f"   Parar: vastai destroy instance {instance_id}")
    # Salva info pra sync
    (ROOT / ".vast_math_instance").write_text(json.dumps({
        "instance_id": instance_id,
        "ssh_host": ssh_host,
        "ssh_port": ssh_port,
        "timestamp": time.time(),
    }))

# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║   F51 MATH CLOUD TRAINING DEPLOY         ║")
    print("║   RTX 3060 12GB ~ $0.43/h                ║")
    print("╚══════════════════════════════════════════╝")
    print()

    # Search
    offers = search_gpus()
    if not offers:
        print("❌ Nenhuma GPU disponível nos critérios")
        sys.exit(1)

    print(f"📊 {len(offers)} GPUs encontradas:")
    for i, off in enumerate(offers[:5]):
        gpu = off.get("gpu_name", off.get("gpu_model", "?"))
        vram = off.get("gpu_ram", 0)
        price = off.get("min_bid", off.get("price", 0))
        print(f"  {i+1}. {gpu} | {vram:.0f}GB | ${price:.3f}/h | ID={off['id']}")

    # Pick first (cheapest)
    chosen = offers[0]
    gpu_name = chosen.get("gpu_name", "GPU")
    price = chosen.get("min_bid", 0)
    print(f"\n🎯 Selecionada: {gpu_name} @ ${price:.3f}/h")
    print(f"   Seu saldo: $5.57 → ~{5.57/price:.0f}h de treino")
    print()

    resp = input("Alugar? [S/n] ").strip().lower()
    if resp and resp != 's':
        print("Cancelado.")
        sys.exit(0)

    instance_id = rent_gpu(chosen["id"])
    ssh_host, ssh_port = wait_online(instance_id)

    upload_project(ssh_host, ssh_port)
    setup_env(ssh_host, ssh_port)
    run_training(ssh_host, ssh_port, instance_id)

    print()
    print("╔══════════════════════════════════════════╗")
    print("║  TREINO MATEMÁTICO RODANDO NA NUVEM! 🧮  ║")
    print("╚══════════════════════════════════════════╝")
