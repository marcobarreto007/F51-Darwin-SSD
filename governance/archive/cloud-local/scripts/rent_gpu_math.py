#!/usr/bin/env python
"""
F51 Cloud GPU Rental — Script Único (qualquer agente executa)
=============================================================

Aluga GPU no Vast.ai, sobe o código, treina matemática, sincroniza checkpoints.

PRÉ-REQUISITOS (já configurados nesta máquina):
    - vastai CLI instalado
    - Chave SSH registrada no Vast.ai
    - Chave API configurada: vastai set api-key <KEY>

USO:
    python scripts/rent_gpu_math.py

O script faz TUDO sozinho — não precisa saber SSH, SCP, tar, nada.
"""

import subprocess
import sys
import time
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

# ═══════════════════════════════════════════
# CONFIGURAÇÃO — ajuste aqui se quiser
# ═══════════════════════════════════════════
GPU_CONFIG = {
    "gpu_min_vram": 10,         # GB mínimo de VRAM
    "gpu_max_price": 0.50,      # $/h máximo
    "image": "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    "disk_gb": 40,              # GB de disco na instância
}

TRAIN_CONFIG = {
    "batch_size": 8,
    "block_size": 512,
    "max_steps": 50000,
    "save_every": 2500,
    "learning_rate": 3e-4,
}

# ═══════════════════════════════════════════
# PASSO 1: Verificar saldo
# ═══════════════════════════════════════════
def check_balance():
    """Verifica se tem crédito suficiente."""
    print("\n💰 Verificando saldo...")
    res = subprocess.run(["vastai", "show", "user"], capture_output=True, text=True)
    # Procura por "Credit" na saída
    for line in res.stdout.split("\n"):
        if "Credit" in line and "@" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                try:
                    credit = float(p)
                    print(f"   Saldo: ${credit:.2f}")
                    return credit
                except ValueError:
                    continue
    print("❌ Não foi possível ler o saldo. Rode: vastai show user")
    sys.exit(1)

# ═══════════════════════════════════════════
# PASSO 2: Buscar GPUs
# ═══════════════════════════════════════════
def search_gpus():
    """Busca as GPUs mais baratas que atendem os critérios."""
    vram = GPU_CONFIG["gpu_min_vram"]
    price = GPU_CONFIG["gpu_max_price"]
    query = f"gpu_ram>={vram} cuda_vers>=12.0 rentable=true"
    
    print(f"\n🔍 Buscando GPUs (≥{vram}GB, ≤${price}/h)...")
    res = subprocess.run(
        ["vastai", "search", "offers", query, "--order", "price", "--limit", "10", "--raw"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print(f"❌ Erro na busca: {res.stderr}")
        sys.exit(1)
    
    data = json.loads(res.stdout)
    if not data:
        print("❌ Nenhuma GPU encontrada.")
        sys.exit(1)
    
    # Filtrar por preço máximo
    affordable = []
    for off in data:
        p = off.get("min_bid", 999)
        if p <= price:
            affordable.append(off)
    
    if not affordable:
        print(f"❌ Nenhuma GPU ≤${price}/h. As mais baratas:")
        for off in data[:3]:
            print(f"   {off.get('gpu_name','?')} {off.get('gpu_ram',0):.0f}GB ${off.get('min_bid',0):.3f}/h")
        sys.exit(1)
    
    print(f"\n📊 {len(affordable)} GPUs disponíveis:")
    print(f"   {'#':<3s} {'GPU':<20s} {'VRAM':>5s}  {'$/h':>7s}  {'Horas*':>6s}  {'Local':<20s}")
    print(f"   {'-'*3} {'-'*20} {'-'*5}  {'-'*7}  {'-'*6}  {'-'*20}")
    
    for i, off in enumerate(affordable[:10]):
        gpu = off.get("gpu_name", "?")[:20]
        vram_gb = off.get("gpu_ram", 0)
        p = off.get("min_bid", 0)
        hours = 5.57 / p if p > 0 else 0
        loc = off.get("geolocation", "?")[:20]
        print(f"   {i+1:<3d} {gpu:<20s} {vram_gb:>4.0f}GB ${p:>6.3f}  {hours:>5.1f}h  {loc:<20s}")
    
    print(f"   *Horas estimadas com $5.57 de crédito")
    return affordable

# ═══════════════════════════════════════════
# PASSO 3: Alugar instância
# ═══════════════════════════════════════════
def rent_instance(offer):
    """Aluga a GPU e espera ficar online."""
    offer_id = offer["id"]
    gpu_name = offer.get("gpu_name", "GPU")
    price = offer.get("min_bid", 0)
    
    print(f"\n🛒 Alugando: {gpu_name} @ ${price:.3f}/h")
    print(f"   (Não cobra nada até a máquina estar online)")
    
    # Criar instância
    res = subprocess.run(
        ["vastai", "create", "instance", str(offer_id),
         "--image", GPU_CONFIG["image"],
         "--disk", str(GPU_CONFIG["disk_gb"]),
         "--raw"],
        capture_output=True, text=True,
    )
    
    if res.returncode != 0:
        print(f"❌ Erro ao criar instância: {res.stderr}")
        sys.exit(1)
    
    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError:
        print(f"❌ Resposta inválida: {res.stdout[:200]}")
        sys.exit(1)
    
    instance_id = data.get("new_contract") or data.get("instance_id") or data.get("id")
    if not instance_id:
        print(f"❌ Não encontrei instance_id na resposta: {data}")
        sys.exit(1)
    
    print(f"   Instância ID: {instance_id}")
    
    # Aguardar ficar online
    print(f"\n⏳ Aguardando máquina ligar (pode levar 1-3 minutos)...")
    start = time.time()
    ssh_host, ssh_port = None, None
    
    while time.time() - start < 300:  # 5 min timeout
        res = subprocess.run(
            ["vastai", "show", "instance", str(instance_id), "--raw"],
            capture_output=True, text=True,
        )
        try:
            data = json.loads(res.stdout)
        except json.JSONDecodeError:
            time.sleep(5)
            continue
        
        status = data.get("actual_status", "?")
        elapsed = int(time.time() - start)
        print(f"   [{elapsed:>3d}s] Status: {status}")
        
        if status == "running":
            ssh_host = data.get("ssh_host", "ssh2.vast.ai")
            ssh_port = data.get("ssh_port", "22")
            print(f"\n✅ MÁQUINA ONLINE!")
            print(f"   SSH: ssh -p {ssh_port} root@{ssh_host}")
            break
        
        time.sleep(10)
    
    if not ssh_host:
        print("❌ Timeout — a instância não ficou online em 5 minutos.")
        print(f"   Verifique manualmente: vastai show instance {instance_id}")
        sys.exit(1)
    
    return instance_id, ssh_host, ssh_port

# ═══════════════════════════════════════════
# PASSO 4: Upload do código e corpus
# ═══════════════════════════════════════════
def ssh(ssh_host, ssh_port, command):
    """Executa comando via SSH."""
    return subprocess.run(
        ["ssh", "-p", str(ssh_port),
         "-o", "StrictHostKeyChecking=no",
         "-o", "ConnectTimeout=10",
         f"root@{ssh_host}", command],
        capture_output=True, text=True,
    )

def upload_code(ssh_host, ssh_port):
    """Empacota e envia o projeto para a GPU."""
    print(f"\n📦 Empacotando projeto...")
    
    # Criar tar com tudo que precisa
    tar_path = ROOT / "_cloud_upload.tar"
    dirs_to_pack = ["f51_darwin", "scripts", "tokenizer", "data/corpus"]
    files_to_pack = ["requirements_cloud.txt", "setup_cloud.sh"]
    
    # Verifica o que existe
    existing = []
    for d in dirs_to_pack:
        p = ROOT / d
        if p.exists():
            existing.append(d)
        else:
            print(f"   ⚠️  {d}/ não encontrado, pulando...")
    for f in files_to_pack:
        p = ROOT / f
        if p.exists():
            existing.append(f)
    
    cmd = ["tar", "-cf", str(tar_path)] + existing
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    
    size_mb = tar_path.stat().st_size / 1e6
    print(f"   Tamanho: {size_mb:.0f} MB")
    
    # Criar diretório remoto
    print(f"\n📤 Enviando para a GPU ({size_mb:.0f} MB)...")
    ssh(ssh_host, ssh_port, "mkdir -p /workspace/F51-Darwin-SSD")
    
    # SCP
    t0 = time.time()
    subprocess.run(
        ["scp", "-P", str(ssh_port),
         "-o", "StrictHostKeyChecking=no",
         str(tar_path),
         f"root@{ssh_host}:/workspace/F51-Darwin-SSD/"],
        check=True,
    )
    elapsed = time.time() - t0
    print(f"   Upload: {elapsed:.0f}s ({size_mb/elapsed:.0f} MB/s)")
    
    # Extrair
    print("📂 Extraindo...")
    ssh(ssh_host, ssh_port, 
        "cd /workspace/F51-Darwin-SSD && tar -xf _cloud_upload.tar")
    
    # Limpar
    tar_path.unlink(missing_ok=True)
    print("✅ Código + corpus na GPU!")

# ═══════════════════════════════════════════
# PASSO 5: Instalar dependências
# ═══════════════════════════════════════════
def setup_environment(ssh_host, ssh_port):
    """Instala PyTorch e dependências na GPU."""
    print("\n🔧 Instalando dependências...")
    
    # Verificar GPU
    res = ssh(ssh_host, ssh_port, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    print(f"   GPU detectada: {res.stdout.strip()}")
    
    # Instalar
    res = ssh(ssh_host, ssh_port,
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 "
        "&& pip install tqdm pyyaml --break-system-packages"
    )
    if res.returncode != 0:
        print(f"❌ Erro instalando: {res.stderr[:300]}")
        sys.exit(1)
    
    print("✅ Ambiente pronto!")

# ═══════════════════════════════════════════
# PASSO 6: Rodar treino matemático
# ═══════════════════════════════════════════
def launch_training(ssh_host, ssh_port, instance_id):
    """Inicia o treino em background e salva as info de conexão."""
    print("\n🧮 Lançando treino matemático...")
    
    bs = TRAIN_CONFIG["batch_size"]
    blk = TRAIN_CONFIG["block_size"]
    steps = TRAIN_CONFIG["max_steps"]
    save = TRAIN_CONFIG["save_every"]
    lr = TRAIN_CONFIG["learning_rate"]
    
    cmd = (
        f"cd /workspace/F51-Darwin-SSD && "
        f"nohup python scripts/train_cloud.py "
        f"--device cuda "
        f"--batch-size {bs} "
        f"--block-size {blk} "
        f"--steps {steps} "
        f"--save-every {save} "
        f"--lr {lr} "
        f"--corpus data/corpus "
        f"--tokenizer tokenizer/f51_bpe "
        f"--checkpoint-dir checkpoints/math_cloud "
        f"> /workspace/train_math.log 2>&1 &"
    )
    
    res = ssh(ssh_host, ssh_port, cmd)
    if res.returncode != 0:
        print(f"❌ Erro ao lançar treino: {res.stderr}")
        sys.exit(1)
    
    # Aguardar o processo iniciar
    time.sleep(5)
    res = ssh(ssh_host, ssh_port, "head -20 /workspace/train_math.log")
    print(f"\n📋 Início do log:")
    print(res.stdout[:500])
    
    # Salvar info de conexão
    info = {
        "instance_id": instance_id,
        "ssh_host": ssh_host,
        "ssh_port": ssh_port,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "gpu": GPU_CONFIG,
        "training": TRAIN_CONFIG,
    }
    (ROOT / ".vast_math_instance.json").write_text(json.dumps(info, indent=2))
    
    return info

# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════╗")
    print("║   F51 MATH CLOUD — ONE-CLICK GPU RENTAL      ║")
    print("║   Aluga, sobe código, treina matemática      ║")
    print("╚══════════════════════════════════════════════╝")
    
    # 1. Saldo
    credit = check_balance()
    if credit < 1.0:
        print(f"❌ Saldo muito baixo: ${credit:.2f}. Mínimo $1.00.")
        sys.exit(1)
    
    # 2. Buscar
    offers = search_gpus()
    
    # 3. Escolher (automático: a mais barata com mais VRAM)
    # Ordena por: VRAM descendente, preço ascendente
    offers.sort(key=lambda o: (-o.get("gpu_ram", 0), o.get("min_bid", 999)))
    chosen = offers[0]
    
    gpu_name = chosen.get("gpu_name", "GPU")
    price = chosen.get("min_bid", 0)
    vram = chosen.get("gpu_ram", 0)
    
    print(f"\n🎯 Seleção automática: {gpu_name} | {vram:.0f}GB | ${price:.3f}/h")
    print(f"   Estimativa: ~{credit/price:.1f}h de treino com ${credit:.2f}")
    
    # Auto-confirmar (sem prompt — o agente não interage)
    print("\n⏩ Iniciando aluguel automaticamente...")
    
    # 4. Alugar
    instance_id, ssh_host, ssh_port = rent_instance(chosen)
    
    # 5. Upload
    upload_code(ssh_host, ssh_port)
    
    # 6. Setup
    setup_environment(ssh_host, ssh_port)
    
    # 7. Treinar
    info = launch_training(ssh_host, ssh_port, instance_id)
    
    # 8. Resumo final
    print("\n" + "=" * 54)
    print("  TREINO MATEMÁTICO RODANDO NA NUVEM! 🧮")
    print("=" * 54)
    print(f"  Instance ID:  {instance_id}")
    print(f"  SSH:          ssh -p {ssh_port} root@{ssh_host}")
    print(f"  Log:          tail -f /workspace/train_math.log")
    print(f"  Checkpoints:  ls /workspace/F51-Darwin-SSD/checkpoints/math_cloud/")
    print()
    print(f"  Para ver progresso:")
    print(f"    ssh -p {ssh_port} root@{ssh_host} 'tail -30 /workspace/train_math.log'")
    print()
    print(f"  Para baixar checkpoints:")
    print(f"    scp -P {ssh_port} root@{ssh_host}:/workspace/F51-Darwin-SSD/checkpoints/math_cloud/*.pt checkpoints/math_cloud/")
    print()
    print(f"  Para DESTRUIR a instância:")
    print(f"    vastai destroy instance {instance_id}")
    print("=" * 54)

if __name__ == "__main__":
    main()
