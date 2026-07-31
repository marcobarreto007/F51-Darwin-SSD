#!/usr/bin/env python
"""
F51 Maquinista — Orquestrador de Treino em Nuvem
=================================================
Estratégia para pobres ambiciosos:
  1. Uma GPU principal (RTX 6000 Ada 48GB) para treino pesado
  2. GPUs baratas de backup (RTX 3060 12GB) para continuar se a principal cair
  3. Checkpoints sincronizados entre instâncias via `vastai copy`
  4. Templates pré-configurados para startup instantâneo
  5. Stop de instâncias ociosas para economizar

Comandos:
  python research/maquinista.py status     — Ver estado de tudo
  python research/maquinista.py sync       — Sincronizar checkpoints entre instâncias
  python research/maquinista.py launch     — Lançar nova instância de treino
  python research/maquinista.py kill-all   — Matar tudo (emergência)
"""

import subprocess, sys, json, time, os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]

# ═══════════════════════════════════════════════════
# CONFIGURAÇÃO DO MAQUINISTA
# ═══════════════════════════════════════════════════

TRAIN_CONFIG = {
    "batch_size": 8,
    "block_size": 512,
    "max_steps": 200000,
    "save_every": 2000,
    "eval_every": 500,
    "learning_rate": 3e-4,
}

GPU_TIERS = {
    "main": {  # GPU principal — a melhor que o dinheiro compra
        "gpu_name": "RTX_6000Ada",
        "min_vram": 40,
        "max_price": 0.70,
        "disk": 40,
        "label": "f51-main",
    },
    "backup": {  # GPU reserva — barata, para continuar se a principal cair
        "gpu_name": "RTX_3060",
        "min_vram": 10,
        "max_price": 0.50,
        "disk": 40,
        "label": "f51-backup",
    },
    "cheap": {  # GPU de exploração — a mais barata possível
        "gpu_name": None,
        "min_vram": 8,
        "max_price": 0.30,
        "disk": 30,
        "label": "f51-explore",
    },
}

CKPT_DIR = "/workspace/F51-Darwin-SSD/checkpoints/cloud"
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"

# ═══════════════════════════════════════════════════
# COMANDOS
# ═══════════════════════════════════════════════════

def run_vastai(*args):
    """Run vastai CLI command and return output."""
    res = subprocess.run(["vastai"] + list(args), capture_output=True, text=True)
    return res.stdout.strip()

def run_vastai_json(*args):
    """Run vastai CLI with --raw and parse JSON."""
    res = subprocess.run(["vastai"] + list(args) + ["--raw"], capture_output=True, text=True)
    if res.returncode != 0:
        return None
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError:
        return None

def get_instances():
    """Get all current instances."""
    data = run_vastai_json("show", "instances")
    if not data:
        return []
    # Handle both list and dict responses
    if isinstance(data, dict):
        return data.get("instances", [])
    return data if isinstance(data, list) else []

def get_balance():
    """Get current credit balance."""
    data = run_vastai_json("show", "user")
    if isinstance(data, dict):
        return float(data.get("credit", 0))
    return 0.0

# ═══════════════════════════════════════════════════
# STATUS
# ═══════════════════════════════════════════════════

def cmd_status():
    """Show complete status of all cloud resources."""
    instances = get_instances()
    balance = get_balance()

    total_burn = 0.0
    running = [i for i in instances if i.get("actual_status") == "running"]
    exited = [i for i in instances if i.get("actual_status") == "exited"]

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  F51 MAQUINISTA — STATUS DA FROTA                        ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Crédito:  ${balance:.2f}                                         ║")
    print(f"║  Running:  {len(running)}  |  Exited: {len(exited)}                                  ║")
    print("╠══════════════════════════════════════════════════════════╣")

    for inst in instances:
        iid = inst.get("id", "?")
        gpu = inst.get("gpu_name", "?")
        vram = inst.get("gpu_ram", 0)
        status = inst.get("actual_status", "?")
        price = inst.get("min_bid", 0)
        gpu_util = inst.get("gpu_util", 0)
        ssh_host = inst.get("ssh_host", "")
        ssh_port = inst.get("ssh_port", "")

        if status == "running":
            total_burn += price
            icon = "🟢" if gpu_util > 10 else "💤"
        elif status == "exited":
            icon = "💀"
        else:
            icon = "⏳"

        print(f"║  {icon} {iid}  {gpu:<20s} {vram:.0f}GB  "
              f"${price:.3f}/h  GPU:{gpu_util}%  {status:<8s} ║")
        if ssh_host and status == "running":
            print(f"║     ssh -p {ssh_port} root@{ssh_host:<30s} ║")

    print("╠══════════════════════════════════════════════════════════╣")
    hours_left = balance / total_burn if total_burn > 0 else float('inf')
    print(f"║  Queima:   ${total_burn:.3f}/h                                     ║")
    print(f"║  Restam:   {hours_left:.1f}h com crédito atual                       ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Check training progress on main instance
    for inst in running:
        host = inst.get("ssh_host", "")
        port = inst.get("ssh_port", "")
        if host and port:
            try:
                res = subprocess.run(
                    ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no",
                     "-o", "ConnectTimeout=5", f"root@{host}",
                     "tail -3 /workspace/train_log.txt 2>/dev/null | grep -oP 'step=\\s*\\d+' | tail -1"
                     ],
                    capture_output=True, text=True, timeout=10
                )
                if res.stdout.strip():
                    step = res.stdout.strip()
                    print(f"  📊 {inst['id']}: {step}")
            except:
                pass

# ═══════════════════════════════════════════════════
# SYNC — copiar checkpoints entre instâncias
# ═══════════════════════════════════════════════════

def cmd_sync():
    """Sync checkpoints from main instance to all backups."""
    instances = get_instances()
    running = [i for i in instances if i.get("actual_status") == "running"]

    if len(running) < 2:
        print("❌ Precisa de pelo menos 2 instâncias rodando para sincronizar.")
        return

    main = running[0]
    main_id = main["id"]

    print(f"🔄 Sincronizando checkpoints de {main_id} → backups...")

    for backup in running[1:]:
        backup_id = backup["id"]
        print(f"   {main_id} → {backup_id} ...")

        # Copy checkpoints via vastai copy (direct instance-to-instance!)
        res = subprocess.run(
            ["vastai", "copy",
             f"{main_id}:{CKPT_DIR}/",
             f"{backup_id}:{CKPT_DIR}/"],
            capture_output=True, text=True
        )
        if res.returncode == 0:
            print(f"   ✅ {backup_id} sincronizado")
        else:
            print(f"   ❌ Erro: {res.stderr[:100]}")

    # Also download to local
    print(f"\n📥 Baixando para máquina local...")
    local_ckpt = ROOT / "checkpoints/cloud_sync"
    local_ckpt.mkdir(parents=True, exist_ok=True)

    res = subprocess.run(
        ["vastai", "copy",
         f"{main_id}:{CKPT_DIR}/",
         f"local:{local_ckpt}/"],
        capture_output=True, text=True
    )
    if res.returncode == 0:
        print(f"   ✅ Checkpoints baixados para {local_ckpt}")
    else:
        print(f"   ⚠️  Download local: {res.stderr[:100]}")

# ═══════════════════════════════════════════════════
# LAUNCH — lançar nova instância de treino
# ═══════════════════════════════════════════════════

def find_best_gpu(tier: str):
    """Find the best GPU offer for a given tier."""
    cfg = GPU_TIERS[tier]
    vram = cfg["min_vram"]
    price = cfg["max_price"]

    query = f"gpu_ram>={vram} cuda_vers>=12.0 rentable=true"
    if cfg["gpu_name"]:
        query += f" gpu_name={cfg['gpu_name']}"

    offers = run_vastai_json("search", "offers", query, "--order", "price", "--limit", "5")
    if not offers:
        return None

    # Filter by price
    affordable = [o for o in offers if o.get("min_bid", 999) <= price]
    if not affordable:
        return None

    # Pick best: lowest price with most VRAM
    affordable.sort(key=lambda o: (o.get("min_bid", 999), -o.get("gpu_ram", 0)))
    return affordable[0]

def launch_instance(tier: str, resume_from: Optional[str] = None):
    """Launch a new training instance."""
    cfg = GPU_TIERS[tier]
    offer = find_best_gpu(tier)

    if not offer:
        print(f"❌ Nenhuma GPU '{tier}' disponível nos critérios.")
        return None

    gpu_name = offer.get("gpu_name", "?")
    price = offer.get("min_bid", 0)
    vram = offer.get("gpu_ram", 0)
    offer_id = offer["id"]

    print(f"🛒 Alugando: {gpu_name} ({vram:.0f}GB) @ ${price:.3f}/h")

    # Build onstart script
    onstart = (
        "apt-get update -qq && apt-get install -y -qq rsync git 2>/dev/null; "
        "pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124; "
        "pip install -q tqdm pyyaml; "
        "mkdir -p /workspace/F51-Darwin-SSD"
    )

    # Create instance
    res = subprocess.run(
        ["vastai", "create", "instance", str(offer_id),
         "--image", IMAGE,
         "--disk", str(cfg["disk"]),
         "--label", cfg["label"],
         "--onstart", onstart,
         "--raw"],
        capture_output=True, text=True
    )

    if res.returncode != 0:
        print(f"❌ Erro: {res.stderr}")
        return None

    try:
        data = json.loads(res.stdout)
        instance_id = data.get("new_contract") or data.get("instance_id")
        print(f"✅ Instância {instance_id} criada. Aguardando ficar online...")
        return instance_id
    except json.JSONDecodeError:
        print(f"❌ Resposta inválida")
        return None

def wait_online(instance_id: str, timeout: int = 300):
    """Wait for instance to be running and return SSH details."""
    start = time.time()
    while time.time() - start < timeout:
        res = subprocess.run(
            ["vastai", "show", "instance", str(instance_id), "--raw"],
            capture_output=True, text=True
        )
        try:
            data = json.loads(res.stdout)
        except json.JSONDecodeError:
            time.sleep(5)
            continue

        status = data.get("actual_status")
        elapsed = int(time.time() - start)
        print(f"   [{elapsed}s] {status}")

        if status == "running":
            ssh_host = data.get("ssh_host", "")
            ssh_port = data.get("ssh_port", "")
            print(f"   ✅ Online! ssh -p {ssh_port} root@{ssh_host}")
            return ssh_host, ssh_port

        time.sleep(10)

    print("❌ Timeout esperando instância.")
    return None, None

def upload_and_train(ssh_host: str, ssh_port: str, instance_id: str, resume_from: Optional[str] = None):
    """Upload F51 code and launch training."""
    print("📦 Enviando código...")

    # Create tar locally
    tar_path = ROOT / "_f51_upload.tar"
    subprocess.run(
        ["tar", "-cf", str(tar_path),
         "f51_darwin", "scripts", "tokenizer", "requirements_cloud.txt"],
        cwd=str(ROOT), check=True
    )

    # SCP to instance
    subprocess.run(
        ["scp", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no",
         str(tar_path), f"root@{ssh_host}:/workspace/F51-Darwin-SSD/"],
        check=True
    )

    # Extract
    subprocess.run(
        ["ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no",
         f"root@{ssh_host}",
         "cd /workspace/F51-Darwin-SSD && tar -xf _f51_upload.tar"],
        check=True
    )

    tar_path.unlink(missing_ok=True)
    print("✅ Código enviado!")

    # Copy checkpoints from main if resuming
    if resume_from:
        print(f"📥 Copiando checkpoints da instância {resume_from}...")
        subprocess.run(
            ["vastai", "copy",
             f"{resume_from}:{CKPT_DIR}/",
             f"{instance_id}:{CKPT_DIR}/"],
            check=True
        )
        print("✅ Checkpoints copiados!")

    # Launch training in tmux
    bs = TRAIN_CONFIG["batch_size"]
    blk = TRAIN_CONFIG["block_size"]
    steps = TRAIN_CONFIG["max_steps"]
    save = TRAIN_CONFIG["save_every"]
    lr = TRAIN_CONFIG["learning_rate"]

    train_cmd = (
        f"cd /workspace/F51-Darwin-SSD && "
        f"python3 -u src/scripts/train_cloud.py "
        f"--device cuda --batch-size {bs} --block-size {blk} "
        f"--steps {steps} --save-every {save} --eval-every 500 "
        f"--lr {lr} --corpus data/corpus --tokenizer tokenizer/f51_bpe "
        f"--checkpoint-dir checkpoints/cloud "
        f"--no-weights "
        f"2>&1 | tee /workspace/train_log.txt"
    )

    print("🧮 Lançando treino...")
    subprocess.run(
        ["ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no",
         f"root@{ssh_host}",
         f"tmux new-session -d -s f51 bash -c '{train_cmd}'"],
        check=True
    )

    print(f"✅ Treino iniciado em {instance_id}!")
    print(f"   Monitor: ssh -p {ssh_port} root@{ssh_host} 'tail -f /workspace/train_log.txt'")
    print(f"   Anexar:  ssh -p {ssh_port} root@{ssh_host} 'tmux attach -t f51'")

def cmd_launch(tier: str = "main"):
    """Launch a new training instance."""
    # Find running instances to potentially copy checkpoints from
    instances = get_instances()
    running = [i for i in instances if i.get("actual_status") == "running"]

    resume_from = None
    if running:
        print(f"🔄 Instância principal já rodando: {running[0]['id']}")
        print(f"   Vou lançar backup sincronizado.")
        resume_from = running[0]["id"]
        tier = "backup"

    instance_id = launch_instance(tier, resume_from)
    if not instance_id:
        return

    ssh_host, ssh_port = wait_online(instance_id)
    if not ssh_host:
        return

    upload_and_train(ssh_host, ssh_port, instance_id, resume_from)

    # Save instance info
    info_path = ROOT / ".f51_instances.json"
    info = {}
    if info_path.exists():
        info = json.loads(info_path.read_text())
    info[instance_id] = {
        "tier": tier,
        "ssh_host": ssh_host,
        "ssh_port": ssh_port,
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "resume_from": resume_from,
    }
    info_path.write_text(json.dumps(info, indent=2))

# ═══════════════════════════════════════════════════
# STOP — parar sem destruir (economiza estado)
# ═══════════════════════════════════════════════════

def cmd_stop(instance_id: str):
    """Stop an instance without destroying it."""
    instances = get_instances()
    inst = next((i for i in instances if str(i.get("id")) == instance_id), None)
    if not inst:
        print(f"❌ Instância {instance_id} não encontrada.")
        return

    # Save checkpoints to volume or download first
    print(f"⏸️  Parando {instance_id}...")

    # Download checkpoints to local first
    local_ckpt = ROOT / f"checkpoints/cloud_{instance_id}"
    local_ckpt.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["vastai", "copy", f"{instance_id}:{CKPT_DIR}/", f"local:{local_ckpt}/"],
        check=False
    )

    # Stop instance
    res = run_vastai("stop", "instance", instance_id)
    print(f"   {res}")

def cmd_kill_all():
    """Destroy all instances (emergency)."""
    instances = get_instances()
    if not instances:
        print("Nenhuma instância para destruir.")
        return

    print(f"⚠️  Destruindo {len(instances)} instâncias...")
    for inst in instances:
        iid = inst.get("id")
        # Try to download checkpoints first
        local_ckpt = ROOT / f"checkpoints/cloud_{iid}"
        local_ckpt.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["vastai", "copy", f"{iid}:{CKPT_DIR}/", f"local:{local_ckpt}/"],
            timeout=30
        )
        # Destroy
        run_vastai("destroy", "instance", str(iid), "-y")
        print(f"   💀 {iid} destruída (checkpoints em {local_ckpt})")

# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Uso: maquinista.py <comando>")
        print("  status    — Ver estado da frota")
        print("  sync      — Sincronizar checkpoints")
        print("  launch    — Lançar nova instância de treino")
        print("  stop <id> — Parar instância (salva checkpoints)")
        print("  kill-all  — Destruir tudo (emergência)")
        return

    cmd = sys.argv[1]

    if cmd == "status":
        cmd_status()
    elif cmd == "sync":
        cmd_sync()
    elif cmd == "launch":
        tier = sys.argv[2] if len(sys.argv) > 2 else "main"
        cmd_launch(tier)
    elif cmd == "stop":
        if len(sys.argv) < 3:
            print("Uso: maquinista.py stop <instance_id>")
            return
        cmd_stop(sys.argv[2])
    elif cmd == "kill-all":
        confirm = input("Tem certeza? Vai destruir TODAS instâncias. [s/N] ")
        if confirm.lower() == 's':
            cmd_kill_all()
    else:
        print(f"Comando desconhecido: {cmd}")

if __name__ == "__main__":
    main()
