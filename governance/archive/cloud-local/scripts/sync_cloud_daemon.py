#!/usr/bin/env python
"""
F51 Cloud Sync Daemon — roda em background, baixa checkpoints da nuvem.

Faz polling do cloud_latest.json a cada N minutos.
Quando detecta step novo, baixa o .pt via SCP.
Atualiza checkpoints/cloud/cloud_latest.json local.
O serve_f51.py detecta e recarrega automaticamente.

Uso:
    python scripts/sync_cloud_daemon.py
    python scripts/sync_cloud_daemon.py --interval 300
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLOUD_HOST = "root@ssh2.vast.ai"
CLOUD_PORT = "32632"
CLOUD_LATEST = "/workspace/F51-Darwin-SSD/checkpoints/cloud/cloud_latest.json"
CLOUD_CHECKPOINT_DIR = "/workspace/F51-Darwin-SSD/checkpoints/cloud"
LOCAL_DIR = ROOT / "checkpoints" / "cloud"
LOCAL_LATEST = LOCAL_DIR / "cloud_latest.json"


def ssh_get_json(remote_path: str) -> dict | None:
    """Fetch a JSON file from the cloud instance via SSH."""
    try:
        result = subprocess.run(
            [
                "ssh", "-o", "StrictHostKeyChecking=no",
                "-p", CLOUD_PORT, CLOUD_HOST,
                f"cat {remote_path} 2>/dev/null",
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return json.loads(result.stdout)
    except Exception:
        return None


def download_checkpoint(remote_path: str, local_path: Path) -> bool:
    """Download a single checkpoint file via SCP."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                "scp", "-o", "StrictHostKeyChecking=no",
                "-P", CLOUD_PORT,
                f"{CLOUD_HOST}:{remote_path}",
                str(local_path),
            ],
            capture_output=True, text=True, timeout=600,  # 10 min timeout for large files
        )
        return result.returncode == 0 and local_path.exists()
    except Exception:
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="F51 Cloud Sync Daemon")
    parser.add_argument("--interval", type=int, default=300,
                       help="Poll interval in seconds (default: 300 = 5 min)")
    parser.add_argument("--once", action="store_true",
                       help="Run once and exit (for testing)")
    args = parser.parse_args()

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    local_step = 0
    if LOCAL_LATEST.exists():
        try:
            local_step = json.loads(LOCAL_LATEST.read_text()).get("step", 0)
        except Exception:
            pass

    print(f"F51 Cloud Sync Daemon")
    print(f"  Nuvem: {CLOUD_HOST}:{CLOUD_PORT}")
    print(f"  Local: {LOCAL_DIR}")
    print(f"  Step local: {local_step}")
    print(f"  Intervalo: {args.interval}s")
    if args.once:
        print(f"  Modo: once (executa e sai)")
    print()

    while True:
        try:
            now = datetime.now(timezone.utc).strftime("%H:%M:%S")

            cloud_data = ssh_get_json(CLOUD_LATEST)
            if cloud_data is None:
                print(f"[{now}] Nuvem offline ou sem checkpoint ainda.")
                if args.once:
                    break
                time.sleep(args.interval)
                continue

            cloud_step = cloud_data.get("step", 0)
            cloud_path = cloud_data.get("path", "")
            cloud_loss = cloud_data.get("loss", "?")

            if cloud_step <= local_step:
                print(f"[{now}] Sem novidades (cloud={cloud_step} local={local_step})")
                if args.once:
                    break
                time.sleep(args.interval)
                continue

            print(f"[{now}] NOVO CHECKPOINT! step={cloud_step} loss={cloud_loss}")

            local_pt = LOCAL_DIR / f"cloud_step_{cloud_step:07d}.pt"

            if local_pt.exists():
                print(f"  Ja existe: {local_pt.name}")
            else:
                print(f"  Baixando {cloud_path} ...")
                ok = download_checkpoint(cloud_path, local_pt)
                if not ok:
                    print(f"  ERRO no download. Tentando novamente no proximo ciclo.")
                    if args.once:
                        break
                    time.sleep(args.interval)
                    continue
                size_mb = local_pt.stat().st_size / 1e6
                print(f"  Download OK: {size_mb:.0f}MB")

            # Update local pointer
            LOCAL_LATEST.write_text(json.dumps({
                "path": str(local_pt.relative_to(ROOT)),
                "step": cloud_step,
                "loss": cloud_loss,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "cloud_sync",
            }, indent=2) + "\n")
            local_step = cloud_step
            print(f"  Local atualizado: step={cloud_step}")
            print(f"  Server vai detectar no proximo ciclo de auto-reload (60s).")

            if args.once:
                break

        except KeyboardInterrupt:
            print("\nSync daemon encerrado.")
            break
        except Exception as e:
            print(f"[{now}] Erro: {e}")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
