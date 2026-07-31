#!/usr/bin/env python3
"""
F51 Darwin-X — Real-time Training Dashboard
Shows: training metrics, GPU, blockchain, checkpoint, SHA-256 hashes.
Updates every 2s. Ctrl+C to exit.

Usage:
  .venv_nitro\\Scripts\\python.exe src\\tools\\darwin_dashboard.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOT = ROOT / "workspace" / "03_CHECKPOINTS_100M_FULL_V9"
BLOCKCHAIN_FILE = ROOT / "workspace" / "runtime" / "organism" / "blockchain" / "blocks.jsonl"
LEDGER_FILE = CKPT_ROOT / "causal_events.jsonl"
DOPAMINE_FILE = CKPT_ROOT / "dopamine_status.txt"
POINTER_FILE = CKPT_ROOT / "organism_latest.json"


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "—"
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16] + "..."


def gpu_info() -> list[dict]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        gpus = []
        for line in out.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            gpus.append(
                {
                    "idx": parts[0],
                    "name": parts[1],
                    "util": parts[2],
                    "mem_used": parts[3],
                    "mem_total": parts[4],
                    "temp": parts[5],
                }
            )
        return gpus
    except Exception:
        return []


def python_processes() -> list[dict]:
    try:
        import psutil
    except ImportError:
        return []
    procs = []
    for p in psutil.process_iter(["pid", "name", "memory_info", "cpu_percent"]):
        if p.info["name"] and "python" in p.info["name"].lower():
            try:
                mem = p.info["memory_info"].rss / (1024**3)
            except Exception:
                mem = 0
            procs.append({"pid": p.info["pid"], "mem_gb": round(mem, 1)})
    return procs


def pointer_info() -> dict[str, Any]:
    if not POINTER_FILE.exists():
        return {"status": "NO POINTER"}
    try:
        d = json.loads(POINTER_FILE.read_text())
        return {
            "cycle": d.get("cycle", "?"),
            "step": d.get("step", "?"),
            "version": d.get("checkpoint_version", "?"),
            "base_id": d.get("base_checkpoint_id", "?")[:32] + "..." if d.get("base_checkpoint_id") else "?",
            "saved_at": d.get("saved_at", "?"),
        }
    except Exception:
        return {"status": "ERROR"}


def dopamine_info() -> dict[str, Any]:
    if not DOPAMINE_FILE.exists():
        return {}
    try:
        text = DOPAMINE_FILE.read_text().strip()
        # Parse key=value pairs
        info = {}
        for part in text.split():
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    info[k] = round(float(v), 4)
                except ValueError:
                    info[k] = v
        return info
    except Exception:
        return {}


def blockchain_info() -> dict[str, Any]:
    info = {
        "blocks": 0,
        "head_hash": "—",
        "last_cycle": "—",
        "size_kb": 0,
    }
    if not BLOCKCHAIN_FILE.exists():
        info["status"] = "NO FILE (blockchain not recording)"
        return info
    try:
        lines = BLOCKCHAIN_FILE.read_text().strip().split("\n")
        info["blocks"] = len(lines)
        info["size_kb"] = round(BLOCKCHAIN_FILE.stat().st_size / 1024, 1)
        if lines:
            last = json.loads(lines[-1])
            info["head_hash"] = last.get("block_hash", "?")[:16] + "..."
            info["last_cycle"] = last.get("header", {}).get("cycle", "?")
        info["status"] = "ACTIVE"
    except Exception as e:
        info["status"] = f"ERROR: {e}"
    return info


def ledger_info() -> dict[str, Any]:
    info = {"lines": 0, "size_kb": 0, "head_hash": "—"}
    if not LEDGER_FILE.exists():
        info["status"] = "NO FILE"
        return info
    try:
        text = LEDGER_FILE.read_text().strip()
        if text:
            lines = text.split("\n")
            info["lines"] = len(lines)
            info["size_kb"] = round(LEDGER_FILE.stat().st_size / 1024, 1)
            last = json.loads(lines[-1])
            info["head_hash"] = last.get("event_hash", "?")[:16] + "..."
        info["status"] = "ACTIVE"
    except Exception as e:
        info["status"] = f"ERROR: {e}"
    return info


def file_hashes() -> dict[str, str]:
    ptr = pointer_info()
    cp_filename = ptr.get("path", "")
    cp_full = CKPT_ROOT / cp_filename if cp_filename else None
    return {
        "checkpoint.pt": sha256_file(cp_full) if cp_full and cp_full.exists() else "—",
        "organism_latest.json": sha256_file(POINTER_FILE),
        "blocks.jsonl": sha256_file(BLOCKCHAIN_FILE),
        "causal_events.jsonl": sha256_file(LEDGER_FILE),
        "config.yaml": sha256_file(ROOT / "src" / "configs" / "darwin_x_100m.yaml"),
        "start_script.ps1": sha256_file(ROOT / "src" / "scripts" / "start_100m_auto.ps1"),
    }


def render() -> None:
    clear_screen()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        🧬 F51 DARWIN-X 100M — REAL-TIME DASHBOARD          ║")
    print(f"║        {now}                              ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    # ── GPU ──
    print("║ 🖥️  GPU                                                      ║")
    for g in gpu_info():
        util_bar = "█" * (int(g["util"]) // 5) + "░" * (20 - int(g["util"]) // 5)
        print(
            f"║  GPU{g['idx']} {g['name'][:28]:28s} │ {util_bar} {g['util']:>3s}% │ "
            f"{g['mem_used']:>5s}/{g['mem_total']:>5s} MiB │ {g['temp']}°C ║"
        )
    if not gpu_info():
        print("║  (nvidia-smi unavailable)                                    ║")

    # ── Python processes ──
    procs = python_processes()
    if procs:
        print(f"║ 🐍 {len(procs)} Python processes (total: {sum(p['mem_gb'] for p in procs):.1f} GB)                              ║")

    print("╠══════════════════════════════════════════════════════════════╣")

    # ── Training ──
    print("║ 🏃 TRAINING                                                  ║")
    dop = dopamine_info()
    ptr = pointer_info()
    if dop:
        step = dop.get("step", "?")
        loss = dop.get("loss", "?")
        jepa = dop.get("jepa_raw", "?")
        dope = dop.get("dope", "?")
        delta = dop.get("delta", "?")
        gmc = dop.get("gmc", "?")
        print(f"║  step={step}  loss={loss}  jepa={jepa}  dopamine={dope}  ║")
        print(f"║  delta_loss={delta}  gmc={gmc}                             ║")
    else:
        print("║  (no dopamine status — training may be stopped)              ║")

    print(f"║  checkpoint: cycle={ptr.get('cycle','?')} step={ptr.get('step','?')} v{ptr.get('version','?')}  "
          f"saved={ptr.get('saved_at','?')} ║")
    print(f"║  base_id: {ptr.get('base_id','?')} ║")

    print("╠══════════════════════════════════════════════════════════════╣")

    # ── Blockchain ──
    print("║ ⛓️  BLOCKCHAIN                                                ║")
    bc = blockchain_info()
    print(f"║  blocks: {bc['blocks']:>6d}  │  head: {bc['head_hash']}  │  {bc['size_kb']} KB  │  "
          f"cycle={bc['last_cycle']} ║")
    print(f"║  status: {bc.get('status', '?')}")

    # ── SHA-256 hashes ──
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║ 🔐 SHA-256 FINGERPRINTS                                      ║")
    hashes = file_hashes()
    for name, h in hashes.items():
        if h != "—":
            print(f"║  {name:30s}  {h} ║")
        else:
            print(f"║  {name:30s}  (file not found)                              ║")

    # ── Organ identities (from latest checkpoint pointer) ──
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║ 🧬 ORGAN IDENTITIES (latest checkpoint)                      ║")
    try:
        import torch

        latest = json.loads(POINTER_FILE.read_text())
        cp_path = CKPT_ROOT / latest["path"]
        if cp_path.exists():
            payload = torch.load(cp_path, map_location="cpu", weights_only=False)
            oir = payload.get("organ_identity_report", {})
            identities = oir.get("identities", {})
            for name, ident in sorted(identities.items()):
                short = ident[-20:] if len(ident) > 20 else ident
                print(f"║  organ:{name:22s} ...{short} ║")
            if not identities:
                print("║  (no organ identities in checkpoint)                        ║")
        else:
            print("║  (checkpoint file not found)                                ║")
    except Exception as e:
        print(f"║  ERROR: {str(e)[:58]}")

    print("╠══════════════════════════════════════════════════════════════╣")
    print("║ 📋 LEDGER                                                    ║")
    ld = ledger_info()
    print(f"║  events: {ld['lines']:>6d}  │  head: {ld['head_hash']}  │  {ld['size_kb']} KB ║")
    print(f"║  status: {ld.get('status', '?')}")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("  Ctrl+C to exit  │  Refresh: 2s  │  dashboard v1")
    print(f"  config SHA-256: {hashes.get('config.yaml', '?')}")


def main() -> None:
    try:
        while True:
            render()
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")


if __name__ == "__main__":
    main()
