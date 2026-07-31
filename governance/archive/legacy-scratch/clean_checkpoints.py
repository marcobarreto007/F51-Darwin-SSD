"""HISTORICAL ONLY: destructive checkpoint loop; do not execute."""

import time
from pathlib import Path


ckpt_dir = Path("checkpoints/organism")
while True:
    ckpts = sorted(
        ckpt_dir.glob("organism_cycle_*.pt"), key=lambda path: path.stat().st_mtime
    )
    if len(ckpts) > 2:
        for old in ckpts[:-2]:
            old.unlink()
            print(f"Deletado: {old.name}")
    time.sleep(300)
