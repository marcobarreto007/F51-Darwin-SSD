#!/usr/bin/env python3
"""F51 auto pipeline: tokenize, train, monitor, grow.

This script is intentionally conservative. It reuses an existing valid token bin
before attempting expensive tokenization, and it fails with an actionable error
if no tokenizer entrypoint exists.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from f51_darwin.artifacts import resolve_token_bin

CORPUS = ROOT / "data" / "corpus"


def step(message: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {message}")
    print(f"{'=' * 60}")


def tokenize() -> bool:
    """Step 1: ensure a reusable token binary exists."""

    existing_tokens = resolve_token_bin(ROOT, min_bytes=100_000_000)
    if existing_tokens is not None:
        print(f"  {existing_tokens.name} ja existe ({existing_tokens.stat().st_size / 1e9:.1f} GB)")
        return True

    step("1/4 TOKENIZANDO")
    tokenizer_script = ROOT / "tokenize_cloud.py"
    if not tokenizer_script.exists():
        tokenizer_script = ROOT / "tokenize_fast.py"
    if not tokenizer_script.exists():
        print("  ERRO: nenhum token bin valido e nenhum script de tokenizacao encontrado.")
        print("  Esperado: data/tokens.bin, data/tokens_full.bin ou data/tokens_chunk_aa.")
        return False

    result = subprocess.run(
        [sys.executable, str(tokenizer_script)],
        cwd=str(ROOT),
        check=False,
    )
    return result.returncode == 0


def train() -> subprocess.Popen:
    """Step 2: start training in the background."""

    step("2/4 TREINANDO DARWIN-X 600M")
    train_script = ROOT / "research" / "train_darwin_x.py"
    token_bin = resolve_token_bin(ROOT, min_bytes=4)
    if token_bin is None:
        raise FileNotFoundError("No token bin found for Darwin-X training.")

    command = [
        sys.executable,
        "-u",
        str(train_script),
        "--config",
        str(ROOT / "src" / "configs" / "darwin_x_600m.yaml"),
        "--token-bin",
        str(token_bin),
        "--tokenizer",
        str(ROOT / "tokenizer" / "f51_bpe"),
    ]

    log = ROOT / "pipeline_train.log"
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
    print(f"  PID: {process.pid} | Log: {log}")
    return process


def monitor(process: subprocess.Popen, check_every: int = 300) -> None:
    """Step 3: monitor training progress."""

    step("3/4 MONITORANDO")
    checkpoint_dir = ROOT / "checkpoints" / "nanfree"
    last_count = len(list(checkpoint_dir.glob("step_*.pt"))) if checkpoint_dir.exists() else 0

    while process.poll() is None:
        time.sleep(check_every)
        current = len(list(checkpoint_dir.glob("step_*.pt"))) if checkpoint_dir.exists() else 0
        if current > last_count:
            print(f"  +{current - last_count} checkpoints novos ({current} total)")
            last_count = current
            continue

        log = ROOT / "pipeline_train.log"
        if log.exists():
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[-3:]:
                if "loss=" in line or "step=" in line:
                    print(f"  {line.strip()[:120]}")


def grow() -> None:
    """Step 4: report grow trigger."""

    step("4/4 CRESCENDO A FAMILIA DARWIN-X")
    print("  Aguardando 600M estabilizar com perda finita e benchmark registrado.")
    print("  Proximo salto deve manter tokenizer, órgãos e contrato de checkpoint.")
    print("  Manual: python research/train_darwin_x.py --config src/configs/darwin_x_600m.yaml")


def main() -> int:
    print("F51 AUTO PIPELINE")
    print(f"  Corpus: {CORPUS}")
    token_bin = resolve_token_bin(ROOT, min_bytes=4)
    print(f"  Tokens: {token_bin if token_bin else 'missing'}")

    if not tokenize():
        print("Tokenizacao falhou")
        return 1

    process = train()
    try:
        monitor(process)
    except KeyboardInterrupt:
        print("\n  Pipeline pausado. Treino continua em background.")

    grow()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
