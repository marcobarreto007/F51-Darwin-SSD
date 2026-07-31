from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from f51_darwin.tokenizer import F51BPETokenizer


def run(command: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False, timeout=timeout)


def load_config(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def split_ssh_target(ssh_host: str) -> tuple[str, str]:
    parts = ssh_host.split()
    port = "22"
    target = ""
    index = 0
    while index < len(parts):
        part = parts[index]
        if part == "-p":
            index += 1
            port = parts[index]
        elif "@" in part:
            target = part
        index += 1
    if not target:
        raise ValueError(f"cannot parse ssh target from {ssh_host!r}")
    return target, port


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def tokenize_batch(
    *,
    batch_name: str,
    approved_dir: Path,
    tokenizer_dir: Path,
    output_bin: Path,
    output_manifest: Path,
    remote_bin_path: str,
    remote_tokenizer_dir: str,
) -> dict[str, object]:
    tokenizer = F51BPETokenizer.load(tokenizer_dir)
    files = sorted(approved_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no approved .txt files in {approved_dir}")

    output_bin.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    documents = 0
    tokens = 0
    with output_bin.open("wb") as handle:
        buffer: list[int] = []
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
            encoded = tokenizer.encode(text, add_eos=True)
            if not encoded:
                continue
            for token_id in encoded:
                if token_id < 0 or token_id >= tokenizer.vocab_size:
                    raise ValueError(f"token id out of vocab: {token_id}")
            documents += 1
            tokens += len(encoded)
            buffer.extend(encoded)
            if len(buffer) >= 1_000_000:
                handle.write(struct.pack(f"<{len(buffer)}i", *buffer))
                buffer.clear()
        if buffer:
            handle.write(struct.pack(f"<{len(buffer)}i", *buffer))

    if tokens < 2:
        raise ValueError("tokenized batch is too small")

    digest = sha256_file(output_bin)
    manifest = {
        "batch": batch_name,
        "documents": documents,
        "tokens": tokens,
        "dtype": "int32-le",
        "tokenizer_dir": remote_tokenizer_dir,
        "tokenizer_name": tokenizer.metadata.name,
        "vocab_size": tokenizer.vocab_size,
        "input_dir": str(approved_dir),
        "input_manifest": "",
        "output_bin": remote_bin_path,
        "output_sha256": digest,
    }
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tokenize a local approved batch and upload tokens to NUCLEO.")
    parser.add_argument("--batch-name", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "corpus_factory.yaml"))
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--target", choices=["master", "worker"], default="master")
    parser.add_argument("--vocab-label", default="80k")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(Path(args.config))
    local_workspace = Path(args.workspace or str(config["local_workspace"]))
    target_config = config["cloud"][args.target]
    remote_workspace = str(target_config["workspace"]).rstrip("/")
    remote_target, remote_port = split_ssh_target(str(target_config["ssh"]))

    batch_dir = local_workspace / "batches" / args.batch_name
    approved_dir = batch_dir / "pack" / "approved"
    tokenizer_dir = Path(args.tokenizer_dir) if args.tokenizer_dir else local_workspace / "tokenizer" / "f51_bpe_80k"
    local_token_dir = local_workspace / "tokens" / args.vocab_label
    output_bin = local_token_dir / f"{args.batch_name}.int32.bin"
    output_manifest = local_token_dir / f"{args.batch_name}.tokens.json"
    remote_token_dir = f"{remote_workspace}/tokens/{args.vocab_label}"
    remote_bin = f"{remote_token_dir}/{args.batch_name}.int32.bin"
    remote_manifest = f"{remote_token_dir}/{args.batch_name}.tokens.json"
    remote_tokenizer = f"{remote_workspace}/tokenizer/f51_bpe_{args.vocab_label}"

    if not approved_dir.exists():
        raise FileNotFoundError(f"approved dir not found: {approved_dir}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"tokenizer dir not found: {tokenizer_dir}")

    plan = {
        "batch": args.batch_name,
        "local_bin": str(output_bin),
        "local_manifest": str(output_manifest),
        "remote_bin": remote_bin,
        "remote_manifest": remote_manifest,
        "dry_run": args.dry_run,
    }

    if args.dry_run:
        print(json.dumps({"ok": True, "plan": plan}, indent=2, sort_keys=True))
        return 0

    if not args.force and (output_bin.exists() or output_manifest.exists()):
        raise FileExistsError(f"local token output exists for batch {args.batch_name}; use --force")
    if args.force:
        output_bin.unlink(missing_ok=True)
        output_manifest.unlink(missing_ok=True)

    manifest = tokenize_batch(
        batch_name=args.batch_name,
        approved_dir=approved_dir,
        tokenizer_dir=tokenizer_dir,
        output_bin=output_bin,
        output_manifest=output_manifest,
        remote_bin_path=remote_bin,
        remote_tokenizer_dir=remote_tokenizer,
    )

    plan.update(
        {
            "documents": manifest["documents"],
            "tokens": manifest["tokens"],
            "bytes": output_bin.stat().st_size,
            "sha256": manifest["output_sha256"],
        }
    )

    check = run(
        [
            "ssh",
            "-p",
            remote_port,
            remote_target,
            f"test ! -e {remote_bin!r} && test ! -e {remote_manifest!r} && mkdir -p {remote_token_dir!r}",
        ],
        timeout=60,
    )
    if check.returncode != 0 and not args.force:
        print(check.stderr or check.stdout, file=sys.stderr)
        return check.returncode or 1

    for local, remote in ((output_bin, remote_bin), (output_manifest, remote_manifest)):
        completed = run(["scp", "-P", remote_port, str(local), f"{remote_target}:{remote}"], timeout=600)
        if completed.returncode != 0:
            print(completed.stderr or completed.stdout, file=sys.stderr)
            return completed.returncode or 1

    print(json.dumps({"ok": True, "plan": plan}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
