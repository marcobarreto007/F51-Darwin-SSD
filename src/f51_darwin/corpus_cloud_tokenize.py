from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Protocol

from f51_darwin.corpus_cloud_common import safe_batch_name as _safe_batch_name
from f51_darwin.corpus_pack_upload import _split_ssh_target


class CommandRunner(Protocol):
    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        ...


@dataclass(frozen=True)
class CloudTokenizePlan:
    remote_host: str
    remote_port: int
    remote_workspace: str
    batch_name: str
    remote_normalized_dir: str
    remote_input_manifest: str
    remote_tokenizer_dir: str
    remote_token_bin: str
    remote_token_manifest: str
    command: list[str]


@dataclass(frozen=True)
class CloudTokenizeResult:
    ok: bool
    dry_run: bool
    plan: CloudTokenizePlan
    executed: list[list[str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _remote_tokenize_script(
    *,
    workspace: str,
    batch_name: str,
    tokenizer_dir: str,
    vocab_label: str,
) -> str:
    q_workspace = shlex.quote(workspace.rstrip("/"))
    q_batch = shlex.quote(batch_name)
    q_tokenizer = shlex.quote(tokenizer_dir.rstrip("/"))
    q_vocab_label = shlex.quote(vocab_label)
    return f"""set -euo pipefail
WORKSPACE={q_workspace}
BATCH={q_batch}
TOKENIZER_DIR={q_tokenizer}
VOCAB_LABEL={q_vocab_label}
INPUT_DIR="$WORKSPACE/normalized/$BATCH"
INPUT_MANIFEST="$WORKSPACE/manifests/$BATCH.jsonl"
TOKEN_DIR="$WORKSPACE/tokens/$VOCAB_LABEL"
OUT_BIN="$TOKEN_DIR/$BATCH.int32.bin"
OUT_MANIFEST="$TOKEN_DIR/$BATCH.tokens.json"
TMP_BIN="$OUT_BIN.tmp"
test -d "$INPUT_DIR"
test -f "$INPUT_MANIFEST"
test -d "$TOKENIZER_DIR"
if [ -e "$OUT_BIN" ] || [ -e "$OUT_MANIFEST" ] || [ -e "$TMP_BIN" ]; then
  echo "token output already exists for batch: $BATCH" >&2
  exit 2
fi
mkdir -p "$TOKEN_DIR"
PYTHONPATH="$WORKSPACE:${{PYTHONPATH:-}}" python3 - "$INPUT_DIR" "$INPUT_MANIFEST" "$TOKENIZER_DIR" "$TMP_BIN" "$OUT_MANIFEST" "$BATCH" <<'PY'
import hashlib
import json
import struct
import sys
from pathlib import Path

from f51_darwin.tokenizer import F51BPETokenizer

input_dir = Path(sys.argv[1])
input_manifest = Path(sys.argv[2])
tokenizer_dir = Path(sys.argv[3])
tmp_bin = Path(sys.argv[4])
out_manifest = Path(sys.argv[5])
batch = sys.argv[6]

tokenizer = F51BPETokenizer.load(tokenizer_dir)
files = sorted(input_dir.glob("*.txt"))
if not files:
    raise SystemExit("no approved text files found")

hasher = hashlib.sha256()
documents = 0
tokens = 0
with tmp_bin.open("wb") as handle:
    buffer = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        encoded = tokenizer.encode(text, add_eos=True)
        if not encoded:
            continue
        for token_id in encoded:
            if token_id < 0 or token_id >= tokenizer.vocab_size:
                raise SystemExit(f"token id out of vocab: {{token_id}}")
        documents += 1
        tokens += len(encoded)
        buffer.extend(encoded)
        if len(buffer) >= 1_000_000:
            payload = struct.pack(f"<{{len(buffer)}}i", *buffer)
            handle.write(payload)
            hasher.update(payload)
            buffer.clear()
    if buffer:
        payload = struct.pack(f"<{{len(buffer)}}i", *buffer)
        handle.write(payload)
        hasher.update(payload)

if tokens < 2:
    tmp_bin.unlink(missing_ok=True)
    raise SystemExit("tokenized batch is too small")

record = {{
    "batch": batch,
    "documents": documents,
    "tokens": tokens,
    "dtype": "int32-le",
    "tokenizer_dir": str(tokenizer_dir),
    "tokenizer_name": tokenizer.metadata.name,
    "vocab_size": tokenizer.vocab_size,
    "input_dir": str(input_dir),
    "input_manifest": str(input_manifest),
    "output_bin": str(tmp_bin.with_suffix("")),
    "output_sha256": hasher.hexdigest(),
}}
out_manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
mv "$TMP_BIN" "$OUT_BIN"
python3 -c 'import json, sys; p=sys.argv[1]; data=json.load(open(p, encoding="utf-8")); data["output_bin"]=sys.argv[2]; open(p, "w", encoding="utf-8").write(json.dumps(data, indent=2, sort_keys=True)+"\\n")' "$OUT_MANIFEST" "$OUT_BIN"
printf 'TOKENIZE_OK batch=%s bin=%s manifest=%s\\n' "$BATCH" "$OUT_BIN" "$OUT_MANIFEST"
"""


def build_cloud_tokenize_plan(
    *,
    ssh_host: str,
    remote_workspace: str,
    batch_name: str,
    tokenizer_dir: str | None = None,
    vocab_label: str = "80k",
) -> CloudTokenizePlan:
    target, port = _split_ssh_target(ssh_host)
    workspace = remote_workspace.rstrip("/")
    safe_batch = _safe_batch_name(batch_name)
    tokenizer = tokenizer_dir.rstrip("/") if tokenizer_dir else f"{workspace}/tokenizer/f51_bpe_{vocab_label}"
    script = _remote_tokenize_script(
        workspace=workspace,
        batch_name=safe_batch,
        tokenizer_dir=tokenizer,
        vocab_label=vocab_label,
    )
    command = ["ssh", "-p", str(port), target, "bash", "-lc", script]
    return CloudTokenizePlan(
        remote_host=target,
        remote_port=port,
        remote_workspace=workspace,
        batch_name=safe_batch,
        remote_normalized_dir=f"{workspace}/normalized/{safe_batch}",
        remote_input_manifest=f"{workspace}/manifests/{safe_batch}.jsonl",
        remote_tokenizer_dir=tokenizer,
        remote_token_bin=f"{workspace}/tokens/{vocab_label}/{safe_batch}.int32.bin",
        remote_token_manifest=f"{workspace}/tokens/{vocab_label}/{safe_batch}.tokens.json",
        command=command,
    )


def tokenize_remote_batch(
    *,
    ssh_host: str,
    remote_workspace: str,
    batch_name: str,
    tokenizer_dir: str | None = None,
    vocab_label: str = "80k",
    dry_run: bool = False,
    runner: CommandRunner | None = None,
) -> CloudTokenizeResult:
    plan = build_cloud_tokenize_plan(
        ssh_host=ssh_host,
        remote_workspace=remote_workspace,
        batch_name=batch_name,
        tokenizer_dir=tokenizer_dir,
        vocab_label=vocab_label,
    )
    if dry_run:
        return CloudTokenizeResult(ok=True, dry_run=True, plan=plan)

    run = runner or (lambda command: subprocess.run(command, check=False, text=True, capture_output=True))
    completed = run(plan.command)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() if completed.stderr else ""
        stdout = completed.stdout.strip() if completed.stdout else ""
        return CloudTokenizeResult(
            ok=False,
            dry_run=False,
            plan=plan,
            executed=[plan.command],
            errors=[stderr or stdout or "remote tokenize command failed"],
        )
    return CloudTokenizeResult(ok=True, dry_run=False, plan=plan, executed=[plan.command])
