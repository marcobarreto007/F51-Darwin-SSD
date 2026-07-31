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
class CloudTokenVerifyPlan:
    remote_host: str
    remote_port: int
    remote_workspace: str
    batch_name: str
    remote_token_manifest: str
    remote_token_bin: str
    command: list[str]


@dataclass(frozen=True)
class CloudTokenVerifyResult:
    ok: bool
    dry_run: bool
    plan: CloudTokenVerifyPlan
    executed: list[list[str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _remote_token_verify_script(*, workspace: str, batch_name: str, vocab_label: str) -> str:
    q_workspace = shlex.quote(workspace.rstrip("/"))
    q_batch = shlex.quote(batch_name)
    q_vocab_label = shlex.quote(vocab_label)
    return f"""set -euo pipefail
WORKSPACE={q_workspace}
BATCH={q_batch}
VOCAB_LABEL={q_vocab_label}
TOKEN_DIR="$WORKSPACE/tokens/$VOCAB_LABEL"
TOKEN_BIN="$TOKEN_DIR/$BATCH.int32.bin"
TOKEN_MANIFEST="$TOKEN_DIR/$BATCH.tokens.json"
test -f "$TOKEN_BIN"
test -f "$TOKEN_MANIFEST"
python3 - "$TOKEN_BIN" "$TOKEN_MANIFEST" "$BATCH" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

token_bin = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
batch = sys.argv[3]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
errors = []

if manifest.get("batch") != batch:
    errors.append(f"batch mismatch: manifest={{manifest.get('batch')}} expected={{batch}}")
if manifest.get("dtype") != "int32-le":
    errors.append(f"dtype mismatch: {{manifest.get('dtype')}}")

tokens = int(manifest.get("tokens", -1))
if tokens < 2:
    errors.append(f"invalid token count: {{tokens}}")

size = token_bin.stat().st_size
if size % 4 != 0:
    errors.append(f"token bin size is not int32 aligned: {{size}}")
if tokens >= 0 and size != tokens * 4:
    errors.append(f"token bin size mismatch: size={{size}} tokens={{tokens}} expected={{tokens * 4}}")

hasher = hashlib.sha256()
with token_bin.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        hasher.update(chunk)
digest = hasher.hexdigest()
if digest != manifest.get("output_sha256"):
    errors.append("token bin sha256 mismatch")

if str(manifest.get("output_bin")) != str(token_bin):
    errors.append(f"output_bin mismatch: {{manifest.get('output_bin')}} expected={{token_bin}}")

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(5)

print(json.dumps({{
    "status": "TOKEN_VERIFY_OK",
    "batch": batch,
    "tokens": tokens,
    "bytes": size,
    "sha256": digest,
}}, sort_keys=True))
PY
"""


def build_cloud_token_verify_plan(
    *,
    ssh_host: str,
    remote_workspace: str,
    batch_name: str,
    vocab_label: str = "80k",
) -> CloudTokenVerifyPlan:
    target, port = _split_ssh_target(ssh_host)
    workspace = remote_workspace.rstrip("/")
    safe_batch = _safe_batch_name(batch_name)
    script = _remote_token_verify_script(
        workspace=workspace,
        batch_name=safe_batch,
        vocab_label=vocab_label,
    )
    command = ["ssh", "-p", str(port), target, "bash", "-lc", script]
    return CloudTokenVerifyPlan(
        remote_host=target,
        remote_port=port,
        remote_workspace=workspace,
        batch_name=safe_batch,
        remote_token_manifest=f"{workspace}/tokens/{vocab_label}/{safe_batch}.tokens.json",
        remote_token_bin=f"{workspace}/tokens/{vocab_label}/{safe_batch}.int32.bin",
        command=command,
    )


def verify_remote_token_batch(
    *,
    ssh_host: str,
    remote_workspace: str,
    batch_name: str,
    vocab_label: str = "80k",
    dry_run: bool = False,
    runner: CommandRunner | None = None,
) -> CloudTokenVerifyResult:
    plan = build_cloud_token_verify_plan(
        ssh_host=ssh_host,
        remote_workspace=remote_workspace,
        batch_name=batch_name,
        vocab_label=vocab_label,
    )
    if dry_run:
        return CloudTokenVerifyResult(ok=True, dry_run=True, plan=plan)

    run = runner or (lambda command: subprocess.run(command, check=False, text=True, capture_output=True))
    completed = run(plan.command)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() if completed.stderr else ""
        stdout = completed.stdout.strip() if completed.stdout else ""
        return CloudTokenVerifyResult(
            ok=False,
            dry_run=False,
            plan=plan,
            executed=[plan.command],
            errors=[stderr or stdout or "remote token verification failed"],
        )
    return CloudTokenVerifyResult(ok=True, dry_run=False, plan=plan, executed=[plan.command])
