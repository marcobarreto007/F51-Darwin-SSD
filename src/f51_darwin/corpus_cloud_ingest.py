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
class CloudIngestPlan:
    remote_host: str
    remote_port: int
    remote_pack_path: str
    remote_workspace: str
    batch_name: str
    remote_normalized_dir: str
    remote_manifest_path: str
    command: list[str]


@dataclass(frozen=True)
class CloudIngestResult:
    ok: bool
    dry_run: bool
    plan: CloudIngestPlan
    executed: list[list[str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _remote_ingest_script(*, pack_path: str, workspace: str, batch_name: str) -> str:
    q_pack = shlex.quote(pack_path)
    q_workspace = shlex.quote(workspace.rstrip("/"))
    q_batch = shlex.quote(batch_name)
    return f"""set -euo pipefail
PACK={q_pack}
WORKSPACE={q_workspace}
BATCH={q_batch}
SHA_FILE="$PACK.sha256"
NORMALIZED_DIR="$WORKSPACE/normalized/$BATCH"
MANIFEST_DIR="$WORKSPACE/manifests"
MANIFEST_PATH="$MANIFEST_DIR/$BATCH.jsonl"
TMP_DIR="$(mktemp -d)"
cleanup() {{ rm -rf "$TMP_DIR"; }}
trap cleanup EXIT
test -f "$PACK"
test -f "$SHA_FILE"
sha256sum -c "$SHA_FILE"
if [ -e "$NORMALIZED_DIR" ] || [ -e "$MANIFEST_PATH" ]; then
  echo "batch already ingested: $BATCH" >&2
  exit 2
fi
mkdir -p "$WORKSPACE/normalized" "$MANIFEST_DIR"
tar -xzf "$PACK" -C "$TMP_DIR"
test -f "$TMP_DIR/manifest.jsonl"
if [ ! -d "$TMP_DIR/approved" ]; then
  echo "pack has no approved directory" >&2
  exit 3
fi
mkdir "$NORMALIZED_DIR"
find "$TMP_DIR/approved" -maxdepth 1 -type f -name '*.txt' -print0 | xargs -0 -I{{}} cp "{{}}" "$NORMALIZED_DIR/"
cp "$TMP_DIR/manifest.jsonl" "$MANIFEST_PATH"
APPROVED_COUNT="$(find "$NORMALIZED_DIR" -maxdepth 1 -type f -name '*.txt' | wc -l)"
MANIFEST_APPROVED="$(python3 -c 'import json, sys; print(sum(1 for line in open(sys.argv[1], encoding="utf-8") if line.strip() and json.loads(line).get("decision") == "approve"))' "$MANIFEST_PATH")"
if [ "$APPROVED_COUNT" -ne "$MANIFEST_APPROVED" ]; then
  echo "approved count mismatch: files=$APPROVED_COUNT manifest=$MANIFEST_APPROVED" >&2
  exit 4
fi
printf 'INGEST_OK batch=%s approved=%s manifest=%s normalized=%s\\n' "$BATCH" "$APPROVED_COUNT" "$MANIFEST_PATH" "$NORMALIZED_DIR"
"""


def build_cloud_ingest_plan(
    *,
    ssh_host: str,
    remote_pack_path: str,
    remote_workspace: str,
    batch_name: str,
) -> CloudIngestPlan:
    target, port = _split_ssh_target(ssh_host)
    safe_batch = _safe_batch_name(batch_name)
    workspace = remote_workspace.rstrip("/")
    script = _remote_ingest_script(
        pack_path=remote_pack_path,
        workspace=workspace,
        batch_name=safe_batch,
    )
    command = ["ssh", "-p", str(port), target, "bash", "-lc", script]
    return CloudIngestPlan(
        remote_host=target,
        remote_port=port,
        remote_pack_path=remote_pack_path,
        remote_workspace=workspace,
        batch_name=safe_batch,
        remote_normalized_dir=f"{workspace}/normalized/{safe_batch}",
        remote_manifest_path=f"{workspace}/manifests/{safe_batch}.jsonl",
        command=command,
    )


def ingest_remote_pack(
    *,
    ssh_host: str,
    remote_pack_path: str,
    remote_workspace: str,
    batch_name: str,
    dry_run: bool = False,
    runner: CommandRunner | None = None,
) -> CloudIngestResult:
    plan = build_cloud_ingest_plan(
        ssh_host=ssh_host,
        remote_pack_path=remote_pack_path,
        remote_workspace=remote_workspace,
        batch_name=batch_name,
    )
    if dry_run:
        return CloudIngestResult(ok=True, dry_run=True, plan=plan)

    run = runner or (lambda command: subprocess.run(command, check=False, text=True, capture_output=True))
    completed = run(plan.command)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() if completed.stderr else ""
        stdout = completed.stdout.strip() if completed.stdout else ""
        return CloudIngestResult(
            ok=False,
            dry_run=False,
            plan=plan,
            executed=[plan.command],
            errors=[stderr or stdout or "remote ingest command failed"],
        )
    return CloudIngestResult(ok=True, dry_run=False, plan=plan, executed=[plan.command])
