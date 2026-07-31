from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Protocol

from f51_darwin.corpus_pack_upload import _split_ssh_target


class CommandRunner(Protocol):
    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        ...


@dataclass(frozen=True)
class CloudTokenIndexPlan:
    remote_host: str
    remote_port: int
    remote_workspace: str
    vocab_label: str
    remote_token_dir: str
    remote_index_path: str
    force: bool
    command: list[str]


@dataclass(frozen=True)
class CloudTokenIndexResult:
    ok: bool
    dry_run: bool
    plan: CloudTokenIndexPlan
    executed: list[list[str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _remote_token_index_script(*, workspace: str, vocab_label: str, force: bool) -> str:
    q_workspace = shlex.quote(workspace.rstrip("/"))
    q_vocab_label = shlex.quote(vocab_label)
    force_value = "1" if force else "0"
    return f"""set -euo pipefail
WORKSPACE={q_workspace}
VOCAB_LABEL={q_vocab_label}
FORCE={force_value}
TOKEN_DIR="$WORKSPACE/tokens/$VOCAB_LABEL"
INDEX_PATH="$TOKEN_DIR/index.json"
TMP_INDEX="$INDEX_PATH.tmp"
test -d "$TOKEN_DIR"
if [ -e "$INDEX_PATH" ] && [ "$FORCE" != "1" ]; then
  echo "token index already exists: $INDEX_PATH" >&2
  exit 2
fi
python3 - "$TOKEN_DIR" "$TMP_INDEX" "$VOCAB_LABEL" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

token_dir = Path(sys.argv[1])
tmp_index = Path(sys.argv[2])
vocab_label = sys.argv[3]
records = []
errors = []

for manifest_path in sorted(token_dir.glob("*.tokens.json")):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    batch = str(manifest.get("batch", ""))
    output_bin = Path(str(manifest.get("output_bin", "")))
    if not batch:
        errors.append(f"{{manifest_path}}: missing batch")
        continue
    if output_bin.name != f"{{batch}}.int32.bin":
        errors.append(f"{{manifest_path}}: output bin name mismatch")
        continue
    if not output_bin.exists():
        errors.append(f"{{manifest_path}}: output bin missing: {{output_bin}}")
        continue
    tokens = int(manifest.get("tokens", -1))
    size = output_bin.stat().st_size
    if tokens < 2:
        errors.append(f"{{manifest_path}}: invalid token count {{tokens}}")
    if size % 4 != 0:
        errors.append(f"{{manifest_path}}: non-int32-aligned size {{size}}")
    if size != tokens * 4:
        errors.append(f"{{manifest_path}}: size mismatch size={{size}} tokens={{tokens}}")
    hasher = hashlib.sha256()
    with output_bin.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != manifest.get("output_sha256"):
        errors.append(f"{{manifest_path}}: sha256 mismatch")
    records.append({{
        "batch": batch,
        "bin": str(output_bin),
        "manifest": str(manifest_path),
        "tokens": tokens,
        "bytes": size,
        "sha256": digest,
        "documents": int(manifest.get("documents", 0)),
        "vocab_size": int(manifest.get("vocab_size", 0)),
        "tokenizer_dir": str(manifest.get("tokenizer_dir", "")),
    }})

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(5)
if not records:
    raise SystemExit("no token manifests found")

index = {{
    "version": 1,
    "vocab_label": vocab_label,
    "batches": records,
    "batch_count": len(records),
    "total_tokens": sum(record["tokens"] for record in records),
    "total_bytes": sum(record["bytes"] for record in records),
}}
tmp_index.write_text(json.dumps(index, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps({{
    "status": "TOKEN_INDEX_OK",
    "batch_count": index["batch_count"],
    "total_tokens": index["total_tokens"],
    "total_bytes": index["total_bytes"],
}}, sort_keys=True))
PY
mv "$TMP_INDEX" "$INDEX_PATH"
printf 'TOKEN_INDEX_WRITTEN path=%s\\n' "$INDEX_PATH"
"""


def build_cloud_token_index_plan(
    *,
    ssh_host: str,
    remote_workspace: str,
    vocab_label: str = "80k",
    force: bool = False,
) -> CloudTokenIndexPlan:
    target, port = _split_ssh_target(ssh_host)
    workspace = remote_workspace.rstrip("/")
    script = _remote_token_index_script(
        workspace=workspace,
        vocab_label=vocab_label,
        force=force,
    )
    command = ["ssh", "-p", str(port), target, "bash", "-lc", script]
    return CloudTokenIndexPlan(
        remote_host=target,
        remote_port=port,
        remote_workspace=workspace,
        vocab_label=vocab_label,
        remote_token_dir=f"{workspace}/tokens/{vocab_label}",
        remote_index_path=f"{workspace}/tokens/{vocab_label}/index.json",
        force=force,
        command=command,
    )


def build_remote_token_index(
    *,
    ssh_host: str,
    remote_workspace: str,
    vocab_label: str = "80k",
    force: bool = False,
    dry_run: bool = False,
    runner: CommandRunner | None = None,
) -> CloudTokenIndexResult:
    plan = build_cloud_token_index_plan(
        ssh_host=ssh_host,
        remote_workspace=remote_workspace,
        vocab_label=vocab_label,
        force=force,
    )
    if dry_run:
        return CloudTokenIndexResult(ok=True, dry_run=True, plan=plan)

    run = runner or (lambda command: subprocess.run(command, check=False, text=True, capture_output=True))
    completed = run(plan.command)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() if completed.stderr else ""
        stdout = completed.stdout.strip() if completed.stdout else ""
        return CloudTokenIndexResult(
            ok=False,
            dry_run=False,
            plan=plan,
            executed=[plan.command],
            errors=[stderr or stdout or "remote token index command failed"],
        )
    return CloudTokenIndexResult(ok=True, dry_run=False, plan=plan, executed=[plan.command])
