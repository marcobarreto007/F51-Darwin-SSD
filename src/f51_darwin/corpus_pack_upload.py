from __future__ import annotations

import hashlib
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from f51_darwin.corpus_pack_verifier import verify_corpus_pack


class CommandRunner(Protocol):
    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        ...


@dataclass(frozen=True)
class UploadPlan:
    pack_path: str
    pack_sha256: str
    remote_host: str
    remote_port: int
    remote_dir: str
    remote_pack_path: str
    remote_sha_path: str
    commands: list[list[str]]


@dataclass(frozen=True)
class UploadResult:
    ok: bool
    dry_run: bool
    plan: UploadPlan
    executed: list[list[str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _split_ssh_target(ssh_host: str) -> tuple[str, int]:
    parts = shlex.split(ssh_host)
    if not parts or parts[0] != "ssh":
        raise ValueError("ssh_host must look like: ssh -p PORT root@HOST")
    port = 22
    target = ""
    index = 1
    while index < len(parts):
        part = parts[index]
        if part == "-p":
            index += 1
            port = int(parts[index])
        elif not part.startswith("-"):
            target = part
        index += 1
    if not target:
        raise ValueError("ssh_host missing root@host target")
    return target, port


def build_upload_plan(
    *,
    pack_path: Path,
    ssh_host: str,
    remote_workspace: str,
    batch_name: str | None = None,
) -> UploadPlan:
    pack_path = pack_path.resolve()
    if not pack_path.exists():
        raise FileNotFoundError(f"pack not found: {pack_path}")
    target, port = _split_ssh_target(ssh_host)
    digest = file_sha256(pack_path)
    safe_batch = batch_name or pack_path.parent.name or pack_path.stem
    safe_batch = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in safe_batch)
    remote_dir = f"{remote_workspace.rstrip('/')}/incoming/{safe_batch}"
    remote_pack_path = f"{remote_dir}/{pack_path.name}"
    remote_sha_path = f"{remote_pack_path}.sha256"
    quoted_dir = shlex.quote(remote_dir)
    quoted_pack = shlex.quote(remote_pack_path)
    quoted_sha = shlex.quote(remote_sha_path)
    commands = [
        ["ssh", "-p", str(port), target, f"mkdir -p {quoted_dir}"],
        ["scp", "-P", str(port), str(pack_path), f"{target}:{remote_pack_path}"],
        [
            "ssh",
            "-p",
            str(port),
            target,
            f"printf '%s  %s\\n' {shlex.quote(digest)} {quoted_pack} > {quoted_sha}",
        ],
        ["ssh", "-p", str(port), target, f"sha256sum -c {quoted_sha}"],
    ]
    return UploadPlan(
        pack_path=str(pack_path),
        pack_sha256=digest,
        remote_host=target,
        remote_port=port,
        remote_dir=remote_dir,
        remote_pack_path=remote_pack_path,
        remote_sha_path=remote_sha_path,
        commands=commands,
    )


def upload_verified_pack(
    *,
    pack_path: Path,
    ssh_host: str,
    remote_workspace: str,
    batch_name: str | None = None,
    dry_run: bool = False,
    runner: CommandRunner | None = None,
) -> UploadResult:
    report = verify_corpus_pack(pack_path)
    if not report.ok:
        plan = build_upload_plan(
            pack_path=pack_path,
            ssh_host=ssh_host,
            remote_workspace=remote_workspace,
            batch_name=batch_name,
        )
        return UploadResult(
            ok=False,
            dry_run=dry_run,
            plan=plan,
            errors=[f"pack verification failed: {error}" for error in report.errors],
        )

    plan = build_upload_plan(
        pack_path=pack_path,
        ssh_host=ssh_host,
        remote_workspace=remote_workspace,
        batch_name=batch_name,
    )
    if dry_run:
        return UploadResult(ok=True, dry_run=True, plan=plan)

    executed: list[list[str]] = []
    run = runner or (lambda command: subprocess.run(command, check=False, text=True, capture_output=True))
    errors: list[str] = []
    for command in plan.commands:
        executed.append(command)
        completed = run(command)
        if completed.returncode != 0:
            stderr = completed.stderr.strip() if completed.stderr else ""
            stdout = completed.stdout.strip() if completed.stdout else ""
            errors.append(stderr or stdout or f"command failed: {' '.join(command)}")
            break
    return UploadResult(
        ok=not errors,
        dry_run=False,
        plan=plan,
        executed=executed,
        errors=errors,
    )
