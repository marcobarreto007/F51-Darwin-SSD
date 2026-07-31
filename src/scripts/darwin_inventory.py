from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

from f51_darwin.artifacts import (
    TOKEN_BIN_AUDIT_CANDIDATES,
    audit_checkpoint_files,
    audit_checkpoint_pointers,
    audit_token_bins,
)
from f51_darwin.dataset_layout import WorkspacePaths


INVENTORY_EXCLUDED_DIRS = {
    ".git",
    ".venv_clean",
    ".venv_nitro",
    "__pycache__",
    ".playwright-mcp",
    "workspace",
}


def visible_project_files(root: Path, *, limit: int = 200) -> list[str]:
    files: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in INVENTORY_EXCLUDED_DIRS for part in relative.parts):
            continue
        files.append(str(relative))
    return sorted(files)[:limit]


def run(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except FileNotFoundError as exc:
        return {"command": command, "returncode": 127, "stdout": "", "stderr": str(exc)}


def main() -> int:
    workspace = WorkspacePaths.from_project(ROOT)
    dataset_root = workspace.root
    feast_token = workspace.feast_token_bin
    checkpoint_root = workspace.checkpoints
    files = visible_project_files(ROOT)
    checks = {
        name: {
            "exists": (ROOT / name).exists(),
            "kind": "dir" if (ROOT / name).is_dir() else "file" if (ROOT / name).is_file() else "missing",
        }
        for name in [
            "pyproject.toml",
            "requirements.txt",
            "src",
            "src/f51_darwin",
            "src/scripts",
            "src/tests",
            "governance/docs",
            "governance/audit",
            "research",
            "workspace",
        ]
    }
    torch_state: dict[str, object]
    try:
        import torch

        torch_state = {
            "present": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        torch_state = {"present": False, "error": repr(exc)}
    payload = {
        "root": str(ROOT),
        "workspace_root": str(dataset_root),
        "checks": checks,
        "canonical_token_bins": [
            item.to_dict()
            for item in audit_token_bins(ROOT, [feast_token])
        ],
        "legacy_local_token_bins": [
            item.to_dict()
            for item in audit_token_bins(ROOT, TOKEN_BIN_AUDIT_CANDIDATES)
            if item.exists
        ],
        "selected_token_bin": str(feast_token) if feast_token.is_file() else None,
        "checkpoint_pointers": [
            item.to_dict()
            for item in audit_checkpoint_pointers(
                ROOT,
                [
                    checkpoint_root / "organism_latest.json",
                    checkpoint_root / "latest.json",
                ],
            )
        ],
        "checkpoint_files": [
            item.to_dict()
            for item in audit_checkpoint_files(
                ROOT,
                [checkpoint_root],
            )
        ],
        "files_first_200": files,
        "git_status": run(["git", "status", "--short"]),
        "git_log": run(["git", "log", "--oneline", "-5"]),
        "python": sys.version,
        "torch": torch_state,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
