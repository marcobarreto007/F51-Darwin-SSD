#!/usr/bin/env python3
"""Validate local Markdown links in the current, non-archived documentation."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
INLINE_CODE = re.compile(r"`([^`\n]+)`")
INTERNAL_PATH = re.compile(
    r"(?<![\w.-])((?:(?:src|governance)/(?:scripts|tools|research|f51_darwin|configs|tests|docs|audit|archive)|"
    r"research|scripts|tools|f51_darwin|configs|tests|docs|audit|archive)[\\/]"
    r"[A-Za-z0-9_.\\/-]+)"
)
EXCLUDED_PREFIXES = ("governance/archive/", "governance/docs/superpowers/", "governance/docs/_historico/")


def _markdown_files(root: Path) -> list[Path]:
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "*.md"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return [
                root / item
                for item in result.stdout.splitlines()
                if item
                and item.replace("\\", "/") != "DIARIO_DE_BORDO.md"
                and not item.replace("\\", "/").startswith(EXCLUDED_PREFIXES)
            ]
    return [
        path
        for path in root.rglob("*.md")
        if path.relative_to(root).as_posix() != "DIARIO_DE_BORDO.md"
        if not path.relative_to(root).as_posix().startswith(EXCLUDED_PREFIXES)
    ]


def check_links(root: Path) -> list[dict[str, object]]:
    root = root.resolve()
    findings: list[dict[str, object]] = []
    for document in _markdown_files(root):
        if not document.is_file():
            continue
        text = document.read_text(encoding="utf-8", errors="replace")
        for match in LINK.finditer(text):
            raw = match.group(1).strip()
            if raw.startswith("<") and ">" in raw:
                raw = raw[1 : raw.index(">")]
            else:
                raw = raw.split(maxsplit=1)[0]
            target = unquote(raw).split("#", 1)[0]
            if not target or re.match(r"^(?:https?://|mailto:|data:)", target, re.I):
                continue
            if re.match(r"^[A-Za-z]:[/\\]", target) or target.startswith(("/", "\\\\")):
                findings.append(
                    {
                        "type": "absolute_local_link",
                        "document": document.relative_to(root).as_posix(),
                        "line": text.count("\n", 0, match.start()) + 1,
                        "target": target,
                    }
                )
                continue
            resolved = (document.parent / target).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                exists = False
            else:
                exists = resolved.exists()
            if not exists:
                findings.append(
                    {
                        "type": "broken_link",
                        "document": document.relative_to(root).as_posix(),
                        "line": text.count("\n", 0, match.start()) + 1,
                        "target": target,
                    }
                )
        for code in INLINE_CODE.finditer(text):
            for match in INTERNAL_PATH.finditer(code.group(1)):
                target = match.group(1).replace("\\", "/").rstrip(".,:;")
                target = re.sub(r":\d+(?:-\d+)?$", "", target)
                if "*" in target or (root / target).exists():
                    continue
                findings.append(
                    {
                        "type": "broken_inline_path",
                        "document": document.relative_to(root).as_posix(),
                        "line": text.count("\n", 0, code.start()) + 1,
                        "target": target,
                    }
                )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    findings = check_links(args.root)
    print(json.dumps({"status": "pass" if not findings else "fail", "findings": findings}, indent=2))
    return 0 if not findings else 2


if __name__ == "__main__":
    sys.exit(main())
