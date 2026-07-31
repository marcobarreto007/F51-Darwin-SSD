#!/usr/bin/env python3
"""Regenerate the Nitro runtime SBOM from the live interpreter.

check_nitro_runtime.py verifies the SBOM but cannot rebuild it, so the only
way to reconcile a drifted environment was to hand-edit the CycloneDX file.
Hand-editing a supply-chain record is the exact gesture the rest of this
system exists to prevent: it produces an attestation with no record of what
changed or why.

This tool regenerates the component list from the installed distributions and
prints the delta first. Regeneration is an ATTESTATION -- it asserts that the
packages now present are approved -- so --apply is required to write, and the
delta is emitted for the record either way.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SBOM = ROOT / "governance/audit/sbom/nitro-runtime.cyclonedx.json"
PROVENANCE = ROOT / "governance/audit/provenance/nitro-runtime.json"


def normalize(name: str) -> str:
    return name.lower().replace("_", "-")


def installed_components() -> list[dict[str, str]]:
    seen: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        seen[normalize(name)] = dist.version or "0"
    return [{"name": n, "version": v} for n, v in sorted(seen.items())]


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate the Nitro SBOM")
    parser.add_argument("--sbom", type=Path, default=SBOM)
    parser.add_argument("--apply", action="store_true",
                        help="Write the file. Without it, only report the delta.")
    args = parser.parse_args()

    document = json.loads(args.sbom.read_text(encoding="utf-8"))
    recorded = {normalize(c["name"]): c.get("version", "")
                for c in document.get("components", [])}
    current = {c["name"]: c["version"] for c in installed_components()}

    added = sorted(set(current) - set(recorded))
    removed = sorted(set(recorded) - set(current))
    changed = sorted(
        n for n in set(current) & set(recorded)
        if current[n] != recorded[n]
    )

    print(f"recorded : {len(recorded)}")
    print(f"installed: {len(current)}")
    print(f"\nadded   ({len(added)}): {', '.join(added) if added else 'none'}")
    print(f"removed ({len(removed)}): {', '.join(removed) if removed else 'none'}")
    print(f"changed ({len(changed)}): "
          f"{', '.join(f'{n} {recorded[n]}->{current[n]}' for n in changed) or 'none'}")

    if not (added or removed or changed):
        print("\nSBOM already matches the environment.")
        return 0

    if not args.apply:
        print("\nDry run. Regenerating asserts that every added package is "
              "approved for this runtime; pass --apply to write.")
        return 1

    document["components"] = installed_components()
    document.setdefault("metadata", {}).setdefault("properties", [])
    properties = [p for p in document["metadata"]["properties"]
                  if p.get("name") != "f51:regenerated"]
    properties.append({
        "name": "f51:regenerated",
        "value": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    document["metadata"]["properties"] = properties

    payload = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False)
    args.sbom.write_text(payload + "\n", encoding="utf-8")
    digest = hashlib.sha256(args.sbom.read_bytes()).hexdigest()

    print(f"\nwritten: {args.sbom}")
    print(f"sha256 : {digest}")

    if PROVENANCE.is_file():
        record = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        stale = [k for k, v in record.items()
                 if isinstance(v, str) and len(v) == 64 and v != digest]
        if stale:
            print(f"\nprovenance record still carries the previous digest under "
                  f"{stale}; update {PROVENANCE.name} before the checker will pass.")
    print("SBOM_REGENERATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
