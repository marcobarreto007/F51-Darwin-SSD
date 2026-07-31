#!/usr/bin/env python3
"""Circuit registry — searchable catalog of stamped circuits.

Takes the output of scan_model_circuits.py and builds a queryable,
hash-indexed registry with:
  - circuit_id → full stamp lookup
  - function_label → channel list
  - layer → top channels
  - donor_sha256 → circuit inventory
  - compatibility matrix between donors
"""

from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any


def load_catalog(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_registry(catalogs: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple catalog scans into a unified registry."""
    registry: dict[str, Any] = {
        "schema": "darwin-circuit-registry-v1",
        "by_circuit_id": {},
        "by_function": {"factual": [], "syntactic": [], "positional": [], "mixed": [], "noise": []},
        "by_layer": {},
        "by_donor": {},
        "compatibility_matrix": [],
        "total_stamps": 0,
    }

    seen = set()
    for cat in catalogs:
        donor = cat.get("checkpoint_sha256", "unknown")
        if donor not in registry["by_donor"]:
            registry["by_donor"][donor] = {"stamps": [], "count": 0}

        for stamp in cat.get("stamps", []):
            cid = stamp["circuit_id"]
            if cid in seen:
                continue
            seen.add(cid)

            registry["by_circuit_id"][cid] = stamp
            registry["by_function"][stamp["label"]].append(cid)

            layer = str(stamp["layer"])
            if layer not in registry["by_layer"]:
                registry["by_layer"][layer] = []
            registry["by_layer"][layer].append(cid)

            registry["by_donor"][donor]["stamps"].append(cid)
            registry["by_donor"][donor]["count"] += 1
            registry["total_stamps"] += 1

    # Sort by-layer lists by attribution
    for layer in registry["by_layer"]:
        registry["by_layer"][layer].sort(
            key=lambda cid: registry["by_circuit_id"][cid]["attribution"],
            reverse=True,
        )

    return registry


def query(registry: dict[str, Any], **filters: Any) -> list[dict[str, Any]]:
    """Query the registry by any combination of filters."""
    results = []
    for cid, stamp in registry["by_circuit_id"].items():
        match = True
        for key, value in filters.items():
            if key == "layer" and stamp.get("layer") != value:
                match = False
            elif key == "label" and stamp.get("label") != value:
                match = False
            elif key == "min_confidence" and stamp.get("confidence", 0) < value:
                match = False
            elif key == "min_attribution" and stamp.get("attribution", 0) < value:
                match = False
            elif key == "donor" and stamp.get("donor_sha256") != value:
                match = False
        if match:
            results.append(stamp)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Circuit registry populator")
    parser.add_argument("--catalogs", nargs="+", required=True, help="Catalog JSON files")
    parser.add_argument("--output", default="workspace/runtime/circuit-catalog/registry.json")
    parser.add_argument("--query-layer", type=int, help="Query: filter by layer")
    parser.add_argument("--query-label", help="Query: filter by function label")
    parser.add_argument("--query-min-confidence", type=float, default=0.5)
    parser.add_argument("--query-top", type=int, default=10, help="Top N results")
    args = parser.parse_args()

    print("Circuit Registry Builder")
    print(f"  Loading {len(args.catalogs)} catalog(s)...")

    catalogs = []
    for path_str in args.catalogs:
        cat = load_catalog(Path(path_str))
        catalogs.append(cat)
        print(f"    {path_str}: {len(cat.get('stamps', []))} stamps")

    registry = build_registry(catalogs)
    print(f"\n  Registry built: {registry['total_stamps']} unique stamps")
    print(f"  Functions: { {k: len(v) for k, v in registry['by_function'].items()} }")
    print(f"  Layers: {len(registry['by_layer'])}")
    print(f"  Donors: {len(registry['by_donor'])}")

    # Run query if requested
    filters = {}
    if args.query_layer is not None:
        filters["layer"] = args.query_layer
    if args.query_label:
        filters["label"] = args.query_label
    if args.query_min_confidence:
        filters["min_confidence"] = args.query_min_confidence

    if filters:
        results = query(registry, **filters)
        print(f"\n  Query ({filters}): {len(results)} matches")
        for r in results[:args.query_top]:
            print(f"    {r['circuit_id'][:16]}... layer={r['layer']} ch={r['channel']} "
                  f"label={r['label']} conf={r.get('confidence', 0):.2f} attr={r.get('attribution', 0):.3f}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(registry, indent=2, sort_keys=True))
    print(f"\nRegistry saved: {out_path}")
    print("REGISTRY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
