#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from f51_darwin.transplant.bundle import extract_first_slice_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a verified first-slice organ bundle."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument(
        "--gaba-layers",
        default="0",
        help="Comma-separated donor GABA layer indices.",
    )
    args = parser.parse_args()

    layers = tuple(
        int(value.strip())
        for value in args.gaba_layers.split(",")
        if value.strip()
    )
    manifest = extract_first_slice_bundle(
        args.checkpoint,
        args.output,
        source_checkpoint_sha256=args.checkpoint_sha256,
        gaba_layers=layers,
    )
    parameter_count = sum(
        math.prod(record.shape)
        for record in manifest.tensors
    )
    print(
        json.dumps(
            {
                "status": "verified",
                "source_checkpoint_sha256": manifest.source_checkpoint_sha256,
                "organs": manifest.organs,
                "tensor_count": len(manifest.tensors),
                "parameter_count": parameter_count,
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
