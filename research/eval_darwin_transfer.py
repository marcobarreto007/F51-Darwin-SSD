from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from f51_darwin.transfer.checkpoint import load_transfer_checkpoint
from f51_darwin.transfer.donor import load_frozen_teacher, sha256_file
from f51_darwin.transfer.experience import ExperienceMemory
from f51_darwin.transfer.runtime import DarwinTransferRuntime
from f51_darwin.transfer.student import DarwinTransferStudent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="Darwin learns through")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    teacher, tokenizer, _ = load_frozen_teacher(args.donor, device=args.device)
    metadata_path = args.checkpoint.with_suffix(args.checkpoint.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    student = DarwinTransferStudent.from_teacher(
        teacher, replaced_layers=tuple(metadata["replaced_layers"])
    ).to(args.device)
    load_transfer_checkpoint(
        student,
        args.checkpoint,
        expected_donor_manifest_sha256=sha256_file(
            args.donor / "donor_manifest.json"
        ),
    )
    memory = ExperienceMemory(
        capacity=1024, top_k=64, threshold=0.995, min_observations=2
    )
    runtime = DarwinTransferRuntime(
        student, student.model.transformer.wte, memory
    )
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)
    outputs = [runtime.predict_next(ids, learn=True) for _ in range(3)]
    print(
        json.dumps(
            {
                "paths": [output.path for output in outputs],
                "backbone_calls": runtime.backbone_calls,
                "experience_hits": runtime.experience_hits,
            }
        )
    )


if __name__ == "__main__":
    main()
