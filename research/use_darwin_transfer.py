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


def checkpoint_from_pointer(root: Path) -> Path:
    pointer = root / "published.json"
    if pointer.is_file():
        return Path(json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"])
    latest = root / "latest.pt"
    if latest.is_file():
        return latest
    raise FileNotFoundError(f"no published or latest checkpoint below {root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--prompt", default="Darwin learns through")
    parser.add_argument("--observe-text")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--control",
        choices=("correct", "frozen", "shuffled", "reset"),
        default="correct",
    )
    args = parser.parse_args()

    checkpoint = checkpoint_from_pointer(args.checkpoint_root)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    layers = tuple(payload["metadata"]["replaced_layers"])
    teacher, tokenizer, _ = load_frozen_teacher(args.donor, device=args.device)
    student = DarwinTransferStudent.from_teacher(
        teacher, replaced_layers=layers
    ).to(args.device)
    load_transfer_checkpoint(
        student,
        checkpoint,
        expected_donor_manifest_sha256=sha256_file(
            args.donor / "donor_manifest.json"
        ),
    )
    student.eval()

    if args.memory.is_file():
        memory = ExperienceMemory.load(args.memory)
    else:
        memory = ExperienceMemory(
            capacity=4096,
            top_k=64,
            threshold=0.995,
            min_observations=1,
        )
    if args.control == "frozen":
        memory.freeze()
    elif args.control == "shuffled":
        memory.shuffle(seed=51)
    elif args.control == "reset":
        memory.reset()

    runtime = DarwinTransferRuntime(
        student,
        student.model.transformer.wte,
        memory,
    )
    if args.observe_text:
        observed = tokenizer(
            args.observe_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(args.device)
        for position in range(1, observed.shape[1]):
            runtime.observe(
                observed[:, :position],
                target_id=int(observed[0, position]),
                vocab_size=student.config.vocab_size,
            )

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)
    generated, paths = runtime.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        learn=args.control == "correct",
    )
    if args.control != "frozen":
        memory.save(args.memory)
    print(
        json.dumps(
            {
                "text": tokenizer.decode(generated[0], skip_special_tokens=True),
                "paths": paths,
                "backbone_calls": runtime.backbone_calls,
                "experience_hits": runtime.experience_hits,
                "memory_slots": len(memory),
                "control": args.control,
                "checkpoint": str(checkpoint),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
