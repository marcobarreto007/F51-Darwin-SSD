from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.data_factory import DataFactory, DataFactoryPaths, load_factory_config
from f51_darwin.synthetic_pt_generator import generate_portuguese_documents


GENERATOR_MODEL = "F51-Synthetic-PT-Generator-v0"
GENERATOR_CHECKPOINT = "synthetic://procedural/pt-20k-v0"
PROMPT = "procedural portuguese bootstrap corpus 20000 words"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate ~20k words of synthetic Portuguese and register candidates only.",
    )
    parser.add_argument("--config", default=str(ROOT / "src" / "configs" / "data_factory.yaml"))
    parser.add_argument("--target-words", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument(
        "--write-source",
        default=str(ROOT / "data" / "generated" / "candidates" / "source" / "f51_synthetic_pt_20k.txt"),
        help="Optional combined source dump inside candidates zone.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw = load_factory_config(Path(args.config))
    paths = DataFactoryPaths.from_project(ROOT, raw)
    factory = DataFactory(paths)

    report = generate_portuguese_documents(target_words=args.target_words, seed=args.seed)
    source_path = Path(args.write_source)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("\n\n".join(report.documents) + "\n", encoding="utf-8")

    records = []
    for index, document in enumerate(report.documents, start=1):
        record = factory.register_synthetic(
            text=document,
            generator_model=GENERATOR_MODEL,
            generator_checkpoint=GENERATOR_CHECKPOINT,
            prompt=f"{PROMPT}; chunk={index}",
            dataset_version=str(raw.get("dataset_version", "v0")),
        )
        records.append(
            {
                "id": record.id,
                "words": len(document.split()),
                "path": str(paths.candidates / f"{record.id}.json"),
            }
        )

    payload = {
        "documents": len(report.documents),
        "target_words": report.target_words,
        "word_count": report.word_count,
        "source_path": str(source_path),
        "candidates_registered": len(records),
        "generator_model": GENERATOR_MODEL,
        "generator_checkpoint": GENERATOR_CHECKPOINT,
        "rule": "Synthetic data is never training data by default.",
        "next_steps": [
            "python src/tools/audit_dataset_candidates.py",
            "python src/tools/promote_approved_data.py --approve-id <id> --build-corpus",
        ],
        "records": records[:5],
        "records_truncated": len(records) > 5,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
