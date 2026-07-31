#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from f51_darwin.transplant_16b.verification import (
    KnowledgeMetrics,
    evaluate_knowledge_gates,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate paired Darwin-Smol knowledge gates."
    )
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rows = tuple(
        KnowledgeMetrics(**row) for row in payload.get("metrics", ())
    )
    report = evaluate_knowledge_gates(rows)
    if report.passed:
        print("KNOWLEDGE_GATES_PASS seeds=17,29,43")
        return 0
    print(
        "KNOWLEDGE_GATES_FAIL status=engineering_transplant_only "
        + " failures="
        + ";".join(report.failures)
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
