from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


REQUIRED_SEEDS = (17, 29, 43)


@dataclass(frozen=True)
class KnowledgeMetrics:
    seed: int
    bpb: float
    random_bpb: float
    kl: float
    random_kl: float
    top1: float
    random_top1: float
    generation_valid: bool


@dataclass(frozen=True)
class KnowledgeGateReport:
    passed: bool
    status: str
    failures: tuple[str, ...]
    metrics: tuple[KnowledgeMetrics, ...]


def evaluate_knowledge_gates(
    metrics: Iterable[KnowledgeMetrics],
) -> KnowledgeGateReport:
    rows = tuple(sorted(metrics, key=lambda row: row.seed))
    failures: list[str] = []
    if tuple(row.seed for row in rows) != REQUIRED_SEEDS:
        failures.append(
            f"seeds must be exactly {REQUIRED_SEEDS}, got "
            f"{tuple(row.seed for row in rows)}"
        )
    for row in rows:
        checks = {
            "bpb_10pct": row.bpb <= 0.90 * row.random_bpb,
            "kl_25pct": row.kl <= 0.75 * row.random_kl,
            "top1_2x": row.top1 >= 2.0 * row.random_top1,
            "paired_bpb": row.bpb < row.random_bpb,
            "paired_kl": row.kl < row.random_kl,
            "generation": row.generation_valid,
        }
        failed = sorted(name for name, passed in checks.items() if not passed)
        if failed:
            failures.append(f"seed={row.seed}: {','.join(failed)}")
    passed = not failures
    return KnowledgeGateReport(
        passed=passed,
        status=(
            "knowledge_gates_passed"
            if passed
            else "engineering_transplant_only"
        ),
        failures=tuple(failures),
        metrics=rows,
    )
