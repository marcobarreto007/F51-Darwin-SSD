from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import yaml

from f51_darwin.evolution_score import EvolutionMetrics, calculate_evolution_score, should_grow
from f51_darwin.expert_pool import ExpertModule, ExpertPool, ModuleState
from f51_darwin.pruning import ablation_decision, mark_dead, quarantine_if_low_score


def main() -> int:
    config = yaml.safe_load((ROOT / "src" / "configs" / "evolution_cycle.yaml").read_text())
    pool = ExpertPool()
    pool.add_expert(
        "candidate_ssd_memory_0001",
        ExpertModule(d_model=16),
        created_at_cycle=1,
        state=ModuleState.CANDIDATE,
    )
    metrics = EvolutionMetrics(
        new_gain=0.42,
        retention=0.88,
        forgetting=0.06,
        compute_cost=0.09,
        redundancy=0.04,
    )
    score = calculate_evolution_score(metrics)
    pool.set_score("candidate_ssd_memory_0001", score)
    grow = should_grow(
        new_loss=2.4,
        replay_loss_improved=False,
        local_adaptation_failed=True,
        evolution_score=score,
        min_evolution_score=config["growth_rule"]["min_evolution_score"],
    )
    if grow:
        pool.set_state("candidate_ssd_memory_0001", ModuleState.ACTIVE)
    else:
        pool.set_state("candidate_ssd_memory_0001", ModuleState.QUARANTINE)

    prune_candidate = pool.add_expert(
        "weak_attention_binding_0001",
        ExpertModule(d_model=16),
        created_at_cycle=1,
        state=ModuleState.ACTIVE,
        score=-0.2,
    )
    quarantine_if_low_score(
        pool,
        prune_candidate.id,
        threshold=config["pruning_rule"]["low_score_threshold"],
    )
    ablation = ablation_decision(
        baseline_score=0.700,
        ablated_score=0.696,
        threshold=config["pruning_rule"]["ablation_delta_threshold"],
    )
    if ablation.action == "quarantine":
        mark_dead(pool, prune_candidate.id)

    report_path = ROOT / config["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "cycle": config["cycle_name"],
        "formula": "new_gain + retention - forgetting - compute_cost - redundancy",
        "candidate_score": score,
        "candidate_promoted": grow,
        "ablation": ablation.__dict__,
        "experts": pool.metadata(),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

