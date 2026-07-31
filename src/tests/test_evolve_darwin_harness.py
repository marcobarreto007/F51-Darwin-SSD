from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("evolve_darwin", ROOT / "research" / "evolve_darwin.py")
assert SPEC is not None
evolve_darwin = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = evolve_darwin
SPEC.loader.exec_module(evolve_darwin)


def test_fixed_verifier_smoke_does_not_claim_evolution() -> None:
    engine = evolve_darwin.EvoEngine(model_probe=False)
    report = engine.run(1)
    assert report.accepted == 4
    assert report.rejected == 0
    assert engine.proof_status() == "VERIFIER_SMOKE_ONLY"
    assert not report.model_generated_knowledge_smoke


def test_model_probe_measures_learning_without_overclaiming() -> None:
    probe = evolve_darwin.run_model_probe(seed=51, train_steps=8, max_tokens=2)
    assert probe.enabled
    assert probe.loss_before is not None
    assert probe.loss_after is not None
    assert probe.loss_after < probe.loss_before
    assert probe.learning_smoke_passed
    assert len(probe.generated) == 2
    assert probe.promotion_gate["action"] == "quarantine"
    assert not probe.promotion_gate["passed"]
    assert not probe.promotion_gate["gates"]["verified_generation"]
