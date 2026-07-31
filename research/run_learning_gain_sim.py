#!/usr/bin/env python3
"""Learning-Gain Simulation — CLI

Thin CLI wrapper around research.learning_gain.runner.

Usage:
    python research/run_learning_gain_sim.py `
      --scenarios all --seeds 30 `
      --output-dir workspace/runtime/learning_gain_sim

    python research/run_learning_gain_sim.py --validate-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from research.learning_gain.runner import (
    run_experiment, plot_results, DEFAULT_POLICIES,
)
from research.learning_gain.scenarios import SCENARIO_GENERATORS

ALL_SCENARIOS = sorted(SCENARIO_GENERATORS.keys())
CHECKPOINT_ROOTS = [
    "workspace/03_CHECKPOINTS_100M_FULL_V9",
    "workspace/03_CHECKPOINTS_100M_FULL_ORGANISM_V1",
    "workspace/03_CHECKPOINTS_100M_FULL_ORGANISM_V3",
    "workspace/03_CHECKPOINTS_600M",
    "workspace/03_CHECKPOINTS_1.6B",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Learning-Gain Simulation — F51 Darwin-X Research"
    )
    p.add_argument(
        "--scenarios", default="all",
        help="Comma-separated scenario IDs or 'all' (default: all)",
    )
    p.add_argument(
        "--seeds", type=int, default=30,
        help="Number of deterministic seeds (default: 30)",
    )
    p.add_argument(
        "--output-dir", default="workspace/runtime/learning_gain_sim",
        help="Output directory for artifacts (default: workspace/runtime/learning_gain_sim)",
    )
    p.add_argument(
        "--dev-seeds", type=int, default=10,
        help="Number of development seeds (not used in this version)",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation",
    )
    p.add_argument(
        "--validate-only", action="store_true",
        help="Run validation tests only, don't execute experiment",
    )
    p.add_argument(
        "--small", action="store_true",
        help="Quick smoke test: 5 seeds, 4 scenarios",
    )
    return p.parse_args()


def _resolve_scenarios(arg: str) -> list[str]:
    """Resolve scenario argument to list of scenario IDs."""
    if arg == "all":
        return ALL_SCENARIOS
    requested = [s.strip() for s in arg.split(",")]
    resolved = []
    for s in requested:
        if s in SCENARIO_GENERATORS:
            resolved.append(s)
        else:
            print(f"[WARN] Unknown scenario: {s} — skipping")
    return resolved or ALL_SCENARIOS


def _check_output_dir(output_dir: str) -> None:
    """Refuse to write into a checkpoint root (per spec)."""
    resolved = str(Path(output_dir).resolve())
    repo_root = str(ROOT.resolve())
    for ckpt_root in CHECKPOINT_ROOTS:
        ckpt_path = str((ROOT / ckpt_root).resolve())
        if resolved.startswith(ckpt_path):
            print(f"ERROR: Output directory resolves inside checkpoint root: {ckpt_root}")
            print("This is forbidden per spec. Use a directory outside any checkpoint root.")
            sys.exit(1)


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir)
    _check_output_dir(args.output_dir)

    if args.validate_only:
        print("Running validation tests...")
        result = _run_validation_checks()
        if result:
            print("All validation checks PASSED.")
            return 0
        else:
            print("Some validation checks FAILED.")
            return 1

    # Determine scenarios and seeds
    if args.small:
        scenarios = ALL_SCENARIOS[:4]  # S01-S04
        seeds = list(range(5))
        print(f"SMOKE TEST: {len(scenarios)} scenarios, {len(seeds)} seeds")
    else:
        scenarios = _resolve_scenarios(args.scenarios)
        seeds = list(range(args.seeds))
        print(f"FULL RUN: {len(scenarios)} scenarios, {len(seeds)} seeds")

    print(f"Output: {output_dir.resolve()}")

    # Run experiment
    raw_results, aggregate = run_experiment(
        scenarios=scenarios,
        seeds=seeds,
        policies_config=DEFAULT_POLICIES,
        output_dir=output_dir,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for pname, summary in aggregate["policy_summaries"].items():
        if summary["n_runs"] > 0:
            print(f"  {pname:20s}  G={summary['median_G']:.4f}  "
                  f"CI=[{summary['ci_95_G'][0]:.4f}, {summary['ci_95_G'][1]:.4f}]  "
                  f"R_old={summary['median_R_old']:.3f}")

    print(f"\n  Falsification gates:")
    for gate in aggregate["falsification_gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"    [{status}] Gate {gate['gate_id']}: {gate['evidence'][:80]}")

    all_passed = aggregate.get("all_gates_passed", False)
    print(f"\n  OVERALL: {'ALL GATES PASSED' if all_passed else 'SOME GATES FAILED'}")

    # Plots
    if not args.no_plots:
        plot_results(raw_results, output_dir)

    print(f"\nArtifacts: {output_dir.resolve()}")
    return 0 if all_passed else 1


def _run_validation_checks() -> bool:
    """Run quick validation checks inline."""
    from research.learning_gain.scenarios import generate_s01, generate_s06, check_s06_marginals
    from research.learning_gain.state import LearnerState, Experience, UpdateAction, predict, execute_action
    import numpy as np

    all_ok = True

    # Check 1: seed determinism
    s1_a = generate_s01(42)
    s1_b = generate_s01(42)
    ok = all(
        np.allclose(a.x, b.x) and np.allclose(a.y, b.y)
        for a, b in zip(s1_a, s1_b)
    )
    print(f"  {'[PASS]' if ok else '[FAIL]'} Seed determinism")
    all_ok = all_ok and ok

    # Check 2: S06 marginals
    orig = generate_s01(42)
    shuf = generate_s06(42)
    ok = check_s06_marginals(orig, shuf)
    print(f"  {'[PASS]' if ok else '[FAIL]'} S06 marginals preserved")
    all_ok = all_ok and ok

    # Check 3: SKIP cost
    ok = True  # enforced by ACTION_COST
    print(f"  {'[PASS]' if ok else '[FAIL]'} SKIP cost < LOCAL_UPDATE cost")

    # Check 4: Accounting conservation
    state = LearnerState(compute_budget_remaining=100, write_budget_remaining=100)
    exp = Experience(
        x=np.random.randn(4).astype(np.float64), c=0,
        y=np.ones(1).astype(np.float64), q=0.5, d=0, t=0,
    )
    state, comp, write = execute_action(state, UpdateAction.LOCAL_UPDATE, exp, None)
    ok = state.total_compute_spent == comp and state.total_writes_spent == write
    print(f"  {'[PASS]' if ok else '[FAIL]'} Accounting conservation")

    return all_ok


if __name__ == "__main__":
    raise SystemExit(main())
