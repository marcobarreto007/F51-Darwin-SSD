"""Paired-seed execution and artifact assembly.

Runs all policies across all scenarios with all seeds, assembles
raw results, and generates aggregate reports.

Data flow is ONE-WAY (per spec):
    config + seed -> scenario stream -> policy -> evaluator -> results
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from research.learning_gain.state import (
    LearnerState, Experience, EligibilityTrace,
    UpdateAction, BudgetExceededError,
    cosine_similarity, predict, execute_action,
    deliver_delayed_consequences,
)
from research.learning_gain.scenarios import (
    SCENARIO_GENERATORS, get_scenario_config,
)
from research.learning_gain.evaluator import (
    create_probes, Evaluator, compute_G, check_falsification_gates,
)
from research.learning_gain.policies import (
    Policy, FrozenPolicy, AlwaysUpdatePolicy, SurpriseOnlyPolicy,
    GainAdaptivePolicy, RandomGatePolicy,
)

EPSILON = 1e-10
SCHEMA_VERSION = "1.0.0"


# ── Config ─────────────────────────────────────────────────────────

DEFAULT_POLICIES: list[tuple[str, dict[str, Any]]] = [
    ("Frozen", {}),
    ("AlwaysUpdate", {"similarity_threshold": 0.5}),
    ("SurpriseOnly", {"surprise_threshold": 0.5}),
    ("GainAdaptive", {
        "similarity_threshold": 0.4,
        "lambda_write": 0.05,
        "lambda_interference": 0.1,
        "lambda_compute": 0.01,
    }),
    ("RandomGate", {"update_probability": 0.3, "seed": 0}),
]


def _make_policy(name: str, params: dict[str, Any]) -> Policy:
    """Policy factory."""
    mapping: dict[str, type[Policy]] = {
        "Frozen": FrozenPolicy,
        "AlwaysUpdate": AlwaysUpdatePolicy,
        "SurpriseOnly": SurpriseOnlyPolicy,
        "GainAdaptive": GainAdaptivePolicy,
        "RandomGate": RandomGatePolicy,
    }
    cls = mapping[name]
    return cls(**params)


# ── Single Run ─────────────────────────────────────────────────────

def run_single(
    policy: Policy,
    experiences: list[Experience],
    scenario_id: str,
    seed: int,
    scenario_config: dict[str, Any],
) -> dict[str, Any]:
    """Run one policy on one scenario stream with one seed.

    Returns per-row dict with all metrics.
    """
    n = len(experiences)
    warmup_end = max(1, int(n * scenario_config.get("warmup_ratio", 0.1)))

    # Budget from scenario config
    compute_budget = scenario_config.get("compute_budget", float("inf"))
    write_budget = scenario_config.get("write_budget", 10_000)

    # Create probes from seed
    probes = create_probes(experiences, seed)

    # Initialize state
    state = LearnerState(
        max_slots=scenario_config.get("max_slots", 16),
        max_traces=scenario_config.get("max_traces", 64),
        compute_budget_remaining=float(compute_budget),
        write_budget_remaining=int(write_budget),
    )

    # Warm-up
    warmup_exps = experiences[:warmup_end]
    state = policy.warm_up(state, warmup_exps)

    # Run stream
    predictions_log: list[dict[str, Any]] = []
    action_log: list[dict[str, Any]] = []
    experience_log: dict[int, Experience] = {exp.t: exp for exp in experiences}

    for exp in experiences[warmup_end:]:
        # Predict
        pred, conf, matched = predict(state, exp.x)
        predictions_log.append({
            "t": exp.t,
            "prediction": pred.tolist(),
            "confidence": conf,
            "matched_slot": matched,
        })

        # Decide action
        try:
            action = policy.decide_action(state, exp, pred, conf, matched)
        except Exception:
            action = UpdateAction.SKIP

        # Track policy stats
        policy.record_action(action)

        # Execute action (actual costs, not estimated)
        try:
            state, comp_cost, write_cost = execute_action(
                state, action, exp, matched,
            )
        except BudgetExceededError:
            # Budget exhausted: force SKIP
            state, comp_cost, write_cost = execute_action(
                state, UpdateAction.SKIP, exp, None,
            )
            action = UpdateAction.SKIP

        action_log.append({
            "t": exp.t,
            "action": action.value,
            "compute_cost": comp_cost,
            "write_cost": write_cost,
        })

        # Add eligibility trace if consequence is delayed
        if exp.d > 0:
            if len(state.traces) < state.max_traces:
                state.traces.append(EligibilityTrace(
                    experience_ref=exp.t,
                    remaining_delay=exp.d,
                    initial_delay=exp.d,
                    feature_snapshot=exp.x.copy(),
                ))

        # Deliver delayed consequences
        deliver_delayed_consequences(state, exp.t, experience_log)

        # Recurrence tracking: if this experience matched a slot
        # that was also the best match last time for similar x
        if matched is not None and matched < len(state.slots):
            state.slots[matched].recurrence_count += 1

    # Evaluate
    evaluator = Evaluator(probes, scenario_id)
    metrics = evaluator.evaluate(
        policy.name, state, predictions_log, action_log, experiences,
    )
    metrics["seed"] = seed
    metrics["policy_stats"] = policy.get_stats()

    return metrics


# ── Full Experiment ────────────────────────────────────────────────

def run_experiment(
    scenarios: list[str],
    seeds: list[int],
    policies_config: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    output_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the full experiment: all policies x scenarios x seeds.

    Args:
        scenarios: list of scenario IDs (S01-S10)
        seeds: list of deterministic seeds
        policies_config: optional policy list override
        output_dir: optional output directory for raw results

    Returns:
        (raw_results: list of per-row dicts, aggregate_report: dict)
    """
    if policies_config is None:
        policies_config = DEFAULT_POLICIES

    raw_results: list[dict[str, Any]] = []

    for scenario_id in scenarios:
        scenario_config = get_scenario_config(scenario_id)
        generator = SCENARIO_GENERATORS[scenario_id]

        for seed in seeds:
            # Generate deterministic stream
            experiences = generator(seed)

            # All policies see the SAME stream for this seed (seed pairing)
            for policy_name, policy_params in policies_config:
                policy = _make_policy(policy_name, policy_params)

                try:
                    row = run_single(
                        policy, experiences, scenario_id, seed, scenario_config,
                    )
                except Exception as exc:
                    row = {
                        "policy": policy_name,
                        "scenario": scenario_id,
                        "seed": seed,
                        "error": str(exc),
                        "delta_q_future": 0.0,
                        "R_old": 0.0,
                        "X_unseen": 0.0,
                        "total_compute": 0.0,
                        "total_writes": 0,
                        "interference_cost": 0.0,
                    }

                raw_results.append(row)

    # Compute G for each row (relative to Frozen baseline for same scenario+seed)
    frozen_rows = [r for r in raw_results if r["policy"] == "Frozen"]
    frozen_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for fr in frozen_rows:
        frozen_by_key[(fr["scenario"], fr["seed"])] = fr

    for row in raw_results:
        key = (row["scenario"], row["seed"])
        frozen_ref = frozen_by_key.get(key)
        if frozen_ref and "error" not in row:
            row["G"] = compute_G(row, frozen_ref)
        else:
            row["G"] = 0.0

    # Aggregate report
    aggregate = _build_aggregate(raw_results, scenarios, seeds, policies_config)

    # Write raw results if output_dir provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_artifacts(raw_results, aggregate, output_dir, scenarios, seeds)

    return raw_results, aggregate


def _build_aggregate(
    raw_results: list[dict[str, Any]],
    scenarios: list[str],
    seeds: list[int],
    policies_config: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Build aggregate JSON report from raw results."""
    policy_names = [p[0] for p in policies_config]

    # Per-policy summary
    summaries: dict[str, dict[str, Any]] = {}
    for pname in policy_names:
        p_rows = [r for r in raw_results if r["policy"] == pname and "error" not in r]
        if not p_rows:
            summaries[pname] = {"n_runs": 0}
            continue
        Gs = [r.get("G", 0.0) for r in p_rows]
        summaries[pname] = {
            "n_runs": len(p_rows),
            "median_G": float(np.median(Gs)),
            "mean_G": float(np.mean(Gs)),
            "median_compute": float(np.median([r.get("total_compute", 0) for r in p_rows])),
            "median_writes": float(np.median([r.get("total_writes", 0) for r in p_rows])),
            "median_R_old": float(np.median([r.get("R_old", 0) for r in p_rows])),
            "median_X_unseen": float(np.median([r.get("X_unseen", 0) for r in p_rows])),
            "ci_95_G": _bootstrap_ci(Gs),
        }

    # Per-scenario summary
    scenario_summaries: dict[str, dict[str, Any]] = {}
    for sid in scenarios:
        s_rows = [r for r in raw_results if r["scenario"] == sid and "error" not in r]
        s_summary: dict[str, Any] = {"n_runs": len(s_rows)}
        for pname in policy_names:
            ps_rows = [r for r in s_rows if r["policy"] == pname]
            if ps_rows:
                s_summary[pname] = {
                    "median_G": float(np.median([r.get("G", 0) for r in ps_rows])),
                    "median_compute": float(np.median([r.get("total_compute", 0) for r in ps_rows])),
                }
        scenario_summaries[sid] = s_summary

    # Falsification gates
    gates = check_falsification_gates(raw_results, scenarios)

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_scenarios": len(scenarios),
        "n_seeds": len(seeds),
        "n_policies": len(policy_names),
        "policy_summaries": summaries,
        "scenario_summaries": scenario_summaries,
        "falsification_gates": gates,
        "all_gates_passed": all(g["passed"] for g in gates),
    }


def _clean_for_json(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _write_artifacts(
    raw_results: list[dict[str, Any]],
    aggregate: dict[str, Any],
    output_dir: Path,
    scenarios: list[str],
    seeds: list[int],
) -> None:
    """Write immutable config, raw results, aggregate JSON, and Markdown report."""

    # Config hash
    config_str = json.dumps({
        "scenarios": scenarios,
        "seeds": seeds,
        "schema_version": SCHEMA_VERSION,
    }, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

    # Immutable config
    config = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": config_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenarios": scenarios,
        "seeds": seeds,
        "n_seeds": len(seeds),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")

    # Raw results (JSONL)
    with open(output_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for row in raw_results:
            # Clean non-serializable types
            clean_row: dict[str, Any] = {}
            for k, v in row.items():
                if isinstance(v, (np.integer,)):
                    clean_row[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean_row[k] = float(v)
                elif isinstance(v, np.ndarray):
                    clean_row[k] = v.tolist()
                elif isinstance(v, dict):
                    clean_row[k] = {
                        kk: (int(vv) if isinstance(vv, (np.integer,))
                             else float(vv) if isinstance(vv, (np.floating,))
                             else vv)
                        for kk, vv in v.items()
                    }
                else:
                    clean_row[k] = v
            f.write(json.dumps(clean_row, ensure_ascii=False) + "\n")

    # Aggregate JSON
    clean_aggregate = _clean_for_json(aggregate)
    (output_dir / "aggregate.json").write_text(
        json.dumps(clean_aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown report
    md = _generate_markdown_report(aggregate, raw_results, scenarios, config_hash)
    (output_dir / "report.md").write_text(md, encoding="utf-8")


def _generate_markdown_report(
    aggregate: dict[str, Any],
    raw_results: list[dict[str, Any]],
    scenarios: list[str],
    config_hash: str,
) -> str:
    """Generate Markdown report."""
    lines: list[str] = []
    lines.append("# Learning-Gain Simulation — Results")
    lines.append("")
    lines.append(f"- **Config hash:** `{config_hash}`")
    lines.append(f"- **Timestamp:** {aggregate['timestamp']}")
    lines.append(f"- **Scenarios:** {aggregate['n_scenarios']}")
    lines.append(f"- **Seeds:** {aggregate['n_seeds']}")
    lines.append(f"- **Policies:** {aggregate['n_policies']}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Policy Summary")
    lines.append("")
    lines.append("| Policy | N | Median G | Mean G | 95% CI | Median Compute | Median Writes | Median R_old |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for pname, summary in aggregate["policy_summaries"].items():
        if summary["n_runs"] == 0:
            lines.append(f"| {pname} | 0 | — | — | — | — | — | — |")
        else:
            ci = summary.get("ci_95_G", [0, 0])
            lines.append(
                f"| {pname} | {summary['n_runs']} | "
                f"{summary['median_G']:.4f} | {summary['mean_G']:.4f} | "
                f"[{ci[0]:.4f}, {ci[1]:.4f}] | "
                f"{summary['median_compute']:.1f} | {summary['median_writes']:.0f} | "
                f"{summary['median_R_old']:.3f} |"
            )
    lines.append("")
    lines.append("## Falsification Gates")
    lines.append("")
    lines.append("| Gate | Passed | Evidence |")
    lines.append("|---|---|---|")
    for gate in aggregate["falsification_gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"| {gate['gate_id']} | {status} | {gate['evidence']} |")
    lines.append("")
    all_passed = aggregate.get("all_gates_passed", False)
    lines.append(f"**Overall: {'ALL GATES PASSED' if all_passed else 'SOME GATES FAILED'}**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Scenario Detail")
    lines.append("")
    for sid in scenarios:
        s_rows = [r for r in raw_results if r["scenario"] == sid and "error" not in r]
        lines.append(f"### {sid}")
        lines.append("")
        lines.append("| Policy | Median G | Median Compute | Median R_old | Median X_unseen |")
        lines.append("|---|---|---|---|---|")
        for pname in aggregate["policy_summaries"]:
            ps_rows = [r for r in s_rows if r["policy"] == pname]
            if ps_rows:
                lines.append(
                    f"| {pname} | "
                    f"{np.median([r.get('G', 0) for r in ps_rows]):.4f} | "
                    f"{np.median([r.get('total_compute', 0) for r in ps_rows]):.1f} | "
                    f"{np.median([r.get('R_old', 0) for r in ps_rows]):.3f} | "
                    f"{np.median([r.get('X_unseen', 0) for r in ps_rows]):.3f} |"
                )
        lines.append("")

    return "\n".join(lines)


# ── Statistics ─────────────────────────────────────────────────────

def _bootstrap_ci(values: list[float], n_resamples: int = 10000) -> list[float]:
    """Bootstrap 95% confidence interval for median."""
    if len(values) < 5:
        return [0.0, 0.0]
    rng = np.random.RandomState(42)
    medians: list[float] = []
    arr = np.array(values)
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        medians.append(float(np.median(sample)))
    medians.sort()
    lower = medians[int(0.025 * len(medians))]
    upper = medians[int(0.975 * len(medians))]
    return [round(lower, 4), round(upper, 4)]


# ── Plotting (optional) ────────────────────────────────────────────

def plot_results(raw_results: list[dict[str, Any]], output_dir: Path) -> None:
    """Generate diagnostic plots from raw results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-policy G distribution
    policy_names = sorted(set(r["policy"] for r in raw_results))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # G distribution boxplot
    ax = axes[0, 0]
    data = []
    labels = []
    for pname in policy_names:
        Gs = [r.get("G", 0) for r in raw_results if r["policy"] == pname and "error" not in r]
        if Gs:
            data.append(Gs)
            labels.append(pname)
    ax.boxplot(data, labels=labels)
    ax.set_title("G Distribution by Policy")
    ax.set_ylabel("G (Net Learning Gain)")
    ax.grid(True, alpha=0.3)

    # Compute vs G scatter
    ax = axes[0, 1]
    for pname in policy_names:
        p_rows = [r for r in raw_results if r["policy"] == pname and "error" not in r]
        if p_rows:
            compute = [r.get("total_compute", 0) for r in p_rows]
            Gs = [r.get("G", 0) for r in p_rows]
            ax.scatter(compute, Gs, alpha=0.5, label=pname, s=10)
    ax.set_xlabel("Total Compute")
    ax.set_ylabel("G")
    ax.set_title("Compute Cost vs Learning Gain")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Per-scenario G heatmap
    ax = axes[1, 0]
    scenarios_list = sorted(set(r["scenario"] for r in raw_results))
    matrix = np.zeros((len(policy_names), len(scenarios_list)))
    for pi, pname in enumerate(policy_names):
        for sj, sid in enumerate(scenarios_list):
            vals = [r.get("G", 0) for r in raw_results
                    if r["policy"] == pname and r["scenario"] == sid and "error" not in r]
            matrix[pi, sj] = np.median(vals) if vals else 0.0
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(scenarios_list)))
    ax.set_xticklabels(scenarios_list, fontsize=7)
    ax.set_yticks(range(len(policy_names)))
    ax.set_yticklabels(policy_names, fontsize=8)
    ax.set_title("Median G: Policy x Scenario")
    plt.colorbar(im, ax=ax)

    # Falsification gates summary
    ax = axes[1, 1]
    ax.axis("off")
    aggregate = _build_aggregate_for_plot(raw_results)
    gates = aggregate.get("falsification_gates", [])
    status_text = "FALSIFICATION GATES:\n\n"
    for g in gates:
        s = "PASS" if g["passed"] else "FAIL"
        status_text += f"[{s}] Gate {g['gate_id']}: {g['evidence'][:80]}\n"
    ax.text(0.05, 0.95, status_text, transform=ax.transAxes,
            fontfamily="monospace", fontsize=8, verticalalignment="top")

    fig.suptitle("Learning-Gain Simulation Results", fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "results_plot.png", dpi=150)
    plt.close(fig)


def _build_aggregate_for_plot(raw_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Quick aggregate for plotting."""
    scenarios = sorted(set(r["scenario"] for r in raw_results))
    return {
        "falsification_gates": check_falsification_gates(raw_results, scenarios),
    }
