"""Inference demo — GainAdaptive vs Frozen on S01+S02+S05.

Shows: predictions vs ground truth, update decisions, learning curves.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

from research.learning_gain.state import (
    LearnerState, Experience, UpdateAction, cosine_similarity,
    predict, execute_action,
)
from research.learning_gain.scenarios import generate_s01, generate_s02, generate_s05
from research.learning_gain.policies import FrozenPolicy, GainAdaptivePolicy


def run_inference_demo():
    print("=" * 72)
    print("  LEARNING-GAIN INFERENCE DEMO")
    print("  GainAdaptive vs Frozen — S01 (estável) + S02 (novidade rara) + S05 (mudança abrupta)")
    print("=" * 72)

    for scenario_name, scenario_gen in [
        ("S01 — Estável", lambda: generate_s01(42)),
        ("S02 — Novidade Rara", lambda: generate_s02(42)),
        ("S05 — Mudança Abrupta", lambda: generate_s05(42)),
    ]:
        print(f"\n{'─'*72}")
        print(f"  {scenario_name}")
        print(f"{'─'*72}")

        experiences = scenario_gen()
        n = len(experiences)
        warmup_end = max(1, n // 10)
        mid = n // 2

        # ── GainAdaptive ──
        ga = GainAdaptivePolicy(
            similarity_threshold=0.3,
            lambda_write=0.02,
            lambda_interference=0.05,
            lambda_compute=0.005,
        )
        state_ga = LearnerState(max_slots=16, max_traces=64)

        # Warm-up
        for exp in experiences[:warmup_end]:
            _, _, matched = predict(state_ga, exp.x)
            if matched is None or cosine_similarity(exp.x, state_ga.slots[matched].prototype) < 0.5:
                state_ga, _, _ = execute_action(state_ga, UpdateAction.ALLOCATE, exp, None, lr=0.05)
            else:
                state_ga, _, _ = execute_action(state_ga, UpdateAction.LOCAL_UPDATE, exp, matched, lr=0.05)

        # ── Frozen ──
        frozen = FrozenPolicy()
        state_frozen = LearnerState(max_slots=16, max_traces=64)
        for exp in experiences[:warmup_end]:
            _, _, matched = predict(state_frozen, exp.x)
            if matched is None or cosine_similarity(exp.x, state_frozen.slots[matched].prototype) < 0.5:
                state_frozen, _, _ = execute_action(state_frozen, UpdateAction.ALLOCATE, exp, None, lr=0.05)
            else:
                state_frozen, _, _ = execute_action(state_frozen, UpdateAction.LOCAL_UPDATE, exp, matched, lr=0.05)

        # Track errors over time
        ga_errors: list[float] = []
        frozen_errors: list[float] = []
        ga_actions: list[str] = []
        ga_slot_counts: list[int] = []

        for exp in experiences[warmup_end:]:
            # GA prediction
            pred_ga, conf_ga, matched_ga = predict(state_ga, exp.x)
            err_ga = float(np.sum((pred_ga - exp.y) ** 2))
            ga_errors.append(err_ga)

            # GA decision
            action = ga.decide_action(state_ga, exp, pred_ga, conf_ga, matched_ga)
            ga.record_action(action)
            ga_actions.append(action.value)
            try:
                state_ga, _, _ = execute_action(state_ga, action, exp, matched_ga)
            except Exception:
                pass
            if matched_ga is not None and matched_ga < len(state_ga.slots):
                state_ga.slots[matched_ga].recurrence_count += 1
            ga_slot_counts.append(len(state_ga.slots))

            # Frozen prediction
            pred_fr, conf_fr, matched_fr = predict(state_frozen, exp.x)
            err_fr = float(np.sum((pred_fr - exp.y) ** 2))
            frozen_errors.append(err_fr)
            action_fr = frozen.decide_action(state_frozen, exp, pred_fr, conf_fr, matched_fr)
            state_frozen, _, _ = execute_action(state_frozen, action_fr, exp, matched_fr)

        # ── Report ──
        post_warmup = n - warmup_end

        # Split into early and late
        quarter = post_warmup // 4
        ga_early = np.mean(ga_errors[:quarter]) if quarter > 0 else 0
        ga_late = np.mean(ga_errors[-quarter:]) if quarter > 0 else 0
        fr_early = np.mean(frozen_errors[:quarter]) if quarter > 0 else 0
        fr_late = np.mean(frozen_errors[-quarter:]) if quarter > 0 else 0

        # In S05, split by regime change
        if "S05" in scenario_name:
            change_idx = max(0, mid - warmup_end)
            ga_pre = np.mean(ga_errors[max(0,change_idx-quarter//2):change_idx]) if change_idx > 0 else 0
            ga_post = np.mean(ga_errors[change_idx:change_idx+quarter//2]) if change_idx+quarter//2 < len(ga_errors) else 0
            fr_pre = np.mean(frozen_errors[max(0,change_idx-quarter//2):change_idx]) if change_idx > 0 else 0
            fr_post = np.mean(frozen_errors[change_idx:change_idx+quarter//2]) if change_idx+quarter//2 < len(frozen_errors) else 0

        print(f"  Experiências: {n} (warmup={warmup_end}, avaliação={post_warmup})")
        print(f"  Slots GA: {len(state_ga.slots)} | Frozen: {len(state_frozen.slots)}")
        print(f"  Compute GA: {state_ga.total_compute_spent:.0f} | Frozen: {state_frozen.total_compute_spent:.0f}")
        print(f"  Writes GA: {state_ga.total_writes_spent} | Frozen: {state_frozen.total_writes_spent}")
        print(f"  Interference GA: {state_ga.total_interference_events} | Frozen: {state_frozen.total_interference_events}")
        print()
        print(f"  Erro médio — GA: early={ga_early:.4f} late={ga_late:.4f}")
        print(f"  Erro médio — Frozen: early={fr_early:.4f} late={fr_late:.4f}")
        if "S05" in scenario_name:
            print(f"  S05 pré-mudança — GA: {ga_pre:.4f} Frozen: {fr_pre:.4f}")
            print(f"  S05 pós-mudança — GA: {ga_post:.4f} Frozen: {fr_post:.4f}")
        print()
        print(f"  Ações GA: skips={ga.n_skips} updates={ga.n_updates} allocs={ga.n_allocates} cons={ga.n_consolidates}")
        print(f"  Ações Frozen: skips={frozen.n_skips} updates={frozen.n_updates}")

        # Show sample predictions
        print(f"\n  Amostras de predição (final da stream):")
        sample_indices = list(range(post_warmup - 10, post_warmup))
        for i in sample_indices:
            if i < len(ga_errors):
                exp = experiences[warmup_end + i]
                pred_ga_i, conf_ga_i, _ = predict(state_ga, exp.x)
                pred_fr_i, conf_fr_i, _ = predict(state_frozen, exp.x)
                true_y = exp.y[0]
                action_i = ga_actions[i] if i < len(ga_actions) else "?"
                print(f"    t={exp.t:4d} | true={true_y:+.4f} | "
                      f"GA={pred_ga_i[0]:+.4f} (c={conf_ga_i:.2f}) | "
                      f"FR={pred_fr_i[0]:+.4f} (c={conf_fr_i:.2f}) | "
                      f"GA_action={action_i}")

        # ── Verdict ──
        ga_improvement = fr_late - ga_late
        compute_ratio = state_ga.total_compute_spent / max(1, state_frozen.total_compute_spent)
        print(f"\n  VEREDITO: GA late error {ga_late:.4f} vs Frozen {fr_late:.4f} "
              f"(Δ={ga_improvement:+.4f}, compute ratio={compute_ratio:.2f}x)")
        if ga_improvement > 0.001:
            print(f"  ✓ GA MELHOR que Frozen no final (+{ga_improvement:.4f} menos erro)")
        elif ga_improvement < -0.001:
            print(f"  ✗ GA PIOR que Frozen no final ({ga_improvement:+.4f} mais erro)")
        else:
            print(f"  ~ GA EMPATA com Frozen (diferença dentro do ruído)")

    print(f"\n{'='*72}")
    print("  FIM DA DEMO")
    print(f"{'='*72}")


if __name__ == "__main__":
    run_inference_demo()
