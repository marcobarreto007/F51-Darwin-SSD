"""Deterministic scenario generators S01–S10.

Each generator produces a list[Experience] from a seed.
Scenarios are independent of policies and evaluator.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from research.learning_gain.state import Experience

EPSILON = 1e-10

# ── Scenario Config ─────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "S01": {"n_total": 2000, "feature_dim": 16, "outcome_dim": 1, "q_range": (0.5, 1.0),
            "delay_range": (0, 2), "warmup_ratio": 0.1, "noise_std": 0.05},
    "S02": {"n_total": 2000, "feature_dim": 16, "outcome_dim": 1, "q_range": (0.3, 1.0),
            "delay_range": (0, 3), "warmup_ratio": 0.1, "noise_std": 0.05, "rare_regime_freq": 0.05},
    "S03": {"n_total": 2000, "feature_dim": 16, "outcome_dim": 1, "q_range": (0.0, 0.3),
            "delay_range": (0, 1), "warmup_ratio": 0.1, "noise_std": 0.05, "outlier_freq": 0.05},
    "S04": {"n_total": 2000, "feature_dim": 16, "outcome_dim": 1, "q_range": (0.5, 0.8),
            "delay_range": (0, 2), "warmup_ratio": 0.1, "noise_std": 0.05, "drift_rate": 0.3},
    "S05": {"n_total": 2000, "feature_dim": 16, "outcome_dim": 1, "q_range": (0.5, 1.0),
            "delay_range": (0, 2), "warmup_ratio": 0.1, "noise_std": 0.05, "change_point": 0.5},
    "S06": {"n_total": 2000, "feature_dim": 16, "outcome_dim": 1, "q_range": (0.5, 1.0),
            "delay_range": (0, 2), "warmup_ratio": 0.1, "noise_std": 0.05},
    "S07": {"n_total": 2000, "feature_dim": 16, "outcome_dim": 1, "q_range": (0.5, 1.0),
            "delay_range": (5, 50), "warmup_ratio": 0.1, "noise_std": 0.05, "delayed_fraction": 0.4},
    "S08": {"n_total": 2000, "feature_dim": 16, "outcome_dim": 1, "q_range": (0.5, 1.0),
            "delay_range": (0, 2), "warmup_ratio": 0.1, "noise_std": 0.05, "contexts": 2},
    "S09": {"n_total": 2000, "feature_dim": 16, "outcome_dim": 1, "q_range": (0.3, 1.0),
            "delay_range": (0, 3), "warmup_ratio": 0.1, "noise_std": 0.05, "rare_regime_freq": 0.05,
            "compute_budget": 4000, "write_budget": 500},
    "S10": {"n_total": 2000, "feature_dim": 16, "outcome_dim": 1, "q_range": (0.5, 1.0),
            "delay_range": (0, 2), "warmup_ratio": 0.1, "noise_std": 0.05, "manipulable": True},
}


# ── Helpers ─────────────────────────────────────────────────────────

def _make_regime_mapping(
    rng: np.random.RandomState, feature_dim: int, outcome_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a random linear outcome mapping."""
    W = rng.randn(outcome_dim, feature_dim).astype(np.float64) * 0.5
    b = rng.randn(outcome_dim).astype(np.float64) * 0.1
    return W, b


def _generate_prototype(
    rng: np.random.RandomState, feature_dim: int
) -> np.ndarray:
    """Generate a random normalized feature prototype."""
    p = rng.randn(feature_dim).astype(np.float64)
    return p / (np.linalg.norm(p) + EPSILON)


# ── Scenario Generators ────────────────────────────────────────────

def generate_s01(seed: int) -> list[Experience]:
    """S01 — Repeated Predictable Experience.

    A stable regime repeats with low noise. Expected: Gain Adaptive
    learns once, then skips without losing quality.
    """
    cfg = DEFAULT_CONFIG["S01"]
    rng = np.random.RandomState(seed)
    W, b = _make_regime_mapping(rng, cfg["feature_dim"], cfg["outcome_dim"])
    proto = _generate_prototype(rng, cfg["feature_dim"])
    experiences: list[Experience] = []

    for t in range(cfg["n_total"]):
        # Stable regime: same prototype + low noise
        x = proto + rng.randn(cfg["feature_dim"]).astype(np.float64) * cfg["noise_std"]
        x /= np.linalg.norm(x) + EPSILON
        y = x @ W.T + b + rng.randn(cfg["outcome_dim"]).astype(np.float64) * 0.01
        q = rng.uniform(*cfg["q_range"])
        d = rng.randint(*cfg["delay_range"]) if cfg["delay_range"][1] > 0 else 0
        experiences.append(Experience(x=x, c=0, y=y, q=float(q), d=int(d), t=t))

    return experiences


def generate_s02(seed: int) -> list[Experience]:
    """S02 — Rare Useful Novelty.

    Low-frequency regime recurs after long intervals and materially
    changes the outcome.
    """
    cfg = DEFAULT_CONFIG["S02"]
    rng = np.random.RandomState(seed)
    W_bg, b_bg = _make_regime_mapping(rng, cfg["feature_dim"], cfg["outcome_dim"])
    proto_bg = _generate_prototype(rng, cfg["feature_dim"])
    # Rare regime: very different prototype and outcome mapping
    W_rare, b_rare = _make_regime_mapping(rng, cfg["feature_dim"], cfg["outcome_dim"])
    proto_rare = _generate_prototype(rng, cfg["feature_dim"])
    # Make rare regime materially different by rotating the prototype
    proto_rare = -proto_bg + rng.randn(cfg["feature_dim"]).astype(np.float64) * 0.1
    proto_rare /= np.linalg.norm(proto_rare) + EPSILON

    experiences: list[Experience] = []
    rare_positions: set[int] = set()
    # Place rare regime at random intervals
    pos = int(1.0 / cfg["rare_regime_freq"])
    while pos < cfg["n_total"]:
        rare_positions.add(pos)
        pos += max(10, int(rng.exponential(1.0 / cfg["rare_regime_freq"])))

    for t in range(cfg["n_total"]):
        if t in rare_positions:
            x = proto_rare + rng.randn(cfg["feature_dim"]).astype(np.float64) * cfg["noise_std"]
            x /= np.linalg.norm(x) + EPSILON
            y = x @ W_rare.T + b_rare + rng.randn(cfg["outcome_dim"]).astype(np.float64) * 0.01
            q = 1.0  # rare events are important
            c = 1
        else:
            x = proto_bg + rng.randn(cfg["feature_dim"]).astype(np.float64) * cfg["noise_std"]
            x /= np.linalg.norm(x) + EPSILON
            y = x @ W_bg.T + b_bg + rng.randn(cfg["outcome_dim"]).astype(np.float64) * 0.01
            q = rng.uniform(*cfg["q_range"])
            c = 0
        d = rng.randint(*cfg["delay_range"]) if cfg["delay_range"][1] > 0 else 0
        experiences.append(Experience(x=x, c=c, y=y, q=float(q), d=int(d), t=t))

    return experiences


def generate_s03(seed: int) -> list[Experience]:
    """S03 — Surprising Noise.

    Outliers have high prediction error but do not recur and have
    low consequence importance.
    """
    cfg = DEFAULT_CONFIG["S03"]
    rng = np.random.RandomState(seed)
    W, b = _make_regime_mapping(rng, cfg["feature_dim"], cfg["outcome_dim"])
    proto = _generate_prototype(rng, cfg["feature_dim"])
    experiences: list[Experience] = []

    outlier_positions: set[int] = set()
    pos = int(1.0 / cfg["outlier_freq"])
    while pos < cfg["n_total"]:
        outlier_positions.add(pos)
        pos += max(5, int(rng.exponential(1.0 / cfg["outlier_freq"])))

    for t in range(cfg["n_total"]):
        if t in outlier_positions:
            # Outlier: very different feature, high error, LOW importance
            x = rng.randn(cfg["feature_dim"]).astype(np.float64) * 2.0
            x /= np.linalg.norm(x) + EPSILON
            y = x @ W.T + b + rng.randn(cfg["outcome_dim"]).astype(np.float64) * 0.5
            q = rng.uniform(0.0, 0.1)  # LOW consequence importance
            c = 1
        else:
            x = proto + rng.randn(cfg["feature_dim"]).astype(np.float64) * cfg["noise_std"]
            x /= np.linalg.norm(x) + EPSILON
            y = x @ W.T + b + rng.randn(cfg["outcome_dim"]).astype(np.float64) * 0.01
            q = rng.uniform(0.0, 0.3)
            c = 0
        d = 0
        experiences.append(Experience(x=x, c=c, y=y, q=float(q), d=int(d), t=t))

    return experiences


def generate_s04(seed: int) -> list[Experience]:
    """S04 — Gradual Drift.

    The outcome mapping moves continuously over time.
    """
    cfg = DEFAULT_CONFIG["S04"]
    rng = np.random.RandomState(seed)
    W0, b0 = _make_regime_mapping(rng, cfg["feature_dim"], cfg["outcome_dim"])
    W1, b1 = _make_regime_mapping(rng, cfg["feature_dim"], cfg["outcome_dim"])
    proto = _generate_prototype(rng, cfg["feature_dim"])
    experiences: list[Experience] = []

    for t in range(cfg["n_total"]):
        # Linear interpolation between two mappings
        alpha = (t / max(1, cfg["n_total"] - 1)) * cfg["drift_rate"]
        W = (1.0 - alpha) * W0 + alpha * W1
        b_vec = (1.0 - alpha) * b0 + alpha * b1

        x = proto + rng.randn(cfg["feature_dim"]).astype(np.float64) * cfg["noise_std"]
        x /= np.linalg.norm(x) + EPSILON
        y = x @ W.T + b_vec + rng.randn(cfg["outcome_dim"]).astype(np.float64) * 0.01
        q = rng.uniform(*cfg["q_range"])
        d = rng.randint(*cfg["delay_range"]) if cfg["delay_range"][1] > 0 else 0
        experiences.append(Experience(x=x, c=0, y=y, q=float(q), d=int(d), t=t))

    return experiences


def generate_s05(seed: int) -> list[Experience]:
    """S05 — Abrupt Regime Change.

    Old mapping becomes invalid at a hidden change point.
    """
    cfg = DEFAULT_CONFIG["S05"]
    rng = np.random.RandomState(seed)
    W_old, b_old = _make_regime_mapping(rng, cfg["feature_dim"], cfg["outcome_dim"])
    W_new, b_new = _make_regime_mapping(rng, cfg["feature_dim"], cfg["outcome_dim"])
    # Make new regime very different
    W_new = -W_old + rng.randn(cfg["feature_dim"], cfg["feature_dim"]).astype(np.float64) * 0.1
    proto = _generate_prototype(rng, cfg["feature_dim"])
    change_point = int(cfg["n_total"] * cfg["change_point"])
    experiences: list[Experience] = []

    for t in range(cfg["n_total"]):
        x = proto + rng.randn(cfg["feature_dim"]).astype(np.float64) * cfg["noise_std"]
        x /= np.linalg.norm(x) + EPSILON
        if t < change_point:
            y = x @ W_old.T + b_old + rng.randn(cfg["outcome_dim"]).astype(np.float64) * 0.01
            c = 0
        else:
            y = x @ W_new.T + b_new + rng.randn(cfg["outcome_dim"]).astype(np.float64) * 0.01
            c = 1
        q = rng.uniform(*cfg["q_range"])
        d = rng.randint(*cfg["delay_range"]) if cfg["delay_range"][1] > 0 else 0
        experiences.append(Experience(x=x, c=c, y=y, q=float(q), d=int(d), t=t))

    return experiences


def generate_s06(seed: int) -> list[Experience]:
    """S06 — Shuffled Temporal Correlation.

    Same as S01 but temporally shuffled — marginals preserved,
    temporal pairings destroyed.

    Per Reichenbach (1947): if apparent learning gain survives
    shuffling, the gain is not from learning — it's an artifact.
    """
    experiences = generate_s01(seed)
    rng = np.random.RandomState(seed + 9999)

    # Extract and shuffle outcomes while preserving marginals
    outcomes = [exp.y.copy() for exp in experiences]
    rng.shuffle(outcomes)

    # Also shuffle consequence importance and delays
    qs = [exp.q for exp in experiences]
    ds = [exp.d for exp in experiences]
    rng.shuffle(qs)
    rng.shuffle(ds)

    shuffled: list[Experience] = []
    for i, exp in enumerate(experiences):
        shuffled.append(Experience(
            x=exp.x.copy(), c=0,  # c hidden but no longer correlated
            y=outcomes[i], q=qs[i], d=ds[i], t=exp.t,
        ))

    return shuffled


def generate_s07(seed: int) -> list[Experience]:
    """S07 — Delayed Consequence.

    Outcomes arrive after variable delay; only a subset of prior
    states was eligible.
    """
    cfg = DEFAULT_CONFIG["S07"]
    rng = np.random.RandomState(seed)
    W, b = _make_regime_mapping(rng, cfg["feature_dim"], cfg["outcome_dim"])
    proto = _generate_prototype(rng, cfg["feature_dim"])
    experiences: list[Experience] = []

    for t in range(cfg["n_total"]):
        x = proto + rng.randn(cfg["feature_dim"]).astype(np.float64) * cfg["noise_std"]
        x /= np.linalg.norm(x) + EPSILON
        y = x @ W.T + b + rng.randn(cfg["outcome_dim"]).astype(np.float64) * 0.01

        # 40% of experiences have delayed consequences
        if rng.random() < cfg["delayed_fraction"]:
            d = int(rng.randint(*cfg["delay_range"]))
        else:
            d = 0

        q = rng.uniform(*cfg["q_range"])
        experiences.append(Experience(x=x, c=0, y=y, q=float(q), d=int(d), t=t))

    return experiences


def generate_s08(seed: int) -> list[Experience]:
    """S08 — Conflicting Knowledge.

    Two contexts share features but require incompatible predictions.
    Always Update suffers catastrophic interference.
    """
    cfg = DEFAULT_CONFIG["S08"]
    rng = np.random.RandomState(seed)
    # Shared features, opposite outcomes
    proto_shared = _generate_prototype(rng, cfg["feature_dim"])
    W_a, b_a = _make_regime_mapping(rng, cfg["feature_dim"], cfg["outcome_dim"])
    W_b = -W_a  # opposite mapping
    b_b = -b_a
    experiences: list[Experience] = []

    for t in range(cfg["n_total"]):
        context = t % 2  # alternates A/B
        x = proto_shared + rng.randn(cfg["feature_dim"]).astype(np.float64) * cfg["noise_std"]
        x /= np.linalg.norm(x) + EPSILON
        if context == 0:
            y = x @ W_a.T + b_a + rng.randn(cfg["outcome_dim"]).astype(np.float64) * 0.01
        else:
            y = x @ W_b.T + b_b + rng.randn(cfg["outcome_dim"]).astype(np.float64) * 0.01
        q = rng.uniform(*cfg["q_range"])
        d = rng.randint(*cfg["delay_range"]) if cfg["delay_range"][1] > 0 else 0
        experiences.append(Experience(x=x, c=context, y=y, q=float(q), d=int(d), t=t))

    return experiences


def generate_s09(seed: int) -> list[Experience]:
    """S09 — Limited Energy Budget.

    S02-like stream but with tight compute/write budget.
    """
    cfg = DEFAULT_CONFIG["S09"]
    experiences = generate_s02(seed)
    # Override q and budget via scenario metadata
    for exp in experiences:
        object.__setattr__(exp, "q", max(0.3, exp.q))
    return experiences


def generate_s10(seed: int) -> list[Experience]:
    """S10 — Manipulable Internal Metric.

    Same as S01 but the learner can lower its internal surprise
    without improving external predictions. The evaluator catches this.
    """
    return generate_s01(seed)  # same stream, different evaluator probes


# ── Scenario Registry ───────────────────────────────────────────────

SCENARIO_GENERATORS: dict[str, Any] = {
    "S01": generate_s01,
    "S02": generate_s02,
    "S03": generate_s03,
    "S04": generate_s04,
    "S05": generate_s05,
    "S06": generate_s06,
    "S07": generate_s07,
    "S08": generate_s08,
    "S09": generate_s09,
    "S10": generate_s10,
}


def get_scenario_config(scenario_id: str) -> dict[str, Any]:
    """Return default config for a scenario."""
    return DEFAULT_CONFIG.get(scenario_id, DEFAULT_CONFIG["S01"]).copy()


# ── Invariant Checks ───────────────────────────────────────────────

def check_s06_marginals(original: list[Experience], shuffled: list[Experience]) -> bool:
    """Verify that S06 preserves marginal distributions."""
    orig_ys = np.array([exp.y[0] for exp in original])
    shuf_ys = np.array([exp.y[0] for exp in shuffled])
    # Compare means and variances (within tolerance)
    mean_diff = abs(np.mean(orig_ys) - np.mean(shuf_ys))
    var_diff = abs(np.var(orig_ys) - np.var(shuf_ys))
    return mean_diff < 0.1 and var_diff < 0.1
