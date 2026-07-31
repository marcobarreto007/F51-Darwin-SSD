from __future__ import annotations

from .dependencies import *  # noqa: F403

@dataclass
class DarwinOrganismConfig:
    organism_name: str = "F51-Darwin-Organism"
    version: str = "v1.0"
    project_root: str = "."
    device: str = "cuda"
    corpus_dir: str = "workspace/02_CORPUS/approved"
    tokenizer_dir: str = "workspace/tokenizer/f51_bpe_80k"
    checkpoint_root: str = "workspace/03_CHECKPOINTS"
    runs_dir: str = "workspace/runtime/runs"
    organism_state_dir: str = "workspace/runtime/organism"
    batch_size: int = 1
    block_size: int = 1024  # batch=1 melhor throughput: 142 tok/s
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.01
    optimizer_name: str = "adamw"
    max_steps_per_cycle: int = 500
    # ── Canary / observability (Task 3) ──
    holdout_tokens: int = 0
    holdout_batches: int = 16
    holdout_seed: int = 999
    metrics_jsonl: str | None = None
    canary_run_id: str | None = None
    canary_metadata_json: str | None = None
    eval_every: int = 100
    save_every: int = 100
    grad_clip: float = 1.0
    seed: int = 51
    # Data-order contract. ``sequential`` preserves legacy checkpoints;
    # ``permuted_blocks`` is the deterministic, stateless production sampler
    # for new lineages.
    sampler_mode: str = "sequential"
    sampler_version: int = 1
    replay_capacity: int = 128
    replay_sample_size: int = 8
    replay_ratio: float = 0.02  # quase zero replay, foco em dados frescos
    replay_warmup_examples: int = 4
    replay_add_every: int = 20
    min_evolution_score: float = 0.001  # Quase impossível morrer
    max_active_modules: int = 999  # Todos podem viver
    max_synthetic_ratio: float = 0.10
    heartbeat_enabled: bool = True
    ghost_enabled: bool = True
    jepa_enabled: bool = True
    curiosity_enabled: bool = True
    spider_enabled: bool = True
    # R2: linear LR warmup over N steps (0 disables). Critical right after an
    # optimizer rebuild or a cold resume so the full LR never hits at once.
    warmup_steps: int = 0
    # R2b: cosine decay phase after warmup.  lr_decay_steps=0 preserves the
    # legacy warmup-only behaviour.  When set, LR decays from initial_lr to
    # initial_lr * lr_final_ratio over lr_decay_steps following a half-cosine.
    lr_decay_steps: int = 0
    lr_final_ratio: float = 0.1
    # R3: gradient accumulation micro-batches per optimizer.step().
    accum_steps: int = 1
    # Finite long-run budget. None leaves one-shot commands unchanged.
    max_train_tokens: int | None = None
    # R4: write the cycle .pt in a background thread so training is not stalled
    # for minutes on the ~11 GB checkpoint of the 2.5B lineage.
    async_checkpoint: bool = True
    # Training-only causal bus. Disabled is the exact legacy path.
    causal_mode: str = "disabled"
    # Explicit authorization for a v7 lineage to adopt causal checkpoint v8.
    causal_v8_migration: bool = False
    dopamine_reward_weight: float = 0.0  # 0.15 = tesao real de aprender
    sleep_max_regression: float = 0.0
    causal_ledger_jsonl: str = (
        "workspace/runtime/organism/causal/events.jsonl"
    )
    # ── Autonomous Drives (MUE-X inspired) ──
    autonomous_drives: bool = True   # Enable stagnation detection, emotional modulation, organ audit
    organ_rl_enabled: bool = True    # Enable RL optimizer for organ configuration
    exploration_pressure: float = 1.0  # Current exploration pressure multiplier
    # ── Blockchain & Senate ──
    blockchain_enabled: bool = False
    blockchain_path: str = "workspace/runtime/organism/blockchain/blocks.jsonl"
    senate_enabled: bool = False
    senate_interval: int = 500         # steps between senate convenings
    senate_min_weight: float = 0.02    # minimum loss weight for any active organ
    senate_death_threshold: float = 0.05  # reputation below which organ is marked for death
    senate_rising_bonus: float = 0.10  # bonus for "rising" reputation organs
    senate_newborn_period: int = 3     # probation cycles for a newborn organ

    def __post_init__(self) -> None:
        sampler_mode = str(self.sampler_mode).strip().lower()
        if sampler_mode not in {"sequential", "permuted_blocks"}:
            raise ValueError(
                "sampler_mode must be one of: permuted_blocks, sequential"
            )
        if isinstance(self.sampler_version, bool) or self.sampler_version != 1:
            raise ValueError("sampler_version must be 1")
        self.sampler_mode = sampler_mode
        if self.max_train_tokens is not None:
            if isinstance(self.max_train_tokens, bool) or self.max_train_tokens <= 0:
                raise ValueError("max_train_tokens must be a positive integer")
            self.max_train_tokens = int(self.max_train_tokens)
        if not isinstance(self.causal_v8_migration, bool):
            raise TypeError("causal_v8_migration must be an explicit boolean")
        if (
            isinstance(self.sleep_max_regression, bool)
            or not isinstance(self.sleep_max_regression, (int, float))
            or not math.isfinite(float(self.sleep_max_regression))
            or float(self.sleep_max_regression) < 0.0
        ):
            raise ValueError(
                "sleep_max_regression must be finite and non-negative"
            )
        self.sleep_max_regression = float(self.sleep_max_regression)
        mode = str(self.causal_mode).strip().lower()
        allowed = {"disabled", "control", "shadow", "enforce"}
        if mode not in allowed:
            raise ValueError(
                "causal_mode must be one of: "
                + ", ".join(sorted(allowed))
            )
        self.causal_mode = mode
