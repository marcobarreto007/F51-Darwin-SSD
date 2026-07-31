from __future__ import annotations

from .dependencies import *  # noqa: F403
from .config import DarwinOrganismConfig
from .control import DarwinOrganism
from .checkpoint_root import preflight_checkpoint_root
from .support import *  # noqa: F403

# ═══════════════════════════════════════════════════════════
# CANARY GUARDS (Task 4)
# ═══════════════════════════════════════════════════════════

def validate_canary_paths(*, canonical_root: Path, canary_root: Path) -> None:
    canonical = canonical_root.resolve()
    canary = canary_root.resolve()
    # The isolated root must remain inside the external checkpoint workspace;
    # sibling, parent, repository-local, and arbitrary absolute paths are not
    # valid canary artifact destinations.
    if canary == canonical or canonical not in canary.parents:
        raise ValueError("canary checkpoint root must be isolated from canonical root")


def validate_canary_shape(command: str, cycles: int, steps: int) -> None:
    if command != "cycle" or cycles != 1 or steps != 250:
        raise ValueError("canary requires exactly one cycle and 250 steps")


def validate_canary_lineage(
    *,
    resume: str | None,
    fresh_start: bool,
    base_checkpoint_sha256: str | None,
) -> None:
    """Require exactly one explicit canary lineage source."""
    if bool(resume) == bool(fresh_start):
        raise ValueError(
            "canary requires exactly one of --resume or --fresh-start"
        )
    if resume and not base_checkpoint_sha256:
        raise ValueError("resumed canary requires --base-checkpoint-sha256")
    if fresh_start and base_checkpoint_sha256:
        raise ValueError(
            "fresh-start canary must not declare a base checkpoint SHA-256"
        )


def validate_run247_lineage(
    command: str, resume: str | None, fresh_start: bool
) -> None:
    """Refuse an accidental new lineage in unattended training mode."""
    if command not in {"run247", "train-budget"}:
        return
    if resume and fresh_start:
        raise ValueError("--resume and --fresh-start are mutually exclusive")
    if not resume and not fresh_start:
        raise ValueError(
            f"{command} requires --resume to preserve model, optimizer, and "
            "gradient-mutational state; use --fresh-start only for an "
            "explicitly approved new lineage"
        )


def validate_train_budget(command: str, max_train_tokens: int | None) -> None:
    if command != "train-budget":
        return
    if (
        max_train_tokens is None
        or isinstance(max_train_tokens, bool)
        or max_train_tokens <= 0
    ):
        raise ValueError("train-budget requires a positive --max-train-tokens")


def remaining_budget_steps(
    train_tokens_seen: int,
    max_train_tokens: int,
    batch_tokens: int,
    max_steps_per_cycle: int,
) -> int:
    if train_tokens_seen < 0:
        raise ValueError("train_tokens_seen must be non-negative")
    if max_train_tokens <= 0:
        raise ValueError("max_train_tokens must be positive")
    if batch_tokens <= 0:
        raise ValueError("batch_tokens must be positive")
    if max_steps_per_cycle <= 0:
        raise ValueError("max_steps_per_cycle must be positive")
    remaining = max(0, max_train_tokens - train_tokens_seen)
    return min(max_steps_per_cycle, remaining // batch_tokens)


def finish_one_shot_cycle(organism: DarwinOrganism) -> None:
    organism._join_pending_save()


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════


def _build_parser(workspace_paths, default_config):
    workspace_paths = WorkspacePaths.from_project(ROOT)
    parser = argparse.ArgumentParser(description="F51 Darwin Organism — O bicho VIVO")
    parser.add_argument(
        "command",
        choices=[
            "bootstrap",
            "cycle",
            "status",
            "evolve",
            "school",
            "serve",
            "run247",
            "train-budget",
        ],
    )
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--token-bin", default=None)
    parser.add_argument("--tokenizer", default=str(workspace_paths.tokenizer / "f51_bpe_80k"))
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Explicitly create a new lineage; required for run247 without --resume.",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--sampler-mode",
        choices=["sequential", "permuted_blocks"],
        default="sequential",
        help="Training data-order contract. Use permuted_blocks only for a new lineage.",
    )
    parser.add_argument(
        "--sampler-version",
        type=int,
        choices=[1],
        default=1,
        help="Version of the selected sampler contract.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Mid-cycle durability save every N total steps, in addition to "
        "the save at each cycle boundary (0 disables mid-cycle saves).",
    )
    parser.add_argument("--eval-every", type=int, default=500, help="Eval every N steps (run247)")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "dae_hybrid"],
        default="adamw",
        help="Explicit optimizer identity; dae_hybrid is required for DAE full runs.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Linear LR warmup over N steps; critical after a cold resume or "
        "optimizer rebuild (0 disables).",
    )
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        default=0,
        help="Cosine LR decay over N steps after warmup (0 = legacy warmup-only).",
    )
    parser.add_argument(
        "--lr-final-ratio",
        type=float,
        default=0.1,
        help="LR floor as fraction of initial_lr after cosine decay (default 0.1).",
    )
    parser.add_argument(
        "--dopamine-reward-weight",
        type=float,
        default=0.0,
        help="Dopamine-driven intrinsic reward: scales gradient by "
        "(1 + (1-dopamine)*weight). 0.15 = tesao real de aprender.",
    )
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation micro-batches per optimizer.step().",
    )
    parser.add_argument(
        "--causal-mode",
        choices=["disabled", "control", "shadow", "enforce"],
        default="disabled",
        help="Causal training arm. Non-disabled modes require explicit v8 migration.",
    )
    parser.add_argument(
        "--causal-v8-migration",
        action="store_true",
        default=False,
        help="Explicitly authorize checkpoint schema v8 for causal/loss-v2 training.",
    )
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    parser.add_argument("--port", type=int, default=5151)
    # ── Autonomous Drives (MUE-X inspired) ──
    parser.add_argument(
        "--autonomous-drives",
        action="store_true",
        default=True,
        help="Enable autonomous drives: stagnation detection, emotional modulation, organ audit.",
    )
    parser.add_argument(
        "--no-autonomous-drives",
        action="store_false",
        dest="autonomous_drives",
        help="Disable autonomous drives (legacy behaviour).",
    )
    parser.add_argument(
        "--organ-rl",
        action="store_true",
        default=True,
        help="Enable RL optimizer for organ configuration selection.",
    )
    parser.add_argument(
        "--no-organ-rl",
        action="store_false",
        dest="organ_rl",
        help="Disable organ RL optimizer.",
    )
    # ── Blockchain & Senate ──
    parser.add_argument("--blockchain-enabled", action="store_true", default=False,
                        help="Enable organ blockchain ledger")
    parser.add_argument("--blockchain-path", type=str, default=None,
                        help="Path to blockchain JSONL file")
    parser.add_argument("--senate-enabled", action="store_true", default=False,
                        help="Enable OrganSenate for dynamic loss weight redistribution")
    parser.add_argument("--senate-interval", type=int, default=500,
                        help="Steps between senate convenings (default: 500, one per cycle)")
    parser.add_argument("--senate-min-weight", type=float, default=0.02,
                        help="Minimum loss weight for any active organ")
    parser.add_argument("--senate-death-threshold", type=float, default=0.05,
                        help="Reputation score below which an organ is marked for death")
    parser.add_argument("--verify-blockchain", action="store_true", default=False,
                        help="Verify full blockchain integrity before starting")
    # ── Canary (Task 4) ──
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--metrics-jsonl", default=None)
    parser.add_argument("--canary-metadata-json", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--holdout-tokens", type=int, default=0)
    parser.add_argument("--holdout-batches", type=int, default=16)
    parser.add_argument("--holdout-seed", type=int, default=999)
    parser.add_argument("--require-clean-worktree", action="store_true")
    parser.add_argument(
        "--source-git-commit",
        default=None,
        help="Commit identity for a verified git-archive deployment without .git metadata.",
    )
    parser.add_argument("--base-checkpoint-sha256", default=None)
    return parser


def _prepare_args(args, parser):
    try:
        validate_run247_lineage(args.command, args.resume, args.fresh_start)
        validate_train_budget(args.command, args.max_train_tokens)
    except ValueError as exc:
        parser.error(str(exc))

    needs_tokens = {
        "bootstrap",
        "cycle",
        "evolve",
        "school",
        "serve",
        "run247",
        "train-budget",
    }
    if args.command in needs_tokens and args.token_bin is None:
        args.token_bin = str(resolve_feast_token_bin(ROOT, require=True))

    # ── Canary pre-validation (Task 4) ──
    frozen_git_commit: str | None = None
    if args.canary:
        validate_canary_shape(args.command, args.cycles, args.steps)
        if not all((args.checkpoint_root, args.metrics_jsonl,
                    args.canary_metadata_json, args.run_id)):
            parser.error("--canary requires isolated root, metrics, metadata, and run ID")
        try:
            validate_canary_lineage(
                resume=args.resume,
                fresh_start=args.fresh_start,
                base_checkpoint_sha256=args.base_checkpoint_sha256,
            )
        except ValueError as exc:
            parser.error(str(exc))
        canary_model_raw = yaml.safe_load(
            Path(args.config).read_text(encoding="utf-8")
        )
        canonical_root = Path(
            canary_model_raw.get("checkpoint_root")
            or resolve_organism_checkpoints_root(ROOT)
        )
        if not canonical_root.is_absolute():
            canonical_root = ROOT / canonical_root
        validate_canary_paths(canonical_root=canonical_root, canary_root=Path(args.checkpoint_root))
        try:
            commit, dirty = tracked_worktree_state(
                ROOT,
                source_commit=args.source_git_commit,
            )
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        if args.require_clean_worktree and dirty:
            parser.error(f"tracked worktree is dirty: {dirty}")
        if args.resume:
            actual_sha = sha256_file(Path(args.resume))
            if actual_sha.lower() != args.base_checkpoint_sha256.lower():
                parser.error("base checkpoint SHA-256 mismatch")
        frozen_git_commit = commit
    return frozen_git_commit


def _build_organism(args, parser, workspace_paths, inventory_main):
    if args.command == "serve":
        if args.resume is None:
            from f51_darwin.serving.runtime import _latest_checkpoint

            args.resume = str(_latest_checkpoint(ROOT))
    if args.command == "status":
        if inventory_main is None:
            parser.error("status command requires the inventory adapter")
        raise SystemExit(inventory_main())
    # ── Checkpoint isolation: read checkpoint_root from model config YAML ──
    model_raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    yaml_checkpoint_root = model_raw.get("checkpoint_root")
    effective_checkpoint_root = (
        args.checkpoint_root
        if args.checkpoint_root is not None
        else str(yaml_checkpoint_root or workspace_paths.checkpoints)
    )
    resolved_checkpoint_root = Path(effective_checkpoint_root)
    if not resolved_checkpoint_root.is_absolute():
        resolved_checkpoint_root = ROOT / resolved_checkpoint_root
    resolved_checkpoint_root = resolved_checkpoint_root.resolve()
    if args.command in {"run247", "train-budget"}:
        preflight_checkpoint_root(
            root=resolved_checkpoint_root,
            model_raw=model_raw,
            mode="resume" if args.resume else "fresh_start",
            resume=args.resume,
        )
    effective_checkpoint_root = str(resolved_checkpoint_root)
    cfg = DarwinOrganismConfig(
        project_root=str(ROOT),
        device=args.device,
        corpus_dir=str(workspace_paths.corpus / "approved"),
        checkpoint_root=effective_checkpoint_root,
        tokenizer_dir=args.tokenizer,
        runs_dir=str(workspace_paths.runs),
        batch_size=args.batch_size,
        block_size=args.block_size,
        sampler_mode=args.sampler_mode,
        sampler_version=args.sampler_version,
        learning_rate=args.lr,
        optimizer_name=args.optimizer,
        max_steps_per_cycle=args.steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        heartbeat_enabled=True,
        ghost_enabled=True,
        jepa_enabled=True,
        curiosity_enabled=True,
        spider_enabled=True,
        warmup_steps=args.warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_final_ratio=args.lr_final_ratio,
        accum_steps=max(1, args.accum_steps),
        dopamine_reward_weight=args.dopamine_reward_weight,
        max_train_tokens=args.max_train_tokens,
        causal_mode=args.causal_mode,
        causal_v8_migration=args.causal_v8_migration,
        # ── Autonomous Drives ──
        autonomous_drives=args.autonomous_drives,
        organ_rl_enabled=args.organ_rl,
        # ── Blockchain & Senate ──
        blockchain_enabled=args.blockchain_enabled,
        blockchain_path=args.blockchain_path or "workspace/runtime/organism/blockchain/blocks.jsonl",
        senate_enabled=args.senate_enabled,
        senate_interval=args.senate_interval,
        senate_min_weight=args.senate_min_weight,
        senate_death_threshold=args.senate_death_threshold,
        # Canary
        holdout_tokens=args.holdout_tokens,
        holdout_batches=args.holdout_batches,
        holdout_seed=args.holdout_seed,
        metrics_jsonl=args.metrics_jsonl,
        canary_run_id=args.run_id,
        canary_metadata_json=args.canary_metadata_json,
    )
    if args.canary:
        cfg.checkpoint_root = str(Path(args.checkpoint_root).resolve())
    cfg.checkpoint_creation_mode = "resume" if args.resume else "fresh_start"

    org = DarwinOrganism(cfg, args.config)
    return org


def _run_command(args, org, frozen_git_commit):
    if args.command == "bootstrap":
        org.bootstrap(token_bin=args.token_bin, resume=args.resume)
        print(json.dumps(org.status(), indent=2))

    elif args.command == "cycle":
        org.bootstrap(token_bin=args.token_bin, resume=args.resume)
        # ── Canary: baseline heldout before training ──
        if args.canary:
            org.metric_channels = MetricChannels.create(window_size=20)
            org._metrics_started_at = time.time()
            baseline_heldout = org.evaluate_holdout(phase="baseline")
        for c in range(args.cycles):
            report = org.run_cycle(steps=args.steps)
            print(f"\nCycle {org.cycle} complete: {json.dumps(report, indent=2)}")
        # ── Canary: join async save and final heldout ──
        if args.canary:
            finish_one_shot_cycle(org)
            final_heldout = org.evaluate_holdout(phase="candidate")
            print(json.dumps({
                "status": "candidate_saved",
                "run_id": args.run_id,
                "git_commit": frozen_git_commit,
                "checkpoint_root": str(Path(args.checkpoint_root).resolve()),
                "baseline_heldout": asdict(baseline_heldout),
                "heldout": asdict(final_heldout),
            }, sort_keys=True), flush=True)

    elif args.command == "school":
        org.bootstrap(token_bin=args.token_bin, resume=args.resume)
        # Train first, then exam
        for c in range(args.cycles):
            org.run_cycle(steps=args.steps)
            org.school_cycle()

    elif args.command == "serve":
        org.bootstrap(token_bin=args.token_bin, resume=args.resume)
        from f51_darwin.serving.runtime import serve_organism

        serve_organism(org, host=args.host, port=args.port)
    elif args.command == "run247":
        _run247(args, org)
    elif args.command == "train-budget":
        _run_train_budget(args, org)


def _run_train_budget(args, org):
    org.bootstrap(token_bin=args.token_bin, resume=args.resume)
    batch_tokens = int(args.batch_size) * int(args.block_size)
    print(
        "\n"
        f"TRAIN-BUDGET INICIADO tokens={org.train_tokens_seen}/"
        f"{args.max_train_tokens} batch_tokens={batch_tokens}",
        flush=True,
    )
    try:
        while not getattr(org, "_emergency_save_requested", False):
            cycle_steps = remaining_budget_steps(
                train_tokens_seen=org.train_tokens_seen,
                max_train_tokens=args.max_train_tokens,
                batch_tokens=batch_tokens,
                max_steps_per_cycle=args.steps,
            )
            if cycle_steps == 0:
                break
            report = org.run_cycle(steps=cycle_steps)
            print(
                json.dumps(
                    {
                        "status": "budget_progress",
                        "cycle": org.cycle,
                        "step": org.total_steps,
                        "train_tokens_seen": org.train_tokens_seen,
                        "max_train_tokens": args.max_train_tokens,
                        "loss_end": report.get("loss_end"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        org._join_pending_save()
    print(
        json.dumps(
            {
                "status": "budget_complete"
                if org.train_tokens_seen + batch_tokens > args.max_train_tokens
                else "budget_paused",
                "cycle": org.cycle,
                "step": org.total_steps,
                "train_tokens_seen": org.train_tokens_seen,
                "max_train_tokens": args.max_train_tokens,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _run247(args, org):
        # ── KILL SWITCH: halt file blocks 24/7 mode regardless of caller
        # (wrapper script, direct CLI, another agent/process) ──
        kill_switch = Path(__file__).resolve().parents[3] / "TRAINING_HALTED.flag"
        if kill_switch.exists():
            print(f"TRAINING_HALTED.flag presente em {kill_switch} -- abortando run247 sem bootstrap.")
            return
        # ── 24/7 MODE: O bebê nunca dorme ──
        org.bootstrap(token_bin=args.token_bin, resume=args.resume)

        print("\n" + "─" * 60)
        print("  ORGANISMO 24/7 INICIADO — Ctrl+C para parar")
        print("─" * 60 + "\n")

        # MathGenius via env var (seguro)
        math_genius = None
        wolfram_key = os.environ.get("WOLFRAM_APP_ID")
        if wolfram_key:
            try:
                from f51_darwin.math_genius import MathGenius
                math_genius = MathGenius(
                    wolfram_key,
                    org.root / org.cfg.organism_state_dir / "math_genius",
                )
                print("  🧠 Math Genius: ativo (Wolfram)")
            except Exception as e:
                print(f"  🧠 Math Genius: falhou ({e})")
        else:
            print("  🧠 Math Genius: desativado (set WOLFRAM_APP_ID to enable)")

        steps_since_math = 0
        steps_since_generate = 0
        math_discoveries = 0
        report: dict = {}

        try:
            while True:
                # R5: a SIGINT/SIGTERM flips this flag cooperatively; stop the
                # 24/7 loop so the partial progress saved by run_cycle holds.
                if getattr(org, "_emergency_save_requested", False):
                    break
                if kill_switch.exists():
                    print(f"TRAINING_HALTED.flag presente em {kill_switch} -- parando loop 24/7 apos o ciclo atual.")
                    break
                # ── TRAIN + EVOLVE: Um ciclo completo ──
                report = org.run_cycle(steps=args.steps)
                steps_since_math += args.steps
                steps_since_generate += args.steps

                # ── MATH GENIUS: Pesquisa matemática periódica ──
                if math_genius and steps_since_math >= 5000:
                    try:
                        discoveries = math_genius.research_cycle(cycles=3)
                        if discoveries:
                            corpus_text = math_genius.to_training_corpus()
                            org.data_factory.register_candidate(
                                text=corpus_text,
                                source_type=SourceType.EDITED,
                                source_path="organism://math_genius",
                                generator_model="math_genius",
                                generator_checkpoint=f"cycle_{org.cycle}",
                                dataset_version=f"organism_cycle_{org.cycle}",
                            )
                            math_discoveries += len(discoveries)
                            print(
                                f"  🧠 +{len(discoveries)} descobertas em "
                                "quarentena; aguardando aprovacao explicita "
                                f"(total: {math_discoveries})"
                            )
                    except Exception as e:
                        print(f"  🧠 Math: {e}")
                    steps_since_math = 0

                # ── LIVE GENERATE: Auto-avaliação periódica ──
                if steps_since_generate >= 2000:
                    try:
                        org.model.eval()
                        prompts = [
                            "Quem é o Darwin?",
                            "Explique o que é um transformer.",
                            "Soli Deo Gloria significa",
                        ]
                        for prompt in prompts:
                            out = org.model.live_generate(
                                prompt, org.tokenizer,
                                max_tokens=50, temperature=0.7,
                            )
                            print(f"  💬 [{prompt[:30]}...] → {out['text'][:80]}")
                        org.model.train()
                    except Exception as e:
                        print(f"  💬 live_generate: {e}")
                    steps_since_generate = 0

                # ── HEARTBEAT ──
                now = datetime.now(timezone.utc).strftime("%H:%M:%S")
                loss = report.get("loss_end", "?")
                active = report.get("modules_active", 0)
                died = report.get("modules_died", 0)
                drives_info = ""
                drives_diag = report.get("autonomous_drives")
                if drives_diag:
                    strat = drives_diag.get("emotions", {}).get("strategy", "?")
                    mood = drives_diag.get("emotions", {}).get("mood", "?")
                    pressure = drives_diag.get("stagnation", {}).get("exploration_pressure", 1.0)
                    drives_info = f" 🧬{strat} {mood} p={pressure:.1f}x"
                print(
                    f"  [{now}] cycle={org.cycle} step={org.total_steps:>7d} "
                    f"loss={loss:.4f} active={active}🟢 died={died}☠️{drives_info}",
                    flush=True,
                )
                # R5: re-check after the cycle so a signal received during
                # run_cycle still exits the 24/7 loop instead of starting a
                # new cycle. run_cycle already saved the partial progress.
                if getattr(org, "_emergency_save_requested", False):
                    break

        except KeyboardInterrupt:
            # Fallback when the platform did not install the signal handler.
            # run_cycle may have been interrupted mid-flight; ensure a save.
            print("\n\n  ⚠️  KeyboardInterrupt — finalizando.")
            try:
                org._save_cycle(report)
            except Exception as exc:
                print(f"  ⚠️  save final falhou: {exc}")

        # R4: guarantee the (possibly async) checkpoint reaches disk before
        # the process exits, regardless of which path stopped the loop.
        org._join_pending_save()
        print("\n" + "─" * 60)
        print("  ORGANISMO PAUSADO")
        print(f"  Step final: {org.total_steps}")
        print(f"  Cycles: {org.cycle}")
        print("─" * 60)


def main(
    argv: list[str] | None = None,
    *,
    default_config: str | Path = CANONICAL_CONFIG,
    inventory_main: Any | None = None,
) -> int:
    workspace_paths = WorkspacePaths.from_project(ROOT)
    parser = _build_parser(workspace_paths, default_config)
    args = parser.parse_args(argv)
    frozen_git_commit = _prepare_args(args, parser)
    org = _build_organism(args, parser, workspace_paths, inventory_main)
    _run_command(args, org, frozen_git_commit)
    return 0
