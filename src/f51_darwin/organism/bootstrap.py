from __future__ import annotations

from .dependencies import *  # noqa: F403
from .checkpoint import *  # noqa: F403
from .causal_adapters import (
    CuriosityPriorityAdapter,
    ExplicitRequestAdapter,
    GABAInterventionAdapter,
    SpiderCalibrationAdapter,
)
from .causal_bus import AblationArm, OrganCausalBus
from .causal_ledger import CausalLedger
from .config import DarwinOrganismConfig
from .support import *  # noqa: F403


def _rewind_ledger_to_checkpoint(
    ledger: CausalLedger | None,
    resume_payload: dict[str, Any],
) -> None:
    """Truncate the causal ledger to the checkpoint's ``ledger_head``.

    On crash recovery the ledger may contain events from steps that were
    executed after the last checkpoint save.  Those steps were lost — the
    model state rewound to the checkpoint — so the ledger must be rewound
    too.  If the checkpoint head is already the current head this is a
    no-op.

    When the checkpoint head is not found in the current chain at all
    (different training session), the old ledger is archived and a fresh
    chain begins — the alternative is a hard crash that blocks all resume.
    """
    if ledger is None:
        return
    contract = resume_payload.get("causal_contract")
    if not isinstance(contract, dict):
        return
    ckpt_head = str(contract.get("ledger_head") or "")
    if not ckpt_head:
        return
    current_head = ledger.head_hash
    if current_head == ckpt_head:
        return
    kept = ledger.truncate_to_head(ckpt_head)
    if kept == 0:
        # Checkpoint belongs to a different session — the ledger chain is
        # irrecoverable.  Archive the old file and start a fresh chain so
        # resume can proceed.
        _archive_stale_ledger(ledger)
        print(
            "   ⚠️  Ledger head do checkpoint não encontrado na chain atual — "
            "ledger arquivado e novo iniciado (sessão diferente)"
        )
        return
    # Reload internal state from the truncated file.
    ledger.recover_head()


def _archive_stale_ledger(ledger: CausalLedger) -> None:
    """Rename the current ledger file so a fresh chain can start."""
    import time
    path = ledger.path
    if not path.exists():
        return
    ts = int(time.time())
    archived = path.with_name(f"{path.stem}_dead_session_{ts}{path.suffix}")
    path.rename(archived)
    # Re-initialise internal state for a fresh chain.
    ledger._head_hash = "0" * 64
    ledger._next_sequence = 0
    ledger._attempts.clear()
    ledger._fingerprint = None


class _DarwinBootstrapMixin:
    """O organismo Darwin-X VIVO — com todos os orgaos ativos."""

    def __init__(self, config: DarwinOrganismConfig, model_config_path: str):
        self.cfg = config
        self.root = Path(config.project_root)
        mark_runtime_use(self.root, actor="darwin_organism")
        self._causal_requests: list[dict[str, Any]] = []

        # Load model config
        raw = yaml.safe_load(Path(model_config_path).read_text(encoding="utf-8"))
        raw['heartbeat_enabled'] = config.heartbeat_enabled
        self.model_config = DarwinXConfig.from_mapping(raw)
        self.training_contract_id = training_contract_identity()
        # Fail before creating the causal ledger, model, or training state.
        self.checkpoint_version = checkpoint_version_for_runtime(
            loss_semantics_version=self.model_config.loss_semantics_version,
            causal_mode=self.cfg.causal_mode,
            causal_v8_migration=self.cfg.causal_v8_migration,
        )
        self._configure_causal_runtime()

        requested_device = config.device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        self.model: DarwinXModel = None
        self.tokenizer: F51BPETokenizer = None
        self.optimizer = None
        self.scaler = None

        # ORGAOS
        self.replay = ReplayBuffer(capacity=config.replay_capacity, seed=config.seed)
        self.firewall = DataFirewall(FirewallConfig(min_chars=200, quality_pass_score=0.60))
        self.brainstem = Brainstem(BrainstemConfig(max_vram_gb=80, max_loss=50, max_active_modules=8))
        self.soul = F51Soul(self.root / config.organism_state_dir)

        # FASE 1: ExpertPool + LegacyLayers + LineageTracker
        self.expert_pool = ExpertPool()
        self.legacy = LegacyLayers(self.root / config.organism_state_dir / "legacy")
        self.lineage = LineageTracker(self.root / config.runs_dir / "lineage")

        self.cycle = 0
        self.total_steps = 0
        self.train_tokens_seen = 0
        self.start_step = 0
        self.amp_dtype = None
        self._ashes_streak: dict[str, int] = {}
        self.base_checkpoint_id = ""
        self.tokenizer_id = ""
        self.resume_checkpoint_path: Path | None = None
        self.resume_online_learning_state: dict[str, Any] | None = None
        self.resume_migration: dict[str, Any] | None = None
        self.online_state_authoritative = False
        self._last_tok_s: float = 0.0
        # R4 (async checkpoint) + R5 (emergency save) bookkeeping.
        self._save_thread: threading.Thread | None = None
        self._save_error: Exception | None = None
        self._emergency_save_requested: bool = False

        # ── Autonomous Drives (MUE-X inspired) ──
        from f51_darwin.organism.autonomous_drives import AutonomousDarwinDrives
        from f51_darwin.organism.organ_rl import OrganRLOptimizer
        self.autonomous_drives = AutonomousDarwinDrives(
            enabled=getattr(config, "autonomous_drives", True)
        )
        self.organ_rl = OrganRLOptimizer(
            state_path=Path(config.checkpoint_root) / "organ_rl_state.json"
        )
        self._last_cycle_strategy: Any = None
        self._last_saved_cycle: int | None = None

    def _configure_causal_runtime(self, fresh_start: bool = False) -> None:
        """Create the local causal runtime only for an explicitly enabled arm."""
        mode = str(getattr(self.cfg, "causal_mode", "disabled")).strip().lower()
        self.causal_bus = None
        self.causal_ledger = None
        if mode == "disabled":
            return

        arms = {
            "control": AblationArm.CONTROL,
            "shadow": AblationArm.SHADOW,
            "enforce": AblationArm.APPLY,
        }
        if mode not in arms:
            raise ValueError(
                "causal_mode must be one of: "
                "control, disabled, enforce, shadow"
            )

        workspace_root = (Path(self.root) / "workspace").resolve()
        if getattr(self.cfg, "checkpoint_root", None):
            ledger_path = (Path(self.root) / self.cfg.checkpoint_root / "causal_events.jsonl").resolve()
        else:
            ledger_relative = Path(
                getattr(
                    self.cfg,
                    "causal_ledger_jsonl",
                    "workspace/runtime/organism/causal/events.jsonl",
                )
            )
            ledger_path = (Path(self.root) / ledger_relative).resolve()
        if not ledger_path.is_relative_to(workspace_root):
            raise ValueError(
                "causal ledger must remain inside the local workspace"
            )
        if fresh_start and ledger_path.exists():
            try:
                ledger_path.unlink()
            except Exception:
                pass

        self.causal_ledger = CausalLedger(ledger_path)
        self.causal_bus = OrganCausalBus(
            arm=arms[mode],
            adapters=(
                ExplicitRequestAdapter(),
                GABAInterventionAdapter(),
                CuriosityPriorityAdapter(),
                SpiderCalibrationAdapter(),
            ),
            intent_recorder=self.causal_ledger,
        )

    def bootstrap(self, token_bin: str = None, resume: str = None):
        """Initialize all organs. Phase 0."""
        print("🧬 BOOTSTRAP: Inicializando organismo...")
        self._configure_causal_runtime(fresh_start=(resume is None))
        # R5: cooperative emergency-save on SIGINT/SIGTERM (main thread only).
        self._install_signal_handlers()

        # Tokenizer
        tok_path = self.root / self.cfg.tokenizer_dir
        if self.model_config.tokenizer == "smol_49152_transplant_v1":
            from f51_darwin.transplant_16b.tokenizer import (
                SmolTokenizerAdapter,
            )

            self.tokenizer = SmolTokenizerAdapter.load(tok_path)
        elif self.model_config.tokenizer == "f51_bpe":
            self.tokenizer = F51BPETokenizer.load(tok_path)
        else:
            raise ValueError(
                f"unsupported tokenizer contract: {self.model_config.tokenizer}"
            )
        self.tokenizer_id = tokenizer_identity(self.tokenizer)
        print(f"   Tokenizer: {self.tokenizer.vocab_size} tokens")

        # Model
        print(f"   Modelo: {self.model_config.model_name} ({self.model_config.d_model}d, {self.model_config.n_layers}L, {self.model_config.fine_experts}E)")
        self.model = DarwinXModel(self.model_config)

        self.amp_dtype, _ = amp_settings(self.device, "bf16")
        # Master weights stay fp32; torch.amp.autocast (training.py/lifecycle.py/
        # checkpoint_mixin.py, all keyed off self.amp_dtype) downcasts to bf16
        # only for the duration of forward/backward. Previously the model itself
        # was permanently cast to bf16 here, which also forced the AdamW moment
        # buffers (exp_avg/exp_avg_sq) into bf16 with no fp32 copy anywhere —
        # confirmed via the production checkpoint (cycle 46/step 14500) that
        # optimizer.state[p]['exp_avg'/'exp_avg_sq'] were bfloat16. Near a loss
        # plateau, gradient updates smaller than half a bf16 ULP (~3e-5 at the
        # model's typical weight magnitude) round away to nothing, so Adam could
        # silently stall regardless of any reward/organ signal.

        resume_payload = self._restore_resume(resume)
        self._place_model()
        self._activate_organs(resume_payload)
        self._configure_optimizer(resume_payload)
        self._load_token_data(token_bin)
        validate_training_data_contract(
            (resume_payload or {}).get("training_data_contract"),
            self.training_data_contract,
            is_resume=resume_payload is not None,
        )
        self._register_experts(resume_payload)
        self._finish_bootstrap(resume_payload)

    def _restore_resume(self, resume: str | None):
        resume_payload = None
        if resume:
            self.resume_checkpoint_path = Path(resume).resolve()
            # Validate checkpoint integrity before unpickling
            with open(self.resume_checkpoint_path, 'rb') as f:
                header = f.read(4)
            if header != b'PK\x03\x04':
                raise RuntimeError(f"Checkpoint {self.resume_checkpoint_path} is not a valid zip/torch archive")
            resume_payload = torch.load(
                self.resume_checkpoint_path,
                map_location="cpu",
                weights_only=False,  # required: checkpoint contains dicts, optimizer state, replay buffer
            )
            # Strip _orig_mod prefix if present (torch.compile artifact)
            state = {}
            for k, v in resume_payload["model_state_dict"].items():
                state[k.replace("_orig_mod.", "")] = v
            checkpoint_version = int(resume_payload.get("version", 0))
            declared_base_id = resume_payload.get("base_checkpoint_id")
            if checkpoint_version >= 7:
                if checkpoint_version > 9:
                    raise ValueError(
                        f"unsupported checkpoint version {checkpoint_version}"
                    )
                embedded_cfg = resume_payload.get("config")
                if not isinstance(embedded_cfg, dict):
                    raise ValueError("v7 checkpoint is missing its embedded model config")
                embedded_normalized = DarwinXConfig.from_mapping(embedded_cfg)
                # Structural-only comparison: loss weights, heartbeat
                # intervals, and calibration thresholds may change between
                # resume cycles without breaking the model contract.
                from f51_darwin.organism.checkpoint_root import (
                    _OPERATIONAL_CONFIG_FIELDS,
                )
                embedded_structural = {
                    k: v
                    for k, v in embedded_normalized.__dict__.items()
                    if k not in _OPERATIONAL_CONFIG_FIELDS
                }
                runtime_structural = {
                    k: v
                    for k, v in self.model_config.__dict__.items()
                    if k not in _OPERATIONAL_CONFIG_FIELDS
                }
                if embedded_structural != runtime_structural:
                    changed = sorted(
                        key
                        for key in set(embedded_structural) | set(runtime_structural)
                        if embedded_structural.get(key) != runtime_structural.get(key)
                    )
                    explicit_loss_v2_migration = (
                        checkpoint_version == 7
                        and self.checkpoint_version >= 8
                        and changed == ["loss_semantics_version"]
                        and embedded_normalized.loss_semantics_version == 1
                    )
                    if not explicit_loss_v2_migration:
                        raise ValueError(
                            "checkpoint config is incompatible with current "
                            "runtime config: " + ", ".join(changed)
                        )
                checkpoint_identity = validate_v7_checkpoint_identity(
                    resume_payload, state
                )
                if checkpoint_version in (8, 9):
                    if self.checkpoint_version not in (8, 9):
                        raise ValueError(
                            "causal checkpoint (v8/v9) cannot resume into the "
                            "legacy loss-v1/causal-disabled runtime"
                        )
                    # Truncate ledger to checkpoint head on crash recovery:
                    # events recorded after the last checkpoint save (lost
                    # steps) would otherwise cause a ledger_head mismatch.
                    _rewind_ledger_to_checkpoint(
                        self.causal_ledger, resume_payload
                    )
                    validate_v8_causal_contract(
                        resume_payload,
                        loss_semantics_version=(
                            self.model_config.loss_semantics_version
                        ),
                        causal_mode=self.cfg.causal_mode,
                        causal_v8_migration=self.cfg.causal_v8_migration,
                        ledger_head=current_causal_ledger_head(
                            self.causal_ledger
                        ),
                    )
                    # ── v9 blockchain integrity verification ──
                    if checkpoint_version >= 9:
                        self._verify_blockchain_integrity()
                        _ckpt_block_hash = str(
                            resume_payload.get("block_hash") or ""
                        )
                        if _ckpt_block_hash:
                            self._rewind_blockchain_to_checkpoint(
                                _ckpt_block_hash
                            )
                elif self.checkpoint_version >= 8:
                    # A v7 source is always legacy evidence. The explicit flag
                    # authorizes its next save as v8+; no causal fields are read.
                    self.resume_migration = {
                        "schema": "darwin-v7-to-v8-causal-migration-v1",
                        "from_version": 7,
                        "training_contract_id": self.training_contract_id,
                        "loss_semantics_version": (
                            self.model_config.loss_semantics_version
                        ),
                        "causal_mode": self.cfg.causal_mode,
                    }
            else:
                self.resume_migration = validate_legacy_v6_resume(
                    resume_payload,
                    state,
                    self.model,
                    self.model_config,
                    self.resume_checkpoint_path,
                )
                identity_note = (
                    "verificada"
                    if self.resume_migration["legacy_identity_recomputed"]
                    else "schema historico preservado; hash atual difere"
                )
                print(
                    "   🧬 Migração v6→v7 autorizada: "
                    f"config exata, {len(self.resume_migration['missing_initialized_buffers'])} "
                    f"buffers novos, identidade {identity_note}"
                )
            declared_tokenizer_id = resume_payload.get("tokenizer_id")
            if declared_tokenizer_id and declared_tokenizer_id != self.tokenizer_id:
                raise ValueError("checkpoint tokenizer identity does not match loaded tokenizer")
            # Any explicit checkpoint is authoritative. Legacy checkpoints
            # without an embedded adapter intentionally resume with a fresh one
            # instead of reading a potentially newer global state file.
            self.online_state_authoritative = True
            if int(resume_payload.get("version", 0)) >= 6:
                embedded = resume_payload.get("online_learning_state")
                self.resume_online_learning_state = embedded if isinstance(embedded, dict) else None
            if checkpoint_version >= 7:
                topology_manifest = resume_payload.get("topology_manifest")
                if not isinstance(topology_manifest, dict):
                    raise ValueError("v7 checkpoint is missing topology_manifest")
                repaired = DarwinXModel.restore_topology(topology_manifest, self.model)
                print(f"   Topology Manifest v7: {repaired} reparos antes dos tensores")
                initialized = migrate_mutational_state_for_load(state, self.model)
                if initialized:
                    if self.resume_migration is None:
                        self.resume_migration = {
                            "schema": "darwin-v7-mutational-state-v2",
                            "from_version": checkpoint_version,
                        }
                    self.resume_migration[
                        "initialized_mutational_buffers"
                    ] = initialized
                    print(
                        "   Gradient Mutational State v2: "
                        f"{len(initialized)} buffers inicializados com seguranca"
                    )
            incompatible = self.model.load_state_dict(
                state, strict=checkpoint_version >= 7
            )
            if checkpoint_version == 6:
                actual_missing = sorted(incompatible.missing_keys)
                expected_missing = self.resume_migration["missing_initialized_buffers"]
                if actual_missing != expected_missing or incompatible.unexpected_keys:
                    raise ValueError("v6 migration load result differs from validated anatomy")
            # Reanimate organ gates that collapsed to zero during cold storage.
            # Must run BEFORE backbone_identity so the content hash reflects the
            # actual runtime state and lineage_root.json stays consistent.
            _gaba_reinit = 0
            for _name, _param in self.model.named_parameters():
                if _name.endswith(".gaba.residual_gate") and _param.abs() < 1e-8:
                    _param.data.fill_(0.02)
                    _gaba_reinit += 1
            if _gaba_reinit:
                print(f"   🧬 GABA gates reinitialized: {_gaba_reinit}")
            _ttm_gate = getattr(self.model, "ttm_residual_gate", None)
            if _ttm_gate is not None and _ttm_gate.abs() < 1e-8:
                _ttm_gate.data.fill_(0.01)
                print("   ⏱️  TTM gate reinitialized")
            self.base_checkpoint_id = backbone_identity(
                self.model.state_dict(), self.model_config
            )
            if self.resume_migration is not None:
                self.resume_migration["migrated_runtime_base_checkpoint_id"] = self.base_checkpoint_id
            training_state = resume_payload.get("training_state", {})
            organism_state = resume_payload.get("organism", {})
            self.start_step = int(training_state.get("step", organism_state.get("total_steps", 0)))
            self.total_steps = self.start_step
            self.cycle = int(training_state.get("cycle", organism_state.get("cycle", 0)))
            self.train_tokens_seen = int(training_state.get("train_tokens_seen", 0))
            if self.train_tokens_seen < 0:
                raise ValueError("checkpoint train_tokens_seen must be non-negative")
            # R2: honor the saved warmup ramp when the launcher did not pass
            # --warmup-steps explicitly, so a cold resume still warms up.
            saved_warmup = int(training_state.get("warmup_steps", 0))
            if saved_warmup and not int(getattr(self.cfg, "warmup_steps", 0)):
                self.cfg.warmup_steps = saved_warmup
            print(
                f"   Resumed from cycle {self.cycle}, step {self.start_step} "
                f"(missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)})"
            )
            # GATE 0.1 FIX: carrega replay buffer do checkpoint
            saved_replay = resume_payload.get("replay_buffer", [])
            if saved_replay:
                if isinstance(saved_replay, dict):
                    self.replay.load_state_dict(saved_replay)
                else:
                    self.replay.load_records(saved_replay)
                print(f"   📦 Replay: {len(self.replay)} exemplos restaurados")
            saved_streak = organism_state.get("ashes_streak", {})
            self._ashes_streak = {
                str(expert_id): int(streak)
                for expert_id, streak in saved_streak.items()
            }
        else:
            self.base_checkpoint_id = backbone_identity(
                self.model.state_dict(), self.model_config
            )

        return resume_payload

    def _place_model(self):
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e9

        # 🖥️ DUAL GPU: Pipeline parallelism se 2+ GPUs disponíveis
        gpu_count = torch.cuda.device_count() if self.device.type == "cuda" else 0
        if gpu_count >= 2:
            # Move directly from CPU into the final placement. Moving the full
            # model to cuda:0 first OOMs the 2.5B class before it can be split.
            split_layer = int(
                os.environ.get("DUAL_GPU_SPLIT", "")
                or self.model.recommended_dual_gpu_split(gpu0=0, gpu1=1)
            )
            ok = self.model.enable_dual_gpu(
                gpu0=0,
                gpu1=1,
                split_layer=split_layer,
            )
            if ok:
                gpu0_name = torch.cuda.get_device_name(0)
                gpu1_name = torch.cuda.get_device_name(1)
                print(f"   🖥️  DUAL GPU: {gpu0_name[:20]}... + {gpu1_name[:20]}...")
                print(
                    f"        Pipeline: layers 0-{split_layer - 1} → GPU0, "
                    f"layers {split_layer}-{self.model_config.n_layers - 1} → GPU1"
                )
                print(f"   Params: {total_params:.2f}B across cuda:0 + cuda:1")
            else:
                print(f"   ⚠️  DUAL GPU: {gpu_count} GPUs detectadas mas split falhou")
                self.model = self.model.to(self.device)
                print(f"   Params: {total_params:.2f}B on {self.device}")
        else:
            self.model = self.model.to(self.device)
            print(f"   Params: {total_params:.2f}B on {self.device}")


    def _activate_organs(self, resume_payload: dict[str, Any] | None):
        # 🔥 ATIVAR TODOS OS ÓRGÃOS
        if self.cfg.heartbeat_enabled or self.cfg.ghost_enabled or self.cfg.curiosity_enabled:
            from f51_darwin.curiosity import CuriosityDrive
            curiosity = CuriosityDrive() if self.cfg.curiosity_enabled else None
            # Ghost is handled internally by the model's _ghost_loss / _causal_ghost_loss
            # The standalone GhostBrain requires GhostCorpus which is not available in training.
            ghost = None
            self.model.activate_organism(
                curiosity=curiosity,
                ghost_brain=ghost,
                jepa=self.model.jepa_predictor,
                organism_ref=self,
            )
            organs = []
            if self.model.heartbeat: organs.append("coração")
            if curiosity: organs.append("curiosidade")
            if self.cfg.ghost_enabled: organs.append("ghost (modelo interno)")
            print(f"   🧬 Órgãos ativos: {', '.join(organs)}")
            if resume_payload:
                self.model.load_heartbeat_state_dict(resume_payload.get("heartbeat_state"))
            # A checkpoint saved on a different device topology (e.g. a
            # dual-GPU pipeline-split box) can carry heartbeat/TTM tensors
            # tagged to a device that doesn't exist here. And even on a
            # fresh-start (no resume_payload at all), TestTimeMemory.proj_value
            # has been observed off-device despite activate_organism()'s own
            # .to() call (proj_key never surfaces the same bug because
            # retrieve() only exercises it once memory has entries -- on a
            # fresh model with an empty memory it silently never runs, while
            # proj_value's self-reconstruction term in the forward pass runs
            # unconditionally every step). Re-home unconditionally, not just
            # on resume, so a stray CPU/cuda:1 tensor never survives either
            # path onto a single-GPU host.
            if self.model.heartbeat is not None:
                placement_device = next(self.model.parameters()).device
                self.model.heartbeat.to(device=placement_device)


    # JEPA predictor LR multiplier. Was bumped to 10x at some point without
    # updating the "3x" comment/log strings below it — 3x is the value this
    # code has actually been documented (and presumably tuned) for, so we
    # revert to it rather than keep an undocumented 10x. On resume,
    # _configure_optimizer re-pins the JEPA group's lr to this multiplier
    # explicitly, since load_state_dict() would otherwise restore whatever
    # multiplier the checkpoint happened to be saved with.
    JEPA_LR_MULTIPLIER = 3.0

    def _jepa_param_groups(self):
        """Parameter groups: JEPA predictor at JEPA_LR_MULTIPLIER x LR to
        prevent collapse, and 1D params (norms, biases, a_log, d_skip) split
        into a weight_decay=0 group per standard practice (GPT-2/nanoGPT/
        LLaMA-style) -- decaying scale/bias parameters isn't the regularizer
        it is for dense matrices, and for dt_proj.bias specifically it drags
        softplus(bias) out of the calibrated [dt_min, dt_max] init range
        over time (found 2026-07-26, see ssm_core.py init docstring).

        Group order is NOT stable across code versions -- resume matches
        groups by (name, shape) remap on load_state_dict failure (see
        _configure_optimizer), not by positional index, so adding groups
        here is safe for existing checkpoints (momentum for params that
        moved groups is cold-reinitialized, nothing crashes).
        """
        jepa_ids = set()
        jepa_predictor = getattr(self.model, "jepa_predictor", None)
        if jepa_predictor is not None:
            jepa_ids = {id(p) for p in jepa_predictor.parameters()}
        backbone_decay, backbone_no_decay = [], []
        jepa_decay, jepa_no_decay = [], []
        for name, p in self.model.named_parameters():
            is_jepa = id(p) in jepa_ids
            no_decay = p.ndim <= 1  # biases, RMSNorm.weight, a_log, d_skip
            if is_jepa:
                (jepa_no_decay if no_decay else jepa_decay).append(p)
            else:
                (backbone_no_decay if no_decay else backbone_decay).append(p)
        groups = [{"params": backbone_decay}]
        if backbone_no_decay:
            groups.append({"params": backbone_no_decay, "weight_decay": 0.0})
        if jepa_decay:
            groups.append(
                {"params": jepa_decay, "lr": self.cfg.learning_rate * self.JEPA_LR_MULTIPLIER}
            )
        if jepa_no_decay:
            groups.append(
                {
                    "params": jepa_no_decay,
                    "lr": self.cfg.learning_rate * self.JEPA_LR_MULTIPLIER,
                    "weight_decay": 0.0,
                }
            )
        return groups

    def _configure_optimizer(self, resume_payload: dict[str, Any] | None):
        # Optimizer identity is explicit. AdamW remains the v7 compatibility
        # path; DAE hybrid is the active fresh-lineage candidate.
        saved_opt_type = (resume_payload or {}).get("optimizer_type", "")
        saved_opt_state = (resume_payload or {}).get("optimizer_state_dict", None)

        if self.cfg.optimizer_name == "dae_hybrid":
            self.optimizer = build_optimizer(
                "dae_hybrid",
                self.model.named_parameters(),
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
            )
            if saved_opt_state is not None:
                if "daehybridoptimizer" not in saved_opt_type.lower():
                    raise ValueError(
                        "optimizer identity mismatch: requested dae_hybrid, "
                        f"checkpoint contains {saved_opt_type or 'unknown'}"
                    )
                try:
                    self.optimizer.load_state_dict(saved_opt_state)
                    print("   ⚡ Optimizer: DAE Hybrid v1 (retomado)")
                except (KeyError, TypeError, ValueError) as exc:
                    self.optimizer = build_optimizer(
                        "dae_hybrid",
                        self.model.named_parameters(),
                        lr=self.cfg.learning_rate,
                        weight_decay=self.cfg.weight_decay,
                    )
                    print(
                        "   ⚡ Optimizer: DAE Hybrid fresh; "
                        f"state rejeitado ({exc})"
                    )
            else:
                print("   ⚡ Optimizer: DAE Hybrid v1 (do zero)")
        elif saved_opt_state is not None and "adamw" in saved_opt_type.lower():
            # Resume with JEPA + no-decay parameter groups (_jepa_param_groups,
            # up to 4 groups: backbone/backbone_no_decay/jepa/jepa_no_decay).
            # Group count can differ from what an older checkpoint was saved
            # with -- load_state_dict below falls back to (name,shape) remap
            # on mismatch instead of crashing (see except clause).
            param_groups = self._jepa_param_groups()
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay, fused=True,
            )
            try:
                self.optimizer.load_state_dict(saved_opt_state)
                # Rescale LR when launcher overrides the checkpoint's base LR.
                # initial_lr anchors the warmup/decay schedule — updating it
                # here lets --lr changes survive resume without cold-start.
                _saved_lr = float(
                    (saved_opt_state.get("param_groups") or [{}])[0].get("lr", 0)
                )
                _cfg_lr = float(self.cfg.learning_rate)
                if _saved_lr > 0 and abs(_cfg_lr - _saved_lr) / (_saved_lr + 1e-8) > 0.01:
                    _ratio = _cfg_lr / _saved_lr
                    for _group in self.optimizer.param_groups:
                        _group["lr"] = _group["lr"] * _ratio
                        if "initial_lr" in _group:
                            _group["initial_lr"] = _group["initial_lr"] * _ratio
                    _rescale_msg = f"LR rescalado {_saved_lr:.1e}→{_cfg_lr:.1e}, "
                else:
                    _rescale_msg = ""
                # load_state_dict restores whatever multiplier the checkpoint
                # was saved with (possibly the old 10x). The ratio rescale
                # above preserves that multiplier rather than correcting it,
                # so JEPA's group lr is pinned explicitly to the current
                # JEPA_LR_MULTIPLIER relative to the (possibly just-rescaled)
                # backbone lr on every resume.
                if len(self.optimizer.param_groups) > 1:
                    _backbone_lr = float(self.optimizer.param_groups[0]["lr"])
                    _jepa_lr = _backbone_lr * self.JEPA_LR_MULTIPLIER
                    # Identify JEPA groups by param identity, not position --
                    # group count/order now varies (no_decay split), so a
                    # hardcoded index 1 would silently pin the wrong group.
                    _jepa_ids_now = set()
                    _jepa_predictor_now = getattr(self.model, "jepa_predictor", None)
                    if _jepa_predictor_now is not None:
                        _jepa_ids_now = {id(p) for p in _jepa_predictor_now.parameters()}
                    for _group in self.optimizer.param_groups:
                        _group_params = _group.get("params", [])
                        _is_jepa_group = bool(_group_params) and id(_group_params[0]) in _jepa_ids_now
                        if _is_jepa_group:
                            for _key in ("lr", "initial_lr"):
                                if _key in _group:
                                    _group[_key] = _jepa_lr
                print(
                    f"   ⚡ Optimizer: AdamW (retomado, {_rescale_msg}"
                    f"JEPA ×{self.JEPA_LR_MULTIPLIER:g} LR)"
                )
            except (KeyError, TypeError, ValueError) as exc:
                if int((resume_payload or {}).get("version", 0)) == 6:
                    self.optimizer = torch.optim.AdamW(
                        self.model.parameters(), lr=self.cfg.learning_rate,
                        weight_decay=self.cfg.weight_decay, fused=True,
                    )
                    try:
                        inserted = migrate_adamw_state_with_new_baselines(
                            saved_opt_state, self.model, self.optimizer
                        )
                        print(
                            "   ⚡ Optimizer: AdamW v6 migrado "
                            f"({len(saved_opt_state['state'])} momenta restaurados, "
                            f"{len(inserted)} baselines novos)"
                        )
                    except (KeyError, TypeError, ValueError) as migration_exc:
                        self.optimizer = torch.optim.AdamW(
                            self.model.parameters(), lr=self.cfg.learning_rate,
                            weight_decay=self.cfg.weight_decay, fused=True,
                        )
                        print(
                            "   ⚡ Optimizer: AdamW fresh; momentum v6 rejeitado "
                            f"({migration_exc})"
                        )
                else:
                    # v7+: try to remap by (name, shape) before discarding all momentum.
                    try:
                        from f51_darwin.organism.checkpoint import (
                            remap_optimizer_state_from_checkpoint,
                        )
                        old_model_state = resume_payload.get("model_state_dict", {})
                        if old_model_state:
                            stats = remap_optimizer_state_from_checkpoint(
                                saved_opt_state,
                                old_model_state,
                                self.optimizer,
                                dict(self.model.named_parameters()),
                            )
                            print(
                                "   ⚡ Optimizer: AdamW remapeado por (name, shape) — "
                                f"{stats['preserved']} momentos preservados, "
                                f"{stats['initialised']} cold init, "
                                f"{stats['dropped']} descartados"
                            )
                        else:
                            raise ValueError("checkpoint has no model_state_dict")
                    except Exception as remap_exc:
                        param_groups = self._jepa_param_groups()
                        self.optimizer = torch.optim.AdamW(
                            param_groups,
                            lr=self.cfg.learning_rate,
                            weight_decay=self.cfg.weight_decay, fused=True,
                        )
                        print(
                            "   ⚡ Optimizer: AdamW fresh; "
                            f"remapeamento falhou ({remap_exc})"
                        )
        elif saved_opt_state is not None and "adafactor" in saved_opt_type.lower():
            # Legacy Adafactor checkpoint — restart with AdamW fresh
            param_groups = self._jepa_param_groups()
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay, fused=True,
            )
            print(f"   ⚡ Optimizer: AdamW (fresh, legado Adafactor descartado, JEPA ×{self.JEPA_LR_MULTIPLIER:g} LR)")
        else:
            # Fresh start — AdamW with JEPA predictor at JEPA_LR_MULTIPLIER x LR
            param_groups = self._jepa_param_groups()
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay, fused=True,
            )
            n_jepa = sum(1 for g in param_groups[1:])
            print(f"   ⚡ Optimizer: AdamW (do zero, JEPA ×{self.JEPA_LR_MULTIPLIER:g} LR, {n_jepa} grupo preditor)")
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=False)
        # Offload AdamW moments to CPU RAM after creation — frees ~1.2 GB
        # across both GPUs.  move_optimizer_state_for_parameters restores
        # them transparently before each step().
        from f51_darwin.organism.checkpoint import offload_optimizer_states_to_cpu
        offload_optimizer_states_to_cpu(self.optimizer)


    def _load_token_data(self, token_bin: str | None):
        # Token data
        self.token_ids, self.token_path = load_token_ids(self.root, token_bin=token_bin)
        self.token_count = len(self.token_ids)
        print(f"   Tokens: {self.token_count/1e9:.1f}B ({self.token_path})")

        # ── Canary holdout split (Task 3) ──
        self.train_token_ids = self.token_ids
        self.holdout_token_ids = None
        self.holdout_starts: tuple[tuple[int, ...], ...] = ()
        self.holdout_definition: dict[str, object] | None = None
        self.token_source_identity: dict[str, object] = {}
        self.canary_metadata: dict[str, object] = {}
        if self.cfg.holdout_tokens > 0:
            split = split_tail_holdout(
                self.token_ids,
                holdout_tokens=self.cfg.holdout_tokens,
                block_size=self.cfg.block_size,
            )
            self.train_token_ids = split.train_tokens
            self.holdout_token_ids = split.holdout_tokens
            self.holdout_starts = fixed_batch_starts(
                token_count=split.holdout_token_count,
                block_size=self.cfg.block_size,
                batch_size=self.cfg.batch_size,
                batches=self.cfg.holdout_batches,
                seed=self.cfg.holdout_seed,
            )
            self.holdout_definition = {
                "source_token_count": split.source_token_count,
                "train_stop": split.train_stop,
                "holdout_start": split.holdout_start,
                "holdout_token_count": split.holdout_token_count,
                "seed": self.cfg.holdout_seed,
                "batches": self.cfg.holdout_batches,
            }
            self.token_source_identity = canary_token_source_identity(
                self.token_path, self.token_count
            )
            if not self.cfg.canary_metadata_json:
                raise RuntimeError("canary runtime metadata JSON is required when holdout_tokens > 0")
            self.canary_metadata = json.loads(
                Path(self.cfg.canary_metadata_json).read_text(encoding="utf-8-sig")
            )
        if not self.token_source_identity:
            self.token_source_identity = token_source_state(
                self.token_path, self.token_count
            )
        self.training_data_contract = build_training_data_contract(
            sampler_mode=self.cfg.sampler_mode,
            sampler_version=self.cfg.sampler_version,
            base_seed=self.cfg.seed,
            block_size=self.cfg.block_size,
            batch_size=self.cfg.batch_size,
            source_identity=self.token_source_identity,
            train_token_count=len(self.train_token_ids),
            holdout_definition=self.holdout_definition,
        )


    def _register_experts(self, resume_payload: dict[str, Any] | None):
        # Module registry (one per MoE expert) — referencia aos experts REAIS
        self.legacy.set_causal_refs(self.expert_pool, self.model)
        saved_experts = {
            entry.get("id"): entry
            for entry in ((resume_payload or {}).get("organism", {}).get("expert_pool", []))
            if entry.get("id")
        }
        for layer in range(self.model_config.n_layers):
            for e in range(self.model_config.fine_experts):
                mid = f"L{layer}_E{e}"
                # GATE 0.2 FIX: nao criar ExpertModule sombra (2.13 GB desperdicados)
                # Apenas registra metadados apontando pro expert real
                real_expert = self.model.blocks[layer].moe.fine_experts[e]
                saved = saved_experts.get(mid, {})
                try:
                    module_state = ModuleState(saved.get("state", ModuleState.ACTIVE.value))
                except ValueError:
                    module_state = ModuleState.ACTIVE
                self.expert_pool.add_expert(
                    mid, None,
                    created_at_cycle=int(saved.get("created_at_cycle", 0)),
                    state=module_state,
                    score=float(saved.get("score", 0.0)),
                )
                # Bind ao expert REAL do DarwinXModel MoE
                self.expert_pool.bind_real_expert(mid, layer, e, real_expert)
                self.expert_pool.records[mid].usage_count = int(saved.get("usage_count", 0))
                # Registra nascimento no LineageTracker
                if not self.lineage.has_module(mid):
                    self.lineage.record_birth(
                        mid, cycle=0, parent_ids=[],
                        d_model=self.model_config.d_model,
                    )
                # Coloca na camada GPU (quente) por padrão
                persisted_tier = self.legacy.expert_tier(mid)
                self.legacy.update_expert_tier(
                    mid, persisted_tier, persisted_tier or LayerTier.GPU,
                )

        # GATE 0.3: Conecta Legacy Layers ao ExpertPool e Modelo para acoes causais
        self.legacy.set_causal_refs(self.expert_pool, self.model)

        # Selar o DNA (semente imutável — só define uma vez)
        self.legacy.set_dna(
            architecture={
                'model_name': self.model_config.model_name,
                'd_model': self.model_config.d_model,
                'n_layers': self.model_config.n_layers,
                'fine_experts': self.model_config.fine_experts,
            },
            tokenizer_config={'vocab_size': self.tokenizer.vocab_size},
        )

    def _verify_blockchain_integrity(self) -> None:
        """Verify the blockchain ledger integrity before resume.

        1. Opens the causal ledger and verifies the complete chain.
        2. If the chain is cryptographically broken, the old ledger is
           archived and a fresh chain is started so resume can proceed.

        This is the blockchain counterpart to the causal contract validation
        in ``validate_v8_causal_contract`` — it guards against on-disk
        corruption and external tampering at the JSONL level.
        """
        _ledger = getattr(self, "causal_ledger", None)
        if _ledger is None:
            return

        verification = _ledger.verify_chain()
        if verification.valid:
            return

        print(
            "   ⚠️  Blockchain integrity check failed: "
            + "; ".join(verification.errors)
        )
        _archive_stale_ledger(_ledger)
        print(
            "   ⚠️  Ledger arquivado e novo iniciado "
            "(blockchain integrity failure)"
        )

    def _rewind_blockchain_to_checkpoint(
        self, checkpoint_block_hash: str
    ) -> None:
        """Truncate the blockchain to the checkpoint's block hash.

        On crash recovery the blockchain may contain blocks from steps that
        were executed after the last checkpoint save.  Those blocks are
        orphaned — the model state rewound to the checkpoint — so the
        blockchain must be truncated to match.

        When *checkpoint_block_hash* is not found in the current chain
        (different training session), the old ledger is archived and a
        fresh chain begins — the alternative is a hard crash that blocks
        all resume.
        """
        _ledger = getattr(self, "causal_ledger", None)
        if _ledger is None:
            return

        current_head = _ledger.head_hash
        if current_head == checkpoint_block_hash:
            return

        kept = _ledger.truncate_to_head(checkpoint_block_hash)
        if kept == 0:
            # Checkpoint belongs to a different session — the blockchain
            # chain is irrecoverable.  Archive the old file and start a
            # fresh chain so resume can proceed.
            _archive_stale_ledger(_ledger)
            print(
                "   ⚠️  Block hash do checkpoint não encontrado na chain "
                "atual — blockchain arquivado e novo iniciado "
                "(sessão diferente)"
            )
            return

        # Reload internal state from the truncated file.
        _ledger.recover_head()

    def _finish_bootstrap(self, resume_payload: dict[str, Any] | None):
        # Data firewall
        paths = DataFactoryPaths.from_project(self.root)
        self.data_factory = DataFactory(paths, firewall=self.firewall)

        # Brainstem baseline
        self.brainstem.set_baseline_loss(15.0)

        # Restore stochastic streams only after bootstrap consumed RNG.
        if resume_payload:
            restore_rng_state(resume_payload.get("rng_state"))

        # Soul blessing
        n_experts = len(self.expert_pool.records)
        print(f"   🧬 ExpertPool: {n_experts} experts registrados")
        print(f"   🧬 LegacyLayers: {self.legacy.status()['total_experts']} na camada GPU")
        print(f"   🧬 LineageTracker: {self.lineage._total_births} nascimentos")
        print(f"\n   {self.soul.blessing()}")
        print("   🧬 ORGANISMO VIVO. Pronto para evoluir.\n")

        # ── Blockchain & Senate ──
        # Instantiated independently of causal_bus so blocks are recorded
        # even when causal_mode is "disabled".
        if getattr(self.cfg, 'blockchain_enabled', False):
            try:
                from f51_darwin.organism.blockchain import OrganLedger
                from f51_darwin.organism.organ_identity import ORGAN_DEFINITIONS
                from f51_darwin.organism.organ_reputation import OrganReputationTracker
                from f51_darwin.organism.organ_senate import OrganSenate
            except ImportError as exc:
                print(
                    f"   ⚠️  Blockchain: import falhou — {exc}. "
                    f"Blockchain desabilitada para este run.",
                    flush=True,
                )
            else:
                # Imports succeeded — instantiate the blockchain runtime.
                blockchain_path = Path(getattr(
                    self.cfg, 'blockchain_path',
                    'workspace/runtime/organism/blockchain/blocks.jsonl',
                ))
                if not blockchain_path.is_absolute():
                    blockchain_path = (Path(self.root) / blockchain_path).resolve()
                self.organ_ledger = OrganLedger(blockchain_path)

                self.organ_reputation_tracker = OrganReputationTracker(
                    organ_definitions=ORGAN_DEFINITIONS,
                )

                if getattr(self.cfg, 'senate_enabled', False):
                    self.organ_senate = OrganSenate(
                        min_weight=float(getattr(self.cfg, 'senate_min_weight', 0.02)),
                        death_threshold=float(getattr(self.cfg, 'senate_death_threshold', 0.05)),
                        rising_bonus=float(getattr(self.cfg, 'senate_rising_bonus', 0.10)),
                        newborn_period=int(getattr(self.cfg, 'senate_newborn_period', 3)),
                    )

                organs = []
                if getattr(self, 'organ_ledger', None) is not None:
                    organs.append('ledger')
                if getattr(self, 'organ_senate', None) is not None:
                    organs.append('senate')
                if getattr(self, 'organ_reputation_tracker', None) is not None:
                    organs.append('reputation')
                if organs:
                    print(
                        f"   🔗 Blockchain: {', '.join(organs)} ativos → "
                        f"{self.organ_ledger.path}",
                        flush=True,
                    )

        # Save initial state
        self._save_state()
