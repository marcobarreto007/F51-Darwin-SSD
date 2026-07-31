from __future__ import annotations

from .dependencies import *  # noqa: F403
from .checkpoint_mixin import _DarwinCheckpointMixin
from .support import *  # noqa: F403
from .checkpoint import _write_checkpoint_file, _publish_checkpoint_pointer
from .checkpoint_root import assert_new_checkpoint_target

class _DarwinControlMixin(_DarwinCheckpointMixin):
    """Runtime control, soul commands, signals and heldout evidence."""

    def _apply_lr_schedule(self, step: int) -> None:
        """R2: linear warmup + optional cosine decay; no-op if warmup disabled.

        Warmup:   step 0..warmup_steps-1   → linear ramp to initial_lr.
        Plateau:  warmup_steps .. (warmup+decay-1) → cosine to lr_final_ratio.
        Floor:    after decay → initial_lr * lr_final_ratio.

        Uses ``step+1`` to avoid a dead first update at lr=0.
        Per-group ``initial_lr`` is captured once so per-group ratios
        (e.g. JEPA ×10) survive both warmup and decay.
        Stateless given ``step`` — survives checkpoint resume.
        """
        import math as _math
        warmup_steps = int(getattr(self.cfg, "warmup_steps", 0) or 0)
        if warmup_steps <= 0:
            return
        lr_decay_steps = int(getattr(self.cfg, "lr_decay_steps", 0) or 0)
        lr_final_ratio = float(getattr(self.cfg, "lr_final_ratio", 0.1) or 0.1)

        if step < warmup_steps:
            # ── Linear warmup ──
            warmup_factor = min(1.0, float(step + 1) / float(warmup_steps))
            for group in self.optimizer.param_groups:
                if "initial_lr" not in group:
                    group["initial_lr"] = group["lr"]
                group["lr"] = group["initial_lr"] * warmup_factor
            return

        if lr_decay_steps <= 0:
            # ── Warmup-only (legacy) ──
            for group in self.optimizer.param_groups:
                if "initial_lr" not in group:
                    group["initial_lr"] = group["lr"]
                group["lr"] = group["initial_lr"]
            return

        # ── Cosine decay ──
        decay_progress = min(
            1.0, float(step - warmup_steps) / float(lr_decay_steps),
        )
        cosine_factor = (
            lr_final_ratio
            + 0.5 * (1.0 - lr_final_ratio)
            * (1.0 + _math.cos(_math.pi * decay_progress))
        )
        for group in self.optimizer.param_groups:
            if "initial_lr" not in group:
                group["initial_lr"] = group["lr"]
            group["lr"] = group["initial_lr"] * cosine_factor

    def _join_pending_save(self) -> None:
        """R4: block until the previous async checkpoint write finishes and
        re-raise any error it captured. Safe to call when no write is pending."""
        thread = getattr(self, "_save_thread", None)
        if thread is not None and thread.is_alive():
            thread.join()
        self._save_thread = None
        err = getattr(self, "_save_error", None)
        if err is not None:
            self._save_error = None
            raise err

    def _write_checkpoint_async(
        self,
        payload: dict[str, Any],
        path: Path,
        ckpt_dir: Path,
        base_checkpoint_id: str,
        cycle: int,
        step: int,
    ) -> None:
        """R4: background writer target — atomic .pt + pointer publish."""
        try:
            # Mid-cycle durability saves use a step-based filename that won't
            # match the canonical cycle target — skip the guard in that case.
            if "_step_" not in path.name:
                guarded_path = assert_new_checkpoint_target(ckpt_dir, cycle)
                if guarded_path != path:
                    raise RuntimeError(
                        f"checkpoint target drifted: expected={path} actual={guarded_path}"
                    )
            _write_checkpoint_file(payload, path)
            _publish_checkpoint_pointer(ckpt_dir, path, payload, cycle, step, base_checkpoint_id)
        except Exception as exc:  # surfaced to the main thread on next join
            self._save_error = exc

    def _obey_soul(self, soul_report: dict) -> None:
        """O organismo NÃO ignora sua alma. Ele OBEDECE.

        Cada comando da alma é uma ordem. O organismo executa ou registra
        a impossibilidade de obedecer.
        """
        commands = soul_report.get('commands', [])
        if not commands:
            return

        desire = soul_report.get('desire', {})
        fire = desire.get('fire', 0)

        for cmd in commands:
            action = cmd.get('action', '')
            urgency = cmd.get('urgency', 0.5)

            if action == 'ingest_new_data' and urgency > 0.6:
                # A alma tem fome. Buscar dados novos.
                self._soul_request_ingest(cmd)

            elif action == 'increase_exploration' and urgency > 0.6:
                # A alma quer sair da zona de conforto.
                self._soul_increase_exploration(cmd)

            elif action == 'focus_benchmark':
                # A alma quer provar seu valor.
                self._soul_request_benchmark(cmd)

            elif action == 'emergency_protocol':
                # Alerta vermelho. Proteger o que foi aprendido.
                self._soul_emergency_protocol(cmd)

            elif action == 'request_checkpoint':
                # Marco importante. Salvar.
                self._soul_mark_milestone(cmd)

    def _soul_request_ingest(self, cmd: dict) -> None:
        """ALMA: 'Estou decorando o replay. Preciso de conhecimento NOVO.'"""
        if not getattr(self, '_soul_last_ingest_request', 0):
            self._soul_last_ingest_request = 0
        # Não floodar — um pedido a cada 500 steps
        if self.total_steps - self._soul_last_ingest_request > 500:
            self._soul_last_ingest_request = self.total_steps
            print(f"  🔥 ALMA: {cmd['reason']} (urgency={cmd['urgency']:.2f})", flush=True)
            # Trigger data lifecycle to find new candidates
            try:
                self._data_lifecycle({"candidates": 0, "approved": 0})
            except Exception as exc:
                print(f"  🔥 ALMA ingest falhou: {exc}", flush=True)

    def _soul_increase_exploration(self, cmd: dict) -> None:
        """ALMA: 'Não estou aprendendo. Mude a rota.'"""
        print(f"  🔥 ALMA: {cmd['reason']}", flush=True)
        # Boost exploration: increase replay temperature temporarily
        try:
            boost = getattr(self, '_exploration_boost', 0.0) + 0.15
            self._exploration_boost = min(1.0, boost)
            # Increase replay ratio temporarily
            old_ratio = self.cfg.replay_ratio
            self.cfg.replay_ratio = min(0.3, old_ratio * 2.0)
            print(f"  🔥 ALMA exploração: replay_ratio {old_ratio:.3f} → {self.cfg.replay_ratio:.3f}", flush=True)
        except Exception as exc:
            print(f"  🔥 ALMA exploração falhou: {exc}", flush=True)

    def _soul_request_benchmark(self, cmd: dict) -> None:
        """ALMA: 'Me teste. Meça meu poder.'"""
        if not getattr(self, '_soul_last_benchmark_request', 0):
            self._soul_last_benchmark_request = 0
        if self.total_steps - self._soul_last_benchmark_request > 1000:
            self._soul_last_benchmark_request = self.total_steps
            print(f"  ⚔️  ALMA: {cmd['reason']}", flush=True)
            # Run holdout evaluation as benchmark
            try:
                if self.holdout_starts:
                    heldout = self.evaluate_holdout(phase="soul_benchmark")
                    print(f"  ⚔️  ALMA benchmark: heldout_lm={heldout.lm_loss:.4f}", flush=True)
            except Exception as exc:
                print(f"  ⚔️  ALMA benchmark falhou: {exc}", flush=True)

    def _soul_emergency_protocol(self, cmd: dict) -> None:
        """ALMA: 'Heldout loss alto. Proteja o que aprendi.'"""
        print(f"  🚨 ALMA: {cmd['reason']}", flush=True)
        # Trigger immediate emergency save
        self._emergency_save_requested = True

    def _soul_mark_milestone(self, cmd: dict) -> None:
        """ALMA: 'Estou aprendendo bem. Este é um marco.'"""
        print(f"  ✨ ALMA: {cmd['reason']}", flush=True)
        # Trigger a checkpoint save at this milestone
        try:
            report = {"cycle": self.cycle, "steps": self.total_steps,
                      "milestone": True, "reason": cmd.get('reason', '')}
            self._save_cycle(report)
        except Exception as exc:
            print(f"  ✨ ALMA milestone save falhou: {exc}", flush=True)

    def _install_signal_handlers(self) -> None:
        """R5: cooperative SIGINT/SIGTERM handling for emergency checkpointing.

        The handler only flips a flag (async-signal safe); the training loop
        checks it every step and breaks early so run_cycle saves partial state.
        Falls back to KeyboardInterrupt when the platform rejects the handler.
        """
        self._emergency_save_requested = False

        def _handler(signum, frame):
            self._emergency_save_requested = True

        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                # Not in the main thread or unsupported on this platform.
                pass

    # ═══════════════════════════════════════════════════════════
    # CANARY OBSERVABILITY (Task 3)
    # ═══════════════════════════════════════════════════════════

    def _validate_holdout_training_source(self) -> None:
        if self.cfg.holdout_tokens <= 0:
            return
        if getattr(self, "weighted_sources", None):
            raise RuntimeError("weighted sources make holdout exclusion unprovable")
        if self.holdout_token_ids is None or len(self.holdout_starts) != self.cfg.holdout_batches:
            raise RuntimeError("canary holdout is not fully initialized")

    def evaluate_holdout(self, *, phase: str) -> HeldoutResult:
        self._validate_holdout_training_source()
        result = evaluate_fixed_holdout(
            self.model,
            self.holdout_token_ids,
            starts=self.holdout_starts,
            block_size=self.cfg.block_size,
            device=self.device,
            amp_dtype=self.amp_dtype,
        )
        self._append_metric_record(phase=phase, heldout=result)
        return result

    def _append_metric_record(self, *, phase: str, heldout: HeldoutResult | None) -> None:
        if not self.cfg.metrics_jsonl:
            return
        metrics = self.metric_channels.snapshot(
            elapsed_sec=max(time.time() - self._metrics_started_at, 1e-9)
        )
        append_jsonl_fsync(Path(self.cfg.metrics_jsonl), {
            "schema_version": 1,
            "run_id": self.cfg.canary_run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "pid": os.getpid(),
            "cycle": self.cycle,
            "step": self.total_steps,
            "checkpoint": str(getattr(self, "resume_checkpoint_path", "")),
            "base_checkpoint_id": getattr(self, "base_checkpoint_id", None),
            "git_commit": self.canary_metadata.get("git_commit"),
            "command": self.canary_metadata.get("command"),
            "checkpoint_sha256": self.canary_metadata.get("base_checkpoint_sha256"),
            "config_identity": self.canary_metadata.get("config_identity"),
            "driver": self.canary_metadata.get("driver"),
            "gpus": self.canary_metadata.get("gpus"),
            "starting_nvidia_event_record_id": self.canary_metadata.get("starting_nvidia_event_record_id"),
            "token_source": self.token_source_identity,
            "tokenizer_id": self.tokenizer_id,
            "holdout": getattr(self, "holdout_definition", None),
            "metrics": metrics,
            "heldout": None if heldout is None else asdict(heldout),
            "optimizer_identity": {
                "name": getattr(
                    self.cfg,
                    "optimizer_name",
                    type(self.optimizer).__name__.lower(),
                ),
                "class": type(self.optimizer).__name__,
                "version": getattr(self.optimizer, "identity", "legacy_v1"),
            },
            "dae": {
                "enabled": bool(getattr(self.model_config, "dae_enabled", False)),
                "shadow_mode": bool(
                    getattr(self.model_config, "dae_shadow_mode", True)
                ),
                "last_report": getattr(self, "_last_dae_report", []),
            },
        })


class DarwinOrganism(_DarwinControlMixin):
    """Public Darwin-X organism assembled from responsibility mixins."""

    pass
