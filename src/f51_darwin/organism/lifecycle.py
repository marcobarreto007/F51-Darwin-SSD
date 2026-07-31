from __future__ import annotations

import copy
import inspect

from .dependencies import *  # noqa: F403
from .bootstrap import _DarwinBootstrapMixin
from .support import *  # noqa: F403
from f51_darwin.organism.checkpoint import remap_optimizer_state

class _DarwinLifecycleMixin(_DarwinBootstrapMixin):
    """Lifecycle, data and temperature responsibilities."""

    def run_cycle(self, steps: int = None) -> dict:
        """Execute one full organism cycle."""
        self.cycle += 1
        steps = steps or self.cfg.max_steps_per_cycle

        print(f"\n{'='*60}")
        print(f"  🔄 CYCLE {self.cycle} — {steps} steps")
        print(f"  {'='*60}")

        report = {"cycle": self.cycle, "steps": 0, "loss_start": 0, "loss_end": 0,
                  "modules_active": 0, "modules_died": 0, "modules_born": 0,
                  "modules_resurrected": 0, "candidates": 0, "approved": 0,
                  "replay_loss": 0, "replay_train_count": 0, "replay_fraction": 0.0,
                  "forgetting": 0}

        # ── AUTONOMOUS DRIVES: apply strategy from previous cycle ──
        drives = getattr(self, "autonomous_drives", None)
        if drives is not None and drives.enabled and self._last_cycle_strategy is not None:
            drives.apply_strategy_to_config(self.cfg, self._last_cycle_strategy)

        # STAGE 1-4: Data lifecycle
        self._data_lifecycle(report)

        # STAGE 5-7: Train + Eval
        self._train_cycle(steps, report)

        # STAGE 7.5: Execute structural actions at SAFE boundary
        self._execute_structural_boundary(report)

        # ── OrganSenate: convene at cycle boundary ──
        if getattr(self.cfg, "senate_enabled", False):
            self._convene_organ_senate()

        # STAGE 8: Evolution — read neuroendocrine state instead of old predictors
        self._evolution_cycle(report)

        model_config = getattr(self, "model_config", None)
        if getattr(model_config, "sleep_enabled", False):
            self._sleep_cycle(report)
        if getattr(model_config, "unified_mesh_enabled", False):
            self._evolution_death_loop(report)

        # ── AUTONOMOUS DRIVES: cycle boundary — select strategy for next cycle ──
        if drives is not None and drives.enabled:
            loss_end = report.get("loss_end", 0)
            drive_result = drives.cycle_boundary(
                cycle=self.cycle,
                loss_end=loss_end,
                model_config=self.model_config,
                expert_pool=getattr(self, "expert_pool", None),
                total_steps=self.total_steps,
            )
            self._last_cycle_strategy = drive_result["strategy"]
            report["autonomous_drives"] = drive_result["diagnostics"]

        # STAGE 9-10: every completed cycle is a scientific checkpoint.  The
        # CLI ``--save-every`` flag remains accepted only for compatibility;
        # cycle boundaries are the atomic resume unit for run247.
        self._save_cycle(report)

        return report

    @staticmethod
    def _structural_log_details(model, *, status: str) -> tuple[str, ...]:
        details: list[str] = []
        for block_index, block in enumerate(getattr(model, "blocks", ())):
            moe = getattr(block, "moe", None)
            log = getattr(moe, "_structural_event_log", None)
            if not isinstance(log, list) or not log:
                continue
            latest = next(
                (
                    event
                    for event in reversed(log)
                    if isinstance(event, dict)
                    and event.get("status") == status
                    and any(
                        key in event
                        for key in (
                            "neurogenesis",
                            "prune_experts",
                            "expand_experts",
                        )
                    )
                ),
                None,
            )
            if latest is None:
                continue
            if latest.get("neurogenesis"):
                details.append(f"proposed:create:B{block_index}")
            for expert_index in latest.get("prune_experts", ()):
                details.append(
                    f"proposed:prune:B{block_index}_E{int(expert_index)}"
                )
            for expert_index in latest.get("expand_experts", ()):
                details.append(
                    f"proposed:expand:B{block_index}_E{int(expert_index)}"
                )
        return tuple(sorted(details))

    @staticmethod
    def _structural_report_details(
        structural_report,
        *,
        expected: tuple[str, ...],
    ) -> tuple[str, ...]:
        counts = {
            "create": sum(
                int(item.get("neurogenesis", 0))
                for item in structural_report
            ),
            "prune": sum(
                int(item.get("pruned", 0)) for item in structural_report
            ),
            "expand": sum(
                int(item.get("expanded", 0)) for item in structural_report
            ),
        }
        expected_counts = {
            action: sum(
                f"proposed:{action}:" in detail for detail in expected
            )
            for action in counts
        }
        if counts != expected_counts:
            return ()
        return expected

    @staticmethod
    def _structural_preflight_errors(model) -> tuple[str, ...]:
        errors: list[str] = []
        for block_index, block in enumerate(getattr(model, "blocks", ())):
            moe = getattr(block, "moe", None)
            preflight = getattr(moe, "structural_preflight_error", None)
            if not callable(preflight):
                continue
            log = getattr(moe, "_structural_event_log", ())
            proposal = next(
                (
                    event
                    for event in reversed(log)
                    if isinstance(event, dict)
                    and event.get("status") == "proposed_not_applied"
                ),
                None,
            )
            if proposal is None:
                continue
            reason = preflight(proposal)
            if reason is not None:
                errors.append(f"B{block_index}:{reason}")
        return tuple(errors)

    def _execute_structural_boundary(self, report: dict) -> None:
        """Mutate topology only for an exact pending causal authorization."""
        from f51_darwin.organism.causal_bus import (
            OrganCausalBus,
            Phase,
            StepIdentity,
            StepOutcome,
        )
        from f51_darwin.organism.unified_mesh import (
            StructuralProposalAdapter,
            UnifiedControlMesh,
            execute_evolution_actions,
        )

        causal_bus = getattr(self, "causal_bus", None)
        proposal_count = 0
        proposal_details: list[str] = []
        decision = None
        boundary_bus = None
        identity = None
        if causal_bus is not None:
            signals = UnifiedControlMesh(self.model).collect_signals(
                getattr(self, "_last_training_output", None)
            )
            proposals = execute_evolution_actions(self.model, signals)
            proposal_count = int(proposals.pruning_decisions)
            proposal_details = sorted(proposals.details)
            serial = int(getattr(self, "_causal_attempt_serial", 0))
            self._causal_attempt_serial = serial + 1
            proposal_bytes = json.dumps(
                {
                    "count": proposal_count,
                    "details": proposal_details,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            identity = StepIdentity(
                run_id=str(
                    getattr(self.cfg, "canary_run_id", None)
                    or getattr(
                        self.cfg,
                        "organism_name",
                        "darwin-organism",
                    )
                ),
                cycle=int(getattr(self, "cycle", 0)),
                optimizer_step=int(getattr(self, "total_steps", 0)),
                accumulation_window=0,
                batch_digest=hashlib.sha256(proposal_bytes).hexdigest(),
                rng_digest=hashlib.sha256(
                    torch.random.get_rng_state().cpu().numpy().tobytes()
                ).hexdigest(),
                base_checkpoint_id=str(
                    getattr(self, "base_checkpoint_id", "")
                ),
                training_contract_id=str(
                    getattr(
                        self,
                        "training_contract_id",
                        "darwin-organism-causal-training-v1",
                    )
                ),
                ablation_plan_id="cycle-boundary",
                attempt_id=f"cycle-boundary-{serial}",
            )
            boundary_bus = OrganCausalBus(
                arm=causal_bus.arm,
                adapters=(StructuralProposalAdapter(),),
                intent_recorder=getattr(self, "causal_ledger", None),
            )
            decision = boundary_bus.decide(
                identity,
                Phase.CYCLE_BOUNDARY,
                {
                    "proposal_count": proposal_count,
                    "proposals": proposal_details,
                },
            )
            boundary_bus.record_intent(identity, (decision,))

        effective = () if decision is None else decision.effective
        pending_details = self._structural_log_details(
            self.model,
            status="proposed_not_applied",
        )
        effective_value = (
            dict(effective[0].value) if len(effective) == 1 else {}
        )
        exact_effective = (
            len(effective) == 1
            and effective[0].target.value == "STRUCTURAL_ACTION"
            and effective[0].operation.value == "QUEUE"
            and effective[0].subject == "model.deferred_structural_actions"
            and effective_value.get("proposal_count") == proposal_count
            and tuple(effective_value.get("actions", ()))
            == tuple(proposal_details)
        )
        pending_matches = pending_details == tuple(proposal_details)
        preflight_errors = (
            self._structural_preflight_errors(self.model)
            if causal_bus is not None and effective and pending_matches
            else ()
        )
        authorized = causal_bus is None or (
            exact_effective and pending_matches and not preflight_errors
        )
        report["structural"] = {
            "authorized": authorized,
            "proposal_count": proposal_count,
            "proposals": proposal_details,
            "pending_actions": list(pending_details),
            "preflight_errors": list(preflight_errors),
            "effective_intervention_ids": [
                item.intervention_id for item in effective
            ],
        }

        def close_outcome(
            *,
            accepted_ids: tuple[str, ...] = (),
            error: str | None = None,
        ) -> None:
            if boundary_bus is None or identity is None:
                return
            outcome = StepOutcome(
                identity=identity,
                optimizer_step_applied=False,
                accepted_intervention_ids=accepted_ids,
                error=error,
            )
            ledger = getattr(self, "causal_ledger", None)
            if ledger is not None:
                ledger.record_outcome(outcome)
            boundary_bus.feedback(outcome)

        if not authorized:
            contract_error = None
            if preflight_errors:
                report["structural"]["error"] = "structural_preflight_rejected"
                contract_error = (
                    "structural_contract_error:preflight:"
                    + ",".join(preflight_errors)
                )
            elif effective and not pending_matches:
                report["structural"]["error"] = "pending_actions_mismatch"
                contract_error = (
                    "structural_contract_error:pending_actions_mismatch"
                )
            elif effective and not exact_effective:
                report["structural"]["error"] = "intervention_mismatch"
                contract_error = (
                    "structural_contract_error:intervention_mismatch"
                )
            close_outcome(error=contract_error)
            return

        # Expert birth/death/expansion happens after optimizer.step and before
        # the next forward. Preserve optimizer moments across any shape change.
        named_params_fn = getattr(self.model, "named_parameters", None)
        old_named_params = (
            dict(named_params_fn()) if callable(named_params_fn) else {}
        )
        try:
            structural_report = self.model.execute_structural_actions()
            if causal_bus is not None:
                executed_details = self._structural_log_details(
                    self.model,
                    status="executed",
                )
                report_details = self._structural_report_details(
                    structural_report,
                    expected=tuple(proposal_details),
                )
                if (
                    executed_details != tuple(proposal_details)
                    or report_details != tuple(proposal_details)
                ):
                    raise RuntimeError("structural execution mismatch")

            topology_changed = any(
                item.get("pruned", 0)
                + item.get("expanded", 0)
                + item.get("neurogenesis", 0)
                > 0
                for item in structural_report
            )
            total_pruned = sum(
                item.get("pruned", 0) for item in structural_report
            )
            total_expanded = sum(
                item.get("expanded", 0) for item in structural_report
            )
            total_born = sum(
                item.get("neurogenesis", 0) for item in structural_report
            )
            remap_stats = None
            if topology_changed:
                old_optimizer = self.optimizer
                old_lr = old_optimizer.param_groups[0]["lr"]
                # Preserve JEPA ×10 LR group when the predictor exists.
                jepa_predictor = getattr(self.model, "jepa_predictor", None)
                if jepa_predictor is not None:
                    new_optimizer = torch.optim.AdamW(
                        self._jepa_param_groups(),
                        lr=old_lr,
                        weight_decay=self.cfg.weight_decay,
                        fused=True,
                    )
                else:
                    new_optimizer = build_optimizer(
                        self.cfg.optimizer_name,
                        self.model.named_parameters(),
                        lr=old_lr,
                        weight_decay=self.cfg.weight_decay,
                    )
                remap_stats = remap_optimizer_state(
                    old_optimizer,
                    old_named_params,
                    new_optimizer,
                    dict(self.model.named_parameters()),
                )
                self.optimizer = new_optimizer
                self._apply_lr_schedule(self.total_steps)
                # Offload moments to CPU after topology change (matches fresh-start path).
                from f51_darwin.organism.checkpoint import offload_optimizer_states_to_cpu
                offload_optimizer_states_to_cpu(self.optimizer)
        except Exception as exc:
            close_outcome(
                error=f"structural_error:{type(exc).__name__}:{exc}"
            )
            raise

        close_outcome(
            accepted_ids=tuple(
                item.intervention_id for item in effective
            )
        )
        if not topology_changed:
            return
        if total_pruned:
            print(f"  ☠️  Apoptose: {total_pruned} experts removidos")
        if total_expanded:
            print(f"  📈 Expansão: {total_expanded} experts cresceram")
        if total_born:
            print(f"  🆕 Neurogênese: {total_born} experts nasceram")
        print(
            # ``remap_stats`` is guaranteed for a topology change.
            f"  ⚡ Topologia: {remap_stats['preserved']} momentos preservados, "
            f"{remap_stats['initialised']} fresh (neurogênese/expansão), "
            f"{remap_stats['dropped']} descartados (prune)"
        )
        # ── Sync ExpertPool with real MoE topology ──
        try:
            expert_pool = getattr(self, "expert_pool", None)
            if expert_pool is not None:
                for block_idx, sreport in enumerate(structural_report):
                    for pruned_idx in sreport.get("pruned", ()):
                        expert_pool.mark_dead(f"L{block_idx}_E{int(pruned_idx)}")
                    for new_idx in sreport.get("neurogenesis", ()):
                        expert_id = f"L{block_idx}_E{int(new_idx)}"
                        if not expert_pool.has_expert(expert_id):
                            expert_pool.add_expert(
                                expert_id, None,
                                created_at_cycle=self.cycle,
                                state=ModuleState.ACTIVE,
                            )
        except Exception as exc:
            print(f"  ⚠️ ExpertPool sync: {exc}")

    # ═══════════════════════════════════════════════════════════
    # PLASTICITY ENGINE — 5 Preditores → Temperatura → Legacy
    # (Ported from src/f51_darwin/organism.py)
    # ═══════════════════════════════════════════════════════════

    def compute_temperature(
        self,
        expert_id: str,
        score: float = 0.0,
        hidden_states: torch.Tensor | None = None,
        loss: float | None = None,
    ) -> float:
        """5 preditores votam na temperatura do expert.

        Ghost x 3  +  JEPA x 2  +  Spider x 4  +  Consensus x 2  +  Curiosity x 1
        Range: 0 a 12
        """
        temperature = 0.0

        # GHOST TOKEN (peso 3) — variance of hidden states
        try:
            if hidden_states is not None:
                ghost_err = float(hidden_states.float().var().mean())
                temperature += max(0.0, 3.0 - ghost_err * 3.0)
            else:
                # Fallback: use score mapping
                temperature += max(0.0, (score + 1.0) * 1.5)  # 0-3
        except Exception:
            temperature += 1.5

        # JEPA (peso 2) — representation quality
        try:
            if hidden_states is not None and hasattr(self.model, 'jepa_predictor') and self.model.jepa_predictor is not None:
                jepa_quality = float(hidden_states.float().norm() / max(hidden_states.numel(), 1))
                temperature += jepa_quality * 2.0
            else:
                temperature += max(0.0, (score + 1.0) * 1.0)  # 0-2
        except Exception:
            temperature += 1.0

        # SPIDER SENSE (peso 4) — danger = HOT (needs attention)
        try:
            if hidden_states is not None and hasattr(self.model, 'spider_sense') and self.model.spider_sense is not None:
                from f51_darwin.spider_sense import SpiderSense
                danger = float(self.model.spider_sense(hidden_states).mean())
                temperature += danger * 4.0
            else:
                # Higher score = less danger = more stable = hotter
                temperature += max(0.0, (score + 1.0) * 2.0)  # 0-4
        except Exception:
            temperature += 2.0

        # CONSENSUS (peso 2) — historical score agreement
        if not hasattr(self, '_consensus_scores'):
            self._consensus_scores: dict[str, list[float]] = {}
        if expert_id not in self._consensus_scores:
            self._consensus_scores[expert_id] = []
        self._consensus_scores[expert_id].append(score)
        scores = self._consensus_scores[expert_id][-10:]
        if scores:
            consensus = sum(scores) / len(scores)
            temperature += max(0.0, (consensus + 1.0) * 1.0)  # 0-2

        # CURIOSITY (peso 1) — exploration reward
        try:
            if hasattr(self.model, 'curiosity') and self.model.curiosity is not None and loss is not None:
                reward = float(self.model.curiosity.evaluate_curiosity_reward(
                    hidden_states.float() if hidden_states is not None else torch.zeros(1, 1, self.model_config.d_model),
                    jepa_prediction_error=float(loss),
                    domain="training",
                    loss_improved=(loss < 10.0),
                ))
                temperature += max(0.0, reward * 1.0)
            else:
                temperature += 0.5
        except Exception:
            temperature += 0.5

        return min(12.0, max(0.0, temperature))

    def _data_lifecycle(self, report: dict):
        """Stages 1-4: extract (GroundedExtractor) → firewall → promote."""
        # STAGE 1: Extract project-grounded candidates via GroundedExtractor
        try:
            extractor = GroundedExtractor(self.root)
            extractions = extractor.extract_project_knowledge(
                "f51_darwin",
                extraction_types=["module_contract", "architecture_note"],
            )
            for ext in extractions[:10]:  # Limit per cycle
                if ext.content.strip():
                    source_type = SourceType.REAL
                    if ext.epistemic_level.value in ("inferred", "hypothesis"):
                        source_type = SourceType.EDITED
                    self.data_factory.register_candidate(
                        text=ext.content,
                        source_type=source_type,
                        source_path=ext.evidence.source_file,
                        generator_model="grounded_extractor" if source_type == SourceType.EDITED else None,
                        generator_checkpoint=f"cycle_{self.cycle}" if source_type == SourceType.EDITED else None,
                        dataset_version=f"organism_cycle_{self.cycle}",
                    )
        except Exception as e:
            print(f"   ⚠️ GroundedExtractor: {e}")

        # STAGES 2-4: Firewall audit + promote
        try:
            candidates = self.data_factory.list_candidates()
            for record, text in candidates[:10]:
                decision = self.data_factory.audit_candidate(record, text)
                report["candidates"] += 1
                if decision.status == DatasetStatus.APPROVED:
                    report["approved"] += 1
            if report["approved"] > 0:
                self.data_factory.promote_approved_to_corpus(build_corpus=False)
        except Exception as e:
            print(f"   ⚠️ Data lifecycle: {e}")

    def _sleep_cycle(self, report: dict) -> None:
        """Run replay sleep as a protected evaluate/promote transaction."""
        try:
            from f51_darwin.organism.unified_mesh import (
                UnifiedControlMesh,
                run_protected_sleep,
                run_sleep_cycle,
            )
            signals = UnifiedControlMesh(self.model).collect_signals(
                getattr(self, "_last_training_output", None)
            )
            proposal = run_sleep_cycle(self.model, signals)
            if not proposal.proposed:
                report["sleep"] = asdict(proposal)
                return
            protected_inputs: dict[str, Any] = {}

            def protected_batches():
                if "batches" not in protected_inputs:
                    protected_inputs["batches"] = (
                        self._protected_sleep_batches()
                    )
                return protected_inputs["batches"]

            sleep_report = run_protected_sleep(
                self.model,
                self.optimizer,
                enabled=True,
                prepare=lambda: protected_batches(),
                consolidate=lambda: self._consolidate_sleep_replay(
                    protected_batches()[1]
                ),
                evaluate=lambda: self._evaluate_protected_sleep(
                    *protected_batches(),
                ),
                snapshot_organs=self._snapshot_sleep_organs,
                restore_organs=self._restore_sleep_organs,
                max_regression=float(
                    getattr(self.cfg, "sleep_max_regression", 0.0)
                ),
            )
            sleep_report.proposed_prunable_params = (
                proposal.proposed_prunable_params
            )
            report["sleep"] = asdict(sleep_report)
        except Exception as exc:
            self._record_organ_error(report, "sleep_rem", exc)

    def _protected_sleep_batches(self):
        block_size = int(getattr(self.cfg, "block_size", 0))
        batch_size = int(getattr(self.cfg, "batch_size", 1))
        tokens = getattr(self, "train_token_ids", None)
        fresh_batch = None
        if (
            tokens is not None
            and block_size > 0
            and len(tokens) >= block_size + batch_size
        ):
            rows = [
                list(tokens[offset : offset + block_size])
                for offset in range(batch_size)
            ]
            fresh_batch = torch.tensor(
                rows,
                dtype=torch.long,
                device=self.device,
            )
        replay_samples = []
        replay = getattr(self, "replay", None)
        if replay is not None and len(replay) > 0:
            replay_samples = replay.sample(
                min(
                    int(getattr(self.cfg, "replay_sample_size", 1)),
                    len(replay),
                )
            )
        return fresh_batch, replay_samples

    def _evaluate_protected_sleep(
        self,
        fresh_batch: torch.Tensor | None,
        replay_samples,
    ) -> dict[str, float]:
        if fresh_batch is None:
            raise RuntimeError("protected sleep has no fixed fresh batch")
        if not replay_samples:
            raise RuntimeError("protected sleep has no replay batch")
        if not getattr(self, "holdout_starts", ()):
            raise RuntimeError("protected sleep has no fixed heldout batches")

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                fresh_output = self._protected_eval_forward(
                    fresh_batch,
                    domain="protected_sleep_fresh",
                )
                fresh_loss = getattr(
                    fresh_output,
                    "lm_loss",
                    getattr(fresh_output, "loss", None),
                )
                if not isinstance(fresh_loss, torch.Tensor):
                    raise RuntimeError(
                        "protected sleep fresh loss is unavailable"
                    )
                replay_loss = self._evaluate_protected_replay_loss(
                    replay_samples
                )
                if replay_loss is None:
                    raise RuntimeError(
                        "protected sleep replay loss is unavailable"
                    )
                heldout = evaluate_fixed_holdout(
                    self.model,
                    self.holdout_token_ids,
                    starts=self.holdout_starts,
                    block_size=self.cfg.block_size,
                    device=self.device,
                    amp_dtype=self.amp_dtype,
                )
            return {
                "fresh": float(fresh_loss.detach().cpu()),
                "replay": float(replay_loss),
                "heldout": float(heldout.lm_loss),
            }
        finally:
            self.model.train(was_training)

    def _protected_eval_forward(
        self,
        batch: torch.Tensor,
        *,
        domain: str,
    ):
        parameters = inspect.signature(self.model.forward).parameters
        supports_heartbeat = (
            "heartbeat" in parameters
            or any(
                item.kind is inspect.Parameter.VAR_KEYWORD
                for item in parameters.values()
            )
        )
        kwargs: dict[str, Any] = {
            "labels": batch,
            "domain": domain,
        }
        if supports_heartbeat:
            kwargs["heartbeat"] = False
        return self.model(batch, **kwargs)

    def _evaluate_protected_replay_loss(self, replay_samples) -> float | None:
        weighted_loss = 0.0
        evaluated = 0
        microbatch_size = max(int(self.cfg.batch_size), 1)
        for offset in range(0, len(replay_samples), microbatch_size):
            chunk = replay_samples[offset : offset + microbatch_size]
            batch = self._batch_from_replay(chunk)
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_dtype is not None,
            ):
                output = self._protected_eval_forward(
                    batch,
                    domain="protected_sleep_replay",
                )
            loss = getattr(output, "loss", None)
            if not isinstance(loss, torch.Tensor) or not torch.isfinite(loss):
                continue
            weighted_loss += float(loss.detach().cpu()) * len(chunk)
            evaluated += len(chunk)
        if evaluated == 0:
            return None
        return weighted_loss / evaluated

    def _consolidate_sleep_replay(self, replay_samples) -> None:
        if not replay_samples:
            raise RuntimeError("protected sleep requires replay consolidation")
        batch = self._batch_from_replay(replay_samples)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_dtype is not None,
        ):
            output = self.model(
                batch,
                labels=batch,
                domain="protected_sleep_replay",
            )
            loss = getattr(output, "loss", None)
        if not isinstance(loss, torch.Tensor) or not torch.isfinite(loss):
            raise RuntimeError("protected sleep consolidation loss is invalid")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            float(getattr(self.cfg, "grad_clip", 1.0)),
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _snapshot_sleep_organs(self) -> dict[str, Any]:
        runtime_names = (
            "_vertical_bias",
            "_ghost_loss",
            "_router_entropy",
            "_expert_usage",
            "_pending_autonomic_actions",
            "_sleep_active",
            "_sleep_steps_remaining",
            "_nitro_step_counter",
            "_autonomic_forward_uses",
            "_autonomic_usage_accumulator",
            "_autonomic_entropy_sum",
            "_autonomic_entropy_observations",
            "_dae_observation_accumulator",
            "_replay_buffer",
            "_replay_count",
            "_signature_inputs",
            "_signature_buffer",
            "_ghost_predator_buffer",
            "_marked_for_apoptosis",
            "_disabled_experts",
            "_structural_event_log",
        )
        moe_runtime = []
        for block_index, block in enumerate(
            getattr(self.model, "blocks", ())
        ):
            moe = getattr(block, "moe", None)
            if moe is None:
                continue
            runtime_state_fn = getattr(moe, "runtime_state_dict", None)
            moe_runtime.append(
                {
                    "block_index": block_index,
                    "runtime_state": (
                        copy.deepcopy(runtime_state_fn())
                        if callable(runtime_state_fn)
                        else None
                    ),
                    "values": (
                        {}
                        if callable(runtime_state_fn)
                        else {
                            name: copy.deepcopy(getattr(moe, name))
                            for name in runtime_names
                            if hasattr(moe, name)
                        }
                    ),
                }
            )
        replay = getattr(self, "replay", None)
        return {
            "replay": (
                None
                if replay is None
                else copy.deepcopy(replay.state_dict())
            ),
            "metric_channels": copy.deepcopy(
                getattr(self, "metric_channels", None)
            ),
            "last_training_metrics": copy.deepcopy(
                getattr(self, "_last_training_metrics", None)
            ),
            "last_training_output": copy.deepcopy(
                getattr(self, "_last_training_output", None)
            ),
            "causal_requests": copy.deepcopy(
                getattr(self, "_causal_requests", [])
            ),
            "moe_runtime": moe_runtime,
        }

    def _restore_sleep_organs(self, state: dict[str, Any]) -> None:
        replay_state = state.get("replay")
        replay = getattr(self, "replay", None)
        if replay is not None and replay_state is not None:
            replay.load_state_dict(replay_state)
        self.metric_channels = state.get("metric_channels")
        self._last_training_metrics = state.get("last_training_metrics")
        self._last_training_output = state.get("last_training_output")
        self._causal_requests = list(state.get("causal_requests", ()))
        blocks = getattr(self.model, "blocks", ())
        for entry in state.get("moe_runtime", ()):
            block_index = int(entry["block_index"])
            moe = blocks[block_index].moe
            runtime_state = entry.get("runtime_state")
            load_runtime_state_fn = getattr(
                moe, "load_runtime_state_dict", None
            )
            if runtime_state is not None:
                if not callable(load_runtime_state_fn):
                    raise RuntimeError(
                        "MoE runtime snapshot has no compatible loader"
                    )
                load_runtime_state_fn(copy.deepcopy(runtime_state))
                continue
            for name, value in entry["values"].items():
                current = getattr(moe, name, None)
                if isinstance(current, torch.Tensor) and isinstance(
                    value, torch.Tensor
                ):
                    value = value.to(
                        device=current.device,
                        dtype=current.dtype,
                    )
                else:
                    value = copy.deepcopy(value)
                setattr(moe, name, value)

    def _evolution_death_loop(self, report: dict) -> None:
        """Stage 8.5: Observe legacy plasticity proposals without mutation."""
        try:
            from f51_darwin.organism.unified_mesh import (
                UnifiedControlMesh,
                execute_evolution_actions,
            )
            signals = UnifiedControlMesh(self.model).collect_signals(
                getattr(self, "_last_training_output", None)
            )
            evo_report = execute_evolution_actions(self.model, signals)
            if 'evolution' not in report:
                report['evolution'] = {}
            report['evolution']['deaths_this_cycle'] = evo_report.deaths
            report['evolution']['quarantines_this_cycle'] = evo_report.quarantines
            report['evolution']['expansions_this_cycle'] = evo_report.expansions
            report['evolution']['pruning_decisions'] = evo_report.pruning_decisions
            report['evolution']['proposals'] = list(evo_report.details)
        except Exception as exc:
            self._record_organ_error(report, "unified_mesh", exc)

    @staticmethod
    def _record_organ_error(report: dict, organ: str, error: Exception) -> None:
        degraded = report.setdefault("degraded_organs", [])
        if organ not in degraded:
            degraded.append(organ)
        report.setdefault("organ_errors", {})[organ] = str(error)
