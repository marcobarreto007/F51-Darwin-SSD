from __future__ import annotations

import hashlib
import inspect
from types import SimpleNamespace

from .dependencies import *  # noqa: F403
from .checkpoint import *  # noqa: F403
from .causal_adapters import (
    CausalExecutionError,
    CausalTrainingExecutor,
    accepted_ids,
    detached_losses,
    gradient_norm,
    training_context,
    training_step_identity,
)
from .causal_bus import AblationArm, Phase, StepOutcome
from .lifecycle import _DarwinLifecycleMixin
from .support import *  # noqa: F403


def _accepts_step_digest(model) -> bool:
    """Return whether the model forward contract accepts the causal digest."""
    try:
        parameters = inspect.signature(model.forward).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.name == "step_digest" for parameter in parameters)


def _compute_gmc_signal(model, optimizer) -> float:
    """Gradient-Momentum Coupling (GMC): cosine similarity entre gradiente e
    primeiro momento do AdamW.

    Retorna valor em [-1, 1]. Alto = gradiente alinhado com momentum
    (progresso real). Baixo/negativo = ruido ou conflito (revisao necessaria).
    """
    if optimizer is None:
        return 0.0
    dot = 0.0
    g_norm_sq = 0.0
    m_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        state = optimizer.state.get(param)
        if state is None or "exp_avg" not in state:
            continue
        g = param.grad.detach().float().cpu()
        m = state["exp_avg"].detach().float().cpu()
        dot += float((g * m).sum())
        g_norm_sq += float(g.pow(2).sum())
        m_norm_sq += float(m.pow(2).sum())
    g_norm = g_norm_sq ** 0.5
    m_norm = m_norm_sq ** 0.5
    if g_norm > 1e-8 and m_norm > 1e-8:
        return float(max(-1.0, min(1.0, dot / (g_norm * m_norm))))
    return 0.0


class _DarwinTrainingMixin(_DarwinLifecycleMixin):
    """Training loop and metric collection."""

    def _train_cycle(self, steps: int, report: dict):
        """Stages 5-7: Train with all organs active."""
        model = self.model
        cfg = self.cfg
        self._validate_holdout_training_source()
        self.metric_channels = MetricChannels.create(window_size=20)
        self._metrics_started_at = time.time()
        last_heldout: HeldoutResult | None = None

        # Curriculum: use weighted multi‑source loader when available
        weighted = getattr(self, 'weighted_sources', None)
        sampler_mode = str(
            getattr(cfg, "sampler_mode", "sequential")
        ).strip().lower()
        sampler_version = int(getattr(cfg, "sampler_version", 1))
        if weighted:
            if sampler_mode != "sequential":
                raise RuntimeError(
                    "permuted_blocks sampler is incompatible with weighted_sources"
                )
            from f51_darwin.data import WeightedCorpusLoader
            loader = WeightedCorpusLoader(
                weighted,
                block_size=cfg.block_size,
                batch_size=cfg.batch_size,
                seed=cfg.seed + self.cycle,
                device=self.device,
            )
        else:
            loader_tokens = getattr(self, "train_token_ids", self.token_ids)
            loader_seed = (
                cfg.seed
                if sampler_mode == "permuted_blocks"
                else cfg.seed + self.cycle
            )
            loader = CausalLMDataLoader(
                loader_tokens, block_size=cfg.block_size,
                batch_size=cfg.batch_size, seed=loader_seed,
                sampler_mode=sampler_mode,
                sampler_version=sampler_version,
                device=self.device,
            )
        loader.set_step(self.total_steps)

        model.train()
        losses = []
        replay_losses = []
        replay_train_losses = []  # GATE 0.1: replay que entrou no gradiente
        fresh_train_count = 0
        tokens_processed = 0
        t0 = time.time()
        # R3: gradient accumulation over N micro-batches (1 = legacy behaviour).
        accum_steps = max(1, int(getattr(cfg, "accum_steps", 1)))
        # R6: brainstem homeostasis guard (None when constructed via __new__).
        brainstem = getattr(self, "brainstem", None)
        # Start the first accumulation window with a clean grad buffer.
        self.optimizer.zero_grad(set_to_none=True)
        # Defaults so the report finalization is safe even when the loop exits
        # before the first logging step (e.g. an R5 emergency break at step 1).
        cur_lm = 0.0
        cur_ppl = 0.0

        for local_step in range(1, steps + 1):
            max_train_tokens = getattr(cfg, "max_train_tokens", None)
            batch_tokens = int(cfg.batch_size) * int(cfg.block_size)
            if (
                max_train_tokens is not None
                and int(getattr(self, "train_tokens_seen", 0)) + batch_tokens
                > max_train_tokens
            ):
                break
            # R5: honour an emergency-save request within ~1 step instead of
            # losing up to steps_per_cycle of progress on crash/SIGTERM.
            if getattr(self, "_emergency_save_requested", False):
                print("  🧯 Emergency save requested — interrompendo ciclo.")
                break

            replay_batch = None
            if (
                len(self.replay) >= cfg.replay_warmup_examples
                and replay_step_due(local_step, cfg.replay_ratio)
            ):
                samples = self.replay.sample(cfg.batch_size)
                replay_batch = self._batch_from_replay(samples)
                batch = replay_batch
            else:
                batch = loader.next_batch()
            batch_kind = "replay" if replay_batch is not None else "fresh"
            causal_bus = getattr(self, "causal_bus", None)
            optimizer_due = (local_step % accum_steps == 0) or (local_step == steps)
            causal_identity = None
            causal_decisions = []
            raw_losses = {}
            effective_losses = {}
            causal_executor = None
            pre_loss = None
            pre_backward = None
            pre_optimizer = None

            if causal_bus is not None:
                # SpiderRAM: persistent CPU memory so SpiderCalibrationAdapter
                # can read the *previous* step's danger (adapters fire before
                # the forward pass, so current losses aren't available yet).
                if getattr(self, "spider_ram", None) is None:
                    from f51_darwin.spider_sense import SpiderRAM
                    self.spider_ram = SpiderRAM()
                causal_identity = training_step_identity(
                    self,
                    batch,
                    accumulation_window=(local_step - 1) % accum_steps,
                )
                causal_executor = CausalTrainingExecutor(model)
                static_context = training_context(
                    raw_losses={},
                    effective_losses={},
                    requests=tuple(
                        getattr(self, "_causal_requests", ())
                    ),
                    optimizer_due=optimizer_due,
                    spider_ram=self.spider_ram.state(),
                    # Same one-step-lag pattern as spider_ram: GABAInterventionAdapter
                    # fires at PRE_LOSS, before this step's forward pass exists, so
                    # it can only see what the PREVIOUS step committed below.
                    gaba_observations=getattr(self, "_gaba_ram", None),
                )
                pre_loss = causal_bus.decide(
                    causal_identity, Phase.PRE_LOSS, static_context
                )
                causal_decisions.append(pre_loss)
                pre_backward = causal_bus.decide(
                    causal_identity, Phase.PRE_BACKWARD, static_context
                )
                causal_decisions.append(pre_backward)
                if optimizer_due:
                    pre_optimizer = causal_bus.decide(
                        causal_identity,
                        Phase.PRE_OPTIMIZER,
                        static_context,
                    )
                    causal_decisions.append(pre_optimizer)

                # One aggregate intent is durable before model.forward or any
                # training mutation in this attempt.
                causal_bus.record_intent(
                    causal_identity, causal_decisions
                )
                blocked = next(
                    (
                        decision
                        for decision in causal_decisions
                        if decision.blocked
                    ),
                    None,
                )
                if blocked is not None:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.metric_channels.skipped_updates += 1
                    self._complete_causal_training_step(
                        causal_identity,
                        causal_decisions,
                        raw_losses,
                        effective_losses,
                        optimizer_step_applied=False,
                        applied_intervention_ids=(),
                        error=_causal_block_reason(blocked),
                    )
                    continue
                try:
                    skip_update = causal_executor.should_skip(
                        pre_backward
                    )
                    if pre_optimizer is not None:
                        skip_update = (
                            causal_executor.should_skip(pre_optimizer)
                            or skip_update
                        )
                except CausalExecutionError as exc:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.metric_channels.skipped_updates += 1
                    self._complete_causal_training_step(
                        causal_identity,
                        causal_decisions,
                        raw_losses,
                        effective_losses,
                        optimizer_step_applied=False,
                        applied_intervention_ids=(
                            causal_executor.applied_intervention_ids
                        ),
                        error=str(exc),
                    )
                    continue
                if skip_update:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.metric_channels.skipped_updates += 1
                    effective_ids = accepted_ids(causal_decisions)
                    applied_ids = (
                        causal_executor.applied_intervention_ids
                    )
                    self._complete_causal_training_step(
                        causal_identity,
                        causal_decisions,
                        raw_losses,
                        effective_losses,
                        optimizer_step_applied=False,
                        applied_intervention_ids=applied_ids,
                        error=(
                            None
                            if set(applied_ids) == set(effective_ids)
                            else "update_skipped_before_other_interventions"
                        ),
                    )
                    continue

            try:
                # Compute causal step_digest for single-pass ghost when bus is active
                step_digest = None
                if causal_bus is not None:
                    step_digest = (
                        f"step-{self.total_steps}-"
                        f"{hashlib.sha256(batch.cpu().numpy().tobytes()).hexdigest()[:16]}"
                    )
                with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                    forward_kwargs = {
                        "labels": batch,
                        "domain": f"cycle_{self.cycle}",
                    }
                    if step_digest is not None and _accepts_step_digest(model):
                        forward_kwargs["step_digest"] = step_digest
                    output = model(batch, **forward_kwargs)
            except Exception as exc:
                if causal_bus is not None:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.metric_channels.skipped_updates += 1
                    self._complete_causal_training_step(
                        causal_identity,
                        causal_decisions,
                        raw_losses,
                        effective_losses,
                        optimizer_step_applied=False,
                        applied_intervention_ids=(
                            causal_executor.applied_intervention_ids
                        ),
                        error=f"forward_error:{type(exc).__name__}:{exc}",
                    )
                raise
            self._last_training_output = _organ_output_snapshot(output)
            # ── Commit GABA observations into neuroendocrine state ──
            if getattr(getattr(self, "model_config", None), "gaba_enabled", False):
                try:
                    n_gaba = model.commit_gaba_observations(output)
                except Exception:
                    n_gaba = 0
            else:
                n_gaba = 0
            if causal_bus is not None and n_gaba > 0:
                # Feed this step's GABA telemetry to the NEXT step's
                # GABAInterventionAdapter.observe() (one-step lag, same
                # reason as spider_ram: the adapter fires at PRE_LOSS,
                # before this step's forward pass exists yet).
                gaba_summary = []
                for obs in output.gaba_observations:
                    excitation = obs.get("sample_excitation")
                    inhibition = obs.get("sample_inhibition")
                    if not hasattr(excitation, "detach") or not hasattr(inhibition, "detach"):
                        continue
                    excitation = excitation.detach().float()
                    inhibition = inhibition.detach().float()
                    gaba_summary.append({
                        "mean_inhibition": float(inhibition.mean().cpu()),
                        "max_inhibition": float(inhibition.max().cpu()),
                        "inhibited_ratio": float(
                            (inhibition > excitation).float().mean().cpu()
                        ),
                    })
                self._gaba_ram = gaba_summary
            self._last_training_metrics = {
                **self.metric_channels.snapshot(
                    elapsed_sec=time.time() - self._metrics_started_at
                ),
                "heldout": None if last_heldout is None else asdict(last_heldout),
            }

            loss = output.loss
            if loss is None or not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                self.metric_channels.skipped_updates += 1
                if causal_bus is not None:
                    self._complete_causal_training_step(
                        causal_identity,
                        causal_decisions,
                        raw_losses,
                        effective_losses,
                        optimizer_step_applied=False,
                        applied_intervention_ids=(
                            causal_executor.applied_intervention_ids
                        ),
                        error="nonfinite_or_missing_loss",
                    )
                continue

            loss_val = float(loss.detach().cpu())
            if causal_bus is not None:
                raw_losses = detached_losses(output, loss)
                effective_losses = dict(raw_losses)
                # Feed SpiderRAM with this step's telemetry so the NEXT
                # step's SpiderCalibrationAdapter sees real danger data.
                if getattr(self, "spider_ram", None) is not None:
                    self.spider_ram.update_from_output(output)

            # R6: brainstem refuses to backpropagate pathological loss spikes
            # (explosion/collapse). Reset the accumulation window and skip.
            if brainstem is not None:
                homeo = brainstem.check_loss(loss_val)
                if not homeo.alive:
                    for violation in homeo.violations:
                        print(f"  🧠 BRAINSTEM (skip step): {violation}")
                    self.optimizer.zero_grad(set_to_none=True)
                    self.metric_channels.skipped_updates += 1
                    if causal_bus is not None:
                        self._complete_causal_training_step(
                            causal_identity,
                            causal_decisions,
                            raw_losses,
                            effective_losses,
                            optimizer_step_applied=False,
                            applied_intervention_ids=(
                                causal_executor.applied_intervention_ids
                            ),
                            error=(
                                "brainstem_rejected:"
                                + "|".join(homeo.violations)
                            ),
                        )
                    continue

            if causal_bus is not None:
                try:
                    loss, effective_aux = causal_executor.adjust_loss(
                        pre_loss, output, loss
                    )
                except CausalExecutionError as exc:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.metric_channels.skipped_updates += 1
                    self._complete_causal_training_step(
                        causal_identity,
                        causal_decisions,
                        raw_losses,
                        effective_losses,
                        optimizer_step_applied=False,
                        applied_intervention_ids=(
                            causal_executor.applied_intervention_ids
                        ),
                        error=str(exc),
                    )
                    continue
                if effective_aux is not None:
                    effective_losses["aux"] = effective_aux
                effective_losses["total"] = float(loss.detach().cpu())
                # Reflect effective (weighted) losses from output when
                # adjust_loss was not invoked — keeps effective_losses
                # faithful to the actual optimisation objective.
                for attr, key in (
                    ("effective_jepa_loss", "jepa"),
                    ("ghost_loss", "ghost"),
                    ("effective_spider_loss", "spider"),
                ):
                    value = getattr(output, attr, None)
                    if isinstance(value, torch.Tensor):
                        effective_losses[key] = float(value.detach().cpu())

            # ── Decision Engine: modulate exploration from confidence ──
            if getattr(
                getattr(self, "model_config", None),
                "decision_engine_enabled",
                False,
            ):
                try:
                    from f51_darwin.decision_engine import DecisionFactors, decide
                    factors = DecisionFactors(
                        veracity=float(max(0.0, 1.0 - loss_val / 10.0)),
                        consensus=0.5,
                        proxy_reward=float(max(0.0, 1.0 - loss_val / 15.0)),
                        robustness=0.7,
                        uncertainty=float(min(1.0, loss_val / 8.0)),
                        danger=float(min(1.0, max(0.0, loss_val - 5.0) / 10.0)),
                    )
                    self._last_decision_confidence = decide(factors)
                except Exception:
                    self._last_decision_confidence = 0.5
            # R2: linear LR warmup (no-op when warmup_steps <= 0).
            self._apply_lr_schedule(self.total_steps)

            # ── Dopamina: tesao real de aprender ──
            # Dopamina alta = aprendizado produtivo → amplifica gradiente
            # (reforco positivo, ciclo virtuoso).
            # Dopamina baixa = confuso/ruidoso → reduz gradiente (cautela).
            # Centrado em 0.5: acima expande, abaixo contrai.
            dopamine_weight = getattr(cfg, "dopamine_reward_weight", 0.0)
            if dopamine_weight > 0.0 and output.heartbeat_stats:
                dopamine = float(output.heartbeat_stats.get("dopamine", 0.5))
                dopamine_clamped = max(0.0, min(1.0, dopamine))
                reward_mod = 1.0 + (dopamine_clamped - 0.5) * 2.0 * dopamine_weight
                loss = loss * reward_mod

            # R3: scale loss so accumulated micro-batches average correctly,
            # then step the optimizer only at the accumulation boundary.
            (loss / accum_steps).backward()

            # ── GMC: Gradient-Momentum Coupling ──
            # Injeta o alinhamento gradiente×momentum no heartbeat para o
            # PROXIMO passo. Computado aqui (pos-backward, pre-optimizer-step)
            # porque o gradiente so existe depois de backward().
            if optimizer_due and hasattr(model, "heartbeat") and model.heartbeat is not None:
                try:
                    gmc = _compute_gmc_signal(model, self.optimizer)
                    model.heartbeat.inject_gmc(gmc)
                except Exception as _e:
                    # Log once per organism lifetime so we can debug
                    if not hasattr(self, "_gmc_error_logged"):
                        self._gmc_error_logged = True
                        try:
                            print(f"  ⚠️ GMC computation failed: {_e}", flush=True)
                        except Exception:
                            pass

            optimizer_step_applied = False
            grad_before = gradient_norm(model) if causal_bus is not None else None
            grad_after = grad_before
            if optimizer_due:
                try:
                    apply_active = getattr(model, "apply_active_gradient_actions", None)
                    self._last_dae_report = (
                        apply_active() if callable(apply_active) else []
                    )
                    if pre_optimizer is not None:
                        causal_executor.scale_gradients(pre_optimizer)
                        causal_executor.clip_gradients(
                            pre_optimizer, cfg.grad_clip
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.grad_clip
                        )
                    grad_after = (
                        gradient_norm(model)
                        if causal_bus is not None
                        else None
                    )
                    move_optimizer_state_for_parameters(
                        self.optimizer, model.parameters()
                    )
                    assert_optimizer_device_invariants(model, self.optimizer)
                    self.optimizer.step()
                    optimizer_step_applied = True
                except CausalExecutionError as exc:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.metric_channels.skipped_updates += 1
                    self._complete_causal_training_step(
                        causal_identity,
                        causal_decisions,
                        raw_losses,
                        effective_losses,
                        optimizer_step_applied=False,
                        gradient_norm_before=grad_before,
                        gradient_norm_after=None,
                        applied_intervention_ids=(
                            causal_executor.applied_intervention_ids
                        ),
                        error=str(exc),
                    )
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                from f51_darwin.organism.checkpoint import (
                    offload_optimizer_states_to_cpu,
                )
                offload_optimizer_states_to_cpu(self.optimizer)
                apply_autonomic = getattr(model, "apply_pending_autonomic_actions", None)
                if callable(apply_autonomic):
                    apply_autonomic()
                # A server controller may pause only once the optimizer step is
                # complete; with accumulation that is every N micro-batches.
                pause_checkpoint = getattr(self, "_runtime_pause_checkpoint", None)
                if callable(pause_checkpoint):
                    pause_checkpoint(model)

            if causal_bus is not None:
                self._complete_causal_training_step(
                    causal_identity,
                    causal_decisions,
                    raw_losses,
                    effective_losses,
                    optimizer_step_applied=optimizer_step_applied,
                    gradient_norm_before=grad_before,
                    gradient_norm_after=grad_after,
                    applied_intervention_ids=(
                        causal_executor.applied_intervention_ids
                    ),
                    error=None,
                )

            # ── Blockchain: record a block after each optimizer step ──
            if optimizer_step_applied:
                self._maybe_record_block()

            if causal_bus is not None:
                loss_val = float(loss.detach().cpu())
            losses.append(loss_val)
            tokens_processed += batch.numel()
            self.train_tokens_seen = int(getattr(self, "train_tokens_seen", 0)) + batch.numel()
            self.total_steps += 1

            # ── Mid-cycle save every save_every steps ──
            _save_every = int(getattr(self.cfg, "save_every", 0) or 0)
            if _save_every > 0 and self.total_steps % _save_every == 0:
                try:
                    self._save_cycle(report, mid_cycle_step=self.total_steps)
                except Exception as _save_err:
                    print(f"  ⚠️ mid-cycle save failed (step {self.total_steps}): {_save_err}")

            lm_loss_val = (
                float(output.lm_loss.detach().cpu())
                if getattr(output, "lm_loss", None) is not None
                else loss_val
            )
            self.metric_channels.record_update(
                batch_kind=batch_kind,
                total_loss=loss_val,
                lm_loss=lm_loss_val,
                tokens=batch.numel(),
                optimizer_step=optimizer_due,
            )

            # Track replay loss separately when replay was mixed in
            if replay_batch is not None and output.lm_loss is not None:
                replay_train_losses.append(loss_val)
            else:
                fresh_train_count += 1
                should_seed = (
                    len(self.replay) < cfg.replay_warmup_examples
                    or fresh_train_count % max(cfg.replay_add_every, 1) == 0
                )
                if should_seed:
                    # Compute curiosity-driven priority for replay
                    priority = 1.0
                    try:
                        if hasattr(model, 'curiosity') and model.curiosity is not None:
                            priority = float(model.curiosity.evaluate_curiosity_reward(
                                batch.float().mean(dim=0, keepdim=True),
                                jepa_prediction_error=float(loss_val),
                                domain="training",
                                loss_improved=(loss_val < getattr(self, '_prev_loss', float('inf'))),
                            ))
                            priority = max(0.01, min(10.0, priority))
                    except Exception:
                        priority = 1.0
                    for row in batch.detach().cpu():
                        self.replay.add(row.tolist(), label=f"step_{self.total_steps}", priority=priority)
                    self._prev_loss = float(loss_val)

            self._heartbeat_if_due(
                local_step, loss_val, tokens_processed, model, cfg
            )
            last_heldout = self._evaluate_if_due(
                local_step, model, cfg, replay_losses, last_heldout
            )
            metric_snapshot = self.metric_channels.snapshot(
                elapsed_sec=time.time() - self._metrics_started_at
            )
            metric_snapshot["heldout"] = (
                None if last_heldout is None else asdict(last_heldout)
            )
            self._last_training_metrics = metric_snapshot

            cur_lm, cur_ppl = self._log_if_due(
                local_step=local_step,
                output=output,
                loss_val=loss_val,
                losses=losses,
                replay_batch=replay_batch,
                replay_train_losses=replay_train_losses,
                batch_kind=batch_kind,
                tokens_processed=tokens_processed,
                started_at=t0,
                current=(cur_lm, cur_ppl),
            )
            # ── AUTONOMOUS DRIVES: feed per-step metrics ──
            drives = getattr(self, "autonomous_drives", None)
            if drives is not None and drives.enabled:
                _jl = getattr(output, "jepa_loss", None)
                _jepa = float(_jl.detach().cpu()) if hasattr(_jl, "detach") else float(_jl or 0.0)
                _dope = float(
                    output.heartbeat_stats.get("dopamine_raw", 0.5)
                    if output.heartbeat_stats else 0.5
                )
                _brainstem_rejected = (
                    brainstem is not None
                    and not getattr(
                        getattr(brainstem, "check_loss", lambda x: type("ok", (), {"alive": True})())(loss_val),
                        "alive", True
                    )
                )
                drives.feed_step(
                    dopamine_raw=_dope,
                    jepa_loss=_jepa,
                    loss_val=loss_val,
                    brainstem_rejected=_brainstem_rejected,
                )

            # Write dopamine status EVERY step (not just every 50)
            if output.heartbeat_stats:
                try:
                    import os as _os
                    _root = getattr(cfg, "checkpoint_root", "workspace/03_CHECKPOINTS_100M_FULL_V3")
                    _path = _os.path.join(_root, "dopamine_status.txt")
                    _s = output.heartbeat_stats
                    with open(_path, "w") as _f:
                        _jl = getattr(output, "jepa_loss", 0.0)
                        _raw_jepa = float(_jl.detach() if hasattr(_jl, "detach") else (_jl or 0.0))
                        _f.write(
                            f"step={self.total_steps} "
                            f"loss={loss_val:.3f} "
                            f"jepa_raw={_raw_jepa:.4f} "
                            f"dope={_s.get('dopamine_raw', '?')} "
                            f"delta={_s.get('dopamine_delta', '?')} "
                            f"gmc={_s.get('dopamine_gmc', '?')} "
                            f"jepa_bonus={_s.get('dopamine_jepa', '?')}\n"
                        )
                except Exception:
                    pass

        report["train_tokens_seen"] = int(getattr(self, "train_tokens_seen", 0))
        self._finalize_training_report(
            report, losses, replay_losses, last_heldout, cur_lm, cur_ppl
        )

    def _complete_causal_training_step(
        self,
        identity,
        decisions,
        raw_losses,
        effective_losses,
        *,
        optimizer_step_applied,
        applied_intervention_ids,
        gradient_norm_before=None,
        gradient_norm_after=None,
        error=None,
    ) -> None:
        applied_set = set(applied_intervention_ids)
        ordered_applied_ids = tuple(
            intervention.intervention_id
            for decision in decisions
            for intervention in decision.effective
            if intervention.intervention_id in applied_set
        )
        outcome = StepOutcome(
            identity=identity,
            optimizer_step_applied=optimizer_step_applied,
            accepted_intervention_ids=ordered_applied_ids,
            raw_losses=raw_losses,
            effective_losses=effective_losses,
            gradient_norm_before=gradient_norm_before,
            gradient_norm_after=gradient_norm_after,
            error=error,
        )
        causal_bus = self.causal_bus
        ledger = getattr(self, "causal_ledger", None)
        if ledger is not None:
            ledger.record_outcome(outcome)
        causal_bus.feedback(outcome)

    def _heartbeat_if_due(self, local_step, loss_val, tokens_processed, model, cfg):
        if local_step % 50 != 0:
            return
        status = {
            "loss": loss_val,
            "fresh_loss": metric_channel_lm_loss(
                self.metric_channels.fresh, default=loss_val
            ),
            "replay_loss": metric_channel_lm_loss(
                self.metric_channels.replay, default=None
            ),
            "tokens_this_cycle": tokens_processed,
            "step": self.total_steps,
            "params": sum(parameter.numel() for parameter in model.parameters()),
            "epochs": self.total_steps
            * cfg.batch_size
            * cfg.block_size
            / max(self.token_count, 1),
        }
        self._obey_soul(self.soul.heartbeat(status))

    def _evaluate_if_due(
        self, local_step, model, cfg, replay_losses, last_heldout
    ):
        if local_step % cfg.eval_every != 0:
            return last_heldout
        model.eval()
        with torch.no_grad():
            if len(self.replay) > 0:
                samples = self.replay.sample(
                    min(cfg.replay_sample_size, len(self.replay))
                )
                if samples:
                    replay_loss = self._evaluate_replay_loss(model, samples)
                    if replay_loss is not None:
                        replay_losses.append(replay_loss)
        model.train()
        if self.holdout_starts:
            return self.evaluate_holdout(phase="periodic")
        return last_heldout

    def _log_if_due(
        self,
        *,
        local_step,
        output,
        loss_val,
        losses,
        replay_batch,
        replay_train_losses,
        batch_kind,
        tokens_processed,
        started_at,
        current,
    ):
        if local_step % 50 != 0 and local_step != 1:
            return current
        tok_s = tokens_processed / max(time.time() - started_at, 0.001)
        self._last_tok_s = tok_s
        avg_loss = sum(losses[-20:]) / max(len(losses[-20:]), 1)
        snapshot = self.metric_channels.snapshot(
            elapsed_sec=time.time() - self._metrics_started_at
        )
        values = {
            name: float(value.detach().cpu()) if value is not None else 0.0
            for name, value in {
                "lm": getattr(output, "lm_loss", None),
                "mtp": getattr(output, "mtp_loss", None),
                "jepa": getattr(output, "jepa_loss", None),
                "jepa_dist": getattr(output, "jepa_dist_loss", None),
                "aux": getattr(output, "aux_loss", None),
                "ghost": getattr(output, "ghost_loss", None),
            }.items()
        }
        perplexity = math.exp(min(values["lm"], 10.0))
        heartbeat = self._heartbeat_label(output)
        replay_info = (
            f" 🎯replay={replay_train_losses[-1]:.2f}"
            if replay_batch is not None and replay_train_losses
            else ""
        )
        print(
            f"  step={self.total_steps:>7d} kind={batch_kind} |"
            f" loss={loss_val:.3f} avg20={avg_loss:.3f} |"
            f" lm={values['lm']:.3f} ppl={perplexity:.1f} |"
            f" mtp={values['mtp']:.3f} jepa={values['jepa']:.3f} "
            f"jd={values['jepa_dist']:.4f} "
            f"aux={values['aux']:.3f} gho={values['ghost']:.3f} |"
            f" fresh={snapshot['fresh']['lm_loss']} "
            f"replay={snapshot['replay']['lm_loss']} |"
            f" tok/s={tok_s:.0f} {heartbeat}{replay_info}",
            flush=True,
        )
        return values["lm"], perplexity

    @staticmethod
    def _heartbeat_label(output) -> str:
        stats = output.heartbeat_stats
        if not stats:
            return ""
        beat = stats.get("beat", "?")
        dopamine = stats.get("dopamine")
        if isinstance(dopamine, (int, float)):
            return f"💓beat={beat} dope={dopamine:.2f}"
        if "batch_skipped" in stats:
            return f"💓beat={beat} batch={stats['batch_skipped']}"
        return f"💓beat={beat}"

    def _finalize_training_report(
        self, report, losses, replay_losses, last_heldout, cur_lm, cur_ppl
    ):
        metrics = self.metric_channels.snapshot(
            elapsed_sec=time.time() - self._metrics_started_at
        )
        if self.holdout_starts:
            if last_heldout is None:
                last_heldout = self.evaluate_holdout(phase="cycle_end")
            else:
                self._append_metric_record(phase="cycle_end", heldout=last_heldout)
        report.update(
            steps=self.metric_channels.successful_updates
            + self.metric_channels.skipped_updates,
            loss_start=losses[0] if losses else 0,
            loss_end=losses[-1] if losses else 0,
            lm_end=cur_lm,
            ppl_end=cur_ppl,
            replay_loss=replay_losses[-1] if replay_losses else 0,
            replay_train_count=self.metric_channels.replay.count,
            fresh_train_count=self.metric_channels.fresh.count,
            replay_fraction=metrics["replay_fraction"],
            metrics=metrics,
            heldout=None if last_heldout is None else asdict(last_heldout),
        )
        if len(replay_losses) >= 2:
            report["forgetting"] = forgetting_proxy(
                replay_losses[0], replay_losses[-1]
            )
        self._last_training_metrics = {
            **metrics,
            "heldout": None if last_heldout is None else asdict(last_heldout),
        }

    def _convene_organ_senate(self) -> None:
        """Convene the OrganSenate at the cycle boundary.

        1. Read reputations from the blockchain ledger.
        2. Redistribute loss weights.
        3. Record the decision as a special block on the chain.
        4. Update ``self.loss_policy`` with the new weights.

        This is a no-op when ``blockchain_enabled`` or ``senate_enabled``
        is False, or when the required modules are not installed.
        """
        if not getattr(self.cfg, 'blockchain_enabled', False):
            return
        if not getattr(self.cfg, 'senate_enabled', False):
            return

        organ_ledger = getattr(self, 'organ_ledger', None)
        organ_senate = getattr(self, 'organ_senate', None)
        organ_reputation_tracker = getattr(self, 'organ_reputation_tracker', None)
        if organ_ledger is None or organ_senate is None or organ_reputation_tracker is None:
            return

        try:
            from f51_darwin.organism.blockchain import Block  # noqa: F811
            from f51_darwin.organism.organ_reputation import OrganReputationTracker  # noqa: F811
            from f51_darwin.organism.organ_senate import SenateLedger  # noqa: F811
        except ImportError:
            return

        if self.total_steps % self.cfg.senate_interval != 0:
            return

        # Compute reputations from the ledger
        reputations = organ_reputation_tracker.compute_from_ledger(
            organ_ledger.path,
            self._get_current_organ_identities(),
            self.cycle,
        )

        # Senate decides
        decision = organ_senate.convene(
            reputations=reputations,
            current_weights=self._get_current_loss_weights(),
            cycle=self.cycle,
            ledger_head_hash=organ_ledger.get_head_hash(),
        )

        # Record the senate block on the chain
        senate_block = SenateLedger.create_senate_block(
            decision=decision,
            prev_block_hash=organ_ledger.get_head_hash(),
            block_number=organ_ledger._next_block_number,
            step_key=self._current_step_key(),
        )
        organ_ledger.append_block(Block.from_dict(senate_block))

        # Apply new loss weights
        self._apply_loss_weights(decision.new_weights)

        print(
            f"  🏛️  Senate cycle {self.cycle}: "
            f"up={decision.promotions} down={decision.demotions} "
            f"dead={decision.deaths}",
            flush=True,
        )

    def _maybe_record_block(self) -> None:
        """Record a block on the blockchain after each optimizer step.

        Creates a block with:
        - Header: block_number, prev_block_hash, merkle_root, step_key, timestamp
        - Transactions: one per phase (PRE_LOSS, PRE_BACKWARD, PRE_OPTIMIZER)
        - Merkle root computed over the transactions

        This is a no-op when ``blockchain_enabled`` is False or when the
        required modules are not installed.
        """
        if not getattr(self.cfg, "blockchain_enabled", False):
            return
        organ_ledger = getattr(self, 'organ_ledger', None)
        if organ_ledger is None:
            # blockchain_enabled is True but bootstrap did not create organ_ledger —
            # this is a configuration bug; surface it once.
            if not getattr(self, '_bc_missing_ledger_warned', False):
                self._bc_missing_ledger_warned = True
                print(
                    "  ⚠️  Blockchain: blockchain_enabled=True mas organ_ledger "
                    "is None — bootstrap não criou o ledger. Verifique imports "
                    "em _finish_bootstrap.",
                    flush=True,
                )
            return

        try:
            from f51_darwin.organism.blockchain import Block, BlockHeader, MerkleTree  # noqa: F811
        except ImportError:
            if not getattr(self, '_bc_import_warned', False):
                self._bc_import_warned = True
                print(
                    "  ⚠️  Blockchain: não foi possível importar Block/BlockHeader/"
                    "MerkleTree de f51_darwin.organism.blockchain.",
                    flush=True,
                )
            return

        transactions = self._build_block_transactions()
        block = Block(
            header=BlockHeader(
                block_number=0,  # OrganLedger.append_block() assigns the real number
                prev_block_hash=organ_ledger.get_head_hash(),
                merkle_root=MerkleTree.build(transactions).root,
                timestamp_unix_ms=int(time.time() * 1000),
                step_key=self._current_step_key(),
                cycle=self.cycle,
                optimizer_step=self.total_steps,
            ),
            transactions=transactions,
        )
        block.block_hash = block.compute_hash()
        organ_ledger.append_block(block)

        # First block written — confirm the path so operators know where to look.
        if not getattr(self, '_bc_first_block_logged', False):
            self._bc_first_block_logged = True
            print(
                f"  🔗 Blockchain: primeiro bloco (#0) escrito em "
                f"{organ_ledger.path}",
                flush=True,
            )

    # ── Senate / Blockchain helpers (stubs that subclasses or organs override) ──

    def _get_current_organ_identities(self) -> dict[str, str]:
        """Return the mapping of organ_name -> identity string currently active.

        Override in subclasses or organ mixins to provide real identities.
        """
        return {}

    def _get_current_loss_weights(self) -> dict[str, float]:
        """Return the current per-organ loss weight mapping.

        Override in subclasses or organ mixins to provide real weights.
        """
        return {}

    def _apply_loss_weights(self, new_weights: dict[str, float]) -> None:
        """Apply new per-organ loss weights to the active training config.

        Override in subclasses or organ mixins to apply weights at runtime.
        """
        pass

    def _current_step_key(self) -> str:
        """Return a unique step key for the current training step."""
        return f"step-{self.total_steps}-cycle-{self.cycle}"

    def _build_block_transactions(self) -> list[dict[str, object]]:
        """Aggregate phase decisions into block transactions.

        Override in subclasses or organ mixins to provide real transaction data.
        """
        return []


def _organ_output_snapshot(output):
    """Keep organ control signals without retaining the full autograd graph."""
    values = {}
    for name in (
        "loss", "lm_loss", "mtp_loss", "jepa_loss", "aux_loss", "ghost_loss",
        "heartbeat_stats", "spider_confidence", "decision_factors",
    ):
        value = getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            value = value.detach()
        values[name] = value
    return SimpleNamespace(**values)


def _causal_block_reason(decision) -> str:
    reasons = ",".join(item.reason for item in decision.rejected)
    return f"{decision.phase.value} blocked" + (
        f":{reasons}" if reasons else ""
    )
