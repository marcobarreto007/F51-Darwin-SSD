from __future__ import annotations

import copy
import math
import uuid
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.layers import ExpertFFN, FineRouter
from f51_darwin.darwin_x_core.neuroendocrine import NeuroendocrineSystem


def _make_moe_backward_hook(moe_layer):
    """Autonomic backward hook — the layer regulates itself from LOCAL signals only.

    Biological principle: Neurons don't know the global loss. They only know:
    - Gradient magnitude arriving (how much upstream "demands" change)
    - Firing frequency (how often they're used)
    - Signal variance (is learning happening or stagnating?)

    This hook updates hormones purely from gradients + local state captured in forward.
    No external loss value needed — the organism is self-contained.
    """
    def hook(module, grad_input, grad_output):
        moe_layer._autonomic_forward_uses = max(
            0, moe_layer._autonomic_forward_uses - 1
        )
        if moe_layer._autonomic_forward_uses > 0:
            return

        # Preserve signed geometry as well as magnitude. A scalar norm cannot
        # reveal direction reversals or define a projection subspace.
        expert_grad_norms, expert_grad_sketches = (
            moe_layer._compute_expert_gradient_signatures()
        )

        # Global stress: magnitude of gradient flowing back from upstream
        if grad_output[0] is not None:
            upstream_stress = grad_output[0].norm().item() / (grad_output[0].numel() ** 0.5)
        else:
            upstream_stress = 0.0

        # Router entropy and usage were captured in forward pass (local state)
        router_entropy = (
            moe_layer._autonomic_entropy_sum /
            max(moe_layer._autonomic_entropy_observations, 1)
        )
        expert_usage = moe_layer._autonomic_usage_accumulator
        if expert_usage is None:
            expert_usage = moe_layer._expert_usage
        expert_usage = expert_usage / (expert_usage.sum() + 1e-8)
        moe_layer._autonomic_usage_accumulator = None
        moe_layer._autonomic_entropy_sum = 0.0
        moe_layer._autonomic_entropy_observations = 0
        expert_routed = expert_usage > 0
        expert_trainable = torch.tensor(
            [any(parameter.requires_grad for parameter in expert.parameters())
             for expert in moe_layer.fine_experts],
            device=expert_grad_norms.device,
            dtype=torch.bool,
        )
        if not bool(expert_routed.any().item()) and expert_grad_norms.sum() == 0:
            return

        if moe_layer.config.dae_enabled:
            # Gradient accumulation microbatches belong to one optimizer
            # event. Aggregate detached evidence now and commit it once at the
            # optimizer boundary.
            moe_layer._accumulate_dae_observation(
                expert_grad_norms=expert_grad_norms,
                expert_usage=expert_usage,
                router_entropy=router_entropy,
                upstream_stress=upstream_stress,
                ghost_loss=moe_layer._ghost_loss if moe_layer._ghost_loss > 0 else None,
                expert_grad_sketches=expert_grad_sketches,
                expert_routed=expert_routed,
                expert_trainable=expert_trainable,
            )
        else:
            # Preserve v7 shadow semantics for existing lineages.
            moe_layer.neuroendocrine.update_from_local_signals(
                expert_grad_norms=expert_grad_norms,
                expert_usage=expert_usage,
                router_entropy=router_entropy,
                upstream_grad_norm=upstream_stress,
                ghost_loss=moe_layer._ghost_loss if moe_layer._ghost_loss > 0 else None,
                expert_grad_sketches=expert_grad_sketches,
                expert_routed=expert_routed,
                expert_trainable=expert_trainable,
            )
            moe_layer._pending_autonomic_actions = (
                moe_layer._propose_autonomic_actions()
            )
    return hook


class _NitroExpertPlacementMixin:
    """Expert placement and Nitro cache lifecycle."""

    def _init_expert_devices(self, default_device: torch.device) -> None:
        """Record the initial GPU device for every expert (called once)."""
        if self._expert_gpu:
            return
        for i in range(len(self.fine_experts)):
            try:
                dev = next(self.fine_experts[i].parameters()).device
            except StopIteration:
                dev = default_device
            self._expert_gpu[i] = dev if dev.type == "cuda" else None

    def _evict_cold_experts(self) -> None:
        """Move the least-used experts to CPU RAM, freezing their grads."""
        hot = set(self.fine_router.get_hot_experts(self.gpu_capacity))
        for i in range(len(self.fine_experts)):
            if i in hot:
                continue
            if self._expert_gpu.get(i) is None:
                continue  # already on CPU
            expert = self.fine_experts[i]
            expert.to("cpu")
            for p in expert.parameters():
                p.requires_grad = False
            self._expert_gpu[i] = None

    def _restore_expert(self, idx: int, target_device: torch.device) -> None:
        """Bring a CPU expert back to its assigned GPU."""
        if idx in self._disabled_experts:
            raise RuntimeError(f"expert {idx} is disabled and cannot be restored")
        expert = self.fine_experts[idx]
        # Preserve dtype: CPU experts lose bf16, must restore it
        target_dtype = next(self.fine_experts[0].parameters()).dtype if len(self.fine_experts) > 0 else torch.bfloat16
        expert.to(device=target_device, dtype=target_dtype)
        for p in expert.parameters():
            p.requires_grad = True
        self._expert_gpu[idx] = target_device

    def _nitro_tick(self) -> None:
        """Periodic Nitro housekeeping (called after forward)."""
        self._nitro_step_counter += 1
        if self._nitro_step_counter % self._nitro_evict_every == 0:
            self._evict_cold_experts()


class _ActiveGradientMixin:
    """DAE evidence, protected-gradient projection and safe-boundary actions."""

    @staticmethod
    def _signed_block_sketch(gradient: torch.Tensor, dimension: int) -> torch.Tensor:
        """Compress one signed gradient tensor without a parameter-sized matrix."""
        flat = gradient.detach().reshape(1, 1, -1)
        elements = flat.numel()
        if elements < dimension:
            flat = F.pad(flat, (0, dimension - elements))
        pooled = F.adaptive_avg_pool1d(flat, dimension).reshape(dimension).float()
        return pooled * math.sqrt(max(elements, 1))

    def _compute_expert_gradient_signatures(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact norms and deterministic signed 64-D sketches per expert."""
        device = self.fine_router.expert_usage_count.device
        dimension = self.neuroendocrine.gradient_sketch_dim
        norms = torch.zeros(len(self.fine_experts), device=device)
        sketches = torch.zeros(
            len(self.fine_experts), dimension, device=device, dtype=torch.float32
        )
        for i, expert in enumerate(self.fine_experts):
            total_squared_norm = torch.zeros((), device=device, dtype=torch.float32)
            gradient_count = 0
            for parameter_index, parameter in enumerate(expert.parameters()):
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                gradient_norm = gradient.norm().float().to(device=device)
                total_squared_norm.add_(gradient_norm.square())
                component = self._signed_block_sketch(gradient, dimension)
                # A fixed rotation prevents identically positioned blocks from
                # different parameter tensors from collapsing onto one another.
                component = torch.roll(
                    component, shifts=(17 * parameter_index) % dimension
                )
                sketches[i].add_(component.to(device=device))
                gradient_count += 1
            if gradient_count:
                norms[i] = total_squared_norm.sqrt()
                sketches[i].div_(math.sqrt(gradient_count))
        return norms, sketches

    def _compute_expert_grad_norms(self) -> torch.Tensor:
        """Compatibility wrapper returning only exact per-expert norms."""
        norms, _ = self._compute_expert_gradient_signatures()
        return norms

    def _propose_autonomic_actions(self) -> dict[str, Any]:
        """Translate hormonal state into an auditable action proposal."""
        plasticity = self.neuroendocrine.plasticity_decision()
        return {
            'plasticity': plasticity,
            'shadow_mode': bool(self.config.dae_shadow_mode),
            'neurogenesis': bool(plasticity['create']),
            'prune_experts': list(plasticity['prune']),
            'expand_experts': list(plasticity['expand']),
            'protect_experts': list(plasticity['protect']),
            'ignore_experts': list(plasticity['ignore']),
            'sleep_phase': self.neuroendocrine.should_sleep(),
            'legacy_hormonal_proposal': {
                'neurogenesis': self.neuroendocrine.should_trigger_neurogenesis(),
                'prune_experts': self.neuroendocrine.should_trigger_pruning(),
                'expand_experts': self.neuroendocrine.should_expand_capacity(),
            },
        }

    @torch.no_grad()
    def _accumulate_dae_observation(
        self,
        *,
        expert_grad_norms: torch.Tensor,
        expert_usage: torch.Tensor,
        router_entropy: float,
        upstream_stress: float,
        ghost_loss: float | None,
        expert_grad_sketches: torch.Tensor,
        expert_routed: torch.Tensor,
        expert_trainable: torch.Tensor,
    ) -> None:
        values = {
            'expert_grad_norms': expert_grad_norms.detach().float(),
            'expert_usage': expert_usage.detach().float(),
            'expert_grad_sketches': expert_grad_sketches.detach().float(),
        }
        accumulator = self._dae_observation_accumulator
        if accumulator is None:
            self._dae_observation_accumulator = {
                **{key: value.clone() for key, value in values.items()},
                'router_entropy': float(router_entropy),
                'upstream_stress': float(upstream_stress),
                'ghost_loss': float(ghost_loss or 0.0),
                'expert_routed': expert_routed.detach().clone(),
                'expert_trainable': expert_trainable.detach().clone(),
                'count': 1,
            }
            return
        for key, value in values.items():
            accumulator[key].add_(value)
        accumulator['router_entropy'] += float(router_entropy)
        accumulator['upstream_stress'] += float(upstream_stress)
        accumulator['ghost_loss'] += float(ghost_loss or 0.0)
        accumulator['expert_routed'].logical_or_(expert_routed)
        accumulator['expert_trainable'].logical_or_(expert_trainable)
        accumulator['count'] += 1

    @torch.no_grad()
    def _commit_dae_observation(self) -> bool:
        accumulator = self._dae_observation_accumulator
        if accumulator is None:
            return False
        self._dae_observation_accumulator = None
        count = max(1, int(accumulator['count']))
        self.neuroendocrine.update_from_local_signals(
            expert_grad_norms=accumulator['expert_grad_norms'] / count,
            expert_usage=accumulator['expert_usage'] / count,
            router_entropy=accumulator['router_entropy'] / count,
            upstream_grad_norm=accumulator['upstream_stress'] / count,
            ghost_loss=(
                accumulator['ghost_loss'] / count
                if accumulator['ghost_loss'] > 0 else None
            ),
            expert_grad_sketches=accumulator['expert_grad_sketches'] / count,
            expert_routed=accumulator['expert_routed'],
            expert_trainable=accumulator['expert_trainable'],
        )
        self._pending_autonomic_actions = self._propose_autonomic_actions()
        return True

    @staticmethod
    @torch.no_grad()
    def _subtract_block_sketch_projection(
        gradient: torch.Tensor,
        projection: torch.Tensor,
    ) -> None:
        """Subtract a 64-D sketch component from contiguous gradient blocks.

        This is deliberately an approximate, bounded inverse of
        ``_signed_block_sketch``.  It changes only block means and therefore
        must not be described as an exact full-dimensional projection.
        """
        flat = gradient.view(-1)
        elements = flat.numel()
        dimension = projection.numel()
        if elements == 0:
            return
        scale = math.sqrt(max(elements, 1))
        for block_idx, indices in enumerate(
            torch.tensor_split(
                torch.arange(elements, device=gradient.device),
                min(dimension, elements),
            )
        ):
            if indices.numel() == 0:
                continue
            correction = projection[block_idx].to(
                device=gradient.device, dtype=gradient.dtype
            ) / scale
            # Advanced indexing returns a copy, so assignment is required for
            # the correction to reach the original gradient storage.
            flat[indices] = flat[indices] - correction

    @torch.no_grad()
    def _project_expert_gradient_from_protected_sketch(
        self,
        expert_idx: int,
    ) -> bool:
        """Remove only conflicting protected sketch components for one expert."""
        rank = int(self.neuroendocrine.protected_rank[expert_idx].item())
        if rank <= 0:
            return False
        basis = self.neuroendocrine.protected_subspace[expert_idx, :rank]
        dimension = self.neuroendocrine.gradient_sketch_dim
        changed = False
        for parameter_index, parameter in enumerate(
            self.fine_experts[expert_idx].parameters()
        ):
            gradient = parameter.grad
            if gradient is None or not bool(torch.isfinite(gradient).all().item()):
                continue
            component = self._signed_block_sketch(gradient, dimension)
            component = torch.roll(
                component, shifts=(17 * parameter_index) % dimension
            ).to(device=basis.device, dtype=torch.float32)
            component_norm = component.norm()
            if component_norm <= 1e-12:
                continue
            unit_component = component / component_norm
            coefficients = basis @ unit_component
            conflicting = coefficients.clamp(max=0.0)
            if not bool((conflicting < 0).any().item()):
                continue
            # Negative coefficients identify opposition to protected memory.
            # Their reconstruction is the component to remove.
            projection = (conflicting @ basis) * component_norm
            projection = torch.roll(
                projection, shifts=-((17 * parameter_index) % dimension)
            )
            self._subtract_block_sketch_projection(gradient, projection)
            changed = True
        return changed

    @torch.no_grad()
    def apply_active_gradient_actions(self) -> dict[str, Any]:
        """Apply shape-preserving DAE actions before clipping/optimizer.step."""
        self._commit_dae_observation()
        actions = self._pending_autonomic_actions or self._propose_autonomic_actions()
        shadow_mode = bool(
            not self.config.dae_enabled or actions.get('shadow_mode', True)
        )
        report: dict[str, Any] = {
            'shadow_mode': shadow_mode,
            'ignored': [],
            'protected': [],
            'updated': [],
            'nonfinite_rejected': 0,
        }
        if shadow_mode:
            return report

        plasticity = actions.get('plasticity', {})
        for expert_idx in plasticity.get('ignore', []):
            if not 0 <= expert_idx < len(self.fine_experts):
                continue
            for parameter in self.fine_experts[expert_idx].parameters():
                if parameter.grad is not None:
                    parameter.grad.zero_()
            report['ignored'].append(expert_idx)

        for expert_idx in plasticity.get('protect', []):
            if not 0 <= expert_idx < len(self.fine_experts):
                continue
            before = sum(
                float(parameter.grad.float().square().sum().item())
                for parameter in self.fine_experts[expert_idx].parameters()
                if parameter.grad is not None
            ) ** 0.5
            changed = self._project_expert_gradient_from_protected_sketch(expert_idx)
            after = sum(
                float(parameter.grad.float().square().sum().item())
                for parameter in self.fine_experts[expert_idx].parameters()
                if parameter.grad is not None
            ) ** 0.5
            if changed:
                report['protected'].append(expert_idx)
                report.setdefault('projection_norm_delta', {})[str(expert_idx)] = (
                    before - after
                )

        for expert_idx in plasticity.get('update', []):
            if not 0 <= expert_idx < len(self.fine_experts):
                continue
            gain = float((
                1.0 + 0.1 * self.neuroendocrine.dopamine[expert_idx]
            ).clamp(
                self.config.dae_update_gain_min,
                self.config.dae_update_gain_max,
            ).item())
            finite = True
            for parameter in self.fine_experts[expert_idx].parameters():
                if parameter.grad is None:
                    continue
                if not bool(torch.isfinite(parameter.grad).all().item()):
                    finite = False
                    report['nonfinite_rejected'] += 1
                    break
            if not finite:
                continue
            for parameter in self.fine_experts[expert_idx].parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(gain)
            report['updated'].append({'expert': expert_idx, 'gain': gain})
        return report

    def apply_pending_autonomic_actions(self) -> dict[str, Any]:
        """Apply only actions whose optimizer/checkpoint invariants are safe.

        Sleep phase is safe (no topology change). Structural actions (neurogenesis,
        pruning, expansion) are recorded for execution at safe boundary.
        """
        actions = self._pending_autonomic_actions or self._propose_autonomic_actions()
        self._pending_autonomic_actions = None
        applied: list[str] = []
        shadow_mode = bool(actions.get('shadow_mode', False))
        structural = {
            'neurogenesis': bool(actions.get('neurogenesis')),
            'prune_experts': list(actions.get('prune_experts', [])),
            'expand_experts': list(actions.get('expand_experts', [])),
        }
        if actions.get('sleep_phase', False) and not self._sleep_active:
            self._enter_sleep_phase()
            applied.append('sleep_phase')
        has_structural = (
            structural['neurogenesis'] or structural['prune_experts'] or
            structural['expand_experts']
        )
        if has_structural:
            self._structural_event_log.append({
                'step': int(self.neuroendocrine.step_count.item()),
                'status': 'shadow_observation' if shadow_mode else 'proposed_not_applied',
                'plasticity': dict(actions.get('plasticity', {})),
                **structural,
            })
            self._structural_event_log[:] = self._structural_event_log[-128:]
        return {
            'applied': applied,
            'structural': structural,
            'shadow_mode': shadow_mode,
            'plasticity': dict(actions.get('plasticity', {})),
        }

    def execute_structural_actions(self) -> dict[str, Any]:
        """Execute topology-changing actions at a SAFE boundary.

        MUST be called between cycles, after optimizer.step() and before
        the next forward(). This is the only place where expert birth,
        death, and expansion are safe — the autograd graph is gone and
        the optimizer can be rebuilt.

        Returns a report of what was executed.
        """
        executed: dict[str, Any] = {'neurogenesis': 0, 'pruned': 0, 'expanded': 0}
        if not self._structural_event_log:
            return executed

        # Process the most recent proposal
        latest = self._structural_event_log[-1]
        if latest.get('status') != 'proposed_not_applied':
            return executed

        rejection_reason = self.structural_preflight_error(latest)
        if rejection_reason is not None:
            latest['status'] = 'rejected_not_applied'
            latest['rejection_reason'] = rejection_reason
            return executed

        requested_prunes = [int(index) for index in latest.get('prune_experts', [])]
        requested_expands = [int(index) for index in latest.get('expand_experts', [])]

        # ── Prune: remove dead experts (highest indices first to preserve validity) ──
        for expert_idx in sorted(requested_prunes, reverse=True):
            self._prune_expert(expert_idx)
            if not self._apoptosis(expert_idx):
                raise RuntimeError(
                    f"structural prune was preflighted but refused: {expert_idx}"
                )
            executed['pruned'] += 1

        # ── Expand: grow high-BDNF experts ──
        for expert_idx in requested_expands:
            self._expand_expert(expert_idx)
            executed['expanded'] += 1

        # ── Neurogenesis: create new expert ──
        if latest.get('neurogenesis'):
            new_idx = self._create_expert()
            if new_idx is not None:
                executed['neurogenesis'] = 1

        # Mark as executed
        latest['status'] = 'executed'
        latest['executed_at_step'] = int(self.neuroendocrine.step_count.item())
        return executed

    def structural_preflight_error(
        self,
        proposal: dict[str, Any] | None = None,
    ) -> str | None:
        """Validate the exact pending topology proposal without mutation."""

        latest = proposal
        if latest is None:
            latest = next(
                (
                    event
                    for event in reversed(self._structural_event_log)
                    if isinstance(event, dict)
                    and event.get('status') == 'proposed_not_applied'
                ),
                None,
            )
        if latest is None:
            return 'missing_pending_proposal'
        requested_prunes = [
            int(index) for index in latest.get('prune_experts', [])
        ]
        requested_expands = [
            int(index) for index in latest.get('expand_experts', [])
        ]
        minimum_experts = max(1, int(self.config.fine_experts) // 2)
        if len(set(requested_prunes)) != len(requested_prunes):
            return 'duplicate_prune'
        if any(
            index < 0 or index >= len(self.fine_experts)
            for index in requested_prunes
        ):
            return 'invalid_prune_index'
        if len(self.fine_experts) - len(requested_prunes) < minimum_experts:
            return 'minimum_expert_floor'
        if requested_prunes and requested_expands:
            return 'mixed_prune_expand'
        if len(set(requested_expands)) != len(requested_expands):
            return 'duplicate_expand'
        if any(
            index < 0 or index >= len(self.fine_experts)
            for index in requested_expands
        ):
            return 'invalid_expand_index'
        if latest.get('neurogenesis') and len(self.fine_experts) >= (
            int(self.config.fine_experts) + 8
        ):
            return 'neurogenesis_capacity'
        return None

    def _apply_autonomic_actions(self) -> dict[str, Any]:
        """Safe compatibility wrapper; call after ``optimizer.step``."""
        return self.apply_pending_autonomic_actions()


class _ExpertLifecycleMixin:
    """Shape-changing expert operations executed only at safe boundaries."""

    def _prune_expert(self, expert_idx: int) -> None:
        """Kill an expert: gate to zero, offload to CPU. Marks for apoptosis."""
        if expert_idx >= len(self.fine_experts):
            return
        if self._expert_gpu.get(expert_idx) is not None:
            self.fine_experts[expert_idx].to('cpu')
            for p in self.fine_experts[expert_idx].parameters():
                p.requires_grad = False
            self._expert_gpu[expert_idx] = None
            # Push router bias down so this expert stops receiving tokens
            self.fine_router.expert_bias.data[expert_idx] -= 3.0
        self._disabled_experts.add(expert_idx)
        # Mark for apoptosis: will be physically removed during next sleep phase
        self._marked_for_apoptosis = getattr(self, '_marked_for_apoptosis', set())
        self._marked_for_apoptosis.add(expert_idx)

    def _apoptosis(self, expert_idx: int) -> bool:
        """Physically remove a dead expert and compact all arrays.

        Biological: microglia engulf dead neurons, recycling the space.
        This is REAL death — the expert ceases to exist as an object.

        Returns True if removal succeeded.
        """
        if expert_idx >= len(self.fine_experts):
            return False
        if len(self.fine_experts) <= self.config.fine_experts // 2:
            return False  # Keep minimum experts alive

        # ── 1. Remove expert from module list ──
        old_expert_gpu = dict(self._expert_gpu)
        dead_expert = self.fine_experts[expert_idx]
        # Move to CPU first to free GPU memory
        dead_expert.to('cpu')
        del self.fine_experts[expert_idx]
        dead_id = self._expert_ids.pop(expert_idx)
        self._expert_parent_ids.pop(expert_idx)
        self._expert_birth_steps.pop(expert_idx)

        # ── 2. Compact neuroendocrine buffers ──
        new_count = len(self.fine_experts)
        ne = self.neuroendocrine
        device = ne.dopamine.device
        compact_names = (
            'dopamine', 'bdnf', 'expert_grad_norm_ema', 'expert_usage_ema',
            'baseline_dopamine', 'grad_magnitude', 'grad_direction_ema',
            'previous_grad_sketch', 'grad_direction_stability', 'grad_sign_flips',
            'grad_stagnation_steps', 'grad_observation_count', 'protected_subspace',
            'protected_strength', 'protected_rank', 'grad_conflict',
            'grad_orthogonal_residual',
        )
        keep = torch.tensor(
            [index for index in range(new_count + 1) if index != expert_idx],
            device=device,
            dtype=torch.long,
        )
        for buf_name in compact_names:
            old = getattr(ne, buf_name)
            new_tensor = old.detach().index_select(0, keep).clone()
            if isinstance(old, nn.Parameter):
                setattr(ne, buf_name, nn.Parameter(new_tensor))
            else:
                ne.register_buffer(buf_name, new_tensor)
        ne.num_experts = new_count

        # ── 3. Compact router ──
        d_model = self.fine_router.router.in_features
        old_weight = self.fine_router.router.weight.data
        new_router = nn.Linear(d_model, new_count, bias=False).to(
            device=device, dtype=old_weight.dtype
        )
        with torch.no_grad():
            src_idx = 0
            for i in range(new_count + 1):
                if i == expert_idx:
                    continue
                if src_idx < new_count:
                    new_router.weight[src_idx] = old_weight[i]
                    src_idx += 1
        self.fine_router.router = new_router
        self.fine_router.num_experts = new_count

        # ── 4. Compact expert_usage_buffer (on self, not router) ──
        old_usage_buf = self._expert_usage_buffer
        new_usage_buf = torch.zeros(new_count, device=device, dtype=old_usage_buf.dtype)
        src_idx = 0
        for i in range(new_count + 1):
            if i == expert_idx:
                continue
            if src_idx < new_count:
                new_usage_buf[src_idx] = old_usage_buf[i]
                src_idx += 1
        self.register_buffer('_expert_usage_buffer', new_usage_buf)

        # ── 5. Compact expert_bias (buffer since 2026-07-26) and others ──
        for buf_name in ['expert_usage_count', 'expert_usage_ema', 'expert_miss_count', 'external_bias']:
            old_buf = getattr(self.fine_router, buf_name)
            new_buf = torch.zeros(new_count, device=device, dtype=old_buf.dtype)
            src_idx = 0
            for i in range(new_count + 1):
                if i == expert_idx:
                    continue
                if src_idx < new_count:
                    new_buf[src_idx] = old_buf[i]
                    src_idx += 1
            setattr(self.fine_router, buf_name, new_buf)

        old_bias = self.fine_router.expert_bias.data
        new_bias = torch.zeros(new_count, device=device)
        src_idx = 0
        for i in range(new_count + 1):
            if i == expert_idx:
                continue
            if src_idx < new_count:
                new_bias[src_idx] = old_bias[i]
                src_idx += 1
        # register_buffer (nao nn.Parameter) -- expert_bias e um buffer
        # atualizado por regra deterministica, nao por gradiente, desde
        # 2026-07-26 (aux-loss-free balancing, ver FineRouter.__init__).
        # Reatribuir via nn.Parameter aqui re-registraria como parametro
        # treinavel na proxima vez que o otimizador for (re)construido.
        self.fine_router.register_buffer('expert_bias', new_bias)

        # ── 5. Clean up GPU tracking ──
        self._expert_gpu = {
            (idx if idx < expert_idx else idx - 1): assigned
            for idx, assigned in old_expert_gpu.items()
            if idx != expert_idx
        }
        self._disabled_experts = {
            idx if idx < expert_idx else idx - 1
            for idx in self._disabled_experts
            if idx != expert_idx
        }
        self._marked_for_apoptosis = {
            idx if idx < expert_idx else idx - 1
            for idx in self._marked_for_apoptosis
            if idx != expert_idx
        }
        self.fine_router.top_k = min(self.fine_router.top_k, new_count)
        self._structural_event_log.append({
            'step': int(ne.step_count.item()),
            'status': 'apoptosis',
            'expert_uuid': dead_id,
        })
        self._structural_event_log[:] = self._structural_event_log[-128:]

        return True

    def _expand_expert(self, expert_idx: int) -> None:
        """Grow expert capacity GRADUALLY using gradient-based growth.

        Biological principle: Dendritic arborization — neurons grow new dendritic
        branches gradually, preserving existing synapses while adding capacity.

        Previous implementation destroyed the expert and recreated it.
        This implementation grows in-place, preserving all existing connections.
        """
        old_expert = self.fine_experts[expert_idx]
        old_hidden = old_expert.gate_proj.out_features
        growth_factor = 0.5  # Grow by 50%, not 2x (too aggressive)
        new_hidden = int(old_hidden * (1 + growth_factor))
        added_neurons = new_hidden - old_hidden
        d_model = old_expert.gate_proj.in_features

        # Create new expert with grown capacity
        new_expert = ExpertFFN(d_model, new_hidden, self.config.dropout)
        target_device = self._expert_gpu.get(expert_idx)
        if target_device is not None:
            new_expert.to(target_device)

        with torch.no_grad():
            # Copy ALL old weights (preservation)
            new_expert.gate_proj.weight[:old_hidden] = old_expert.gate_proj.weight
            new_expert.up_proj.weight[:old_hidden] = old_expert.up_proj.weight
            new_expert.down_proj.weight[:, :old_hidden] = old_expert.down_proj.weight

            # Initialize NEW neurons (growth) with Xavier/Kaiming
            # New connections start weak and strengthen through learning
            nn.init.xavier_uniform_(new_expert.gate_proj.weight[old_hidden:])
            nn.init.xavier_uniform_(new_expert.up_proj.weight[old_hidden:])
            nn.init.xavier_uniform_(new_expert.down_proj.weight[:, old_hidden:])

            # CRITICAL: Scale new connections to start WEAK (LTD baseline)
            # They must earn their strength through gradient-driven LTP
            new_expert.gate_proj.weight[old_hidden:] *= 0.1
            new_expert.up_proj.weight[old_hidden:] *= 0.1
            new_expert.down_proj.weight[:, old_hidden:] *= 0.1

        # ── IN-PLACE GROWTH: resize weights without replacing the module ──
        expert = self.fine_experts[expert_idx]
        # Determine target dtype from existing params
        target_dtype = expert.gate_proj.weight.dtype
        target_device = expert.gate_proj.weight.device
        with torch.no_grad():
            for old_layer, new_layer, name in [
                (expert.gate_proj, new_expert.gate_proj, 'gate'),
                (expert.up_proj, new_expert.up_proj, 'up'),
                (expert.down_proj, new_expert.down_proj, 'down'),
            ]:
                old_layer.out_features = new_hidden if name != 'down' else old_layer.out_features
                old_layer.in_features = old_layer.in_features
                if name == 'down':
                    old_layer.in_features = new_hidden
                # Ensure dtype matches (bf16 model, float32 weights = crash)
                new_weight = new_layer.weight.data.to(dtype=target_dtype, device=target_device)
                old_layer.weight = nn.Parameter(new_weight)
                if old_layer.bias is not None:
                    old_layer.bias = nn.Parameter(new_layer.bias.data.to(dtype=target_dtype, device=target_device))
        del new_expert
        if target_device is not None:
            self._expert_gpu[expert_idx] = (
                target_device if target_device.type == "cuda" else None
            )

    def _create_expert(self) -> int | None:
        """Create a new expert from scratch if there's capacity.
        Returns the new expert index, or None if no room."""
        current = len(self.fine_experts)
        max_experts = getattr(self.config, 'fine_experts', current)
        if current >= max_experts + 8:  # allow overshoot of 8
            return None

        best_idx = int(self.neuroendocrine.bdnf.argmax())
        parent_id = self._expert_ids[best_idx]
        # Create a child with the parent's current anatomy, including growth.
        d_model = self.config.d_model
        hidden = self.fine_experts[best_idx].gate_proj.out_features
        new_expert = ExpertFFN(d_model, hidden, self.config.dropout)

        # Initialize from best existing expert's weights + noise
        with torch.no_grad():
            for new_p, old_p in zip(new_expert.parameters(), self.fine_experts[best_idx].parameters()):
                new_p.copy_(old_p + torch.randn_like(old_p) * 0.1)

        self.fine_experts.append(new_expert)
        self._expert_ids.append(uuid.uuid4().hex)
        self._expert_parent_ids.append(parent_id)
        self._expert_birth_steps.append(int(self.neuroendocrine.step_count.item()))

        # ── CRÍTICO: Expand neuroendocrine system FIRST ──
        # This prevents IndexError when hormonal gates try to access new expert
        self.neuroendocrine.expand_capacity(new_size=current + 1, from_expert_idx=best_idx)

        # ── Expand ALL router components for new expert ──
        device = self.fine_router.expert_bias.device

        # Expand router weight matrix
        old_router_weight = self.fine_router.router.weight.data.to(device)
        new_router = nn.Linear(d_model, current + 1, bias=False).to(device)
        with torch.no_grad():
            new_router.weight[:current] = old_router_weight
            new_router.weight[current:] = old_router_weight[best_idx:best_idx+1] * 0.5
        self.fine_router.router = new_router
        self.fine_router.num_experts = current + 1

        # Expand buffers and parameters
        for buf_name in ['expert_usage_count', 'expert_usage_ema', 'expert_miss_count', 'external_bias']:
            old_buf = getattr(self.fine_router, buf_name)
            new_buf = torch.zeros(current + 1, device=device, dtype=old_buf.dtype)
            new_buf[:current] = old_buf.to(device)
            setattr(self.fine_router, buf_name, new_buf)

        # expert_bias e buffer (nao Parameter) desde 2026-07-26 -- ver
        # comentario em FineRouter.__init__ e no outro compact-site acima.
        old_bias = self.fine_router.expert_bias.data.to(device)
        new_bias = torch.zeros(current + 1, device=device)
        new_bias[:current] = old_bias
        self.fine_router.register_buffer('expert_bias', new_bias)

        # Expand _expert_usage_buffer (used for tracking)
        old_usage_buf = self._expert_usage_buffer
        new_usage_buf = torch.zeros(current + 1, device=device, dtype=old_usage_buf.dtype)
        new_usage_buf[:current] = old_usage_buf.to(device)
        self.register_buffer('_expert_usage_buffer', new_usage_buf)

        # Place on correct device with correct dtype (bf16 like the rest of the model)
        target_device = self._expert_gpu.get(0)
        if target_device is not None:
            target_dtype = self.fine_experts[0].gate_proj.weight.dtype if len(self.fine_experts) > 0 else torch.bfloat16
            new_expert.to(device=target_device, dtype=target_dtype)
            self.fine_router.router.to(target_device)
            self._expert_gpu[current] = target_device

        # Reset neurogenesis signal
        self.neuroendocrine.norepinephrine.zero_()
        self._structural_event_log.append({
            'step': int(self.neuroendocrine.step_count.item()),
            'status': 'birth',
            'expert_uuid': self._expert_ids[current],
            'parent_uuid': parent_id,
        })
        self._structural_event_log[:] = self._structural_event_log[-128:]
        return current

    def _add_to_replay_buffer(self, routing_indices: torch.Tensor) -> None:
        """Store routing traces for later consolidation diagnostics.

        Biological principle: During waking, hippocampus encodes experiences.
        During sleep, these are replayed to neocortex for long-term consolidation.
        """
        if len(self._replay_buffer) < self._replay_capacity:
            self._replay_buffer.append(routing_indices.detach().cpu())
        else:
            idx = int(self.fine_router.total_steps.item()) % self._replay_capacity
            self._replay_buffer[idx] = routing_indices.detach().cpu()

    def _enter_sleep_phase(self) -> None:
        """Enter a shape-preserving consolidation window.

        Routing traces alone are not training examples, so this phase does not
        claim gradient replay and does not destructively prune/fuse weights.
        """
        self.neuroendocrine.trigger_sleep_phase()
        self._sleep_active = True
        self._sleep_steps_remaining = self._sleep_phase_duration
        self._replay_count += len(self._replay_buffer)

    def _enter_sleep_phase_unsafe(self) -> None:
        """ACTUAL sleep phase: synaptic pruning, replay, consolidation.

        Biological phases (simplified):
        1. NREM: Replay of hippocampal memories to neocortex
        2. REM: Synaptic pruning and consolidation
        3. Wake: Reset acetylcholine, refreshed
        """
        self.neuroendocrine.trigger_sleep_phase()
        self._sleep_active = True
        self._sleep_steps_remaining = self._sleep_phase_duration

        # === SYNAPTIC PRUNING (NREM phase) ===
        # Experts with consistently low dopamine AND low BDNF get pruned
        with torch.no_grad():
            dopamine = self.neuroendocrine.dopamine
            bdnf = self.neuroendocrine.bdnf
            # Weak synapses: low DA AND low BDNF (not rewarded, not growing)
            weak_mask = (dopamine < -0.5) & (bdnf < 0.5)
            for expert_idx in weak_mask.nonzero().squeeze(-1).tolist():
                if expert_idx < len(self.fine_experts):
                    self._prune_synapses(expert_idx, pruning_rate=0.2)

        # === EXPERT FUSION (REM phase) ===
        self._fuse_redundant_experts()

        # === APOPTOSIS (cleanup phase) ===
        # Physically remove experts marked for death.
        # Sorted descending so removal indices stay valid.
        for expert_idx in sorted(self._marked_for_apoptosis, reverse=True):
            self._apoptosis(expert_idx)
        self._marked_for_apoptosis.clear()

    def _prune_synapses(self, expert_idx: int, pruning_rate: float = 0.2) -> None:
        """Prune weak synapses in an expert (LTD: long-term depression).

        Biological principle: Synapses that don't fire together, don't wire together.
        Weakest 20% of weights are set to zero, creating sparsity.
        """
        if expert_idx >= len(self.fine_experts):
            return

        expert = self.fine_experts[expert_idx]
        with torch.no_grad():
            for name, param in expert.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    # Find weakest synapses (lowest absolute magnitude)
                    flat = param.abs().view(-1)
                    num_prune = int(flat.numel() * pruning_rate)
                    if num_prune > 0:
                        weakest = flat.topk(num_prune, largest=False).indices
                        keep = torch.ones(flat.numel(), dtype=torch.bool, device=param.device)
                        keep[weakest] = False
                        param.mul_(keep.view_as(param))

    def _capture_activation_signature(self, x: torch.Tensor) -> None:
        """Store representative inputs for activation signature computation.

        Builds a buffer of recent inputs to compare expert outputs. The method
        remains experimental and is not enabled for destructive fusion.
        """
        if len(self._signature_inputs) >= self._signature_capacity:
            idx = int(self.fine_router.total_steps.item()) % self._signature_capacity
            self._signature_inputs[idx] = x.detach().cpu()
        else:
            self._signature_inputs.append(x.detach().cpu())

    def _fuse_redundant_experts(self) -> list[tuple[int, int]]:
        """Return no fusion until functional similarity is measured.

        Two scalar BDNF values cannot establish that experts implement the same
        function. Destructive fusion is disabled until held-out activation or
        output similarity provides a causal, shape-compatible criterion.
        """
        return []

    def _fuse_redundant_experts_unsafe(self) -> None:
        """Detect and merge functionally redundant experts via Activation Signatures.

        Replaces the BDNF proxy with real behavioral evidence:
        if two experts produce near-identical outputs for the same inputs,
        they are functionally redundant — regardless of their weights.

        Cost: O(E * K * d_model) with K=100 inputs — negligible vs O(E^2 * d^2).
        """
        if len(self.fine_experts) < 2 or len(self._signature_inputs) < 10:
            return

        n_experts = len(self.fine_experts)
        signatures = torch.zeros(n_experts, self.config.d_model)

        with torch.no_grad():
            for i, expert in enumerate(self.fine_experts):
                dev = next(expert.parameters()).device
                sig_sum = torch.zeros(self.config.d_model)
                count = 0
                for x_stored in self._signature_inputs:
                    out = expert(x_stored.to(dev)).mean(dim=(0, 1))
                    sig_sum += out.cpu()
                    count += 1
                if count > 0:
                    signatures[i] = sig_sum / count

        # Pairwise cosine similarity of activation signatures
        norms = F.normalize(signatures, dim=-1)
        sim = norms @ norms.T  # [E, E]

        merged = set()
        for i in range(n_experts):
            if i in merged:
                continue
            for j in range(i + 1, n_experts):
                if j in merged:
                    continue
                if sim[i, j] > 0.95:  # near-identical outputs
                    self._merge_experts(source_idx=j, target_idx=i)
                    merged.add(j)

        if merged:
            self._signature_inputs.clear()

    def _merge_experts(self, source_idx: int, target_idx: int) -> None:
        """Merge source expert into target expert (cortical remapping).

        After fusion, source expert is marked as dead and will be pruned.
        """
        if source_idx >= len(self.fine_experts) or target_idx >= len(self.fine_experts):
            return

        source = self.fine_experts[source_idx]
        target = self.fine_experts[target_idx]

        with torch.no_grad():
            # Average the weights (simple fusion)
            for s_param, t_param in zip(source.parameters(), target.parameters()):
                t_param.data = 0.5 * (t_param.data + s_param.data)

            # Reduce router bias for source (mark as redundant)
            self.fine_router.expert_bias.data[source_idx] -= 5.0


class _MoEForwardMixin:
    """Token routing, expert execution and auxiliary statistics."""

    def _pre_forward_actions(self) -> None:
        """Apply hormonal state that affects THIS forward pass.

        The backward hook only observes signals and proposes lifecycle actions.
        Here we handle state that changes HOW tokens are processed right now:
        - Sleep: activate replay behavior for the next few forwards
        - Stress: cortisol already baked into expert_gate()
        """
        # ── Sleep mode: replay old examples for consolidation ──
        if self._sleep_active:
            if self._sleep_steps_remaining > 0:
                self._sleep_steps_remaining -= 1
                # During sleep, we would replay from buffer (requires integration with training loop)
                # For now, sleep mode = reduced exploration, higher consolidation
            else:
                self._sleep_active = False
                # Wake up: refresh hormonal state
                self.neuroendocrine.trigger_sleep_phase()  # Resets ACh

    def _apply_vertical_bias(self) -> torch.Tensor | None:
        """Apply routing bias from the previous layer before routing.

        If layer N routed math tokens to expert #3, layer N+1 receives a
        small bias toward its own expert #3. This creates representational
        continuity across layers — no more isolated islands.
        """
        bias = self._vertical_bias
        self._vertical_bias = None
        if bias is None or bias.numel() != self.fine_router.expert_bias.numel():
            return None
        return (bias - bias.mean()) * self.config.vertical_routing_scale

    def _emit_vertical_bias(self) -> torch.Tensor:
        """Produce routing bias for the NEXT layer based on our routing decisions.

        Returns a bias tensor shaped [num_experts] that encodes which experts
        were active. The next layer will receive this as _vertical_bias.
        """
        usage = self._expert_usage_buffer
        total = usage.sum()
        if total == 0:
            return torch.zeros_like(usage)
        # Normalize: experts that got more tokens emit stronger bias
        return (usage / total).detach().clone()

    def _dispatch_fine_experts(
        self,
        flat_x: torch.Tensor,
        flat_weights: torch.Tensor,
        flat_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Roteia tokens para os fine experts com UM sync GPU->CPU por camada.

        O dispatch anterior iterava ``flat_indices.unique()`` em Python e, para
        cada expert, fazia ``int(t.item())`` e ``torch.where(mask)``. Cada uma
        dessas duas operacoes bloqueia ate a GPU terminar, entao o custo era
        2 syncs x experts_ativos x camadas. Medido no 100M (32 experts,
        12 camadas): 381 chamadas de ``nonzero`` por forward, 274 ms de CPU
        contra 20.6 ms de CUDA.

        Aqui os tokens sao ordenados por expert uma unica vez com ``argsort``
        estavel; as fronteiras de cada expert saem de um unico ``.tolist()``.
        A ordenacao estavel preserva a ordem (linha, slot) que ``torch.where``
        produzia, de modo que cada expert recebe as mesmas linhas na mesma
        ordem e a saida e bit a bit identica — ver
        ``src/tests/test_moe_dispatch_equivalence.py``.
        """
        output = torch.zeros_like(flat_x)
        num_tokens, top_k = flat_indices.shape
        if num_tokens == 0:
            return output

        # Achata (token, slot) -> uma atribuicao por linha.
        assignments = flat_indices.reshape(-1)
        token_rows = torch.arange(
            num_tokens, device=flat_indices.device
        ).repeat_interleave(top_k)
        assignment_weights = flat_weights.reshape(-1)

        # Ordenacao ESTAVEL: dentro de um expert, mantem a ordem crescente de
        # (linha, slot), identica a varredura row-major de torch.where.
        order = torch.argsort(assignments, stable=True)
        sorted_experts = assignments[order]
        sorted_rows = token_rows[order]
        sorted_weights = assignment_weights[order]

        # Unico ponto de sincronizacao da camada: fronteiras dos grupos.
        unique_experts, counts = torch.unique_consecutive(
            sorted_experts, return_counts=True
        )
        expert_ids = unique_experts.tolist()
        group_sizes = counts.tolist()

        start = 0
        for expert_idx, size in zip(expert_ids, group_sizes):
            stop = start + size
            rows = sorted_rows[start:stop]

            # ── Nitro: lazy-restore cold expert from CPU ──
            if self.nitro_enabled and self._expert_gpu.get(expert_idx) is None:
                self._restore_expert(expert_idx, flat_x.device)

            expert = self.fine_experts[expert_idx]
            # ── NEUROENDOCRINE: pass system to expert for gating ──
            expert_out = expert(flat_x[rows], self.neuroendocrine, expert_idx)
            output = output.index_add(
                0, rows, sorted_weights[start:stop].unsqueeze(-1) * expert_out
            )
            start = stop

        return output

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        # ── PRE-FORWARD: Aplicar decisões hormonais acumuladas ──
        self._pre_forward_actions()

        # ── VERTICAL: aplicar viés de roteamento da camada anterior ──
        vertical_bias = self._apply_vertical_bias()

        batch, seq_len, d_model = x.shape
        weights, indices, logits = self.fine_router(x, routing_bias=vertical_bias)
        flat_x = x.reshape(-1, d_model)
        flat_weights = weights.reshape(-1, self.config.experts_per_token)
        flat_indices = indices.reshape(-1, self.config.experts_per_token)

        # ── Lazy GPU device init (once) ──
        if not self._expert_gpu:
            self._init_expert_devices(x.device)

        # ── Group tokens by expert (batched dispatch) ──
        output = self._dispatch_fine_experts(flat_x, flat_weights, flat_indices)

        if self.shared_experts:
            # Soma direta (nao media) -- DeepSeekMoE trata shared experts com
            # peso de combine fixo em 1 cada; a media com 2 shared experts
            # cortava a contribuicao do branch compartilhado pela metade do
            # esperado (achado 2026-07-26).
            shared = torch.zeros_like(flat_x)
            for expert in self.shared_experts:
                shared = shared + expert(flat_x)
            output = output + shared

        output = output.reshape(batch, seq_len, d_model)
        aux = self._auxiliary_stats(logits, indices)
        num_experts = len(self.fine_experts)
        current_usage = F.one_hot(indices.reshape(-1), num_experts).float().mean(dim=0)
        self._expert_usage_buffer.copy_(current_usage.detach().to(self._expert_usage_buffer.device))

        # ── Nitro: periodic eviction ──
        if self.nitro_enabled and self.training:
            self._nitro_tick()

        # ── Collect metrics for hormonal update (will be used in backward) ──
        if self.training:
            probs = F.softmax(logits, dim=-1)
            self._router_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().detach()
            # Use ACTUAL expert count (may grow via neurogenesis)
            self._expert_usage = current_usage.detach()
            self._autonomic_forward_uses += 1
            if (self._autonomic_usage_accumulator is None or
                    self._autonomic_usage_accumulator.shape != current_usage.shape):
                self._autonomic_usage_accumulator = torch.zeros_like(current_usage)
            self._autonomic_usage_accumulator.add_(current_usage.detach())
            self._autonomic_entropy_sum += self._router_entropy
            self._autonomic_entropy_observations += 1

            # ── Store experience for replay + activation signatures ──
            step = int(self.fine_router.total_steps.item())
            if step % 10 == 0:
                self._add_to_replay_buffer(indices)
            if step % 5 == 0:
                self._capture_activation_signature(x.detach())

        return output, aux

    def _auxiliary_stats(self, logits: torch.Tensor, indices: torch.Tensor) -> dict[str, Any]:
        num_experts = len(self.fine_experts)  # Use ACTUAL count
        probs = F.softmax(logits, dim=-1)
        usage = F.one_hot(indices.reshape(-1), num_experts).float().mean(dim=0)
        mean_prob = probs.reshape(-1, num_experts).mean(dim=0)
        load_balance_loss = num_experts * torch.sum(mean_prob * usage)
        router_z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        # Base aux_loss — scaling is applied in DarwinXModel.forward()
        aux_loss = 0.01 * load_balance_loss + 1e-3 * router_z_loss
        return {
            "aux_loss": aux_loss,
            "load_balance_loss": load_balance_loss.detach(),
            "router_z_loss": router_z_loss.detach(),
            "tokens_per_expert": usage.detach().cpu().tolist(),
            "shared_experts": self.config.shared_experts,
            "dead_experts": [i for i, u in enumerate(usage.detach().cpu().tolist()) if u < 0.001],
        }


class DeepSeekStyleMoE(
    _NitroExpertPlacementMixin,
    _ActiveGradientMixin,
    _ExpertLifecycleMixin,
    _MoEForwardMixin,
    nn.Module,
):
    """DeepSeek-style MoE assembled from behavior-preserving capability mixins."""

    _RUNTIME_STATE_FIELDS = (
        "_router_entropy",
        "_expert_usage",
        "_autonomic_forward_uses",
        "_autonomic_usage_accumulator",
        "_autonomic_entropy_sum",
        "_autonomic_entropy_observations",
        "_dae_observation_accumulator",
        "_sleep_active",
        "_sleep_steps_remaining",
        "_replay_buffer",
        "_replay_count",
        "_pending_autonomic_actions",
        "_structural_event_log",
        "_vertical_bias",
        "_marked_for_apoptosis",
        "_signature_inputs",
        "_signature_buffer",
        "_disabled_experts",
        "_ghost_predator_buffer",
        "_ghost_loss",
        "_nitro_step_counter",
    )

    def runtime_state_dict(self) -> dict[str, Any]:
        """Snapshot Python-side state mutated by forward and lifecycle hooks."""

        return {
            "schema": "darwin-moe-runtime-v1",
            "values": {
                name: copy.deepcopy(getattr(self, name))
                for name in self._RUNTIME_STATE_FIELDS
                if hasattr(self, name)
            },
            "expert_devices": [
                next(expert.parameters()).device
                for expert in self.fine_experts
            ],
            "expert_requires_grad": [
                [parameter.requires_grad for parameter in expert.parameters()]
                for expert in self.fine_experts
            ],
            "expert_gpu": copy.deepcopy(self._expert_gpu),
        }

    def load_runtime_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the exact Python-side runtime snapshot."""

        if not isinstance(state, dict) or state.get("schema") != (
            "darwin-moe-runtime-v1"
        ):
            raise ValueError("invalid Darwin MoE runtime state")
        devices = list(state.get("expert_devices", ()))
        requires_grad = list(state.get("expert_requires_grad", ()))
        if len(devices) != len(self.fine_experts) or len(requires_grad) != len(
            self.fine_experts
        ):
            raise ValueError("Darwin MoE runtime expert anatomy mismatch")
        for expert, device, flags in zip(
            self.fine_experts, devices, requires_grad
        ):
            parameters = list(expert.parameters())
            if len(flags) != len(parameters):
                raise ValueError("Darwin MoE runtime gradient flags mismatch")
            expert.to(device=device)
            for parameter, enabled in zip(expert.parameters(), flags):
                parameter.requires_grad = bool(enabled)
        for name, value in dict(state.get("values", {})).items():
            if name not in self._RUNTIME_STATE_FIELDS:
                raise ValueError(f"unexpected Darwin MoE runtime field: {name}")
            setattr(self, name, copy.deepcopy(value))
        self._expert_gpu = copy.deepcopy(state.get("expert_gpu", {}))

    def __init__(self, config: DarwinXConfig) -> None:
        super().__init__()
        self.config = config
        self.fine_router = FineRouter(config.d_model, config.fine_experts, config.experts_per_token)
        self.fine_experts = nn.ModuleList(
            [
                ExpertFFN(config.d_model, config.fine_expert_hidden_dim, config.dropout)
                for _ in range(config.fine_experts)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                ExpertFFN(config.d_model, config.shared_expert_hidden_dim, config.dropout)
                for _ in range(config.shared_experts)
            ]
        )

        # ── Neuroendocrine System ──
        self.neuroendocrine = NeuroendocrineSystem(
            num_experts=config.fine_experts,
            dopamine_tau=200.0,
            norepinephrine_tau=150.0,
            cortisol_tau=300.0,
            bdnf_tau=400.0,
            ach_tau=500.0,
        )

        # ── Nitro CPU offloading ──
        self.nitro_enabled: bool = bool(config.nitro_enabled)
        self.gpu_capacity: int = getattr(config, "nitro_gpu_expert_capacity", 4)
        self._expert_gpu: dict[int, torch.device | None] = {}
        self._expert_ids: list[str] = [uuid.uuid4().hex for _ in self.fine_experts]
        self._expert_parent_ids: list[str | None] = [None for _ in self.fine_experts]
        self._expert_birth_steps: list[int] = [0 for _ in self.fine_experts]
        self._nitro_step_counter: int = 0
        self._nitro_evict_every: int = 100

        # ── Local state captured during forward for hormonal update ──
        self._router_entropy: float = 0.0
        self._expert_usage = torch.zeros(self.config.fine_experts)
        self.register_buffer('_expert_usage_buffer', torch.zeros(self.config.fine_experts))
        self._autonomic_forward_uses: int = 0
        self._autonomic_usage_accumulator: torch.Tensor | None = None
        self._autonomic_entropy_sum: float = 0.0
        self._autonomic_entropy_observations: int = 0
        self._dae_observation_accumulator: dict[str, Any] | None = None

        # ── Autonomic regulation: backward hook updates hormones from LOCAL signals ──
        self.register_full_backward_hook(_make_moe_backward_hook(self))

        # ── Sleep phase infrastructure ──
        self._sleep_active: bool = False
        self._sleep_steps_remaining: int = 0
        self._sleep_phase_duration: int = 5  # Number of forward steps in sleep mode

        # ── Replay buffer for consolidation (hippocampal replay during sleep) ──
        self._replay_capacity: int = 128
        self._replay_buffer: list[torch.Tensor] = []
        self._replay_count: int = 0

        # ── Structural action pipeline: propose in backward, execute at safe boundary ──
        self._pending_autonomic_actions: dict[str, Any] | None = None
        self._structural_event_log: list[dict[str, Any]] = []

        # ── Vertical connections: routing influence from previous layer ──
        self._vertical_bias: torch.Tensor | None = None

        # ── Apoptosis tracking: experts marked for physical removal ──
        self._marked_for_apoptosis: set[int] = set()

        # ── Activation Signature Buffer for expert fusion detection ──
        # Stores recent expert outputs to compute functional similarity.
        # Experimental output-signature buffer. This is an internal hypothesis,
        # not a literature-backed fusion criterion yet.
        self._signature_capacity: int = 100
        self._signature_inputs: list[torch.Tensor] = []  # recent inputs
        self._signature_buffer: list[torch.Tensor] = []  # flattened expert outputs
        self._disabled_experts: set[int] = set()

        # ── Ghost Predator: test old domain knowledge ──
        self._ghost_predator_buffer: list[torch.Tensor] = []
        self._ghost_predator_size: int = 32
        self._ghost_loss: float = 0.0  # last ghost test loss
        self._pending_autonomic_actions: dict[str, Any] | None = None
        self._structural_event_log: list[dict[str, Any]] = []
