from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class _NeuroSignalMixin:
    """Local-signal hormone and gradient-geometry updates."""

    @torch.no_grad()
    def _novelty_from_entropy(self, router_entropy: float) -> float:
        """Return bounded positive entropy surprise, not absolute entropy."""
        entropy = max(0.0, float(router_entropy))
        if not bool(self.router_entropy_initialized.item()):
            self.router_entropy_ema.fill_(entropy)
            self.router_entropy_initialized.fill_(True)
            return 0.0
        baseline = float(self.router_entropy_ema.item())
        alpha = 0.02
        self.router_entropy_ema.mul_(1.0 - alpha).add_(alpha * entropy)
        max_entropy = max(math.log(max(self.num_experts, 2)), 1e-6)
        return min(1.0, max(0.0, entropy - baseline) / max_entropy)

    @torch.no_grad()
    def _update_loss_history(self, loss: float) -> float:
        """Update sliding window and return loss trend."""
        ptr = int(self.loss_ptr.item())
        self.loss_history[ptr] = loss
        self.loss_ptr.fill_((ptr + 1) % 100)
        if self.step_count < 100:
            return 0.0
        next_ptr = int(self.loss_ptr.item())
        ordered = torch.cat((self.loss_history[next_ptr:], self.loss_history[:next_ptr]))
        older = ordered[:50]
        recent = ordered[50:]
        return (recent.mean() - older.mean()).clamp(-1.0, 1.0).item()

    @torch.no_grad()
    def update(
        self,
        expert_grad_norms: torch.Tensor,
        expert_usage: torch.Tensor,
        router_entropy: float,
        loss: float,
        global_grad_norm: float,
    ) -> dict[str, Any]:
        """Update hormonal state based on current step metrics.

        This is called once per training step (not per token).
        """
        self.step_count += 1

        # Update EMAs
        alpha = 0.02
        self.expert_grad_norm_ema = (1 - alpha) * self.expert_grad_norm_ema + alpha * expert_grad_norms
        self.expert_usage_ema = (1 - alpha) * self.expert_usage_ema + alpha * expert_usage

        # === DOPAMINE: Reward prediction error ===
        # High gradient norm = expert is learning = reward
        grad_reward = (expert_grad_norms - self.expert_grad_norm_ema).clamp(-1, 1)
        dopamine_decay = math.exp(-1.0 / self.tau['dopamine'])
        self.dopamine = self.dopamine * dopamine_decay + 0.1 * grad_reward

        # === NOREPINEPHRINE: Novelty detection ===
        # Router entropy spike = unexpected routing = novelty
        novelty_signal = self._novelty_from_entropy(router_entropy)
        ne_decay = math.exp(-1.0 / self.tau['norepinephrine'])
        self.norepinephrine.mul_(ne_decay).add_((1.0 - ne_decay) * novelty_signal)

        # === CORTISOL: Stress response ===
        loss_trend = self._update_loss_history(loss)
        grad_stress = (global_grad_norm - 5.0) / 10.0  # normalize around 5.0
        # Positive trend means recent loss is worse than the older window.
        stress_signal = max(0, loss_trend) + max(0, grad_stress)
        cort_decay = math.exp(-1.0 / self.tau['cortisol'])
        self.cortisol = self.cortisol * cort_decay + 0.03 * stress_signal

        # === BDNF: Activity-dependent growth ===
        # Consistent high usage + high grad = growth factor
        usage_score = (self.expert_usage_ema / self.expert_usage_ema.mean()).clamp(0, 3)
        grad_score = (self.expert_grad_norm_ema / 0.1).clamp(0, 3)
        bdnf_growth = usage_score * grad_score
        bdnf_decay = math.exp(-1.0 / self.tau['bdnf'])
        self.bdnf = self.bdnf * bdnf_decay + 0.01 * bdnf_growth

        # === ACETYLCHOLINE: Sleep pressure ===
        # Accumulates with training, decays during "sleep" (reset externally)
        ach_decay = math.exp(-1.0 / self.tau['ach'])
        self.ach_pressure.mul_(ach_decay).add_(1.0 - ach_decay)

        return self.state()

    @torch.no_grad()
    def _measure_gradient_geometry(
        self,
        unit_sketches: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Measure conflict and unexplained residual against historical bases."""
        self.grad_conflict.masked_fill_(valid, 0.0)
        self.grad_orthogonal_residual.masked_fill_(valid, 1.0)
        for expert_idx in valid.nonzero(as_tuple=False).flatten().tolist():
            rank = int(self.protected_rank[expert_idx].item())
            if rank == 0:
                continue
            basis = self.protected_subspace[expert_idx, :rank]
            coefficients = basis @ unit_sketches[expert_idx]
            projection = coefficients @ basis
            residual = unit_sketches[expert_idx] - projection
            self.grad_orthogonal_residual[expert_idx] = residual.norm().clamp(0.0, 1.0)
            self.grad_conflict[expert_idx] = (-coefficients.min()).clamp(0.0, 1.0)

    @torch.no_grad()
    def _update_protected_memory(
        self,
        unit_sketches: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Update a signed basis from retention-associated gradient evidence."""
        for expert_idx in valid.nonzero(as_tuple=False).flatten().tolist():
            vector = unit_sketches[expert_idx]
            rank = int(self.protected_rank[expert_idx].item())
            basis = self.protected_subspace[expert_idx]
            strength = self.protected_strength[expert_idx]
            strength.mul_(0.995)

            if rank:
                coefficients = basis[:rank] @ vector
                closest = int(coefficients.abs().argmax().item())
                if coefficients[closest].abs() >= 0.8:
                    aligned = vector * coefficients[closest].sign()
                    candidate = 0.95 * basis[closest] + 0.05 * aligned
                    other = torch.cat(
                        (basis[:closest], basis[closest + 1:rank]), dim=0
                    )
                    if other.numel():
                        candidate = candidate - (other @ candidate) @ other
                    basis[closest].copy_(F.normalize(candidate, dim=0))
                    strength[closest].add_(1.0)
                    continue
                residual = vector - (coefficients @ basis[:rank])
            else:
                residual = vector

            residual_norm = residual.norm()
            if residual_norm <= 1e-6:
                continue
            residual = residual / residual_norm
            if rank < self.protected_basis_capacity:
                basis[rank].copy_(residual)
                strength[rank] = 1.0
                self.protected_rank[expert_idx] = rank + 1
            else:
                replace = int(strength.argmin().item())
                other = torch.cat((basis[:replace], basis[replace + 1:]), dim=0)
                if other.numel():
                    residual = residual - (other @ residual) @ other
                    residual = F.normalize(residual, dim=0)
                basis[replace].copy_(residual)
                strength[replace] = 1.0

    @torch.no_grad()
    def update_from_local_signals(
        self,
        expert_grad_norms: torch.Tensor,
        expert_usage: torch.Tensor,
        router_entropy: float,
        upstream_grad_norm: float,
        ghost_loss: float | None = None,
        expert_grad_sketches: torch.Tensor | None = None,
        expert_routed: torch.Tensor | None = None,
        expert_trainable: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Update hormonal state from LOCAL signals + Ghost Predator.

        Ghost Predator: if the model retains OLD knowledge → dopamine boost.
        If it forgets → cortisol spike. This makes the organism WISE, not AGGRESSIVE.
        """
        self.step_count += 1

        # Ensure inputs are on the same device as buffers
        dev = self.dopamine.device
        expert_grad_norms = expert_grad_norms.to(device=dev, dtype=torch.float32)
        expert_usage = expert_usage.to(device=dev, dtype=torch.float32)
        if expert_routed is None:
            expert_routed = expert_usage > 0
        else:
            expert_routed = expert_routed.to(device=dev, dtype=torch.bool)
        if expert_trainable is None:
            expert_trainable = torch.ones_like(expert_routed)
        else:
            expert_trainable = expert_trainable.to(device=dev, dtype=torch.bool)
        if expert_grad_sketches is not None:
            expert_grad_sketches = expert_grad_sketches.to(device=dev, dtype=torch.float32)
            expected_shape = (self.num_experts, self.gradient_sketch_dim)
            if tuple(expert_grad_sketches.shape) != expected_shape:
                raise ValueError(
                    "expert_grad_sketches must have shape "
                    f"{expected_shape}, got {tuple(expert_grad_sketches.shape)}"
                )

        alpha = 0.02
        previous_norm_ema = self.expert_grad_norm_ema.clone().clamp_min(1e-8)
        updated_norm_ema = (1 - alpha) * self.expert_grad_norm_ema + alpha * expert_grad_norms
        self.expert_grad_norm_ema.copy_(torch.where(
            expert_routed, updated_norm_ema, self.expert_grad_norm_ema
        ))
        self.expert_usage_ema.mul_(1 - alpha).add_(alpha * expert_usage)

        # === DOPAMINE: Reward prediction error (local + ghost) ===
        grad_reward = (expert_grad_norms - self.expert_grad_norm_ema).clamp(-1, 1)

        # Ghost Predator: reward retention, punish forgetting
        ghost_bonus = 0.0
        ghost_punishment = 0.0
        ghost_retained = False
        if ghost_loss is not None and ghost_loss > 0:
            previous_ghost = self.ghost_loss_ema.item()
            ghost_retained = ghost_loss <= previous_ghost
            self.ghost_loss_ema.mul_(0.99).add_(0.01 * ghost_loss)
            ghost_delta = (previous_ghost - ghost_loss) / max(previous_ghost, 0.1)
            ghost_bonus = max(0.0, ghost_delta) * 0.3
            ghost_punishment = max(0.0, -ghost_delta) * 0.3

        dopamine_decay = math.exp(-1.0 / self.tau['dopamine'])
        self.dopamine = self.dopamine * dopamine_decay + 0.1 * (grad_reward + ghost_bonus)

        # === NOREPINEPHRINE: Novelty detection ===
        novelty_signal = self._novelty_from_entropy(router_entropy)
        ne_decay = math.exp(-1.0 / self.tau['norepinephrine'])
        self.norepinephrine.mul_(ne_decay).add_((1.0 - ne_decay) * novelty_signal)

        # === CORTISOL: Stress (local + ghost punishment) ===
        grad_stress = (upstream_grad_norm - 0.5)
        grad_instability = expert_grad_norms.std(unbiased=False) / (expert_grad_norms.mean() + 1e-8)
        stress_signal = max(0, grad_stress) + min(1.0, grad_instability) + ghost_punishment
        cort_decay = math.exp(-1.0 / self.tau['cortisol'])
        self.cortisol = self.cortisol * cort_decay + 0.03 * stress_signal

        # === BDNF: Activity-dependent growth ===
        usage_score = (self.expert_usage_ema / self.expert_usage_ema.mean()).clamp(0, 3)
        grad_score = (self.expert_grad_norm_ema / 0.1).clamp(0, 3)
        bdnf_growth = usage_score * grad_score
        bdnf_decay = math.exp(-1.0 / self.tau['bdnf'])
        self.bdnf = self.bdnf * bdnf_decay + 0.01 * bdnf_growth

        # === ACETYLCHOLINE: Sleep pressure ===
        ach_decay = math.exp(-1.0 / self.tau['ach'])
        self.ach_pressure.mul_(ach_decay).add_(1.0 - ach_decay)

        # ═══════════════════════════════════════════════════════════
        # GRADIENT MUTATIONAL STATE — extract all signals from gradient
        # ═══════════════════════════════════════════════════════════
        # Relative magnitude is comparable across experts and model scales.
        relative_magnitude = (expert_grad_norms / previous_norm_ema).clamp(0.0, 10.0)
        next_magnitude = 0.9 * self.grad_magnitude + 0.1 * relative_magnitude
        self.grad_magnitude.copy_(torch.where(
            expert_routed, next_magnitude, self.grad_magnitude
        ))

        # Absence of routing is not evidence of a dead expert.
        significant = expert_grad_norms > (0.1 * previous_norm_ema).clamp_min(1e-12)
        eligible = expert_routed & expert_trainable
        next_stagnation = torch.where(
            significant,
            torch.zeros_like(self.grad_stagnation_steps),
            self.grad_stagnation_steps + 1,
        )
        self.grad_stagnation_steps.copy_(torch.where(
            eligible, next_stagnation, self.grad_stagnation_steps
        ))

        if expert_grad_sketches is not None:
            sketch_norms = expert_grad_sketches.norm(dim=1, keepdim=True)
            valid = eligible & (sketch_norms.squeeze(1) > 1e-12)
            unit_sketches = expert_grad_sketches / sketch_norms.clamp_min(1e-12)
            observed_before = self.grad_observation_count > 0
            comparable = valid & observed_before

            direction_cosine = F.cosine_similarity(
                unit_sketches, self.grad_direction_ema, dim=1, eps=1e-8
            )
            reversal = 0.5 * (
                1.0 - F.cosine_similarity(
                    unit_sketches, self.previous_grad_sketch, dim=1, eps=1e-8
                )
            )
            self.grad_direction_stability.copy_(torch.where(
                comparable,
                0.95 * self.grad_direction_stability + 0.05 * direction_cosine,
                self.grad_direction_stability,
            ))
            self.grad_sign_flips.copy_(torch.where(
                comparable,
                0.9 * self.grad_sign_flips + 0.1 * reversal,
                self.grad_sign_flips,
            ))

            blended_direction = F.normalize(
                0.95 * self.grad_direction_ema + 0.05 * unit_sketches, dim=1
            )
            self.grad_direction_ema.copy_(torch.where(
                valid.unsqueeze(1),
                torch.where(
                    observed_before.unsqueeze(1), blended_direction, unit_sketches
                ),
                self.grad_direction_ema,
            ))
            self.previous_grad_sketch.copy_(torch.where(
                valid.unsqueeze(1), unit_sketches, self.previous_grad_sketch
            ))
            self.grad_observation_count.add_(valid.to(dtype=torch.long))

            self._measure_gradient_geometry(unit_sketches, valid)
            if ghost_retained and int(self.step_count.item()) % 10 == 0:
                self._update_protected_memory(unit_sketches, valid)

        return self.state()


class _NeuroPlasticityMixin:
    """Plasticity policy and observable endocrine state."""

    def state(self) -> dict[str, Any]:
        return {
            'dopamine_per_expert': self.dopamine.tolist(),
            'norepinephrine': self.norepinephrine.item(),
            'cortisol': self.cortisol.item(),
            'bdnf_per_expert': self.bdnf.tolist(),
            'ach_pressure': self.ach_pressure.item(),
            'grad_magnitude_mean': self.grad_magnitude.mean().item(),
            'grad_stagnation_max': int(self.grad_stagnation_steps.max().item()),
            'grad_sign_flips_mean': self.grad_sign_flips.mean().item(),
            'grad_direction_stability_mean': self.grad_direction_stability.mean().item(),
            'grad_conflict_mean': self.grad_conflict.mean().item(),
            'grad_orthogonal_residual_mean': self.grad_orthogonal_residual.mean().item(),
            'grad_observations_min': int(self.grad_observation_count.min().item()),
            'protected_rank_max': int(self.protected_rank.max().item()),
        }

    def plasticity_decision(self) -> dict[str, Any]:
        """Route expert plasticity from signed gradient geometry.

        Decisions are evidence only while the MoE is in shadow mode. They do
        not alter token routing or model topology from inside backward.
        """
        decisions: dict[str, Any] = {
            'ignore': [], 'update': [], 'protect': [], 'expand': [],
            'prune': [], 'create': False,
        }
        minimum_evidence = 20
        for expert_idx in range(self.num_experts):
            magnitude = self.grad_magnitude[expert_idx].item()
            stagnation = int(self.grad_stagnation_steps[expert_idx].item())
            reversal = self.grad_sign_flips[expert_idx].item()
            stability = self.grad_direction_stability[expert_idx].item()
            conflict = self.grad_conflict[expert_idx].item()
            residual = self.grad_orthogonal_residual[expert_idx].item()
            observations = int(self.grad_observation_count[expert_idx].item())

            if observations >= minimum_evidence and reversal > 0.6 and magnitude < 0.5:
                decisions['ignore'].append(expert_idx)
                continue
            if observations >= minimum_evidence and stagnation > 500 and magnitude < 0.1:
                decisions['prune'].append(expert_idx)
                continue
            if observations >= 2 and conflict > 0.25:
                decisions['protect'].append(expert_idx)
                continue
            if (observations >= minimum_evidence and magnitude > 1.5 and
                    stability > 0.75 and residual > 0.4 and
                    self.bdnf[expert_idx].item() > 0.5):
                decisions['expand'].append(expert_idx)
                continue
            decisions['update'].append(expert_idx)

        observed = self.grad_observation_count >= minimum_evidence
        if (bool(observed.any().item()) and
                self.norepinephrine.item() > 0.05 and
                self.grad_magnitude[observed].mean().item() > 1.2 and
                self.grad_direction_stability[observed].mean().item() > 0.7 and
                self.grad_orthogonal_residual[observed].mean().item() > 0.6):
            decisions['create'] = True
        return decisions

    def expert_gate(self, expert_idx: int) -> torch.Tensor:
        """Get the gating factor for a specific expert.

        Returns a neutral-at-baseline multiplicative factor in [0.5, 1.5].
        Combines dopamine (learned baseline + rapid reward) and cortisol (suppression).
        """
        base_dop = self.baseline_dopamine[expert_idx]
        rapid_dop = self.dopamine[expert_idx]
        total_dop = (base_dop + rapid_dop).clamp(-4, 4)

        # Cortisol suppresses all experts under stress
        cort_suppression = (self.cortisol * 0.5).clamp(0, 2)

        # Plasticity decisions intentionally do not alter token routing. Doing
        # so would manufacture the low-gradient evidence used for pruning.

        # Exponential modulation keeps
        # an untouched model neutral while allowing bounded up/down regulation.
        log_gate = (total_dop - cort_suppression).clamp(
            min=math.log(0.5), max=math.log(1.5)
        )
        gate = torch.exp(log_gate)
        return gate

    def should_trigger_neurogenesis(self) -> bool:
        """High norepinephrine indicates novelty → create new expert."""
        return self.norepinephrine.item() > 0.05  # 10x mais sensivel

    def should_trigger_pruning(self) -> list[int]:
        """High cortisol + low dopamine → prune weak experts."""
        if self.cortisol.item() < 0.05:  # 10x mais sensivel
            return []
        threshold = self.expert_usage_ema.mean() * 0.1
        weak = (self.expert_usage_ema < threshold).nonzero().squeeze(-1).tolist()
        return weak

    def should_expand_capacity(self) -> list[int]:
        """High BDNF → expert should grow (more hidden_dim)."""
        if self.bdnf.max() < 0.5:  # 10x mais sensivel
            return []
        high_bdnf = (self.bdnf > 0.3).nonzero().squeeze(-1).tolist()  # 10x mais sensivel
        return high_bdnf

    def should_sleep(self) -> bool:
        """High homeostatic sleep pressure triggers consolidation.

        ``ach_pressure`` remains as a checkpoint-compatible legacy name. The
        accumulated quantity is adenosine-like pressure, not a claim that ACh
        itself monotonically accumulates during wakefulness.
        """
        return self.ach_pressure.item() > 0.1  # 10x mais sensivel (100 steps)

    def trigger_sleep_phase(self) -> None:
        """Reset acetylcholine after sleep phase."""
        self.ach_pressure.zero_()


class NeuroendocrineSystem(_NeuroSignalMixin, _NeuroPlasticityMixin, nn.Module):
    """Per-layer endocrine controller with stable registered state."""

    _FP32_MUTATIONAL_BUFFERS = (
        "grad_magnitude",
        "grad_direction_ema",
        "previous_grad_sketch",
        "grad_direction_stability",
        "grad_sign_flips",
        "protected_subspace",
        "protected_strength",
        "grad_conflict",
        "grad_orthogonal_residual",
    )

    def __init__(
        self,
        num_experts: int,
        dopamine_tau: float = 200.0,
        norepinephrine_tau: float = 150.0,
        cortisol_tau: float = 300.0,
        bdnf_tau: float = 400.0,
        ach_tau: float = 500.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.gradient_sketch_dim = 64
        self.protected_basis_capacity = 4
        self.tau = {
            'dopamine': dopamine_tau,
            'norepinephrine': norepinephrine_tau,
            'cortisol': cortisol_tau,
            'bdnf': bdnf_tau,
            'ach': ach_tau,
        }

        # Baseline dopamine is learned, but an untrained organism starts neutral.
        # The router has its own small noise to break expert symmetry.
        self.baseline_dopamine = nn.Parameter(torch.zeros(num_experts))

        # Hormonal state (not learned, updated via forward dynamics)
        self.register_buffer('dopamine', torch.zeros(num_experts))
        self.register_buffer('norepinephrine', torch.tensor(0.0))
        self.register_buffer('cortisol', torch.tensor(0.0))
        self.register_buffer('bdnf', torch.zeros(num_experts))
        self.register_buffer('ach_pressure', torch.tensor(0.0))

        # Memory for temporal dynamics
        self.register_buffer('loss_history', torch.zeros(100))
        self.register_buffer('loss_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('step_count', torch.zeros(1, dtype=torch.long))

        # ── Ghost Predator: tests old knowledge, rewards retention, punishes forgetting ──
        self.register_buffer('ghost_loss_ema', torch.tensor(10.0))

        # ═══════════════════════════════════════════════════════════
        # GRADIENT MUTATIONAL STATE — The gradient is the model's only
        # real signal from the world. Every architectural decision
        # (birth, growth, protection, death) derives from it.
        # ═══════════════════════════════════════════════════════════
        # Versioned so v7 checkpoints created before this geometry existed can
        # initialize it explicitly without weakening strict anatomy loading.
        self.register_buffer('mutational_state_version', torch.tensor(2, dtype=torch.long))
        # Relative magnitude: current norm divided by that expert's own EMA.
        self.register_buffer('grad_magnitude', torch.zeros(num_experts))
        # Signed, per-expert directional state in a compact deterministic space.
        self.register_buffer(
            'grad_direction_ema', torch.zeros(num_experts, self.gradient_sketch_dim)
        )
        self.register_buffer(
            'previous_grad_sketch', torch.zeros(num_experts, self.gradient_sketch_dim)
        )
        self.register_buffer('grad_direction_stability', torch.zeros(num_experts))
        # EMA of direction reversal: 0=same direction, 1=opposite direction.
        self.register_buffer('grad_sign_flips', torch.zeros(num_experts))
        # Counts only routed, trainable observations with insignificant gradient.
        self.register_buffer(
            'grad_stagnation_steps', torch.zeros(num_experts, dtype=torch.long)
        )
        self.register_buffer(
            'grad_observation_count', torch.zeros(num_experts, dtype=torch.long)
        )
        # Rank-limited basis of retention-associated historical directions.
        self.register_buffer(
            'protected_subspace',
            torch.zeros(
                num_experts, self.protected_basis_capacity, self.gradient_sketch_dim
            ),
        )
        self.register_buffer(
            'protected_strength',
            torch.zeros(num_experts, self.protected_basis_capacity),
        )
        self.register_buffer('protected_rank', torch.zeros(num_experts, dtype=torch.long))
        self.register_buffer('grad_conflict', torch.zeros(num_experts))
        self.register_buffer('grad_orthogonal_residual', torch.zeros(num_experts))
        self.register_buffer('ghost_memory_size', torch.zeros(1))

        # Expert-specific tracking
        self.register_buffer('expert_grad_norm_ema', torch.ones(num_experts) * 0.1)
        self.register_buffer('expert_usage_ema', torch.ones(num_experts) / num_experts)
        self.register_buffer('router_entropy_ema', torch.tensor(0.0))
        self.register_buffer('router_entropy_initialized', torch.tensor(False))

    def _apply(self, fn, recurse: bool = True):
        """Move control state with the module without reducing its precision.

        ``Module.to(dtype=torch.bfloat16)`` normally casts floating-point
        buffers together with model weights. Signed gradient geometry is a
        control-plane signal, not a model activation: its matrix products must
        remain FP32 while the experts themselves run in BF16.
        """
        super()._apply(fn, recurse=recurse)
        for name in self._FP32_MUTATIONAL_BUFFERS:
            buffer = self._buffers.get(name)
            if buffer is not None and buffer.dtype != torch.float32:
                self._buffers[name] = buffer.float()
        return self

    @torch.no_grad()
    def expand_capacity(self, new_size: int, from_expert_idx: int = 0) -> None:
        """Expand all hormonal buffers to accommodate new experts (neurogenesis).

        Biological principle: When neurogenesis occurs, the hypothalamic-pituitary
        axis extends its regulatory reach to new neurons. Hormonal baseline for
        new neurons is inherited from the "parent" neuron that spawned them.

        Args:
            new_size: Target number of experts
            from_expert_idx: Parent expert whose hormonal state is inherited
        """
        if new_size <= self.num_experts:
            return

        delta = new_size - self.num_experts
        device = self.dopamine.device

        # Expand learned parameters (with gradient flow)
        new_baseline = nn.Parameter(torch.zeros(new_size, device=device))
        new_baseline.data[:self.num_experts] = self.baseline_dopamine.data
        # Inherit baseline from parent, add small noise for differentiation
        parent_baseline = self.baseline_dopamine.data[from_expert_idx]
        new_baseline.data[self.num_experts:] = parent_baseline + torch.randn(delta, device=device) * 0.05
        del self.baseline_dopamine  # Remove old parameter
        self.register_parameter('baseline_dopamine', new_baseline)

        # Expand buffers (save old, create new, assign)
        def expand_buffer(old_buf, default_value=0.0):
            new_buf = torch.zeros(new_size, device=device, dtype=old_buf.dtype)
            new_buf[:self.num_experts] = old_buf
            if default_value != 0.0:
                new_buf[self.num_experts:] = default_value
            return new_buf

        old_dopamine = self.dopamine.clone()
        old_bdnf = self.bdnf.clone()
        old_grad_ema = self.expert_grad_norm_ema.clone()
        old_usage_ema = self.expert_usage_ema.clone()

        self.register_buffer('dopamine', expand_buffer(old_dopamine, 0.0))
        self.register_buffer('bdnf', expand_buffer(old_bdnf, 0.0))
        self.register_buffer('expert_grad_norm_ema', expand_buffer(old_grad_ema, 0.1))
        self.register_buffer('expert_usage_ema', expand_buffer(old_usage_ema, 1.0 / new_size))

        # ── Expand gradient mutational state buffers ──
        one_dimensional = (
            'grad_magnitude', 'grad_direction_stability', 'grad_sign_flips',
            'grad_stagnation_steps', 'grad_observation_count', 'protected_rank',
            'grad_conflict', 'grad_orthogonal_residual',
        )
        for name in one_dimensional:
            old = getattr(self, name)
            new = torch.zeros(new_size, device=device, dtype=old.dtype)
            new[:self.num_experts] = old
            self.register_buffer(name, new)

        matrix_names = (
            'grad_direction_ema', 'previous_grad_sketch',
            'protected_subspace', 'protected_strength',
        )
        for name in matrix_names:
            old = getattr(self, name)
            new = torch.zeros(
                (new_size, *old.shape[1:]), device=device, dtype=old.dtype
            )
            new[:self.num_experts] = old
            self.register_buffer(name, new)

        # Update size
        self.num_experts = new_size
