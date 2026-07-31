from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from f51_darwin.gaba_inhibition import GABAergicLayer
from f51_darwin.inter_hemispheric import InterHemisphericSystem
from f51_darwin.jepa_v2 import JEPAHeadV2

from .organs import HeartbeatOrgan


@dataclass
class OrganSlotOutput:
    hidden: torch.Tensor
    auxiliary_loss: torch.Tensor | None
    observation: dict[str, Any] | None


class OrganAdapter(nn.Module):
    def __init__(
        self,
        receptor_dim: int = 768,
        organ_dim: int = 512,
    ) -> None:
        super().__init__()
        self.receptor_dim = int(receptor_dim)
        self.organ_dim = int(organ_dim)
        self.input_norm = nn.LayerNorm(self.receptor_dim)
        self.to_organ = nn.Linear(self.receptor_dim, self.organ_dim)
        self.from_organ = nn.Linear(self.organ_dim, self.receptor_dim)

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.receptor_dim:
            raise ValueError(
                "recipient hidden must have shape "
                f"[batch, sequence, {self.receptor_dim}]"
            )
        return self.to_organ(self.input_norm(hidden))

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.organ_dim:
            raise ValueError(
                "organ hidden must have shape "
                f"[batch, sequence, {self.organ_dim}]"
            )
        return self.from_organ(hidden)


class GABAResidualSlot(nn.Module):
    def __init__(
        self,
        organ: GABAergicLayer,
        *,
        receptor_dim: int = 768,
        organ_dim: int = 512,
        max_residual_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        if organ.config.d_model != organ_dim:
            raise ValueError("GABA organ dimension does not match adapter")
        if max_residual_ratio < 0.0:
            raise ValueError("max_residual_ratio must be non-negative")
        self.organ = organ
        self.adapter = OrganAdapter(receptor_dim, organ_dim)
        self.external_gate = nn.Parameter(torch.zeros(()))
        self.max_residual_ratio = float(max_residual_ratio)
        for parameter in self.organ.parameters():
            parameter.requires_grad_(False)

    def _bounded(self, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        residual_norm = residual.float().norm(dim=-1, keepdim=True)
        maximum = (
            hidden.float().norm(dim=-1, keepdim=True)
            * self.max_residual_ratio
        )
        scale = torch.clamp(
            maximum
            / residual_norm.clamp_min(torch.finfo(torch.float32).eps),
            max=1.0,
        )
        return residual * scale.to(dtype=residual.dtype)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        enabled: bool,
        mutate_state: bool = False,
    ) -> OrganSlotOutput:
        canonical = self.adapter.encode(hidden)
        delta, observation = self.organ(
            canonical,
            mutate_state=mutate_state,
        )
        projected = self._bounded(hidden, self.adapter.decode(delta))
        if enabled:
            combined = hidden + torch.tanh(self.external_gate) * projected
        else:
            combined = hidden
        return OrganSlotOutput(
            hidden=combined,
            auxiliary_loss=None,
            observation=observation,
        )


class IHSResidualSlot(nn.Module):
    """Pre-block inter-hemispheric transplant with an exact zero gate."""

    def __init__(
        self,
        organ: InterHemisphericSystem,
        *,
        receptor_dim: int = 768,
        organ_dim: int = 512,
        max_residual_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        if int(organ.config.d_model) != organ_dim:
            raise ValueError("IHS organ dimension does not match adapter")
        if max_residual_ratio < 0.0:
            raise ValueError("max_residual_ratio must be non-negative")
        self.organ = organ
        self.adapter = OrganAdapter(receptor_dim, organ_dim)
        self.external_gate = nn.Parameter(torch.zeros(()))
        self.max_residual_ratio = float(max_residual_ratio)
        for parameter in self.organ.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        enabled: bool,
    ) -> OrganSlotOutput:
        canonical = self.adapter.encode(hidden)
        transformed = self.organ(canonical)
        projected = self.adapter.decode(transformed)
        residual_norm = projected.float().norm(dim=-1, keepdim=True)
        maximum = (
            hidden.float().norm(dim=-1, keepdim=True)
            * self.max_residual_ratio
        )
        scale = torch.clamp(
            maximum
            / residual_norm.clamp_min(torch.finfo(torch.float32).eps),
            max=1.0,
        )
        bounded = projected * scale.to(dtype=projected.dtype)
        gate = (
            torch.tanh(self.external_gate)
            if enabled
            else torch.zeros_like(self.external_gate)
        )
        combined = hidden + gate * bounded if enabled else hidden
        if not torch.isfinite(combined).all():
            raise FloatingPointError("non-finite transplanted IHS output")
        return OrganSlotOutput(
            hidden=combined,
            auxiliary_loss=None,
            observation={
                "gate": float(gate.detach()),
                "residual_ratio": float(
                    (
                        bounded.float().norm(dim=-1)
                        / hidden.float().norm(dim=-1).clamp_min(
                            torch.finfo(torch.float32).eps
                        )
                    ).mean().detach()
                ),
            },
        )


class HeartbeatSlot(nn.Module):
    """Stateful Heartbeat sidecar; shadow observes without mutating donor state."""

    def __init__(
        self,
        organ: HeartbeatOrgan,
        *,
        receptor_dim: int = 768,
        organ_dim: int = 512,
    ) -> None:
        super().__init__()
        if organ.d_model != organ_dim:
            raise ValueError("Heartbeat organ dimension does not match adapter")
        self.organ = organ
        self.adapter = OrganAdapter(receptor_dim, organ_dim)
        for parameter in self.organ.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        enabled: bool,
        jepa_error: float = 0.0,
        loss: float = 0.0,
    ) -> OrganSlotOutput:
        canonical = self.adapter.encode(hidden)
        if enabled:
            observation = self.organ.beat(
                canonical,
                jepa_error=float(jepa_error),
                loss=float(loss),
                memory_used=False,
            )
        else:
            state = self.organ.export_state()
            observation = {
                "beat": int(state.get("beat", 0)),
                "dopamine": float(state.get("dopamine", 0.5)),
                "memory": {
                    "slots_used": len(state.get("tt_memory_slots", ())),
                    "capacity": self.organ.memory_capacity,
                },
                "shadow": True,
            }
        anchor = (
            canonical.sum() * 0.0
            if enabled
            else canonical.detach().sum() * 0.0
        )
        return OrganSlotOutput(
            hidden=hidden,
            auxiliary_loss=anchor,
            observation=observation,
        )


class JEPAAuxiliarySlot(nn.Module):
    def __init__(
        self,
        organ: JEPAHeadV2,
        *,
        receptor_dim: int = 768,
        organ_dim: int = 512,
    ) -> None:
        super().__init__()
        if organ.d_model != organ_dim:
            raise ValueError("JEPA organ dimension does not match adapter")
        self.organ = organ
        self.adapter = OrganAdapter(receptor_dim, organ_dim)
        for parameter in self.organ.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        enabled: bool,
    ) -> OrganSlotOutput:
        if hidden.shape[1] < 2:
            raise ValueError("JEPA requires at least two sequence positions")
        canonical = self.adapter.encode(hidden)
        predicted, target, _ = self.organ(canonical)
        loss = (
            1.0
            - F.cosine_similarity(
                predicted,
                target.detach(),
                dim=-1,
            )
        ).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite transplanted JEPA loss")
        contribution = loss if enabled else loss.detach() * 0.0
        return OrganSlotOutput(
            hidden=hidden,
            auxiliary_loss=contribution,
            observation=None,
        )


class SpiderSenseSlot(nn.Module):
    """Sidecar slot: Spider-Sense confidence, minimal gradient path.

    Encodes GPT-2 768d → organ 512d → Spider MLP → confidence [B,T].
    Returns a tiny auxiliary loss (0.001 * |confidence - 0.5|) to create
    a gradient path for the adapter. Full calibration loss requires
    ground-truth correctness labels (not available in this experiment).
    """

    def __init__(
        self,
        organ: nn.Module,  # SpiderSense
        *,
        receptor_dim: int = 768,
        organ_dim: int = 512,
    ) -> None:
        super().__init__()
        self.organ = organ
        self.adapter = OrganAdapter(receptor_dim, organ_dim)
        for parameter in self.organ.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        enabled: bool,
    ) -> OrganSlotOutput:
        canonical = self.adapter.encode(hidden)
        confidence = self.organ(canonical)  # [B, T]
        if not torch.isfinite(confidence).all():
            raise FloatingPointError("non-finite transplanted Spider confidence")
        # Minimal gradient path: encourage confidence away from 0.5 (uncertain)
        aux = (confidence - 0.5).abs().mean() * 0.001
        loss = aux if enabled else aux.detach() * 0.0
        obs = {
            "confidence_mean": float(confidence.detach().mean()),
            "confidence_std": float(confidence.detach().std()),
        }
        return OrganSlotOutput(
            hidden=hidden,  # no residual injection
            auxiliary_loss=loss,
            observation=obs,
        )


class MTPSlot(nn.Module):
    """Sidecar slot: Multi-Token Prediction heads.

    Encodes GPT-2 768d → organ 512d → MTP Linear heads → decode → loss.
    Predicts future hidden states from current context.
    """

    def __init__(
        self,
        heads: nn.ModuleList,  # list of nn.Linear(512,512,bias=False)
        *,
        receptor_dim: int = 768,
        organ_dim: int = 512,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.organ = heads  # alias for lifecycle.py compatibility
        self.depth = int(depth)
        self.adapter = OrganAdapter(receptor_dim, organ_dim)
        for head in self.heads:
            for parameter in head.parameters():
                parameter.requires_grad_(False)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        enabled: bool,
    ) -> OrganSlotOutput:
        if hidden.shape[1] < self.depth + 2:
            return OrganSlotOutput(hidden=hidden, auxiliary_loss=None, observation=None)
        canonical = self.adapter.encode(hidden)  # [B, T, 512]
        total_loss = torch.tensor(0.0, device=hidden.device)
        for offset, head in enumerate(self.heads, start=2):
            if hidden.shape[1] <= offset:
                continue
            current = canonical[:, :-offset, :]  # [B, T-offset, 512]
            target = canonical[:, offset:, :].detach()  # [B, T-offset, 512]
            predicted = head(current)  # [B, T-offset, 512]
            loss = (1.0 - F.cosine_similarity(
                predicted.reshape(-1, canonical.size(-1)),
                target.reshape(-1, canonical.size(-1)),
                dim=-1,
            )).mean()
            total_loss = total_loss + loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError("non-finite transplanted MTP loss")
        contribution = total_loss if enabled else total_loss.detach() * 0.0
        return OrganSlotOutput(
            hidden=hidden,
            auxiliary_loss=contribution,
            observation={"mtp_loss": float(total_loss.detach())},
        )


class TTMResidualSlot(nn.Module):
    """Residual injection slot: Test-Time Memory retrieval via gate.

    Uses a minimal TestTimeMemory (proj_key + proj_value) to retrieve
    cached logits for similar hidden states. Injects via learned scalar gate.
    """

    def __init__(
        self,
        proj_key: nn.Linear,
        proj_value: nn.Linear,
        *,
        receptor_dim: int = 768,
        organ_dim: int = 512,
        memory_capacity: int = 1024,
    ) -> None:
        super().__init__()
        self.adapter = OrganAdapter(receptor_dim, organ_dim)
        self.organ = nn.ModuleDict({
            "proj_key": proj_key,
            "proj_value": proj_value,
        })
        for param in self.organ.parameters():
            param.requires_grad_(False)
        self.external_gate = nn.Parameter(torch.zeros(()))
        self._memory: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._capacity = memory_capacity

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        enabled: bool,
    ) -> OrganSlotOutput:
        canonical = self.adapter.encode(hidden)  # [B, T, 512]
        B, T, D = canonical.shape
        # Simple pooled retrieval: mean over sequence
        q = canonical.mean(dim=1)  # [B, 512]
        key = self.organ["proj_key"](q)  # [B, 128]
        value = self.organ["proj_value"](q)  # [B, 512]

        # Retrieve from memory (cosine similarity, all on same device)
        device = hidden.device
        best_value = torch.zeros_like(value)
        best_sim = torch.zeros(B, device=device)
        for mem_key, mem_value in self._memory:
            mk = mem_key.to(device)
            mv = mem_value.to(device)
            sim = F.cosine_similarity(key, mk.unsqueeze(0).expand(B, -1), dim=-1)
            mask = sim > best_sim
            best_sim = torch.where(mask, sim, best_sim)
            for b in range(B):
                if mask[b]:
                    best_value[b] = mv

        # Write to memory (if enabled, store on CPU)
        if enabled and len(self._memory) < self._capacity:
            self._memory.append((key[0].detach().cpu(), value[0].detach().cpu()))
        elif enabled:
            self._memory.pop(0)
            self._memory.append((key[0].detach().cpu(), value[0].detach().cpu()))

        # Decode and inject via gate
        delta = self.adapter.decode(best_value.unsqueeze(1).expand(-1, T, -1))  # [B, T, 768]
        gate = torch.tanh(self.external_gate) if enabled else torch.zeros_like(self.external_gate)
        combined = hidden + gate * delta * 0.15  # max_scale = 0.15

        return OrganSlotOutput(
            hidden=combined,
            auxiliary_loss=None,
            observation={"gate": float(gate.detach()), "memory_size": len(self._memory)},
        )
