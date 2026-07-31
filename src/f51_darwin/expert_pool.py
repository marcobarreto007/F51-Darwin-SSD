from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import torch
from torch import nn


class ModuleState(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    FROZEN = "frozen"
    QUARANTINE = "quarantine"
    MERGED = "merged"
    DEAD = "dead"


class ExpertModule(nn.Module):
    """Expert FFN standalone (usado para novos experts criados pelo organismo)."""
    def __init__(self, d_model: int, mlp_ratio: int = 2) -> None:
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ExpertRecord:
    id: str
    state: ModuleState
    usage_count: int
    score: float
    created_at_cycle: int
    # GATE 0.2: referencia ao expert REAL no DarwinXModel
    block_index: int = -1
    expert_index: int = -1
    _real_expert: nn.Module | None = field(default=None, repr=False)


class ExpertPool(nn.Module):
    """Pool de experts conectado aos experts REAIS do DarwinXModel MoE.
    
    GATE 0.2: Cada ExpertRecord referencia um expert real em 
    DarwinXModel.blocks[block_index].moe.fine_experts[expert_index].
    Metodos freeze/activate/mask operam nos parametros reais.
    """

    def __init__(self) -> None:
        super().__init__()
        self.modules_by_id = nn.ModuleDict()
        self.records: dict[str, ExpertRecord] = {}
        # GATE 0.2: router bias storage
        self._router_biases: dict[str, float] = {}

    def add_expert(
        self,
        expert_id: str,
        module: nn.Module | None,
        created_at_cycle: int,
        state: ModuleState = ModuleState.CANDIDATE,
        score: float = 0.0,
    ) -> ExpertRecord:
        if expert_id in self.records:
            raise ValueError(f"Expert already exists: {expert_id}")
        if module is not None:
            self.modules_by_id[expert_id] = module
        record = ExpertRecord(
            id=expert_id,
            state=state,
            usage_count=0,
            score=score,
            created_at_cycle=created_at_cycle,
        )
        self.records[expert_id] = record
        return record

    # ═══ GATE 0.2: Bind ao expert real do DarwinXModel ═══

    def bind_real_expert(
        self, expert_id: str, block_index: int, expert_index: int,
        real_module: nn.Module,
    ) -> None:
        """Conecta um registro do pool ao expert REAL do modelo MoE."""
        record = self.records.get(expert_id)
        if record is None:
            raise KeyError(f"Expert not found: {expert_id}")
        record.block_index = block_index
        record.expert_index = expert_index
        record._real_expert = real_module

    def freeze_expert(self, expert_id: str) -> bool:
        """Congela parametros (requires_grad=False). NAO altera estado de vida."""
        record = self.records.get(expert_id)
        if record is None or record._real_expert is None:
            return False
        for p in record._real_expert.parameters():
            p.requires_grad = False
        return True

    def activate_expert(self, expert_id: str) -> bool:
        """Reativa parametros (requires_grad=True). NAO altera estado de vida."""
        record = self.records.get(expert_id)
        if record is None or record._real_expert is None:
            return False
        for p in record._real_expert.parameters():
            p.requires_grad = True
        return True

    def mask_expert(self, expert_id: str, bias: float = -5.0) -> bool:
        """Adiciona vies negativo no router para evitar selecao."""
        record = self.records.get(expert_id)
        if record is None:
            return False
        self._router_biases[expert_id] = bias
        # Aplica o bias no router real se disponivel
        if record._real_expert is not None and record.block_index >= 0:
            # O bias vai ser aplicado pelo FineRouter.forward()
            pass
        return True

    def unmask_expert(self, expert_id: str) -> bool:
        """Remove o vies do router."""
        self._router_biases.pop(expert_id, None)
        return True

    def get_router_bias(self, block_index: int, expert_index: int) -> float:
        """Retorna o bias do router para um expert especifico."""
        for rid, record in self.records.items():
            if record.block_index == block_index and record.expert_index == expert_index:
                return self._router_biases.get(rid, 0.0)
        return 0.0

    def sync_usage_from_model(self, block_index: int, usage_counts: list[float]) -> None:
        """Sincroniza usage_count dos experts reais para o pool."""
        for rid, record in self.records.items():
            if record.block_index == block_index and record.expert_index < len(usage_counts):
                record.usage_count = int(usage_counts[record.expert_index])

    # ═══ Metodos originais ═══

    def set_state(self, expert_id: str, state: ModuleState) -> ExpertRecord:
        record = self.records[expert_id]
        record.state = state
        return record

    def set_score(self, expert_id: str, score: float) -> ExpertRecord:
        record = self.records[expert_id]
        record.score = float(score)
        return record

    def record_usage(self, expert_id: str, count: int = 1) -> ExpertRecord:
        record = self.records[expert_id]
        record.usage_count += count
        return record

    def active_records(self) -> list[ExpertRecord]:
        return [
            record
            for record in self.records.values()
            if record.state in {ModuleState.ACTIVE, ModuleState.FROZEN}
        ]

    def route_active(self, x: torch.Tensor, expert_ids: Iterable[str] | None = None) -> torch.Tensor:
        selected = list(expert_ids) if expert_ids is not None else [r.id for r in self.active_records()]
        if not selected:
            return x
        outputs = []
        for expert_id in selected:
            record = self.records[expert_id]
            module = record._real_expert
            if module is None and expert_id in self.modules_by_id:
                module = self.modules_by_id[expert_id]
            if module is None:
                continue
            self.record_usage(expert_id)
            outputs.append(module(x))
        if not outputs:
            return x
        return x + torch.stack(outputs, dim=0).mean(dim=0)

    def metadata(self) -> list[dict[str, object]]:
        return [
            {
                "id": record.id,
                "state": record.state.value,
                "usage_count": record.usage_count,
                "score": record.score,
                "created_at_cycle": record.created_at_cycle,
                "block_index": record.block_index,
                "expert_index": record.expert_index,
                "has_real_expert": record._real_expert is not None,
            }
            for record in self.records.values()
        ]
