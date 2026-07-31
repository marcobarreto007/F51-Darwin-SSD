from __future__ import annotations

import torch
from torch import nn

from .adapters import ZeroGatedCognitiveAdapter
from .contracts import (
    CANONICAL_ORGAN_IDS,
    COGNITIVE_ARCHITECTURE_VERSION,
    COGNITIVE_ORGAN_WIDTH,
    CognitiveForwardMetadata,
    CognitivePulseEvent,
    OrganKind,
    ResidualCondition,
)
from .executive import (
    ExecutiveAction,
    ExecutiveDecision,
    UniversalExecutive,
)
from .memory import UniversalMemory
from .pulse import CognitivePulse, CognitivePulseRecord
from .world_model import (
    NUM_PROPOSAL_SLOTS,
    HierarchicalWorldModel,
)


class CognitiveRuntime(nn.Module):
    """Three-organ cognitive runtime: Memory -> WorldModel -> Executive.

    Shadow mode (gates zero):
        All three organs execute but produce zero residual.
        Hidden states and logits are exactly preserved.
        Used for contract validation and identity proof.

    Active mode (gates > 0):
        Memory retrieves, WorldModel proposes, Executive selects,
        and the selected plan conditions the decoder through bounded
        adapters.  All events are recorded in the CognitivePulse.
    """

    def __init__(self, d_model: int, max_scale: float) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.max_scale = float(max_scale)

        # --- Organs ----------------------------------------------------
        self.memory = UniversalMemory(
            d_model, max_scale,
            recall_threshold=0.5,
            margin_threshold=0.0,
            high_confidence_threshold=float("inf"),
        )
        self.world_model = HierarchicalWorldModel(d_model, max_scale)
        self.executive = UniversalExecutive()

        # --- Legacy adapters (backward compat with shadow probe) -------
        self.memory_adapter = ZeroGatedCognitiveAdapter(d_model, max_scale)
        self.world_model_adapter = ZeroGatedCognitiveAdapter(d_model, max_scale)

        # --- State -----------------------------------------------------
        self._pulse = CognitivePulse()
        self._mode: str = "shadow"

    # -- Mode control -------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    def set_active(self) -> None:
        self._mode = "active"

    def set_shadow(self) -> None:
        self._mode = "shadow"

    # -- Shadow path (backward-compatible, zero-impact) ---------------------

    def _zero_gate_path(
        self,
        hidden: torch.Tensor,
        adapter: ZeroGatedCognitiveAdapter,
        organ: OrganKind,
    ) -> torch.Tensor:
        last = hidden[:, -1:, :]
        values = adapter.encode(last)
        positions = torch.full(
            (hidden.shape[0], 1),
            hidden.shape[1] - 1,
            device=hidden.device,
            dtype=torch.long,
        )
        return adapter.apply_condition(
            hidden,
            ResidualCondition(
                condition_id=f"shadow:{organ.value}",
                organ=organ,
                values=values,
                positions=positions,
            ),
        )

    def observe_shadow(
        self,
        hidden: torch.Tensor,
        metadata: CognitiveForwardMetadata,
    ) -> tuple[torch.Tensor, CognitivePulseRecord]:
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        result = self._zero_gate_path(
            hidden,
            self.memory_adapter,
            OrganKind.MEMORY,
        )
        result = self._zero_gate_path(
            result,
            self.world_model_adapter,
            OrganKind.WORLD_MODEL,
        )
        if not torch.equal(result, hidden):
            raise RuntimeError("shadow cognition changed hidden state")
        record = self._pulse.append(CognitivePulseEvent.shadow(metadata))
        return result, record

    # -- Active path --------------------------------------------------------

    def observe_active(
        self,
        hidden: torch.Tensor,
        metadata: CognitiveForwardMetadata,
        *,
        goal_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, CognitivePulseRecord]:
        """Full cognitive cycle: Memory -> WorldModel -> Executive -> condition.

        Args:
            hidden: [B, T, d_model] backbone hidden states
            metadata: forward metadata for pulse recording
            goal_embedding: [B, 512] optional goal representation

        Returns:
            (conditioned_hidden, pulse_record)
        """
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")

        result = hidden
        memory_ids: list[str] = []
        candidate_ids: list[str] = []
        selected_id: str | None = None
        compute_spent = 0

        # --- 1. UniversalMemory: recall ---------------------------------
        recalls = self.memory.recall(result, top_k=4, require_verified=True)
        memory_embeddings: list[torch.Tensor] = []
        for rc in recalls:
            if not rc.abstained:
                memory_ids.append(rc.record.memory_id)
                memory_embeddings.append(rc.record.value_embedding)

        # Apply memory conditioning if gate is open
        if memory_embeddings:
            best_mem = memory_embeddings[0]
            pos = torch.full(
                (result.shape[0], 1),
                result.shape[1] - 1,
                device=result.device,
                dtype=torch.long,
            )
            result = self.memory.readout.condition_hidden(result, best_mem, pos)
            compute_spent += 1

        # --- 2. HierarchicalWorldModel: propose -------------------------
        traj_candidates = self.world_model.propose_trajectories(
            result, goal_embedding, memory_embeddings
        )
        cand_dicts = [
            {"z_short": c.z_short, "z_medium": c.z_medium, "z_long": c.z_long}
            for c in traj_candidates
        ]

        # --- 3. UniversalExecutive: score and decide --------------------
        mem_scores = torch.tensor(
            [rc.score for rc in recalls if not rc.abstained][:NUM_PROPOSAL_SLOTS]
            or [0.0]
        )
        scored = self.executive.score_candidates(
            cand_dicts,
            goal_embedding=goal_embedding,
            memory_recall_scores=mem_scores,
        )
        for s in scored:
            candidate_ids.append(s.candidate_id)

        decision = self.executive.decide(scored)

        # --- 4. Apply plan conditioning if SELECT -----------------------
        if (
            decision.action == ExecutiveAction.SELECT
            and decision.selected_candidate_id is not None
        ):
            selected_id = decision.selected_candidate_id
            idx = int(decision.selected_candidate_id.split("_")[-1])
            if 0 <= idx < len(traj_candidates):
                c = traj_candidates[idx]
                pos = torch.full(
                    (result.shape[0], 1),
                    result.shape[1] - 1,
                    device=result.device,
                    dtype=torch.long,
                )
                result = self.world_model.plan_adapter.condition(
                    result,
                    c.z_short.unsqueeze(0).expand(result.shape[0], -1),
                    c.z_medium.unsqueeze(0).expand(result.shape[0], -1),
                    c.z_long.unsqueeze(0).expand(result.shape[0], -1),
                    pos,
                )
                compute_spent += 1

        # --- 5. Emit pulse ----------------------------------------------
        event = CognitivePulseEvent(
            schema="darwin-cognitive-pulse-v1",
            mode="active",
            step_id=metadata.step_id,
            checkpoint_id=metadata.checkpoint_id,
            context_digest=metadata.context_digest,
            selected_memory_ids=tuple(memory_ids),
            candidate_ids=tuple(candidate_ids),
            selected_candidate_id=selected_id,
            prediction_error=None,
            uncertainty=(
                float(scored[0].trajectory_uncertainty) if scored else None
            ),
            compute_spent=compute_spent,
        )
        record = self._pulse.append(event)
        return result, record

    # -- Manifest -----------------------------------------------------------

    def manifest(self) -> dict[str, object]:
        return {
            "architecture_version": COGNITIVE_ARCHITECTURE_VERSION,
            "organ_width": COGNITIVE_ORGAN_WIDTH,
            "organ_ids": list(CANONICAL_ORGAN_IDS),
            "influence_paths": [
                OrganKind.MEMORY.value,
                OrganKind.WORLD_MODEL.value,
            ],
            "executive_authority": "selection_only",
            "pulse_authority": "events_only",
            "memory_slots": self.memory.slot_count,
            "memory_verified": self.memory.verified_count,
        }

    def pulse_state_dict(self) -> dict[str, object]:
        return self._pulse.state_dict()
