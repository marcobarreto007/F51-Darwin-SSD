"""UniversalMemory — persistent associative memory organ.

Writes key→value associations from verified experience, recalls relevant
memories by query, and persists snapshots across sessions.  All neural
operations happen in the canonical 512-dim cognitive space.

Gate contract:
- gate=0.0 → zero residual, backbone untouched
- gate>0.0 → bounded residual injected at declared positions
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from f51_darwin.hashing import canonical_sha256

from .contracts import COGNITIVE_ORGAN_WIDTH


MEMORY_STATE_SCHEMA = "darwin-universal-memory-state-v1"

# A taxonomia de proveniencia decide o que pode ser recuperado sob
# require_verified, e o decaimento de quarentena em consolidate() depende dela.
# Ate 2026-07-31 ela existia apenas como comentario ao lado do campo: nada
# validava a escrita, "model_verified" nunca era atribuido por nenhum caminho,
# e uma string arbitraria entrava no registro em silencio. Um valor invalido
# nao falhava -- apenas nunca casava com nenhum filtro, tornando o registro
# invisivel para require_verified e imune ao decaimento.
PROVENANCE_VALUES = frozenset({"human", "model_verified", "model_quarantine"})
VERIFICATION_STATES = frozenset({"unverified", "verified", "contradicted"})


# ── Record types ────────────────────────────────────────────────────────────


@dataclass
class MemoryRecord:
    memory_id: str
    key_embedding: torch.Tensor   # [512]
    value_embedding: torch.Tensor  # [512]
    event_type: str               # explicit_teaching | verified_correction | calibrated_novelty
    source_digest: str
    provenance: str               # human | model_verified | model_quarantine
    verification_state: str        # unverified | verified | contradicted
    surprise: float
    confidence: float
    created_step: int
    last_access_step: int
    access_count: int
    tags: tuple[str, ...] = ()
    # Additional keys for the same memory, each [512]. A fact can be asked for
    # in many ways, and a single key forces one phrasing to stand for all of
    # them -- which is what caps paraphrase recall regardless of how well the
    # encoder is trained. Recall scores a record by its best-matching key, so
    # storing several phrasings at teach time widens the entry without
    # duplicating the entry itself.
    alt_keys: tuple[torch.Tensor, ...] = ()

    @property
    def all_keys(self) -> tuple[torch.Tensor, ...]:
        return (self.key_embedding, *self.alt_keys)

    @property
    def is_verified(self) -> bool:
        return self.verification_state == "verified"

    @property
    def is_quarantined(self) -> bool:
        return self.provenance == "model_quarantine"


@dataclass
class MemoryRecall:
    record: MemoryRecord
    score: float
    calibrated_probability: float
    abstained: bool
    abstention_reason: str | None
    # Gap between this candidate and the next-best one. Absolute cosine drifts
    # with phrasing, so the top-1/top-2 gap is the more stable confidence
    # signal when deciding whether the store actually holds the answer.
    margin: float = 0.0


# ── Memory readout adapter ──────────────────────────────────────────────────


class MemoryReadoutAdapter(nn.Module):
    """Converts a 512-dim memory value into a decoder-conditioning residual.

    Trained with frozen backbone to make memory values decodable by the brain.
    """

    def __init__(self, d_model: int, max_scale: float) -> None:
        super().__init__()
        self.output_adapter = nn.Linear(COGNITIVE_ORGAN_WIDTH, d_model, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))
        self.max_scale = float(max_scale)
        self.d_model = int(d_model)

    def effective_scale(self) -> torch.Tensor:
        return self.max_scale * torch.tanh(self.gate)

    def condition_hidden(
        self,
        hidden: torch.Tensor,
        memory_value: torch.Tensor,  # [B, 512] or [B, 1, 512]
        positions: torch.Tensor,     # [B, 1]
    ) -> torch.Tensor:
        gate_val = self.effective_scale().to(device=hidden.device, dtype=hidden.dtype)
        if bool(self.gate.detach().eq(0).item()):
            return hidden
        if memory_value.ndim == 2:
            memory_value = memory_value.unsqueeze(1)
        projected = self.output_adapter(memory_value.to(device=hidden.device, dtype=hidden.dtype))
        reference = hidden.gather(
            dim=1,
            index=positions.unsqueeze(-1).expand(-1, -1, self.d_model),
        )
        projected_norm = projected.float().norm(dim=-1, keepdim=True)
        reference_norm = reference.float().norm(dim=-1, keepdim=True)
        factor = (reference_norm / projected_norm.clamp_min(1e-6)).clamp(max=1.0)
        bounded = projected * factor.to(dtype=projected.dtype)
        delta = gate_val * bounded
        result = hidden.clone()
        result.scatter_add_(
            dim=1,
            index=positions.unsqueeze(-1).expand(-1, -1, self.d_model),
            src=delta,
        )
        return result


# ── The organ ───────────────────────────────────────────────────────────────


class UniversalMemory(nn.Module):
    """Associative memory organ with learned key/value projection and readout.

    Writes are gated by three independent triggers (explicit_teaching,
    verified_correction, calibrated_novelty).  Recalls are scored by
    cosine similarity against a learned key projector and thresholded
    for abstention.  Unverified model-generated memories enter quarantine.

    The organ preserves its own state_dict for checkpointing and can
    snapshot/restore its memory store independently of model weights.
    """

    def __init__(
        self,
        d_model: int,
        max_scale: float = 0.15,
        *,
        recall_top_k: int = 4,
        recall_threshold: float = 0.6,
        novelty_threshold: float = 0.85,
        margin_threshold: float = 0.0,
        high_confidence_threshold: float = float("inf"),
        key_encoder_hidden: int | None = None,
        max_slots: int = 4096,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        self.d_model = int(d_model)
        self.max_scale = float(max_scale)
        self.recall_top_k = int(recall_top_k)
        self.recall_threshold = float(recall_threshold)
        self.novelty_threshold = float(novelty_threshold)
        # 0.0 disables the margin gate, preserving score-only abstention.
        self.margin_threshold = float(margin_threshold)
        # A score at or above this short-circuits the margin gate: when the
        # absolute match is overwhelming, near neighbours are irrelevant.
        # Defaults to +inf, which leaves the margin gate unconditional.
        self.high_confidence_threshold = float(high_confidence_threshold)
        self.max_slots = int(max_slots)

        # Key/value projectors: d_model → 512.
        # Bias matters here: hidden states for these prompts share a large
        # constant component (phrasing/template), and entity identity lives in
        # a small residual around it.  Without a bias the projection cannot
        # subtract that offset, so the dominant direction survives into the
        # cosine and swamps the signal the recall threshold has to read.
        # A single linear map has to separate entity identity from phrasing in
        # a space where phrasing dominates: a 1-NN probe over centred raw
        # hidden states never retrieves another phrasing of the same entity
        # (accuracy 0.000 for last-token, mean, max and last-8 pooling). One
        # hidden layer gives the projector room to do that non-linearly.
        # Defaults to None, which keeps the original single Linear so existing
        # adapter checkpoints stay loadable.
        self.key_encoder_hidden = (
            None if key_encoder_hidden is None else int(key_encoder_hidden)
        )
        if self.key_encoder_hidden is None:
            self.key_encoder = nn.Linear(d_model, COGNITIVE_ORGAN_WIDTH, bias=True)
        else:
            self.key_encoder = nn.Sequential(
                nn.Linear(d_model, self.key_encoder_hidden, bias=True),
                nn.GELU(),
                nn.Linear(self.key_encoder_hidden, COGNITIVE_ORGAN_WIDTH, bias=True),
            )
        self.value_encoder = nn.Linear(d_model, COGNITIVE_ORGAN_WIDTH, bias=True)

        # Readout: 512 → d_model with bounded gate
        self.readout = MemoryReadoutAdapter(d_model, max_scale)

        # In-memory store
        self._store: dict[str, MemoryRecord] = {}
        self._step_counter: int = 0

    # ── encode / decode ──────────────────────────────────────────────────

    @property
    def _key_dtype(self) -> torch.dtype:
        """dtype of the key projector, which may be a Linear or a Sequential."""
        return next(self.key_encoder.parameters()).dtype

    def encode_query(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project the last-token hidden state into memory-key space.

        Args:
            hidden: [B, T, d_model]

        Returns:
            [B, 512]
        """
        query = hidden[:, -1, :].to(self._key_dtype)
        return self.key_encoder(query)

    def encode_value(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project tokens into memory-value space.

        Args:
            hidden: [B, T, d_model]

        Returns:
            [B, 512]
        """
        pooled = hidden.mean(dim=1).to(self.value_encoder.weight.dtype)
        return self.value_encoder(pooled)

    # ── write ────────────────────────────────────────────────────────────

    def teach(
        self,
        key_hidden: torch.Tensor,
        value_hidden: torch.Tensor,
        event_type: str = "explicit_teaching",
        *,
        provenance: str = "human",
        confidence: float = 1.0,
        tags: tuple[str, ...] = (),
        alt_key_hiddens: Sequence[torch.Tensor] = (),
    ) -> str:
        """Explicitly write an association into memory.

        ``alt_key_hiddens`` are further phrasings of the same question. They
        cost one 512-float vector each and are the cheapest way to lift
        paraphrase recall, because they move the burden off the encoder having
        to map every wording to one point.
        """
        if event_type not in {
            "explicit_teaching",
            "verified_correction",
            "calibrated_novelty",
        }:
            raise ValueError(f"unsupported event_type: {event_type}")
        if key_hidden.ndim not in (2, 3) or value_hidden.ndim not in (2, 3):
            raise ValueError("key and value hidden must be 2D or 3D tensors")

        with torch.no_grad():
            if key_hidden.ndim == 3:
                key_emb = self.encode_query(key_hidden)
            else:
                key_emb = self.key_encoder(key_hidden)
            if value_hidden.ndim == 3:
                val_emb = self.encode_value(value_hidden)
            else:
                val_emb = self.value_encoder(value_hidden)
            alt_embeddings = []
            for hidden in alt_key_hiddens:
                if hidden.ndim not in (2, 3):
                    raise ValueError("alt key hidden must be 2D or 3D tensors")
                emb = (
                    self.encode_query(hidden) if hidden.ndim == 3
                    else self.key_encoder(hidden)
                )
                alt_embeddings.append(emb.squeeze(0) if emb.ndim == 2 else emb)

        return self._write_raw(
            key_embedding=key_emb.squeeze(0) if key_emb.ndim == 2 else key_emb,
            value_embedding=val_emb.squeeze(0) if val_emb.ndim == 2 else val_emb,
            event_type=event_type,
            provenance=provenance,
            verification_state="verified" if provenance == "human" else "unverified",
            confidence=confidence,
            tags=tags,
            alt_keys=tuple(alt_embeddings),
        )

    def write_if_novel(
        self,
        key_hidden: torch.Tensor,
        value_hidden: torch.Tensor,
        event_type: str = "calibrated_novelty",
    ) -> str | None:
        """Write only if the query is sufficiently novel vs existing memories."""
        with torch.no_grad():
            query = self.encode_query(key_hidden) if key_hidden.ndim == 3 else self.key_encoder(key_hidden)
            if query.ndim == 2:
                query = query.squeeze(0)
            if len(self._store) == 0:
                top_score = -1.0
            else:
                keys = torch.stack([r.key_embedding for r in self._store.values()])
                sims = F.cosine_similarity(query.unsqueeze(0), keys, dim=-1)
                top_score = float(sims.max())
            if top_score >= self.novelty_threshold:
                return None
        return self.teach(
            key_hidden,
            value_hidden,
            event_type=event_type,
            provenance="model_quarantine",
            confidence=float(max(0.0, 1.0 - top_score)),
        )

    def _write_raw(
        self,
        key_embedding: torch.Tensor,
        value_embedding: torch.Tensor,
        event_type: str,
        provenance: str,
        verification_state: str,
        confidence: float,
        tags: tuple[str, ...],
        alt_keys: tuple[torch.Tensor, ...] = (),
    ) -> str:
        # Funil unico de escrita: validar aqui cobre teach, write_if_novel e
        # qualquer caminho futuro.
        if provenance not in PROVENANCE_VALUES:
            raise ValueError(
                f"provenance {provenance!r} outside {sorted(PROVENANCE_VALUES)}"
            )
        if verification_state not in VERIFICATION_STATES:
            raise ValueError(
                f"verification_state {verification_state!r} outside "
                f"{sorted(VERIFICATION_STATES)}"
            )
        if len(self._store) >= self.max_slots:
            self._evict_lru()
        memory_id = canonical_sha256({
            "key": key_embedding.detach().cpu().tolist(),
            "step": self._step_counter,
            "event": event_type,
        })[:32]
        self._step_counter += 1
        self._store[memory_id] = MemoryRecord(
            memory_id=memory_id,
            key_embedding=key_embedding.detach().cpu().clone(),
            value_embedding=value_embedding.detach().cpu().clone(),
            event_type=event_type,
            source_digest="",
            provenance=provenance,
            verification_state=verification_state,
            surprise=float(1.0 - confidence),
            confidence=float(confidence),
            created_step=self._step_counter,
            last_access_step=self._step_counter,
            access_count=0,
            tags=tags,
            alt_keys=tuple(k.detach().cpu().clone() for k in alt_keys),
        )
        return memory_id

    def _evict_lru(self) -> None:
        if not self._store:
            return
        victim = min(self._store.values(), key=lambda r: (r.access_count, r.last_access_step))
        del self._store[victim.memory_id]

    # ── recall ───────────────────────────────────────────────────────────

    def recall(
        self,
        query_hidden: torch.Tensor,
        top_k: int | None = None,
        *,
        require_verified: bool = False,
    ) -> list[MemoryRecall]:
        """Retrieve top-k memories for a query, with abstention.

        Args:
            query_hidden: [B, T, d_model] or [D] or [B, D]
            top_k: override recall_top_k
            require_verified: only return verified memories

        Returns:
            list of MemoryRecall, sorted by score descending
        """
        k = self.recall_top_k if top_k is None else int(top_k)
        if len(self._store) == 0:
            return [MemoryRecall(
                record=MemoryRecord(
                    memory_id="", key_embedding=torch.zeros(512),
                    value_embedding=torch.zeros(512),
                    event_type="abstention", source_digest="",
                    provenance="", verification_state="unverified",
                    surprise=0.0, confidence=0.0,
                    created_step=0, last_access_step=0, access_count=0,
                ),
                score=0.0, calibrated_probability=0.0,
                abstained=True, abstention_reason="empty_store",
            )]

        with torch.no_grad():
            if query_hidden.ndim == 3:
                query = self.encode_query(query_hidden)
                # For batched queries, pool across batch to get a single query
                if query.ndim == 2 and query.shape[0] > 1:
                    query = query.mean(dim=0)  # [B, 512] -> [512]
                elif query.ndim == 2:
                    query = query.squeeze(0)
            else:
                query = self.key_encoder(query_hidden.unsqueeze(0)).squeeze(0)
            if query.ndim == 2:
                query = query.squeeze(0)

            candidates = list(self._store.values())
            if require_verified:
                candidates = [r for r in candidates if r.is_verified]
            if not candidates:
                return [MemoryRecall(
                    record=MemoryRecord(
                        memory_id="", key_embedding=torch.zeros(512),
                        value_embedding=torch.zeros(512),
                        event_type="abstention", source_digest="",
                        provenance="", verification_state="unverified",
                        surprise=0.0, confidence=0.0,
                        created_step=0, last_access_step=0, access_count=0,
                    ),
                    score=0.0, calibrated_probability=0.0,
                    abstained=True, abstention_reason="no_verified_candidates",
                )]

            # A record scores by its best-matching key, so records carrying
            # several phrasings are compared on equal footing with single-key
            # ones rather than being ranked once per key.
            flat_keys, owner = [], []
            for position, record in enumerate(candidates):
                for key in record.all_keys:
                    flat_keys.append(key)
                    owner.append(position)
            record_keys = torch.stack(flat_keys)
            owner_index = torch.tensor(owner, device=record_keys.device)
            flat_sims = F.cosine_similarity(
                query.unsqueeze(0).to(record_keys.device), record_keys, dim=-1
            )
            sims = torch.full(
                (len(candidates),), -1.0,
                dtype=flat_sims.dtype, device=flat_sims.device,
            ).scatter_reduce(0, owner_index, flat_sims, reduce="amax")
            scores, indices = sims.sort(descending=True)

            # Gap between the best candidate and the runner-up. A query whose
            # answer is genuinely in the store separates from the field; a
            # query about something never taught matches everything about
            # equally well, however high its absolute cosine happens to be.
            top_margin = (
                float(scores[0] - scores[1]) if len(candidates) > 1 else float("inf")
            )

            results: list[MemoryRecall] = []
            for i in range(min(k, len(candidates))):
                idx = int(indices[i])
                score = float(scores[i])
                record = candidates[idx]
                record.last_access_step = self._step_counter
                record.access_count += 1
                below_score = score < self.recall_threshold
                # The margin gate only qualifies the top-1 claim; lower-ranked
                # candidates are reported on score alone.  A score above the
                # high-confidence band skips it: an exact re-encode of a taught
                # query scores 1.0 while sitting only ~0.09 clear of its
                # neighbours, and an unconditional margin gate rejects it.
                below_margin = (
                    i == 0
                    and self.margin_threshold > 0.0
                    and top_margin < self.margin_threshold
                    and score < self.high_confidence_threshold
                )
                abstained = below_score or below_margin
                reason = None
                if below_score:
                    reason = "below_threshold"
                elif below_margin:
                    reason = "ambiguous_margin"
                prob = float(torch.sigmoid(torch.tensor((score - self.recall_threshold) * 10.0)))
                results.append(MemoryRecall(
                    record=record,
                    score=score,
                    calibrated_probability=prob,
                    abstained=abstained,
                    abstention_reason=reason,
                    margin=top_margin if i == 0 else 0.0,
                ))

            return results

    def best_value(
        self,
        query_hidden: torch.Tensor,
        *,
        require_verified: bool = True,
    ) -> torch.Tensor | None:
        """Return the best memory value embedding [512] or None if abstained."""
        recalls = self.recall(query_hidden, top_k=1, require_verified=require_verified)
        if not recalls or recalls[0].abstained:
            return None
        return recalls[0].record.value_embedding

    # ── consolidation ────────────────────────────────────────────────────

    def consolidate(self) -> dict[str, Any]:
        """Run off-critical-path memory housekeeping.

        Returns a report of actions taken.
        """
        report: dict[str, Any] = {
            "duplicates_merged": 0,
            "contradictions_detected": 0,
            "decayed": 0,
            "total_before": len(self._store),
        }
        # Merge duplicates (same key within 0.99 cosine sim)
        records = list(self._store.values())
        merged: set[str] = set()
        for i, ri in enumerate(records):
            if ri.memory_id in merged:
                continue
            for rj in records[i + 1:]:
                if rj.memory_id in merged:
                    continue
                sim = float(F.cosine_similarity(
                    ri.key_embedding.unsqueeze(0), rj.key_embedding.unsqueeze(0)
                ))
                if sim > 0.99 and ri.event_type == rj.event_type:
                    # Keep the verified one, decay the other
                    if ri.is_verified and not rj.is_verified:
                        del self._store[rj.memory_id]
                        merged.add(rj.memory_id)
                        report["duplicates_merged"] += 1
                    elif rj.is_verified and not ri.is_verified:
                        del self._store[ri.memory_id]
                        merged.add(ri.memory_id)
                        report["duplicates_merged"] += 1

        # Decay unverified, never-accessed memories older than 100 steps
        to_decay = [
            rid for rid, r in self._store.items()
            if not r.is_verified and r.access_count == 0
            and (self._step_counter - r.created_step) > 100
        ]
        for rid in to_decay:
            del self._store[rid]
            report["decayed"] += 1

        report["total_after"] = len(self._store)
        return report

    # ── persistence ──────────────────────────────────────────────────────

    def encoder_snapshot(self) -> dict[str, Any]:
        """Encoder weights plus their fingerprint, for exact restoration.

        The memory store is only meaningful under the projections that wrote
        it, so a durable memory needs a way to put those projections back.
        """
        return {
            "fingerprint": self._encoder_fingerprint(),
            "weights": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.state_dict().items()
                if name.startswith(("key_encoder", "value_encoder"))
            },
        }

    def restore_encoder(self, snapshot: Mapping[str, Any]) -> None:
        """Restore encoder weights and verify the fingerprint matches.

        Raises if restoration did not reproduce the recorded projections, so
        a silent partial restore cannot pass as a successful one.
        """
        weights = snapshot.get("weights")
        if not isinstance(weights, Mapping) or not weights:
            raise ValueError("encoder snapshot carries no weights")
        current = self.state_dict()
        for name, tensor in weights.items():
            if name not in current:
                raise ValueError(f"encoder snapshot names unknown tensor {name!r}")
            if tuple(current[name].shape) != tuple(tensor.shape):
                raise ValueError(
                    f"encoder snapshot shape mismatch on {name!r}: "
                    f"{tuple(tensor.shape)} vs {tuple(current[name].shape)}"
                )
        with torch.no_grad():
            for name, tensor in weights.items():
                current[name].copy_(tensor.to(current[name].device))

        expected = str(snapshot.get("fingerprint", ""))
        if expected:
            actual = self._encoder_fingerprint()
            if actual != expected:
                raise ValueError(
                    f"encoder restore did not reproduce the recorded projections: "
                    f"expected {expected[:16]}..., got {actual[:16]}..."
                )

    def _encoder_fingerprint(self) -> str:
        """Digest of the projections that produced the stored key embeddings.

        Keys are stored ENCODED. A snapshot restored into a module whose
        key_encoder differs encodes the query with one projection and compares
        it against keys written by another, which does not fail -- it returns
        plausible, wrong scores. Measured: recall 1.0/1.0/1.0 became
        -0.007/0.101/0.050 with records restored and encoders left fresh.
        Binding the fingerprint into the snapshot turns that silent corruption
        into a loud error.
        """
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            if not name.startswith(("key_encoder", "value_encoder")):
                continue
            data = tensor.detach().to(device="cpu").contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(data.dtype).encode("ascii"))
            digest.update(str(tuple(data.shape)).encode("ascii"))
            digest.update(
                memoryview(data.reshape(-1).view(torch.uint8).numpy()).cast("B")
            )
        return digest.hexdigest()

    def memory_state_dict(self) -> dict[str, Any]:
        """Serializable snapshot of the memory store."""
        records_payload = []
        for r in self._store.values():
            records_payload.append({
                "memory_id": r.memory_id,
                "key_embedding": r.key_embedding.tolist(),
                "value_embedding": r.value_embedding.tolist(),
                "event_type": r.event_type,
                "source_digest": r.source_digest,
                "provenance": r.provenance,
                "verification_state": r.verification_state,
                "surprise": r.surprise,
                "confidence": r.confidence,
                "created_step": r.created_step,
                "last_access_step": r.last_access_step,
                "access_count": r.access_count,
                "tags": list(r.tags),
                "alt_keys": [k.tolist() for k in r.alt_keys],
            })
        payload: dict[str, Any] = {
            "schema": MEMORY_STATE_SCHEMA,
            "step_counter": self._step_counter,
            "records": records_payload,
            "encoder_fingerprint": self._encoder_fingerprint(),
        }
        payload["snapshot_sha256"] = canonical_sha256(payload)
        return payload

    def load_memory_state(
        self,
        state: Mapping[str, Any],
        *,
        verify: bool = True,
    ) -> None:
        """Restore memory store from a snapshot.

        ``verify=True`` checks two things the snapshot already carried but
        nothing enforced: that the payload matches its own digest, and that
        the encoders in this module are the ones that wrote the keys. Pass
        ``verify=False`` only to import a snapshot deliberately, accepting
        that recall scores will be meaningless until the encoders are
        restored with ``load_state_dict``.
        """
        if state.get("schema") != MEMORY_STATE_SCHEMA:
            raise ValueError("unsupported memory state schema")

        if verify:
            declared = str(state.get("snapshot_sha256", ""))
            recomputed = canonical_sha256(
                {k: v for k, v in state.items() if k != "snapshot_sha256"}
            )
            if not declared:
                raise ValueError(
                    "memory snapshot carries no snapshot_sha256; refusing to "
                    "load unverifiable state"
                )
            if declared != recomputed:
                raise ValueError(
                    f"memory snapshot digest mismatch: declared {declared[:16]}... "
                    f"but content hashes to {recomputed[:16]}..."
                )

            stored_fp = state.get("encoder_fingerprint")
            if stored_fp is not None:
                current_fp = self._encoder_fingerprint()
                if stored_fp != current_fp:
                    raise ValueError(
                        "memory snapshot was written by different encoders "
                        f"({str(stored_fp)[:16]}... vs {current_fp[:16]}...). "
                        "Restore the module weights with load_state_dict before "
                        "load_memory_state, or pass verify=False to accept "
                        "meaningless recall scores."
                    )

        self._store.clear()
        self._step_counter = int(state.get("step_counter", 0))
        for raw in state.get("records", []):
            rid = str(raw["memory_id"])
            # O carregamento reconstroi registros sem passar pelo funil de
            # escrita, entao um snapshot estrangeiro poderia injetar valores
            # fora da taxonomia e produzir registros invisiveis aos filtros.
            raw_prov = str(raw.get("provenance", "model_quarantine"))
            raw_state = str(raw.get("verification_state", "unverified"))
            if raw_prov not in PROVENANCE_VALUES:
                raise ValueError(
                    f"record {rid}: provenance {raw_prov!r} outside "
                    f"{sorted(PROVENANCE_VALUES)}"
                )
            if raw_state not in VERIFICATION_STATES:
                raise ValueError(
                    f"record {rid}: verification_state {raw_state!r} outside "
                    f"{sorted(VERIFICATION_STATES)}"
                )
            self._store[rid] = MemoryRecord(
                memory_id=rid,
                key_embedding=torch.tensor(raw["key_embedding"], dtype=torch.float32),
                value_embedding=torch.tensor(raw["value_embedding"], dtype=torch.float32),
                event_type=str(raw["event_type"]),
                source_digest=str(raw.get("source_digest", "")),
                provenance=str(raw.get("provenance", "model_quarantine")),
                verification_state=str(raw.get("verification_state", "unverified")),
                surprise=float(raw.get("surprise", 0.0)),
                confidence=float(raw.get("confidence", 0.0)),
                created_step=int(raw.get("created_step", 0)),
                last_access_step=int(raw.get("last_access_step", 0)),
                access_count=int(raw.get("access_count", 0)),
                tags=tuple(raw.get("tags", ())),
                alt_keys=tuple(
                    torch.tensor(k, dtype=torch.float32)
                    for k in raw.get("alt_keys", ())
                ),
            )

    @property
    def slot_count(self) -> int:
        return len(self._store)

    @property
    def verified_count(self) -> int:
        return sum(1 for r in self._store.values() if r.is_verified)

    @property
    def quarantined_count(self) -> int:
        return sum(1 for r in self._store.values() if r.is_quarantined)
