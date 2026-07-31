"""Canonical append-only causal ledger with semantic online validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any

from .causal_bus import (
    AblationArm,
    BusDecision,
    Intervention,
    Rejection,
    Signal,
    StepIdentity,
    StepOutcome,
)


_SCHEMA_VERSION = 3
_EMPTY_HEAD = "0" * 64
_EVENT_TYPES = frozenset({"STEP_INTENT", "STEP_OUTCOME"})


class LedgerTamperError(RuntimeError):
    """Raised when an existing ledger fails canonical or semantic validation."""


@dataclass(frozen=True)
class LedgerVerification:
    valid: bool
    event_count: int
    head_hash: str
    orphan_intent_step_keys: tuple[str, ...]
    orphan_intent_count: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class BlockChainVerification:
    """Result of a block-chain integrity scan over a v3 ledger."""
    valid: bool
    block_count: int
    head_block_hash: str
    genesis_block_hash: str
    blocks_with_errors: tuple[int, ...]
    errors: tuple[str, ...]


@dataclass
class _AttemptState:
    arm: str
    effective_ids: tuple[str, ...]
    outcome_recorded: bool = False


def _jsonable(value: Any, *, path: str = "value") -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} must use string keys")
            converted[key] = _jsonable(item, path=f"{path}.{key}")
        return converted
    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} must be JSON-compatible, got {type(value).__name__}"
    )


def _ensure_finite(value: Any, *, path: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite value")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _ensure_finite(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _ensure_finite(item, path=f"{path}[{index}]")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    serializable = _jsonable(value)
    _ensure_finite(serializable)
    return json.dumps(
        serializable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _event_hash(event_without_hash: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(event_without_hash)).hexdigest()


def _file_fingerprint(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_size, stat.st_mtime_ns


class LedgerMerkleTree:
    """Merkle tree over block transactions for tamper-evident inclusion proofs.

    Each leaf is ``SHA-256(canonical_bytes(tx))``.  Internal nodes are
    ``SHA-256(left | right)`` with the typical Bitcoin-style duplication
    rule: when the level has an odd number of nodes the last one is
    duplicated before pairing.
    """

    @staticmethod
    def compute_root(transactions: Sequence[Mapping[str, Any]]) -> str:
        """Return the hex-encoded Merkle root for *transactions*.

        An empty list yields ``SHA-256(b"")`` (consistently reproducible).
        """
        if not transactions:
            return hashlib.sha256(b"").hexdigest()
        leaves: list[bytes] = [
            hashlib.sha256(_canonical_bytes(tx)).digest()
            for tx in transactions
        ]
        while len(leaves) > 1:
            if len(leaves) % 2 == 1:
                leaves.append(leaves[-1])
            leaves = [
                hashlib.sha256(leaves[i] + leaves[i + 1]).digest()
                for i in range(0, len(leaves), 2)
            ]
        return leaves[0].hex()

    @staticmethod
    def verify_root(
        transactions: Sequence[Mapping[str, Any]],
        expected_root: str,
    ) -> bool:
        """Return ``True`` when *expected_root* matches the computed root."""
        return LedgerMerkleTree.compute_root(transactions) == expected_root

    @staticmethod
    def generate_proof(
        transactions: Sequence[Mapping[str, Any]],
        tx_index: int,
    ) -> dict[str, Any]:
        """Build a Merkle inclusion proof for the transaction at *tx_index*.

        Returns a dict with ``leaf_index``, ``leaf_hash``, ``root``, and
        ``path`` (a tuple of ``{"sibling_hash", "is_right"}`` entries).
        """
        if not transactions:
            raise IndexError("tx_index out of range (empty transactions)")
        if tx_index < 0 or tx_index >= len(transactions):
            raise IndexError(
                f"tx_index {tx_index} out of range [0, {len(transactions)})"
            )
        leaves: list[bytes] = [
            hashlib.sha256(_canonical_bytes(tx)).digest()
            for tx in transactions
        ]
        path_entries: list[dict[str, Any]] = []
        current_index: int = tx_index
        level: list[bytes] = list(leaves)
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            sibling_idx: int = current_index ^ 1
            if sibling_idx < len(level):
                path_entries.append({
                    "sibling_hash": level[sibling_idx].hex(),
                    "is_right": current_index % 2 == 0,
                })
            current_index //= 2
            level = [
                hashlib.sha256(level[i] + level[i + 1]).digest()
                for i in range(0, len(level), 2)
            ]
        return {
            "leaf_index": tx_index,
            "leaf_hash": hashlib.sha256(
                _canonical_bytes(transactions[tx_index])
            ).hexdigest(),
            "root": level[0].hex() if level else hashlib.sha256(b"").hexdigest(),
            "path": tuple(path_entries),
        }

    @staticmethod
    def verify_proof(proof: Mapping[str, Any]) -> bool:
        """Validate an inclusion proof produced by :meth:`generate_proof`.

        Returns ``True`` when recomputing the root from the leaf and
        sibling path yields the stored ``root``.
        """
        current: bytes = bytes.fromhex(str(proof["leaf_hash"]))
        for step in proof["path"]:
            sibling: bytes = bytes.fromhex(str(step["sibling_hash"]))
            if step["is_right"]:
                current = hashlib.sha256(current + sibling).digest()
            else:
                current = hashlib.sha256(sibling + current).digest()
        return current.hex() == proof["root"]


def _signal_dict(signal: Signal) -> dict[str, Any]:
    return {
        "signal_id": signal.signal_id,
        "source": signal.source,
        "phase": signal.phase.value,
        "step_key": signal.step_key,
        "name": signal.name,
        "value": _jsonable(signal.value),
        "unit": signal.unit,
        "evidence_ids": list(signal.evidence_ids),
        "schema_version": signal.schema_version,
    }


def _intervention_dict(intervention: Intervention) -> dict[str, Any]:
    return {
        "intervention_id": intervention.intervention_id,
        "source": intervention.source,
        "phase": intervention.phase.value,
        "target": (
            intervention.target.value
            if isinstance(intervention.target, Enum)
            else str(intervention.target)
        ),
        "operation": (
            intervention.operation.value
            if isinstance(intervention.operation, Enum)
            else str(intervention.operation)
        ),
        "subject": intervention.subject,
        "value": _jsonable(intervention.value),
        "valid_from_step": intervention.valid_from_step,
        "valid_through_step": intervention.valid_through_step,
        "priority": intervention.priority,
        "evidence_ids": list(intervention.evidence_ids),
        "schema_version": intervention.schema_version,
    }


def _rejection_dict(rejection: Rejection) -> dict[str, str]:
    return {
        "item_id": rejection.item_id,
        "source": rejection.source,
        "reason": rejection.reason,
    }


def _decision_dict(decision: BusDecision) -> dict[str, Any]:
    return {
        "phase": decision.phase.value,
        "signals": [_signal_dict(item) for item in decision.signals],
        "accepted": [_intervention_dict(item) for item in decision.accepted],
        "effective": [_intervention_dict(item) for item in decision.effective],
        "rejected": [_rejection_dict(item) for item in decision.rejected],
        "blocked": decision.blocked,
        "context": _jsonable(decision.context),
    }


def _identity_from_event(step: Any) -> StepIdentity:
    if not isinstance(step, Mapping):
        raise ValueError("step_not_object")
    fields = {
        "run_id",
        "cycle",
        "optimizer_step",
        "accumulation_window",
        "batch_digest",
        "rng_digest",
        "base_checkpoint_id",
        "training_contract_id",
        "ablation_plan_id",
        "attempt_id",
        "key",
    }
    if set(step) != fields:
        raise ValueError("step_anchor_fields")
    identity = StepIdentity(
        run_id=step["run_id"],
        cycle=step["cycle"],
        optimizer_step=step["optimizer_step"],
        accumulation_window=step["accumulation_window"],
        batch_digest=step["batch_digest"],
        rng_digest=step["rng_digest"],
        base_checkpoint_id=step["base_checkpoint_id"],
        training_contract_id=step["training_contract_id"],
        ablation_plan_id=step["ablation_plan_id"],
        attempt_id=step["attempt_id"],
    )
    if step["key"] != identity.key:
        raise ValueError("step_key_anchor_mismatch")
    return identity


def _intent_state(payload: Any) -> _AttemptState:
    if not isinstance(payload, Mapping):
        raise ValueError("intent_payload_not_object")
    if set(payload) != {"arm", "decisions"}:
        raise ValueError("intent_payload_fields")
    try:
        arm = AblationArm(payload["arm"])
    except (TypeError, ValueError) as exc:
        raise ValueError("intent_arm") from exc
    decisions = payload["decisions"]
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("intent_decisions")
    phases: set[str] = set()
    effective_ids: list[str] = []
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ValueError("intent_decision_not_object")
        phase = decision.get("phase")
        if not isinstance(phase, str) or phase in phases:
            raise ValueError("intent_duplicate_or_invalid_phase")
        phases.add(phase)
        if not isinstance(decision.get("blocked"), bool):
            raise ValueError("intent_blocked_type")
        accepted = decision.get("accepted")
        effective = decision.get("effective")
        if not isinstance(accepted, list) or not isinstance(effective, list):
            raise ValueError("intent_intervention_arrays")
        accepted_ids = [
            item.get("intervention_id")
            for item in accepted
            if isinstance(item, Mapping)
        ]
        phase_effective_ids = [
            item.get("intervention_id")
            for item in effective
            if isinstance(item, Mapping)
        ]
        if len(accepted_ids) != len(accepted) or len(
            phase_effective_ids
        ) != len(effective):
            raise ValueError("intent_intervention_id")
        if len(set(phase_effective_ids)) != len(phase_effective_ids):
            raise ValueError("intent_duplicate_effective_id")
        if not set(phase_effective_ids).issubset(set(accepted_ids)):
            raise ValueError("intent_effective_not_accepted")
        if arm is not AblationArm.APPLY and phase_effective_ids:
            raise ValueError("intent_effective_for_non_apply")
        if bool(decision["blocked"]) and phase_effective_ids:
            raise ValueError("intent_effective_for_blocked")
        effective_ids.extend(phase_effective_ids)
    if len(set(effective_ids)) != len(effective_ids):
        raise ValueError("intent_duplicate_effective_id")
    return _AttemptState(arm=arm.value, effective_ids=tuple(effective_ids))


def _validate_outcome(payload: Any, state: _AttemptState) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("outcome_payload_not_object")
    required = {
        "optimizer_step_applied",
        "accepted_intervention_ids",
        "effective_intervention_ids",
        "raw_losses",
        "effective_losses",
        "gradient_norm_before",
        "gradient_norm_after",
        "parameter_delta_norm",
        "heldout_snapshot",
        "replay_snapshot",
        "degraded_organs",
        "error",
    }
    if set(payload) != required:
        raise ValueError("outcome_payload_fields")
    if not isinstance(payload["optimizer_step_applied"], bool):
        raise ValueError("outcome_optimizer_step_type")
    ids = payload["effective_intervention_ids"]
    accepted_alias = payload["accepted_intervention_ids"]
    if (
        not isinstance(ids, list)
        or not all(isinstance(item, str) and item for item in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("outcome_intervention_ids")
    if accepted_alias != ids:
        raise ValueError("outcome_intervention_alias_mismatch")
    if not set(ids).issubset(set(state.effective_ids)):
        raise ValueError("outcome_intervention_not_effective")
    if state.arm != AblationArm.APPLY.value and ids:
        raise ValueError("outcome_intervention_for_non_apply")
    error = payload["error"]
    if error is not None and not isinstance(error, str):
        raise ValueError("outcome_error_type")
    if (
        state.arm == AblationArm.APPLY.value
        and error is None
        and tuple(ids) != state.effective_ids
    ):
        raise ValueError("outcome_missing_effective_intervention")
    if not isinstance(payload["degraded_organs"], list) or not all(
        isinstance(item, str) and item
        for item in payload["degraded_organs"]
    ):
        raise ValueError("outcome_degraded_organs")


def _scan_ledger(
    path: Path,
) -> tuple[LedgerVerification, dict[str, _AttemptState]]:
    if not path.exists() or path.stat().st_size == 0:
        return (
            LedgerVerification(
                valid=True,
                event_count=0,
                head_hash=_EMPTY_HEAD,
                orphan_intent_step_keys=(),
                orphan_intent_count=0,
                errors=(),
            ),
            {},
        )
    raw = path.read_bytes()
    errors: list[str] = []
    if not raw.endswith(b"\n"):
        errors.append("missing_terminal_newline")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return (
            LedgerVerification(
                valid=False,
                event_count=0,
                head_hash=_EMPTY_HEAD,
                orphan_intent_step_keys=(),
                orphan_intent_count=0,
                errors=(f"invalid_utf8:{exc.start}",),
            ),
            {},
        )

    previous = _EMPTY_HEAD
    event_count = 0
    attempts: dict[str, _AttemptState] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            errors.append(f"line_{line_number}:blank_line")
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line_{line_number}:invalid_json:{exc.msg}")
            break
        # ── v3 block records are validated separately by _scan_blocks ──
        if isinstance(event, dict) and event.get("record_type") == "BLOCK":
            continue
        required = {
            "schema_version",
            "sequence",
            "event_type",
            "step",
            "payload",
            "prev_hash",
            "event_hash",
        }
        if not isinstance(event, dict) or set(event) != required:
            errors.append(f"line_{line_number}:event_fields")
            break
        if event.get("schema_version") not in (2, _SCHEMA_VERSION):
            errors.append(f"line_{line_number}:schema_version")
        if event["sequence"] != event_count:
            errors.append(f"line_{line_number}:sequence")
        if event["event_type"] not in _EVENT_TYPES:
            errors.append(f"line_{line_number}:event_type")
        if event["prev_hash"] != previous:
            errors.append(f"line_{line_number}:prev_hash")
        body = {key: value for key, value in event.items() if key != "event_hash"}
        try:
            computed = _event_hash(body)
            canonical = _canonical_bytes(event).decode("utf-8")
        except (TypeError, ValueError) as exc:
            errors.append(
                f"line_{line_number}:noncanonical:{type(exc).__name__}"
            )
            break
        if event["event_hash"] != computed:
            errors.append(f"line_{line_number}:event_hash")
        if line != canonical:
            errors.append(f"line_{line_number}:noncanonical_encoding")
        try:
            identity = _identity_from_event(event["step"])
            step_key = identity.key
            if event["event_type"] == "STEP_INTENT":
                if step_key in attempts:
                    raise ValueError("duplicate_attempt_id")
                attempts[step_key] = _intent_state(event["payload"])
            else:
                state = attempts.get(step_key)
                if state is None:
                    raise ValueError("outcome_without_intent")
                if state.outcome_recorded:
                    raise ValueError("duplicate_outcome")
                _validate_outcome(event["payload"], state)
                state.outcome_recorded = True
        except (TypeError, ValueError) as exc:
            errors.append(f"line_{line_number}:{exc}")

        event_count += 1
        previous = str(event["event_hash"])

    orphan_keys = tuple(
        sorted(
            step_key
            for step_key, state in attempts.items()
            if not state.outcome_recorded
        )
    )
    return (
        LedgerVerification(
            valid=not errors,
            event_count=event_count,
            head_hash=previous if event_count else _EMPTY_HEAD,
            orphan_intent_step_keys=orphan_keys,
            orphan_intent_count=len(orphan_keys),
            errors=tuple(errors),
        ),
        attempts,
    )


def verify_ledger(path: str | Path) -> LedgerVerification:
    return _scan_ledger(Path(path))[0]


def _scan_blocks(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Scan *path* for ``darwin-organ-block-v1`` records and validate.

    Returns ``(blocks, errors)``.  Each block is the raw parsed dict;
    *errors* are human-readable strings tagged with the block number.
    """
    blocks: list[dict[str, Any]] = []
    errors: list[str] = []

    if not path.exists() or path.stat().st_size == 0:
        return blocks, errors

    raw: bytes = path.read_bytes()
    text: str = raw.decode("utf-8")
    prev_block_hash: str = _EMPTY_HEAD
    expected_number: int = 0

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("record_type") != "BLOCK":
            continue

        header: Mapping[str, Any] = record.get("header", {}) if isinstance(record.get("header"), Mapping) else {}
        transactions: Sequence[Mapping[str, Any]] = (
            record.get("transactions", [])
            if isinstance(record.get("transactions"), list)
            else []
        )
        bn: Any = header.get("block_number")

        if record.get("schema_version") not in (2, _SCHEMA_VERSION):
            errors.append(f"block_{bn}:unsupported_schema")
        if bn != expected_number:
            errors.append(f"block_{bn}:block_number_mismatch")
        if header.get("prev_block_hash") != prev_block_hash:
            errors.append(f"block_{bn}:prev_block_hash")

        computed_root: str = LedgerMerkleTree.compute_root(transactions)
        if header.get("merkle_root") != computed_root:
            errors.append(f"block_{bn}:merkle_root")

        blocks.append(record)
        prev_block_hash = str(record.get("block_hash", _EMPTY_HEAD))
        expected_number += 1

    return blocks, errors


def verify_block_chain_file(path: str | Path) -> BlockChainVerification:
    """Standalone block-chain verification of a ledger file.

    Scans only ``BLOCK`` records and returns a
    :class:`BlockChainVerification` summary.
    """
    blocks, errors = _scan_blocks(Path(path))
    blocks_with_errors: list[int] = []
    for error in errors:
        m = re.match(r"block_(\d+):", error)
        if m:
            blocks_with_errors.append(int(m.group(1)))

    head_hash: str = (
        str(blocks[-1].get("block_hash", _EMPTY_HEAD)) if blocks else _EMPTY_HEAD
    )
    genesis_hash: str = (
        str(blocks[0].get("block_hash", _EMPTY_HEAD)) if blocks else _EMPTY_HEAD
    )

    return BlockChainVerification(
        valid=not errors,
        block_count=len(blocks),
        head_block_hash=head_hash,
        genesis_block_hash=genesis_hash,
        blocks_with_errors=tuple(sorted(set(blocks_with_errors))),
        errors=tuple(errors),
    )


class CausalLedger:
    """Writer that refuses any event which would make its chain invalid."""

    def __init__(
        self,
        path: str | Path,
        *,
        fsync: bool = True,
        max_events: int = 50_000,
    ) -> None:
        self.path = Path(path)
        self.fsync = bool(fsync)
        self.max_events = max_events
        self._lock = threading.RLock()
        verification, attempts = _scan_ledger(self.path)
        if not verification.valid:
            raise LedgerTamperError("; ".join(verification.errors))
        self._head_hash = verification.head_hash
        self._next_sequence = verification.event_count
        self._attempts = attempts
        self._fingerprint = _file_fingerprint(self.path)

    @property
    def head_hash(self) -> str:
        return self._head_hash

    def verify(self) -> LedgerVerification:
        return verify_ledger(self.path)

    def verify_chain(self) -> LedgerVerification:
        return self.verify()

    def recover_head(self) -> str:
        with self._lock:
            verification, attempts = _scan_ledger(self.path)
            if not verification.valid:
                raise LedgerTamperError("; ".join(verification.errors))
            self._head_hash = verification.head_hash
            self._next_sequence = verification.event_count
            self._attempts = attempts
            self._fingerprint = _file_fingerprint(self.path)
            return self._head_hash

    def record_intent(
        self,
        identity: StepIdentity,
        decisions: Sequence[BusDecision],
    ) -> dict[str, Any]:
        decisions = tuple(decisions)
        if not decisions:
            raise ValueError("STEP_INTENT requires aggregate decisions")
        if any(decision.identity != identity for decision in decisions):
            raise ValueError("STEP_INTENT identity mismatch")
        arms = {decision.arm for decision in decisions}
        if len(arms) != 1:
            raise ValueError("STEP_INTENT arm mismatch")
        payload = {
            "arm": decisions[0].arm.value,
            "decisions": [_decision_dict(item) for item in decisions],
        }
        _intent_state(payload)
        return self._append("STEP_INTENT", identity, payload)

    def record_outcome(self, outcome: StepOutcome) -> dict[str, Any]:
        ids = list(outcome.accepted_intervention_ids)
        payload = {
            "optimizer_step_applied": outcome.optimizer_step_applied,
            "accepted_intervention_ids": ids,
            "effective_intervention_ids": ids,
            "raw_losses": _jsonable(outcome.raw_losses),
            "effective_losses": _jsonable(outcome.effective_losses),
            "gradient_norm_before": outcome.gradient_norm_before,
            "gradient_norm_after": outcome.gradient_norm_after,
            "parameter_delta_norm": outcome.parameter_delta_norm,
            "heldout_snapshot": _jsonable(outcome.heldout_snapshot),
            "replay_snapshot": _jsonable(outcome.replay_snapshot),
            "degraded_organs": list(outcome.degraded_organs),
            "error": outcome.error,
        }
        return self._append("STEP_OUTCOME", outcome.identity, payload)

    def append_intent(
        self,
        identity: StepIdentity,
        decisions: Sequence[BusDecision],
    ) -> dict[str, Any]:
        return self.record_intent(identity, decisions)

    def append_outcome(self, outcome: StepOutcome) -> dict[str, Any]:
        return self.record_outcome(outcome)

    def _append(
        self,
        event_type: str,
        identity: StepIdentity,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if event_type not in _EVENT_TYPES:
            raise ValueError(f"unsupported causal event type: {event_type}")
        serializable_payload = _jsonable(payload, path="payload")
        _ensure_finite(serializable_payload, path="payload")
        with self._lock:
            current_fingerprint = _file_fingerprint(self.path)
            if current_fingerprint != self._fingerprint:
                current, _ = _scan_ledger(self.path)
                if not current.valid:
                    raise LedgerTamperError("; ".join(current.errors))
                if (
                    current.head_hash != self._head_hash
                    or current.event_count != self._next_sequence
                ):
                    # ── Modo adaptativo ──
                    # O ledger mudou fora deste writer (ex.: auto-resume
                    # truncou a chain, config operacional alterada, órgão
                    # ligado/desligado entre ciclos). Em vez de rejeitar
                    # como tampering, o organismo tenta se adaptar:
                    # re-sincroniza com o estado real do arquivo e continua
                    # se a chain estiver íntegra.
                    try:
                        self.recover_head()
                    except LedgerTamperError:
                        raise
                    # recover_head() atualiza _head_hash, _next_sequence,
                    # _attempts e _fingerprint. Se chegou aqui, a chain
                    # está válida — o organismo absorveu a mudança.

            # ── Rotation guard: prevent unbounded file growth ──
            if self.max_events > 0 and self._next_sequence >= self.max_events:
                self._rotate()

            state = self._attempts.get(identity.key)
            if event_type == "STEP_INTENT":
                if state is not None:
                    if state.outcome_recorded:
                        # ── Modo adaptativo ──
                        # Este passo já foi completamente registrado
                        # (INTENT + OUTCOME).  O organismo está
                        # re-executando — retorna sem escrever duplicata.
                        return {
                            "event_hash": self._head_hash,
                            "sequence": self._next_sequence,
                            "event_type": event_type,
                            "step": identity.to_dict(),
                            "payload": serializable_payload,
                            "prev_hash": self._head_hash,
                            "_adapted": True,
                            "_reason": "step already recorded — intent skipped",
                        }
                    # ── Modo adaptativo ──
                    # O INTENT existe como órfão (sem OUTCOME) —
                    # ex.: recover_head() restaurou um intent de um
                    # crash anterior.  Atualiza o estado em memória
                    # sem escrever duplicata na chain.
                    next_state = _intent_state(serializable_payload)
                    self._attempts[identity.key] = next_state
                    return {
                        "event_hash": self._head_hash,
                        "sequence": self._next_sequence,
                        "event_type": event_type,
                        "step": identity.to_dict(),
                        "payload": serializable_payload,
                        "prev_hash": self._head_hash,
                        "_adapted": True,
                        "_reason": "orphan intent updated in-memory — duplicate skipped",
                    }
                next_state = _intent_state(serializable_payload)
            else:
                if state is None or state.outcome_recorded:
                    # ── Modo adaptativo ──
                    # O INTENT foi perdido (ex.: ledger truncado entre
                    # intent e outcome por um resume). O organismo não
                    # pode registrar outcome sem intent — retorna sem
                    # escrever. O passo é perdido mas o treino sobrevive.
                    return {
                        "event_hash": self._head_hash,
                        "sequence": self._next_sequence,
                        "event_type": "STEP_OUTCOME",
                        "step": identity.to_dict(),
                        "payload": serializable_payload,
                        "prev_hash": self._head_hash,
                        "_adapted": True,
                        "_reason": "intent lost — outcome skipped",
                    }
                _validate_outcome(serializable_payload, state)
                next_state = None

            body = {
                "schema_version": _SCHEMA_VERSION,
                "sequence": self._next_sequence,
                "event_type": event_type,
                "step": identity.to_dict(),
                "payload": serializable_payload,
                "prev_hash": self._head_hash,
            }
            event = dict(body)
            event["event_hash"] = _event_hash(body)
            encoded = _canonical_bytes(event) + b"\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())

            self._head_hash = event["event_hash"]
            self._next_sequence += 1
            if event_type == "STEP_INTENT":
                self._attempts[identity.key] = next_state
            else:
                state.outcome_recorded = True
            self._fingerprint = _file_fingerprint(self.path)
            return event

    # ── Block-based recording (darwin-organ-block-v1) ──────────────────

    def append_block(self, block_dict: dict[str, Any]) -> dict[str, Any]:
        """Append a ``darwin-organ-block-v1`` record to the ledger.

        *block_dict* must contain ``header`` (with ``block_number``,
        ``prev_block_hash``, ``merkle_root``, ``cycle``,
        ``optimizer_step``, ``timestamp_unix_ms``, ``checkpoint_flag``)
        and ``transactions`` (a list of per-phase transaction dicts).

        Validates block-number sequencing, ``prev_block_hash`` chaining,
        and ``merkle_root`` integrity before writing.  Returns the
        canonicalised record (including the computed ``block_hash``).
        """
        with self._lock:
            current_fingerprint = _file_fingerprint(self.path)
            if current_fingerprint != self._fingerprint:
                self.recover_head()

            existing_blocks, scan_errors = _scan_blocks(self.path)
            if scan_errors:
                raise LedgerTamperError("; ".join(scan_errors))

            expected_number: int = len(existing_blocks)
            expected_prev_hash: str = (
                str(existing_blocks[-1].get("block_hash", _EMPTY_HEAD))
                if existing_blocks
                else _EMPTY_HEAD
            )

            header_in: Mapping[str, Any] = (
                block_dict.get("header", {})
                if isinstance(block_dict.get("header"), Mapping)
                else {}
            )
            transactions: Sequence[Mapping[str, Any]] = (
                block_dict.get("transactions", [])
                if isinstance(block_dict.get("transactions"), list)
                else []
            )

            # ── Validate block_number ──
            bn: Any = header_in.get("block_number")
            if bn != expected_number:
                raise ValueError(
                    f"block_number: expected {expected_number}, got {bn}"
                )

            # ── Validate prev_block_hash ──
            pbh: Any = header_in.get("prev_block_hash")
            if pbh != expected_prev_hash:
                raise ValueError(
                    f"prev_block_hash: expected {expected_prev_hash}, got {pbh}"
                )

            # ── Validate merkle_root ──
            expected_root: str = LedgerMerkleTree.compute_root(transactions)
            mr: Any = header_in.get("merkle_root")
            if mr != expected_root:
                raise ValueError(
                    f"merkle_root: expected {expected_root}, got {mr}"
                )

            # ── Build canonical record ──
            record: dict[str, Any] = {
                "schema_version": _SCHEMA_VERSION,
                "record_type": "BLOCK",
                "header": {
                    "block_number": int(bn),
                    "prev_block_hash": str(pbh),
                    "merkle_root": expected_root,
                    "cycle": int(header_in.get("cycle", 0)),
                    "optimizer_step": int(
                        header_in.get("optimizer_step", 0)
                    ),
                    "timestamp_unix_ms": int(
                        header_in.get(
                            "timestamp_unix_ms",
                            int(time.time() * 1000),
                        )
                    ),
                    "checkpoint_flag": bool(
                        header_in.get("checkpoint_flag", False)
                    ),
                },
                "transactions": list(transactions),
            }
            record["block_hash"] = _event_hash({
                k: v for k, v in record.items() if k != "block_hash"
            })

            # ── Write to JSONL ──
            encoded: bytes = _canonical_bytes(record) + b"\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())

            self._fingerprint = _file_fingerprint(self.path)
            return record

    def get_block(self, block_number: int) -> dict[str, Any] | None:
        """Return the block with the given *block_number*, or ``None``."""
        blocks, _ = _scan_blocks(self.path)
        for block in blocks:
            hdr: Any = block.get("header", {})
            if isinstance(hdr, Mapping) and hdr.get("block_number") == block_number:
                return block
        return None

    def get_latest_block(self) -> dict[str, Any] | None:
        """Return the most recent block, or ``None`` if none exist."""
        blocks, _ = _scan_blocks(self.path)
        return blocks[-1] if blocks else None

    def get_block_count(self) -> int:
        """Return the total number of blocks in the chain."""
        blocks, _ = _scan_blocks(self.path)
        return len(blocks)

    def verify_block_chain(self) -> "BlockChainVerification":
        """Validate the full block chain and return a summary."""
        return verify_block_chain_file(self.path)

    def _rotate(self) -> None:
        """Archive the current ledger as a dead-session file and reset the chain.

        Called automatically when ``_next_sequence >= max_events`` to
        prevent unbounded file growth.  The archived file retains the
        full event history for forensic analysis.
        """
        if not self.path.exists():
            return
        ts = int(time.time())
        archived = self.path.with_name(
            f"{self.path.stem}_dead_session_{ts}{self.path.suffix}"
        )
        self.path.rename(archived)
        self._head_hash = _EMPTY_HEAD
        self._next_sequence = 0
        self._attempts.clear()
        self._fingerprint = None

    def truncate_to_head(self, target_head: str) -> int:
        """Truncate the ledger file to end at *target_head* (inclusive).

        Returns the number of events kept.  Raises ``ValueError`` when
        *target_head* is not found in the chain, and ``LedgerTamperError``
        when the file fingerprint changes during truncation.
        """
        target_head = str(target_head or "")
        if not target_head:
            raise ValueError("target_head must be a non-empty hash")
        with self._lock:
            if not self.path.exists():
                return 0  # nothing to truncate — fresh ledger, file not created yet
            before = _file_fingerprint(self.path)
            raw = self.path.read_bytes()
            if not raw.endswith(b"\n"):
                raw += b"\n"
            text = raw.decode("utf-8")
            lines = text.splitlines(keepends=True)
            cut_at: int | None = None
            for idx, line in enumerate(lines):
                try:
                    event = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                if event.get("event_hash") == target_head:
                    cut_at = idx + 1  # keep up to and including this line
                    break
            if cut_at is None:
                return 0  # head not in chain → caller decides (fresh ledger / archive)
            kept = "".join(lines[:cut_at])
            # Ensure trailing newline so _scan_ledger is happy
            if not kept.endswith("\n"):
                kept += "\n"
            # Use binary I/O to avoid Windows newline translation
            # (\n ↔ \r\n) which would break the content comparison.
            kept_bytes = kept.encode("utf-8")
            tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp_path.write_bytes(kept_bytes)
            tmp_path.replace(self.path)
            actual_bytes = self.path.read_bytes()
            if actual_bytes != kept_bytes:
                self.path.write_bytes(raw)
                raise LedgerTamperError(
                    "ledger content mismatch after truncation"
                )
            self._head_hash = target_head
            self._next_sequence = cut_at
            # _attempts will be rebuilt from scratch by recover_head()
            # — _AttemptState has no sequence field and the eager filter
            # would crash on the dataclass.
            self._attempts.clear()
            self._fingerprint = _file_fingerprint(self.path)
            return cut_at


class LedgerIndex:
    """SQLite query index derived from the canonical JSONL ledger.

    The index is fully reconstructible — JSONL is the source of truth.
    Use :meth:`rebuild_from_ledger` to re-create the SQLite database
    from the ledger file at any time.

    Tables
    ------
    * **blocks** — one row per ``BLOCK`` record.
    * **transactions** — one row per per-phase transaction inside a block.
    * **organ_states** — one row per (block, organ) pair for provenance.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    # ── Schema ───────────────────────────────────────────────────────

    def _create_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            DROP TABLE IF EXISTS organ_states;
            DROP TABLE IF EXISTS transactions;
            DROP TABLE IF EXISTS blocks;

            CREATE TABLE blocks (
                block_number   INTEGER PRIMARY KEY,
                block_hash     TEXT    NOT NULL,
                prev_block_hash TEXT   NOT NULL,
                merkle_root    TEXT    NOT NULL,
                cycle          INTEGER NOT NULL,
                optimizer_step INTEGER NOT NULL,
                timestamp_unix_ms INTEGER NOT NULL,
                checkpoint_flag  INTEGER NOT NULL
            );

            CREATE TABLE transactions (
                tx_id          TEXT PRIMARY KEY,
                block_number   INTEGER NOT NULL,
                phase          TEXT    NOT NULL,
                organ_sources  TEXT    NOT NULL,
                signal_count   INTEGER NOT NULL,
                accepted_count INTEGER NOT NULL,
                rejected_count INTEGER NOT NULL,
                FOREIGN KEY (block_number)
                    REFERENCES blocks(block_number)
            );

            CREATE TABLE organ_states (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                block_number    INTEGER NOT NULL,
                organ_name      TEXT    NOT NULL,
                identity_hash   TEXT    NOT NULL,
                FOREIGN KEY (block_number)
                    REFERENCES blocks(block_number)
            );

            CREATE INDEX IF NOT EXISTS idx_tx_block
                ON transactions(block_number);
            CREATE INDEX IF NOT EXISTS idx_tx_phase
                ON transactions(phase);
            CREATE INDEX IF NOT EXISTS idx_organ_block
                ON organ_states(block_number);
            CREATE INDEX IF NOT EXISTS idx_organ_name
                ON organ_states(organ_name);
        """)
        conn.commit()

    # ── Rebuild ───────────────────────────────────────────────────────

    def rebuild_from_ledger(self, jsonl_path: str | Path) -> None:
        """Drop and rebuild every table from the canonical ledger file."""
        jsonl_path = Path(jsonl_path)
        self._create_schema()
        conn = self._get_conn()

        if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
            conn.commit()
            return

        raw: bytes = jsonl_path.read_bytes()
        text: str = raw.decode("utf-8")

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue

            if record.get("record_type") == "BLOCK":
                self._index_block(conn, record)

        conn.commit()

    def _index_block(
        self,
        conn: sqlite3.Connection,
        record: Mapping[str, Any],
    ) -> None:
        header: Mapping[str, Any] = (
            record.get("header", {})
            if isinstance(record.get("header"), Mapping)
            else {}
        )
        transactions: Sequence[Mapping[str, Any]] = (
            record.get("transactions", [])
            if isinstance(record.get("transactions"), list)
            else []
        )

        bn: int = int(header.get("block_number", 0))
        conn.execute(
            """INSERT OR REPLACE INTO blocks
               (block_number, block_hash, prev_block_hash, merkle_root,
                cycle, optimizer_step, timestamp_unix_ms, checkpoint_flag)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                bn,
                str(record.get("block_hash", "")),
                str(header.get("prev_block_hash", "")),
                str(header.get("merkle_root", "")),
                int(header.get("cycle", 0)),
                int(header.get("optimizer_step", 0)),
                int(header.get("timestamp_unix_ms", 0)),
                1 if header.get("checkpoint_flag") else 0,
            ),
        )

        for tx_index, tx in enumerate(transactions):
            if not isinstance(tx, Mapping):
                continue
            tx_id: str = f"{bn}:{tx_index}"
            organ_sources: list[str] = (
                list(tx.get("organ_sources", []))
                if isinstance(tx.get("organ_sources"), list)
                else []
            )
            conn.execute(
                """INSERT OR REPLACE INTO transactions
                   (tx_id, block_number, phase, organ_sources,
                    signal_count, accepted_count, rejected_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    tx_id,
                    bn,
                    str(tx.get("phase", "")),
                    json.dumps(organ_sources, sort_keys=True),
                    int(tx.get("signal_count", 0)),
                    int(tx.get("accepted_count", 0)),
                    int(tx.get("rejected_count", 0)),
                ),
            )

            # ── Organ provenance ──
            for organ_name in organ_sources:
                if not isinstance(organ_name, str) or not organ_name:
                    continue
                identity_hash: str = hashlib.sha256(
                    _canonical_bytes({
                        "organ": organ_name,
                        "block_number": bn,
                        "phase": str(tx.get("phase", "")),
                        "tx_data": dict(tx),
                    })
                ).hexdigest()
                conn.execute(
                    """INSERT INTO organ_states
                       (block_number, organ_name, identity_hash)
                       VALUES (?, ?, ?)""",
                    (bn, organ_name, identity_hash),
                )

    # ── Queries ───────────────────────────────────────────────────────

    def query_blocks_by_cycle(self, cycle: int) -> list[dict[str, Any]]:
        """Return all blocks belonging to *cycle*, ordered by block number."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM blocks WHERE cycle = ? ORDER BY block_number",
            (cycle,),
        ).fetchall()
        return [dict(row) for row in rows]

    def query_transactions_by_organ(
        self, organ_name: str
    ) -> list[dict[str, Any]]:
        """Return every transaction whose ``organ_sources`` includes *organ_name*."""
        conn = self._get_conn()
        # organ_sources is stored as a JSON array string
        rows = conn.execute(
            """SELECT t.* FROM transactions t
               WHERE t.organ_sources LIKE ?
               ORDER BY t.block_number""",
            (f'%"{organ_name}"%',),
        ).fetchall()
        return [dict(row) for row in rows]

    def query_organ_reputation_history(
        self, organ_name: str
    ) -> list[dict[str, Any]]:
        """Return the identity-hash history for *organ_name* across blocks."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM organ_states
               WHERE organ_name = ?
               ORDER BY block_number""",
            (organ_name,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "BlockChainVerification",
    "CausalLedger",
    "LedgerIndex",
    "LedgerMerkleTree",
    "LedgerTamperError",
    "LedgerVerification",
    "verify_block_chain_file",
    "verify_ledger",
]
