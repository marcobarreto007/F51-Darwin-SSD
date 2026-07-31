"""Regression test for the CausalLedger self-deadlock (FIXED).

History: ``CausalLedger.__init__`` used to create
``self._lock = threading.Lock()`` (non-reentrant). ``_append()`` acquires
``self._lock`` and, while still holding it, may detect that the on-disk file
fingerprint diverged from the in-memory fingerprint ("adaptive mode" — a
second writer touched the same file). When that happens it calls
``self.recover_head()``, which itself does ``with self._lock:`` — a second
acquisition of the same lock, by the same thread that already holds it. With
a plain ``threading.Lock`` that is a guaranteed self-deadlock in CPython: the
thread blocks forever on a lock it already owns and will never release.

Confirmed empirically (2026-07-24): this test failed to reproduce a hang
before the fix — the thread was still alive after the join timeout. The fix
was to change ``self._lock`` to ``threading.RLock()`` in
``causal_ledger.py``, which allows the same thread to safely re-enter the
lock from ``recover_head()``. This test now asserts the FIXED behavior
(the call returns within the timeout) so a future regression back to a
non-reentrant lock turns this test red again.

This test never touches anything under ``workspace/`` — it only uses a
pytest ``tmp_path`` fixture. It never imports torch and never starts any
training. The call is executed in a background ``daemon`` thread with a
bounded ``join(timeout=...)`` so this test process itself can never hang
indefinitely even if a future regression reintroduces the deadlock.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

from f51_darwin.organism.causal_bus import (
    AblationArm,
    Intervention,
    InterventionOperation,
    InterventionTarget,
    OrganCausalBus,
    Phase,
    Signal,
    StepIdentity,
)
from f51_darwin.organism.causal_ledger import CausalLedger

JOIN_TIMEOUT_SECONDS = 5.0


def _identity(*, optimizer_step: int) -> StepIdentity:
    return StepIdentity(
        run_id="deadlock-repro-run",
        cycle=1,
        optimizer_step=optimizer_step,
        accumulation_window=1,
        batch_digest=f"batch:{optimizer_step}",
        rng_digest=f"rng:{optimizer_step}",
        base_checkpoint_id="base:deadlock-repro",
        training_contract_id="contract:deadlock-repro",
        ablation_plan_id="plan:deadlock-repro",
    )


class _MinimalAdapter:
    """Smallest possible adapter that yields one signal and one intervention.

    Mirrors ``LedgerAdapter`` in ``src/tests/test_causal_ledger.py`` so the
    resulting ``BusDecision`` is a realistic, valid payload for
    ``record_intent``.
    """

    adapter_id = "deadlock-repro-adapter"
    version = "v1"

    def observe(self, identity, phase, context):
        return (
            Signal(
                signal_id=f"lm/{identity.optimizer_step}",
                source="deadlock-repro-adapter:v1",
                phase=phase,
                step_key=identity.key,
                name="lm_loss",
                value=context.get("lm_loss", 1.0),
                unit="nats/token",
            ),
        )

    def propose(self, identity, phase, signals, context):
        return (
            Intervention(
                intervention_id=f"aux-zero/{identity.optimizer_step}",
                source="deadlock-repro-adapter:v1",
                phase=phase,
                target=InterventionTarget.LOSS_TERM,
                operation=InterventionOperation.SET_SCALE,
                subject="aux",
                value=0.0,
                valid_from_step=identity.optimizer_step,
                valid_through_step=identity.optimizer_step,
                evidence_ids=(signals[0].signal_id,),
            ),
        )

    def feedback(self, outcome):
        return None


def test_second_writer_touching_shared_file_deadlocks_first_writer_append(
    tmp_path: Path,
) -> None:
    """Reproduce the self-deadlock via the AUTOMATIC path inside ``_append``.

    Deliberately does **not** call ``recover_head()`` manually anywhere —
    that would take the safe path exercised by
    ``test_public_aliases_and_recover_head_follow_an_external_valid_append``
    in ``src/tests/test_causal_ledger.py`` and would prove nothing about the
    dangerous branch inside ``_append`` itself.

    Sequence:
      1. ``writer_1`` and ``writer_2`` both open the same empty ledger file.
         Both observe ``head_hash == EMPTY`` and ``event_count == 0``.
      2. ``writer_2`` appends a STEP_INTENT. The file on disk now has one
         event; ``writer_2``'s in-memory fingerprint/head/sequence are
         updated accordingly. ``writer_1`` is not notified — its in-memory
         state is now stale.
      3. ``writer_1.record_intent(...)`` is called for a *different* step,
         from a background thread. Inside ``_append``:
           - ``with self._lock:`` is acquired (first acquisition — succeeds).
           - ``_file_fingerprint(self.path)`` no longer matches
             ``writer_1._fingerprint`` (because of step 2).
           - The freshly scanned ``current.head_hash``/``event_count`` do
             not match ``writer_1._head_hash``/``_next_sequence`` either
             (writer_2 wrote a real event) → "adaptive mode" branch taken.
           - ``self.recover_head()`` is invoked *while the lock from the
             outer ``with`` is still held* → ``recover_head()`` does
             ``with self._lock:`` again → non-reentrant re-acquisition by
             the same thread → permanent block.
      4. The call runs in a daemon thread with ``join(timeout=5)``. If the
         thread is still alive after the timeout, the deadlock reproduced.
    """
    path = tmp_path / "causal.jsonl"

    writer_1 = CausalLedger(path, fsync=False)
    writer_2 = CausalLedger(path, fsync=False)

    # Precondition: both writers agree on the empty-ledger baseline.
    assert writer_1.head_hash == "0" * 64
    assert writer_2.head_hash == "0" * 64

    # ── Step 2: writer_2 mutates the shared file behind writer_1's back ──
    bus_2 = OrganCausalBus(
        arm=AblationArm.SHADOW,
        adapters=(_MinimalAdapter(),),
    )
    identity_2 = _identity(optimizer_step=1)
    decision_2 = bus_2.decide(identity_2, Phase.PRE_LOSS, {"lm_loss": 1.0})
    writer_2.append_intent(identity_2, (decision_2,))

    # Sanity: the file changed under writer_1 without it knowing.
    assert writer_2.head_hash != "0" * 64
    assert writer_1.head_hash == "0" * 64  # stale — writer_1 has no idea

    # ── Step 3: drive writer_1 through the automatic recovery branch ──
    bus_1 = OrganCausalBus(
        arm=AblationArm.SHADOW,
        adapters=(_MinimalAdapter(),),
    )
    identity_1 = _identity(optimizer_step=2)
    decision_1 = bus_1.decide(identity_1, Phase.PRE_LOSS, {"lm_loss": 2.0})

    result_queue: "queue.Queue[object]" = queue.Queue()

    def _call_record_intent() -> None:
        try:
            outcome = writer_1.record_intent(identity_1, (decision_1,))
            result_queue.put(("returned", outcome))
        except BaseException as exc:  # noqa: BLE001 - want any outcome, not a crash of the test
            result_queue.put(("raised", exc))

    worker = threading.Thread(
        target=_call_record_intent,
        name="writer_1-record_intent",
        daemon=True,  # never blocks pytest process exit even if deadlocked
    )
    worker.start()
    worker.join(timeout=JOIN_TIMEOUT_SECONDS)

    deadlocked = worker.is_alive()
    assert not deadlocked, (
        "DEADLOCK DETECTED: writer_1.record_intent() did not return "
        f"within {JOIN_TIMEOUT_SECONDS}s. Thread is blocked inside "
        "_append()'s automatic recover_head() call."
    )

    outcome_kind, payload = result_queue.get_nowait()
    assert outcome_kind == "returned", f"expected successful return, got {outcome_kind}: {payload!r}"
    assert isinstance(payload, dict) and payload.get("event_type") == "STEP_INTENT"
