from __future__ import annotations

import contextvars
import json
import multiprocessing as mp
import os
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest

from f51_darwin.circuits.identity import canonical_json_bytes
from f51_darwin.circuits.ledger import CircuitLedger, LedgerError
import f51_darwin.circuits.ledger as ledger_module
import f51_darwin.circuits.pointer as pointer_module
from f51_darwin.circuits.pointer import (
    CheckpointCandidate,
    PointerConflict,
    PointerError,
    publish_active_pointer,
    read_active_pointer,
)


def _event(number: int) -> dict[str, object]:
    return {
        "kind": "circuit_lifecycle",
        "number": number,
        "event_count": number + 1,
        "event_tail_sha256": f"{number + 1:064x}",
        "transaction_manifest_sha256": f"{number + 11:064x}",
    }


def _ledger_with_three_records(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    path = tmp_path / "ledger.jsonl"
    ledger = CircuitLedger(path)
    records = [ledger.append(_event(number)).to_dict() for number in range(3)]
    return path, records


def test_ledger_appends_canonical_hash_chain_and_externalizes_transaction_tail(
    tmp_path: Path,
) -> None:
    path, records = _ledger_with_three_records(tmp_path)

    assert records[0]["previous_record_sha256"] == "0" * 64
    assert records[1]["previous_record_sha256"] == records[0]["record_sha256"]
    assert records[2]["previous_record_sha256"] == records[1]["record_sha256"]
    assert (
        CircuitLedger(
            path,
            expected_tail_sha256=records[2]["record_sha256"],
            expected_record_count=3,
        ).verify()
        == records[2]["record_sha256"]
    )
    assert path.read_bytes() == b"".join(
        canonical_json_bytes(record) + b"\n" for record in records
    )


@pytest.mark.parametrize("tamper", ["delete", "reorder", "edit", "truncate"])
def test_ledger_rejects_deleted_reordered_edited_or_truncated_records(
    tmp_path: Path, tamper: str
) -> None:
    path, records = _ledger_with_three_records(tmp_path)
    lines = path.read_bytes().splitlines(keepends=True)
    if tamper == "delete":
        lines.pop(1)
    elif tamper == "reorder":
        lines[0], lines[1] = lines[1], lines[0]
    elif tamper == "edit":
        payload = json.loads(lines[1])
        payload["event"]["number"] = 99
        lines[1] = canonical_json_bytes(payload) + b"\n"
    else:
        lines[-1] = lines[-1][:-7]
    path.write_bytes(b"".join(lines))

    with pytest.raises(LedgerError):
        CircuitLedger(
            path,
            expected_tail_sha256=records[-1]["record_sha256"],
            expected_record_count=3,
        ).verify()


def test_ledger_rejects_canonical_terminal_truncation_against_external_anchor(
    tmp_path: Path,
) -> None:
    path, records = _ledger_with_three_records(tmp_path)
    path.write_bytes(b"".join(path.read_bytes().splitlines(keepends=True)[:-1]))

    with pytest.raises(LedgerError, match="tail"):
        CircuitLedger(
            path,
            expected_tail_sha256=records[-1]["record_sha256"],
            expected_record_count=3,
        ).verify()


def test_ledger_rejects_canonical_alternate_terminal_history(
    tmp_path: Path,
) -> None:
    path, records = _ledger_with_three_records(tmp_path)
    lines = path.read_bytes().splitlines(keepends=True)
    alternate = json.loads(lines[-1])
    alternate["event"]["number"] = 999
    unsigned = {
        key: value for key, value in alternate.items() if key != "record_sha256"
    }
    from f51_darwin.circuits.identity import canonical_sha256

    alternate["record_sha256"] = canonical_sha256(unsigned)
    lines[-1] = canonical_json_bytes(alternate) + b"\n"
    path.write_bytes(b"".join(lines))

    with pytest.raises(LedgerError, match="tail"):
        CircuitLedger(
            path,
            expected_tail_sha256=records[-1]["record_sha256"],
            expected_record_count=3,
        ).verify()


def test_fresh_unanchored_nonempty_ledger_cannot_verify_or_append(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    CircuitLedger(path).append(_event(0))
    fresh = CircuitLedger(path)

    with pytest.raises(LedgerError, match="trusted.*tail.*count|anchor"):
        fresh.verify()
    with pytest.raises(LedgerError, match="trusted.*tail.*count|anchor"):
        fresh.append(_event(1))


def test_competing_anchored_appenders_serialize_and_only_one_wins(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    first = CircuitLedger(path).append(_event(0))
    appenders = [
        CircuitLedger(
            path,
            expected_tail_sha256=first.record_sha256,
            expected_record_count=1,
        )
        for _ in range(2)
    ]

    def append(index: int) -> str:
        try:
            return appenders[index].append(_event(index + 1)).record_sha256
        except LedgerError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(append, range(2)))

    assert results.count("rejected") == 1
    winner = next(result for result in results if result != "rejected")
    assert (
        CircuitLedger(
            path,
            expected_tail_sha256=winner,
            expected_record_count=2,
        ).verify()
        == winner
    )


def test_ledger_append_fsyncs_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[Path] = []
    monkeypatch.setattr(
        ledger_module,
        "_fsync_directory",
        lambda path: observed.append(Path(path)),
    )

    CircuitLedger(tmp_path / "ledger.jsonl").append(_event(0))

    assert observed == [tmp_path]


def test_ledger_rejects_symlink_lock_without_touching_external_target(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external.lock"
    external.write_bytes(b"external")
    lock = tmp_path / ".ledger.jsonl.lock"
    try:
        lock.symlink_to(external)
    except OSError:
        pytest.skip("file symlink creation is unavailable")

    with pytest.raises(LedgerError, match="symlink|reparse|lock"):
        CircuitLedger(tmp_path / "ledger.jsonl").append(_event(0))

    assert external.read_bytes() == b"external"
    assert not (tmp_path / "ledger.jsonl").exists()


def _checkpoint(path: Path, content: bytes) -> str:
    import hashlib

    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _candidate(
    path: Path,
    sha256: str,
    ledger_path: Path,
    ledger_sha256: str,
    ledger_count: int,
):
    return CheckpointCandidate(
        path=path,
        sha256=sha256,
        ledger_path=ledger_path,
        ledger_sha256=ledger_sha256,
        ledger_count=ledger_count,
    )


def test_pointer_publication_is_compare_and_swap_and_binds_ledger_tail(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = CircuitLedger(ledger_path)
    ledger_a = ledger.append(_event(0)).record_sha256
    checkpoint_a = tmp_path / "a.pt"
    sha_a = _checkpoint(checkpoint_a, b"checkpoint-a")
    pointer_root = tmp_path / "pointers"
    pointer_root.mkdir()

    first = publish_active_pointer(
        pointer_root,
        "0" * 64,
        _candidate(checkpoint_a, sha_a, ledger_path, ledger_a, 1),
    )
    assert first.checkpoint_sha256 == sha_a
    assert first.ledger_sha256 == ledger_a

    ledger_b = ledger.append(_event(1)).record_sha256
    checkpoint_b = tmp_path / "b.pt"
    sha_b = _checkpoint(checkpoint_b, b"checkpoint-b")
    second = publish_active_pointer(
        pointer_root,
        sha_a,
        _candidate(checkpoint_b, sha_b, ledger_path, ledger_b, 2),
    )
    assert second.parent_checkpoint_sha256 == sha_a

    ledger_c = ledger.append(_event(2)).record_sha256
    checkpoint_c = tmp_path / "c.pt"
    sha_c = _checkpoint(checkpoint_c, b"checkpoint-c")
    with pytest.raises(PointerConflict):
        publish_active_pointer(
            pointer_root,
            sha_a,
            _candidate(checkpoint_c, sha_c, ledger_path, ledger_c, 3),
        )

    assert read_active_pointer(pointer_root) == second


def test_pointer_replace_failure_preserves_original_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = CircuitLedger(ledger_path)
    ledger_a = ledger.append(_event(0)).record_sha256
    checkpoint_a = tmp_path / "a.pt"
    sha_a = _checkpoint(checkpoint_a, b"checkpoint-a")
    root = tmp_path / "pointers"
    root.mkdir()
    first = publish_active_pointer(
        root,
        "0" * 64,
        _candidate(checkpoint_a, sha_a, ledger_path, ledger_a, 1),
    )

    ledger_b = ledger.append(_event(1)).record_sha256
    checkpoint_b = tmp_path / "b.pt"
    sha_b = _checkpoint(checkpoint_b, b"checkpoint-b")
    pointer_bytes = (root / "active.json").read_bytes()

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("injected crash before replace")

    monkeypatch.setattr("f51_darwin.circuits.pointer.os.replace", fail_replace)
    with pytest.raises(PointerError, match="replace"):
        publish_active_pointer(
            root,
            sha_a,
            _candidate(checkpoint_b, sha_b, ledger_path, ledger_b, 2),
        )

    assert (root / "active.json").read_bytes() == pointer_bytes
    assert read_active_pointer(root) == first
    assert not list(root.glob(".active.json.*.tmp"))


def test_pointer_rejects_checkpoint_or_ledger_mismatch(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_sha = CircuitLedger(ledger_path).append(_event(0)).record_sha256
    checkpoint = tmp_path / "candidate.pt"
    checkpoint_sha = _checkpoint(checkpoint, b"candidate")
    root = tmp_path / "pointers"
    root.mkdir()

    with pytest.raises(PointerError, match="checkpoint"):
        publish_active_pointer(
            root,
            "0" * 64,
            _candidate(checkpoint, "f" * 64, ledger_path, ledger_sha, 1),
        )
    with pytest.raises(PointerError, match="ledger"):
        publish_active_pointer(
            root,
            "0" * 64,
            _candidate(checkpoint, checkpoint_sha, ledger_path, "f" * 64, 1),
        )


def test_pointer_rejects_candidate_ledger_that_does_not_descend_from_current(
    tmp_path: Path,
) -> None:
    ledger_a_path = tmp_path / "ledger-a.jsonl"
    ledger_a_sha = CircuitLedger(ledger_a_path).append(_event(0)).record_sha256
    checkpoint_a = tmp_path / "a.pt"
    sha_a = _checkpoint(checkpoint_a, b"a")
    root = tmp_path / "pointers"
    root.mkdir()
    publish_active_pointer(
        root,
        "0" * 64,
        _candidate(checkpoint_a, sha_a, ledger_a_path, ledger_a_sha, 1),
    )

    alternate_path = tmp_path / "alternate.jsonl"
    alternate_sha = CircuitLedger(alternate_path).append(_event(99)).record_sha256
    checkpoint_b = tmp_path / "b.pt"
    sha_b = _checkpoint(checkpoint_b, b"b")

    with pytest.raises(PointerError, match="ancestor|prefix|history"):
        publish_active_pointer(
            root,
            sha_a,
            _candidate(checkpoint_b, sha_b, alternate_path, alternate_sha, 1),
        )

    assert read_active_pointer(root).checkpoint_sha256 == sha_a


def test_pointer_acquires_lock_before_validating_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_sha = CircuitLedger(ledger_path).append(_event(0)).record_sha256
    checkpoint = tmp_path / "candidate.pt"
    checkpoint_sha = _checkpoint(checkpoint, b"candidate")
    root = tmp_path / "pointers"
    root.mkdir()
    locked = False
    real_lock = pointer_module._publication_lock
    real_validate = pointer_module._validate_checkpoint

    @contextmanager
    def observed_lock(
        path: Path,
        ledger_lock: Path | None = None,
        callback: object = None,
    ):
        nonlocal locked
        with real_lock(path, ledger_lock, callback):
            locked = True
            try:
                yield
            finally:
                locked = False

    def guarded_validate(path: Path, sha256: str) -> Path:
        assert locked
        return real_validate(path, sha256)

    monkeypatch.setattr(pointer_module, "_publication_lock", observed_lock)
    monkeypatch.setattr(pointer_module, "_validate_checkpoint", guarded_validate)

    publish_active_pointer(
        root,
        "0" * 64,
        _candidate(checkpoint, checkpoint_sha, ledger_path, ledger_sha, 1),
    )


def test_pointer_revalidates_candidate_immediately_before_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_sha = CircuitLedger(ledger_path).append(_event(0)).record_sha256
    checkpoint = tmp_path / "candidate.pt"
    checkpoint_sha = _checkpoint(checkpoint, b"candidate")
    root = tmp_path / "pointers"
    root.mkdir()
    real_validate = pointer_module._validate_checkpoint
    candidate_validations = 0

    def mutate_before_second_validation(path: Path, sha256: str) -> Path:
        nonlocal candidate_validations
        if path == checkpoint:
            candidate_validations += 1
            if candidate_validations == 2:
                checkpoint.write_bytes(b"mutated-after-first-validation")
        return real_validate(path, sha256)

    monkeypatch.setattr(
        pointer_module,
        "_validate_checkpoint",
        mutate_before_second_validation,
    )

    with pytest.raises(PointerError, match="checkpoint"):
        publish_active_pointer(
            root,
            "0" * 64,
            _candidate(checkpoint, checkpoint_sha, ledger_path, ledger_sha, 1),
        )
    assert not (root / "active.json").exists()


def test_pointer_replace_fsyncs_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_sha = CircuitLedger(ledger_path).append(_event(0)).record_sha256
    checkpoint = tmp_path / "candidate.pt"
    checkpoint_sha = _checkpoint(checkpoint, b"candidate")
    root = tmp_path / "pointers"
    root.mkdir()
    observed: list[Path] = []
    monkeypatch.setattr(
        pointer_module,
        "_fsync_directory",
        lambda path: observed.append(Path(path)),
    )

    publish_active_pointer(
        root,
        "0" * 64,
        _candidate(checkpoint, checkpoint_sha, ledger_path, ledger_sha, 1),
    )

    assert observed == [root]


def test_pointer_rejects_symlink_lock_without_touching_external_target(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_sha = CircuitLedger(ledger_path).append(_event(0)).record_sha256
    checkpoint = tmp_path / "candidate.pt"
    checkpoint_sha = _checkpoint(checkpoint, b"candidate")
    root = tmp_path / "pointers"
    root.mkdir()
    external = tmp_path / "external.lock"
    external.write_bytes(b"external")
    try:
        (root / ".active.lock").symlink_to(external)
    except OSError:
        pytest.skip("file symlink creation is unavailable")

    with pytest.raises(PointerError, match="symlink|reparse|lock"):
        publish_active_pointer(
            root,
            "0" * 64,
            _candidate(checkpoint, checkpoint_sha, ledger_path, ledger_sha, 1),
        )

    assert external.read_bytes() == b"external"
    assert not (root / "active.json").exists()


def _assert_alias_publication_rejected_promptly(
    *,
    root: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    ledger_path: Path,
    ledger_sha256: str,
) -> None:
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        from f51_darwin.circuits.pointer import (
            CheckpointCandidate,
            PointerError,
            publish_active_pointer,
        )

        root, checkpoint, checkpoint_sha, ledger, ledger_sha = sys.argv[1:]
        candidate = CheckpointCandidate(
            path=Path(checkpoint),
            sha256=checkpoint_sha,
            ledger_path=Path(ledger),
            ledger_sha256=ledger_sha,
            ledger_count=1,
        )
        try:
            publish_active_pointer(Path(root), "0" * 64, candidate)
        except PointerError as exc:
            if "alias" not in str(exc) and "collid" not in str(exc):
                raise
            print("REJECTED")
        else:
            raise SystemExit("publication unexpectedly succeeded")
        """
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(root),
            str(checkpoint),
            checkpoint_sha256,
            str(ledger_path),
            ledger_sha256,
        ],
        cwd=Path.cwd(),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "REJECTED"


def test_pointer_rejects_hardlink_alias_of_held_pointer_lock_without_hanging(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_sha = CircuitLedger(ledger_path).append(_event(0)).record_sha256
    root = tmp_path / "pointers"
    root.mkdir()
    try:
        os.link(tmp_path / ".ledger.jsonl.lock", root / ".active.lock")
    except OSError:
        pytest.skip("hardlink creation is unavailable")
    checkpoint = tmp_path / "candidate.pt"
    checkpoint_sha = _checkpoint(checkpoint, b"candidate")

    _assert_alias_publication_rejected_promptly(
        root=root,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        ledger_path=ledger_path,
        ledger_sha256=ledger_sha,
    )

    assert not (root / "active.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows case alias regression")
def test_pointer_rejects_windows_case_alias_of_held_pointer_lock(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pointers"
    root.mkdir()
    ledger_path = root / "ACTIVE"
    ledger_sha = CircuitLedger(ledger_path).append(_event(0)).record_sha256
    checkpoint = tmp_path / "candidate.pt"
    checkpoint_sha = _checkpoint(checkpoint, b"candidate")

    _assert_alias_publication_rejected_promptly(
        root=root,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        ledger_path=ledger_path,
        ledger_sha256=ledger_sha,
    )

    assert not (root / "active.json").exists()


def _published_a_and_candidate_b(
    tmp_path: Path,
) -> tuple[
    Path,
    CircuitLedger,
    object,
    bytes,
    Path,
    str,
    str,
]:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = CircuitLedger(ledger_path)
    ledger_a = ledger.append(_event(0)).record_sha256
    checkpoint_a = tmp_path / "a.pt"
    sha_a = _checkpoint(checkpoint_a, b"checkpoint-a")
    root = tmp_path / "pointers"
    root.mkdir()
    pointer_a = publish_active_pointer(
        root,
        "0" * 64,
        _candidate(checkpoint_a, sha_a, ledger_path, ledger_a, 1),
    )
    pointer_a_bytes = (root / "active.json").read_bytes()
    ledger_b = ledger.append(_event(1)).record_sha256
    checkpoint_b = tmp_path / "b.pt"
    sha_b = _checkpoint(checkpoint_b, b"checkpoint-b")
    return (
        root,
        ledger,
        pointer_a,
        pointer_a_bytes,
        checkpoint_b,
        sha_b,
        ledger_b,
    )


def test_pointer_rejects_mutation_immediately_after_final_checkpoint_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        root,
        ledger,
        pointer_a,
        pointer_a_bytes,
        checkpoint_b,
        sha_b,
        ledger_b,
    ) = _published_a_and_candidate_b(tmp_path)
    real_validate = pointer_module._validate_checkpoint
    candidate_validations = 0

    def validate_then_mutate(path: Path, sha256: str) -> Path:
        nonlocal candidate_validations
        validated = real_validate(path, sha256)
        if path == checkpoint_b:
            candidate_validations += 1
            if candidate_validations == 2:
                checkpoint_b.write_bytes(b"poison-after-final-validation")
        return validated

    monkeypatch.setattr(pointer_module, "_validate_checkpoint", validate_then_mutate)

    with pytest.raises(PointerError, match="checkpoint|changed|identity"):
        publish_active_pointer(
            root,
            pointer_a.checkpoint_sha256,
            _candidate(
                checkpoint_b,
                sha_b,
                ledger.path,
                ledger_b,
                2,
            ),
        )

    assert (root / "active.json").read_bytes() == pointer_a_bytes
    assert read_active_pointer(root) == pointer_a


def test_pointer_restores_exact_old_bytes_after_post_replace_poison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        root,
        ledger,
        pointer_a,
        pointer_a_bytes,
        checkpoint_b,
        sha_b,
        ledger_b,
    ) = _published_a_and_candidate_b(tmp_path)
    real_replace = pointer_module.os.replace
    pointer_replace_count = 0

    def replace_then_poison(source: object, destination: object) -> None:
        nonlocal pointer_replace_count
        real_replace(source, destination)
        if Path(destination) == root / "active.json":
            pointer_replace_count += 1
            if pointer_replace_count == 1:
                checkpoint_b.write_bytes(b"poison-after-pointer-replace")

    monkeypatch.setattr(pointer_module.os, "replace", replace_then_poison)

    with pytest.raises(PointerError, match="checkpoint|publication"):
        publish_active_pointer(
            root,
            pointer_a.checkpoint_sha256,
            _candidate(
                checkpoint_b,
                sha_b,
                ledger.path,
                ledger_b,
                2,
            ),
        )

    assert pointer_replace_count == 2
    assert (root / "active.json").read_bytes() == pointer_a_bytes
    assert read_active_pointer(root) == pointer_a


def test_pointer_rejects_same_hash_inode_replacement_before_pointer_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        root,
        ledger,
        pointer_a,
        pointer_a_bytes,
        checkpoint_b,
        sha_b,
        ledger_b,
    ) = _published_a_and_candidate_b(tmp_path)
    real_validate = pointer_module._validate_checkpoint
    candidate_validations = 0

    def validate_then_replace_inode(path: Path, sha256: str) -> Path:
        nonlocal candidate_validations
        validated = real_validate(path, sha256)
        if path == checkpoint_b:
            candidate_validations += 1
            if candidate_validations == 2:
                replacement = tmp_path / "replacement.pt"
                replacement.write_bytes(checkpoint_b.read_bytes())
                os.replace(replacement, checkpoint_b)
        return validated

    monkeypatch.setattr(
        pointer_module,
        "_validate_checkpoint",
        validate_then_replace_inode,
    )

    with pytest.raises(PointerError, match="identity|changed"):
        publish_active_pointer(
            root,
            pointer_a.checkpoint_sha256,
            _candidate(
                checkpoint_b,
                sha_b,
                ledger.path,
                ledger_b,
                2,
            ),
        )

    assert (root / "active.json").read_bytes() == pointer_a_bytes
    assert read_active_pointer(root) == pointer_a


def test_lock_namespace_replacement_cannot_create_concurrent_critical_sections(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "namespace.lock"
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        from f51_darwin.circuits.ledger import LedgerError, _exclusive_lock

        mode, lock_path = sys.argv[1:]
        try:
            with _exclusive_lock(Path(lock_path)):
                print("ENTERED", flush=True)
                if mode == "hold":
                    sys.stdin.readline()
            print("RELEASED", flush=True)
        except LedgerError as exc:
            print(f"FAILED_CLOSED:{exc}", flush=True)
        """
    )
    command = [sys.executable, "-c", script]
    holder = subprocess.Popen(
        [*command, "hold", str(lock_path)],
        cwd=Path.cwd(),
        env=os.environ.copy(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    contender: subprocess.Popen[str] | None = None
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "ENTERED"
        if os.name == "nt":
            with pytest.raises(OSError):
                lock_path.unlink()
        else:
            lock_path.unlink()
            lock_path.write_bytes(b"replacement")

        contender = subprocess.Popen(
            [*command, "once", str(lock_path)],
            cwd=Path.cwd(),
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.5)
        assert contender.poll() is None, (
            "replacement lock admitted a concurrent critical section"
        )

        assert holder.stdin is not None
        holder.stdin.write("\n")
        holder.stdin.flush()
        holder_output, holder_error = holder.communicate(timeout=5)
        contender_output, contender_error = contender.communicate(timeout=5)
        assert holder.returncode == 0, holder_error
        assert contender.returncode == 0, contender_error
        assert "RELEASED" in holder_output or "FAILED_CLOSED" in holder_output
        assert (
            "ENTERED" in contender_output
            or "FAILED_CLOSED" in contender_output
        )
    finally:
        for process in (holder, contender):
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate()


def test_nested_locks_in_same_parent_reuse_namespace_without_hanging(
    tmp_path: Path,
) -> None:
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        from f51_darwin.circuits.ledger import _exclusive_locks

        parent = Path(sys.argv[1])
        with _exclusive_locks(
            [parent / "active.lock", parent / "ledger.lock"]
        ):
            print("COORDINATED", flush=True)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path.cwd(),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "COORDINATED"


def test_pointer_restores_a_when_final_lock_path_identity_check_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        root,
        ledger,
        pointer_a,
        pointer_a_bytes,
        checkpoint_b,
        sha_b,
        ledger_b,
    ) = _published_a_and_candidate_b(tmp_path)
    real_verify = getattr(ledger_module, "_verify_lock_path_identity", None)
    verification_count = 0

    def fail_active_release(path: Path, expected: tuple[int, int]) -> None:
        nonlocal verification_count
        verification_count += 1
        if real_verify is not None:
            real_verify(path, expected)
        if verification_count == 4:
            raise LedgerError("lock path identity changed after pointer mutation")

    monkeypatch.setattr(
        ledger_module,
        "_verify_lock_path_identity",
        fail_active_release,
        raising=False,
    )

    with pytest.raises(PointerError, match="lock|restor|publication"):
        publish_active_pointer(
            root,
            pointer_a.checkpoint_sha256,
            _candidate(
                checkpoint_b,
                sha_b,
                ledger.path,
                ledger_b,
                2,
            ),
        )

    assert verification_count >= 4
    assert (root / "active.json").read_bytes() == pointer_a_bytes
    assert read_active_pointer(root) == pointer_a


def test_coordinator_rejects_nested_acquisition_from_copied_context(
    tmp_path: Path,
) -> None:
    def nested() -> None:
        with ledger_module._exclusive_lock(tmp_path / "nested.lock"):
            pass

    with ledger_module._exclusive_lock(tmp_path / "outer.lock"):
        with pytest.raises(LedgerError, match="nested|reentrant|coordinator"):
            contextvars.copy_context().run(nested)


def test_coordinator_acquires_high_low_high_without_ancestor_state(
    tmp_path: Path,
) -> None:
    high = tmp_path / "z"
    low = tmp_path / "a"
    high.mkdir()
    low.mkdir()
    paths = [
        high / "first.lock",
        low / "middle.lock",
        high / "last.lock",
    ]

    with ledger_module._exclusive_locks(paths):
        assert all(path.exists() for path in paths)


def test_coordinator_blocks_contender_until_release_callback_completes(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "callback.lock"
    allow_callback = tmp_path / "allow-callback"
    holder_script = textwrap.dedent(
        """
        import sys
        import time
        from pathlib import Path
        import f51_darwin.circuits.ledger as ledger

        lock_path = Path(sys.argv[1])
        allow = Path(sys.argv[2])
        real_verify = ledger._verify_lock_path_identity
        count = 0

        def fail_release(path, identity):
            global count
            count += 1
            real_verify(path, identity)
            if count == 2:
                raise ledger.LedgerError("injected release integrity failure")

        def callback():
            print("CALLBACK_START", flush=True)
            while not allow.exists():
                time.sleep(0.01)
            print("CALLBACK_DONE", flush=True)

        ledger._verify_lock_path_identity = fail_release
        try:
            with ledger._exclusive_locks(
                [lock_path],
                on_release_integrity_error=callback,
            ):
                print("HELD", flush=True)
                sys.stdin.readline()
        except ledger.LedgerError:
            print("FAILED_CLOSED", flush=True)
        """
    )
    contender_script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from f51_darwin.circuits.ledger import _exclusive_lock

        with _exclusive_lock(Path(sys.argv[1])):
            print("ENTERED", flush=True)
        """
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_script, str(lock_path), str(allow_callback)],
        cwd=Path.cwd(),
        env=os.environ.copy(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    contender: subprocess.Popen[str] | None = None
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "HELD"
        contender = subprocess.Popen(
            [sys.executable, "-c", contender_script, str(lock_path)],
            cwd=Path.cwd(),
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert holder.stdin is not None
        holder.stdin.write("\n")
        holder.stdin.flush()
        assert holder.stdout.readline().strip() == "CALLBACK_START"
        time.sleep(0.5)
        assert contender.poll() is None
        allow_callback.write_text("continue", encoding="utf-8")
        holder_output, holder_error = holder.communicate(timeout=5)
        contender_output, contender_error = contender.communicate(timeout=5)
        assert holder.returncode == 0, holder_error
        assert contender.returncode == 0, contender_error
        assert "CALLBACK_DONE" in holder_output
        assert "FAILED_CLOSED" in holder_output
        assert contender_output.strip() == "ENTERED"
    finally:
        for process in (holder, contender):
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate()


@pytest.mark.skipif(os.name == "nt", reason="POSIX at-fork reset regression")
def test_coordinator_at_fork_child_starts_with_clean_reentrancy_state(
    tmp_path: Path,
) -> None:
    context = mp.get_context("fork")
    child_parent = tmp_path / "child"
    child_parent.mkdir()
    queue = context.Queue()

    def acquire_in_child() -> None:
        try:
            with ledger_module._exclusive_lock(child_parent / "child.lock"):
                queue.put(True)
        except LedgerError:
            queue.put(False)

    with ledger_module._exclusive_lock(tmp_path / "parent.lock"):
        child = context.Process(target=acquire_in_child)
        child.start()
        child.join(5)
        assert child.exitcode == 0
        assert queue.get(timeout=1) is True
    queue.close()
    queue.join_thread()


@pytest.mark.skipif(os.name == "nt", reason="POSIX inherited-lock regression")
def test_fork_child_does_not_keep_parent_lock_alive(
    tmp_path: Path,
) -> None:
    context = mp.get_context("fork")
    child_started = context.Event()
    release_child = context.Event()
    contender_entered = context.Event()
    lock_path = tmp_path / "forked.lock"

    def long_lived_child() -> None:
        child_started.set()
        release_child.wait(10)

    def contender() -> None:
        with ledger_module._exclusive_lock(lock_path):
            contender_entered.set()

    with ledger_module._exclusive_lock(lock_path):
        child = context.Process(target=long_lived_child)
        child.start()
        assert child_started.wait(5)

    second = context.Process(target=contender)
    second.start()
    try:
        assert contender_entered.wait(5)
        assert child.is_alive()
    finally:
        release_child.set()
        child.join(5)
        second.join(5)
        for process in (child, second):
            if process.is_alive():
                process.kill()
                process.join()
    assert child.exitcode == 0
    assert second.exitcode == 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX paused-open fork regression")
def test_concurrent_fork_waits_for_open_descriptor_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = mp.get_context("fork")
    lock_path = tmp_path / "paused-open.lock"
    open_paused = threading.Event()
    resume_open = threading.Event()
    parent_holds = threading.Event()
    release_parent = threading.Event()
    start_contender = context.Event()
    contender_entered = context.Event()
    real_open = ledger_module._open_lock_descriptor
    original_pid = os.getpid()

    def paused_open(path: Path, *, create: bool = True) -> int:
        descriptor = real_open(path, create=create)
        if os.getpid() == original_pid:
            open_paused.set()
            resume_open.wait(5)
        return descriptor

    monkeypatch.setattr(
        ledger_module,
        "_open_lock_descriptor",
        paused_open,
    )
    # Make release rely on descriptor ownership so an inherited open-file
    # description deterministically keeps the advisory lock alive.
    monkeypatch.setattr(ledger_module, "_unlock_file", lambda handle: None)

    def holder() -> None:
        with ledger_module._exclusive_lock(lock_path):
            parent_holds.set()
            release_parent.wait(10)

    def contender() -> None:
        start_contender.wait(10)
        with ledger_module._exclusive_lock(lock_path):
            contender_entered.set()

    second = context.Process(target=contender)
    second.start()
    worker = threading.Thread(target=holder)
    worker.start()
    assert open_paused.wait(5)

    controller = threading.Thread(
        target=lambda: (time.sleep(0.5), resume_open.set())
    )
    controller.start()
    child_pid = os.fork()
    if child_pid == 0:
        time.sleep(10)
        os._exit(0)

    child_reaped = False
    try:
        assert parent_holds.wait(5)
        release_parent.set()
        worker.join(5)
        assert not worker.is_alive()
        start_contender.set()
        assert contender_entered.wait(3)
        child_status = os.waitpid(child_pid, os.WNOHANG)
        child_reaped = child_status != (0, 0)
        assert not child_reaped
    finally:
        release_parent.set()
        resume_open.set()
        worker.join(2)
        controller.join(2)
        if not child_reaped:
            child_status = os.waitpid(child_pid, os.WNOHANG)
            if child_status == (0, 0):
                os.kill(child_pid, 9)
                os.waitpid(child_pid, 0)
        second.join(5)
        if second.is_alive():
            second.kill()
            second.join()
    assert second.exitcode == 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX reentrant at-fork regression")
def test_same_thread_fork_inside_fork_guard_does_not_deadlock() -> None:
    with ledger_module._FORK_GUARD:
        child_pid = os.fork()
        if child_pid == 0:
            with ledger_module._FORK_GUARD:
                os._exit(0)

    waited_pid, status = os.waitpid(child_pid, 0)
    assert waited_pid == child_pid
    assert os.waitstatus_to_exitcode(status) == 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX opener registration regression")
def test_lock_opener_registers_before_same_thread_fork(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = mp.get_context("fork")
    lock_path = tmp_path / "same-thread-open.lock"
    contender_entered = context.Event()
    real_open = ledger_module._open_lock_descriptor
    original_pid = os.getpid()
    child_pids: list[int] = []
    forked = False

    def fork_after_real_open(path: Path, *, create: bool = True) -> object:
        nonlocal forked
        opened = real_open(path, create=create)
        lease = opened.lease
        assert lease is not None
        assert ledger_module._HELD_RESOURCES[id(lease)] is lease
        if os.getpid() == original_pid and not forked:
            forked = True
            child_pid = os.fork()
            if child_pid == 0:
                time.sleep(10)
                os._exit(0)
            child_pids.append(child_pid)
        return opened

    monkeypatch.setattr(
        ledger_module,
        "_open_lock_descriptor",
        fork_after_real_open,
    )
    monkeypatch.setattr(ledger_module, "_unlock_file", lambda handle: None)

    def contender() -> None:
        with ledger_module._exclusive_lock(lock_path):
            contender_entered.set()

    with ledger_module._exclusive_lock(lock_path):
        pass

    second = context.Process(target=contender)
    second.start()
    try:
        assert contender_entered.wait(3)
        assert len(child_pids) == 1
        assert os.waitpid(child_pids[0], os.WNOHANG) == (0, 0)
    finally:
        if child_pids:
            child_status = os.waitpid(child_pids[0], os.WNOHANG)
            if child_status == (0, 0):
                os.kill(child_pids[0], 9)
                os.waitpid(child_pids[0], 0)
        second.join(5)
        if second.is_alive():
            second.kill()
            second.join()
    assert second.exitcode == 0


def _assert_reused_descriptor_survives_fork(
    captured_descriptor: int,
) -> None:
    opened: list[int] = []
    reused_descriptor: int | None = None
    try:
        for _ in range(captured_descriptor + 8):
            descriptor = os.open(os.devnull, os.O_RDONLY)
            opened.append(descriptor)
            if descriptor == captured_descriptor:
                reused_descriptor = descriptor
                break
        assert reused_descriptor == captured_descriptor

        read_fd, write_fd = os.pipe()
        child_pid = os.fork()
        if child_pid == 0:
            os.close(read_fd)
            try:
                os.fstat(reused_descriptor)
                assert not ledger_module._HELD_RESOURCES
                os.write(write_fd, b"PASS")
                os._exit(0)
            except BaseException:
                os._exit(1)

        os.close(write_fd)
        assert os.read(read_fd, 4) == b"PASS"
        os.close(read_fd)
        waited_pid, status = os.waitpid(child_pid, 0)
        assert waited_pid == child_pid
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        for descriptor in opened:
            os.close(descriptor)


@pytest.mark.skipif(os.name == "nt", reason="POSIX fdopen adoption regression")
def test_baseexception_after_fdopen_adoption_cannot_close_reused_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InjectedAdoptionAbort(BaseException):
        pass

    lock_path = tmp_path / "fdopen-adoption-abort.lock"
    captured_descriptor: int | None = None
    real_fdopen = ledger_module.os.fdopen

    def capture_fdopen(
        descriptor: int,
        *args: object,
        **kwargs: object,
    ) -> object:
        nonlocal captured_descriptor
        handle = real_fdopen(descriptor, *args, **kwargs)
        captured_descriptor = descriptor
        return handle

    def abort_after_adoption(lease: object, pending: object) -> None:
        raise InjectedAdoptionAbort

    monkeypatch.setattr(ledger_module.os, "fdopen", capture_fdopen)
    monkeypatch.setattr(
        ledger_module,
        "_after_fdopen_adoption",
        abort_after_adoption,
    )

    with pytest.raises(InjectedAdoptionAbort):
        with ledger_module._exclusive_lock(lock_path):
            pass

    assert captured_descriptor is not None
    assert not ledger_module._HELD_RESOURCES
    assert not ledger_module._ACTIVE_COORDINATORS
    _assert_reused_descriptor_survives_fork(captured_descriptor)


@pytest.mark.skipif(os.name == "nt", reason="POSIX registration rollback regression")
@pytest.mark.parametrize("resource_kind", ["descriptor", "handle"])
def test_post_insert_registration_failure_rolls_back_exact_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource_kind: str,
) -> None:
    class InjectedPostInsert(BaseException):
        pass

    class PostInsertFailureRegistry(dict[int, object]):
        def __setitem__(self, key: int, value: object) -> None:
            super().__setitem__(key, value)
            raise InjectedPostInsert

    registry = PostInsertFailureRegistry()
    monkeypatch.setattr(ledger_module, "_HELD_RESOURCES", registry)
    captured_descriptor: int

    if resource_kind == "descriptor":
        real_os_open = ledger_module.os.open

        def capture_os_open(*args: object, **kwargs: object) -> int:
            nonlocal captured_descriptor
            descriptor = real_os_open(*args, **kwargs)
            captured_descriptor = descriptor
            return descriptor

        monkeypatch.setattr(ledger_module.os, "open", capture_os_open)
        with pytest.raises(InjectedPostInsert):
            ledger_module._open_lock_descriptor(
                tmp_path / "post-insert.lock"
            )
        monkeypatch.setattr(ledger_module.os, "open", real_os_open)
    else:
        captured_descriptor = os.open(os.devnull, os.O_RDONLY)
        handle = os.fdopen(captured_descriptor, "rb", buffering=0)
        with pytest.raises(InjectedPostInsert):
            ledger_module._register_pending_handle(
                handle,
                lambda current: current.close(),
            )
        assert handle.closed

    assert not registry
    _assert_reused_descriptor_survives_fork(captured_descriptor)


@pytest.mark.skipif(os.name == "nt", reason="POSIX lease-transfer regression")
def test_baseexception_after_lease_transfer_cannot_close_reused_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InjectedTransferAbort(BaseException):
        pass

    lock_path = tmp_path / "transfer-abort.lock"
    captured_descriptor: int | None = None
    real_open = ledger_module._open_lock_descriptor

    def capture_open(path: Path, *, create: bool = True) -> object:
        nonlocal captured_descriptor
        pending = real_open(path, create=create)
        if path == lock_path:
            captured_descriptor = pending.descriptor
        return pending

    def abort_after_file_transfer(
        lease: object,
        resource: object,
    ) -> None:
        if isinstance(resource, ledger_module._HeldFileLock):
            raise InjectedTransferAbort

    monkeypatch.setattr(
        ledger_module,
        "_open_lock_descriptor",
        capture_open,
    )
    monkeypatch.setattr(
        ledger_module,
        "_after_lease_transfer",
        abort_after_file_transfer,
    )

    with pytest.raises(InjectedTransferAbort):
        with ledger_module._exclusive_lock(lock_path):
            pass

    assert captured_descriptor is not None
    assert not ledger_module._HELD_RESOURCES
    assert not ledger_module._ACTIVE_COORDINATORS

    _assert_reused_descriptor_survives_fork(captured_descriptor)


@pytest.mark.parametrize("alias_kind", ["case", "hardlink"])
def test_coordinator_identity_order_avoids_two_process_abba(
    tmp_path: Path,
    alias_kind: str,
) -> None:
    first = tmp_path / "first.lock"
    second = tmp_path / "second.lock"
    with ledger_module._exclusive_locks([first, second]):
        pass
    if alias_kind == "case":
        if os.name != "nt":
            pytest.skip("case aliases require Windows")
        reverse_paths = [Path(str(second).upper()), Path(str(first).upper())]
    else:
        first_alias = tmp_path / "z-alias-first.lock"
        second_alias = tmp_path / "a-alias-second.lock"
        try:
            os.link(first, first_alias)
            os.link(second, second_alias)
        except OSError:
            pytest.skip("hardlink creation is unavailable")
        reverse_paths = [second_alias, first_alias]
    marker = tmp_path / "start"
    script = textwrap.dedent(
        """
        import sys
        import time
        from pathlib import Path
        from f51_darwin.circuits.ledger import _exclusive_locks

        marker = Path(sys.argv[1])
        paths = [Path(value) for value in sys.argv[2:]]
        while not marker.exists():
            time.sleep(0.01)
        with _exclusive_locks(paths):
            print("ENTERED", flush=True)
            time.sleep(0.2)
        """
    )
    left = subprocess.Popen(
        [sys.executable, "-c", script, str(marker), str(first), str(second)],
        cwd=Path.cwd(),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    right = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(marker),
            *(str(path) for path in reverse_paths),
        ],
        cwd=Path.cwd(),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        marker.write_text("go", encoding="utf-8")
        left_output, left_error = left.communicate(timeout=8)
        right_output, right_error = right.communicate(timeout=8)
        assert left.returncode == 0, left_error
        assert right.returncode == 0, right_error
        assert left_output.strip() == "ENTERED"
        assert right_output.strip() == "ENTERED"
    finally:
        for process in (left, right):
            if process.poll() is None:
                process.kill()
                process.communicate()


@pytest.mark.skipif(os.name == "nt", reason="POSIX FD cleanup regression")
@pytest.mark.parametrize(
    "failure_stage",
    ["fstat", "identity", "write", "fsync", "fdopen", "flock", "path"],
)
def test_coordinator_pre_yield_failures_close_every_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    descriptor_dir = Path("/proc/self/fd")
    baseline = len(list(descriptor_dir.iterdir()))
    lock_path = tmp_path / f"{failure_stage}.lock"

    def injected(*args: object, **kwargs: object) -> object:
        raise OSError(f"injected {failure_stage} failure")

    if failure_stage == "fstat":
        real_fstat = ledger_module.os.fstat
        calls = 0

        def fail_file_fstat(descriptor: int) -> object:
            nonlocal calls
            calls += 1
            if calls == 2:
                return injected()
            return real_fstat(descriptor)

        monkeypatch.setattr(ledger_module.os, "fstat", fail_file_fstat)
    elif failure_stage == "identity":
        monkeypatch.setattr(
            ledger_module,
            "_file_identity_from_descriptor",
            injected,
        )
    elif failure_stage in {"write", "fsync", "fdopen"}:
        monkeypatch.setattr(ledger_module.os, failure_stage, injected)
    elif failure_stage == "flock":
        monkeypatch.setattr(ledger_module, "_lock_file", injected)
    else:
        monkeypatch.setattr(
            ledger_module,
            "_verify_lock_path_identity",
            injected,
        )

    with pytest.raises(LedgerError, match="injected|acquire"):
        with ledger_module._exclusive_lock(lock_path):
            pass

    assert len(list(descriptor_dir.iterdir())) == baseline
    assert not ledger_module._HELD_RESOURCES
    assert not ledger_module._ACTIVE_COORDINATORS
    monkeypatch.undo()
    with ledger_module._exclusive_lock(lock_path):
        pass


def test_coordinator_cleans_active_state_after_body_and_release_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="body"):
        with ledger_module._exclusive_lock(tmp_path / "body.lock"):
            raise RuntimeError("body")
    assert not ledger_module._ACTIVE_COORDINATORS

    real_verify = ledger_module._verify_lock_path_identity
    count = 0

    def fail_release(path: Path, identity: tuple[int, int]) -> None:
        nonlocal count
        count += 1
        real_verify(path, identity)
        if count == 2:
            raise LedgerError("injected release failure")

    monkeypatch.setattr(
        ledger_module,
        "_verify_lock_path_identity",
        fail_release,
    )
    with pytest.raises(LedgerError, match="release"):
        with ledger_module._exclusive_lock(tmp_path / "release.lock"):
            pass
    assert not ledger_module._ACTIVE_COORDINATORS
    monkeypatch.setattr(
        ledger_module,
        "_verify_lock_path_identity",
        real_verify,
    )
    with ledger_module._exclusive_lock(tmp_path / "after.lock"):
        pass


def test_coordinator_normalizes_open_oserror_to_ledger_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_open(path: Path, *, create: bool = True) -> int:
        raise OSError("injected open failure")

    monkeypatch.setattr(ledger_module, "_open_lock_descriptor", fail_open)

    with pytest.raises(LedgerError, match="open|acquire|injected"):
        with ledger_module._exclusive_lock(tmp_path / "failure.lock"):
            pass
    assert not ledger_module._ACTIVE_COORDINATORS


def test_pointer_normalizes_coordinator_oserror_to_pointer_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_sha = CircuitLedger(ledger_path).append(_event(0)).record_sha256
    checkpoint = tmp_path / "candidate.pt"
    checkpoint_sha = _checkpoint(checkpoint, b"candidate")
    root = tmp_path / "pointers"
    root.mkdir()

    def fail_open(path: Path, *, create: bool = True) -> int:
        raise OSError("injected coordinator open failure")

    monkeypatch.setattr(ledger_module, "_open_lock_descriptor", fail_open)

    with pytest.raises(PointerError, match="lock|open|acquire|injected"):
        publish_active_pointer(
            root,
            "0" * 64,
            _candidate(
                checkpoint,
                checkpoint_sha,
                ledger_path,
                ledger_sha,
                1,
            ),
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX namespace release regression")
def test_pointer_restores_a_when_namespace_release_integrity_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        root,
        ledger,
        pointer_a,
        pointer_a_bytes,
        checkpoint_b,
        sha_b,
        ledger_b,
    ) = _published_a_and_candidate_b(tmp_path)
    real_verify = ledger_module._verify_namespace_path
    verification_count = 0

    def fail_final_namespace_release(
        path: Path,
        identity: tuple[int, int],
    ) -> None:
        nonlocal verification_count
        verification_count += 1
        real_verify(path, identity)
        if verification_count == 3:
            raise LedgerError("injected final namespace release failure")

    monkeypatch.setattr(
        ledger_module,
        "_verify_namespace_path",
        fail_final_namespace_release,
    )

    with pytest.raises(PointerError, match="namespace|integrity|release"):
        publish_active_pointer(
            root,
            pointer_a.checkpoint_sha256,
            _candidate(
                checkpoint_b,
                sha_b,
                ledger.path,
                ledger_b,
                2,
            ),
        )

    assert verification_count >= 3
    assert (root / "active.json").read_bytes() == pointer_a_bytes
    assert read_active_pointer(root) == pointer_a
