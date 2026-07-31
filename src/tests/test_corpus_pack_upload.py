from pathlib import Path

from f51_darwin.corpus_factory_pipeline import build_corpus_pack
from f51_darwin.corpus_pack_upload import build_upload_plan, upload_verified_pack


def _make_pack(tmp_path: Path) -> Path:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "math.txt").write_text(
        (
            "This theorem has a proof using algebra, calculus, statistics, "
            "probability, and a differential equation. "
        )
        * 8,
        encoding="utf-8",
    )
    summary = build_corpus_pack(
        input_dir=input_dir,
        output_dir=output_dir,
        default_license="PD-US",
        source_url_base="https://www.gutenberg.org",
    )
    assert summary.pack_path is not None
    return Path(summary.pack_path)


def test_build_upload_plan_uses_master_incoming_dir(tmp_path: Path) -> None:
    pack = _make_pack(tmp_path)
    plan = build_upload_plan(
        pack_path=pack,
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_workspace="/workspace/f51_corpus_factory",
        batch_name="batch_001",
    )

    assert plan.remote_host == "root@70.30.158.46"
    assert plan.remote_port == 56013
    assert plan.remote_dir == "/workspace/f51_corpus_factory/incoming/batch_001"
    assert plan.remote_pack_path.endswith("/approved_corpus_pack.tar.gz")
    assert len(plan.pack_sha256) == 64
    assert plan.commands[0][:3] == ["ssh", "-p", "56013"]
    assert plan.commands[1][0] == "scp"


def test_upload_verified_pack_dry_run_does_not_execute(tmp_path: Path) -> None:
    pack = _make_pack(tmp_path)
    result = upload_verified_pack(
        pack_path=pack,
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_workspace="/workspace/f51_corpus_factory",
        batch_name="batch_001",
        dry_run=True,
    )

    assert result.ok
    assert result.dry_run
    assert not result.executed


def test_upload_verified_pack_executes_commands_with_runner(tmp_path: Path) -> None:
    pack = _make_pack(tmp_path)
    calls = []

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def runner(command):
        calls.append(command)
        return Completed()

    result = upload_verified_pack(
        pack_path=pack,
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_workspace="/workspace/f51_corpus_factory",
        batch_name="batch_001",
        runner=runner,
    )

    assert result.ok
    assert len(calls) == 4
    assert result.executed == calls


def test_upload_verified_pack_refuses_invalid_pack(tmp_path: Path) -> None:
    bad_pack = tmp_path / "bad.tar.gz"
    bad_pack.write_bytes(b"not a tar")

    result = upload_verified_pack(
        pack_path=bad_pack,
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_workspace="/workspace/f51_corpus_factory",
        dry_run=True,
    )

    assert not result.ok
    assert result.errors
    assert "pack verification failed" in result.errors[0]
