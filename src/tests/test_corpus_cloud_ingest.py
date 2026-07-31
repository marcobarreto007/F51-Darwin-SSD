from f51_darwin.corpus_cloud_ingest import build_cloud_ingest_plan, ingest_remote_pack


def test_build_cloud_ingest_plan_targets_normalized_and_manifest() -> None:
    plan = build_cloud_ingest_plan(
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_pack_path="/workspace/f51_corpus_factory/incoming/batch_001/approved_corpus_pack.tar.gz",
        remote_workspace="/workspace/f51_corpus_factory",
        batch_name="batch_001",
    )

    assert plan.remote_host == "root@70.30.158.46"
    assert plan.remote_port == 56013
    assert plan.remote_normalized_dir == "/workspace/f51_corpus_factory/normalized/batch_001"
    assert plan.remote_manifest_path == "/workspace/f51_corpus_factory/manifests/batch_001.jsonl"
    assert plan.command[:4] == ["ssh", "-p", "56013", "root@70.30.158.46"]
    script = plan.command[-1]
    assert "sha256sum -c" in script
    assert "batch already ingested" in script
    assert "INGEST_OK" in script


def test_ingest_remote_pack_dry_run_does_not_execute() -> None:
    result = ingest_remote_pack(
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_pack_path="/workspace/f51_corpus_factory/incoming/batch_001/approved_corpus_pack.tar.gz",
        remote_workspace="/workspace/f51_corpus_factory",
        batch_name="batch_001",
        dry_run=True,
    )

    assert result.ok
    assert result.dry_run
    assert not result.executed


def test_ingest_remote_pack_executes_with_runner() -> None:
    calls = []

    class Completed:
        returncode = 0
        stdout = "INGEST_OK batch=batch_001 approved=1"
        stderr = ""

    def runner(command):
        calls.append(command)
        return Completed()

    result = ingest_remote_pack(
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_pack_path="/workspace/f51_corpus_factory/incoming/batch_001/approved_corpus_pack.tar.gz",
        remote_workspace="/workspace/f51_corpus_factory",
        batch_name="batch_001",
        runner=runner,
    )

    assert result.ok
    assert result.executed == calls
    assert len(calls) == 1


def test_ingest_remote_pack_reports_remote_failure() -> None:
    class Completed:
        returncode = 2
        stdout = ""
        stderr = "batch already ingested: batch_001"

    result = ingest_remote_pack(
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_pack_path="/workspace/f51_corpus_factory/incoming/batch_001/approved_corpus_pack.tar.gz",
        remote_workspace="/workspace/f51_corpus_factory",
        batch_name="batch_001",
        runner=lambda command: Completed(),
    )

    assert not result.ok
    assert "batch already ingested" in result.errors[0]
