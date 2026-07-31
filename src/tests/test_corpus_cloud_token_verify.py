from f51_darwin.corpus_cloud_token_verify import (
    build_cloud_token_verify_plan,
    verify_remote_token_batch,
)


def test_build_cloud_token_verify_plan_checks_manifest_and_bin() -> None:
    plan = build_cloud_token_verify_plan(
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_workspace="/workspace/f51_corpus_factory",
        batch_name="batch_001",
    )

    assert plan.remote_host == "root@70.30.158.46"
    assert plan.remote_port == 56013
    assert plan.remote_token_bin == "/workspace/f51_corpus_factory/tokens/80k/batch_001.int32.bin"
    assert plan.remote_token_manifest == "/workspace/f51_corpus_factory/tokens/80k/batch_001.tokens.json"
    script = plan.command[-1]
    assert "TOKEN_VERIFY_OK" in script
    assert "token bin size mismatch" in script
    assert "token bin sha256 mismatch" in script
    assert "output_bin mismatch" in script


def test_verify_remote_token_batch_dry_run_does_not_execute() -> None:
    result = verify_remote_token_batch(
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_workspace="/workspace/f51_corpus_factory",
        batch_name="batch_001",
        dry_run=True,
    )

    assert result.ok
    assert result.dry_run
    assert not result.executed


def test_verify_remote_token_batch_executes_with_runner() -> None:
    calls = []

    class Completed:
        returncode = 0
        stdout = '{"status":"TOKEN_VERIFY_OK"}'
        stderr = ""

    def runner(command):
        calls.append(command)
        return Completed()

    result = verify_remote_token_batch(
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_workspace="/workspace/f51_corpus_factory",
        batch_name="batch_001",
        runner=runner,
    )

    assert result.ok
    assert result.executed == calls
    assert len(calls) == 1


def test_verify_remote_token_batch_reports_remote_failure() -> None:
    class Completed:
        returncode = 5
        stdout = ""
        stderr = "token bin sha256 mismatch"

    result = verify_remote_token_batch(
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_workspace="/workspace/f51_corpus_factory",
        batch_name="batch_001",
        runner=lambda command: Completed(),
    )

    assert not result.ok
    assert "sha256 mismatch" in result.errors[0]
