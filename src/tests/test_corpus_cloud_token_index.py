from f51_darwin.corpus_cloud_token_index import (
    build_cloud_token_index_plan,
    build_remote_token_index,
)


def test_build_cloud_token_index_plan_writes_training_index() -> None:
    plan = build_cloud_token_index_plan(
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_workspace="/workspace/f51_corpus_factory",
    )

    assert plan.remote_host == "root@70.30.158.46"
    assert plan.remote_port == 56013
    assert plan.remote_token_dir == "/workspace/f51_corpus_factory/tokens/80k"
    assert plan.remote_index_path == "/workspace/f51_corpus_factory/tokens/80k/index.json"
    assert not plan.force
    script = plan.command[-1]
    assert "TOKEN_INDEX_OK" in script
    assert "TOKEN_INDEX_WRITTEN" in script
    assert "token index already exists" in script
    assert "total_tokens" in script
    assert "sha256 mismatch" in script


def test_build_cloud_token_index_plan_force_allows_overwrite() -> None:
    plan = build_cloud_token_index_plan(
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_workspace="/workspace/f51_corpus_factory",
        force=True,
    )

    assert plan.force
    assert "FORCE=1" in plan.command[-1]


def test_build_remote_token_index_dry_run_does_not_execute() -> None:
    result = build_remote_token_index(
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_workspace="/workspace/f51_corpus_factory",
        dry_run=True,
    )

    assert result.ok
    assert result.dry_run
    assert not result.executed


def test_build_remote_token_index_executes_with_runner() -> None:
    calls = []

    class Completed:
        returncode = 0
        stdout = "TOKEN_INDEX_WRITTEN path=/workspace/f51_corpus_factory/tokens/80k/index.json"
        stderr = ""

    def runner(command):
        calls.append(command)
        return Completed()

    result = build_remote_token_index(
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_workspace="/workspace/f51_corpus_factory",
        runner=runner,
    )

    assert result.ok
    assert result.executed == calls
    assert len(calls) == 1


def test_build_remote_token_index_reports_remote_failure() -> None:
    class Completed:
        returncode = 2
        stdout = ""
        stderr = "token index already exists"

    result = build_remote_token_index(
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_workspace="/workspace/f51_corpus_factory",
        runner=lambda command: Completed(),
    )

    assert not result.ok
    assert "token index already exists" in result.errors[0]
