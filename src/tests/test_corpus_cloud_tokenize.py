from f51_darwin.corpus_cloud_tokenize import build_cloud_tokenize_plan, tokenize_remote_batch


def test_build_cloud_tokenize_plan_targets_batch_outputs() -> None:
    plan = build_cloud_tokenize_plan(
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_workspace="/workspace/f51_corpus_factory",
        batch_name="batch_001",
    )

    assert plan.remote_host == "root@70.30.158.46"
    assert plan.remote_port == 56013
    assert plan.remote_normalized_dir == "/workspace/f51_corpus_factory/normalized/batch_001"
    assert plan.remote_input_manifest == "/workspace/f51_corpus_factory/manifests/batch_001.jsonl"
    assert plan.remote_tokenizer_dir == "/workspace/f51_corpus_factory/tokenizer/f51_bpe_80k"
    assert plan.remote_token_bin == "/workspace/f51_corpus_factory/tokens/80k/batch_001.int32.bin"
    assert plan.remote_token_manifest == "/workspace/f51_corpus_factory/tokens/80k/batch_001.tokens.json"
    script = plan.command[-1]
    assert "F51BPETokenizer.load" in script
    assert "token output already exists" in script
    assert "TOKENIZE_OK" in script
    assert "token id out of vocab" in script


def test_tokenize_remote_batch_dry_run_does_not_execute() -> None:
    result = tokenize_remote_batch(
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_workspace="/workspace/f51_corpus_factory",
        batch_name="batch_001",
        dry_run=True,
    )

    assert result.ok
    assert result.dry_run
    assert not result.executed


def test_tokenize_remote_batch_executes_with_runner() -> None:
    calls = []

    class Completed:
        returncode = 0
        stdout = "TOKENIZE_OK batch=batch_001"
        stderr = ""

    def runner(command):
        calls.append(command)
        return Completed()

    result = tokenize_remote_batch(
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_workspace="/workspace/f51_corpus_factory",
        batch_name="batch_001",
        runner=runner,
    )

    assert result.ok
    assert result.executed == calls
    assert len(calls) == 1


def test_tokenize_remote_batch_reports_remote_failure() -> None:
    class Completed:
        returncode = 2
        stdout = ""
        stderr = "token output already exists for batch: batch_001"

    result = tokenize_remote_batch(
        ssh_host="ssh -p 56013 root@70.30.158.46",
        remote_workspace="/workspace/f51_corpus_factory",
        batch_name="batch_001",
        runner=lambda command: Completed(),
    )

    assert not result.ok
    assert "token output already exists" in result.errors[0]
