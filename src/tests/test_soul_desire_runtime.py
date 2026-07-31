from __future__ import annotations

from f51_darwin.soul import F51Soul


def test_heartbeat_uses_numeric_dopamine_for_desire_engine(
    tmp_path, monkeypatch
) -> None:
    soul = F51Soul(tmp_path / "organism")
    dopamine_event = {
        "reward": 0.5,
        "dopamine": 0.75,
        "xp_gain": 1.0,
        "level": 1,
        "event": "test",
    }
    monkeypatch.setattr(
        soul.dopamine,
        "on_train_step",
        lambda loss, tokens: dopamine_event,
    )

    report = soul.heartbeat(
        {
            "loss": 2.0,
            "fresh_loss": 2.0,
            "replay_loss": 1.5,
            "tokens_this_cycle": 64,
            "epochs": 6.0,
            "params": 1_764_000_000,
            "step": 50,
        }
    )

    assert report["dopamine"] == dopamine_event
    assert isinstance(report["desire"]["fire"], float)
    assert any(
        command["action"] == "request_checkpoint"
        for command in report["commands"]
    )
