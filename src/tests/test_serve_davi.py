from __future__ import annotations

import http.client
import json
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from pathlib import Path

from f51_darwin.data_factory import DataFactory, DataFactoryPaths
from f51_darwin.data_firewall import DataFirewall, FirewallConfig
from f51_darwin.inference_learner import ResearchMemoryStore
from scripts.serve_davi import DaviService, TrainingController, UI_HTML, build_http_server


def test_davi_runtime_state_is_routed_under_workspace() -> None:
    text = Path("src/f51_darwin/serving/runtime.py").read_text(encoding="utf-8")

    assert "WorkspacePaths.from_project" in text
    assert "paths.runs / \"online_learning/research_memory.json\"" in text
    assert "paths.runs / \"online_learning/adapter_state.pt\"" in text
    assert 'default="127.0.0.1"' in text
    assert "resolve_latest_organism_checkpoint(root, require=True)" in text
    assert "scan_latest_organism_checkpoint" not in text


class _Service:
    api_token = "test-local-token"

    def stats(self):
        return {"interactions": 1, "memories": 2}

    def interact(self, prompt):
        return {"prompt": prompt, "response": {"text": "safe", "sources": []}}

    def approve(self, memory_id):
        return {"accepted": True, "memory_id": memory_id}


class _ModeModel:
    def __init__(self) -> None:
        self.training = True

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False


class _ModeOrganism:
    def __init__(self) -> None:
        self.model = _ModeModel()
        self.steps = 0

    def run_cycle(self, *, steps: int):
        for _ in range(steps):
            self.steps += 1
            time.sleep(0.001)
            self._runtime_pause_checkpoint(self.model)
        return {"loss_end": 1.0, "lm_end": 0.5, "ppl_end": 1.6}


def test_training_pause_waits_for_safe_boundary_and_stops_updates() -> None:
    organism = _ModeOrganism()
    controller = TrainingController(organism, steps_per_cycle=100)
    controller.start()
    deadline = time.monotonic() + 2
    while organism.steps < 3 and time.monotonic() < deadline:
        time.sleep(0.005)

    controller.pause()
    paused_at = organism.steps
    time.sleep(0.03)

    assert controller.paused
    assert controller.mode == "inference"
    assert organism.model.training is False
    assert organism.steps == paused_at

    controller.resume()
    assert controller.mode == "train"
    deadline = time.monotonic() + 2
    while organism.steps == paused_at and time.monotonic() < deadline:
        time.sleep(0.005)
    controller.stop()

    assert organism.steps > paused_at
    assert controller._thread is None


def _request(port, method, path, *, token=None, payload=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {}
    body = None
    if token:
        headers["X-Davi-Token"] = token
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload)
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    return response.status, raw, dict(response.getheaders())


@contextmanager
def _running_server(service, *, expected_errors=0):
    server = build_http_server(service, port=0)
    assert server.bound.is_set()
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.01},
        daemon=False,
        name="davi-http-test",
    )
    thread.start()
    assert server.serving.wait(timeout=2)
    try:
        yield server, server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert server.stopped.is_set()
        assert not thread.is_alive()
        assert len(server.server_errors) == expected_errors, server.server_errors


def test_ui_uses_text_content_and_server_is_loopback_token_guarded() -> None:
    service = _Service()
    with _running_server(service) as (_server, port):
        status, html, headers = _request(port, "GET", "/")
        assert status == 200
        assert b"test-local-token" in html
        assert "X-Frame-Options" in headers
        assert "innerHTML" not in UI_HTML

        status, _, _ = _request(
            port, "POST", "/api/interact", payload={"prompt": "hello"}
        )
        assert status == 403

        status, raw, _ = _request(
            port,
            "POST",
            "/api/interact",
            token=service.api_token,
            payload={"prompt": "hello"},
        )
        assert status == 200
        assert json.loads(raw)["response"]["text"] == "safe"

        status, raw, _ = _request(
            port,
            "POST",
            "/api/approve",
            token=service.api_token,
            payload={"memory_id": "web_1"},
        )
        assert status == 200
        assert json.loads(raw)["accepted"] is True


def test_cross_origin_post_is_rejected() -> None:
    service = _Service()
    with _running_server(service) as (_server, port):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            body = json.dumps({"prompt": "hello"})
            connection.request(
                "POST",
                "/api/interact",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Davi-Token": service.api_token,
                    "Origin": "https://attacker.example",
                },
            )
            response = connection.getresponse()
            response.read()
            assert response.status == 403
        finally:
            connection.close()


def test_unauthorized_large_post_stress_has_no_connection_abort() -> None:
    service = _Service()
    body = json.dumps({"prompt": "x" * 60_000})
    with _running_server(service) as (_server, port):
        statuses = []
        for _ in range(3):
            for _ in range(200):
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                try:
                    connection.request(
                        "POST",
                        "/api/interact",
                        body=body,
                        headers={"Content-Type": "application/json"},
                    )
                    response = connection.getresponse()
                    response.read()
                    statuses.append(response.status)
                finally:
                    connection.close()
        assert statuses == [403] * 600


def test_http_server_lifecycle_is_deterministic_across_restarts() -> None:
    for _ in range(100):
        with _running_server(_Service()) as (_server, port):
            status, _, _ = _request(port, "GET", "/")
            assert status == 200


def test_handler_sentinel_is_captured_and_returns_generic_500() -> None:
    class SentinelService(_Service):
        def interact(self, prompt):
            raise LookupError("sentinel-handler-failure")

    with _running_server(SentinelService(), expected_errors=1) as (server, port):
        status, raw, _ = _request(
            port,
            "POST",
            "/api/interact",
            token=SentinelService.api_token,
            payload={"prompt": "hello"},
        )
        assert status == 500
        assert json.loads(raw) == {"error": "internal server error"}
        assert server.server_errors == [
            {
                "source": "handler",
                "type": "LookupError",
                "message": "sentinel-handler-failure",
            }
        ]


class _FirewallLearner:
    def __init__(self, path) -> None:
        self.memory = ResearchMemoryStore(path)
        self.consolidations = 0

    def consolidate(self, memory_id):
        existing = self.memory.get(memory_id)
        if existing is not None and existing.status == "consolidated":
            return {**existing.consolidation, "accepted": True, "idempotent": True}
        self.consolidations += 1
        result = {"accepted": True, "reason": "committed"}
        self.memory.approve(memory_id)
        self.memory.record_consolidation(memory_id, result)
        return result


def _firewall_service(tmp_path):
    learner = _FirewallLearner(tmp_path / "research_memory.json")
    factory = DataFactory(
        DataFactoryPaths.from_root(tmp_path),
        firewall=DataFirewall(FirewallConfig(min_chars=50)),
    )
    return DaviService(learner, organism=SimpleNamespace(data_factory=factory)), learner, factory


def _remember(learner, text):
    memory, _created = learner.memory.remember(
        query="verify evidence",
        text=text,
        sources=[{"title": "Source", "href": "https://example.org/source"}],
    )
    return memory


def test_rejected_canonical_record_can_never_reach_adapter(tmp_path) -> None:
    service, learner, factory = _firewall_service(tmp_path)
    memory = _remember(learner, "short untrusted text")

    provenance = service._register_quarantine(memory.id)

    assert provenance["dataset_status"] == "rejected"
    with pytest.raises(ValueError, match="canonical provenance is not approved: rejected"):
        service.approve(memory.id)
    assert learner.consolidations == 0
    assert factory.ledger.find_by_hash(memory.content_hash).status.value == "rejected"


def test_quarantine_requires_explicit_approval_before_consolidation(tmp_path) -> None:
    service, learner, factory = _firewall_service(tmp_path)
    text = (
        "This imported scientific evidence has explicit provenance and enough "
        "content for inspection, but the non-real source must remain quarantined "
        "until a local operator explicitly approves it."
    )
    memory = _remember(learner, text)
    provenance = service._register_quarantine(memory.id)
    assert provenance["dataset_status"] == "quarantine"
    assert learner.consolidations == 0

    result = service.approve(memory.id)

    assert result["accepted"] is True
    assert learner.consolidations == 1
    record = factory.ledger.find_by_hash(memory.content_hash)
    assert record.status.value == "approved"
    assert (factory.paths.corpus / f"{record.id}.txt").exists()

    repeated = service.approve(memory.id)
    assert repeated["accepted"] is True
    assert repeated.get("idempotent") is True
    assert learner.consolidations == 1
