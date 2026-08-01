#!/usr/bin/env python3
"""Davi unified server — training + inference with mutual exclusion.

One organism owns the model.  A background training thread runs continuous
cycles.  The frontend toggle switches between *training mode* and *inference
mode*.  The two modes are mutually exclusive:
  - In training mode the chat is locked and cycles run freely.
  - In inference mode training pauses at the next safe boundary, the model
    switches to eval, and chat / research / web search are available.

VRAM safety: mode transitions call torch.cuda.empty_cache() and the dual‑GPU
pipeline is never asked to train and generate simultaneously.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[3]
CANONICAL_CONFIG = "src/configs/darwin_x_100m.yaml"

from f51_darwin.data_factory import approve_quarantine_item
from f51_darwin.dataset_states import DatasetStatus, SourceType
from f51_darwin.inference_learner import (
    InferenceLearner,
    OnlineLearningConfig,
)
from f51_darwin.artifacts import resolve_latest_organism_checkpoint
from f51_darwin.dataset_layout import WorkspacePaths
from f51_darwin.serving.burst_detector import BurstDetector
from f51_darwin.serving.prefix_cache import PrefixCache

import torch

# ═══════════════════════════════════════════════════════════════════════════
# UI  (single‑page app — Portuguese)
# ═══════════════════════════════════════════════════════════════════════════

UI_HTML = (
    resources.files("f51_darwin.serving")
    .joinpath("static/davi.html")
    .read_text(encoding="utf-8")
)

# ═══════════════════════════════════════════════════════════════════════════
# GPU helpers
# ═══════════════════════════════════════════════════════════════════════════

def _gpu_stats() -> tuple[str, str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return (
            (lines[0] if len(lines) > 0 else "—"),
            (lines[1] if len(lines) > 1 else "—"),
        )
    except Exception:
        return "—", "—"


# ═══════════════════════════════════════════════════════════════════════════
# Training controller  (background thread, pausable from frontend)
# ═══════════════════════════════════════════════════════════════════════════

class TrainingController:
    """Run organism cycles with an acknowledged train/inference handoff."""

    def __init__(self, organism, *, steps_per_cycle: int = 500) -> None:
        self._org = organism
        self._steps = steps_per_cycle
        self._pause_event = threading.Event()
        self._pause_event.set()        # start NOT paused -> training runs
        self._pause_ack = threading.Event()
        self._running_ack = threading.Event()
        self._running_ack.set()
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # The organism calls this only after an optimizer step has completed.
        organism._runtime_pause_checkpoint = self._pause_at_safe_boundary

        # Live metrics (written by training thread, read by stats endpoint)
        self.loss: float | None = None
        self.lm_loss: float | None = None
        self.ppl: float | None = None
        self.tok_s: float = 0.0
        self.active_experts: int = 0
        self.died_experts: int = 0
        self.last_report: dict[str, Any] = {}

    # ── mode control ──

    @property
    def mode(self) -> str:
        return "inference" if self.paused else "train"

    @property
    def paused(self) -> bool:
        return self._pause_ack.is_set()

    def pause(self) -> None:
        """Return only after the training thread acknowledges a safe pause."""
        if self.paused:
            return
        self._running_ack.clear()
        self._pause_event.clear()
        if not self._running:
            self._on_pause()
            self._pause_ack.set()
            return
        if not self._pause_ack.wait(timeout=60):
            self._pause_event.set()
            self._running_ack.set()
            raise RuntimeError("training did not acknowledge pause within 60 seconds")

    def resume(self) -> None:
        """Resume training."""
        if self._pause_event.is_set() and not self.paused:
            return
        self._pause_event.set()
        if not self._running:
            self._pause_ack.clear()
            self._on_resume()
            self._running_ack.set()
            return
        if not self._running_ack.wait(timeout=60):
            raise RuntimeError("training did not acknowledge resume within 60 seconds")

    def set_mode(self, mode: str) -> str:
        if mode == "inference":
            self.pause()
        else:
            self.resume()
        return self.mode

    def _on_pause(self) -> None:
        """Change model state only after the training thread is quiescent."""
        try:
            self._org.model.eval()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _on_resume(self) -> None:
        """Called when transitioning inference -> train."""
        try:
            self._org.model.train()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _pause_at_safe_boundary(self, model: Any) -> None:
        """Acknowledge a pause after a completed optimizer step or cycle."""
        if self._pause_event.is_set():
            return
        self._on_pause()
        self._pause_ack.set()
        try:
            self._pause_event.wait()
        finally:
            self._pause_ack.clear()
        if self._running:
            self._on_resume()
            self._running_ack.set()

    # ── life-cycle ──

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=False, name="davi-train"
        )
        self._thread.start()

    def stop(self, *, timeout: float = 60.0) -> None:
        self._running = False
        self._pause_event.set()   # unblock thread so it can exit
        self._running_ack.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise RuntimeError(f"training thread did not stop within {timeout:g} seconds")
        self._thread = None

    # ── training loop ──

    def _loop(self) -> None:
        print("[Davi] Training thread started.", flush=True)
        while self._running:
            self._pause_at_safe_boundary(self._org.model)

            try:
                report = self._org.run_cycle(steps=self._steps)
                with self._lock:
                    self.last_report = report
                    self.loss = report.get("loss_end")
                    self.lm_loss = report.get("lm_end")
                    self.ppl = report.get("ppl_end")
                    self.tok_s = getattr(self._org, "_last_tok_s", 0.0)
                    self.active_experts = report.get("modules_active", 0)
                    self.died_experts = report.get("modules_died", 0)
            except Exception:
                traceback.print_exc()
                time.sleep(5)

        print("[Davi] Training thread stopped.", flush=True)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            r = dict(self.last_report)
        return r


# ═══════════════════════════════════════════════════════════════════════════
# Davi service  (HTTP handlers)
# ═══════════════════════════════════════════════════════════════════════════

class DaviService:
    def __init__(
        self,
        learner: InferenceLearner,
        *,
        organism: Any = None,
        training: TrainingController | None = None,
    ) -> None:
        self.learner = learner
        self.organism = organism
        self.training = training
        self.api_token = secrets.token_urlsafe(32)
        self._transaction_lock = threading.RLock()

        # BTB: cache de prefixo compartilhado + detector de burst
        self._btb_prefix_cache = PrefixCache(max_entries=16)
        self._btb_burst_detector = BurstDetector(
            threshold=2,    # X
            prefix_blocks=32,  # Y — 32 * 256 = 8192 tokens de prefixo
            window_s=1.0,   # Z
            warm_copies=1,  # M — single-node: 1 copia
        )

    def set_mode(self, mode: str) -> str:
        """Serialize mode changes with inference and approval transactions."""
        if self.training is None:
            return "inference"
        with self._transaction_lock:
            return self.training.set_mode(mode)

    # ── inference (only when training is paused) ──

    def interact(self, prompt: str) -> dict[str, Any]:
        with self._transaction_lock:
            # Check under the same lock used by set_mode to avoid a resume
            # racing between the guard and the inference transaction.
            if self.training is not None and not self.training.paused:
                raise RuntimeError(
                    "Training is active. Switch to Inference mode in the sidebar first."
                )
            result = self.learner.interact(prompt)
            if result.get("memory_id"):
                provenance = self._register_quarantine(result["memory_id"])
                result["provenance"] = provenance
                if provenance and result.get("memory_status") != "consolidated":
                    result["memory_status"] = provenance["dataset_status"]
            return result

    def _register_quarantine(self, memory_id: str) -> dict[str, str] | None:
        if self.organism is None or not hasattr(self.organism, "data_factory"):
            return None
        memory = self.learner.memory.get(memory_id)
        if memory is None:
            return None
        factory = self.organism.data_factory
        existing = factory.ledger.find_by_hash(memory.content_hash)
        if existing is None:
            record = factory.register_candidate(
                text=memory.text,
                source_type=SourceType.IMPORTED,
                source_path=memory.sources[0]["href"] if memory.sources else "",
                prompt=memory.query,
                dataset_version="online_research_v1",
            )
            factory.audit_candidate(record, memory.text)
            existing = factory.ledger.find_by_hash(memory.content_hash)
        if existing is None:
            return None
        metadata = {
            "dataset_record_id": existing.id,
            "dataset_status": existing.status.value,
        }
        if existing.status == DatasetStatus.REJECTED and memory.status != "consolidated":
            self.learner.memory.set_status(memory_id, "rejected")
        self.learner.memory.record_consolidation(
            memory_id,
            {"accepted": False, "reason": "awaiting_operator_approval", **metadata},
        )
        return metadata

    def approve(self, memory_id: str) -> dict[str, Any]:
        with self._transaction_lock:
            memory = self.learner.memory.get(memory_id)
            if memory is None:
                raise KeyError(memory_id)
            if self.organism is None or not hasattr(self.organism, "data_factory"):
                raise ValueError("canonical data firewall is unavailable")
            record_id = memory.consolidation.get("dataset_record_id")
            if not record_id:
                raise ValueError("memory has no canonical provenance record")

            factory = self.organism.data_factory
            existing = factory.ledger.find_by_hash(memory.content_hash)
            if existing is None or existing.id != record_id:
                raise ValueError("canonical provenance record is missing or mismatched")
            if existing.status == DatasetStatus.QUARANTINE:
                approve_quarantine_item(
                    factory,
                    record_id,
                    reason="approved by local operator through Davi UI",
                )
                existing = factory.ledger.find_by_hash(memory.content_hash)
            if existing is None or existing.status != DatasetStatus.APPROVED:
                status = existing.status.value if existing is not None else "missing"
                raise ValueError(f"canonical provenance is not approved: {status}")

            factory.promote_approved_to_corpus(build_corpus=True)
            return self.learner.consolidate(memory_id)

    # ── stats (includes training live metrics) ──

    def stats(self) -> dict[str, Any]:
        s = self.learner.stats()
        s["cycle"] = getattr(self.organism, "cycle", None)
        s["step"] = getattr(self.organism, "total_steps", None)
        s["experts"] = len(
            getattr(getattr(self.organism, "expert_pool", None), "records", {})
        )
        s["gpu0"], s["gpu1"] = _gpu_stats()

        if self.training is not None:
            s["mode"] = self.training.mode
            snap = self.training.snapshot()
            if self.training.loss is not None:
                s["loss"] = self.training.loss
            if self.training.lm_loss is not None:
                s["lm_loss"] = self.training.lm_loss
            if self.training.ppl is not None:
                s["ppl"] = self.training.ppl
            s["tok_s"] = self.training.tok_s
            s["active_experts"] = self.training.active_experts
            s["died_experts"] = self.training.died_experts
            s["replay_loss"] = snap.get("replay_loss", 0)
            s["forgetting"] = snap.get("forgetting", 0)
        else:
            s["mode"] = "inference"

        # BTB: prefix cache + burst detector stats
        s["btb"] = {
            "prefix_cache": {
                "entries": self._btb_prefix_cache.size(),
                "hits": self._btb_prefix_cache.stats.hits,
                "misses": self._btb_prefix_cache.stats.misses,
                "hit_rate": round(self._btb_prefix_cache.stats.hit_rate, 3),
                "stores": self._btb_prefix_cache.stats.stores,
                "evictions": self._btb_prefix_cache.stats.evictions,
            },
            "burst_detector": {
                "total_arrivals": self._btb_burst_detector.stats.total_arrivals,
                "bursts_detected": self._btb_burst_detector.stats.bursts_detected,
                "active_prefixes": len(self._btb_burst_detector.active_prefixes()),
            },
        }

        return s


# ═══════════════════════════════════════════════════════════════════════════
# HTTP server
# ═══════════════════════════════════════════════════════════════════════════

def _origin_allowed(value: str | None) -> bool:
    if not value:
        return True
    try:
        return urlparse(value).hostname in {"127.0.0.1", "localhost", "::1"}
    except ValueError:
        return False


class DaviHTTPServer(ThreadingHTTPServer):
    """Observable HTTP lifecycle with fail-closed worker error capture."""

    daemon_threads = False
    block_on_close = True

    def __init__(self, server_address, handler_class) -> None:
        self.bound = threading.Event()
        self.serving = threading.Event()
        self.stopped = threading.Event()
        self.server_errors: list[dict[str, str]] = []
        self._error_lock = threading.Lock()
        super().__init__(server_address, handler_class)
        self.bound.set()

    def record_error(self, source: str, exc: BaseException) -> None:
        with self._error_lock:
            self.server_errors.append(
                {"source": source, "type": type(exc).__name__, "message": str(exc)}
            )

    def handle_error(self, request, client_address) -> None:
        exc = sys.exception()
        if exc is not None:
            self.record_error("worker", exc)

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        self.serving.set()
        try:
            super().serve_forever(poll_interval=poll_interval)
        finally:
            self.serving.clear()
            self.stopped.set()


def build_http_server(
    service: DaviService, *, host: str = "127.0.0.1", port: int = 5151
) -> DaviHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "DaviLocal/2.0"

        def log_message(self, format: str, *args: object) -> None:
            return

        def _authorized(self) -> bool:
            return secrets.compare_digest(
                self.headers.get("X-Davi-Token", ""), service.api_token
            ) and _origin_allowed(self.headers.get("Origin"))

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        # ── GET ──

        def do_GET(self) -> None:
            if self.path == "/":
                body = UI_HTML.replace("__TOKEN__", json.dumps(service.api_token)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'unsafe-inline'; "
                    "style-src 'unsafe-inline'; connect-src 'self'",
                )
                self.send_header("X-Frame-Options", "DENY")
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/stats" and self._authorized():
                self._json(200, service.stats())
                return
            if self.path == "/api/btb/stats" and self._authorized():
                self._json(200, service.stats().get("btb", {}))
                return
            self._json(404, {"error": "not found"})

        # ── POST ──

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(400, {"error": "invalid request size"})
                return
            if length < 1 or length > 65_536:
                self.close_connection = True
                self._json(400, {"error": "invalid request size"})
                return
            raw = self.rfile.read(length)
            if len(raw) != length:
                self.close_connection = True
                self._json(400, {"error": "incomplete request body"})
                return
            if not self._authorized():
                self._json(403, {"error": "local API token/origin rejected"})
                return
            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("JSON object required")

                # ── /api/mode ──
                if self.path == "/api/mode":
                    mode = str(payload.get("mode", "train")).strip().lower()
                    if mode not in ("train", "inference"):
                        raise ValueError("mode must be 'train' or 'inference'")
                    if service.training is None:
                        self._json(200, {"mode": "inference", "training_available": False})
                        return
                    new_mode = service.set_mode(mode)
                    self._json(
                        200,
                        {
                            "mode": new_mode,
                            "paused": service.training.paused,
                            "cycle": getattr(service.organism, "cycle", None),
                            "step": getattr(service.organism, "total_steps", None),
                        },
                    )
                    return

                # ── /api/interact ──
                if self.path == "/api/interact":
                    prompt = str(payload.get("prompt", ""))
                    if not prompt.strip() or len(prompt) > 2_000:
                        raise ValueError("prompt must contain 1..2000 characters")
                    self._json(200, service.interact(prompt))
                    return

                # ── /api/approve ──
                if self.path == "/api/approve":
                    memory_id = str(payload.get("memory_id", ""))[:100]
                    if not memory_id:
                        raise ValueError("memory_id is required")
                    self._json(200, service.approve(memory_id))
                    return

                self._json(404, {"error": "not found"})

            except KeyError as exc:
                self._json(404, {"error": f"unknown memory: {exc.args[0]}"})
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})
            except RuntimeError as exc:
                # e.g. "Training is active"
                self._json(409, {"error": str(exc)})
            except Exception as exc:
                self.server.record_error("handler", exc)
                self._json(500, {"error": "internal server error"})

    return DaviHTTPServer((host, int(port)), Handler)


# ═══════════════════════════════════════════════════════════════════════════
# Wiring
# ═══════════════════════════════════════════════════════════════════════════

def build_learner(organism: Any, *, root: Path) -> InferenceLearner:
    # Build the compiled inference engine (torch.compile + KV cache)
    engine = None
    try:
        from f51_darwin.inference_engine import DarwinInferenceEngine

        engine = DarwinInferenceEngine(
            organism.model,
            organism.tokenizer,
            max_seq_len=organism.model.config.context_length,
        )
        print("  🚀 InferenceEngine: torch.compile + KV cache ativo")
    except Exception as exc:
        print(f"  ⚠️ InferenceEngine indisponivel ({exc}) — fallback lento")

    paths = WorkspacePaths.from_project(root)
    return InferenceLearner(
        organism.model,
        organism.tokenizer,
        heartbeat=organism.model.heartbeat,
        inference_engine=engine,
        config=OnlineLearningConfig(
            memory_path=str(paths.runs / "online_learning/research_memory.json"),
            state_path=str(paths.runs / "online_learning/adapter_state.pt"),
            generation_max_tokens=256,
            base_checkpoint_id=getattr(organism, "base_checkpoint_id", ""),
            tokenizer_id=getattr(organism, "tokenizer_id", ""),
        ),
        replay_provider=lambda size: organism.replay.sample(
            min(size, len(organism.replay))
        )
        if len(organism.replay)
        else [],
        initial_state=getattr(organism, "resume_online_learning_state", None),
        allow_state_file=not bool(
            getattr(organism, "online_state_authoritative", False)
        ),
        restore_state_heartbeat=not bool(
            getattr(organism, "online_state_authoritative", False)
        ),
    )


def serve_organism(
    organism: Any,
    *,
    host: str = "127.0.0.1",
    port: int = 5151,
    train: bool = True,
    steps_per_cycle: int = 500,
) -> None:
    root = Path(organism.cfg.project_root).resolve()

    # Training controller (background thread)
    training_ctrl: TrainingController | None = None
    if train:
        training_ctrl = TrainingController(organism, steps_per_cycle=steps_per_cycle)

    service = DaviService(
        build_learner(organism, root=root),
        organism=organism,
        training=training_ctrl,
    )
    httpd = build_http_server(service, host=host, port=port)

    print(f"Davi pronto em http://{host}:{port}")
    if training_ctrl:
        training_ctrl.start()
        print("  ⚡ Treino em background — controlado pelo switch no painel")
    print("  💬 Inferencia liberada quando treino esta pausado")
    print("  🔒 Mutex: treino e inferencia nunca rodam juntos")
    print()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nDavi pausado.")
    finally:
        if training_ctrl:
            training_ctrl.stop()
        httpd.server_close()


def _latest_checkpoint(root: Path) -> Path:
    checkpoint = resolve_latest_organism_checkpoint(root, require=True)
    assert checkpoint is not None
    return checkpoint


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main(
    argv: list[str] | None = None,
    *,
    default_config: str = CANONICAL_CONFIG,
    organism_types: tuple[type, type] | None = None,
) -> None:
    import argparse
    from f51_darwin.dataset_layout import resolve_feast_token_bin
    if organism_types is None:
        raise RuntimeError("Davi CLI requires the organism adapter types")
    DarwinOrganism, DarwinOrganismConfig = organism_types

    parser = argparse.ArgumentParser(
        description="Davi unified server — treino + inferencia"
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--token-bin", default=None)
    parser.add_argument("--config", default=default_config)
    parser.add_argument(
        "--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"]
    )
    parser.add_argument("--port", type=int, default=5151)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--steps", type=int, default=500, help="Steps per training cycle"
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Disable background training (inference-only mode)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Block size (128 dual GPU, 64 single GPU)",
    )
    args = parser.parse_args(argv)
    root = ROOT.resolve()
    paths = WorkspacePaths.from_project(root, require=True)
    if args.token_bin is None:
        args.token_bin = str(resolve_feast_token_bin(root, require=True))
    checkpoint = (
        Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(root)
    )

    config = DarwinOrganismConfig(
        project_root=str(root),
        device=args.device,
        corpus_dir=str(paths.corpus / "approved"),
        tokenizer_dir=str(paths.tokenizer / "f51_bpe_80k"),
        checkpoint_root=str(paths.checkpoints),
        runs_dir=str(paths.runs),
        block_size=args.block_size,
        heartbeat_enabled=True,
        ghost_enabled=True,
        jepa_enabled=True,
        curiosity_enabled=True,
        spider_enabled=True,
    )
    organism = DarwinOrganism(config, str(root / args.config))
    organism.bootstrap(token_bin=args.token_bin, resume=str(checkpoint))

    serve_organism(
        organism,
        host=args.host,
        port=args.port,
        train=not args.no_train,
        steps_per_cycle=args.steps,
    )


if __name__ == "__main__":
    main()
