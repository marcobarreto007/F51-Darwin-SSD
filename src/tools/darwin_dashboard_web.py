#!/usr/bin/env python3
"""
F51 Darwin-X Mission Control — Web Dashboard (FastAPI + SSE + htmx + Chart.js)

Usage:
  .venv_nitro\\Scripts\\python.exe src\\tools\\darwin_dashboard_web.py
  .venv_nitro\\Scripts\\python.exe src\\tools\\darwin_dashboard_web.py --lineage 100m --port 8000

Dependencies (install once):
  .venv_nitro\\Scripts\\pip.exe install fastapi uvicorn

Architecture:
  - SSE (Server-Sent Events) for real-time push: hero, gpu, training, chart_point, etc.
  - SQLite WAL for historical metrics persistence
  - In-memory deques (maxlen=7200) for live chart data
  - Jinja2 template rendering for initial page load
  - Regex-based DIARIO_DE_BORDO.md parser for agent timeline
  - Advisory file lock (msvcrt) for logbook appends
  - Zero dependency on PyTorch or f51_darwin modules
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "src" / "configs" / "darwin_x_100m.yaml"
DEFAULT_CKPT_ROOT = ROOT / "workspace" / "03_CHECKPOINTS_100M_FULL_V9"
DEFAULT_BLOCKCHAIN_FILE = ROOT / "workspace" / "runtime" / "organism" / "blockchain" / "blocks.jsonl"
LOGBOOK_FILE = ROOT / "DIARIO_DE_BORDO.md"
DASHBOARD_DIR = ROOT / "workspace" / "runtime" / "dashboard"
TEMPLATES_DIR = ROOT / "src" / "tools" / "templates"
STATIC_DIR = ROOT / "src" / "tools" / "static"


# ═══════════════════════════════════════════════════════════════════════════════
# DashboardConfig
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DashboardConfig:
    project_root: Path = ROOT
    checkpoint_root: Path = DEFAULT_CKPT_ROOT
    blockchain_file: Path = DEFAULT_BLOCKCHAIN_FILE
    ledger_file: Path = DEFAULT_CKPT_ROOT / "causal_events.jsonl"
    dopamine_file: Path = DEFAULT_CKPT_ROOT / "dopamine_status.txt"
    pointer_file: Path = DEFAULT_CKPT_ROOT / "organism_latest.json"
    config_file: Path = DEFAULT_CONFIG
    logbook_file: Path = LOGBOOK_FILE
    dashboard_dir: Path = DASHBOARD_DIR
    sqlite_path: Path = DASHBOARD_DIR / "metrics.db"
    runs_file: Path = DASHBOARD_DIR / "runs.jsonl"
    timeline_cache: Path = DASHBOARD_DIR / "agent_timeline.json"
    host: str = "127.0.0.1"
    port: int = 8000
    poll_interval_s: float = 2.0
    lineage: str = "100m"
    skip_gpu: bool = False


def resolve_paths(lineage: str | None = None) -> DashboardConfig:
    cfg = DashboardConfig()
    if lineage:
        cfg.lineage = lineage
    cfg.dashboard_dir.mkdir(parents=True, exist_ok=True)

    # Validate minimum required files exist
    if not cfg.logbook_file.exists():
        print(f"[WARN] DIARIO_DE_BORDO.md not found at {cfg.logbook_file}")
    if not TEMPLATES_DIR.exists():
        print(f"[FATAL] Templates directory not found: {TEMPLATES_DIR}")
        sys.exit(1)

    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# SQLite Database Manager
# ═══════════════════════════════════════════════════════════════════════════════

class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def open(self) -> None:
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.execute("PRAGMA journal_mode=WAL")
        self.execute("PRAGMA synchronous=NORMAL")
        self.execute("PRAGMA foreign_keys=ON")
        self.execute("PRAGMA busy_timeout=5000")
        self.execute("PRAGMA cache_size=-8000")
        self.execute("PRAGMA temp_store=MEMORY")
        self.execute("PRAGMA mmap_size=268435456")
        self._create_tables()

    def _create_tables(self) -> None:
        self.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_key      TEXT    NOT NULL,
                value           REAL,
                timestamp_unix_s REAL  NOT NULL,
                cycle           INTEGER,
                step            INTEGER,
                source          TEXT    NOT NULL DEFAULT 'poll',
                created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)
        self.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_key_ts
                ON metrics(metric_key, timestamp_unix_s)
        """)
        self.execute("""
            CREATE TABLE IF NOT EXISTS gpu_snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                gpu_index       INTEGER NOT NULL,
                timestamp_unix_s REAL   NOT NULL,
                util_pct        INTEGER,
                mem_used_mib    INTEGER,
                mem_total_mib   INTEGER,
                temp_c          INTEGER,
                created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)
        self.execute("""
            CREATE INDEX IF NOT EXISTS idx_gpu_ts ON gpu_snapshots(timestamp_unix_s)
        """)
        self.execute("""
            CREATE TABLE IF NOT EXISTS checkpoint_history (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle               INTEGER NOT NULL,
                step                INTEGER NOT NULL,
                filename            TEXT    NOT NULL,
                size_bytes          INTEGER,
                base_checkpoint_id  TEXT,
                sha256              TEXT,
                saved_at            TEXT    NOT NULL,
                detected_at_unix_s  REAL   NOT NULL,
                created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)
        self.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_cp_cycle_step
                ON checkpoint_history(cycle, step)
        """)
        self.execute("""
            CREATE TABLE IF NOT EXISTS dashboard_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type  TEXT    NOT NULL,
                payload     TEXT,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)  # type: ignore[union-attr]

    def executemany(self, sql: str, params: list[tuple]) -> sqlite3.Cursor:
        return self.conn.executemany(sql, params)  # type: ignore[union-attr]

    def insert_metrics_batch(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.executemany(
            """INSERT OR IGNORE INTO metrics
               (metric_key, value, timestamp_unix_s, cycle, step, source)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [(r["metric_key"], r["value"], r["ts"], r.get("cycle"), r.get("step"), "poll") for r in rows],
        )
        self.conn.commit()  # type: ignore[union-attr]

    def insert_gpu_batch(self, rows: list[dict], ts: float) -> None:
        if not rows:
            return
        self.executemany(
            """INSERT INTO gpu_snapshots
               (gpu_index, timestamp_unix_s, util_pct, mem_used_mib, mem_total_mib, temp_c)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [(int(r["idx"]), ts, r.get("util"), r.get("mem_used"), r.get("mem_total"), r.get("temp")) for r in rows],
        )
        self.conn.commit()  # type: ignore[union-attr]

    def query_history(self, metric_key: str, from_ts: float, to_ts: float, limit: int = 1000) -> list[dict]:
        rows = self.execute(
            """SELECT timestamp_unix_s, value FROM metrics
               WHERE metric_key = ? AND timestamp_unix_s >= ? AND timestamp_unix_s <= ?
               ORDER BY timestamp_unix_s ASC LIMIT ?""",
            (metric_key, from_ts, to_ts, limit),
        ).fetchall()
        return [{"ts": r[0], "value": r[1]} for r in rows]

    def checkpoint_wal(self) -> None:
        self.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        if self.conn:
            self.checkpoint_wal()
            self.conn.close()
            self.conn = None


# ═══════════════════════════════════════════════════════════════════════════════
# Ring Buffer (in-memory deque for chart data)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RingBuffer:
    metric_key: str
    capacity: int = 7200
    timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=7200))
    values: deque[float] = field(default_factory=lambda: deque(maxlen=7200))

    def push(self, ts: float, value: float | None) -> None:
        if value is None or not math.isfinite(value):
            return
        self.timestamps.append(ts)
        self.values.append(value)

    def last_value(self) -> float | None:
        return self.values[-1] if self.values else None

    def last_n(self, n: int) -> list[dict]:
        n = min(n, len(self.timestamps))
        result = []
        for i in range(len(self.timestamps) - n, len(self.timestamps)):
            result.append({"ts": self.timestamps[i], "value": self.values[i]})
        return result

    def range(self, from_ts: float, to_ts: float) -> list[dict]:
        return [
            {"ts": self.timestamps[i], "value": self.values[i]}
            for i in range(len(self.timestamps))
            if from_ts <= self.timestamps[i] <= to_ts
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

_START_TIME = time.time()

def _iso8601(ts: float | None = None) -> str:
    if ts is None:
        ts = time.time()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _iso8601_local(ts: float | None = None) -> str:
    if ts is None:
        ts = time.time()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def _human_duration(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def _sha256_short(path: Path) -> str:
    if not path.exists():
        return "--"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16] + "..."


# ═══════════════════════════════════════════════════════════════════════════════
# Data Collectors  (file parsers, subprocess, no torch dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def read_dopamine(path: Path) -> dict[str, Any]:
    """Parse dopamine_status.txt key=value pairs. Returns {} on any error."""
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8").strip()
        result: dict[str, Any] = {}
        for part in text.split():
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            try:
                if k == "step":
                    result[k] = int(v)
                else:
                    result[k] = round(float(v), 6)
            except ValueError:
                pass
        return result
    except Exception:
        return {}


def read_pointer(path: Path) -> dict[str, Any]:
    """Parse organism_latest.json."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_blockchain(path: Path) -> dict[str, Any]:
    """Read blocks.jsonl: count lines, parse last block."""
    result: dict[str, Any] = {
        "blocks": 0, "head_hash": "--", "head_hash_short": "--",
        "last_cycle": None, "last_step": None,
        "size_kb": 0.0, "status": "NO FILE",
        "timestamp": _iso8601(),
    }
    if not path.exists():
        return result
    try:
        stat = path.stat()
        result["size_kb"] = round(stat.st_size / 1024, 1)
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            result["status"] = "EMPTY"
            return result
        lines = [ln for ln in text.split("\n") if ln.strip()]
        result["blocks"] = len(lines)
        last = json.loads(lines[-1])
        bh = last.get("block_hash", "")
        result["head_hash"] = bh
        result["head_hash_short"] = bh[:8] if len(bh) >= 8 else bh
        hdr = last.get("header", {})
        result["last_cycle"] = hdr.get("cycle")
        result["last_step"] = hdr.get("optimizer_step")
        result["status"] = "ACTIVE"
    except json.JSONDecodeError:
        result["status"] = "PARSE ERROR"
    except Exception:
        result["status"] = "PARSE ERROR"
    return result


def read_ledger(path: Path) -> dict[str, Any]:
    """Read causal_events.jsonl head."""
    if not path.exists():
        return {"lines": 0, "size_kb": 0, "head_hash": "--", "status": "NO FILE"}
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return {"lines": 0, "size_kb": 0, "head_hash": "--", "status": "EMPTY"}
        lines = [ln for ln in text.split("\n") if ln.strip()]
        last = json.loads(lines[-1])
        return {
            "lines": len(lines),
            "size_kb": round(path.stat().st_size / 1024, 1),
            "head_hash": last.get("event_hash", "--")[:16] + "...",
            "status": "ACTIVE",
        }
    except Exception:
        return {"lines": 0, "size_kb": 0, "head_hash": "--", "status": "PARSE ERROR"}


async def read_gpu() -> dict[str, Any]:
    """Query nvidia-smi via asyncio subprocess. Returns {gpus: [...], timestamp: ...}."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        gpus = []
        for line in stdout.decode("utf-8", errors="replace").strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gpus.append({
                "idx": parts[0],
                "name": parts[1],
                "util": int(parts[2]) if parts[2] not in ("[Not Supported]", "") else 0,
                "mem_used": int(parts[3]) if parts[3] != "[Not Supported]" else 0,
                "mem_total": int(parts[4]) if parts[4] != "[Not Supported]" else 0,
                "temp": int(parts[5]) if parts[5] not in ("[Not Supported]", "") else None,
            })
        return {"gpus": gpus, "timestamp": _iso8601()}
    except Exception:
        return {"gpus": [], "timestamp": _iso8601()}


def check_training_process() -> bool:
    """Check if a Python process with 'darwin_organism' or 'organism' in command line is running."""
    try:
        result = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "commandline"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout:
            return any(
                keyword in result.stdout.lower()
                for keyword in ["darwin_organism", "organism.py", "darwin_x"]
            )
    except Exception:
        pass
    # Fallback: tasklist
    try:
        result = subprocess.run(
            ["tasklist", "/v", "/fo", "csv"],
            capture_output=True, text=True, timeout=8,
        )
        if result.stdout:
            return "organism" in result.stdout.lower() or "darwin" in result.stdout.lower()
    except Exception:
        pass
    return False


def compute_hashes(config: DashboardConfig) -> dict[str, Any]:
    """Compute SHA-256 fingerprints for integrity section."""
    ptr = read_pointer(config.pointer_file)
    cp_filename = ptr.get("path", "")
    cp_path = config.checkpoint_root / cp_filename if cp_filename else None

    def _hash_or_missing(p: Path | None, label: str) -> dict:
        if p and p.exists():
            h = _sha256_short(p)
            return {"hash_short": h, "hash_full": None, "exists": True, "size_mb": round(p.stat().st_size / (1024**2), 2)}
        return {"hash_short": "--", "hash_full": None, "exists": False, "size_mb": None}

    return {
        "checkpoint": _hash_or_missing(cp_path, "checkpoint"),
        "config": _hash_or_missing(config.config_file, "config"),
        "ledger": _hash_or_missing(config.ledger_file, "ledger"),
        "blocks": _hash_or_missing(config.blockchain_file, "blocks"),
        "script": _hash_or_missing(ROOT / "src" / "scripts" / "start_100m_auto.ps1", "script"),
        "verified_at": _iso8601(),
    }


def read_system_health(training_uptime_s: float | None, training_running: bool) -> dict[str, Any]:
    """Collect system health metrics using psutil if available."""
    try:
        import psutil
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage(str(ROOT))
        return {
            "ram_used_gb": round(ram.used / (1024**3), 1),
            "ram_total_gb": round(ram.total / (1024**3), 0),
            "ram_pct": round(ram.percent, 1),
            "disk_used_gb": round(disk.used / (1024**3), 1),
            "disk_total_gb": round(disk.total / (1024**3), 0),
            "disk_pct": round(disk.percent, 1),
            "disk_free_gb": round(disk.free / (1024**3), 1),
            "process_count": 1 if training_running else 0,
            "process_total_mem_gb": None,
            "uptime": _human_duration(training_uptime_s) if training_running else None,
            "uptime_s": training_uptime_s if training_running else None,
            "dashboard_uptime_s": round(time.time() - _START_TIME, 1),
            "timestamp": _iso8601(),
        }
    except ImportError:
        return {
            "ram_used_gb": None, "ram_total_gb": None, "ram_pct": None,
            "disk_used_gb": None, "disk_total_gb": None, "disk_pct": None,
            "disk_free_gb": None, "process_count": None,
            "process_total_mem_gb": None, "uptime": None, "uptime_s": None,
            "dashboard_uptime_s": round(time.time() - _START_TIME, 1),
            "timestamp": _iso8601(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Logbook Parser  (regex-based, DIARIO_DE_BORDO.md)
# ═══════════════════════════════════════════════════════════════════════════════

_TABLE_ROW_RE = re.compile(r'\|\s*\*\*(.+?)\*\*\s*\|\s*(.+?)\s*\|')
_ENTRY_HEADER_RE = re.compile(
    r'###\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\s+UTC\s*\|\s*(.+?)(?:\s*\(([^)]*)\))?\s*$'
)
_OP_RE = re.compile(r'OPERAÇÃO\s*(?:PRETENDIDA|REALIZADA)?\s*:?\s*(READ|WRITE|EXECUTE|DECIDE|MIXED)', re.IGNORECASE)
_COMMIT_RE = re.compile(r'COMMIT:\s*([a-f0-9]{7,})\s*[-—]\s*(.+)', re.IGNORECASE)
_ENTRADA_RE = re.compile(r'(ENTRADA)', re.IGNORECASE)
_SAIDA_RE = re.compile(r'(SA[ÍI]DA)', re.IGNORECASE)


def parse_estado_atual(text: str) -> dict[str, Any]:
    """Parse ESTADO ATUAL markdown table into a dict."""
    section_start = text.find("## 📍 ESTADO ATUAL")
    if section_start == -1:
        section_start = text.find("## ESTADO ATUAL")
    if section_start == -1:
        return {}
    table_start = text.find("| Campo", section_start)
    if table_start == -1:
        return {}
    table_end = text.find("\n\n", table_start)
    if table_end == -1:
        table_end = len(text)
    table_text = text[table_start:table_end]

    raw: dict[str, str] = {}
    for match in _TABLE_ROW_RE.finditer(table_text):
        raw[match.group(1).strip()] = match.group(2).strip()
    return raw


def parse_agent_entries(text: str, max_entries: int = 30) -> list[dict[str, Any]]:
    """Parse agent entry blocks from DIARIO_DE_BORDO.md."""
    entries: list[dict[str, Any]] = []
    for m in _ENTRY_HEADER_RE.finditer(text):
        date_str, time_str = m.group(1), m.group(2)
        agent_name = m.group(3).strip()

        # Find code block after header
        block_start = text.find("```", m.end())
        if block_start == -1:
            continue
        block_end = text.find("```", block_start + 3)
        if block_end == -1:
            continue
        block = text[block_start + 3:block_end].strip()

        # Determine entry type
        has_entrada = bool(_ENTRADA_RE.search(block))
        has_saida = bool(_SAIDA_RE.search(block))
        if has_entrada and has_saida:
            entry_type = "ENTRADA + SAÍDA"
        elif has_entrada:
            entry_type = "ENTRADA"
        elif has_saida:
            entry_type = "SAÍDA"
        else:
            entry_type = "AUTO"

        # Operation
        op_match = _OP_RE.search(block)
        if op_match:
            operation = op_match.group(1).upper()
        else:
            operation = "MIXED"

        # Summary: first non-empty non-field line
        summary = ""
        for line in block.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if any(stripped.upper().startswith(p) for p in ("OPERAÇÃ", "OPERAÇAO", "ESTADO", "AÇÕES", "OBJETIVO", "COMMIT", "WARNINGS", "ARQUIVOS", "ENTRADA", "SAÍD")):
                continue
            if len(stripped) > 5:
                summary = stripped[:80]
                break
        if not summary:
            obj_match = re.search(r'OBJETIVO:\s*(.+)', block)
            if obj_match:
                summary = obj_match.group(1).strip()[:80]

        # Commit
        commit = None
        cm_match = _COMMIT_RE.search(block)
        if cm_match:
            commit = f"{cm_match.group(1)} — {cm_match.group(2).strip()}"

        # Warnings
        warnings = None
        warn_match = re.search(r'WARNINGS?\s*:?\s*\n(.*?)(?=\n\S|\Z)', block, re.DOTALL)
        if warn_match:
            warnings = warn_match.group(1).strip()

        entries.append({
            "timestamp": f"{date_str[5:]} {time_str}",
            "timestamp_full": f"{date_str}T{time_str}:00Z",
            "agent": agent_name,
            "operation": operation,
            "operation_type": operation.lower(),
            "entry_type": entry_type,
            "summary": summary or "(no summary)",
            "commit": commit,
            "warnings": warnings,
            "estado_encontrado": None,
            "acoes": None,
            "estado_deixado": None,
        })
        if len(entries) >= max_entries:
            break
    return entries


def parse_bugs(text: str) -> list[dict[str, Any]]:
    """Parse Bugs conhecidos section."""
    bugs: list[dict[str, Any]] = []
    section = text.find("## 🔧 Bugs conhecidos")
    if section == -1:
        section = text.find("## Bugs conhecidos")
    if section == -1:
        return bugs
    list_start = text.find("1.", section)
    if list_start == -1:
        return bugs
    list_end = text.find("\n\n", list_start)
    if list_end == -1:
        list_end = len(text)
    list_text = text[list_start:list_end]

    for m in re.finditer(r'(\d+)\.\s+\*\*(.+?)\*\*(?:\s*[:—-]\s*(.+))?', list_text):
        bugs.append({
            "id": int(m.group(1)),
            "title": m.group(2).strip(),
            "description": m.group(3).strip() if m.group(3) else None,
            "status": "known",
        })
    return bugs


def parse_logbook(logbook_path: Path, cache_path: Path) -> dict[str, Any]:
    """Parse entire DIARIO_DE_BORDO.md, with mtime-based caching."""
    if not logbook_path.exists():
        return {"estado_atual": {}, "entries": [], "bugs": [], "parse_error": "Logbook file not found"}

    # Check cache
    if cache_path.exists():
        try:
            cache_mtime = cache_path.stat().st_mtime
            src_mtime = logbook_path.stat().st_mtime
            if cache_mtime >= src_mtime:
                return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        text = logbook_path.read_text(encoding="utf-8")
    except Exception as e:
        return {"estado_atual": {}, "entries": [], "bugs": [], "parse_error": str(e)}

    estado_atual = parse_estado_atual(text)
    entries = parse_agent_entries(text)
    bugs = parse_bugs(text)

    result = {
        "estado_atual": estado_atual,
        "entries": entries,
        "bugs": bugs,
        "entry_count": len(entries),
        "parsed_at": _iso8601(),
    }

    # Write cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    return result


def append_logbook_event(logbook_path: Path, event_type: str, payload: str, timeout_s: float = 2.0) -> bool:
    """Append dashboard event to DIARIO_DE_BORDO.md with advisory file lock (Windows msvcrt)."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = f"""

### {timestamp} | Dashboard (auto)

```
{event_type.upper()}
{payload}
```
"""
    try:
        import msvcrt
        with open(logbook_path, "a", encoding="utf-8") as f:
            start = time.monotonic()
            while True:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() - start > timeout_s:
                        return False
                    time.sleep(0.05)
            try:
                f.write(entry)
                f.flush()
                os.fsync(f.fileno())
            finally:
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# SSE Manager  (broadcast to all connected clients)
# ═══════════════════════════════════════════════════════════════════════════════

class SSEManager:
    def __init__(self):
        self._queues: list[asyncio.Queue] = []
        self._lock = asyncio.Lock()
        self._start_time = _START_TIME

    async def add_client(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._queues.append(q)
        return q

    async def remove_client(self, q: asyncio.Queue) -> None:
        async with self._lock:
            if q in self._queues:
                self._queues.remove(q)

    async def broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        payload = f"event: {event_type}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
        async with self._lock:
            dead: list[asyncio.Queue] = []
            for q in self._queues:
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    dead.append(q)
            for q in dead:
                self._queues.remove(q)

    @property
    def client_count(self) -> int:
        return len(self._queues)


# ═══════════════════════════════════════════════════════════════════════════════
# Global State
# ═══════════════════════════════════════════════════════════════════════════════

sse_manager = SSEManager()
db = DatabaseManager(DASHBOARD_DIR / "metrics.db")
_shutdown_requested = False
_training_started_at: float | None = None
_training_running = False
_previous_metrics: dict[str, Any] = {}
_tick_count = 0
_last_gpu_ts = 0.0
_cached_gpu: dict[str, Any] = {"gpus": [], "timestamp": ""}
_last_hash_compute = 0.0
_cached_hashes: dict[str, Any] = {}
_last_logbook_mtime = 0.0
_cached_timeline: dict[str, Any] = {"entries": [], "entry_count": 0}
_training_uptime_s: float | None = None
_last_dopamine_ts = 0.0
_last_pointer_mtime = 0.0
_cached_pointer: dict[str, Any] = {}

# In-memory deques for charts
ring_buffers: dict[str, RingBuffer] = {
    "loss": RingBuffer("loss"),
    "ppl": RingBuffer("ppl"),
    "tok_s": RingBuffer("tok_s"),
    "dope": RingBuffer("dope"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Main Poll Cycle
# ═══════════════════════════════════════════════════════════════════════════════

async def poll_cycle(config: DashboardConfig) -> dict[str, Any]:
    """Single poll cycle. Returns dict of SSE events to broadcast."""
    global _tick_count, _last_gpu_ts, _cached_gpu, _last_hash_compute
    global _cached_hashes, _previous_metrics, _training_running, _training_started_at
    global _training_uptime_s, _last_dopamine_ts, _last_pointer_mtime, _cached_pointer
    global _last_logbook_mtime, _cached_timeline

    _tick_count += 1
    now = time.time()
    events: dict[str, Any] = {}

    # ── Dopamine ──
    dop = read_dopamine(config.dopamine_file)
    if dop:
        _last_dopamine_ts = now

    # ── Pointer ──
    ptr: dict[str, Any] = {}
    if config.pointer_file.exists():
        try:
            pstat = config.pointer_file.stat()
            if pstat.st_mtime != _last_pointer_mtime:
                ptr = read_pointer(config.pointer_file)
                _cached_pointer = ptr
                _last_pointer_mtime = pstat.st_mtime
            else:
                ptr = _cached_pointer
        except Exception:
            ptr = _cached_pointer

    # ── Training process detection ──
    process_running = check_training_process()
    was_running = _training_running
    _training_running = process_running
    if process_running and not was_running:
        _training_started_at = now
        events["training_status"] = {"status": "running", "changed_at": _iso8601(now), "since_s": 0}
        # Append start event to logbook and runs.jsonl
        append_logbook_event(config.logbook_file, "training_start",
                             f"Dashboard detected training process at {_iso8601_local(now)}")
        _append_run_event(config, "started", now)
    elif not process_running and was_running:
        _training_started_at = None
        _training_uptime_s = None
        events["training_status"] = {"status": "stopped", "changed_at": _iso8601(now), "since_s": 0}
        _append_run_event(config, "stopped", now)

    if process_running and _training_started_at:
        _training_uptime_s = now - _training_started_at

    # ── Hero event ──
    hero = _build_hero(dop, ptr, now)
    events["hero"] = hero

    # ── Training metrics event ──
    train = _build_training(dop, now)
    events["training"] = train

    # ── Chart point ──
    cp = _build_chart_point(dop, now)
    if cp:
        events["chart_point"] = cp

    # ── GPU (cached for 2s) ──
    if now - _last_gpu_ts >= config.poll_interval_s:
        _cached_gpu = await read_gpu()
        _last_gpu_ts = now
    events["gpu"] = _cached_gpu

    # ── Medium tier (every 5th tick ~10s) ──
    if _tick_count % 5 == 0:
        events["blockchain"] = read_blockchain(config.blockchain_file)
        events["health"] = read_system_health(_training_uptime_s, process_running)

    # ── Slow tier (every 30th tick ~60s) ──
    if _tick_count % 30 == 0:
        if now - _last_hash_compute > 55.0:
            _cached_hashes = compute_hashes(config)
            _last_hash_compute = now
        events["hashes"] = _cached_hashes

        # Re-parse logbook if it changed
        if config.logbook_file.exists():
            try:
                lb_mtime = config.logbook_file.stat().st_mtime
                if lb_mtime != _last_logbook_mtime:
                    _cached_timeline = parse_logbook(config.logbook_file, config.timeline_cache)
                    _last_logbook_mtime = lb_mtime
            except Exception:
                pass
        events["timeline"] = _cached_timeline

        # Organ states (from pointer only, no torch)
        events["organs"] = _build_organs(config, ptr)

    # ── Persist to SQLite ──
    _write_metrics_db(dop, now, _cached_gpu.get("gpus", []))

    return events


def _build_hero(dop: dict, ptr: dict, now: float) -> dict[str, Any]:
    prev_loss = _previous_metrics.get("loss")
    cur_loss = dop.get("loss")
    loss_trend = "none"
    if prev_loss is not None and cur_loss is not None:
        diff = cur_loss - prev_loss
        loss_trend = "down" if diff < -0.0001 else ("up" if diff > 0.0001 else "flat")
    return {
        "loss": cur_loss,
        "loss_previous": prev_loss,
        "loss_trend": loss_trend,
        "cycle": dop.get("cycle", ptr.get("cycle")),
        "step": dop.get("step", ptr.get("step")),
        "ckpt_cycle": ptr.get("cycle"),
        "ckpt_step": ptr.get("step"),
        "version": ptr.get("checkpoint_version"),
        "timestamp": _iso8601_local(now),
    }


def _build_training(dop: dict, now: float) -> dict[str, Any]:
    cur_loss = dop.get("loss")
    ppl = math.exp(cur_loss) if cur_loss is not None else None
    prev_loss = _previous_metrics.get("loss")
    prev_ppl = _previous_metrics.get("ppl")
    prev_tok = _previous_metrics.get("tok_s")
    cur_tok = dop.get("tok_s")
    return {
        "loss": cur_loss,
        "ppl": round(ppl, 1) if ppl is not None else None,
        "tok_s": cur_tok,
        "jepa_raw": dop.get("jepa_raw"),
        "dope": dop.get("dope"),
        "delta": dop.get("delta"),
        "gmc": dop.get("gmc"),
        "loss_delta": round(cur_loss - prev_loss, 6) if cur_loss is not None and prev_loss is not None else None,
        "ppl_delta": round(ppl - prev_ppl, 1) if ppl is not None and prev_ppl is not None else None,
        "tok_s_delta": round(cur_tok - prev_tok, 1) if cur_tok is not None and prev_tok is not None else None,
        "timestamp": _iso8601(now),
    }


def _build_chart_point(dop: dict, now: float) -> dict[str, Any] | None:
    cur_loss = dop.get("loss")
    ppl = math.exp(cur_loss) if cur_loss is not None else None
    point = {
        "ts": now,
        "loss": cur_loss,
        "ppl": round(ppl, 1) if ppl is not None else None,
        "tok_s": dop.get("tok_s"),
        "dope": dop.get("dope"),
    }
    # Push to ring buffers
    ring_buffers["loss"].push(now, cur_loss)
    ring_buffers["ppl"].push(now, round(ppl, 1) if ppl is not None else None)
    ring_buffers["tok_s"].push(now, dop.get("tok_s"))
    ring_buffers["dope"].push(now, dop.get("dope"))
    # Update previous metrics
    _previous_metrics["loss"] = cur_loss
    _previous_metrics["ppl"] = round(ppl, 1) if ppl is not None else None
    _previous_metrics["tok_s"] = dop.get("tok_s")
    return point


def _build_organs(config: DashboardConfig, ptr: dict) -> dict[str, Any]:
    """Build organ states from config toggles + pointer base_checkpoint_id."""
    organs: dict[str, Any] = {
        "core": True, "moe": True, "jepa": True, "mtp": True,
        "spider_calibration": True, "spider_sense": True,
        "gaba": True, "inter_hemispheric": True, "ttm_residual": True,
        "heartbeat": True, "ghost": True, "curiosity": False,
        "dae": True, "nitro": True, "decision_engine": False,
        "unified_mesh": True, "sleep": True,
    }
    base_id = ptr.get("base_checkpoint_id", "")
    result: dict[str, Any] = {}
    for name, active in organs.items():
        id_short = base_id[-20:] if base_id else None
        result[name] = {
            "active": active,
            "status": "active" if active else "dead",
            "identity_hash": id_short,
        }
    result["timestamp"] = _iso8601()
    return result


def _write_metrics_db(dop: dict, now: float, gpus: list[dict]) -> None:
    """Write current poll's metrics to SQLite."""
    cycle = dop.get("cycle")
    step = dop.get("step")
    rows = []
    for key in ["loss", "ppl", "tok_s", "dope", "jepa_raw", "delta", "gmc"]:
        val = None
        if key == "ppl":
            loss = dop.get("loss")
            val = round(math.exp(loss), 1) if loss is not None else None
        elif key in dop:
            val = dop[key]
        if val is not None:
            rows.append({"metric_key": key, "value": val, "ts": now, "cycle": cycle, "step": step})
    try:
        db.insert_metrics_batch(rows)
    except Exception:
        pass  # Non-fatal: keep deques alive
    try:
        db.insert_gpu_batch(gpus, now)
    except Exception:
        pass


def _append_run_event(config: DashboardConfig, event: str, ts: float) -> None:
    """Append training start/stop event to runs.jsonl."""
    try:
        ptr = _cached_pointer or read_pointer(config.pointer_file)
        run_entry = {
            "event": event,
            "timestamp": _iso8601(ts),
            "cycle": ptr.get("cycle"),
            "step": ptr.get("step"),
            "base_checkpoint_id": ptr.get("base_checkpoint_id"),
        }
        config.runs_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config.runs_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_entry, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Background Async Loops
# ═══════════════════════════════════════════════════════════════════════════════

async def data_collector_loop(config: DashboardConfig) -> None:
    """Main poll loop. Runs every poll_interval_s."""
    global _shutdown_requested
    while not _shutdown_requested:
        loop_start = time.monotonic()
        try:
            events = await poll_cycle(config)
            for event_type, payload in events.items():
                await sse_manager.broadcast(event_type, payload)
        except Exception as e:
            try:
                db.execute(
                    "INSERT INTO dashboard_events (event_type, payload) VALUES (?, ?)",
                    ("poll_error", str(e)[:500]),
                )
            except Exception:
                pass
        elapsed = time.monotonic() - loop_start
        await asyncio.sleep(max(0.0, config.poll_interval_s - elapsed))


async def heartbeat_loop() -> None:
    """Send SSE heartbeat every 15s."""
    global _shutdown_requested
    while not _shutdown_requested:
        await asyncio.sleep(15.0)
        try:
            await sse_manager.broadcast("heartbeat", {
                "server_time_utc": _iso8601(),
                "dashboard_uptime_s": round(time.time() - _START_TIME, 1),
                "client_count": sse_manager.client_count,
            })
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI Application
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="F51 Darwin-X Mission Control",
    version="2.0.0",
    docs_url=None,
    redoc_url=None,
)

# Static files mount
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    print(f"[WARN] Static directory not found at {STATIC_DIR}")

# Jinja2 templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR)) if TEMPLATES_DIR.exists() else None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Serve the Jinja2 dashboard page with SSR-seeded initial data."""
    if templates is None:
        return HTMLResponse("<h1>Error: templates directory not found</h1>", status_code=500)

    # Build initial context from latest cached data (or load fresh)
    config = request.app.state.config  # type: ignore[attr-defined]
    dop = read_dopamine(config.dopamine_file)
    ptr = _cached_pointer or read_pointer(config.pointer_file)
    gpu = _cached_gpu
    bc = read_blockchain(config.blockchain_file)
    timeline = _cached_timeline or parse_logbook(config.logbook_file, config.timeline_cache)
    hashes = _cached_hashes or compute_hashes(config)

    cur_loss = dop.get("loss")
    ppl_val = round(math.exp(cur_loss), 1) if cur_loss is not None else None

    initial = {
        "hero": {
            "loss": cur_loss,
            "cycle": dop.get("cycle", ptr.get("cycle")),
            "step": dop.get("step", ptr.get("step")),
            "version": ptr.get("checkpoint_version"),
            "timestamp": _iso8601_local(),
        },
        "gpus": gpu.get("gpus", []),
        "training": {
            "loss": cur_loss,
            "ppl": ppl_val,
            "tok_s": dop.get("tok_s"),
            "jepa_raw": dop.get("jepa_raw"),
            "dope": dop.get("dope"),
            "delta": dop.get("delta"),
            "gmc": dop.get("gmc"),
            "loss_delta": None,
            "ppl_delta": None,
            "tok_s_delta": None,
        },
        "blockchain": bc,
        "hashes": hashes,
        "organs": _build_organs(config, ptr),
        "system": read_system_health(_training_uptime_s, _training_running),
        "timeline": timeline.get("entries", []),
        "bugs": timeline.get("bugs", []),
        "estado_atual": timeline.get("estado_atual", {}),
        "render_time_utc": _iso8601(),
        "lineage": config.lineage,
    }
    return templates.TemplateResponse("dashboard.html", {"request": request, **initial})


@app.get("/api/status")
async def api_status(request: Request):
    """Lightweight JSON health check."""
    config = request.app.state.config  # type: ignore[attr-defined]
    return {
        "dashboard": {
            "version": "2.0.0",
            "uptime_s": round(time.time() - _START_TIME, 1),
            "connected_clients": sse_manager.client_count,
            "polls_total": _tick_count,
        },
        "training": {
            "status": "running" if _training_running else "stopped",
            "cycle": _cached_pointer.get("cycle"),
            "step": _cached_pointer.get("step"),
        },
        "system": {"process_count": 1 if _training_running else 0},
    }


@app.get("/api/metrics")
async def api_metrics(metric: str, from_ts: float | None = None, to_ts: float | None = None, limit: int = 1000):
    """Query historical metrics from in-memory buffers + SQLite."""
    now = time.time()
    frm = from_ts if from_ts else (now - 3600)
    to = to_ts if to_ts else now

    buf = ring_buffers.get(metric)
    if buf and buf.count > 0:
        points = buf.range(frm, to)
    else:
        points = db.query_history(metric, frm, to, limit)
    return {
        "metric": metric,
        "points": points,
        "count": len(points),
        "from": _iso8601(frm),
        "to": _iso8601(to),
    }


@app.get("/api/runs")
async def api_runs(request: Request, limit: int = 20):
    """Training run history from runs.jsonl."""
    config = request.app.state.config  # type: ignore[attr-defined]
    runs = []
    if config.runs_file.exists():
        try:
            for line in config.runs_file.read_text(encoding="utf-8").strip().split("\n"):
                if line.strip():
                    runs.append(json.loads(line))
        except Exception:
            pass
    return {"runs": runs[-limit:], "count": len(runs)}


@app.get("/api/sections/{name}")
async def api_section(request: Request, name: str):
    """htmx partial HTML snippets."""
    config = request.app.state.config  # type: ignore[attr-defined]
    if templates is None:
        return HTMLResponse("<div>Error: no templates</div>", status_code=500)

    try:
        if name == "hero":
            dop = read_dopamine(config.dopamine_file)
            ptr = _cached_pointer or read_pointer(config.pointer_file)
            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "hero": _build_hero(dop, ptr, time.time()),
                "_section": "hero",
            })
        elif name == "gpu":
            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "gpus": _cached_gpu.get("gpus", []),
                "_section": "gpu",
            })
        elif name == "blockchain":
            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "blockchain": read_blockchain(config.blockchain_file),
                "_section": "blockchain",
            })
        elif name == "hashes":
            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "hashes": _cached_hashes or compute_hashes(config),
                "_section": "hashes",
            })
        elif name == "timeline":
            timeline = parse_logbook(config.logbook_file, config.timeline_cache)
            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "timeline": timeline.get("entries", []),
                "_section": "timeline",
            })
        elif name == "health":
            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "system": read_system_health(_training_uptime_s, _training_running),
                "_section": "health",
            })
        elif name == "organs":
            ptr = _cached_pointer or read_pointer(config.pointer_file)
            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "organs": _build_organs(config, ptr),
                "_section": "organs",
            })
        else:
            return HTMLResponse(f"<div>Unknown section: {name}</div>", status_code=404)
    except Exception as e:
        return HTMLResponse(f"<div class='section-error'>Error: {e}</div>", status_code=500)


@app.get("/events")
async def sse_stream(request: Request):
    """SSE endpoint — Server-Sent Events stream."""
    queue = await sse_manager.add_client()

    async def event_generator():
        try:
            # Send initial retry:2000 for auto-reconnect
            yield "retry: 2000\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield payload
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            await sse_manager.remove_client(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize database, parse logbook, verify data files, spawn background tasks."""
    global _cached_pointer, _last_pointer_mtime, _cached_timeline, _last_logbook_mtime

    config: DashboardConfig = app.state.config  # type: ignore[attr-defined]

    # Phase 1: Open SQLite
    print("[STARTUP] Opening SQLite database...")
    db.open()

    # Phase 2: Parse logbook
    print("[STARTUP] Parsing DIARIO_DE_BORDO.md...")
    if config.logbook_file.exists():
        _cached_timeline = parse_logbook(config.logbook_file, config.timeline_cache)
        _last_logbook_mtime = config.logbook_file.stat().st_mtime

    # Phase 3: Read current state
    print("[STARTUP] Loading initial data...")
    if config.pointer_file.exists():
        _cached_pointer = read_pointer(config.pointer_file)
        _last_pointer_mtime = config.pointer_file.stat().st_mtime

    # Phase 4: Detect training state
    is_running = check_training_process()
    if is_running:
        print("[STARTUP] Training process detected — recording start in runs.jsonl")
        _append_run_event(config, "startup_detected", time.time())

    # Phase 5: Load GPU state
    print("[STARTUP] Querying GPU state...")
    gpu_data = await read_gpu()
    global _cached_gpu, _last_gpu_ts
    _cached_gpu = gpu_data
    _last_gpu_ts = time.time()

    # Phase 6: Compute hashes
    print("[STARTUP] Computing SHA-256 hashes...")
    global _cached_hashes, _last_hash_compute
    _cached_hashes = compute_hashes(config)
    _last_hash_compute = time.time()

    # Phase 7: Spawn background tasks
    print("[STARTUP] Starting background poll loops...")
    asyncio.create_task(data_collector_loop(config))
    asyncio.create_task(heartbeat_loop())

    # Phase 8: Print banner
    dop = read_dopamine(config.dopamine_file)
    ptr = _cached_pointer
    tl = _cached_timeline
    bc = read_blockchain(config.blockchain_file)
    dop_status = f"step={dop.get('step','?')} loss={dop.get('loss','?')}" if dop else "NOT FOUND (training not running?)"
    bc_status = f"{bc.get('blocks', 0)} blocks" if bc.get('status') == 'ACTIVE' else bc.get('status', '?')
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║    F51 Darwin-X Mission Control — Web Dashboard            ║
║    http://{config.host}:{config.port}/dashboard              {" " * (24 - len(str(config.port))) }║
╠══════════════════════════════════════════════════════════════╣
║    Lineage:      {config.lineage.upper():<44s}║
║    Checkpoint:   cycle {ptr.get('cycle', '?')}, step {ptr.get('step', '?')}{" " * (32 - len(str(ptr.get('step', '?'))))}║
║    Dopamine:     {dop_status:<44s}║
║    Blockchain:   {bc_status:<44s}║
║    SQLite:       {db.db_path}{" " * (44 - len(str(db.db_path)))}║
║    Logbook:      {tl.get('entry_count', 0)} entries parsed{" " * (32 - len(str(tl.get('entry_count', 0))))}║
╠══════════════════════════════════════════════════════════════╣
║    Ctrl+C to stop                                           ║
╚══════════════════════════════════════════════════════════════╝""")


@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown: cancel tasks, flush buffers, close DB."""
    global _shutdown_requested
    print("[SHUTDOWN] Stopping background tasks...")
    _shutdown_requested = True

    # Give tasks time to finish current cycle
    await asyncio.sleep(0.5)

    # Flush ring buffers to SQLite
    print("[SHUTDOWN] Flushing ring buffers to SQLite...")
    now = time.time()
    rows = []
    for key, buf in ring_buffers.items():
        if buf.count == 0:
            continue
        for i in range(buf.count):
            rows.append({"metric_key": key, "value": buf.values[i], "ts": buf.timestamps[i], "cycle": None, "step": None})
    try:
        db.insert_metrics_batch(rows)
    except Exception:
        pass

    # SQLite WAL checkpoint
    print("[SHUTDOWN] Closing SQLite...")
    db.close()

    # Broadcast shutdown
    try:
        await sse_manager.broadcast("shutdown", {"reason": "dashboard_stopped", "at": _iso8601()})
    except Exception:
        pass

    print("[SHUTDOWN] Dashboard stopped.")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="F51 Darwin-X Mission Control Dashboard")
    parser.add_argument("--lineage", default=None, choices=["100m", "600m", "1.6b"], help="Model lineage")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--no-gpu", action="store_true", help="Skip nvidia-smi")
    args = parser.parse_args()

    # Override from env
    host = os.environ.get("F51_DASHBOARD_HOST", args.host)
    port = int(os.environ.get("F51_DASHBOARD_PORT", str(args.port)))
    lineage = os.environ.get("F51_LINEAGE", args.lineage) or None
    skip_gpu = args.no_gpu or os.environ.get("F51_DASHBOARD_NO_GPU") == "1"

    config = resolve_paths(lineage)
    config.host = host
    config.port = port
    config.skip_gpu = skip_gpu

    # Attach config to app state for route access
    app.state.config = config  # type: ignore[attr-defined]

    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
