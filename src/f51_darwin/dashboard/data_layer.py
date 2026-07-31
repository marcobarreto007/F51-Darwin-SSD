"""
F51 Darwin-X Mission Control — Data Layer
==========================================
SQLite persistence, logbook parsing, ring buffers, process detection.

Provides the four core data-layer classes used by the FastAPI dashboard:
  SQLiteManager     — WAL-mode SQLite with 4 tables, batched inserts, vacuum
  LogbookParser     — Regex parser for DIARIO_DE_BORDO.md (cached, reload-on-mtime)
  MetricsRingBuffer — In-memory deques (maxlen=7200) for live chart feeds
  RunDetector       — psutil-based training-process detection

All classes are synchronous (no asyncio); the caller wraps blocking calls in
asyncio.to_thread() where needed.

Author: F51 Darwin-X Dashboard Wave 3
Date:   2026-07-24
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RING_MAXLEN: int = 7200          # 4 hours at 2 s poll interval
GPU_SNAPSHOT_RETENTION_S: int = 86400  # 24 hours

# ---------------------------------------------------------------------------
# SQLite schema DDL
# ---------------------------------------------------------------------------

_METRICS_DDL: str = """
CREATE TABLE IF NOT EXISTS metrics (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_key        TEXT    NOT NULL,
    value             REAL,
    timestamp_unix_s  REAL    NOT NULL,
    cycle             INTEGER,
    step              INTEGER,
    source            TEXT    NOT NULL DEFAULT 'poll',
    created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_metrics_key_ts
    ON metrics(metric_key, timestamp_unix_s);

CREATE INDEX IF NOT EXISTS idx_metrics_ts
    ON metrics(timestamp_unix_s);

CREATE INDEX IF NOT EXISTS idx_metrics_cycle_step
    ON metrics(cycle, step);
"""

_GPU_SNAPSHOTS_DDL: str = """
CREATE TABLE IF NOT EXISTS gpu_snapshots (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    gpu_index         INTEGER NOT NULL,
    timestamp_unix_s  REAL    NOT NULL,
    util_pct          INTEGER,
    mem_used_mib      INTEGER,
    mem_total_mib     INTEGER,
    temp_c            INTEGER,
    created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_gpu_ts
    ON gpu_snapshots(timestamp_unix_s);

CREATE INDEX IF NOT EXISTS idx_gpu_index_ts
    ON gpu_snapshots(gpu_index, timestamp_unix_s);
"""

_CHECKPOINT_HISTORY_DDL: str = """
CREATE TABLE IF NOT EXISTS checkpoint_history (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle               INTEGER NOT NULL,
    step                INTEGER NOT NULL,
    filename            TEXT    NOT NULL,
    size_bytes          INTEGER,
    base_checkpoint_id  TEXT,
    sha256              TEXT,
    saved_at            TEXT    NOT NULL,
    detected_at_unix_s  REAL    NOT NULL,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_cp_cycle_step
    ON checkpoint_history(cycle, step);
"""

_DASHBOARD_EVENTS_DDL: str = """
CREATE TABLE IF NOT EXISTS dashboard_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type      TEXT    NOT NULL,
    payload         TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_devents_type_ts
    ON dashboard_events(event_type, created_at);
"""

_PRAGMA_SQL: str = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;
PRAGMA cache_size = -8000;
PRAGMA temp_store = MEMORY;
PRAGMA mmap_size = 268435456;
PRAGMA auto_vacuum = INCREMENTAL;
"""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AgentEntry:
    """A single agent entry parsed from DIARIO_DE_BORDO.md."""
    timestamp: str                    # "07-24 19:15"
    timestamp_full: str               # "2026-07-24T19:15:00Z"
    agent: str                        # "Claude (F51 Darwin session)"
    operation: str                    # "READ" | "WRITE" | "EXECUTE" | "DECIDE" | "MIXED"
    operation_type: str               # lowercase, for CSS class
    entry_type: str                   # "ENTRADA" | "SAÍDA" | "ENTRADA + SAÍDA"
    summary: str                      # First meaningful line, max ~80 chars
    estado_encontrado: Optional[str] = None
    acoes: Optional[str] = None
    estado_deixado: Optional[str] = None
    commit: Optional[str] = None      # "hash — message" or descriptive text
    warnings: Optional[str] = None
    files_modified: list[str] = field(default_factory=list)


class RunStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    HUNG = "hung"


# ---------------------------------------------------------------------------
# SQLiteManager
# ---------------------------------------------------------------------------

class SQLiteManager:
    """Manages the dashboard SQLite database (WAL mode).

    Creates and maintains four tables:
      - metrics            (time-series training metrics, one row per metric per poll)
      - gpu_snapshots      (per-GPU utilisation / memory / temperature)
      - checkpoint_history (one row per checkpoint save detected)
      - dashboard_events   (internal operational log for the dashboard itself)
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            timeout=10.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_PRAGMA_SQL)
        self._create_tables()

    # -- table creation -------------------------------------------------------

    def _create_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(_METRICS_DDL)
        cur.executescript(_GPU_SNAPSHOTS_DDL)
        cur.executescript(_CHECKPOINT_HISTORY_DDL)
        cur.executescript(_DASHBOARD_EVENTS_DDL)
        self._conn.commit()
        cur.close()

    # -- insert helpers -------------------------------------------------------

    def insert_metrics(
        self,
        step: int,
        cycle: int,
        loss: Optional[float],
        jepa_raw: Optional[float],
        dope: Optional[float],
        delta_loss: Optional[float],
        gmc: Optional[float],
        jepa_bonus: Optional[float],
        tok_per_sec: Optional[float],
        gpu_data: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Insert one row per metric for a single poll tick.

        Values that are None / NaN / Inf are stored as SQL NULL.
        Perplexity is computed from *loss* (ppl = exp(loss)) when loss is valid.
        """
        now_ts = time.time()
        rows: list[tuple[str, Optional[float], float, Optional[int], Optional[int]]] = []

        def _safe(v: Optional[float]) -> Optional[float]:
            if v is None:
                return None
            if not math.isfinite(v):
                return None
            return round(v, 8)

        safe_loss = _safe(loss)
        safe_jepa = _safe(jepa_raw)
        safe_dope = _safe(dope)
        safe_delta = _safe(delta_loss)
        safe_gmc = _safe(gmc)
        safe_jepa_bonus = _safe(jepa_bonus)
        safe_tok_s = _safe(tok_per_sec)
        safe_ppl: Optional[float] = None
        if safe_loss is not None and safe_loss < 50.0:  # guard against insane values
            safe_ppl = round(math.exp(safe_loss), 4)

        rows.append(("loss", safe_loss, now_ts, cycle, step))
        rows.append(("ppl", safe_ppl, now_ts, cycle, step))
        rows.append(("jepa_raw", safe_jepa, now_ts, cycle, step))
        rows.append(("dope", safe_dope, now_ts, cycle, step))
        rows.append(("delta", safe_delta, now_ts, cycle, step))
        rows.append(("gmc", safe_gmc, now_ts, cycle, step))
        rows.append(("jepa_bonus", safe_jepa_bonus, now_ts, cycle, step))
        rows.append(("tok_s", safe_tok_s, now_ts, cycle, step))

        cur = self._conn.cursor()
        cur.executemany(
            "INSERT INTO metrics (metric_key, value, timestamp_unix_s, cycle, step, source) "
            "VALUES (?, ?, ?, ?, ?, 'poll')",
            rows,
        )
        self._conn.commit()
        cur.close()

        # Also insert GPU snapshots if provided
        if gpu_data:
            self._insert_gpu_snapshots_batch(step, now_ts, gpu_data)

    def insert_gpu_snapshot(
        self,
        step: int,
        gpu_index: int,
        util: Optional[int],
        mem_used: Optional[int],
        mem_total: Optional[int],
        temp: Optional[int],
    ) -> None:
        """Insert a single GPU snapshot row."""
        now_ts = time.time()
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO gpu_snapshots (gpu_index, timestamp_unix_s, util_pct, mem_used_mib, mem_total_mib, temp_c) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (gpu_index, now_ts, util, mem_used, mem_total, temp),
        )
        self._conn.commit()
        cur.close()

    def _insert_gpu_snapshots_batch(
        self, step: int, now_ts: float, gpu_data: list[dict[str, Any]]
    ) -> None:
        """Batch-insert GPU snapshots from parsed nvidia-smi data."""
        rows: list[tuple[int, float, Optional[int], Optional[int], Optional[int], Optional[int]]] = []
        for g in gpu_data:
            idx = int(g.get("idx", 0))
            util = self._safe_int(g.get("util"))
            mem_used = self._safe_int(g.get("mem_used"))
            mem_total = self._safe_int(g.get("mem_total"))
            temp = self._safe_int(g.get("temp"))
            rows.append((idx, now_ts, util, mem_used, mem_total, temp))
        if rows:
            cur = self._conn.cursor()
            cur.executemany(
                "INSERT INTO gpu_snapshots (gpu_index, timestamp_unix_s, util_pct, mem_used_mib, mem_total_mib, temp_c) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
            cur.close()

    def insert_checkpoint_event(
        self,
        cycle: int,
        step: int,
        filename: str,
        size_bytes: int,
        base_checkpoint_id: str,
        saved_at: str,
    ) -> None:
        """Record a checkpoint save event (idempotent — UNIQUE on cycle+step)."""
        now_ts = time.time()
        cur = self._conn.cursor()
        try:
            cur.execute(
                "INSERT OR IGNORE INTO checkpoint_history "
                "(cycle, step, filename, size_bytes, base_checkpoint_id, saved_at, detected_at_unix_s) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (cycle, step, filename, size_bytes, base_checkpoint_id, saved_at, now_ts),
            )
            self._conn.commit()
        except sqlite3.Error:
            pass  # unique constraint — already recorded
        finally:
            cur.close()

    def insert_dashboard_event(self, event_type: str, payload: Optional[str] = None) -> None:
        """Log a dashboard-internal event (startup, shutdown, poll_error, …)."""
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO dashboard_events (event_type, payload) VALUES (?, ?)",
            (event_type, payload),
        )
        self._conn.commit()
        cur.close()

    # -- query helpers --------------------------------------------------------

    def get_metrics_range(self, from_step: int, to_step: int) -> list[dict[str, Any]]:
        """Return all metrics rows between two global step numbers (inclusive)."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT metric_key, value, timestamp_unix_s, cycle, step, source "
            "FROM metrics WHERE step >= ? AND step <= ? "
            "ORDER BY step ASC, metric_key ASC",
            (from_step, to_step),
        )
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        return rows

    def get_recent_metrics(self, n: int = 100) -> list[dict[str, Any]]:
        """Return the last *n* metric rows (newest first)."""
        # Get the most recent N distinct timestamps first, then all metrics for those
        cur = self._conn.cursor()
        cur.execute(
            "SELECT metric_key, value, timestamp_unix_s, cycle, step, source "
            "FROM metrics "
            "WHERE timestamp_unix_s IN ("
            "  SELECT DISTINCT timestamp_unix_s FROM metrics "
            "  ORDER BY timestamp_unix_s DESC LIMIT ?"
            ") "
            "ORDER BY timestamp_unix_s ASC, metric_key ASC",
            (n,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        return rows

    def get_recent_metric_points(
        self, metric_key: str, n: int = 1000
    ) -> list[dict[str, Any]]:
        """Return the last *n* {ts, value} points for a single metric key.

        Used to seed chart ring buffers on startup.
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT timestamp_unix_s AS ts, value "
            "FROM metrics "
            "WHERE metric_key = ? AND value IS NOT NULL "
            "ORDER BY timestamp_unix_s DESC LIMIT ?",
            (metric_key, n),
        )
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        # Reverse so they are in chronological order
        rows.reverse()
        return rows

    def get_run_history(self) -> list[dict[str, Any]]:
        """Read runs.jsonl from workspace/runtime/dashboard/ and return parsed entries."""
        runs_path = self._db_path.parent / "runs.jsonl"
        if not runs_path.exists():
            return []
        runs: list[dict[str, Any]] = []
        try:
            for line in runs_path.read_text(encoding="utf-8").strip().split("\n"):
                if line.strip():
                    try:
                        runs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return runs

    def get_gpu_snapshots_recent(self, n: int = 7200) -> list[dict[str, Any]]:
        """Return the most recent GPU snapshot rows."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT gpu_index, timestamp_unix_s, util_pct, mem_used_mib, mem_total_mib, temp_c "
            "FROM gpu_snapshots ORDER BY timestamp_unix_s DESC LIMIT ?",
            (n,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        return rows

    def get_checkpoint_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent checkpoint history entries (newest first)."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT cycle, step, filename, size_bytes, base_checkpoint_id, sha256, saved_at, detected_at_unix_s "
            "FROM checkpoint_history ORDER BY cycle DESC, step DESC LIMIT ?",
            (limit,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        return rows

    # -- maintenance ----------------------------------------------------------

    def vacuum_old_metrics(self, retention_days: int = 30) -> int:
        """Delete metrics older than *retention_days*.  Returns count of rows deleted.

        GPU snapshots are also pruned to the last 24 h.
        """
        cutoff_ts = time.time() - (retention_days * 86400.0)
        cur = self._conn.cursor()

        cur.execute("SELECT COUNT(*) FROM metrics WHERE timestamp_unix_s < ?", (cutoff_ts,))
        deleted_metrics: int = cur.fetchone()[0]

        cur.execute("DELETE FROM metrics WHERE timestamp_unix_s < ?", (cutoff_ts,))

        # Prune GPU snapshots to last 24 h
        cur.execute(
            "DELETE FROM gpu_snapshots WHERE timestamp_unix_s < ("
            "  SELECT MAX(timestamp_unix_s) - ? FROM gpu_snapshots"
            ")",
            (GPU_SNAPSHOT_RETENTION_S,),
        )

        self._conn.commit()
        cur.close()

        # Reclaim disk space
        cur = self._conn.cursor()
        cur.execute("PRAGMA incremental_vacuum")
        cur.close()

        return deleted_metrics

    def checkpoint_wal(self) -> None:
        """Flush WAL journal to the main database file."""
        cur = self._conn.cursor()
        cur.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        cur.close()

    def optimize(self) -> None:
        """Update query planner statistics."""
        cur = self._conn.cursor()
        cur.execute("PRAGMA optimize")
        cur.close()

    def metrics_stats(self) -> dict[str, Any]:
        """Return basic statistics about the database for monitoring."""
        cur = self._conn.cursor()
        stats: dict[str, Any] = {}
        cur.execute("SELECT COUNT(*) FROM metrics")
        stats["metrics_row_count"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM gpu_snapshots")
        stats["gpu_snapshot_count"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM checkpoint_history")
        stats["checkpoint_history_count"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM dashboard_events")
        stats["dashboard_event_count"] = cur.fetchone()[0]
        stats["db_size_mb"] = round(self._db_path.stat().st_size / (1024 * 1024), 2) if self._db_path.exists() else 0.0
        cur.close()
        return stats

    def close(self) -> None:
        """Gracefully close the database: checkpoint WAL, optimize, close connection."""
        try:
            self.checkpoint_wal()
        except sqlite3.Error:
            pass
        try:
            self.optimize()
        except sqlite3.Error:
            pass
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    # -- static helpers -------------------------------------------------------

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None


# ---------------------------------------------------------------------------
# LogbookParser
# ---------------------------------------------------------------------------

class LogbookParser:
    """Regex-based parser for DIARIO_DE_BORDO.md.

    Parses three sections:
      - ESTADO ATUAL table    -> dict
      - Agent entry blocks    -> list[AgentEntry]
      - Bugs conhecidos       -> list[str]

    Caches the parsed result.  ``reload()`` re-parses only if the source file's
    mtime has changed since the last parse.
    """

    # -- compiled regexes -----------------------------------------------------

    _TABLE_ROW_RE = re.compile(r'\|\s*\*\*(.+?)\*\*\s*\|\s*(.+?)\s*\|')
    _ENTRY_HEADER_RE = re.compile(
        r'###\s+(\d{4}-\d{2}-\d{2})\s+~?(\d{2}:\d{2})\s+UTC\s*\|\s*(.+?)\s*(?:[-—–]+\s*(?:ENTRADA|SA[IÍ]DA)(?:\s*\+\s*SA[IÍ]DA)?)?\s*$',
        re.MULTILINE,
    )
    _OP_RE = re.compile(
        r'OPERA[CÇ][AÃ]O\s*(?:PRETENDIDA|REALIZADA)?\s*:\s*(READ|WRITE|EXECUTE|DECIDE|MIXED|PARADA[^;]*)',
        re.IGNORECASE,
    )
    _COMMIT_RE = re.compile(
        r'COMMIT:\s*'
        r'(?:\(?nenhum[^)]*\)?|\(?pending\)?)'
        r'|COMMIT:\s*([a-f0-9]{6,})\s*[-—–]\s*(.+)',
        re.IGNORECASE,
    )
    _COMMIT_HASH_ONLY_RE = re.compile(r'COMMIT:\s*([a-f0-9]{6,})\s*$', re.IGNORECASE)

    def __init__(self, logbook_path: Path) -> None:
        self._path = logbook_path
        self._last_mtime: float = 0.0
        self._estado_atual: dict[str, str] = {}
        self._entries: list[AgentEntry] = []
        self._bugs: list[str] = []

        if self._path.exists():
            self._parse()

    # -- public API -----------------------------------------------------------

    def parse_estado_atual(self) -> dict[str, str]:
        """Return the ESTADO ATUAL table as a flat key-value dict."""
        return dict(self._estado_atual)

    def parse_entries(self) -> list[AgentEntry]:
        """Return all parsed agent entries."""
        return list(self._entries)

    def parse_bugs(self) -> list[str]:
        """Return known bugs as a list of title strings."""
        return list(self._bugs)

    def agent_timeline(self, n: int = 20) -> list[dict[str, Any]]:
        """Return the last *n* entries formatted for UI display.

        Each dict contains: timestamp, timestamp_full, agent, operation,
        operation_type, entry_type, summary, commit, estado_encontrado,
        acoes, estado_deixado, warnings, files_modified.
        """
        entries = self._entries[-n:] if len(self._entries) > n else self._entries
        result: list[dict[str, Any]] = []
        for e in reversed(entries):  # newest first
            result.append({
                "timestamp": e.timestamp,
                "timestamp_full": e.timestamp_full,
                "agent": e.agent,
                "operation": e.operation,
                "operation_type": e.operation_type,
                "entry_type": e.entry_type,
                "summary": e.summary,
                "commit": e.commit,
                "estado_encontrado": e.estado_encontrado,
                "acoes": e.acoes,
                "estado_deixado": e.estado_deixado,
                "warnings": e.warnings,
                "files_modified": e.files_modified,
            })
        return result

    def reload(self) -> bool:
        """Re-parse the logbook if its mtime has changed.  Returns True if re-parsed."""
        if not self._path.exists():
            return False
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            return False
        if mtime <= self._last_mtime:
            return False
        self._parse()
        return True

    # -- internal parse -------------------------------------------------------

    def _parse(self) -> None:
        """Parse the full file and populate internal caches."""
        try:
            text = self._path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return

        self._last_mtime = self._path.stat().st_mtime if self._path.exists() else 0.0
        self._estado_atual = self._parse_estado_atual_block(text)
        self._entries = self._parse_agent_entries(text)
        self._bugs = self._parse_bugs_block(text)

    def _parse_estado_atual_block(self, text: str) -> dict[str, str]:
        """Parse the ESTADO ATUAL markdown table into a flat dict."""
        result: dict[str, str] = {}

        section_start = text.find("## 📍 ESTADO ATUAL")
        if section_start == -1:
            section_start = text.find("## ESTADO ATUAL")
        if section_start == -1:
            return result

        table_start = text.find("| Campo", section_start)
        if table_start == -1:
            table_start = text.find("| **", section_start)
        if table_start == -1:
            return result

        table_end = text.find("\n\n", table_start)
        if table_end == -1:
            table_end = len(text)

        table_text = text[table_start:table_end]

        for match in self._TABLE_ROW_RE.finditer(table_text):
            key = match.group(1).strip()
            value = match.group(2).strip()
            result[key] = value

        return result

    def _parse_agent_entries(self, text: str) -> list[AgentEntry]:
        """Parse all agent entry/exit blocks."""
        entries: list[AgentEntry] = []

        for header_match in self._ENTRY_HEADER_RE.finditer(text):
            date_str = header_match.group(1)
            time_str = header_match.group(2)
            agent_name = header_match.group(3).strip()

            # Find the subsequent code block(s)
            pos = header_match.end()
            # Collect all code blocks until the next header or end of text
            block_parts: list[str] = []
            while pos < len(text):
                fence_start = text.find("```", pos)
                if fence_start == -1:
                    break
                # If there is another header between pos and fence_start, stop
                between = text[pos:fence_start]
                if re.search(r'^###\s+\d{4}-\d{2}-\d{2}', between, re.MULTILINE):
                    break
                fence_end = text.find("```", fence_start + 3)
                if fence_end == -1:
                    break
                block_parts.append(text[fence_start + 3:fence_end].strip())
                pos = fence_end + 3

            if not block_parts:
                continue

            # Merge all code blocks for this entry
            block_content = "\n".join(block_parts)

            entry = self._parse_entry_block(block_content, date_str, time_str, agent_name)
            if entry:
                entries.append(entry)

        return entries

    def _parse_entry_block(
        self, block: str, date_str: str, time_str: str, agent_name: str
    ) -> Optional[AgentEntry]:
        """Parse a single agent-entry code block."""
        entry = AgentEntry(
            timestamp=f"{date_str[5:]} {time_str}",  # "MM-DD HH:MM"
            timestamp_full=f"{date_str}T{time_str}:00Z",
            agent=agent_name,
            operation="READ",
            operation_type="read",
            entry_type="ENTRADA",
            summary="",
        )

        # -- entry type (check first line only) ----------------------------
        first_line = block.split("\n")[0].strip().upper().replace("Í", "I").replace("Ç", "C").replace("Ã", "A")
        if "ENTRADA + SAIDA" in first_line or "ENTRADA" in first_line and "SAIDA" in first_line:
            entry.entry_type = "ENTRADA + SAÍDA"
        elif first_line.startswith("ENTRADA"):
            entry.entry_type = "ENTRADA"
        elif first_line.startswith("SAIDA"):
            entry.entry_type = "SAÍDA"

        # -- operation --------------------------------------------------------
        op_match = self._OP_RE.search(block)
        if op_match:
            raw_op = op_match.group(1).upper()
            # Map variants
            if "PARADA" in raw_op:
                entry.operation = "MIXED"
            elif raw_op in ("READ", "WRITE", "EXECUTE", "DECIDE", "MIXED"):
                entry.operation = raw_op
            else:
                for known in ("READ", "WRITE", "EXECUTE", "DECIDE"):
                    if known in raw_op:
                        entry.operation = known
                        break
                else:
                    entry.operation = "MIXED"
        else:
            # No explicit operation code — infer from context
            entry.operation = self._infer_operation(block, entry.entry_type)
        entry.operation_type = entry.operation.lower()

        # -- summary ----------------------------------------------------------
        entry.summary = self._extract_summary(block)

        # -- commit -----------------------------------------------------------
        entry.commit = self._extract_commit(block)

        # -- section blocks ---------------------------------------------------
        entry.estado_encontrado = self._extract_section(block, "ESTADO ENCONTRADO")
        entry.estado_deixado = self._extract_section(block, "ESTADO DEIXADO")
        entry.acoes = self._extract_section(block, "AÇÕES")
        entry.warnings = self._extract_section(block, "WARNINGS")

        # -- files modified ---------------------------------------------------
        entry.files_modified = self._extract_file_list(block)

        return entry

    @staticmethod
    def _infer_operation(block: str, entry_type: str) -> str:
        """Infer the operation when no explicit code is present in the entry.

        Checks the operation description line for action keywords (accent-
        insensitive, word-boundary match). If keywords from multiple categories
        match, returns MIXED. Falls back to entry-type defaults:
        ENTRADA → READ, SAÍDA → MIXED.
        """
        _accent_map = str.maketrans("ÇÃÕÍÊÁÉÓÚÂçãõíêáéóúâ",
                                     "CAOIEAEOUAcaoieaeoua")

        op_desc_match = re.search(
            r'OPERA[CÇ][AÃ]O\s*(?:PRETENDIDA|REALIZADA)?\s*:\s*(.+)',
            block, re.IGNORECASE,
        )
        if op_desc_match:
            desc = op_desc_match.group(1).translate(_accent_map).lower()

            # Use word-boundary matching to avoid substring false positives
            def _has_word(text: str, keyword: str) -> bool:
                return bool(re.search(r'\b' + re.escape(keyword) + r'\b', text))

            def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
                return any(_has_word(text, kw) for kw in keywords)

            decide_kw = ("decidir", "decide", "planej", "design", "projet",
                         "especific", "arquitet", "dissection", "dissec",
                         "analisar", "analise", "vulnerability", "investig")
            execute_kw = ("test", "executar", "lancar", "launch", "run",
                          "rodar", "canario", "canary", "compilar", "build",
                          "relancamento", "testes")
            write_kw = ("corrigir", "correcao", "correcoes", "fix", "criar",
                        "criacao", "template", "implement", "escrever", "escrita",
                        "ajust", "tuning", "bootstrap", "adicion", "adicionar",
                        "nova", "novo", "applied", "aplicad",
                        "modific", "patch", "escrev")
            read_kw = ("ler", "read-only", "verificacao", "verificacoes",
                       "diagnostic", "diagnostico", "verificar", "auditar",
                       "auditoria", "inspect", "inspecionar", "trace",
                       "checkpoint integrity", "imports")

            matched: set[str] = set()
            if _has_any(desc, decide_kw):
                matched.add("DECIDE")
            if _has_any(desc, execute_kw):
                matched.add("EXECUTE")
            if _has_any(desc, write_kw):
                matched.add("WRITE")
            if _has_any(desc, read_kw):
                matched.add("READ")

            if len(matched) == 1:
                return matched.pop()
            if len(matched) > 1:
                if "EXECUTE" in matched or "WRITE" in matched:
                    return "MIXED"
                return matched.pop()

        # Fallback based on entry type
        if entry_type in ("SAÍDA", "ENTRADA + SAÍDA"):
            return "MIXED"
        return "READ"

    def _extract_summary(self, block: str) -> str:
        """Extract a short summary from the entry block."""
        lines = block.strip().split("\n")

        # Try the first line (often the entry type line)
        first = lines[0].strip() if lines else ""

        # Look for OBJETIVO / OPERAÇÃO lines
        obj_match = re.search(r'OBJETIVO:\s*(.+)', block)
        if obj_match:
            return obj_match.group(1).strip()[:80]

        op_full = re.search(
            r'OPERAÇÃO\s*(?:PRETENDIDA|REALIZADA)?\s*:\s*(.+)', block, re.IGNORECASE
        )
        if op_full:
            return op_full.group(1).strip()[:80]

        # Fallback: first non-empty, non-meta line after header lines
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if any(
                stripped.upper().startswith(p)
                for p in (
                    "ENTRADA", "SAÍDA", "OPER", "ESTADO", "AÇÕES",
                    "OBJETIVO", "COMMIT", "WARNINGS", "ARQUIVOS",
                    "TESTES",
                )
            ):
                continue
            if len(stripped) > 5:
                return stripped[:80]

        # Ultimate fallback: first substantive line
        for line in lines:
            stripped = line.strip()
            if len(stripped) > 10:
                return stripped[:80]

        return first[:80] if first else ""

    def _extract_commit(self, block: str) -> Optional[str]:
        """Extract commit hash and message.

        Only matches lines starting with COMMIT: (case-insensitive), not
        embedded references like "Git commit: hash".
        """
        # Try "hash — message" or "hash -- message" format (line-start anchored)
        match = re.search(r'^COMMIT:\s*([a-f0-9]{6,})\s*[-—–]+\s*(.+)', block,
                          re.IGNORECASE | re.MULTILINE)
        if match:
            return f"{match.group(1)} — {match.group(2).strip()}"

        # Try hash-only (line-start anchored)
        match = re.search(r'^COMMIT:\s*([a-f0-9]{6,})\s*$', block,
                          re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1)

        # Non-hash commit descriptions: "COMMIT: (pending)" or "COMMIT: (nenhum ...)"
        match = re.search(r'^COMMIT:\s*\(?(nenhum[^)]*|pending)\)?', block,
                          re.IGNORECASE | re.MULTILINE)
        if match:
            inner = match.group(1).strip()
            if inner == "pending":
                return "(pending)"
            return f"({inner})"

        return None

    def _parse_bugs_block(self, text: str) -> list[str]:
        """Parse the Bugs conhecidos numbered list into title strings."""
        bugs: list[str] = []

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

        for match in re.finditer(r'\d+\.\s+\*\*(.+?)\*\*(?:\s*:\s*(.+))?', list_text):
            title = match.group(1).strip()
            bugs.append(title)

        return bugs

    # -- static section helpers -----------------------------------------------

    @staticmethod
    def _extract_section(block: str, section_name: str) -> Optional[str]:
        """Extract a named section body from the block.

        Handles both "NAME:" (with colon on same line) and "NAME\\n" (header on its own line).
        Accent-insensitive: ``ACOES`` matches both ``ACOES`` and ``AÇÕES``.
        """
        # Build an accent-insensitive pattern for the section name
        escaped = re.escape(section_name)
        # Replace Portuguese accented chars with character classes
        accent_map = {
            re.escape("Ç"): "[CÇ]",
            re.escape("ç"): "[cç]",
            re.escape("Ã"): "[AÃ]",
            re.escape("ã"): "[aã]",
            re.escape("Õ"): "[OÕ]",
            re.escape("õ"): "[oõ]",
            re.escape("Í"): "[IÍ]",
            re.escape("í"): "[ií]",
            re.escape("Ê"): "[EÊ]",
            re.escape("ê"): "[eê]",
            re.escape("Á"): "[AÁ]",
            re.escape("á"): "[aá]",
            re.escape("É"): "[EÉ]",
            re.escape("é"): "[eé]",
            re.escape("Ó"): "[OÓ]",
            re.escape("ó"): "[oó]",
            re.escape("Ú"): "[UÚ]",
            re.escape("ú"): "[uú]",
            re.escape("Â"): "[AÂ]",
            re.escape("â"): "[aâ]",
        }
        for accented, char_class in accent_map.items():
            escaped = escaped.replace(accented, char_class)

        pattern = rf'{escaped}\s*:?\s*\n(.*?)(?=\n[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ]{{2,}}|\Z)'
        match = re.search(pattern, block, re.DOTALL)
        if match:
            content = match.group(1).strip()
            return content if content else None
        return None

    @staticmethod
    def _extract_file_list(block: str) -> list[str]:
        """Extract modified file paths from ARQUIVOS MODIFICADOS section."""
        section = LogbookParser._extract_section(block, "ARQUIVOS MODIFICADOS")
        if not section:
            return []
        files: list[str] = []
        for line in section.split("\n"):
            stripped = line.strip().lstrip("- ").strip()
            if stripped and not stripped.startswith("(") and not stripped.startswith("DIARIO_DE_BORDO"):
                files.append(stripped)
        return files


# ---------------------------------------------------------------------------
# MetricsRingBuffer
# ---------------------------------------------------------------------------

class MetricsRingBuffer:
    """In-memory ring buffers (maxlen=7200) for live chart data.

    Maintains parallel deques for: step, loss, ppl, tok_s, jepa, dope, delta, gmc.
    Each push stores one tick across all deques.  NaN/Inf/None values are dropped
    (the corresponding deque is not extended, preserving alignment).
    """

    _METRIC_KEYS = ("step", "loss", "ppl", "tok_s", "jepa", "dope", "delta", "gmc")

    def __init__(self, maxlen: int = RING_MAXLEN) -> None:
        self._maxlen = maxlen
        self._deques: dict[str, deque[float]] = {
            key: deque(maxlen=maxlen) for key in self._METRIC_KEYS
        }
        self._timestamps: deque[float] = deque(maxlen=maxlen)

    def push(self, metrics: dict[str, Any]) -> None:
        """Push one poll tick into all deques.

        *metrics* is a flat dict that may contain any of the known keys plus
        extra keys (which are silently ignored).  Values that are None, NaN, or
        Inf are skipped for the affected key.
        """
        now = time.time()
        self._timestamps.append(now)

        for key in self._METRIC_KEYS:
            if key not in metrics:
                continue
            val = metrics[key]
            if val is None:
                continue
            try:
                fval = float(val)
            except (ValueError, TypeError):
                continue
            if not math.isfinite(fval):
                continue
            self._deques[key].append(fval)

    def get_last_n(self, n: int) -> dict[str, list[float]]:
        """Return the last *n* values from each deque.

        Returns a dict keyed by metric name, each value is a list of floats
        (newest last).  This shape is directly consumable by Chart.js.
        """
        return {key: list(dq)[-n:] if len(dq) > n else list(dq) for key, dq in self._deques.items()}

    def get_current(self) -> dict[str, Optional[float]]:
        """Return the most recent value for each metric, or None if the deque is empty."""
        return {
            key: (dq[-1] if dq else None)
            for key, dq in self._deques.items()
        }

    def get_timestamps_last_n(self, n: int) -> list[float]:
        """Return the last *n* timestamps for X-axis alignment."""
        ts = list(self._timestamps)
        return ts[-n:] if len(ts) > n else ts

    def clear(self) -> None:
        """Reset all deques."""
        self._timestamps.clear()
        for dq in self._deques.values():
            dq.clear()

    @property
    def count(self) -> int:
        """Total number of ticks stored (from the step deque)."""
        return len(self._deques["step"])

    @property
    def span_s(self) -> float:
        """Time span from first to last tick in seconds."""
        t = list(self._timestamps)
        if len(t) < 2:
            return 0.0
        return t[-1] - t[0]


# ---------------------------------------------------------------------------
# RunDetector
# ---------------------------------------------------------------------------

class RunDetector:
    """Detect the training process via psutil.

    Looks for python/python.exe processes whose command line contains
    "darwin_organism" (the training entry point module).
    """

    _TRAINING_MARKER = "darwin_organism"

    def __init__(self) -> None:
        self._psutil_available: bool = False
        try:
            import psutil  # noqa: F401

            self._psutil_available = True
        except ImportError:
            pass

    def detect(self) -> RunStatus:
        """Return the training process status.

        Returns:
          RUNNING — at least one training process exists and is not hung
          STOPPED — no training process found
          HUNG    — a training process exists but psutil is unavailable
                    (caller should refine with dopamine timestamp)
        """
        if not self._psutil_available:
            return RunStatus.STOPPED  # Cannot detect — assume stopped

        for proc in self._find_training_processes():
            pid = proc["pid"]
            # Check if the process is alive
            try:
                import psutil

                p = psutil.Process(pid)
                if p.is_running():
                    return RunStatus.RUNNING
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return RunStatus.STOPPED

    def get_process_info(self) -> dict[str, Any]:
        """Return information about all matching training processes.

        Returns a dict with:
          processes: list of {pid, cpu_time_s, mem_gb, uptime_s, cmdline}
          count:     total matching processes
          total_mem_gb: aggregate RSS in GB
        """
        if not self._psutil_available:
            return {"processes": [], "count": 0, "total_mem_gb": 0.0, "psutil_available": False}

        import psutil

        processes: list[dict[str, Any]] = []
        total_mem = 0.0

        for proc_info in self._find_training_processes():
            pid = proc_info["pid"]
            try:
                p = psutil.Process(pid)
                with p.oneshot():
                    rss_gb = p.memory_info().rss / (1024**3)
                    cpu_time = p.cpu_time()
                    create_time = p.create_time()
                    uptime_s = time.time() - create_time if create_time else 0.0

                    processes.append({
                        "pid": pid,
                        "cpu_time_s": round(cpu_time.user + cpu_time.system, 1) if cpu_time else 0.0,
                        "mem_gb": round(rss_gb, 2),
                        "uptime_s": round(uptime_s, 1),
                        "cmdline": " ".join(p.cmdline()[:3]),
                    })
                    total_mem += rss_gb
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return {
            "processes": processes,
            "count": len(processes),
            "total_mem_gb": round(total_mem, 2),
            "psutil_available": True,
        }

    def is_running(self) -> bool:
        """Convenience: True if at least one training process is alive."""
        return self.detect() == RunStatus.RUNNING

    def _find_training_processes(self) -> list[dict[str, Any]]:
        """Iterate over psutil process list and return matching entries."""
        import psutil

        matches: list[dict[str, Any]] = []
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    name = (proc.info.get("name") or "").lower()
                    cmdline = proc.info.get("cmdline") or []
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

                # Must be a Python process
                if "python" not in name:
                    continue

                # Command line must reference darwin_organism
                cmdline_str = " ".join(cmdline).lower() if cmdline else ""
                if self._TRAINING_MARKER in cmdline_str:
                    matches.append({
                        "pid": proc.info["pid"],
                        "name": proc.info["name"],
                        "cmdline": cmdline,
                    })
        except Exception:
            pass

        return matches


# ---------------------------------------------------------------------------
# Utility: append dashboard events to DIARIO_DE_BORDO.md
# ---------------------------------------------------------------------------

def append_logbook_event(
    logbook_path: Path,
    event_type: str,
    payload: str,
    timeout_s: float = 2.0,
) -> bool:
    """Append a dashboard auto-event to DIARIO_DE_BORDO.md.

    Uses ``msvcrt.locking`` (Windows) for advisory file locking.  Returns True
    on success, False on lock timeout or write failure.

    *event_type* is a short label like ``"training_start"``, ``"training_stop"``,
    or ``"training_snapshot"``.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = f"""

### {timestamp} | Dashboard (auto)

```
{event_type.upper()}
{payload}
```
"""

    try:
        with open(logbook_path, "a", encoding="utf-8") as f:
            if os.name == "nt":
                import msvcrt

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
                    try:
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass  # best-effort unlock
            else:
                # Non-Windows: best-effort append (no fcntl.lockf by default)
                f.write(entry)
                f.flush()
                os.fsync(f.fileno())
        return True
    except Exception:
        return False
