# F51 Darwin-X Mission Control — Dashboard Architecture Plan

**Date:** 2026-07-24
**Author:** Claude (F51 Darwin session)
**Status:** Implementation plan — ready for coding

---

## 1. File Tree

Every file to be created or modified, with purpose and line-count estimate.

### 1.1 New files (to be created)

| # | File | Purpose | Est. Lines |
|---|---|---|---|
| 1 | `src/tools/darwin_dashboard_web.py` | FastAPI app, SSE collector loop, logbook parser, SQLite writer, path resolver, all backend logic | ~400 |
| 2 | `src/tools/static/htmx.min.js` | htmx 2.x runtime (14KB, committed — downloaded once) | 1 (minified) |
| 3 | `src/tools/static/chart.min.js` | Chart.js 4.x runtime (65KB, committed — downloaded once) | 1 (minified) |
| 4 | `workspace/runtime/dashboard/metrics.db` | SQLite WAL database (3 tables: metrics, gpu_snapshots, checkpoint_history). Created on first startup. | N/A (binary) |
| 5 | `workspace/runtime/dashboard/runs.jsonl` | Append-only log of training run start/stop events. Written by dashboard on process detection changes. | N/A (runtime) |
| 6 | `workspace/runtime/dashboard/agent_timeline.json` | Cached parsed logbook entries. Regenerated when DIARIO_DE_BORDO.md mtime changes. | N/A (runtime) |
| 7 | `governance/docs/dashboard_architecture_plan.md` | This document. | ~400 |

### 1.2 Existing files (already created, referenced)

| # | File | Purpose | Status |
|---|---|---|---|
| 1 | `src/tools/templates/dashboard.html` | Jinja2 template: 8-section CSS Grid layout, htmx attributes, Chart.js canvases, SSE event handlers | Exists (1545 lines) |
| 2 | `src/tools/static/dashboard.css` | Dark OLED CSS custom properties, reset, layout, component styles | Exists |
| 3 | `src/tools/darwin_dashboard.py` | Terminal-only dashboard v1. Kept as fallback; not modified. | Exists (299 lines) |
| 4 | `DIARIO_DE_BORDO.md` | Agent logbook — parsed by dashboard for agent timeline section | Exists |
| 5 | `src/configs/darwin_x_100m.yaml` | 100M lineage config — `checkpoint_root` field used for path discovery | Exists |

### 1.3 Files NOT created

| Reason | Detail |
|---|---|
| src/tools/static/dashboard.js (intentionally absent) | JavaScript logic inlines in the template `<script>` block (SSE handlers, Chart.js init, htmx swaps). Keeps all front-end logic in one file. |
| src/scripts/dashboard_*.py (intentionally absent) | Single-file backend constraint: `darwin_dashboard_web.py` contains all Python logic. No separate `collectors.py`, `parser.py`, `db.py`. |
| `package.json` / `package-lock.json` | Zero npm. Zero node. Zero build step. |
| `requirements-dashboard.txt` | Only 2 deps (`fastapi`, `uvicorn`). Documented inline in `darwin_dashboard_web.py` docstring. |

---

## 2. Module Dependencies

### 2.1 Import Graph

```
┌─────────────────────────────────────────────────────────┐
│            darwin_dashboard_web.py  (~400 lines)         │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌─────────┐  ┌─────────┐  │
│  │ Path     │  │ Data     │  │ SSE     │  │ Logbook │  │
│  │ Resolver │  │ Poller   │  │ Broad-  │  │ Parser  │  │
│  │          │  │ (async)  │  │ caster  │  │         │  │
│  └──────────┘  └──────────┘  └─────────┘  └─────────┘  │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐   │
│  │ SQLite   │  │ GPU      │  │ FastAPI App + Routes │   │
│  │ Writer   │  │ Subproc  │  │ (Jinja2 + SSE)       │   │
│  └──────────┘  └──────────┘  └──────────────────────┘   │
│                                                         │
│  External imports:                                       │
│    fastapi, uvicorn, jinja2 (via fastapi)               │
│    asyncio, sqlite3, json, pathlib, hashlib, os,        │
│    subprocess, time, datetime, re, threading, signal     │
│    msvcrt (on Windows) / fcntl (on Unix)                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│            dashboard.html  (~1545 lines, exists)         │
│                                                         │
│  Dependencies (loaded from /static/):                    │
│    htmx.min.js  ──  AJAX, SSE extensions                │
│    chart.min.js ──  4 Chart.js instances                 │
│                                                         │
│  Dependencies (inlined <style> or /static/):             │
│    dashboard.css ── shared CSS custom properties         │
│                                                         │
│  Data flow:                                              │
│    SSE stream (/api/sse/stream) → JSON events            │
│    htmx polls (/api/sections/*) → HTML fragments         │
│    Chart.js updates via SSE 'chart_point' events         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 No Circular Dependencies

The architecture is a strict DAG:

```
Data files on disk
    ↓  (read by async poller every 2s)
darwin_dashboard_web.py (in-process data collection)
    ↓  (SSE push + htmx endpoints)
dashboard.html (browser rendering)
```

### 2.3 What darwin_dashboard_web.py does NOT import

- **NOTHING from `src/f51_darwin/`** — zero dependency on the model code, organism runtime, or training modules. The dashboard is a read-only observer that reads text files on disk. It never imports PyTorch, never loads checkpoints, never touches the model.
- **NOT `torch`** — organ identities come from `organism_latest.json` metadata, not from checkpoint file loading.
- **NOT `psutil`** — training process detection uses `subprocess` (PowerShell `Get-Process`) or tasklist.

---

## 3. Configuration — Path Discovery

### 3.1 The DashboardConfig dataclass

```python
@dataclass
class DashboardConfig:
    # ── Root paths ──
    project_root: Path          # auto: __file__ → parents[1]
    checkpoint_root: Path       # from config YAML or auto-detect
    blockchain_dir: Path        # from config YAML or default

    # ── File paths ──
    dopamine_file: Path         # {checkpoint_root}/dopamine_status.txt
    pointer_file: Path          # {checkpoint_root}/organism_latest.json
    ledger_file: Path           # {checkpoint_root}/causal_events.jsonl
    blockchain_file: Path       # {blockchain_dir}/blocks.jsonl
    logbook_file: Path          # {project_root}/DIARIO_DE_BORDO.md
    config_file: Path           # {project_root}/src/configs/darwin_x_100m.yaml

    # ── Dashboard runtime ──
    dashboard_dir: Path         # {project_root}/workspace/runtime/dashboard/
    sqlite_path: Path           # {dashboard_dir}/metrics.db
    runs_file: Path             # {dashboard_dir}/runs.jsonl
    timeline_cache: Path        # {dashboard_dir}/agent_timeline.json

    # ── Server ──
    host: str = "127.0.0.1"
    port: int = 8000
    poll_interval_s: float = 2.0

    # ── Lineage ──
    lineage: str = "100m"       # from --lineage flag or auto-detect
```

### 3.2 Resolution Algorithm (startup)

```
resolve_paths(lineage_override=None) → DashboardConfig:
    1. project_root = Path(__file__).resolve().parents[1]
    2. If --lineage flag: use that lineage name
       Else: scan checkpoint_root dirs, pick the one with organism_latest.json
    3. Load YAML config → extract checkpoint_root, blockchain_path
    4. Build all derived paths
    5. Validate: at minimum DIARIO_DE_BORDO.md and templates/ exist
       (data files may be missing — training not running — OK)
    6. Return DashboardConfig
```

### 3.3 Environment Variables (override mechanism)

| Variable | Overrides |
|---|---|
| `F51_DASHBOARD_HOST` | `--host` default |
| `F51_DASHBOARD_PORT` | `--port` default |
| `F51_LINEAGE` | lineage selector (100m, 600m, 1.6b) |
| `F51_DASHBOARD_NO_GPU` | skip nvidia-smi (set to 1) |

### 3.4 CLI

```powershell
# Start dashboard (auto-detect lineage)
.venv_nitro\Scripts\python.exe src\\tools\\darwin_dashboard_web.py

# Start dashboard (explicit lineage)
.venv_nitro\Scripts\python.exe src\\tools\\darwin_dashboard_web.py --lineage 100m

# Custom port, accessible on LAN
.venv_nitro\Scripts\python.exe src\\tools\\darwin_dashboard_web.py --host 0.0.0.0 --port 8080
```

---

## 4. Startup Sequence

### 4.1 What happens when `python src/tools/darwin_dashboard_web.py` runs

```
PHASE 0: Import + resolve paths        (~50ms)
  ├─ Import fastapi, uvicorn, jinja2, asyncio, sqlite3, ...
  ├─ Create DashboardConfig via path resolution
  ├─ Validate: at least template file exists (fatal if missing)
  └─ Data files may be absent → warn, continue in historical mode

PHASE 1: Init SQLite                    (~20ms)
  ├─ sqlite3.connect(metrics.db, check_same_thread=False)
  ├─ PRAGMA journal_mode=WAL
  ├─ PRAGMA synchronous=NORMAL  (durability trade-off for write speed)
  ├─ CREATE TABLE IF NOT EXISTS metrics (...)
  ├─ CREATE TABLE IF NOT EXISTS gpu_snapshots (...)
  ├─ CREATE TABLE IF NOT EXISTS checkpoint_history (...)
  └─ Enable WAL autocheckpoint (default 1000 pages)

PHASE 2: Parse logbook snapshot         (~10ms)
  ├─ Read DIARIO_DE_BORDO.md
  ├─ Extract ESTADO ATUAL table (regex: |\s*Campo\s*\|\s*Valor\s*| pattern)
  ├─ Extract agent entries (regex: ### YYYY-MM-DD HH:MM UTC | [name])
  ├─ Write agent_timeline.json cache
  └─ If logbook missing → warn, timeline will be empty

PHASE 3: Build FastAPI app              (~20ms)
  ├─ app = FastAPI(title="F51 Darwin-X Mission Control")
  ├─ mount StaticFiles at /static (src/tools/static/)
  ├─ Jinja2Templates at src/tools/templates/
  ├─ Register routes:
  │   GET  /dashboard              → serve Jinja2 template (initial render)
  │   GET  /api/sections/{name}    → htmx partial renders (6 endpoints)
  │   GET  /api/sse/stream         → SSE event stream (long-lived)
  │   GET  /api/status             → JSON health check
  │   GET  /api/history/{metric}?hours=N  → SQLite query for chart backfill
  ├─ Register startup event  → spawn background asyncio tasks
  └─ Register shutdown event → cleanup (section 5)

PHASE 4: Spawn background collectors    (~10ms)
  ├─ asyncio.create_task(data_collector_loop(config))  — polls files every 2s
  ├─ asyncio.create_task(gpu_collector_loop(config))   — polls nvidia-smi every 2s
  └─ asyncio.create_task(logbook_watcher_loop(config)) — polls DIARIO mtime every 10s

PHASE 5: Start uvicorn                  (indefinite)
  ├─ uvicorn.run(app, host=config.host, port=config.port, log_level="info")
  ├─ Print banner: URL, lineage, checkpoint root, data file status
  └─ Block until SIGINT (Ctrl+C) or SIGTERM
```

### 4.2 Banner printed at startup

```
╔══════════════════════════════════════════════════════════════╗
║    F51 Darwin-X Mission Control — Web Dashboard            ║
║    http://127.0.0.1:8000/dashboard                          ║
╠══════════════════════════════════════════════════════════════╣
║    Lineage:      100M (V9)                                  ║
║    Checkpoint:   cycle 46, step 14500                       ║
║    Dopamine:     NOT FOUND (training not running)           ║
║    Blockchain:   NOT FOUND (not yet created)                ║
║    SQLite:       workspace/runtime/dashboard/metrics.db     ║
║    Logbook:      12 entries parsed                          ║
╠══════════════════════════════════════════════════════════════╣
║    Ctrl+C to stop                                           ║
╚══════════════════════════════════════════════════════════════╝
```

### 4.3 "Historical View" mode (training NOT running)

When `dopamine_status.txt` is absent or stale (mtime > 60s old):
- Hero shows last known loss from `organism_latest.json` with "(stopped)" label
- GPU cards show current GPU state (nvidia-smi still works)
- Charts show last 100 points from SQLite history, then flat-line
- ConnectionStatus indicator shows "Training: STOPPED" (red)
- All sections still render — just with empty/last-known data
- Background poller still runs; if training starts and dopamine appears, dashboard detects it and transitions to live mode

### 4.4 Detecting "training just started"

The poller compares dopamine mtime. If the file was absent/missing and suddenly appears (or its step counter jumps forward after being stale), the dashboard:
1. Appends a `run_start` event to `runs.jsonl` with timestamp
2. Emits an SSE `training_status` event: `{"status": "started", "timestamp": "..."}`
3. Updates ConnectionStatus: "Training: LIVE" (green pulse)

---

## 5. Shutdown Sequence

### 5.1 Signal handling (SIGINT / Ctrl+C, SIGTERM)

```python
# Registered as FastAPI shutdown event + signal handler
async def shutdown():
    # 1. Set global flag to stop background tasks
    _shutdown_requested = True

    # 2. Cancel background asyncio tasks with timeout
    for task in [_collector_task, _gpu_task, _logbook_task]:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # 3. Flush in-memory deques to SQLite
    for metric_name, deque in _live_deques.items():
        if deque:
            _flush_deque_to_sqlite(metric_name, deque)

    # 4. SQLite WAL checkpoint (flush WAL to main DB file)
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # 5. Close SQLite connection
    db.close()

    # 6. Append dashboard stop event to DIARIO_DE_BORDO.md
    #    (with advisory lock, 2s timeout)
    _logbook_append_shutdown()

    # 7. Clean up SSE connections
    for queue in _sse_queues:
        queue.put_nowait({"event": "shutdown", "data": "{}"})
```

### 5.2 SSE client disconnect handling

- Each SSE connection has an `asyncio.Queue`.
- On client disconnect (detected via `request.is_disconnected()`), the queue is removed from the broadcast list.
- No server-side state per client — queues are just references. GC handles cleanup.

### 5.3 Crash recovery

- All state is on disk: SQLite WAL survives process death (WAL mode auto-recovers).
- `runs.jsonl` is append-only with fsync after each write.
- On restart, dashboard reads SQLite for chart history, re-parses logbook, and resumes.
- In-memory deques are lost on crash — rebuilt from SQLite on restart (last 7200 points per metric).
- SSE clients see connection drop → browser `EventSource` auto-reconnects (built-in).

---

## 6. Error Handling Strategy

### 6.1 Fatal errors (crash the server)

These conditions cause the server to exit with a clear message:

| Error | Reason |
|---|---|
| Template file not found (`src/tools/templates/dashboard.html`) | Cannot render anything |
| Static directory not found (`src/tools/static/`) | htmx/Chart.js not loadable |
| SQLite database cannot be created (disk full, permissions) | Cannot persist metrics |
| Port already in use | Cannot serve |
| DIARIO_DE_BORDO.md not found | Protocol violation — logbook is mandatory |

### 6.2 Non-fatal errors (log warning, continue)

These conditions are logged at WARNING level and the affected section shows a fallback state:

| Error | Recovery |
|---|---|
| `dopamine_status.txt` missing or unparseable | Hero shows "--" values, "Training stopped" label |
| `organism_latest.json` missing or corrupt JSON | Hero shows "No checkpoint" state |
| `blocks.jsonl` missing | Blockchain panel shows "No blockchain data — training must run with --blockchain-enabled" |
| `causal_events.jsonl` missing | Ledger panel shows "No causal events" |
| `nvidia-smi` returns error or times out (>5s) | GPU cards show "GPU unavailable" |
| `config.yaml` parse fails | Use hardcoded defaults for checkpoint_root |
| Logbook parse error (regex mismatch) | Timeline shows "Parse error" with line number, rest of dashboard works |
| SQLite write fails (disk full mid-session) | In-memory deques continue accumulating, log warning every 60s |
| SSE queue full (slow client) | Drop oldest event, emit `queue_pressure` warning |
| `msvcrt.locking` timeout on logbook write | Queue the append, retry on next poll cycle (max 3 retries) |
| `subprocess` timeout on nvidia-smi | Return empty GPU list, log once per 60s |

### 6.3 Error state indicators (browser-visible)

Each section has a `data-error` attribute that htmx checks:

```html
<!-- Normal state: htmx swaps inner content -->
<div id="section-hero" hx-get="/api/sections/hero" hx-trigger="every 2s">
  ... hero content ...
</div>

<!-- Error state: htmx detects HTTP 500 and shows fallback -->
<!-- The /api/sections/hero endpoint returns 200 with a data-ok attribute -->
<!-- On 500, htmx's error handling shows a dimmed card with the error message -->
```

htxm error extension (inlined in template):

```javascript
document.body.addEventListener('htmx:responseError', function(evt) {
    // Show a subtle yellow strip at top with error summary
    // Auto-dismiss after 5s
    // Do NOT replace the section content — keep last good state visible
});
```

### 6.4 Resilience summary

```
                    ┌─────────────────────────────────────┐
                    │         DOES THIS CRASH?            │
                    ├──────────────┬──────────────────────┤
                    │ dopamine.txt │  NO — show "--"      │
                    │ missing      │                      │
                    ├──────────────┼──────────────────────┤
                    │ organism.json│  NO — show "No ckpt" │
                    │ corrupt JSON │                      │
                    ├──────────────┼──────────────────────┤
                    │ blocks.jsonl │  NO — show "Inactive"│
                    │ missing      │                      │
                    ├──────────────┼──────────────────────┤
                    │ nvidia-smi   │  NO — show "GPU N/A" │
                    │ fails        │                      │
                    ├──────────────┼──────────────────────┤
                    │ SQLite disk  │  NO — keep deques,   │
                    │ full         │  warn every 60s       │
                    ├──────────────┼──────────────────────┤
                    │ DIARIO.md    │  YES — fatal error,  │
                    │ missing      │  protocol violation   │
                    ├──────────────┼──────────────────────┤
                    │ template     │  YES — fatal error,  │
                    │ missing      │  cannot render page   │
                    ├──────────────┼──────────────────────┤
                    │ port in use  │  YES — fatal error,  │
                    │              │  cannot start server  │
                    └──────────────┴──────────────────────┘
```

---

## 7. Testing Plan

### 7.1 Unit tests (no server needed)

#### Test 1: Path resolution
```python
# test: mock project_root, verify all derived paths are correct
# file: create_fake_tree() → resolve_paths() → assert paths match expected
# pass: all 12 paths resolve to correct absolute paths
```

#### Test 2: Logbook parser
```python
# test: parse a known DIARIO_DE_BORDO.md snapshot
# file: mock_diario.md with 5 entries + ESTADO ATUAL table
# assert: 5 entries parsed, correct timestamps, correct agent names
# assert: ESTADO ATUAL fields extracted (cycle=46, step=14500, etc.)
# edge: empty file, file with only header, file with corrupted entry
# edge: entry with multiline WARNINGS block
```

#### Test 3: Dopamine parser
```python
# test: parse "step=14520 loss=5.922 jepa_raw=0.1061 dope=0.0032 delta=-0.0241"
# assert: step=14520, loss=5.922, jepa_raw=0.1061, dope=0.0032, delta=-0.0241
# edge: empty file, malformed line, missing keys, negative values
```

#### Test 4: organism_latest.json parser
```python
# test: parse real file from workspace
# assert: cycle=46, step=14500, checkpoint_version=9
# edge: file missing, file with missing keys, base_checkpoint_id=None
```

#### Test 5: Blockchain JSONL parser
```python
# test: mock blocks.jsonl with 3 blocks
# assert: 3 blocks parsed, block_hash extracted, cycle/step from header
# edge: empty file, file with blank lines, corrupt JSON on last line
```

#### Test 6: SQLite schema
```python
# test: create in-memory SQLite, run schema DDL, insert sample row
# assert: 3 tables exist with correct columns
# assert: WAL mode enabled
# edge: unique constraint on (cycle, step) for metrics table
```

#### Test 7: GPU parser (nvidia-smi output)
```python
# test: mock nvidia-smi CSV output
# assert: 2 GPUs parsed, correct index/name/util/mem/temp
# edge: empty output, no GPUs, timeout
```

### 7.2 Integration tests (start server, hit endpoints)

#### Test 8: Server starts in historical mode
```powershell
# No training running — uses existing checkpoint data
.venv_nitro\Scripts\python.exe src\\tools\\darwin_dashboard_web.py --port 18765 &
sleep 2
curl http://127.0.0.1:18765/api/status
# assert: 200 OK, {"status": "ok", "mode": "historical", "lineage": "100m"}
```

#### Test 9: Dashboard page renders (HTTP 200)
```powershell
curl http://127.0.0.1:18765/dashboard
# assert: 200 OK, Content-Type: text/html
# assert: contains "F51 Darwin-X", "Mission Control"
# assert: contains 8 grid sections
```

#### Test 10: htmx partial endpoints
```powershell
curl http://127.0.0.1:18765/api/sections/hero
# assert: 200 OK, HTML fragment with loss/cycle/step
curl http://127.0.0.1:18765/api/sections/gpu
# assert: 200 OK, HTML fragment with GPU cards (or "GPU unavailable")
curl http://127.0.0.1:18765/api/sections/training
# assert: 200 OK
curl http://127.0.0.1:18765/api/sections/blockchain
# assert: 200 OK
curl http://127.0.0.1:18765/api/sections/hashes
# assert: 200 OK
curl http://127.0.0.1:18765/api/sections/timeline
# assert: 200 OK, HTML fragment with agent entries
```

#### Test 11: SSE stream emits events
```powershell
curl -N http://127.0.0.1:18765/api/sse/stream &
sleep 4
# assert: received 'hero' event with JSON data
# assert: received 'gpu' event with JSON data
# assert: event format is valid SSE (event:NAME\ndata:JSON\n\n)
kill %1
```

#### Test 12: Static files served
```powershell
curl http://127.0.0.1:18765/static/htmx.min.js
# assert: 200 OK, Content-Type: application/javascript
curl http://127.0.0.1:18765/static/chart.min.js
# assert: 200 OK
curl http://127.0.0.1:18765/static/dashboard.css
# assert: 200 OK, Content-Type: text/css
```

### 7.3 Synthetic data tests (training simulation)

#### Test 13: Dopamine update detection
```python
# Setup: write initial dopamine_status.txt with step=100, loss=6.0
# Server polls → emits SSE 'hero' with loss=6.0
# Update: write step=101, loss=5.9 (simulates training step)
# Server polls → detects mtime change → emits SSE 'hero' with loss=5.9
# assert: chart_point event emitted with new data point
```

#### Test 14: Training start/stop detection
```python
# Setup: no dopamine_status.txt (mimic stopped training)
# Server: shows historical mode
# Create: dopamine_status.txt appears (mimic training started)
# Server: detects new file → emits training_status:started
# Delete: dopamine_status.txt (mimic training process died)
# Server: after 3 polls (6s) with no file → emits training_status:stopped
```

#### Test 15: Concurrent SSE clients
```python
# Setup: start server, connect 3 SSE clients
# Simulate: dopamine update
# assert: all 3 clients receive the same event
# Disconnect: kill 1 client
# assert: no broadcast error, remaining 2 clients still receive events
```

### 7.4 Manual QA checklist (browser)

| # | Check | Expected |
|---|---|---|
| 1 | Open `http://localhost:8000/dashboard` | Page loads, dark theme, no JS console errors |
| 2 | Hero section | Shows loss, cycle, step (or "--" if no data) |
| 3 | GPU cards | Shows GPU utilization bars and memory (or "GPU unavailable") |
| 4 | 4 charts | Canvas rendered, axes labeled, dark background |
| 5 | Blockchain panel | Shows block count, head hash (or "No blockchain") |
| 6 | Organ grid | Shows 17 chips with organ names |
| 7 | Hash section | Shows SHA-256 prefixes for config/checkpoint files |
| 8 | Agent timeline | Shows entries from DIARIO_DE_BORDO.md with timestamps |
| 9 | CSS Grid responsive | Resize to 1920→1440→1080: columns change 4→4→3 |
| 10 | Ctrl+C shutdown | Server exits cleanly, "Shutting down" message printed |

### 7.5 Performance checks

| Check | Target |
|---|---|
| Poll cycle CPU usage | < 5% of one core (file reads + JSON parse + SQLite insert) |
| SQLite insert latency | < 2ms per row (WAL mode, single-row inserts) |
| SSE broadcast latency | < 5ms from file detection to queue.put() |
| Template render time | < 50ms for initial page load |
| Memory usage (idle) | < 80MB (Python + FastAPI + asyncio event loop) |
| Memory usage (active) | < 120MB (with deques at capacity 7200x4) |

---

## 8. Internal Architecture of `darwin_dashboard_web.py`

### 8.1 Class/function structure (~400 lines)

```
darwin_dashboard_web.py
│
├── [1-10]   Module docstring + imports
├── [11-40]  DashboardConfig dataclass
├── [41-70]  resolve_paths(lineage=None) → DashboardConfig
├── [71-100] SQLite helpers: init_db(), insert_metric(), query_history()
│
├── [101-160] Data collectors (called by async loop)
│   ├── read_dopamine(path) → dict
│   ├── read_pointer(path) → dict
│   ├── read_blockchain(path) → dict
│   ├── read_ledger(path) → dict
│   ├── read_gpu() → list[dict]
│   └── compute_hashes(config) → dict
│
├── [161-220] Logbook parser
│   ├── parse_estado_atual(text) → dict
│   ├── parse_agent_entries(text) → list[dict]
│   └── regenerate_timeline_cache(config)
│
├── [221-280] SSE infrastructure
│   ├── SSEQueueManager (broadcast list + add/remove)
│   ├── sse_event_generator(queue) → async generator
│   └── broadcast_to_all(event_name, data_dict)
│
├── [281-320] FastAPI app + routes
│   ├── GET /dashboard
│   ├── GET /api/sections/{name}  (6 sections)
│   ├── GET /api/sse/stream
│   ├── GET /api/status
│   └── GET /api/history/{metric}?hours=N
│
├── [321-360] Async background loops
│   ├── data_collector_loop(config)
│   ├── gpu_collector_loop(config)
│   └── logbook_watcher_loop(config)
│
├── [361-380] Shutdown handler
│   └── shutdown_event()
│
└── [381-400] main() — arg parse, config resolve, uvicorn.run()
```

### 8.2 Data flow per 2-second tick

```
┌──────────────────────────────────────────────────────┐
│  data_collector_loop (runs every 2s)                  │
│                                                      │
│  1. stat dopamine_status.txt → if mtime changed:     │
│     ├─ read + parse key=value pairs                  │
│     ├─ broadcast SSE 'hero' event                    │
│     ├─ broadcast SSE 'training_metrics' event        │
│     ├─ broadcast SSE 'chart_point' for loss/ppl/tok/dop│
│     ├─ append to in-memory deques (maxlen=7200)      │
│     └─ every 10th tick: flush deques to SQLite       │
│                                                      │
│  2. stat organism_latest.json → if mtime changed:    │
│     ├─ read + parse JSON                             │
│     └─ broadcast SSE 'hero' event (checkpoint info)  │
│                                                      │
│  3. stat blocks.jsonl → if size changed:             │
│     ├─ read last line (new block)                    │
│     ├─ broadcast SSE 'blockchain' event              │
│     └─ insert checkpoint_history if new cycle        │
│                                                      │
│  4. Training process detection:                      │
│     ├─ tasklist /fi "imagename eq python.exe"        │
│     ├─ if state changed (none→found or found→none)   │
│     └─ append event to runs.jsonl                     │
│                                                      │
│  5. Every 30 ticks (60s):                            │
│     ├─ recompute SHA-256 hashes                      │
│     └─ broadcast SSE 'hashes' event                  │
│                                                      │
│  gpu_collector_loop (runs every 2s, separate task)   │
│     ├─ nvidia-smi --query-gpu=... --format=csv       │
│     ├─ parse CSV output                              │
│     ├─ broadcast SSE 'gpu' event                     │
│     ├─ insert into gpu_snapshots table               │
│     └─ on nvidia-smi error: emit empty GPU event     │
│                                                      │
│  logbook_watcher_loop (runs every 10s, separate task)│
│     ├─ stat DIARIO_DE_BORDO.md mtime                 │
│     ├─ if changed: re-parse all entries              │
│     ├─ regenerate agent_timeline.json cache          │
│     └─ broadcast SSE 'timeline' event                │
└──────────────────────────────────────────────────────┘
```

### 8.3 SSE event types emitted

| Event Name | Payload | Frequency | Consumer |
|---|---|---|---|
| `hero` | `{loss, cycle, step, checkpoint_cycle, checkpoint_step, saved_at, base_id_short}` | 2s | Hero section + CycleStepChip |
| `gpu` | `[{idx, name, util, mem_used, mem_total, temp}]` | 2s | GPU cards |
| `training` | `{step, loss, jepa_raw, dope, delta, gmc, tok_s, ppl}` | 2s | MetricRow |
| `chart_point` | `{ts, loss, ppl, tok_s, dopamine, step}` | 2s | All 4 Chart.js instances |
| `blockchain` | `{blocks, head_hash, last_cycle, size_kb}` | 10s | BlockchainCard |
| `organs` | `[{name, identity_short, status}]` | 60s | OrganChip grid |
| `hashes` | `{config_sha, checkpoint_sha, pointer_sha, blocks_sha, ledger_sha}` | 60s | IntegrityRow |
| `timeline` | `[{timestamp, agent, operation, summary}]` | on logbook change | AgentTimeline |
| `health` | `{cpu_pct, mem_gb, disk_free_gb, uptime_s, training_running}` | 10s | SystemHealthBar |
| `training_status` | `{status: "started"|"stopped", timestamp}` | on change | ConnectionStatus |
| `retry` | `{after_ms: 2000}` | on SSE setup | EventSource reconnect |

---

## 9. SQLite Schema

```sql
-- Table 1: Training metrics (one row per step)
CREATE TABLE IF NOT EXISTS metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL NOT NULL,           -- unix epoch with fractional seconds
    step        INTEGER NOT NULL,
    cycle       INTEGER NOT NULL,
    loss        REAL,
    ppl         REAL,                    -- perplexity (exp(loss))
    tok_s       REAL,                    -- tokens per second
    jepa_raw    REAL,
    dope        REAL,                    -- dopamine signal
    delta       REAL,                    -- delta_loss
    gmc         REAL,                    -- gradient magnitude coefficient
    lm_loss     REAL,
    aux_loss    REAL,
    UNIQUE(cycle, step)                  -- deduplicate on re-poll
);

-- Table 2: GPU snapshots (one row per GPU per 10s)
CREATE TABLE IF NOT EXISTS gpu_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL NOT NULL,
    gpu_index   INTEGER NOT NULL,
    util_pct    INTEGER,
    mem_used_mb INTEGER,
    mem_total_mb INTEGER,
    temp_c      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_gpu_ts ON gpu_snapshots(timestamp);

-- Table 3: Checkpoint history (one row per checkpoint save)
CREATE TABLE IF NOT EXISTS checkpoint_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL NOT NULL,
    cycle       INTEGER NOT NULL,
    step        INTEGER NOT NULL,
    version     INTEGER NOT NULL,
    base_id     TEXT,
    size_bytes  INTEGER,
    path        TEXT
);
```

---

## 10. Key Design Decisions

### 10.1 Why SSE over WebSocket?

- SSE is unidirectional (server to client) — dashboard only needs push, never client-to-server commands.
- SSE auto-reconnects (browser built-in `EventSource`), no custom reconnect logic.
- SSE works through HTTP proxies; WebSocket sometimes doesn't.
- Fewer lines of code: `EventSource` in browser, `StreamingResponse` in FastAPI.
- htmx has native SSE extension (`hx-ext="sse"`).

### 10.2 Why in-memory deques + periodic SQLite flush?

- Charts need fast access to the last N points (array append + shift).
- SQLite SELECT for every chart render would add latency.
- Deques kept in memory; flushed to SQLite every 10 ticks (20s) for crash recovery.
- On restart, last 7200 points loaded from SQLite into deques.

### 10.3 Why single-file backend?

- 400 lines is maintainable by a single developer.
- No cross-module imports to debug at 2am when training crashes.
- Easy to audit — one file to read top to bottom.
- The 400-line limit is enforced by the plan: if a section grows, simplify rather than extract.

### 10.4 Why htmx partials + SSE (two transport modes)?

- Initial page load: Jinja2 renders complete HTML with all sections.
- htmx polls `/api/sections/{name}` every N seconds for the sections that change at different rates.
- SSE pushes real-time data (hero, GPU, chart points) for sub-2-second updates and chart data.
- This hybrid approach avoids the "SSE for everything" complexity (SSE requires JavaScript parsing per event) and the "htmx-only" latency (polls can't go below 1s without hammering the server).
- In the current template design, SSE is the primary transport; htmx sections are fallback.

---

## 11. Download Instructions (for implementer)

Before running the dashboard for the first time:

```powershell
# 1. Install pip deps (2 packages, ~25MB)
.venv_nitro\Scripts\pip.exe install fastapi uvicorn

# 2. Download htmx 2.x (14KB)
#    From: https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js
#    Save to: src/tools/static/htmx.min.js
Invoke-WebRequest -Uri "https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js" `
    -OutFile "src/tools/static/htmx.min.js"

# 3. Download Chart.js 4.x (65KB)
#    From: https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js
#    Save to: src/tools/static/chart.min.js
Invoke-WebRequest -Uri "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js" `
    -OutFile "src/tools/static/chart.min.js"

# 4. Verify static files exist
Get-ChildItem src/tools/static/htmx.min.js, src/tools/static/chart.min.js, src/tools/static/dashboard.css

# 5. Start dashboard
.venv_nitro\Scripts\python.exe src\\tools\\darwin_dashboard_web.py
```

---

## 12. What This Plan Does NOT Cover

- **Multi-lineage support (600M, 1.6B)**: v1 targets 100M only. The `--lineage` flag is future-proofing; switching lineages requires the lineage to have a checkpoint_root with organism_latest.json.
- **Authentication**: Dashboard runs on localhost only (`--host 127.0.0.1`). Add nginx reverse proxy with basic auth for remote access.
- **Alerting/notifications**: Dashboard is a read-only observer. It never sends emails, Slack messages, or push notifications.
- **Mobile layout**: v1 targets desktop browsers (1080p+). Mobile breakpoints are future work.
- **Organ weights detail**: The organ grid shows organ names and active/inactive status from organism_latest.json metadata. Individual weight values require checkpoint loading (torch) which is excluded by design.
- **Training control**: Dashboard cannot start/stop training. Use `src/scripts/start_100m_auto.ps1` for that.
- **Dual GPU optimization advice**: The dashboard shows metrics for both GPUs but does not recommend GPU allocation changes.
