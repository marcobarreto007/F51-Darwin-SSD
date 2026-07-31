# F51 Darwin-X Mission Control -- Data Flow & Wire Protocol

> **Version:** 1.0.0
> **Date:** 2026-07-24
> **Wave:** Wave 3 (implementation)
> **Audience:** Backend coders implementing `darwin_dashboard_web.py`

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  Training Process (organism.py)                                  │
│    │  writes every ~2s:                                          │
│    ├── dopamine_status.txt     (key=value lines)                 │
│    ├── causal_events.jsonl     (JSONL, append-only)             │
│    ├── blocks.jsonl            (JSONL, append-only)             │
│    └── organism_latest.json    (overwritten on checkpoint save)  │
└──────────────────┬───────────────────────────────────────────────┘
                   │  filesystem (same machine)
┌──────────────────▼───────────────────────────────────────────────┐
│  Dashboard Backend (darwin_dashboard_web.py, FastAPI+uvicorn)    │
│    │                                                             │
│    ├── PollerLoop (asyncio, 2s tick)                             │
│    │     reads mtime+size → parses deltas → 3 priority queues    │
│    │                                                             │
│    ├── SQLite (WAL mode, metrics.db)                             │
│    │     writes every poll cycle, reads for history seeding      │
│    │                                                             │
│    ├── LogbookParser (regex, cached, file-locked)                │
│    │                                                             │
│    └── SSEManager                                                │
│          broadcasts typed events to all connected browsers       │
└──────────────────┬───────────────────────────────────────────────┘
                   │  HTTP SSE (text/event-stream)
                   │  single persistent connection per client
┌──────────────────▼───────────────────────────────────────────────┐
│  Browser (dashboard.html + htmx + Chart.js)                      │
│    ├── 4x Chart.js canvases (loss, ppl, tok_s, dopamine)         │
│    ├── In-memory deques (max 7200 points = 4 hours)              │
│    ├── DOM updates via JS EventSource handlers                   │
│    └── htmx for section-level swaps (blockchain, organs, etc.)   │
└──────────────────────────────────────────────────────────────────┘
```

**Key constraint:** All data sources are local files written by the training process.
The dashboard NEVER requires the training process to be running. When training is
stopped, the dashboard shows last-known values with `STALE` indicators.

---

## 2. Source File Formats (confirmed from actual workspace data)

### 2.1 `dopamine_status.txt`

```
Location:  <CKPT_ROOT>/dopamine_status.txt
Format:    space-separated key=value pairs, one line per step
Written:   training loop, every step (~2s)
Atomic:    NO -- file is rewritten in-place. Read size before parse to detect
           partial writes.

Example (actual from cycle 47):
  step=14525 loss=5.287 jepa_raw=0.1073 dope=0.159 delta=0.0221 gmc=0.0051 jepa_bonus=0.03

Observed keys (varies by training phase):
  REQUIRED: step, loss
  COMMON:   jepa_raw, dope, delta, gmc, jepa_bonus, tok_s, ppl
  OPTIONAL: mtp, ghost, spider, lm_loss, aux_loss
  The set of keys is NOT guaranteed stable. Parse only what is present.
  All values are numeric except `step` which is an integer.
```

### 2.2 `organism_latest.json`

```
Location:  <CKPT_ROOT>/organism_latest.json
Format:    single JSON object, overwritten atomically on each checkpoint save
Written:   every ~100 steps (~200s)
Atomic:    YES (write-to-temp then rename)

Example (actual):
{
  "version": 1,
  "checkpoint_version": 9,
  "path": "organism_cycle_046_step_014500.pt",
  "cycle": 46,
  "step": 14500,
  "size_bytes": 987173360,
  "base_checkpoint_id": "darwin-model-core-v1:2b648ee2784b60ee16d6a69b3e49979a699e0f9a7345392642f2070992b207b7",
  "saved_at": "2026-07-24T12:01:04.301826+00:00"
}

Schema:
  version               integer   -- pointer format version
  checkpoint_version    integer   -- checkpoint schema version (currently 9)
  path                  string    -- filename relative to CKPT_ROOT
  cycle                 integer   -- cycle number at save time
  step                  integer   -- global step at save time
  size_bytes            integer   -- file size in bytes
  base_checkpoint_id    string    -- canonical model identity
  saved_at              string    -- ISO 8601 UTC timestamp
```

### 2.3 `causal_events.jsonl`

```
Location:  <CKPT_ROOT>/causal_events.jsonl
Format:    JSONL, one JSON object per line, append-only
Written:   every step (~2s), one INTENT + one OUTCOME per step
Rotation:  CausalLedger rotates at max_events (default 50,000).

Top-level fields (every event):
  event_hash     string    SHA-256 of this event
  event_type     string    "STEP_INTENT" | "STEP_OUTCOME"
  prev_hash      string    SHA-256 of previous event (hash chain)
  schema_version integer   currently 3
  sequence       integer   monotonic counter
  step           object    step identity block

Step identity block:
  ablation_plan_id          string  -- "shadow" | "control" | "enforce"
  attempt_id                string  -- "attempt-N" (unique per step attempt)
  base_checkpoint_id        string  -- model identity
  batch_digest              string  -- SHA-256 of token batch
  cycle                     integer -- current cycle number
  key                       string  -- unique step key
  optimizer_step            integer -- global optimizer step number
  rng_digest                string  -- SHA-256 of RNG state
  run_id                    string  -- "F51-Darwin-Organism"

STEP_INTENT payload (abbreviated -- actual is deeply nested):
  arm       string    current causal arm
  decisions array     phase decisions (PRE_LOSS, PRE_BACKWARD, PRE_OPTIMIZER)

STEP_OUTCOME payload (keys relevant to dashboard):
  effective_losses.lm       number  -- language modeling loss    ★ PRIMARY
  effective_losses.total    number  -- total loss (raw+aux)      ★
  effective_losses.aux      number  -- auxiliary loss
  effective_losses.jepa     number  -- JEPA loss
  effective_losses.ghost    number  -- Ghost (MAE) loss
  effective_losses.mtp      number  -- multi-token prediction loss
  effective_losses.spider   number  -- spider calibration loss
  raw_losses                object  -- same structure, pre-weighting
  gradient_norm_before      number  -- pre-clip gradient norm
  gradient_norm_after       number  -- post-clip gradient norm
  optimizer_step_applied    boolean -- whether optimizer stepped
  error                     null|string -- error message if any
```

### 2.4 `blocks.jsonl`

```
Location:  workspace/runtime/organism/blockchain/blocks.jsonl
Format:    JSONL, one JSON object per line, append-only
Written:   every step (~2s) when --blockchain-enabled
Atomic:    each line is a complete JSON object followed by \n

Example (actual block #168):
{
  "block_hash": "25547e73e07de667fd42cf277aea4a5f80eadcc8bdf8ea443c9c70d891ff4605",
  "transactions": [],
  "header": {
    "block_number": 168,
    "checkpoint_flag": false,
    "checkpoint_sha256": null,
    "cycle": 47,
    "merkle_root": "0000000000000000000000000000000000000000000000000000000000000000",
    "optimizer_step": 14528,
    "organ_identity_merkle_root": "0000000000000000000000000000000000000000000000000000000000000000",
    "prev_block_hash": "dd3becbb14805e888c321170c04687a6d81aaa35f0f58074a6da2042c640bb85",
    "schema": "darwin-organ-block-v1",
    "step_key": "step-14528-cycle-47",
    "timestamp_unix_ms": 1784905923576
  }
}
```

### 2.5 `DIARIO_DE_BORDO.md`

```
Location:  <ROOT>/DIARIO_DE_BORDO.md
Format:    Markdown with structured sections

Key sections parsed by LogbookParser:
  - "## ESTADO ATUAL" → markdown table → dict
  - "## Bugs conhecidos" → numbered list → list of {id, title, status}
  - "### YYYY-MM-DD HH:MM UTC | [Agent Name]" → agent entry block
    Each entry contains:
      OPERAÇÃO: READ|WRITE|EXECUTE|DECIDE|MIXED
      ESTADO ENCONTRADO: bullet list
      AÇÕES: numbered list
      ESTADO DEIXADO: bullet list
      COMMIT: hash -- description
      WARNINGS: text
```

### 2.6 Config YAML

```
Location:  <ROOT>/src/configs/darwin_x_100m.yaml
Format:    YAML (lib: PyYAML)
Keys used by dashboard:
  model_name, d_model, n_layers, n_heads, n_kv_heads,
  context_length, vocab_size,
  fine_experts, shared_experts, experts_per_token,
  blockchain_enabled, senate_enabled,
  *all *_enabled and *_weight keys (organ toggles)
```

---

## 3. SSE Event Catalog

### 3.1 Protocol

```
Endpoint:     GET /dashboard/sse/stream
Content-Type: text/event-stream
Cache-Control: no-cache
Connection:    keep-alive
Reconnection:  client EventSource auto-reconnects; server sends retry: 2000
Heartbeat:     server sends ": heartbeat\n\n" comment line every 15s
               to keep the connection alive through proxies.
```

### 3.2 Event Types and JSON Schemas

Every SSE event format:
```
event: <event_type>
data: <compact JSON, no extraneous whitespace>
<blank line>
```

---

#### `event: heartbeat`

**Frequency:** every 15s from server
**Consumers:** ConnectionStatus component

```json
{
  "server_time_utc": "2026-07-24T19:30:00.000Z",
  "dashboard_uptime_s": 1234.5,
  "client_count": 2
}
```

Schema:
| Field | Type | Description |
|---|---|---|
| server_time_utc | string | ISO 8601 UTC timestamp |
| dashboard_uptime_s | number | Seconds since dashboard process started |
| client_count | integer | Number of connected SSE clients |

---

#### `event: hero`

**Frequency:** every 2s (tier_fast)
**Consumers:** HeroLoss, CycleStepChip, header clock

```json
{
  "loss": 5.2872,
  "loss_previous": 5.2911,
  "loss_trend": "down",
  "cycle": 47,
  "step": 14525,
  "ckpt_cycle": 46,
  "ckpt_step": 14500,
  "version": 9,
  "timestamp": "2026-07-24 19:30:00 UTC"
}
```

Schema:
| Field | Type | Description |
|---|---|---|
| loss | number\|null | Current total loss. null if training stopped. |
| loss_previous | number\|null | Previous tick's loss for trend computation |
| loss_trend | string | "down" \| "up" \| "flat" \| "none" (first tick) |
| cycle | integer\|null | Current cycle number |
| step | integer\|null | Current global step |
| ckpt_cycle | integer\|null | Last checkpoint's cycle |
| ckpt_step | integer\|null | Last checkpoint's step |
| version | integer\|null | Checkpoint schema version |
| timestamp | string | Human-readable UTC time |

Edge cases:
- If `dopamine_status.txt` is missing or unparseable: all values `null`, dashboard shows `--` in gray.
- If training was running but no update for >30s: values are the last known ones, `ConnectionStatus` sets `training_status=hung`.

---

#### `event: gpu`

**Frequency:** every 2s (tier_fast)
**Consumers:** GpuCard x2

```json
{
  "gpus": [
    {"idx": "0", "name": "NVIDIA GeForce RTX 5060 Ti", "util": 85, "mem_used": 7752, "mem_total": 16311, "temp": 72},
    {"idx": "1", "name": "NVIDIA GeForce RTX 3060",     "util": 0,  "mem_used": 3,    "mem_total": 12288, "temp": 45}
  ],
  "timestamp": "2026-07-24T19:30:00Z"
}
```

Schema for each GPU object:
| Field | Type | Description |
|---|---|---|
| idx | string | GPU index from nvidia-smi |
| name | string | GPU product name (untruncated) |
| util | integer | GPU utilization percentage (0-100) |
| mem_used | integer | VRAM used in MiB |
| mem_total | integer | VRAM total in MiB |
| temp | integer | GPU temperature in Celsius. null if unsupported. |

Edge cases:
- If `nvidia-smi` is unavailable: `gpus` is an empty array `[]`.
- If `nvidia-smi` times out (>5s): re-send last known values.
- util > 100 (accounting artifact): clamp bar to 100%, keep label exact.
- Single GPU machine: array has 1 element. Grid adapts.
- No GPUs: array empty. Cards show "No NVIDIA GPUs detected".

---

#### `event: training`

**Frequency:** every 2s (tier_fast)
**Consumers:** MetricRow (7 rows)

```json
{
  "loss": 5.2872,
  "ppl": 197.8,
  "tok_s": 245.3,
  "jepa_raw": 0.1073,
  "dope": 0.159,
  "delta": 0.0221,
  "gmc": 0.0051,
  "loss_delta": -0.0039,
  "ppl_delta": -6.2,
  "tok_s_delta": 5.1,
  "timestamp": "2026-07-24T19:30:00Z"
}
```

Schema:
| Field | Type | Description |
|---|---|---|
| loss | number\|null | Effective total loss |
| ppl | number\|null | Perplexity = exp(loss) |
| tok_s | number\|null | Tokens per second throughput |
| jepa_raw | number\|null | Raw JEPA loss (before weight) |
| dope | number\|null | Dopamine scalar (0-1 range typically) |
| delta | number\|null | Delta loss |
| gmc | number\|null | Gradient magnitude consistency |
| loss_delta | number\|null | loss - previous_loss |
| ppl_delta | number\|null | ppl - previous_ppl |
| tok_s_delta | number\|null | tok_s - previous_tok_s |
| timestamp | string | ISO 8601 UTC timestamp |

Notes:
- `ppl` is computed server-side as `exp(loss)` since the raw field is not in dopamine_status.txt.
- If a metric key is absent from dopamine_status.txt, send `null` for that field. Client shows `--`.
- Delta values are `null` on the first tick after startup (no previous value).

---

#### `event: chart_point`

**Frequency:** every 2s (tier_fast)
**Consumers:** ChartCanvas x4 (Chart.js instances)

```json
{
  "ts": 1784905800.0,
  "loss": 5.2872,
  "ppl": 197.8,
  "tok_s": 245.3,
  "dope": 0.159
}
```

Schema:
| Field | Type | Description |
|---|---|---|
| ts | number | Unix timestamp in seconds (float) for X-axis position |
| loss | number\|null | Loss value for chart-loss |
| ppl | number\|null | Perplexity for chart-ppl |
| tok_s | number\|null | Tokens/s for chart-tok-s |
| dope | number\|null | Dopamine for chart-dopamine |

Notes:
- `NaN`, `Infinity`, and `null` values are NOT pushed into chart deques.
- The `ts` field uses `time.time()` (monotonic float seconds) for consistent X-axis spacing.
- Each chart deque independently trims to `MAX_POINTS=7200` (4 hours at 2s).

---

#### `event: blockchain`

**Frequency:** every 10s (tier_medium)
**Consumers:** BlockchainCard

```json
{
  "blocks": 168,
  "head_hash": "25547e73e07de667f...",
  "head_hash_short": "25547e73",
  "last_cycle": 47,
  "last_step": 14528,
  "size_kb": 95.9,
  "status": "ACTIVE",
  "timestamp": "2026-07-24T19:30:00Z"
}
```

Schema:
| Field | Type | Description |
|---|---|---|
| blocks | integer | Total block count (line count of file) |
| head_hash | string | Full SHA-256 of the most recent block |
| head_hash_short | string | First 8 hex chars of head_hash |
| last_cycle | integer\|null | Cycle of most recent block |
| last_step | integer\|null | Step of most recent block |
| size_kb | number | File size in KB |
| status | string | "ACTIVE" \| "NO FILE" \| "EMPTY" \| "PARSE ERROR" |
| timestamp | string | ISO 8601 UTC |

Status values:
- `"ACTIVE"` -- blocks.jsonl exists with >= 1 blocks, last block parsed successfully
- `"NO FILE"` -- blocks.jsonl does not exist (blockchain not recording or directory not created)
- `"EMPTY"` -- file exists but has 0 lines (first block pending)
- `"PARSE ERROR"` -- last line is invalid JSON

---

#### `event: organs`

**Frequency:** every 60s (tier_slow)
**Consumers:** OrganChip grid (17 chips)

```json
{
  "core":              {"active": true,  "status": "active",    "identity_hash": "5bf0be801def50f3..."},
  "moe":               {"active": true,  "status": "active",    "identity_hash": "..."},
  "jepa":              {"active": true,  "status": "active",    "identity_hash": "..."},
  "mtp":               {"active": true,  "status": "active",    "identity_hash": "..."},
  "spider_calibration":{"active": true,  "status": "active",    "identity_hash": "..."},
  "spider_sense":      {"active": true,  "status": "active",    "identity_hash": "..."},
  "gaba":              {"active": true,  "status": "reanimated","identity_hash": "92ff471a..."},
  "inter_hemispheric": {"active": true,  "status": "active",    "identity_hash": "..."},
  "ttm_residual":      {"active": true,  "status": "reanimated","identity_hash": "..."},
  "heartbeat":         {"active": true,  "status": "active",    "identity_hash": "..."},
  "ghost":             {"active": true,  "status": "active",    "identity_hash": "..."},
  "curiosity":         {"active": false, "status": "dead",      "identity_hash": "80437be5..."},
  "dae":               {"active": true,  "status": "active",    "identity_hash": "052fa5d0..."},
  "nitro":             {"active": true,  "status": "active",    "identity_hash": "..."},
  "decision_engine":   {"active": false, "status": "dead",      "identity_hash": "494246b6..."},
  "unified_mesh":      {"active": true,  "status": "active",    "identity_hash": "..."},
  "sleep":             {"active": true,  "status": "active",    "identity_hash": "..."},
  "timestamp": "2026-07-24T19:30:00Z"
}
```

Schema per organ:
| Field | Type | Description |
|---|---|---|
| active | boolean | Organ has an identity hash in the checkpoint |
| status | string | "active" \| "dormant" \| "dead" \| "reanimated" \| "disabled" |
| identity_hash | string\|null | Full organ identity SHA-256 string, or null |

Status semantics:
- `"active"` -- Identity found in checkpoint, config toggle enabled
- `"reanimated"` -- Identity exists but was recently reanimated by a resume hook (e.g., GABA/TTM gates were zero on load). Purple dot in UI.
- `"dormant"` -- No identity in checkpoint (organ never initialized)
- `"dead"` -- Code exists but output is discarded (curiosity, decision_engine)
- `"disabled"` -- Config toggle is `false`

Status determination logic:
1. Check config: `organ_enabled` → false → `"disabled"`
2. Check known-dead list: `["curiosity", "decision_engine"]` → `"dead"`
3. Check checkpoint identity: present → `"active"`
4. Check reanimation: in reanimated_organs set → `"reanimated"` (overrides #3)
5. No identity → `"dormant"`

---

#### `event: hashes`

**Frequency:** every 60s (tier_slow)
**Consumers:** IntegrityRow (5 rows), footer config hash

```json
{
  "checkpoint": {"hash_short": "d41d8cd98f00b204...", "hash_full": "d41d8cd98f00b204e9800998ecf8427e...", "exists": true,  "size_mb": 938.2},
  "config":     {"hash_short": "3a7f2e1b9c4d5f6a...", "hash_full": "3a7f2e1b9c4d5f6a...",              "exists": true,  "size_mb": 0.003},
  "ledger":     {"hash_short": "--",                   "hash_full": null,                              "exists": false, "size_mb": null},
  "blocks":     {"hash_short": "e99a18c428cb38d5...", "hash_full": "e99a18c428cb38d5...",              "exists": true,  "size_mb": 0.096},
  "script":     {"hash_short": "a1b2c3d4e5f6a7b8...", "hash_full": "a1b2c3d4e5f6a7b8...",              "exists": true,  "size_mb": 0.004},
  "verified_at": "2026-07-24T19:30:00Z",
  "verified_count": 4,
  "total_count": 5
}
```

Schema per file:
| Field | Type | Description |
|---|---|---|
| hash_short | string | First 16 hex chars + "..." or "--" if missing |
| hash_full | string\|null | Full 64-char SHA-256 hex string, or null |
| exists | boolean | Whether the file exists at the expected path |
| size_mb | number\|null | File size in MB, or null if missing |

Files tracked:
| label | path | expected default |
|---|---|---|
| checkpoint | `<CKPT_ROOT>/<organism_latest.path>` | should exist if training ran |
| config | `<ROOT>/src/configs/darwin_x_100m.yaml` | always exists |
| ledger | `<CKPT_ROOT>/causal_events.jsonl` | exists when training runs |
| blocks | `workspace/runtime/organism/blockchain/blocks.jsonl` | may not exist |
| script | `<ROOT>/src/scripts/start_100m_auto.ps1` | always exists |

---

#### `event: health`

**Frequency:** every 10s (tier_medium)
**Consumers:** SystemHealthBar

```json
{
  "ram_used_gb": 18.2,
  "ram_total_gb": 32.0,
  "ram_pct": 56.9,
  "disk_used_gb": 312.5,
  "disk_total_gb": 953.0,
  "disk_pct": 32.8,
  "disk_free_gb": 640.5,
  "process_count": 2,
  "process_total_mem_gb": 16.1,
  "uptime": "4h 32m",
  "uptime_s": 16320,
  "dashboard_uptime_s": 342.1,
  "timestamp": "2026-07-24T19:30:00Z"
}
```

Schema:
| Field | Type | Description |
|---|---|---|
| ram_used_gb | number | Used RAM in GB |
| ram_total_gb | number | Total RAM in GB |
| ram_pct | number | RAM usage percentage |
| disk_used_gb | number | Used disk space in GB |
| disk_total_gb | number | Total disk space in GB |
| disk_pct | number | Disk usage percentage |
| disk_free_gb | number | Free disk space in GB |
| process_count | integer | Number of running Python processes |
| process_total_mem_gb | number | Total RSS of all Python processes in GB |
| uptime | string\|null | Human-readable training uptime ("4h 32m"), null if stopped |
| uptime_s | number\|null | Training uptime in seconds, null if stopped |
| dashboard_uptime_s | number | Dashboard process uptime in seconds |
| timestamp | string | ISO 8601 UTC |

Edge cases:
- `psutil` not installed: all numeric fields null, client shows "psutil not installed".
- `process_count == 0`: expected when training is stopped. Show count in red.
- `ram_pct > 95`: OOM risk. Bar pulses red.
- `disk_pct > 95`: checkpoint save may fail. Bar pulses red.

---

#### `event: timeline`

**Frequency:** every 60s (tier_slow)
**Consumers:** AgentTimelineEntry

```json
{
  "entries": [
    {
      "timestamp": "07-24 19:15",
      "timestamp_full": "2026-07-24T19:15:00Z",
      "agent": "Claude (F51 Darwin session)",
      "operation": "WRITE",
      "operation_type": "write",
      "entry_type": "SAÍDA",
      "summary": "Especificação detalhada de 11 componentes do dashboard web",
      "commit": "39a38ed -- fix(launch): add --blockchain-enabled --senate-enabled",
      "estado_encontrado": "...bullet text...",
      "acoes": "...numbered list...",
      "estado_deixado": "...bullet text...",
      "warnings": "...text...",
      "files_modified": ["path/to/file.py -- desc", "path/to/file2.py -- desc"]
    }
  ],
  "entry_count": 20,
  "parsed_at": "2026-07-24T19:30:00Z"
}
```

Schema per entry:
| Field | Type | Description |
|---|---|---|
| timestamp | string | Short display timestamp "MM-DD HH:MM" |
| timestamp_full | string | ISO 8601 UTC timestamp |
| agent | string | Agent/human name |
| operation | string | "READ" \| "WRITE" \| "EXECUTE" \| "DECIDE" \| "MIXED" |
| operation_type | string | Lowercase for CSS class: "read" \| "write" \| "execute" \| "decide" \| "mixed" |
| entry_type | string | "ENTRADA" \| "SAÍDA" \| "ENTRADA + SAÍDA" |
| summary | string | First meaningful line of actions/objective (max 80 chars) |
| commit | string\|null | "hash -- description" or null if no commit |
| estado_encontrado | string\|null | Full text of ESTADO ENCONTRADO block |
| acoes | string\|null | Full text of AÇÕES block |
| estado_deixado | string\|null | Full text of ESTADO DEIXADO block |
| warnings | string\|null | Full text of WARNINGS block |
| files_modified | array[string]\|null | List of modified file paths |

---

#### `event: training_status`

**Frequency:** sent on change (not periodic)
**Consumers:** ConnectionStatus, HeroLoss, CycleStepChip (state overlays)

```json
{
  "status": "running",
  "changed_at": "2026-07-24T19:30:00Z",
  "since_s": 45.2
}
```

Schema:
| Field | Type | Description |
|---|---|---|
| status | string | "running" \| "stopped" \| "hung" |
| changed_at | string | ISO 8601 when status last changed |
| since_s | number | Seconds since the last data update (for "hung" detection) |

Status determination:
- `"running"` -- dopamine_status.txt updated within last 5s
- `"stopped"` -- training process not running (no Python process with "organism" in cmdline)
- `"hung"` -- training process exists but dopamine_status.txt not updated for >60s

---

### 3.3 Event Routing Matrix

| Event | Frequency | Hero | GPUx2 | MetricRow x7 | Chart x4 | Blockchain | OrganGrid | Integrity x5 | Timeline | Health | ConnStatus |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `hero` | 2s | X | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| `gpu` | 2s | -- | X | -- | -- | -- | -- | -- | -- | -- | -- |
| `training` | 2s | -- | -- | X | -- | -- | -- | -- | -- | -- | -- |
| `chart_point` | 2s | -- | -- | -- | X | -- | -- | -- | -- | -- | -- |
| `blockchain` | 10s | -- | -- | -- | -- | X | -- | -- | -- | -- | -- |
| `health` | 10s | -- | -- | -- | -- | -- | -- | -- | -- | X | -- |
| `organs` | 60s | -- | -- | -- | -- | -- | X | -- | -- | -- | -- |
| `hashes` | 60s | -- | -- | -- | -- | -- | -- | X | -- | -- | -- |
| `timeline` | 60s | -- | -- | -- | -- | -- | -- | -- | X | -- | -- |
| `training_status` | on-change | (overlay) | -- | -- | -- | -- | -- | -- | -- | -- | X |
| `heartbeat` | 15s | -- | -- | -- | -- | -- | -- | -- | -- | -- | X |

---

## 4. REST Endpoints

### 4.1 `GET /dashboard`

**Purpose:** Serve the full HTML page with SSR-seeded initial data.

**Response:** `text/html` -- Jinja2 template rendered with `initial` context dict.

**Context (`initial` dict) shape:**

```python
{
    "hero": {
        "loss": 5.2872,          # float | None
        "cycle": 47,             # int | None
        "step": 14525,           # int | None
        "version": 9,            # int
        "timestamp": "2026-07-24 19:30:00 UTC"  # str
    },
    "gpus": [                    # list, may be empty
        {"idx": "0", "name": "NVIDIA GeForce RTX 5060 Ti",
         "util": 85, "mem_used": 7752, "mem_total": 16311, "temp": 72}
    ],
    "training": {
        "loss": 5.2872,
        "ppl": 197.8,
        "tok_s": 245.3,
        "jepa_raw": 0.1073,
        "dope": 0.159,
        "delta": 0.0221,
        "gmc": 0.0051,
        "loss_delta": -0.0039   # float | None (first tick)
    },
    "blockchain": {
        "blocks": 168,
        "head_hash": "25547e73...",
        "last_cycle": 47,
        "size_kb": 95.9,
        "status": "ACTIVE"
    },
    "organs": {                  # dict of organ_name -> info
        "core": {"active": True, "identity_hash": "5bf0be80..."},
        # ... 17 entries
    },
    "hashes": {                  # dict of label -> hash_short str
        "checkpoint": "d41d8cd9...",
        "config": "3a7f2e1b...",
        "ledger": "--",
        "blocks": "e99a18c4...",
        "script": "a1b2c3d4..."
    },
    "system": {
        "ram_used_gb": 18.2,
        "ram_total_gb": 32.0,
        "disk_used_gb": 312.5,
        "disk_total_gb": 953.0,
        "process_count": 2,
        "uptime": "4h 32m"
    },
    "timeline": [                # list of entries, newest first, max 20
        {
            "timestamp": "07-24 19:15",
            "agent": "Claude (F51 Darwin session)",
            "operation": "WRITE",
            "operation_type": "write",
            "entry_type": "SAÍDA",
            "summary": "Especificação detalhada de 11 componentes..."
        }
    ],
    "render_time_utc": "2026-07-24T19:30:00.000Z"
}
```

**SSR strategy:**
1. Call each collector function once (synchronously during startup).
2. Render template with results.
3. After response is sent, start the asyncio poll loop and SSE connections.
4. Client-side EventSource connects after DOM ready. First SSE `chart_point` event carries `action: "seed"` flag (see chart_point schema extension below).

### 4.2 `GET /dashboard/sse/stream`

**Purpose:** Server-Sent Events stream (section 3).

**Response:** `text/event-stream`

### 4.3 `GET /api/metrics/history`

**Purpose:** Seed Chart.js deques with historical data from SQLite on page load.

**Query params:**

| Param | Type | Default | Description |
|---|---|---|---|
| metric | string | (required) | One of: "loss", "ppl", "tok_s", "dope" |
| from | string | "1h ago" | ISO 8601 UTC or relative ("4h") |
| to | string | "now" | ISO 8601 UTC or "now" |
| limit | integer | 1000 | Max data points returned |

**Response:** `application/json`

```json
{
  "metric": "loss",
  "points": [
    {"ts": 1784902200.0, "value": 5.3012},
    {"ts": 1784902202.0, "value": 5.2987}
  ],
  "count": 2,
  "from": "2026-07-24T17:30:00Z",
  "to": "2026-07-24T19:30:00Z"
}
```

Schema:
| Field | Type | Description |
|---|---|---|
| metric | string | The requested metric name (echo) |
| points | array | Array of {ts: float (unix_s), value: float} |
| count | integer | Number of points returned |
| from | string | Resolved start time (ISO 8601) |
| to | string | Resolved end time (ISO 8601) |

**SQL query (template):**
```sql
SELECT timestamp_unix_s, value
FROM metrics
WHERE metric_key = ?
  AND timestamp_unix_s >= ?
  AND timestamp_unix_s <= ?
ORDER BY timestamp_unix_s ASC
LIMIT ?
```

### 4.4 `GET /api/status`

**Purpose:** Lightweight JSON endpoint for external monitoring (no HTML).

**Response:** `application/json`

```json
{
  "dashboard": {
    "version": "2.0.0",
    "uptime_s": 342.1,
    "connected_clients": 2,
    "last_poll_utc": "2026-07-24T19:30:00Z",
    "polls_total": 10245,
    "polls_failed": 3
  },
  "training": {
    "status": "stopped",
    "last_data_utc": "2026-07-24T08:44:00Z",
    "seconds_since_data": 38760,
    "cycle": 46,
    "step": 14500,
    "loss": null
  },
  "system": {
    "ram_pct": 56.9,
    "disk_pct": 32.8,
    "python_processes": 0
  }
}
```

### 4.5 `GET /api/runs`

**Purpose:** Training run history from `workspace/runtime/dashboard/runs.jsonl`.

**Query params:**

| Param | Type | Default | Description |
|---|---|---|---|
| limit | integer | 20 | Max runs returned |

**Response:** `application/json`

```json
{
  "runs": [
    {
      "run_id": "run-20260724-084400",
      "started_at": "2026-07-24T08:44:00Z",
      "stopped_at": "2026-07-24T11:05:00Z",
      "start_cycle": 45,
      "start_step": 14300,
      "end_cycle": 46,
      "end_step": 14500,
      "total_steps": 200,
      "exit_reason": "manual_stop"
    }
  ],
  "count": 1
}
```

Schema per run:
| Field | Type | Description |
|---|---|---|
| run_id | string | Unique run identifier |
| started_at | string | ISO 8601 UTC start time |
| stopped_at | string\|null | ISO 8601 UTC stop time (null if still running) |
| start_cycle | integer | Cycle number at run start |
| start_step | integer | Global step at run start |
| end_cycle | integer\|null | Cycle at run end (null if still running) |
| end_step | integer\|null | Step at run end (null if still running) |
| total_steps | integer\|null | Total steps in this run (null if still running) |
| exit_reason | string\|null | "manual_stop" \| "crash" \| "oom" \| "checkpoint_save_failed" \| null |

### 4.6 `GET /api/sections/{name}`

**Purpose:** htmx-driven partial updates for SSR-fallback sections.
These endpoints return HTML snippets rendered from Jinja2 partial templates.
Used when SSE is disconnected, or to populate sections that are NOT SSE-driven.

**Valid names:** `hero`, `gpu`, `training`, `blockchain`, `hashes`, `organs`, `timeline`, `health`

**Response per section:**

- `/api/sections/hero` -- HTML fragment for `#hero-section` (replaces innerHTML)
- `/api/sections/gpu` -- HTML fragment with both GPU cards (replaces innerHTML of both `#gpu0-section` and `#gpu1-section`)
- etc.

Each handler calls the corresponding collector function, renders a Jinja2 snippet
(`templates/sections/<name>.html` or inline Jinja2 string), and returns `text/html`.

### 4.7 Static Files

```
/static/htmx.min.js    → src/tools/static/htmx.min.js     (14KB)
/static/chart.min.js   → src/tools/static/chart.min.js    (65KB)
/static/dashboard.css  → src/tools/static/dashboard.css   (external CSS, non-critical)
```

Served by `StaticFiles` mount on FastAPI app.
All three files committed to repo. Zero CDN dependencies.

---

## 5. SQLite Schema

### 5.1 PRAGMA Settings

```sql
-- Applied on every connection open
PRAGMA journal_mode = WAL;           -- Write-Ahead Logging for concurrent reads
PRAGMA synchronous = NORMAL;         -- Safe with WAL, better performance
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;          -- 5s busy timeout
PRAGMA cache_size = -8000;           -- 8 MB page cache
PRAGMA temp_store = MEMORY;
PRAGMA mmap_size = 268435456;        -- 256 MB memory-mapped I/O
PRAGMA auto_vacuum = INCREMENTAL;    -- reclaim space on vacuum
```

### 5.2 Table: `metrics`

Stores one row per metric per poll cycle. This is the primary table for chart history.

```sql
CREATE TABLE IF NOT EXISTS metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_key      TEXT    NOT NULL,               -- "loss", "ppl", "tok_s", "dope", "jepa_raw", "delta", "gmc"
    value           REAL,                           -- NULL if metric was unavailable this cycle
    timestamp_unix_s REAL  NOT NULL,                -- Unix timestamp in seconds (float)
    cycle           INTEGER,                        -- Current cycle at time of reading
    step            INTEGER,                        -- Current step at time of reading
    source          TEXT    NOT NULL DEFAULT 'poll', -- "poll" | "backfill" | "manual"
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_metrics_key_ts
    ON metrics(metric_key, timestamp_unix_s);

CREATE INDEX IF NOT EXISTS idx_metrics_ts
    ON metrics(timestamp_unix_s);

CREATE INDEX IF NOT EXISTS idx_metrics_cycle_step
    ON metrics(cycle, step);
```

**Insert pattern (batched, per poll cycle):**

```sql
INSERT INTO metrics (metric_key, value, timestamp_unix_s, cycle, step, source)
VALUES
    ('loss',   5.2872, 1784905800.0, 47, 14525, 'poll'),
    ('ppl',    197.8,  1784905800.0, 47, 14525, 'poll'),
    ('tok_s',  245.3,  1784905800.0, 47, 14525, 'poll'),
    ('dope',   0.159,  1784905800.0, 47, 14525, 'poll'),
    ('jepa_raw',0.1073,1784905800.0, 47, 14525, 'poll'),
    ('delta',  0.0221, 1784905800.0, 47, 14525, 'poll'),
    ('gmc',    0.0051, 1784905800.0, 47, 14525, 'poll');
```

**Retention:** Keep all rows. Table is append-only. On startup, if row count > 1,000,000,
emit info log "metrics table has N rows (X MB)" but do NOT truncate.
Manual vacuum available via signal or admin endpoint.

### 5.3 Table: `gpu_snapshots`

Stores one row per GPU per poll cycle.

```sql
CREATE TABLE IF NOT EXISTS gpu_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    gpu_index       INTEGER NOT NULL,               -- 0, 1, ...
    timestamp_unix_s REAL   NOT NULL,
    util_pct        INTEGER,                        -- 0-100
    mem_used_mib    INTEGER,
    mem_total_mib   INTEGER,
    temp_c          INTEGER,                        -- Celsius, nullable
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_gpu_ts
    ON gpu_snapshots(timestamp_unix_s);

CREATE INDEX IF NOT EXISTS idx_gpu_index_ts
    ON gpu_snapshots(gpu_index, timestamp_unix_s);
```

**Insert pattern (batched, per poll cycle):**

```sql
INSERT INTO gpu_snapshots (gpu_index, timestamp_unix_s, util_pct, mem_used_mib, mem_total_mib, temp_c)
VALUES (0, 1784905800.0, 85, 7752, 16311, 72),
       (1, 1784905800.0, 0,  3,    12288, 45);
```

**Retention:** Keep last 86,400 snapshots per GPU (24 hours at 2s). Prune older on startup.
Pruning query:
```sql
DELETE FROM gpu_snapshots
WHERE timestamp_unix_s < (
    SELECT MAX(timestamp_unix_s) - 86400 FROM gpu_snapshots
);
```

### 5.4 Table: `checkpoint_history`

Stores one row per checkpoint save detected.

```sql
CREATE TABLE IF NOT EXISTS checkpoint_history (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle               INTEGER NOT NULL,
    step                INTEGER NOT NULL,
    filename            TEXT    NOT NULL,            -- e.g., "organism_cycle_046_step_014500.pt"
    size_bytes          INTEGER,
    base_checkpoint_id  TEXT,
    sha256              TEXT,                        -- NULL until computed (deferred, slow)
    saved_at            TEXT    NOT NULL,            -- ISO 8601 from organism_latest.json
    detected_at_unix_s  REAL   NOT NULL,             -- When dashboard noticed this checkpoint
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_cp_cycle_step
    ON checkpoint_history(cycle, step);
```

**Insert:** On detecting `organism_latest.json` mtime change or cycle/step increment.
UNIQUE constraint prevents duplicates if dashboard restarts and re-scans.

**Retention:** Keep all rows. Table is small (one row per 100 steps).

### 5.5 Table: `dashboard_events`

Internal operational log for dashboard itself.

```sql
CREATE TABLE IF NOT EXISTS dashboard_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type      TEXT    NOT NULL,                -- "startup", "shutdown", "poll_error", "sse_connect", "sse_disconnect", "logbook_write"
    payload         TEXT,                            -- JSON string, free-form
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_devents_type_ts
    ON dashboard_events(event_type, created_at);
```

---

## 6. Data Collector Module

### 6.1 Architecture

```python
# Pseudocode structure of darwin_dashboard_web.py

import asyncio
import json
import os
import time
import hashlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Configuration ──
@dataclass
class DashboardConfig:
    ckpt_root: Path
    blockchain_file: Path
    ledger_file: Path
    dopamine_file: Path
    pointer_file: Path
    config_yaml: Path
    logbook_path: Path
    db_path: Path
    poll_interval_s: float = 2.0
    fast_queue_max: int = 7200
    slow_queue_max: int = 7200

# ── Ring Buffers ──
@dataclass
class ChartBuffer:
    """Ring buffer for one chart metric."""
    metric: str
    maxlen: int = 7200
    timestamps: deque = field(default_factory=lambda: deque(maxlen=7200))
    values: deque = field(default_factory=lambda: deque(maxlen=7200))

    def push(self, ts: float, value: float) -> None:
        if value is None or not isfinite(value):
            return
        self.timestamps.append(ts)
        self.values.append(value)

    def last_n(self, n: int) -> list[dict]:
        """Return last N points as [{ts, value}]."""
        n = min(n, len(self.timestamps))
        return [{"ts": self.timestamps[i], "value": self.values[i]}
                for i in range(-n, 0)]

    def last_value(self) -> Optional[float]:
        return self.values[-1] if self.values else None

# ── File Watcher ──
@dataclass
class FileState:
    path: Path
    last_mtime: float = 0.0
    last_size: int = 0
    last_inode: int = 0  # for detecting rotation (new file = new inode)

    def changed(self) -> bool:
        """Check mtime, size, or inode change. Returns True if file was modified."""
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            # File was deleted (e.g., rotation)
            self.last_mtime = 0.0
            self.last_size = 0
            self.last_inode = 0
            return False  # no data to read
        changed = (
            stat.st_mtime != self.last_mtime or
            stat.st_size != self.last_size or
            stat.st_ino != self.last_inode  # Windows: st_ino may be 0, skip
        )
        self.last_mtime = stat.st_mtime
        self.last_size = stat.st_size
        # On Windows, st_ino may not be available; handle gracefully
        try:
            self.last_inode = stat.st_ino
        except AttributeError:
            pass
        return changed

# ── Main Poller ──
class DataCollector:
    def __init__(self, config: DashboardConfig):
        self.cfg = config
        self.state = {
            "dopamine": FileState(config.dopamine_file),
            "pointer": FileState(config.pointer_file),
            "blockchain": FileState(config.blockchain_file),
            "ledger": FileState(config.ledger_file),
            "logbook": FileState(config.logbook_path),
        }
        self.buffers = {
            "loss": ChartBuffer("loss"),
            "ppl": ChartBuffer("ppl"),
            "tok_s": ChartBuffer("tok_s"),
            "dope": ChartBuffer("dope"),
        }
        self.previous_metrics: dict = {}
        self.cached_hashes: dict = {}
        self.hash_cache_ttl: float = 55.0  # seconds (just under 60s poll)
        self.last_hash_compute: float = 0.0
        self.training_status: str = "stopped"
        self.training_status_changed_at: Optional[float] = None
        self.tick_count: int = 0

    async def poll_cycle(self) -> dict:
        """
        Single poll cycle. Called every 2s.
        Returns a dict of events to dispatch, keyed by SSE event type.
        """
        events = {}
        now = time.time()
        self.tick_count += 1

        # ── Fast tier (every poll) ──
        dopamine = self._read_dopamine()
        gpu_data = await self._read_gpu()  # subprocess, so async
        pointer = self._read_pointer()     # only if changed

        # Build 'hero' event
        events["hero"] = self._build_hero_event(dopamine, pointer)

        # Build 'training' event
        events["training"] = self._build_training_event(dopamine)

        # Build 'chart_point' event
        chart_point = self._build_chart_point(dopamine, now)
        if chart_point:
            events["chart_point"] = chart_point

        # Build 'gpu' event
        events["gpu"] = gpu_data

        # ── Medium tier (every 5th poll ~10s) ──
        if self.tick_count % 5 == 0:
            events["blockchain"] = self._read_blockchain()
            events["health"] = await self._read_system_health()

        # ── Slow tier (every 30th poll ~60s) ──
        if self.tick_count % 30 == 0:
            events["organs"] = await self._read_organ_states()
            events["hashes"] = await self._compute_hashes()
            events["timeline"] = self._read_timeline()

        # ── training_status: only on change ──
        new_status = self._detect_training_status(dopamine)
        if new_status != self.training_status:
            self.training_status = new_status
            self.training_status_changed_at = now
            events["training_status"] = {
                "status": new_status,
                "changed_at": _iso8601(now),
                "since_s": 0.0
            }

        # ── Persist to SQLite ──
        await self._write_metrics_to_db(dopamine, gpu_data)

        return events

    # ── Individual readers ──

    def _read_dopamine(self) -> dict:
        """Parse dopamine_status.txt into dict. Returns {} on any error."""
        path = self.cfg.dopamine_file
        if not path.exists():
            return {}
        try:
            # Read size first to detect partial write
            stat = path.stat()
            text = path.read_text(encoding="utf-8").strip()
            if len(text) != stat.st_size:
                # File was being written during our read. Discard.
                return {}
            result = {}
            for part in text.split():
                if "=" in part:
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

    def _read_pointer(self) -> dict:
        """Parse organism_latest.json."""
        path = self.cfg.pointer_file
        if not self.state["pointer"].changed():
            return {}  # no update since last read
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    async def _read_gpu(self) -> dict:
        """nvidia-smi via asyncio subprocess."""
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
            for line in stdout.decode().strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                gpus.append({
                    "idx": parts[0],
                    "name": parts[1],
                    "util": int(parts[2]) if parts[2] != "[Not Supported]" else 0,
                    "mem_used": int(parts[3]),
                    "mem_total": int(parts[4]),
                    "temp": int(parts[5]) if parts[5] != "[Not Supported]" else None,
                })
            return {"gpus": gpus, "timestamp": _iso860j()}
        except Exception:
            return {"gpus": [], "timestamp": _iso860j()}

    def _read_blockchain(self) -> dict:
        """Read blocks.jsonl: count lines, parse last block."""
        path = self.cfg.blockchain_file
        result = {"blocks": 0, "head_hash": "--", "head_hash_short": "--",
                   "last_cycle": None, "last_step": None,
                   "size_kb": 0.0, "status": "NO FILE",
                   "timestamp": _iso860j()}
        if not path.exists():
            return result
        try:
            stat = path.stat()
            result["size_kb"] = round(stat.st_size / 1024, 1)
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                result["status"] = "EMPTY"
                return result
            lines = text.split("\n")
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

    async def _read_system_health(self) -> dict:
        """psutil: RAM, disk, process count."""
        try:
            import psutil
            ram = psutil.virtual_memory()
            disk = psutil.disk_usage(str(self.cfg.ckpt_root))
            procs = [p for p in psutil.process_iter(["name", "memory_info"])
                     if p.info["name"] and "python" in p.info["name"].lower()]
            total_mem = sum(
                (p.info["memory_info"].rss if p.info["memory_info"] else 0)
                for p in procs
            ) / (1024**3)
            return {
                "ram_used_gb": round(ram.used / (1024**3), 1),
                "ram_total_gb": round(ram.total / (1024**3), 0),
                "ram_pct": round(ram.percent, 1),
                "disk_used_gb": round(disk.used / (1024**3), 1),
                "disk_total_gb": round(disk.total / (1024**3), 0),
                "disk_pct": round(disk.percent, 1),
                "disk_free_gb": round(disk.free / (1024**3), 1),
                "process_count": len(procs),
                "process_total_mem_gb": round(total_mem, 1),
                "uptime": _human_duration(self._training_uptime_s()),
                "uptime_s": self._training_uptime_s(),
                "dashboard_uptime_s": round(self._dashboard_uptime_s(), 1),
                "timestamp": _iso860j(),
            }
        except ImportError:
            return {
                "ram_used_gb": None, "ram_total_gb": None, "ram_pct": None,
                "disk_used_gb": None, "disk_total_gb": None, "disk_pct": None,
                "disk_free_gb": None, "process_count": None,
                "process_total_mem_gb": None, "uptime": None, "uptime_s": None,
                "dashboard_uptime_s": round(self._dashboard_uptime_s(), 1),
                "timestamp": _iso860j(),
            }

    async def _read_organ_states(self) -> dict:
        """Read organ identities from checkpoint and config to determine statuses."""
        # This reads the checkpoint .pt file (expensive, only every 60s)
        # Uses asyncio.to_thread() to avoid blocking
        ...
```

### 6.2 Poll Loop (main coroutine)

```python
async def poll_loop(collector: DataCollector, sse_manager: SSEManager, db: DatabaseManager):
    """Main asyncio loop. Runs every 2s for the lifetime of the dashboard."""
    while True:
        loop_start = time.monotonic()
        try:
            events = await collector.poll_cycle()
            # Dispatch to SSE clients
            for event_type, payload in events.items():
                await sse_manager.broadcast(event_type, payload)
        except Exception as e:
            logger.error(f"Poll cycle error: {e}")
            # Don't crash the loop — one bad poll shouldn't kill the dashboard
            await db.log_dashboard_event("poll_error", str(e))

        # Sleep for the remainder of the poll interval
        elapsed = time.monotonic() - loop_start
        await asyncio.sleep(max(0.0, collector.cfg.poll_interval_s - elapsed))
```

### 6.3 SSEManager

```python
class SSEManager:
    """Manages connected SSE clients and broadcasts events."""

    def __init__(self):
        self._clients: list[asyncio.Queue] = []
        self._lock = asyncio.Lock()
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def add_client(self) -> asyncio.Queue:
        """Register a new SSE client. Returns its personal asyncio.Queue."""
        queue = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._clients.append(queue)
        return queue

    async def remove_client(self, queue: asyncio.Queue) -> None:
        """Remove a disconnected client."""
        async with self._lock:
            if queue in self._clients:
                self._clients.remove(queue)

    async def broadcast(self, event_type: str, data: dict) -> None:
        """Push an SSE event to all connected clients."""
        payload = f"event: {event_type}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
        async with self._lock:
            dead = []
            for q in self._clients:
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    dead.append(q)  # Slow client — drop it
            for q in dead:
                self._clients.remove(q)

    async def start_heartbeat(self, interval: float = 15.0) -> None:
        """Send periodic heartbeat events to keep connections alive."""
        while True:
            await asyncio.sleep(interval)
            await self.broadcast("heartbeat", {
                "server_time_utc": _iso860j(),
                "dashboard_uptime_s": time.monotonic() - self._start_time,
                "client_count": len(self._clients),
            })
```

### 6.4 SSE Endpoint (FastAPI route)

```python
from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse
# OR manual implementation:

@router.get("/dashboard/sse/stream")
async def sse_stream(request: Request):
    """SSE endpoint. Each client gets its own queue."""
    queue = await sse_manager.add_client()

    async def event_generator():
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield payload
                except asyncio.TimeoutError:
                    # Send heartbeat comment to keep connection alive
                    yield ": heartbeat\n\n"
        finally:
            await sse_manager.remove_client(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
```

---

## 7. Logbook Parser API

### 7.1 Module: `LogbookParser`

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import re
import json
import time
import msvcrt  # Windows file locking


@dataclass
class EstadoAtual:
    """Parsed ESTADO ATUAL table."""
    raw: dict = field(default_factory=dict)  # All key=value pairs from the table
    data_hora: Optional[str] = None
    branch: Optional[str] = None
    ultimo_commit: Optional[str] = None          # "hash — description"
    checkpoint_ativo: Optional[str] = None       # "organism_cycle_XXX_step_YYYYYY.pt"
    checkpoint_version: Optional[int] = None
    cycle: Optional[int] = None
    step: Optional[int] = None
    loss: Optional[float] = None
    tok_s: Optional[str] = None
    orgaos_ativos: Optional[int] = None
    experts_ativos: Optional[str] = None
    barramento_causal: Optional[str] = None
    blockchain: Optional[str] = None
    treino_rodando: Optional[bool] = None
    dashboard_web: Optional[str] = None


@dataclass
class AgentEntry:
    """Parsed agent entry/exit block."""
    timestamp: str                   # "07-24 19:15" or "2026-07-24 19:15"
    timestamp_full: str              # ISO 8601
    agent: str                       # "Claude (F51 Darwin session)"
    operation: str                   # "READ" | "WRITE" | "EXECUTE" | "DECIDE" | "MIXED"
    operation_type: str              # lowercase, for CSS
    entry_type: str                  # "ENTRADA" | "SAÍDA" | "ENTRADA + SAÍDA"
    summary: str                     # First meaningful line (max 80 chars)
    estado_encontrado: Optional[str] = None
    acoes: Optional[str] = None
    estado_deixado: Optional[str] = None
    commit: Optional[str] = None     # "hash — message"
    warnings: Optional[str] = None
    files_modified: list[str] = field(default_factory=list)


@dataclass
class KnownBug:
    """Parsed bug from Bugs conhecidos section."""
    id: int
    title: str
    status: str = "known"           # "known" | "fixed" | "mitigated"
    description: Optional[str] = None


@dataclass
class ParsedLogbook:
    """Complete parsed state of DIARIO_DE_BORDO.md."""
    estado_atual: Optional[EstadoAtual] = None
    entries: list[AgentEntry] = field(default_factory=list)
    bugs: list[KnownBug] = field(default_factory=list)
    parsed_at: str = ""
    parse_errors: list[str] = field(default_factory=list)


class LogbookParser:
    """
    Regex-based parser for DIARIO_DE_BORDO.md.

    Parsing is tolerant: malformed entries are skipped with a parse_error logged,
    rather than crashing the entire parse.
    """

    # Regex for ESTADO ATUAL table rows: | **Key** | Value |
    _TABLE_ROW_RE = re.compile(
        r'\|\s*\*\*(.+?)\*\*\s*\|\s*(.+?)\s*\|'
    )

    # Regex for agent entry header: ### YYYY-MM-DD HH:MM UTC | [Name]
    _ENTRY_HEADER_RE = re.compile(
        r'###\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\s+UTC\s*\|\s*(.+?)(?:\s*\(([^)]*)\))?\s*$'
    )

    # Regex for entry footer: the closing ``` of the code block
    _ENTRY_FOOTER_RE = re.compile(r'^```$')

    # Regex for OPERATION line
    _OP_RE = re.compile(r'OPERAÇÃO\s*(?:PRETENDIDA|REALIZADA)?:\s*(READ|WRITE|EXECUTE|DECIDE|MIXED)', re.IGNORECASE)

    # Regex for COMMIT line
    _COMMIT_RE = re.compile(r'COMMIT:\s*([a-f0-9]+)\s*[-—]\s*(.+)', re.IGNORECASE)

    def __init__(self, logbook_path: Path, cache_path: Path):
        self.path = logbook_path
        self.cache_path = cache_path

    def parse(self) -> ParsedLogbook:
        """
        Parse the entire DIARIO_DE_BORDO.md.
        Uses cache if mtime unchanged.
        Returns ParsedLogbook with all sections.
        """
        # Check cache
        if self._cache_valid():
            return self._load_cache()

        # Parse fresh
        text = self._read_with_retry()
        result = ParsedLogbook(parsed_at=_iso860j())

        result.estado_atual = self._parse_estado_atual(text)
        result.entries = self._parse_agent_entries(text)
        result.bugs = self._parse_bugs(text)

        # Write cache
        self._write_cache(result)
        return result

    def _parse_estado_atual(self, text: str) -> Optional[EstadoAtual]:
        """Parse the ESTADO ATUAL markdown table."""
        # Locate the "## ESTADO ATUAL" section
        section_start = text.find("## 📍 ESTADO ATUAL")
        if section_start == -1:
            return None
        # Find the table (starts with "| Campo | Valor |")
        table_start = text.find("| Campo", section_start)
        if table_start == -1:
            return None
        table_end = text.find("\n\n", table_start)
        if table_end == -1:
            table_end = len(text)
        table_text = text[table_start:table_end]

        raw = {}
        for match in self._TABLE_ROW_RE.finditer(table_text):
            key = match.group(1).strip()
            value = match.group(2).strip()
            raw[key] = value

        ea = EstadoAtual(raw=raw)
        # Map known keys
        ea.data_hora = raw.get("Data/Hora")
        ea.branch = raw.get("Branch")
        ea.ultimo_commit = raw.get("Último commit")
        ea.checkpoint_ativo = raw.get("Checkpoint ativo")
        ea.checkpoint_version = self._parse_int(raw.get("Checkpoint version"))
        ea.cycle = self._parse_int(raw.get("Cycle"))
        ea.step = self._parse_int(raw.get("Step"))
        ea.loss = self._try_parse_float(raw.get("Loss (lm)", "").lstrip("~"))
        ea.tok_s = raw.get("tok/s")
        ea.orgaos_ativos = self._parse_int(raw.get("Órgãos ativos"))
        ea.experts_ativos = raw.get("Experts ativos")
        ea.barramento_causal = raw.get("Barramento causal")
        ea.blockchain = raw.get("Blockchain")
        treino = raw.get("Treino rodando?", "")
        ea.treino_rodando = True if "SIM" in treino.upper() or "✅" in treino else (
            False if "PARADO" in treino.upper() or "❌" in treino else None
        )
        ea.dashboard_web = raw.get("Dashboard web")
        return ea

    def _parse_agent_entries(self, text: str) -> list[AgentEntry]:
        """
        Parse all agent entry/exit blocks.
        Each block starts with "### YYYY-MM-DD HH:MM UTC | Name"
        and contains a code-fenced block with structured fields.
        """
        entries = []
        # Find all entry headers
        for header_match in self._ENTRY_HEADER_RE.finditer(text):
            date_str = header_match.group(1)
            time_str = header_match.group(2)
            agent_name = header_match.group(3).strip()

            # Find the subsequent code block
            block_start = text.find("```", header_match.end())
            if block_start == -1:
                continue
            block_end = text.find("```", block_start + 3)
            if block_end == -1:
                continue
            block_content = text[block_start + 3:block_end].strip()

            entry = self._parse_entry_block(
                block_content, date_str, time_str, agent_name
            )
            if entry:
                entries.append(entry)

        return entries

    def _parse_entry_block(self, block: str, date_str: str, time_str: str,
                           agent_name: str) -> Optional[AgentEntry]:
        """Parse a single agent entry code block."""
        entry = AgentEntry(
            timestamp=f"{date_str[5:]} {time_str}",  # "MM-DD HH:MM"
            timestamp_full=f"{date_str}T{time_str}:00Z",
            agent=agent_name,
            operation="READ",     # default
            operation_type="read",
            entry_type="ENTRADA",  # default
            summary="",
        )

        # Detect entry type
        if "ENTRADA + SAÍDA" in block or "ENTRADA" in block and "SAÍDA" in block:
            entry.entry_type = "ENTRADA + SAÍDA"
        elif "ENTRADA" in block.split("\n")[0] if block else False:
            entry.entry_type = "ENTRADA"
        elif "SAÍDA" in block.split("\n")[0] if block else False:
            entry.entry_type = "SAÍDA"

        # Parse OPERATION
        op_match = self._OP_RE.search(block)
        if op_match:
            entry.operation = op_match.group(1).upper()
            entry.operation_type = entry.operation.lower()

        # Parse summary: first non-empty line after header that isn't a field marker
        lines = block.strip().split("\n")
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("OPER") or stripped.startswith("ESTADO") or \
               stripped.startswith("AÇÕES") or stripped.startswith("OBJETIVO") or \
               stripped.startswith("COMMIT") or stripped.startswith("WARNINGS") or \
               stripped.startswith("ARQUIVOS"):
                continue
            if len(stripped) > 5:
                entry.summary = stripped[:80]
                break

        if not entry.summary:
            # Fallback: use the first numbered action or objective line
            obj_match = re.search(r'OBJETIVO:\s*(.+)', block)
            if obj_match:
                entry.summary = obj_match.group(1).strip()[:80]

        # Parse COMMIT
        commit_match = self._COMMIT_RE.search(block)
        if commit_match:
            entry.commit = f"{commit_match.group(1)} — {commit_match.group(2).strip()}"

        # Extract section blocks
        entry.estado_encontrado = self._extract_section(block, "ESTADO ENCONTRADO")
        # For ESTADO DEIXADO, handle both "ESTADO DEIXADO:" and "ESTADO DEIXADO\n"
        entry.estado_deixado = self._extract_section(block, "ESTADO DEIXADO")
        entry.acoes = self._extract_section(block, "AÇÕES")
        entry.warnings = self._extract_section(block, "WARNINGS")

        # Parse files modified
        entry.files_modified = self._extract_file_list(block)

        return entry

    def _parse_bugs(self, text: str) -> list[KnownBug]:
        """Parse the Bugs conhecidos numbered list."""
        bugs = []
        section = text.find("## 🔧 Bugs conhecidos")
        if section == -1:
            return bugs
        # Find the numbered list start
        list_start = text.find("1.", section)
        if list_start == -1:
            return bugs
        list_end = text.find("\n\n", list_start)
        if list_end == -1:
            list_end = len(text)
        list_text = text[list_start:list_end]

        for match in re.finditer(r'(\d+)\.\s+\*\*(.+?)\*\*(?:\s*:\s*(.+))?', list_text):
            bug_id = int(match.group(1))
            title = match.group(2).strip()
            description = match.group(3).strip() if match.group(3) else None
            bugs.append(KnownBug(id=bug_id, title=title, description=description))
        return bugs

    def _cache_valid(self) -> bool:
        """Check if cache is fresh (newer than source file mtime)."""
        if not self.cache_path.exists():
            return False
        try:
            src_mtime = self.path.stat().st_mtime
            cache_mtime = self.cache_path.stat().st_mtime
            return cache_mtime >= src_mtime
        except Exception:
            return False

    def _load_cache(self) -> ParsedLogbook:
        """Load parsed logbook from JSON cache."""
        data = json.loads(self.cache_path.read_text())
        result = ParsedLogbook()
        if data.get("estado_atual"):
            result.estado_atual = EstadoAtual(**data["estado_atual"])
        result.entries = [AgentEntry(**e) for e in data.get("entries", [])]
        result.bugs = [KnownBug(**b) for b in data.get("bugs", [])]
        result.parsed_at = data.get("parsed_at", "")
        return result

    def _write_cache(self, result: ParsedLogbook) -> None:
        """Write parsed result to JSON cache."""
        cache_dir = self.cache_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "estado_atual": result.estado_atual.__dict__ if result.estado_atual else None,
            "entries": [e.__dict__ for e in result.entries],
            "bugs": [b.__dict__ for b in result.bugs],
            "parsed_at": result.parsed_at,
        }
        self.cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _read_with_retry(self) -> str:
        """Read the file with retry on lock contention."""
        for attempt in range(3):
            try:
                return self.path.read_text(encoding="utf-8")
            except PermissionError:
                time.sleep(0.1 * (attempt + 1))
        raise

    @staticmethod
    def _parse_int(value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        # Remove thousand separators and non-digit chars
        cleaned = re.sub(r'[^\d]', '', str(value))
        try:
            return int(cleaned) if cleaned else None
        except ValueError:
            return None

    @staticmethod
    def _try_parse_float(value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _extract_section(block: str, section_name: str) -> Optional[str]:
        """Extract a named section from the block."""
        # Match "SECTION_NAME:" or "SECTION_NAME\n"
        pattern = rf'{section_name}\s*:?\s*\n(.*?)(?=\n\S|\Z)'
        match = re.search(pattern, block, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _extract_file_list(block: str) -> list[str]:
        """Extract modified files from ARQUIVOS MODIFICADOS section."""
        section = LogbookParser._extract_section(block, "ARQUIVOS MODIFICADOS")
        if not section:
            return []
        files = []
        for line in section.split("\n"):
            line = line.strip().lstrip("- ").strip()
            if line and not line.startswith("("):
                files.append(line)
        return files


# ── Dashboard write function ──

def append_logbook_event(
    logbook_path: Path,
    event_type: str,  # "training_start", "training_stop", "training_snapshot"
    payload: str,
    timeout_s: float = 2.0,
) -> bool:
    """
    Append a dashboard event to DIARIO_DE_BORDO.md.

    Uses msvcrt.locking on Windows for advisory file lock.
    Returns True on success, False on lock timeout.

    Thread-safe: dashboard poll loop calls this, but rarely (start/stop/100-step snapshot).
    """
    import msvcrt

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
            # Acquire lock
            start = time.monotonic()
            while True:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() - start > timeout_s:
                        return False
                    time.sleep(0.05)  # 50ms backoff
            try:
                f.write(entry)
                f.flush()
                os.fsync(f.fileno())
            finally:
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        return True
    except Exception:
        return False
```

---

## 8. In-Memory Ring Buffers

### 8.1 Server-Side: ChartPoint Deques

Used to answer `GET /api/metrics/history` without hitting SQLite for the most
recent data, and as a write-through cache.

```python
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class RingBuffer:
    """Fixed-capacity ring buffer for one metric time series."""
    metric_key: str
    capacity: int = 7200
    timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=7200))
    values: deque[float] = field(default_factory=lambda: deque(maxlen=7200))

    def push(self, ts: float, value: Optional[float]) -> None:
        """Append a data point. Silently drops NaN/Inf/None."""
        if value is None:
            return
        if not math.isfinite(value):
            return
        self.timestamps.append(ts)
        self.values.append(value)

    def last_value(self) -> Optional[float]:
        """Most recent value, or None if empty."""
        return self.values[-1] if self.values else None

    def last_n(self, n: int) -> list[dict]:
        """Return last N points as [{ts, value}], newest last."""
        n = min(n, len(self.timestamps))
        items = []
        # deque doesn't support slice, iterate manually
        for i in range(len(self.timestamps) - n, len(self.timestamps)):
            items.append({"ts": self.timestamps[i], "value": self.values[i]})
        return items

    def range(self, from_ts: float, to_ts: float) -> list[dict]:
        """Return all points in [from_ts, to_ts]."""
        items = []
        for i in range(len(self.timestamps)):
            ts = self.timestamps[i]
            if from_ts <= ts <= to_ts:
                items.append({"ts": ts, "value": self.values[i]})
        return items

    @property
    def count(self) -> int:
        return len(self.timestamps)

    @property
    def span_s(self) -> float:
        """Time span from first to last point in seconds."""
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]
```

### 8.2 Client-Side: Chart Deques

The browser maintains its own ring buffers for Chart.js rendering. These are
backed by JavaScript arrays with manual trimming.

```javascript
// Client-side ring buffer (JavaScript)
const MAX_POINTS = 7200;

const chartBuffers = {
  loss:     { timestamps: [], values: [] },
  ppl:      { timestamps: [], values: [] },
  tok_s:    { timestamps: [], values: [] },
  dopamine: { timestamps: [], values: [] },
};

function pushToBuffer(metric, ts, value) {
  if (value === null || value === undefined || !isFinite(value)) return;
  const buf = chartBuffers[metric];
  if (!buf) return;
  buf.timestamps.push(ts);
  buf.values.push(value);
  while (buf.timestamps.length > MAX_POINTS) {
    buf.timestamps.shift();
    buf.values.shift();
  }
}
```

Chart.js update:
```javascript
function pushChartPoint(chart, datasetIndex, ts, value) {
  const ds = chart.data.datasets[datasetIndex];
  if (!ds) return;
  ds.data.push({x: ts, y: value});
  while (ds.data.length > MAX_POINTS) {
    ds.data.shift();
  }
  chart.update('none');  // skip layout/scale, only redraw data
}
```

Client buffers are seeded on page load from `GET /api/metrics/history?metric=X&limit=1000`.
After seeding, the EventSource `chart_point` handler appends live data.

---

## 9. Startup Data Flow

### 9.1 Sequence

```
1. FastAPI app starts (uvicorn.run)
   │
2. Create/open SQLite database
   ├── Apply PRAGMAs (WAL, synchronous=NORMAL, etc.)
   ├── Run CREATE TABLE IF NOT EXISTS for all 4 tables
   └── Run PRAGMA incremental_vacuum; (reclaim free pages)
   │
3. DataCollector.__init__() — instantiate, set file paths, init ring buffers
   │
4. First poll (synchronous, blocking)
   ├── Read dopamine_status.txt → initial hero + training state
   ├── Read organism_latest.json → initial checkpoint info
   ├── Run nvidia-smi (subprocess) → initial GPU state
   ├── Read blocks.jsonl → initial blockchain state
   ├── Read checkpoint .pt → organ identities (expensive, only once)
   ├── Compute SHA-256 hashes → initial integrity state
   └── Parse DIARIO_DE_BORDO.md → initial timeline state
   │
5. Seed SQLite backfill
   ├── Read last 1000 rows per metric from metrics table
   └── Populate ChartRingBuffers from DB (so GET /api/metrics/history works immediately)
   │
6. Build `initial` context dict for SSR Jinja2 template
   │
7. Start asyncio tasks:
   ├── poll_loop() — runs every 2s
   ├── heartbeat() — runs every 15s
   ├── dashboard_event_logger() — periodic dashboard self-logging
   └── (optional) logbook_watcher() — watches DIARIO_DE_BORDO.md mtime
   │
8. Server ready → accept HTTP connections
   │
9. GET / → render dashboard.html with `initial` context
   │
10. Browser loads page:
    ├── Parse HTML, apply CSS
    ├── Initialize Chart.js (4 canvases) with empty data
    ├── Connect EventSource → GET /dashboard/sse/stream
    ├── SSE 'chart_point' first event → seed Chart.js deques
    ├── SSE 'hero', 'training', 'gpu' events → update DOM
    └── htmx starts polling section endpoints as fallback
```

### 9.2 SQLite Backfill

```python
async def backfill_chart_buffers(db: DatabaseManager, buffers: dict[str, RingBuffer]):
    """Load last 1000 points per metric from SQLite into ring buffers."""
    for metric_key, buf in buffers.items():
        rows = await db.fetch(
            "SELECT timestamp_unix_s, value FROM metrics "
            "WHERE metric_key = ? ORDER BY timestamp_unix_s DESC LIMIT 1000",
            (metric_key,)
        )
        # Rows come back newest-first; insert in chronological order
        for ts, val in reversed(rows):
            buf.push(ts, val)
```

### 9.3 Error Recovery During Startup

| Failure | Behavior |
|---|---|
| SQLite file locked | Retry 3x with 1s backoff. If still locked, exit with error. |
| dopamine_status.txt missing | Continue. Set all training metrics to null. Dashboard shows `--`. |
| organism_latest.json missing | Continue. Set checkpoint info to null. IntegrityRow shows "(no file)". |
| nvidia-smi unavailable | GPU array empty. GPU cards show "No NVIDIA GPUs detected". |
| DIARIO_DE_BORDO.md missing | Timeline empty. Show "No agent entries yet" placeholder. |
| config.yaml parse error | Fatal. Exit with message: "Cannot parse config at <path>". |
| blocks.jsonl missing | Normal. BlockchainCard shows "NO FILE" status. |
| workspace/runtime/dashboard/ dir missing | Create it (mkdir -p). |

---

## 10. Shutdown Data Flow

### 10.1 Sequence

```
1. Signal received: SIGINT (Ctrl+C) or SIGTERM
   │
2. Set shutdown flag (asyncio.Event)
   │
3. Cancel poll_loop task
   ├── Wait for current poll cycle to complete (max 5s)
   └── If still running after 5s, force-cancel
   │
4. Flush ring buffers to SQLite
   ├── For each ChartRingBuffer: INSERT any points not yet in DB
   │   (tracked by last_persisted_ts per metric)
   └── COMMIT
   │
5. PRAGMA wal_checkpoint(TRUNCATE);  -- flush WAL to main DB
   │
6. PRAGMA optimize;                  -- update query planner stats
   │
7. Close SQLite connection
   │
8. Broadcast final SSE event: event: shutdown
   data: {"reason": "dashboard_stopped", "at": "<ISO8601>"}
   │
9. Close all SSE client connections
   │
10. Append shutdown entry to runs.jsonl (if a training run was active)
    {
      "run_id": "run-20260724-084400",
      "stopped_at": "2026-07-24T11:05:00Z",
      "exit_reason": "dashboard_shutdown"
    }
    │
11. (Optional) Append dashboard shutdown note to DIARIO_DE_BORDO.md
    via append_logbook_event("dashboard_stop", ...)
    │
12. uvicorn server shutdown complete
```

### 10.2 Shutdown Handler (signal handler)

```python
import signal
import asyncio

shutdown_event = asyncio.Event()

def signal_handler(sig, frame):
    # Schedule the shutdown coroutine
    asyncio.create_task(graceful_shutdown())

async def graceful_shutdown():
    """Gracefully stop the dashboard."""
    logger.info("Shutting down...")

    # 1. Signal poll loop to stop
    shutdown_event.set()

    # 2. Wait for poll loop to finish current cycle
    try:
        await asyncio.wait_for(poll_loop_task, timeout=5.0)
    except asyncio.TimeoutError:
        poll_loop_task.cancel()

    # 3. Flush buffers to DB
    await flush_buffers_to_db()

    # 4. Checkpoint and close SQLite
    await db.checkpoint()
    await db.close()

    # 5. Send final SSE event
    await sse_manager.broadcast("shutdown", {
        "reason": "dashboard_stopped",
        "at": _iso860j(),
    })

    # 6. Close SSE connections
    await sse_manager.close_all()

    # 7. Write shutdown event to runs.jsonl
    await write_run_event("dashboard_stop")

    logger.info("Shutdown complete.")
```

---

## 11. File Plan (complete)

```
src/scripts/
  darwin_dashboard_web.py          # FastAPI app, DataCollector, SSEManager,
                                    # LogbookParser, DatabaseManager, HashCache,
                                    # poll loop, SSE endpoint, REST endpoints
  templates/
    dashboard.html                  # Full Jinja2 template (~1545 lines, already built)
    sections/                       # htmx partial templates (optional — SSE preferred)
      hero.html
      gpu.html
      training.html
      blockchain.html
      organs.html
      hashes.html
      timeline.html
      health.html
  static/
    htmx.min.js                     # htmx 2.x (14KB, committed)
    chart.min.js                    # Chart.js 4.x (65KB, committed)
    dashboard.css                   # External dark theme CSS (extracted from <style>)

workspace/
  runtime/
    dashboard/
      metrics.db                    # SQLite WAL database (auto-created)
      metrics.db-wal                # WAL file (auto-managed)
      metrics.db-shm                # WAL shared memory (auto-managed)
      runs.jsonl                    # Training run log (append-only JSONL)
      agent_timeline.json           # Cached parsed DIARIO_DE_BORDO.md
```

---

## 12. Implementation Checklist (for Wave 3 coders)

### Phase A: Data Layer
- [ ] `DatabaseManager` class: open, PRAGMA, CREATE TABLE, insert, query, checkpoint, close
- [ ] `RingBuffer` class: push, last_n, range, count
- [ ] `FileState` class: changed() with mtime/size/inode
- [ ] Unit tests for all three

### Phase B: Data Collectors
- [ ] `DataCollector.__init__` — set up all paths and FileStates
- [ ] `_read_dopamine` — parse space-separated key=value
- [ ] `_read_pointer` — parse organism_latest.json
- [ ] `_read_gpu` — asyncio subprocess nvidia-smi
- [ ] `_read_blockchain` — count lines, parse last block
- [ ] `_read_system_health` — psutil RAM/Disk/processes
- [ ] `_read_organ_states` — checkpoint load + config → statuses
- [ ] `_compute_hashes` — SHA-256 with 64KB chunking, asyncio.to_thread
- [ ] `_build_hero_event`, `_build_training_event`, `_build_chart_point`
- [ ] Unit tests with mock file contents

### Phase C: SSE
- [ ] `SSEManager` class: add_client, remove_client, broadcast, heartbeat
- [ ] `GET /dashboard/sse/stream` endpoint
- [ ] Client reconnection handling (EventSource auto-reconnect)
- [ ] Dead client cleanup (QueueFull)

### Phase D: REST Endpoints
- [ ] `GET /dashboard` — SSR Jinja2 template with `initial` context
- [ ] `GET /api/metrics/history` — SQLite queries with from/to/limit
- [ ] `GET /api/status` — lightweight JSON status
- [ ] `GET /api/runs` — runs.jsonl reader
- [ ] `GET /api/sections/{name}` — htmx partial fallbacks
- [ ] Static file mount for /static/

### Phase E: Logbook
- [ ] `LogbookParser` class: full implementation with all regex patterns
- [ ] `EstadoAtual`, `AgentEntry`, `KnownBug`, `ParsedLogbook` dataclasses
- [ ] `append_logbook_event` with msvcrt.locking
- [ ] Cache validation (mtime check)

### Phase F: Integration
- [ ] Startup sequence (backfill, first poll, SSR render)
- [ ] Shutdown sequence (flush, checkpoint, SSE close, logbook append)
- [ ] `poll_loop` main coroutine
- [ ] Training status detection (running/stopped/hung)
- [ ] Edge case: no training running (all metrics null, status=stopped)
- [ ] Edge case: training starts while dashboard is running
- [ ] Edge case: file rotation (causal_events.jsonl → dead_session)

### Phase G: Static Assets
- [ ] Download htmx 2.x minified (14KB) → src/tools/static/htmx.min.js
- [ ] Download Chart.js 4.x minified (65KB) → src/tools/static/chart.min.js
- [ ] Extract critical CSS from dashboard.html → src/tools/static/dashboard.css (optional)

### Phase H: Smoke Test
- [ ] Start dashboard with training stopped: all sections show "--" or last-known values
- [ ] Start dashboard with training running: all sections populate with live data
- [ ] Kill training: dashboard shows status=stopped within 2 polls, STALE indicators after 30s
- [ ] Restart training: dashboard detects new data and transitions to running
- [ ] Two browser tabs: both receive SSE events independently
- [ ] Network disconnect: dashboard shows reconnecting overlay, recovers on reconnect
- [ ] Server restart: client reconnects, charts re-seed from SQLite
