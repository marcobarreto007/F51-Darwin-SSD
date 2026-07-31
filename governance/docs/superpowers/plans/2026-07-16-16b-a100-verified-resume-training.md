# Darwin-X 1.6B A100 Verified Resume Training Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume the canonical Darwin-X 1.6B Nitro lineage at cycle 71 on the rented A100 80 GB and train it against the best verified local corpus.

**Architecture:** The local checkpoint and corpus remain authoritative. Large artifacts cross the Vast proxy as independently named chunks, are verified individually, assembled in lexical order, verified again as complete files, and only then published to the training paths. A bounded smoke cycle must resume the full model and AdamW state before `run247` is launched.

**Tech Stack:** Windows PowerShell 7, Python 3.11/3.12, PyTorch 2.6 CUDA 12.4, OpenSSH/SCP, SHA-256, gzip, Vast.ai A100 80 GB.

## Global Constraints

- Canonical lineage: `F51-Darwin-X-1.6B-Nitro`.
- Canonical resume: `organism_cycle_071.pt`, cycle 71, step 40751.
- Corpus: `00_CORPUS_PRINCIPAL_tokens_feast_v2.bin`, exactly 18,548,972,689 little-endian int32 tokens.
- Never use `/workspace/tokens.npy` or the 40 MB bootstrap token file for the production run.
- Never concatenate parallel streams into one remote file.
- Do not publish a remote artifact until its complete SHA-256 matches the local SHA-256.
- Use BF16, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, batch size 1, and block size 64 for the first A100 run.
- Keep smoke checkpoints isolated from the canonical remote checkpoint root.

---

### Task 1: Repair the Soul metric contract

**Files:**
- Modify: `src/scripts/darwin_organism.py`
- Test: `src/tests/test_organism_causal_runtime.py`

**Interfaces:**
- Consumes: `ChannelWindow.snapshot() -> dict[str, float | int | None]`.
- Produces: `metric_channel_lm_loss(channel, default) -> float | None`.

- [x] Add tests proving an LM-loss snapshot becomes a number and an empty window returns the caller's default.
- [x] Replace the dictionary values passed to `F51Soul.heartbeat()` with numeric LM losses.
- [x] Run:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.\.venv_nitro\Scripts\python.exe -m py_compile src\\scripts\\darwin_organism.py src\\f51_darwin\\soul.py
.\.venv_nitro\Scripts\python.exe -m pytest -q -p no:cacheprovider src\\tests\\test_organism_causal_runtime.py
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: compilation succeeds and all focused tests pass.

### Task 2: Freeze authoritative artifact identities

**Files:**
- Create externally: `F51-Dataset-Organizado/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast_v2.bin.manifest.json`
- Read: `F51-Dataset-Organizado/03_CHECKPOINTS/organism_cycle_071.pt`

**Interfaces:**
- Produces corpus bytes, token count, SHA-256, gzip bytes, and gzip SHA-256.
- Produces checkpoint bytes, file SHA-256, tensor identity, topology validity, and optimizer compatibility.

- [x] Compress `feast_v2` deterministically with gzip level 1 while hashing the uncompressed stream.
- [x] Verify the checkpoint with:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.\.venv_nitro\Scripts\python.exe src\\scripts\\inspect_organism_checkpoint.py `
  C:\Users\marco\Desktop\F51-Dataset-Organizado\03_CHECKPOINTS\organism_cycle_071.pt `
  --config src\\configs\\darwin_x_1.6b_nitro.yaml --verify-identity
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Expected: `identity_verified`, `topology_manifest_valid`, `optimizer_resume_compatible`, and `strict_resume_compatible` are all true.

### Task 3: Transfer artifacts as verified independent chunks

**Files:**
- Local source: `%TEMP%\darwin_cloud_stage\00_CORPUS_PRINCIPAL_tokens_feast_v2.bin.gz`
- Local source: `F51-Dataset-Organizado/03_CHECKPOINTS/organism_cycle_071.pt`
- Remote staging: `/workspace/.f51_uploads/`

**Interfaces:**
- Each chunk is named `chunk_000`, `chunk_001`, and so on.
- Each chunk has a local SHA-256 and an identical remote SHA-256.
- Assembly is `cat chunk_000 chunk_001 ... > target.tmp`, followed by a complete hash check and atomic `mv`.

- [ ] Split both artifacts into 768 MiB chunks without loading either complete file into memory.
- [ ] Upload at most four chunks concurrently, each to its own remote path.
- [ ] Verify every remote chunk hash.
- [ ] Assemble each complete remote artifact in lexical chunk order.
- [ ] Verify the complete gzip hash and checkpoint hash before atomic publication.

### Task 4: Materialize and validate the remote corpus

**Files:**
- Remote compressed input: `/workspace/.f51_uploads/feast_v2.bin.gz`
- Remote final corpus: `/workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast_v2.bin`
- Remote manifest: `/workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast_v2.bin.manifest.json`

**Interfaces:**
- `gzip -t` must succeed before decompression.
- Final corpus must be 74,195,890,756 bytes and SHA-256 `9677e9f22f4d78efa7b25c77b2da3cbdb5fb2a499926ac641a75c13b65e25cff`.

- [ ] Run `gzip -t` on the assembled gzip.
- [ ] Decompress into a temporary path.
- [ ] Verify byte count, divisibility by four, token count, and SHA-256.
- [ ] Atomically publish the corpus and its manifest.

### Task 5: Run a bounded A100 resume smoke

**Files:**
- Remote checkpoint: `/workspace/03_CHECKPOINTS/organism_cycle_071.pt`
- Remote smoke root: `/workspace/03_CHECKPOINTS/cloud_smoke_cycle071/`
- Remote log: `/workspace/logs/smoke_cycle071.log`

**Interfaces:**
- Smoke resumes cycle 71, step 40751, Topology Manifest v7, replay buffer, and AdamW momentum.
- Smoke uses two optimizer microsteps over the real `feast_v2` corpus.

- [ ] Synchronize the patched `src/scripts/darwin_organism.py` and confirm its SHA-256.
- [ ] Run remote `py_compile`.
- [ ] Launch one foreground `cycle --steps 2` with BF16, block size 64, batch size 1, and isolated checkpoint root.
- [ ] Require finite loss, no traceback, no CUDA OOM, and a completed isolated checkpoint.

### Task 6: Launch and prove persistent training

**Files:**
- Remote launcher: `/workspace/launch_16b_a100.sh`
- Remote stdout/stderr: `/workspace/logs/run247_16b_a100.log`
- Remote PID file: `/workspace/run247_16b_a100.pid`

**Interfaces:**
- `run247` resumes the canonical cycle-71 checkpoint.
- Production token source is the verified `feast_v2` file.
- The process is detached with `nohup`, logs are append-only, and checkpoints remain under `/workspace/03_CHECKPOINTS/`.

- [ ] Launch `run247` with block size 64, batch size 1, accumulation 1, learning rate 0.00015, and evaluation every 500 steps.
- [ ] Confirm the PID exists after 60 seconds.
- [ ] Confirm A100 utilization, finite first-step loss, real token count, cycle 72, and checkpoint resume identity in the log.
- [ ] Continue monitoring until at least one new remote checkpoint is atomically published.

