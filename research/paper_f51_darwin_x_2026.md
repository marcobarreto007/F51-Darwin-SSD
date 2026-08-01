# Prefix State Reuse in a Hybrid SSD/Attention Model: A Correctness-First Measurement

**Author:** Marco Barreto — F51 Darwin-X Laboratory (independent, local-only)
**Status:** Technical report. Not submitted.
**Artifacts:** `research/btb/state_reuse.py`, `research/btb/verify_state_reuse.py`,
`research/btb/bench_warm_vs_suffix.py`, `src/tests/test_btb_state_reuse_parity.py`,
`research/btb/state_reuse_results.json`, `research/btb/warm_vs_suffix_results.json`

---

## Abstract

Bursty inference workloads — subagent fanout, batch extraction — send many requests sharing
one long prompt prefix. Reusing the prefix's cached state instead of recomputing it is the
obvious optimization, and recent routing work (Biting the Bullet) builds burst detection and
speculative replication on top of it. We ask a narrower question first: **in a hybrid
Selective-State-Space (SSD) + attention model, is prefix state reuse actually correct, and
what does the cache actually cost?**

We report three findings on F51 Darwin-X (153M params, 9 SSD layers / 3 attention layers,
NVIDIA RTX 5060 Ti). First, a **correctness pitfall**: with a populated KV cache where
`k_len > q_len`, PyTorch's `is_causal=True` aligns the triangular mask top-left, so suffix
queries attend only to the first few keys and silently ignore the cached prefix. Second,
with an absolute-position mask the reuse path is **numerically equivalent to full recompute**
(max |Δlogit| ≤ 2e-6, KL ≤ 4.7e-7, top-1 agreement 8/8 across prefixes of 128–2048 tokens).
Third, the hybrid cache splits cleanly: **SSD recurrent state is constant at 684 KiB**
regardless of prefix length, while attention KV grows at 3.0 KiB/token, crossing over at
**~228 tokens**.

Latency is **scale-dependent**. At 153M a fixed per-forward overhead of ~780 ms dominates
compute and the amortized speedup is noise (+29.2%, −25.5%, −0.5%, +5.1%) — a null result. At
517M the same measurement, gated on parity, yields **4.3× to >18×** for bursts of 50 requests
over prefixes of 256–2048 tokens. The effect is real; it is simply invisible below the scale
where compute exceeds framework overhead.

We argue that the correctness gate — and its negative control — is the reusable contribution,
because it is what separates real state reuse from measuring "fewer tokens is faster," and
because every speedup figure here is reported only for a path proven equivalent to recompute.

---

## 1. Motivation and scope

The appeal of prefix reuse in a hybrid architecture is that the SSD half carries a
*constant-size* recurrent state, so replicating a warm prefix between slots should not scale
with prefix length the way a pure-Transformer KV cache does. That intuition is sound but had
not been measured here, and the surrounding claims are easy to get wrong in both directions.

This report deliberately does **not** cover: multi-GPU or RDMA replication (untested), burst
routing policy, or test-time training. It covers exactly what was measured on one device.

### 1.1 What motivated a correctness-first protocol

An earlier benchmark in this repository (`research/btb/run_e2e.py`) reported 50–62% speedups.
Its warm path called the model on the suffix alone and stored a placeholder dict in place of
the cache. It measured "processing 32 tokens is faster than processing 1056," and the
resulting hidden state differed from the true one by 82.9% relative error. The lesson
generalizes: **a reuse benchmark without an output-equivalence gate measures nothing**, and
the failure is invisible in the timing numbers. Everything below is built around that gate.

---

## 2. Method

### 2.1 Cache model

One slot per layer, mirroring `DarwinInferenceEngine._forward_with_cache` in single-device
form for auditability:

* **Attention layers** (`GQACausalAttention`): per-layer `k`, `v` tensors of shape
  `[1, n_kv_heads, filled, head_dim]`, concatenated as the sequence extends. Size **O(N)**.
* **SSD layers** (`SSDMixerOnly`): per-layer `conv_state` and `ssm_state` obtained via
  `block.ssd(..., return_state=True)`. Size **O(1)** in sequence length.

### 2.2 The causal-mask pitfall

`F.scaled_dot_product_attention(..., is_causal=True)` builds its triangular mask aligned to
the **top-left** of the `[q_len, k_len]` score matrix. During prefill `q_len == k_len` and
this is correct. During cached continuation `k_len = prefix + suffix` while `q_len = suffix`,
so a suffix query at absolute position `p` is permitted to attend only to keys
`0 .. (its row index)` — the first few prefix tokens — rather than to everything up to `p`.
The cached prefix is effectively discarded, and no error is raised.

We instead build the mask from absolute positions:

```python
q_pos = position + torch.arange(seq)      # absolute positions of the queries
k_pos = torch.arange(k_len)               # absolute positions of all cached keys
attn_mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
```

Before this fix, warm and cold logits diverged by ~5.6e-2. After it, by ~1e-6.

### 2.3 Parity gate

For each of `burst = 8` distinct suffixes sharing one prefix we compare final-token logits
from the cold path (recompute prefix+suffix from scratch) and the warm path (clone the prefix
cache, continue from absolute position `prefix_len`). We require **all** of:
max |Δlogit| ≤ 2e-2, KL(cold ‖ warm) ≤ 1e-5, and top-1 agreement on every suffix.
`verify_state_reuse.py` exits non-zero and suppresses all timing output if the gate fails.

### 2.4 Negative control

A parity test can pass vacuously if the model ignores its context. `test_btb_state_reuse_parity.py`
therefore also asserts that dropping the prefix entirely — precisely what the earlier benchmark
did — **must** change the logits (max |Δ| > 1e-3). A third test asserts the memory scaling
claim directly: attention KV must grow superlinearly with a 4× token increase while SSD state
stays byte-identical.

### 2.5 Weights

Deterministic random init (seed 51); no checkpoint is loaded. Parity is a statement about the
algebra of the cache and is weight-independent. Latency reflects real FLOPs and kernel
behavior but not generation quality. No claim in this report depends on model quality.

---

## 3. Results

### 3.1 Parity holds

RTX 5060 Ti, `src/configs/darwin_x_100m.yaml`, 153M params, burst = 8, suffix = 16 tokens.

| Prefix | Parity | max abs Δ | KL | top-1 |
| ---: | :---: | ---: | ---: | :---: |
| 128 | OK | 1.0e-06 | 2.44e-07 | 8/8 |
| 512 | OK | 1.0e-06 | 2.43e-07 | 8/8 |
| 1024 | OK | 2.0e-06 | 4.72e-07 | 8/8 |
| 2048 | OK | 2.0e-06 | 2.43e-07 | 8/8 |

Residual is float non-associativity: cold sums the prefix and suffix contributions in one
pass, warm sums them across a concatenation boundary. Three orders of magnitude below the
gate.

### 3.2 Hybrid cache scaling, measured

| Prefix | Attention KV | SSD state | Total |
| ---: | ---: | ---: | ---: |
| 128 | 384 KiB | 684 KiB | 1068 KiB |
| 512 | 1536 KiB | 684 KiB | 2220 KiB |
| 1024 | 3072 KiB | 684 KiB | 3756 KiB |
| 2048 | 6144 KiB | 684 KiB | 6828 KiB |

Attention KV is exactly linear at **3.0 KiB/token** (3 attention layers, `n_kv_heads=2`,
`head_dim=64`, fp32). SSD state is **byte-identical** across a 16× range of prefix lengths —
the O(1) claim is confirmed, not assumed.

**Crossover:** SSD state is the larger of the two below **228 tokens** (684 / 3.0) and the
smaller above it. For short prefixes the hybrid design *costs* memory relative to a
pure-attention cache; the advantage is real but begins later than the framing "constant state
is always cheaper" suggests. At 2048 tokens the SSD half is 10% of the total cache; the
attention half still dominates and still grows linearly. **A hybrid model does not have an
O(1) cache.** It has an O(N) cache with a smaller constant.

### 3.3 Latency is a function of scale, not of the mechanism

**At 153M: null result.**

| Prefix | Cold (ms) | Warm amortized (ms) | Speedup |
| ---: | ---: | ---: | ---: |
| 128 | 1120.3 | 793.0 | +29.2% |
| 512 | 802.1 | 1006.8 | −25.5% |
| 1024 | 944.3 | 948.8 | −0.5% |
| 2048 | 1216.8 | 1155.2 | +5.1% |

No monotonic trend, sign changes twice. A control measured ~780 ms per forward at 32 tokens
versus 1043 ms at 1056 tokens — a 1.34× ratio across a 33× difference in token count.
Per-forward framework overhead, not compute, dominates at this size and swamps whatever the
reuse saves. **No speedup should be claimed at 153M.**

**At 517M (`darwin_x_600m`, d_model 1408, same 3 attention / 9 SSD split): the effect appears.**
Burst of 50 requests, 32-token suffixes, min-of-10 with global warmup, parity gate re-run at
every prefix length:

| Prefix | T_full (ms) | Prefill once (ms) | Warm/request (ms) | Cold total (ms) | Warm total (ms) | Speedup | KL | Parity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 256 | 724.7 | 633.4 | 156.7 | 36 233 | 8 470 | **4.28×** | 2.4e-07 | OK |
| 512 | 1298.7 | 1212.3 | 161.9 | 64 934 | 9 308 | **6.98×** | 2.4e-07 | OK |
| 1024 | 2409.8 | 2371.5 | 175.2 | 120 491 | 11 131 | **10.82×** | 2.4e-07 | OK |
| 2048 | 4670.5 | 4563.4 | 99.8 | 233 526 | 9 552 | **24.45×** | 1.0e-08 | OK |

*Run-to-run variance.* Three independent runs gave 4.19/3.97/4.28× at 256, 10.96/11.43/10.82×
at 1024 — stable within ±5%. The 2048 row is the least stable: warm-per-request measured
160.3 ms and 99.8 ms on two runs, giving 18.5× and 24.4×. Treat 2048 as **≥18×** rather than
as a point estimate; the shorter prefixes are the trustworthy figures.

`T_full` now scales with prefix length (725 → 4671 ms across 256 → 2048 tokens) rather than
sitting on a constant floor: compute finally exceeds overhead. Warm cost per request is
**flat at ~160–173 ms** regardless of prefix length, which is the shape the mechanism predicts
— the suffix work is constant and the prefix is paid once. Speedup therefore grows roughly
linearly with prefix length, exceeding 18× at 2048 tokens.

Cache clone cost is included in the warm figures (`warm_logits` clones before continuing) and
is sub-millisecond at these sizes.

**A note on why the warm path is not more expensive than it looks.** One might expect
continuing from a cache to cost *more* than running the suffix alone, since attention must
scan `k_len = prefix + suffix` keys instead of 32. Measured, the real warm path is
0.86–0.96× the cost of the suffix-alone forward — slightly *cheaper*. The reason is
architectural: only 3 of 12 layers are attention. The extra key-scan is confined to those
three, while the 9 SSD layers and all FFN blocks process 32 tokens either way and dominate
the budget. In a pure-Transformer model this would not hold and the warm path would carry a
visible O(prefix) term.

---

## 4. Threats to validity

* **One device, two model sizes.** §3.3 is measured at 153M and 517M; the 1.6B MoE lineage is
  not covered here and its expert-routing overhead may change the picture. §3.1 and §3.2 are
  architectural and should hold across sizes.
* **The 517M speedup is amortized over a burst of 50** sharing one prefix. Smaller bursts
  amortize the single prefill over fewer requests and the advantage shrinks accordingly.
* **Random weights mean no cache-eviction pressure or real traffic distribution.** These are
  best-case conditions for reuse: one prefix, perfect hit rate, no memory contention.
* **Random weights.** Sound for parity and FLOP-level timing; says nothing about quality.
* **Single-device.** No PCIe or RDMA transfer was performed or timed. The machine has two
  GPUs (RTX 5060 Ti, RTX 3060), so this is testable and simply has not been done — any
  inter-GPU replication figure would be fabricated.
* **fp32 throughout.** A fp16/bf16 runtime halves both cache figures and moves the crossover.
* **Parity is measured on final-token logits**, the quantity that determines the next token.
  Divergence confined to earlier positions would not be caught.

---

## 5. What this does and does not support

Supported:

1. Prefix state reuse in this hybrid architecture is **implementable and numerically exact**,
   given an absolute-position causal mask.
2. The SSD/attention cache split is **real and measurable**: 684 KiB constant vs 3.0 KiB/token.
3. The `is_causal=True` pitfall is a **silent correctness bug** in cached hybrid inference and
   is worth documenting on its own.
4. At 517M, prefix reuse under a parity gate delivers **4.3×–24×** on bursts of 50, growing
   with prefix length — and **nothing measurable at 153M**. Scale is the variable that decides
   whether this optimization is worth implementing at all.

Not supported by anything here: any inter-GPU or RDMA transfer time; any claim about test-time
training; any comparison against vLLM, SGLang, or a published baseline; any claim that a hybrid
cache is O(1); and any speedup at 153M.

This is a technical note, not a conference paper. The parity gate and the negative control are
the parts worth reusing.

---

## 6. Reproduction

```bash
# Correctness gate (CPU, seconds). Exit code is the result.
python -m pytest src/tests/test_btb_state_reuse_parity.py -v

# Parity + cache scaling (GPU). Exits non-zero if parity fails; prints no speedup then.
python research/btb/verify_state_reuse.py \
    --config src/configs/darwin_x_100m.yaml \
    --prefix-tokens 128 512 1024 2048 --suffix-tokens 16 --burst 8

# §3.3 speedup at 517M, with the parity gate re-run at every prefix length.
python research/btb/bench_warm_vs_suffix.py \
    --config src/configs/darwin_x_600m.yaml \
    --prefix-tokens 256 512 1024 2048 --burst 50
```

Recorded output: `research/btb/state_reuse_results.json`,
`research/btb/warm_vs_suffix_results.json`.

---

## References

1. Gu, A., & Dao, T. (2024). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*.
2. Dao, T., & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms
   Through Structured State Space Duality*.
3. Kwon, W., et al. (2023). *Efficient Memory Management for Large Language Model Serving with
   PagedAttention* (vLLM). SOSP.
4. Zheng, L., et al. (2024). *SGLang: Efficient Execution of Structured Language Model Programs*.

> Prior work on burst detection and speculative KV replication motivated this line of
> investigation. That reference is deliberately omitted here pending verification of its
> bibliographic details — it was cited in an earlier draft from an unverified summary, and
> this report cites nothing it has not checked.
