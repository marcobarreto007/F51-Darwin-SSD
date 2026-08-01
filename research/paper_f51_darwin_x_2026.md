# Prefix State Reuse in a Hybrid SSD/Attention Model: A Correctness-First Measurement

**Author:** Marco Barreto — F51 Darwin-X Laboratory (independent, local-only)
**Status:** Technical report / negative-and-partial result. Not submitted.
**Artifacts:** `research/btb/state_reuse.py`, `research/btb/verify_state_reuse.py`,
`src/tests/test_btb_state_reuse_parity.py`, `research/btb/state_reuse_results.json`

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

We report a **null result on latency**. At this model scale a fixed per-forward overhead of
roughly 780 ms dominates compute, and measured amortized speedup is non-monotonic
(+29.2%, −25.5%, −0.5%, +5.1% at 128/512/1024/2048 tokens). We therefore make no performance
claim. We argue that the correctness gate — and its negative control — is the reusable
contribution, because it is what separates real state reuse from measuring "fewer tokens is
faster."

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

### 3.3 Latency: null result

| Prefix | Cold (ms) | Warm amortized (ms) | Speedup |
| ---: | ---: | ---: | ---: |
| 128 | 1120.3 | 793.0 | +29.2% |
| 512 | 802.1 | 1006.8 | −25.5% |
| 1024 | 944.3 | 948.8 | −0.5% |
| 2048 | 1216.8 | 1155.2 | +5.1% |

No monotonic trend, sign changes twice. A separate control measured a fixed cost of ~780 ms
per forward pass at 32 tokens versus 1043 ms at 1056 tokens on this model — a 1.34× ratio
across a 33× difference in token count. Per-forward overhead, not compute, dominates at 153M
parameters, and it swamps whatever the reuse saves. Cache clone cost is charged to the warm
path and is non-trivial at these sizes.

**We make no latency claim.** The numbers above are reported so they are not re-derived and
mistaken for signal later. A meaningful latency measurement needs a regime where compute
dominates: the 600M or 1.6B lineages, and/or prefixes well beyond 2048.

---

## 4. Threats to validity

* **One device, one model size.** Everything in §3.3 may change at 1.6B; §3.1 should not.
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

Not supported by anything here: any speedup figure; any inter-GPU or RDMA transfer time; any
claim about test-time training; any comparison against vLLM, SGLang, or a published baseline;
and any claim that a hybrid cache is O(1).

This is a technical note, not a conference paper. The parity gate and the negative control are
the parts worth reusing.

---

## 6. Reproduction

```bash
# Correctness gate (CPU, seconds). Exit code is the result.
python -m pytest src/tests/test_btb_state_reuse_parity.py -v

# Full measurement (GPU). Exits non-zero if parity fails; prints no speedup in that case.
python research/btb/verify_state_reuse.py \
    --config src/configs/darwin_x_100m.yaml \
    --prefix-tokens 128 512 1024 2048 --suffix-tokens 16 --burst 8
```

Recorded output: `research/btb/state_reuse_results.json`.

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
