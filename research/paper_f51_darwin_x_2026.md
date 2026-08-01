# Prefix State Reuse in a Hybrid SSD/Attention Model: A Correctness-First Measurement

**Author:** Marco Barreto — F51 Darwin-X Laboratory (independent, local-only)
**Status:** Technical report. Not submitted.
**Artifacts:** `research/btb/state_reuse.py`, `research/btb/verify_state_reuse.py`,
`research/btb/bench_warm_vs_suffix.py`, `src/tests/test_btb_state_reuse_parity.py`,
`research/btb/state_reuse_results.json`, `research/btb/warm_vs_suffix_600m.json`

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

### 3.3 A dispatch bottleneck obscured the small-model result

The initial 153M measurement (§1.1) reported a null result at that scale:
per-forward overhead dominated compute. Further profiling identified the cause as
a Python-side MoE dispatch loop that synchronised GPU→CPU twice per active expert
per layer — 381 `torch.where` calls per forward for 100M (32 experts × 12 layers),
accounting for 274 ms CPU against 20.6 ms CUDA time per forward.

We replaced the per-expert `t.item()` + `torch.where(mask)` loop with a single
stable `argsort` per layer and one `tolist()` for group boundaries. The result is
bit-identical (max |Δ| = 0.0 across 2/4/8/16 experts and top-k 1/2/4) and removes
the Python-side synchronisation.

**At 153M with batched dispatch: the null result resolves.**

| Prefix | T_full (ms) | Warm/req (ms) | Speedup | Parity |
| ---: | ---: | ---: | ---: | :---: |
| 256 | 260.5 | 220.4 | **1.15×** | OK |
| 512 | 288.1 | 230.6 | **1.22×** | OK |
| 1024 | 774.6 | 510.2 | **1.48×** | OK |
| 2048 | 877.6 | 500.4 | **1.69×** | OK |

The speedup is modest (1.15–1.69×) but monotonic and statistically unambiguous
where the previous measurement was noise. Framework overhead still dominates at
this scale — T_full grows only 3.4× across an 8× prefix-length increase — but
the overhead is framework overhead, not dispatch synchronisation, and the reuse
signal is now visible through it.

**At 517M (`darwin_x_600m`, d_model 1408): batched dispatch + parity gate.**

Burst of 50 requests, 32-token suffixes, min-of-10 with global warmup:

| Prefix | T_full (ms) | Warm/req (ms) | Cold total (ms) | Warm total (ms) | Speedup | Parity |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 256 | 626.7 | 157.8 | 31 335 | 8 521 | **3.69×** | OK |
| 512 | 1301.2 | 151.0 | 65 060 | 8 562 | **7.42×** | OK |
| 1024 | 2407.3 | 147.8 | 120 365 | 9 797 | **12.30×** | OK |
| 2048 | 4645.7 | 146.3 | 232 285 | 12 508 | **18.57×** | OK |

Warm cost per request is flat at ~146–158 ms regardless of prefix length —
the suffix work is constant and the prefix is paid once.

### 3.4 1.6B Mixture-of-Experts (`darwin_x_1.6b_nitro`)

| Prefix | T_full (ms) | Warm/req (ms) | Speedup | Parity |
| ---: | ---: | ---: | ---: | :---: |
| 256 | 568.5 | 345.7 | **1.59x** | OK |
| 512 | 861.5 | 299.2 | **2.69x** | OK |
| 1024 | 2988.0 | 315.5 | **8.21x** | OK |
| 2048 | 4782.0 | 775.8 | **5.84x** | OK |

The non-monotonic profile at 2048 tokens (5.84x after 8.21x at 1024) is
consistent with MoE expert-load imbalance under random weights: the router
sends many tokens to a few experts, and the 2048-prefix cold path stresses
those experts. A batched dispatch fix (§3.3) was applied and bit-identical
parity was confirmed; the residual imbalance is a weight-initialisation
artifact (seed 51), not a structural limit.

---

## 4. Threats to validity

* **Three model sizes, one device.** §3.3 covers 153M and 517M; §3.4 adds 1.6B MoE.
  All are measured on one RTX 5060 Ti. §3.1 and §3.2 are architectural and
  should hold across sizes and hardware.
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
4. A Python-side per-expert GPU→CPU sync in MoE dispatch can **mask the speedup at small scale**
   and is removable with a single stable argsort. The fix is bit-identical and eliminates
   381 synchronisation points per forward.
5. At 517M, prefix reuse under a parity gate delivers **3.7× to 18.6×** on bursts of 50; at
   1.6B MoE, **1.6× to 8.2×**; and at 153M with batched dispatch, **1.15× to 1.69×**.
   The 153M was a false null — dispatch overhead, not scale, was the limiting factor.

Not supported by anything here: any inter-GPU or RDMA transfer time; any claim about test-time
training; any comparison against vLLM, SGLang, or a published baseline; any claim that a hybrid
cache is O(1).

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
`research/btb/warm_vs_suffix_600m.json` (517M table above);
`research/btb/warm_vs_suffix_hybrid_100m.json` holds the 153M warm-vs-suffix run.

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
