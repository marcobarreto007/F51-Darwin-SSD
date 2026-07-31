# Integrity-Verified Persistent Memory for Neural Networks

**Authors:** Marco (F51 Darwin)
**Submitted to:** arXiv
**Date:** 2026-07-30

## Abstract

Persistent associative memory for neural networks enables knowledge retention across sessions without modifying model weights. However, existing approaches (Titans, Infini-Attention, MemoryGraft) store key-value embeddings without verifying that the encoder used to write them is the same encoder that will read them. When a memory snapshot is loaded into a new model instance with different encoder weights — whether from independent initialization, fine-tuning, or tampering — recall silently degrades while the memory store reports full integrity. We present a memory architecture that prevents this failure mode by embedding a cryptographic fingerprint of the encoder weights into the memory snapshot and refusing to load when the fingerprint does not match. The system additionally provides per-record provenance tracking (human, model-verified, model-quarantine), self-validating snapshot checksums, multi-key records for paraphrase-invariant recall, and exact hash-verified rollback. We demonstrate that without the fingerprint verification, recall drops from 1.0 to -0.007 on a 1.7B-parameter language model after memory reload, and we show that the verification catches this degradation with zero false positives across all tested configurations. 

## 1. Introduction

Neural networks with persistent memory (Graves et al., 2016; Rae et al., 2016; Behrouz et al., 2024) decouple knowledge storage from weight parameters, enabling fact retention without gradient updates. Recent work on neural long-term memory (Titans, Behrouz et al., 2024) and compressive memory (Infini-Attention, Munkhdalai et al., 2024) has shown that associative key-value stores can effectively augment transformer architectures.

However, all existing approaches share a silent failure mode: when a memory snapshot is loaded into a model instance with different encoder weights, the stored key embeddings — which were produced by the original encoder — are queried against embeddings produced by the new encoder. The cosine similarity between them approaches zero, but the memory store reports no errors. The recall appears to abstain due to low confidence, when in fact the retrieval mechanism is fundamentally broken.

## 2. The Encoder Mismatch Problem

[Section demonstrating: load memory into fresh model instance -> recall drops from 1.0 to -0.007, 0.101, 0.050 with no error raised]

## 3. Fingerprint-Verified Memory Architecture

[Section describing: memory_state_dict with encoder_fingerprint, load_memory_state with verify=True, snapshot_sha256 for tamper detection, provenance field, alt_keys for multi-phrasing]

## 4. Experimental Validation

[Section with: 6 passing tests, reload with correct encoder = 5/5 recall, reload with wrong encoder = ValueError raised, wrong encoder with verify=False = recall near zero]

## 5. Related Work

[Section covering: Titans, Infini-Attention, MINJA, MemoryGraft, SMSR, CycloneDX AI-BOM, US 11,568,062 — none do encoder fingerprint verification on reload]

## 6. Conclusion

The encoder fingerprint verification transforms a silent failure mode into a loud integrity invariant. When the encoder does not match, the system refuses to operate rather than silently returning degraded results. This design principle — refuse rather than degrade — applies beyond memory systems to any learned component whose function depends on the compatibility of its parameters with stored state.

## Code Availability

The implementation is available at [GitHub repository URL] under [license]. All experiments were conducted on consumer GPU hardware (NVIDIA RTX 5060 Ti 16GB + RTX 3060 12GB) with no cloud dependency.

## References

[15+ citations to Titans, Infini-Attention, MINJA, MemoryGraft, CycloneDX, relevant patents, EDPB 2026, EU AI Act]
