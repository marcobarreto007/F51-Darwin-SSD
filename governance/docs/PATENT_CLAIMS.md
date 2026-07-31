# Patent Claims — Persistent Neural Memory with Integrity-Verified Reload

**Inventor:** Marco (F51 Darwin)  
**Date:** 2026-07-30  
**Priority:** To be established via patent deposit (CIPO provisional or US provisional)  
**Warning:** Do NOT publish on arXiv or elsewhere before depositing. Public disclosure before deposit forfeits patent rights outside Canada/US (no grace period in Europe, China, Japan). Canada and US allow 12-month grace period for inventor's own disclosure — deposit first, publish after.  
**Jurisdiction:** Canada (CIPO), US (USPTO), PCT (international)  
**Git history:** Proves date of conception (useful in inventorship disputes) but does NOT establish patent priority. Only the deposit date establishes priority.

---

## Independent Claim 1

A method for persistent associative memory in a neural network, comprising:

(a) encoding input representations into key-value pairs using a learned encoder module with trainable weights;

(b) storing said key-value pairs in a memory store;

(c) computing a cryptographic fingerprint of the encoder module weights using a hash function;

(d) embedding said fingerprint in a serialized snapshot of the memory store;

(e) upon reloading the snapshot into a new instance of the neural network, recomputing the fingerprint of the encoder module weights of the new instance;

(f) comparing the recomputed fingerprint with the embedded fingerprint; and

(g) refusing to load the memory store when the fingerprints do not match, thereby preventing silent recall degradation caused by encoder incompatibility between write and read instances.

## Dependent Claim 2

The method of claim 1, wherein the hash function is SHA-256.

## Dependent Claim 3

The method of claim 1, wherein each stored key-value pair includes a provenance field indicating whether the pair originated from human teaching, model-verified correction, or model-generated quarantine.

## Dependent Claim 4

The method of claim 3, wherein memory pairs with model_quarantine provenance are excluded from recall when verified-only retrieval is requested.

## Dependent Claim 5

The method of claim 1, wherein the snapshot includes a self-validating checksum (snapshot_sha256) computed over all snapshot contents excluding the checksum field itself, enabling detection of tampering after serialization.

## Dependent Claim 6

The method of claim 1, wherein the memory store supports multiple alternative key embeddings per record (alt_keys), and recall scores a record by the maximum cosine similarity across all its keys.

## Dependent Claim 7

The method of claim 1, further comprising an exact rollback mechanism wherein the weight tensor of the encoder module is restored to a previously-saved state and verified by hash comparison.

## Dependent Claim 8

The method of claim 1, wherein the memory store exposes a consolidation operation that deduplicates records with cosine similarity above a threshold and decays records with model_quarantine provenance that have never been accessed.

---

## Independent Claim 9 (System)

A system for persistent associative memory in a neural network, comprising:

(a) a learned encoder module configured to encode input representations into key embeddings and value embeddings;

(b) a memory store configured to store records each containing a key embedding, a value embedding, and a provenance field;

(c) a snapshot module configured to serialize the memory store along with a cryptographic fingerprint of the encoder module weights and a self-validating checksum;

(d) a reload module configured to, upon deserialization, recompute the fingerprint of the encoder module weights of the current instance, compare it with the embedded fingerprint, and refuse to load when the fingerprints do not match.

---

## Prior Art Declaration

The following prior art has been identified and does not anticipate the claimed invention:

| Reference | What it does | What it does NOT do |
|---|---|---|
| US 11,568,062 | Hashes model weight segments for provenance | Does not hash memory state; does not verify encoder compatibility on reload; does not refuse load |
| CycloneDX 1.7 / AI-BOM | Cryptographic ML-BOM for model artifacts | Operates at model level, not per-record memory level; does not verify learned encoder compatibility |
| Titans (arXiv 2501.00663) | Neural long-term memory with surprise-based writes | No cryptographic fingerprint; no encoder compatibility verification; no refusal |
| MINJA | Memory poisoning defense via provenance tags | Tags records; does not verify encoder integrity; does not refuse load |
| SMSR / MemoryGraft | Signed memory for multi-agent systems | Signs text content; reindexes on incompatibility (destructive); does not refuse load |

## Distinction from Obviousness

The claimed invention is not obvious over the combination of checksums (TCP, filesystems) and neural memory (Titans, Infini-Attention) because:

(a) Neural network encoders are stochastic learned projections, not deterministic bit-transmission channels. A checksum mismatch in TCP indicates corruption of a known signal. An encoder fingerprint mismatch indicates a different learned projection space — the stored embeddings were encoded with one projection and would be queried with another. A checksum cannot detect this; only a fingerprint of the encoder weights can.

(b) The standard engineering practice for encoder mismatch is reindexing (deleting and recreating the memory). The claimed invention inverts this: it treats the memory as the invariant and refuses to operate when the encoder cannot faithfully retrieve it. This is a design choice that goes against the grain of the literature, which universally treats encoder mismatch as an operational bug to be mitigated, not an integrity invariant to be enforced.
