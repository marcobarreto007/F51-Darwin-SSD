# The Convergence: What Three Null Results Actually Prove

## The three experiments

| Experiment | Result | Interpretation |
|---|---|---|
| Neuron ablation | +0.972 cross-domain | Domain margin = 2.5% of variance |
| SAE features (4096) | 0.94x separation ratio | Same cos_sim as raw neurons |
| Layer transplant (25% FFN) | Net negative | 1/6 improvement, 2/6 degraded |

## What this means

SmolLM2-1.7B has 1.2B FFN parameters with 8192 intermediate dimensions per layer.
For a model this size, the "superposition hypothesis" (Elhage et al., 2022) predicts
that features are stored in COMPRESSED form - multiple concepts share the same neurons
because there aren't enough dimensions to give each concept its own circuit.

**The entanglement is not between Python and Medicine specifically.**
It is between ALL knowledge domains. The model does not have "Python neurons" and
"Medicine neurons" - it has "useful pattern neurons" that both domains use.
This is not a bug. It is the optimal compression strategy for a model this size.

**Scaling matters.** Anthropic's "Scaling Monosemanticity" (2024) showed that features
DO become more separable in larger models (Claude 3 Sonnet, 70B+). The transition
from polysemantic to monosemantic happens with scale. SmolLM2-1.7B is simply below
the threshold where domain-specific circuits can form.

## What actually works (and why)

| Method | Works? | Why |
|---|---|---|
| ROME (add knowledge) | YES (6/6) | Rank-1 update installs a NEW direction without disturbing existing weights |
| UniversalMemory (store externally) | YES (5/5) | Bypasses the FFN entirely - stores in separate key-value space |
| Scanner + SHA-256 (catalog) | YES (196K stamps) | Observational, non-destructive |
| Domain ablation (separate) | NO (+0.972) | Model too small for dedicated circuits |
| SAE separation | NO (0.94x) | Features exist but shared ones dominate centroids |
| Layer transplant (transfer) | NO (net negative) | Instruct knowledge is not localized to specific layers |

## The architectural implication

For models at the 1-3B scale, the correct paradigm is:

    ADDITION, not SEPARATION.
    EXTERNAL MEMORY, not INTERNAL REORGANIZATION.
    CRYPTOGRAPHIC PROOF, not STATISTICAL CONFIDENCE.

The model is a SHARED COMPUTATIONAL SUBSTRATE. You cannot surgically remove
"Python knowledge" because Python knowledge is distributed across the same
neurons that implement reasoning, pattern matching, and English comprehension.
Removing it removes everything.

What you CAN do:
1. Install new knowledge on top (ROME)
2. Store facts externally and recall on demand (UniversalMemory)
3. Catalog every weight with cryptographic identity (scanner + SHA-256)
4. Prove that nothing else changed (rollback + hash verification)

## What this means for the Bluebook

The original vision was "surgical editing of domain-specific knowledge."
The data says: at 1.7B scale, domains are not separable.

The corrected vision:
"Cryptographic catalog of a shared computational substrate, with
provable additive knowledge installation and external associative memory."

This is MORE honest, MORE defensible, and MORE interesting to the MILA
researchers who study superposition and feature entanglement.
