# CODEX 16B CANARY — 2026-07-15

Status: stable_canary

Base: organism_cycle_068.pt (SHA-256 4416A9AF)
Candidate: canary_recovery_20260715_162156/organism_cycle_069.pt
Steps: 250 (36502–36751)

Metrics:
  Fresh: 200 updates, LM ~1.8
  Replay: 50 updates, LM ~1.5
  Baseline heldout: 5.766
  Candidate heldout: 5.668
  Delta: −0.098 (no regression)

Gates: all passed
  - No GPU events
  - Peak temps 57/63 C
  - Checkpoint verified
  - 224 experts, 0 deaths

Decision: stable_canary — hardware stable, checkpoint retomavel.
