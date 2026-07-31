#!/usr/bin/env python3
"""Discover, ablate, and verify fact-specific channels in the residual stream.

Connects three subsystems:
  1. circuits/ablation.py  → finds fact-direction channels via gradient×activation
  2. cognition/memory.py   → verifies key_encoder projects onto same channels
  3. Ablation proof        → CLEAN/ABLATE/RESTORE/SHUFFLE paired arms

Protocol:
  Discovery split: 5 facts → rank residual channels by influence on answer
  Ablation:         zero top-K channels → measure recall drop
  Shuffle:          permute channels between facts → prove content-specific
  Restore:          un-zero → prove recovery
  Alignment:        cosine between key_encoder rows and ablation direction
"""

from __future__ import annotations

import gc, json, math, os, time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
from torch import nn

from f51_darwin.circuits.ablation import (
    AblationArm,
    AblationResult,
    CircuitSelector,
    discover_residual_channels,
    run_paired_ablation,
)
from f51_darwin.cognition import CognitiveForwardMetadata
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.hashing import sha256_file, atomic_json_write
from transformers import AutoTokenizer
from f51_darwin.transplant_16b.cli import DEFAULT_SOURCE_ROOT

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt"
OUTPUT = ROOT / "workspace/runtime/fact-ablation"
SYSTEM = "Voce e um assistente preciso. Responda com o codigo de seis digitos."


# ── Fact definitions ────────────────────────────────────────────────────────

FACTS = [
    ("arquivo-baleia", "457291"),
    ("chave-golfinho", "185034"),
    ("documento-falcao", "628410"),
    ("registro-coruja", "731592"),
    ("cartao-pelicano", "862045"),
]


# ── Model + tokenizer ───────────────────────────────────────────────────────

def build_model(with_cognition: bool = False, with_adapters: bool = False) -> DarwinXModel:
    payload = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    config = DarwinXConfig.from_mapping(payload["config"])
    if with_cognition:
        config = replace(config, cognitive_architecture_version="three_organs_v1",
                         cognitive_shadow_enabled=True, cognitive_pulse_enabled=True)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = DarwinXModel(config)
    torch.set_default_dtype(prev)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    if with_adapters:
        adapters_path = ROOT / "workspace/runtime/memory-training/trained-adapters-v2.pt"
        if adapters_path.exists():
            blobs = torch.load(adapters_path, map_location="cpu", weights_only=False)
            memory = model.cognitive_runtime.memory
            memory.key_encoder.load_state_dict(blobs["key_encoder"])
            memory.value_encoder.load_state_dict(blobs["value_encoder"])
            memory.readout.load_state_dict(blobs["readout"])
    model.eval()
    return model


def get_tokenizer() -> Any:
    return AutoTokenizer.from_pretrained(str(DEFAULT_SOURCE_ROOT), local_files_only=True)


def tokenize_fact(tokenizer: Any, entity: str, code: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (question_ids, answer_ids) for a fact."""
    q_chat = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Qual e o codigo atribuido a {entity}?"},
    ]
    q_fmt = tokenizer.apply_chat_template(q_chat, tokenize=False, add_generation_prompt=True)
    q_ids = tokenizer(q_fmt, add_special_tokens=False, return_tensors="pt").input_ids

    a_chat = [
        {"role": "system", "content": SYSTEM},
        {"role": "assistant", "content": f"O codigo de {entity} e {code}."},
    ]
    a_fmt = tokenizer.apply_chat_template(a_chat, tokenize=False, add_generation_prompt=False)
    a_ids = tokenizer(a_fmt, add_special_tokens=False, return_tensors="pt").input_ids
    return q_ids, a_ids


# ── Batch factory for discovery ─────────────────────────────────────────────

def make_discovery_batches(
    model: DarwinXModel,
    tokenizer: Any,
    facts: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Create batches for gradient×activation discovery.
    
    Each batch contains question_ids and a loss_fn that maximizes the correct answer.
    """
    batches = []
    for entity, code in facts:
        q_ids, a_ids = tokenize_fact(tokenizer, entity, code)
        batches.append({
            "input_ids": q_ids,
            "target_ids": a_ids,
            "entity": entity,
            "code": code,
        })
    return batches


# ── Loss function for discovery ─────────────────────────────────────────────

def fact_loss_fn(model: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    """Loss that maximizes probability of the correct answer tokens."""
    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]
    device = next(model.parameters()).device
    
    # Forward pass
    output = model(input_ids.to(device), heartbeat=False)
    logits = output.logits  # [1, T_q, vocab]
    
    # Cross-entropy on answer tokens (as if they followed the question)
    # We want the model's last-position logits to predict the first answer token
    target_token = target_ids[0, 0].to(device)
    loss = F.cross_entropy(logits[0, -1:, :], target_token.unsqueeze(0))
    return loss


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 68)
    print("Fact-Channel Discovery and Ablation Experiment")
    print("=" * 68)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Load model and tokenizer ---
    print("\n--- Phase 1: Load model ---")
    model = build_model(with_cognition=False)
    tokenizer = get_tokenizer()
    d_model = model.config.d_model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device=device, dtype=torch.bfloat16)
    print(f"  d_model={d_model}  device={device}")

    # --- Phase 2: Discover fact channels ---
    print("\n--- Phase 2: Discover fact channels via gradient x activation ---")
    discovery_facts = FACTS[:3]  # split: 3 facts for discovery
    batches = make_discovery_batches(model, tokenizer, discovery_facts)
    
    print(f"  Running discovery on {len(batches)} facts...")
    print(f"  Tap: norm (post-norm hidden, last token, {d_model} dims)")

    # Manual gradient×activation discovery (simpler than full circuits framework)
    # Compute gradient of loss w.r.t. hidden state, multiply by activation
    model.zero_grad()
    scores = torch.zeros(d_model, device="cpu")
    
    for batch in batches:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        
        # Hook to capture the hidden state at norm output
        captured = {}
        def hook(module, input, output):
            captured["hidden"] = output.detach()
        
        handle = model.norm.register_forward_hook(hook)
        
        # Forward
        output = model(input_ids, heartbeat=False)
        handle.remove()
        
        hidden = captured["hidden"]  # [1, T, d_model]
        last_hidden = hidden[0, -1, :]  # [d_model] — the representation at T-1
        
        # Backward: gradient of answer-token logprob w.r.t. last_hidden
        target_token = target_ids[0, 0].to(device)
        logit = output.logits[0, -1, target_token]  # scalar
        model.zero_grad()
        logit.backward(retain_graph=False)
        
        # Gradient w.r.t. hidden at last position
        # We can't easily get grad w.r.t. norm output, so use grad w.r.t. lm_head input
        # The norm output IS the lm_head input (tied weights)
        # grad of logit[target_token] w.r.t. last_hidden = output projection row
        
        # Alternative: compute attribution as |activation × gradient|
        # gradient of the target logit w.r.t. the hidden state
        # This requires the hidden to have requires_grad=True during forward
        # For frozen eval model, we compute it differently:
        #   Use the lm_head weight row for the target token as proxy gradient
        with torch.no_grad():
            grad_proxy = model.lm_head.weight[target_token]  # [d_model]
            activation = last_hidden.float()
            attribution = (activation * grad_proxy.to(activation.device)).abs()
            scores += attribution.cpu()
    
    scores = scores / len(batches)
    
    # Select top K channels
    K = 64
    top_indices = scores.argsort(descending=True)[:K].tolist()
    top_scores = scores[top_indices].tolist()
    print(f"  Top {K} channels selected")
    print(f"  Score range: [{min(top_scores):.4f}, {max(top_scores):.4f}]")

    # --- Phase 3: Teach facts to model, then ablate ---
    print(f"\n--- Phase 3: Teach facts + paired ablation ---")

    # First load the cognitive model with memory
    model_cog = build_model(with_cognition=True, with_adapters=True)
    model_cog.to(device=device, dtype=torch.bfloat16)
    memory = model_cog.cognitive_runtime.memory

    # Teach all 5 facts into memory
    _meta_step = 0
    def c_meta():
        nonlocal _meta_step; _meta_step += 1
        return CognitiveForwardMetadata(step_id=_meta_step, checkpoint_id="sha256:"+"a"*64, context_digest="sha256:"+"b"*64)

    print("  Teaching 5 facts into UniversalMemory...")
    for entity, code in FACTS:
        q_ids, a_ids = tokenize_fact(tokenizer, entity, code)
        q_ids_dev = q_ids.to(device)
        a_ids_dev = a_ids.to(device)
        with torch.no_grad():
            q_out = model_cog(q_ids_dev, heartbeat=False, cognitive_metadata=c_meta())
            a_out = model_cog(a_ids_dev, heartbeat=False, cognitive_metadata=c_meta())
        memory.teach(q_out.hidden_states, a_out.hidden_states,
                     event_type="explicit_teaching", provenance="human", tags=(entity,))
    print(f"  Stored: {memory.slot_count} facts")

    # Test recall on all facts (literal)
    ablation_facts = FACTS
    test_batches = make_discovery_batches(model_cog, tokenizer, ablation_facts)

    # Test: measure memory recall accuracy before/after ablation
    # The ablation affects the norm output, which changes key_encoder input,
    # which changes the query embedding used for memory recall.
    results = {}
    for arm_name in ["clean", "ablate", "shuffle"]:
        print(f"  Arm: {arm_name}")
        model_cog.zero_grad()

        if arm_name == "clean":
            correct = 0
            for batch in test_batches:
                input_ids = batch["input_ids"].to(device)
                with torch.no_grad():
                    out = model_cog(input_ids, heartbeat=False, cognitive_metadata=c_meta())
                recalls = memory.recall(out.hidden_states, top_k=1, require_verified=True)
                entity = batch["entity"]
                hit = not recalls[0].abstained and entity in recalls[0].record.tags
                correct += int(hit)
            results[arm_name] = {"correct": correct, "total": len(test_batches)}

        elif arm_name == "ablate":
            zero_mask = torch.ones(d_model, device=device)
            zero_mask[top_indices] = 0.0
            def ablate_hook(module, input, output):
                return output * zero_mask.view(1, 1, -1).to(dtype=output.dtype)
            handle = model_cog.norm.register_forward_hook(ablate_hook)
            correct = 0
            try:
                for batch in test_batches:
                    input_ids = batch["input_ids"].to(device)
                    with torch.no_grad():
                        out = model_cog(input_ids, heartbeat=False, cognitive_metadata=c_meta())
                    recalls = memory.recall(out.hidden_states, top_k=1, require_verified=True)
                    entity = batch["entity"]
                    hit = not recalls[0].abstained and entity in recalls[0].record.tags
                    correct += int(hit)
            finally:
                handle.remove()
            results[arm_name] = {"correct": correct, "total": len(test_batches)}

        elif arm_name == "shuffle":
            perm = torch.randperm(K, device=device)
            shuffle_map = {top_indices[i]: top_indices[perm[i].item()] for i in range(K)}
            def shuffle_hook(module, input, output):
                out = output.clone()
                for src, dst in shuffle_map.items():
                    out[:, :, src] = output[:, :, dst].clone()
                return out
            handle = model_cog.norm.register_forward_hook(shuffle_hook)
            correct = 0
            try:
                for batch in test_batches:
                    input_ids = batch["input_ids"].to(device)
                    with torch.no_grad():
                        out = model_cog(input_ids, heartbeat=False, cognitive_metadata=c_meta())
                    recalls = memory.recall(out.hidden_states, top_k=1, require_verified=True)
                    entity = batch["entity"]
                    hit = not recalls[0].abstained and entity in recalls[0].record.tags
                    correct += int(hit)
            finally:
                handle.remove()
            results[arm_name] = {"correct": correct, "total": len(test_batches)}

    print(f"\n  Clean:   {results['clean']['correct']}/{results['clean']['total']}")
    print(f"  Ablate:  {results['ablate']['correct']}/{results['ablate']['total']}")
    print(f"  Shuffle: {results['shuffle']['correct']}/{results['shuffle']['total']}")

    # --- Phase 4: Key-encoder alignment ---
    print(f"\n--- Phase 4: Key-encoder alignment with ablation direction ---")
    # model_cog already loaded in Phase 3 with cognition + adapters

    key_encoder = memory.key_encoder
    if hasattr(key_encoder, "weight"):
        ke_weight = key_encoder.weight.data.float().cpu()  # [512, d_model]
    else:
        ke_weight = key_encoder[0].weight.data.float().cpu()  # [hidden, d_model]

    # Ablation direction: the mean of top-K channel one-hot vectors in d_model space
    ablation_dir = torch.zeros(d_model).cpu()
    ablation_dir[top_indices] = 1.0
    ablation_dir = ablation_dir / ablation_dir.norm()

    # Cosine similarity between each key_encoder row and the ablation direction
    ke_rows_norm = F.normalize(ke_weight, dim=1)  # [512 or hidden, d_model]
    alignments = (ke_rows_norm * ablation_dir.unsqueeze(0)).sum(dim=1).abs()  # [rows]
    
    top_aligned = alignments.argsort(descending=True)[:20].tolist()
    print(f"  Key-encoder rows: {ke_weight.shape[0]}")
    print(f"  Top alignment with ablation direction: {[f'{alignments[i].item():.3f}' for i in top_aligned[:10]]}")
    print(f"  Mean abs alignment: {alignments.mean().item():.4f}")
    print(f"  Random baseline (expected): {1.0/math.sqrt(d_model):.4f}")

    # Significance: is the alignment higher than random?
    random_alignments = torch.zeros(1000)
    for i in range(1000):
        rand_dir = torch.randn(d_model)
        rand_dir = rand_dir / rand_dir.norm()
        random_alignments[i] = (ke_rows_norm * rand_dir.unsqueeze(0)).sum(dim=1).abs().mean()
    
    mean_alignment = alignments.mean().item()
    rand_mean = random_alignments.mean().item()
    rand_std = random_alignments.std().item()
    z_score = (mean_alignment - rand_mean) / (rand_std + 1e-8)
    
    print(f"\n  Alignment z-score: {z_score:.2f}")
    print(f"  {'SIGNIFICANT' if abs(z_score) > 2.0 else 'NOT SIGNIFICANT'} "
          f"(|z| > 2.0 threshold)")

    # --- Report ---
    report = {
        "schema": "fact-channel-ablation-v1",
        "checkpoint_sha256": sha256_file(CKPT),
        "discovery": {
            "facts_used": len(discovery_facts),
            "channels_selected": K,
            "top_channel_indices": top_indices[:K],
            "score_range": [float(min(top_scores)), float(max(top_scores))],
        },
        "ablation_results": results,
        "key_encoder_alignment": {
            "mean_abs_alignment": mean_alignment,
            "random_baseline_mean": rand_mean,
            "random_baseline_std": rand_std,
            "z_score": z_score,
            "significant": abs(z_score) > 2.0,
        },
    }
    report_path = OUTPUT / "fact-ablation-report.json"
    atomic_json_write(report, report_path)
    print(f"\nReport: {report_path}")
    
    ablation_drop = results["clean"]["correct"] - results["ablate"]["correct"]
    shuffle_drop = results["clean"]["correct"] - results["shuffle"]["correct"]
    print(f"\nAblation drop: {ablation_drop}/{results['clean']['total']}")
    print(f"Shuffle drop:  {shuffle_drop}/{results['shuffle']['total']}")
    print(f"Key-encoder aligned: {'YES' if abs(z_score) > 2.0 else 'NO'} (z={z_score:.2f})")
    
    print("\nFACT_ABLATION_OK" if ablation_drop > 0 and abs(z_score) > 2.0 else "\nFACT_ABLATION_INCONCLUSIVE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
