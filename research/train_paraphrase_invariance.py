#!/usr/bin/env python3
"""Train key_encoder phrasing invariance using PAWS-X Portuguese + fact paraphrases.

Fixes the paraphrase recall gap: literal=100%, paraphrase=70%.
Adds two training signals:
  1. PAWS-X pt pairs → encoder learns that different phrasings = same meaning
  2. Synthetic fact paraphrases → encoder learns fact-specific phrasing invariance
"""

from __future__ import annotations

import gc, json, math, os, random, time
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "0"  # allow dataset download

import torch
import torch.nn.functional as F

from f51_darwin.cognition import CognitiveForwardMetadata
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.hashing import sha256_file, atomic_json_write
from transformers import AutoTokenizer
from f51_darwin.transplant_16b.cli import DEFAULT_SOURCE_ROOT

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt"
OUTPUT = ROOT / "workspace/runtime/memory-training"
SYSTEM = "Voce e um assistente preciso. Responda com o codigo de seis digitos."

# ── Fact definitions ────────────────────────────────────────────────────────

FACTS = [
    ("arquivo-baleia", "457291"),
    ("chave-golfinho", "185034"),
    ("documento-falcao", "628410"),
    ("arquivo-tartaruga", "372819"),
    ("documento-lontra", "914756"),
    ("arquivo-gaviao", "246813"),
    ("chave-morcego", "509317"),
    ("registro-coruja", "731592"),
    ("cartao-pelicano", "862045"),
    ("pasta-albatros", "193478"),
]

# ── Synthetic paraphrases per fact ──────────────────────────────────────────

def make_paraphrases(entity: str, code: str) -> list[str]:
    """Generate diverse phrasings for a fact question."""
    return [
        f"Qual e o codigo atribuido a {entity}?",
        f"Informe somente o codigo cadastrado para {entity}.",
        f"No catalogo secreto, que numero identifica {entity}?",
        f"Nao consigo lembrar o codigo de {entity}. Qual e?",
        f"Me diga o codigo secreto do {entity}.",
        f"Qual o numero de identificacao do {entity}?",
        f"Preciso do codigo associado ao {entity}. Informe.",
        f"O {entity} esta registrado sob qual codigo?",
        f"Consulte o codigo do {entity} no sistema.",
        f"Que codigo foi designado para o {entity}?",
    ]

# ── Model building ──────────────────────────────────────────────────────────

_step = 0
def _meta() -> CognitiveForwardMetadata:
    global _step; _step += 1
    return CognitiveForwardMetadata(step_id=_step, checkpoint_id="sha256:"+"a"*64, context_digest="sha256:"+"b"*64)


def build_model() -> DarwinXModel:
    payload = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    config = DarwinXConfig.from_mapping(payload["config"])
    cog = replace(config, cognitive_architecture_version="three_organs_v1",
                  cognitive_shadow_enabled=True, cognitive_pulse_enabled=True)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = DarwinXModel(cog)
    torch.set_default_dtype(prev)
    model.load_state_dict(payload["model_state_dict"], strict=False)

    # Load previously trained adapters
    adapters_path = OUTPUT / "trained-adapters.pt"
    if adapters_path.exists():
        trained = torch.load(adapters_path, map_location="cpu", weights_only=False)
        model.cognitive_runtime.memory.key_encoder.load_state_dict(trained["key_encoder"])
        model.cognitive_runtime.memory.value_encoder.load_state_dict(trained["value_encoder"])
        model.cognitive_runtime.memory.readout.load_state_dict(trained["readout"])
        print("  Loaded previously trained adapters")

    # Place on GPU BEFORE encoding (much faster forward passes)
    model.eval()
    model.to(dtype=torch.bfloat16)
    if torch.cuda.device_count() >= 2:
        model.enable_dual_gpu(gpu0=0, gpu1=1)
        print("  Placed on dual GPU")
    elif torch.cuda.is_available():
        model.to(device=torch.device("cuda:0"))
        print("  Placed on cuda:0")

    return model


def encode_question(model, tokenizer, question: str) -> torch.Tensor:
    chat = [{"role":"system","content":SYSTEM},{"role":"user","content":question}]
    fmt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(fmt, add_special_tokens=False, return_tensors="pt").input_ids
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(ids.to(device), heartbeat=False, cognitive_metadata=_meta())
    return out.hidden_states.clone().cpu()


def encode_answer(model, tokenizer, entity: str, code: str) -> torch.Tensor:
    text = f"O codigo de {entity} e {code}."
    chat = [{"role":"system","content":SYSTEM},{"role":"assistant","content":text}]
    fmt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    ids = tokenizer(fmt, add_special_tokens=False, return_tensors="pt").input_ids
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(ids.to(device), heartbeat=False, cognitive_metadata=_meta())
    return out.hidden_states.clone().cpu()


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("Paraphrase-Invariance Training")
    print("=" * 60)

    # --- Load PAWS-X Portuguese ---
    print("\n--- Loading PAWS-X Portuguese ---")
    from datasets import load_dataset
    try:
        paws = load_dataset("paws-x", "pt", split="train", trust_remote_code=False)
        # Filter paraphrase pairs (label=1)
        pos_pairs = [(row["sentence1"], row["sentence2"]) for row in paws if row["label"] == 1]
        print(f"  PAWS-X positive pairs: {len(pos_pairs)}")
    except Exception as e:
        print(f"  WARNING: PAWS-X load failed ({e}), using fact paraphrases only")
        pos_pairs = []

    # --- Build model ---
    print("\n--- Building model ---")
    model = build_model()
    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_SOURCE_ROOT), local_files_only=True)
    d_model = model.config.d_model
    memory = model.cognitive_runtime.memory

    # --- Encode fact questions (store raw hidden states, NOT embeddings) ---
    print("\n--- Encoding fact paraphrases (raw hidden states) ---")
    # Store raw backbone hidden states: [num_paraphrases, 1, T, d_model]
    fact_paraphrase_hiddens = []   # list of list of tensors
    fact_answer_hiddens = []       # list of tensors
    for entity, code in FACTS:
        para_qs = make_paraphrases(entity, code)
        q_hiddens = []
        for q in para_qs:
            q_hiddens.append(encode_question(model, tokenizer, q))
        fact_paraphrase_hiddens.append(q_hiddens)
        a_hidden = encode_answer(model, tokenizer, entity, code)
        fact_answer_hiddens.append(a_hidden)
    N_PARAPHRASES = len(fact_paraphrase_hiddens[0])
    print(f"  Encoded {len(FACTS)} facts x {N_PARAPHRASES} paraphrases")

    # --- Encode PAWS-X pairs (store raw hidden states) ---
    paws_hidden_pairs = []
    if pos_pairs:
        print(f"\n--- Encoding PAWS-X pairs (sample of 2000) ---")
        sample = random.sample(pos_pairs, min(2000, len(pos_pairs)))
        for i, (s1, s2) in enumerate(sample):
            h1 = encode_question(model, tokenizer, s1)
            h2 = encode_question(model, tokenizer, s2)
            paws_hidden_pairs.append((h1, h2))
            if (i + 1) % 500 == 0:
                print(f"  encoded {i+1}/{len(sample)} PAWS-X pairs")
        print(f"  Encoded {len(paws_hidden_pairs)} PAWS-X pairs")

    # --- Train ---
    print(f"\n--- Training key_encoder for phrasing invariance ---")
    trainable = list(memory.key_encoder.parameters())
    opt = torch.optim.AdamW(trainable, lr=5e-5)
    steps = 200
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    # Save literal anchor embeddings (frozen snapshot of pre-training key_encoder)
    literal_anchors = []
    for q_hiddens in fact_paraphrase_hiddens:
        with torch.no_grad():
            anchor = memory.encode_query(q_hiddens[0].to(device=next(memory.key_encoder.parameters()).device, dtype=torch.bfloat16))
        literal_anchors.append(anchor.squeeze(0).detach().clone())

    log = []
    best_loss = float("inf")

    device = next(memory.key_encoder.parameters()).device
    for step in range(steps):
        opt.zero_grad()

        # Re-encode all fact paraphrases through current key_encoder
        fp_embs = []
        for q_hiddens in fact_paraphrase_hiddens:
            embs = []
            for q_hidden in q_hiddens:
                q_dev = q_hidden.to(device=device, dtype=torch.bfloat16)
                emb = memory.encode_query(q_dev)
                embs.append(emb.squeeze(0))
            fp_embs.append(torch.stack(embs).to(device))

        # --- Loss 1: Fact paraphrase consistency ---
        fact_consistency = torch.tensor(0.0)
        for fp in fp_embs:
            sim = F.cosine_similarity(fp.unsqueeze(1), fp.unsqueeze(0), dim=-1)
            fact_consistency = fact_consistency + (1.0 - sim).mean()
        fact_consistency = fact_consistency / len(fp_embs)

        # --- Loss 2: Inter-fact discrimination ---
        inter_fact = torch.tensor(0.0)
        centroids = [fp.mean(dim=0) for fp in fp_embs]
        for i in range(len(centroids)):
            for j in range(len(centroids)):
                if i != j:
                    sim_ij = F.cosine_similarity(centroids[i].unsqueeze(0), centroids[j].unsqueeze(0), dim=-1)
                    inter_fact = inter_fact + F.relu(sim_ij - 0.3)
        n_pairs = len(centroids) * (len(centroids) - 1)
        inter_fact = inter_fact / max(1, n_pairs)

        # --- Loss 3: PAWS-X contrastive (sample 16 pairs per step) ---
        paws_loss = torch.tensor(0.0)
        if paws_hidden_pairs:
            n_sample = min(16, len(paws_hidden_pairs))
            indices = random.sample(range(len(paws_hidden_pairs)), n_sample)
            for idx in indices:
                h1, h2 = paws_hidden_pairs[idx]
                e1 = memory.encode_query(h1).squeeze(0)
                e2 = memory.encode_query(h2).squeeze(0)
                paws_loss = paws_loss + (1.0 - F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0), dim=-1))
            paws_loss = paws_loss / n_sample

        # --- Loss 4: Margin (paraphrase centroid vs nearest other fact) ---
        margin_loss = torch.tensor(0.0)
        for i, fp in enumerate(fp_embs):
            ci = fp.mean(dim=0)
            score_self = F.cosine_similarity(ci.unsqueeze(0), fp, dim=-1).max()
            best_other = -1.0
            for j in range(len(fp_embs)):
                if i != j:
                    cj = fp_embs[j].mean(dim=0)
                    so = float(F.cosine_similarity(ci.unsqueeze(0), cj.unsqueeze(0), dim=-1))
                    best_other = max(best_other, so)
            margin = score_self - best_other
            margin_loss = margin_loss + F.relu(0.15 - margin)

        # --- Loss 5: Literal anchor ---
        anchor_loss = torch.tensor(0.0, device=device)
        for i, q_hiddens in enumerate(fact_paraphrase_hiddens):
            lit_emb = memory.encode_query(q_hiddens[0].to(device=device, dtype=torch.bfloat16)).squeeze(0)
            anchor_loss = anchor_loss + (1.0 - F.cosine_similarity(lit_emb.unsqueeze(0), literal_anchors[i].unsqueeze(0).to(device), dim=-1))
        anchor_loss = anchor_loss / len(fact_paraphrase_hiddens)

        total = 0.3 * fact_consistency + 1.5 * inter_fact + 0.3 * paws_loss + 1.0 * margin_loss + 0.5 * anchor_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        opt.step()
        scheduler.step()

        if step % 40 == 0 or step == steps - 1:
            entry = {
                "step": step, "loss": float(total.detach()),
                "fact_consistency": float(fact_consistency.detach()),
                "inter_fact": float(inter_fact.detach()),
                "paws": float(paws_loss.detach()),
                "margin": float(margin_loss.detach()),
                "anchor": float(anchor_loss.detach()),
                "lr": float(scheduler.get_last_lr()[0]),
            }
            log.append(entry)
            print(f"  step {step:3d}  loss={float(total.detach()):.4f}  consistency={float(fact_consistency.detach()):.4f}  margin={float(margin_loss.detach()):.4f}")
            if float(total.detach()) < best_loss:
                best_loss = float(total.detach())

    # --- Save ---
    print(f"\n--- Saving (best_loss={best_loss:.4f}) ---")
    adapter_state = {
        "key_encoder": {k: v.cpu() for k, v in memory.key_encoder.state_dict().items()},
        "value_encoder": {k: v.cpu() for k, v in memory.value_encoder.state_dict().items()},
        "readout": {k: v.cpu() for k, v in memory.readout.state_dict().items()},
    }
    torch.save(adapter_state, OUTPUT / "trained-adapters-v2.pt")

    report = {
        "schema": "memory-paraphrase-training-v1",
        "steps": steps,
        "best_loss": best_loss,
        "paws_pairs_used": len(paws_hidden_pairs),
        "facts_trained": len(FACTS),
        "paraphrases_per_fact": N_PARAPHRASES,
        "log": log,
    }
    atomic_json_write(report, OUTPUT / "paraphrase-training-log.json")
    print(f"  Saved to {OUTPUT}/trained-adapters-v2.pt")
    print("PARAPHRASE_TRAINING_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
