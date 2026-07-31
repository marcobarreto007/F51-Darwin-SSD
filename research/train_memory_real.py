#!/usr/bin/env python3
"""Train UniversalMemory adapters on real 1.7B hidden states (v2).

v1 defect: the training objective was cosine(encode_query(q), encode_value(a))
-- query-to-value space -- while `UniversalMemory.recall` scores
cosine(encode_query(q_new), key_embedding) where key_embedding is
encode_query(q_taught) -- query-to-query space.  The optimized objective was
never the measured one, so nothing taught the key encoder that paraphrases of
the same fact belong together.  Paraphrase recall was 0/5.

v2 trains the key encoder directly in query-to-query space: paraphrases of the
same fact are positives, other facts in the batch are negatives.  A secondary
query-to-value term keeps the value space aligned so the readout path retains
signal.  The recall threshold is then calibrated empirically against held-out
positives and never-taught negatives instead of being left at the default 0.6,
which sat below the observed off-diagonal similarity and made abstention
impossible.

Backbone stays frozen throughout; only key_encoder, value_encoder and readout
receive gradients.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.functional as F

from f51_darwin.cognition import CognitiveForwardMetadata
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.hashing import atomic_json_write, sha256_file

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt"
OUTPUT_DIR = ROOT / "workspace/runtime/memory-training"
SYSTEM = "Voce e um assistente preciso. Responda com o codigo de seis digitos."

SEED = 20260729

# Absolute-cosine targets. The calibrated threshold lands between them.
M_POS = 0.85
M_NEG = 0.25
MARGIN_WEIGHT = 2.0

# Hidden width of the key projector. None keeps the original single Linear.
KEY_ENCODER_HIDDEN = int(os.environ.get("DARWIN_KEY_ENCODER_HIDDEN", "0")) or None
OUTPUT_SUFFIX = os.environ.get("DARWIN_ADAPTER_SUFFIX", "")

# Held-out phrasings are never seen in training, so paraphrase generalisation
# is measured on wording the encoder has no exposure to.  Training coverage has
# to be structurally varied (interrogative, imperative, elliptical, indirect),
# not just lexically shuffled -- with only six near-identical templates the
# held-out positives landed at cosine 0.72 against 0.91 on seen phrasings.
TRAIN_TEMPLATES = [
    "Qual e o codigo atribuido a {e}?",
    "Me diga qual codigo pertence a {e}.",
    "Qual o codigo de {e}?",
    "Voce lembra do codigo de {e}?",
    "Informe o codigo registrado para {e}.",
    "{e} tem qual codigo?",
    "Recupere o codigo de {e}.",
    "O codigo de {e} e qual mesmo?",
    "Consulte o registro e diga o codigo de {e}.",
    "Sobre {e}: qual o codigo?",
    "Estou procurando o codigo associado a {e}.",
    "Poderia me informar o codigo de {e}?",
    "Diga os seis digitos de {e}.",
    "Qual identificador numerico foi dado a {e}?",
    "Lembra o que foi registrado para {e}?",
    "Qual foi o codigo definido para {e}?",
]
HELDOUT_TEMPLATES = [
    "Preciso do codigo da {e}.",
    "Qual numero de seis digitos identifica {e}?",
    "Nao consigo lembrar o codigo de {e}, voce sabe?",
    "Me passa o registro numerico de {e}.",
]

# In the raw hidden space the dominant axis is phrasing, not entity: a 1-NN
# probe over centred hiddens never picks another phrasing of the same entity
# (accuracy 0.000 for last-token, mean, max and last-8 pooling).  The key
# encoder therefore has to learn to suppress the phrasing direction, and that
# generalises with entity count far more than with step count -- hence a pool
# wide enough to train on hundreds of distinct entities.
_NOUNS = [
    "aurora", "vega", "cronos", "delta", "orion", "lyra", "atlas", "nimbus",
    "quasar", "helios", "tundra", "zenite", "pantano", "farol", "bussola",
    "granito", "estuario", "meridiano", "obsidiana", "cascata", "planalto",
    "ancora", "bastiao", "corrente", "duna", "enxame", "fenda", "geleira",
    "arrecife", "boreal", "caatinga", "dilema", "escarpa", "fronteira",
    "gaivota", "horizonte", "ilhota", "jangada", "lanterna", "manguezal",
    "nevoeiro", "oceano", "pampa", "quilombo", "riacho", "savana", "torrente",
    "urutau", "varzea", "xisto", "azimute", "brisa", "campina", "desfiladeiro",
    "eclipse", "furacao", "gruta", "hematita", "iceberg", "jazida", "kelvin",
    "litoral", "monsao", "nascente", "orvalho", "penhasco", "quartzo",
    "ravina", "sertao", "trilha", "vulcao",
]
_TRAIN_PREFIXES = ["projeto", "sistema", "unidade", "nucleo", "setor", "modulo"]
# Reserved for calibration only. Thresholds calibrated on entities whose
# prefix token was seen in training transfer optimistically: deployment
# entities carry novel surface forms, scores land lower, and a threshold fitted
# on the easier distribution then rejects real hits. Holding prefixes out makes
# the calibration distribution match the one the organ is judged on.
_CALIB_PREFIXES = ["posto", "celula", "arquivo", "consorcio"]


def build_entities(
    count: int, rng: random.Random, prefixes: list[str], taken: set[str],
) -> list[str]:
    """Distinct entity names drawn from the given prefix set."""
    pool = [f"{p}-{n}" for p in prefixes for n in _NOUNS if f"{p}-{n}" not in taken]
    rng.shuffle(pool)
    if count > len(pool):
        raise ValueError(f"requested {count} entities, pool holds {len(pool)}")
    chosen = pool[:count]
    taken.update(chosen)
    return chosen


def build_facts(entities: list[str], rng: random.Random) -> list[dict[str, Any]]:
    facts = []
    used: set[str] = set()
    for entity in entities:
        while True:
            code = f"{rng.randint(100000, 999999):06d}"
            if code not in used:
                used.add(code)
                break
        facts.append({
            "entity": entity,
            "code": code,
            "answer_text": f"O codigo de {entity} e {code}.",
        })
    return facts


class Encoder:
    """Runs the frozen backbone and returns last-layer hidden states."""

    def __init__(self, model: DarwinXModel, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._step = 0

    def _metadata(self) -> CognitiveForwardMetadata:
        self._step += 1
        return CognitiveForwardMetadata(
            step_id=self._step,
            checkpoint_id="sha256:" + "a" * 64,
            context_digest="sha256:" + "b" * 64,
        )

    def __call__(self, text: str, role: str) -> torch.Tensor:
        chat = [
            {"role": "system", "content": SYSTEM},
            {"role": role, "content": text},
        ]
        formatted = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=(role == "user"),
        )
        ids = self.tokenizer(
            formatted, add_special_tokens=False, return_tensors="pt",
        ).input_ids.to(self.model.token_embedding.weight.device)
        with torch.inference_mode():
            out = self.model(ids, heartbeat=False, cognitive_metadata=self._metadata())
        hidden = out.hidden_states
        if hidden is None:
            raise RuntimeError(
                "model returned hidden_states=None; refusing to train on a "
                "random-noise fallback"
            )
        return hidden.detach().to(dtype=torch.float32, device="cpu").clone()


def info_nce(anchor: torch.Tensor, positive: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE with in-batch negatives."""
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    logits = (a @ p.t()) / temperature
    labels = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    *,
    m_pos: float,
    m_neg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hinge on absolute cosine values.

    InfoNCE only shapes the *ranking* of similarities, so an unrelated query
    can still sit at cosine 0.86 against the whole store -- which is what made
    abstention impossible.  The threshold operates on absolute cosine, so the
    absolute scale has to be trained too: same-fact pairs are pushed above
    m_pos, different-fact pairs below m_neg.
    """
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    sim = a @ p.t()
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    pos = sim[eye]
    neg = sim[~eye]
    loss = F.relu(m_pos - pos).mean() + F.relu(neg - m_neg).mean()
    return loss, pos.mean().detach(), neg.mean().detach()


def calibrate_thresholds(
    positives: list[dict[str, Any]], negatives: list[dict[str, Any]],
) -> dict[str, float]:
    """Jointly pick (recall_threshold, margin_threshold) on the end-to-end task.

    A positive query succeeds only when the organ answers AND returns the
    correct slot; scoring it on cosine alone would credit a confident wrong
    retrieval.  A negative query -- an entity never taught -- succeeds only by
    abstaining.  Each probe carries top-1 score, the top-1/top-2 gap and the
    retrieved id, so both gates sweep against the metric that matters.
    """
    def accepts(p: dict[str, Any], t: float, m: float, hi: float) -> bool:
        return p["score"] >= t and (p["margin"] >= m or p["score"] >= hi)

    score_grid = sorted({round(p["score"], 2) for p in positives + negatives})
    margin_grid = [0.0] + sorted(
        {round(p["margin"], 3) for p in positives + negatives if p["margin"] > 0.0}
    )
    # The short-circuit is only meaningful above the negatives' score range.
    hi_floor = max(n["score"] for n in negatives) if negatives else 1.0
    # 2.0 is unreachable for a cosine, so it encodes "short-circuit disabled"
    # while staying a finite, JSON-safe number.
    high_grid = [2.0] + [
        round(v, 2) for v in sorted({round(p["score"], 2) for p in positives})
        if v > hi_floor
    ]

    best = {
        "threshold": 0.6, "margin_threshold": 0.0,
        "high_confidence_threshold": 2.0, "balanced_accuracy": -1.0,
    }
    for t in score_grid:
        for m in margin_grid:
            for hi in high_grid:
                tp = sum(1 for p in positives if p["correct"] and accepts(p, t, m, hi))
                tn = sum(1 for n in negatives if not accepts(n, t, m, hi))
                tpr = tp / max(len(positives), 1)
                tnr = tn / max(len(negatives), 1)
                bal = 0.5 * (tpr + tnr)
                if bal > best["balanced_accuracy"]:
                    best = {
                        "threshold": float(t),
                        "margin_threshold": float(m),
                        "high_confidence_threshold": float(hi),
                        "balanced_accuracy": float(bal),
                        "true_positive_rate": float(tpr),
                        "true_negative_rate": float(tnr),
                    }
    return best


def main() -> int:
    print("=== UniversalMemory Real Training (v2, query-space objective) ===\n")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)

    ckpt_path = Path(CKPT).resolve()
    print(f"Loading: {ckpt_path}")
    ckpt_sha = sha256_file(ckpt_path)
    print(f"SHA-256: {ckpt_sha}")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    base_config = DarwinXConfig.from_mapping(payload["config"])
    d_model = base_config.d_model
    print(f"d_model={d_model}  n_layers={base_config.n_layers}")

    print("Building cognitive model...")
    cognitive_config = replace(
        base_config,
        cognitive_architecture_version="three_organs_v1",
        cognitive_shadow_enabled=True,
        cognitive_pulse_enabled=True,
    )
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = DarwinXModel(cognitive_config)
    torch.set_default_dtype(prev_dtype)
    if KEY_ENCODER_HIDDEN is not None:
        runtime_memory = model.cognitive_runtime.memory
        rebuilt = type(runtime_memory)(
            d_model, key_encoder_hidden=KEY_ENCODER_HIDDEN,
        )
        runtime_memory.key_encoder = rebuilt.key_encoder
        runtime_memory.key_encoder_hidden = KEY_ENCODER_HIDDEN
        print(f"  key_encoder: MLP {d_model}->{KEY_ENCODER_HIDDEN}->512")
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    print(f"  missing: {len(missing)} (cognitive)  unexpected: {len(unexpected)}")

    runtime = model.cognitive_runtime
    if runtime is None:
        print("ERROR: cognitive_runtime is None")
        return 1
    runtime.set_active()
    memory = runtime.memory

    from transformers import AutoTokenizer
    from f51_darwin.transplant_16b.cli import DEFAULT_SOURCE_ROOT
    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_SOURCE_ROOT), local_files_only=True)
    print("Tokenizer loaded")

    print("Placing backbone on dual GPU...")
    model.to(dtype=torch.bfloat16)
    on_gpu = model.enable_dual_gpu(gpu0=0, gpu1=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  dual_gpu={on_gpu}  device={device}")

    # Adapters run in float32: bf16 cosine carries ~8e-3 of noise, which is
    # larger than the margins the calibrated threshold has to resolve.
    memory.key_encoder.to(device=device, dtype=torch.float32)
    memory.value_encoder.to(device=device, dtype=torch.float32)
    memory.readout.to(device=device, dtype=torch.float32)

    n_train, n_cal_pos, n_cal_neg = 220, 40, 60
    taken: set[str] = set()
    train_facts = build_facts(
        build_entities(n_train, rng, _TRAIN_PREFIXES, taken), rng)
    cal_pos_facts = build_facts(
        build_entities(n_cal_pos, rng, _CALIB_PREFIXES, taken), rng)
    cal_neg_facts = build_facts(
        build_entities(n_cal_neg, rng, _CALIB_PREFIXES, taken), rng)
    print(f"Facts: {len(train_facts)} train ({'/'.join(_TRAIN_PREFIXES)}) / "
          f"{len(cal_pos_facts)} calib-positive / {len(cal_neg_facts)} "
          f"never-taught (held-out prefixes: {'/'.join(_CALIB_PREFIXES)})")

    encode = Encoder(model, tokenizer)

    def encode_group(facts: list[dict[str, Any]], templates: list[str], label: str):
        out = []
        total = len(facts) * len(templates)
        done = 0
        for fact in facts:
            qs = [encode(t.format(e=fact["entity"]), "user") for t in templates]
            out.append(qs)
            done += len(templates)
            if done % 48 == 0:
                print(f"  {label}: {done}/{total}")
        return out

    print("Encoding through frozen backbone...")
    train_q = encode_group(train_facts, TRAIN_TEMPLATES, "train-q")
    train_a = [encode(f["answer_text"], "assistant") for f in train_facts]
    print(f"  train answers: {len(train_a)}")

    trainable = (
        list(memory.key_encoder.parameters())
        + list(memory.value_encoder.parameters())
        + list(memory.readout.parameters())
    )
    opt = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.01)
    steps = 4000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    batch_facts = 32
    temperature = 0.07

    print(f"Training {steps} steps (batch={batch_facts} facts)...")
    log_entries: list[dict[str, Any]] = []
    total_loss = torch.tensor(0.0)

    for step in range(steps):
        opt.zero_grad()
        idx = torch.randperm(len(train_facts))[:batch_facts].tolist()

        anchors, positives, values = [], [], []
        for i in idx:
            t_a, t_p = rng.sample(range(len(TRAIN_TEMPLATES)), 2)
            anchors.append(memory.encode_query(train_q[i][t_a].to(device)))
            positives.append(memory.encode_query(train_q[i][t_p].to(device)))
            values.append(memory.encode_value(train_a[i].to(device)))

        anchor = torch.cat(anchors, dim=0)
        positive = torch.cat(positives, dim=0)
        value = torch.cat(values, dim=0)

        # Primary: query-to-query, the space `recall` actually scores in.
        loss_key = info_nce(anchor, positive, temperature)
        # Secondary: keeps the value space aligned for the readout path.
        loss_qv = info_nce(anchor, value, temperature)
        # Absolute-scale shaping so the calibrated threshold has a gap to sit in.
        loss_m, pos_sim, neg_sim = margin_loss(
            anchor, positive, m_pos=M_POS, m_neg=M_NEG,
        )
        total_loss = loss_key + 0.3 * loss_qv + MARGIN_WEIGHT * loss_m

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        opt.step()
        scheduler.step()

        if step % 100 == 0 or step == steps - 1:
            entry = {
                "step": step,
                "loss": float(total_loss.detach()),
                "loss_key": float(loss_key.detach()),
                "loss_qv": float(loss_qv.detach()),
                "loss_margin": float(loss_m.detach()),
                "cos_pos": float(pos_sim),
                "cos_neg": float(neg_sim),
                "lr": float(scheduler.get_last_lr()[0]),
            }
            log_entries.append(entry)
            print(f"  step {step:4d}  loss={entry['loss']:.4f}  "
                  f"key={entry['loss_key']:.4f}  qv={entry['loss_qv']:.4f}  "
                  f"cos+={entry['cos_pos']:.3f}  cos-={entry['cos_neg']:.3f}")

    # ── calibration ─────────────────────────────────────────────────────────
    print("\nCalibrating recall threshold on held-out phrasings...")
    memory._store.clear()
    memory.recall_threshold = -1.0  # never abstain while collecting raw scores

    cal_pos_q_train = encode_group(cal_pos_facts, TRAIN_TEMPLATES[:1], "calib-teach")
    cal_pos_q_held = encode_group(cal_pos_facts, HELDOUT_TEMPLATES, "calib-pos")
    cal_neg_q_held = encode_group(cal_neg_facts, HELDOUT_TEMPLATES, "calib-neg")

    rids = []
    for fact, qs in zip(cal_pos_facts, cal_pos_q_train):
        a_h = encode(fact["answer_text"], "assistant")
        rids.append(memory.teach(
            qs[0].to(device), a_h.to(device),
            event_type="explicit_teaching", provenance="human",
            tags=(fact["entity"],),
        ))

    def probe(q: torch.Tensor, expect_rid: str | None) -> dict[str, Any]:
        r = memory.recall(q.to(device), top_k=2, require_verified=True)[0]
        return {
            "score": r.score,
            "margin": r.margin,
            "correct": expect_rid is not None and r.record.memory_id == expect_rid,
        }

    # Two positive classes, because both occur in use: the same question asked
    # again (deterministic re-encode, score ~1.0) and a rephrasing the encoder
    # has never seen (score spread wide and low). Calibrating on paraphrases
    # alone leaves the high-confidence band unobserved, and the margin gate
    # then rejects exact matches.
    positives_literal = [
        probe(qs[0], rid) for rid, qs in zip(rids, cal_pos_q_train)
    ]
    positives_para = [
        probe(q, rid)
        for rid, qs in zip(rids, cal_pos_q_held)
        for q in qs
    ]
    positives = positives_literal + positives_para
    negatives = [probe(q, None) for qs in cal_neg_q_held for q in qs]

    correct_slot = sum(1 for p in positives if p["correct"])
    print(f"  literal positives n={len(positives_literal)} "
          f"correct={sum(1 for p in positives_literal if p['correct'])}")
    print(f"  paraphrase positives n={len(positives_para)} "
          f"correct={sum(1 for p in positives_para if p['correct'])}")
    n_pos = len(positives)
    print(f"  positives n={n_pos}  correct-slot={correct_slot}/{n_pos}")
    print(f"    score  min={min(p['score'] for p in positives):.4f} "
          f"mean={sum(p['score'] for p in positives)/n_pos:.4f}")
    print(f"    margin min={min(p['margin'] for p in positives):.4f} "
          f"mean={sum(p['margin'] for p in positives)/n_pos:.4f}")
    n_neg = len(negatives)
    print(f"  negatives n={n_neg}")
    print(f"    score  max={max(n['score'] for n in negatives):.4f} "
          f"mean={sum(n['score'] for n in negatives)/n_neg:.4f}")
    print(f"    margin max={max(n['margin'] for n in negatives):.4f} "
          f"mean={sum(n['margin'] for n in negatives)/n_neg:.4f}")

    calib = calibrate_thresholds(positives, negatives)
    memory.recall_threshold = calib["threshold"]
    memory.margin_threshold = calib["margin_threshold"]
    memory.high_confidence_threshold = calib["high_confidence_threshold"]
    print(f"  threshold={calib['threshold']:.4f}  "
          f"margin_threshold={calib['margin_threshold']:.4f}  "
          f"high_confidence={calib['high_confidence_threshold']:.4f}")
    print(f"  balanced_accuracy={calib['balanced_accuracy']:.4f}  "
          f"tpr={calib['true_positive_rate']:.4f}  tnr={calib['true_negative_rate']:.4f}")

    print(f"\nSaving to {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    adapter_state = {
        "key_encoder": {k: v.detach().cpu() for k, v in memory.key_encoder.state_dict().items()},
        "value_encoder": {k: v.detach().cpu() for k, v in memory.value_encoder.state_dict().items()},
        "readout": {k: v.detach().cpu() for k, v in memory.readout.state_dict().items()},
        "recall_threshold": calib["threshold"],
        "margin_threshold": calib["margin_threshold"],
        "high_confidence_threshold": calib["high_confidence_threshold"],
    }
    adapter_state["key_encoder_hidden"] = KEY_ENCODER_HIDDEN
    torch.save(adapter_state, OUTPUT_DIR / f"trained-adapters{OUTPUT_SUFFIX}.pt")
    training_log = {
        "schema": "memory-adapter-training-v2",
        "checkpoint_sha256": ckpt_sha,
        "d_model": d_model,
        "seed": SEED,
        "objective": (
            "query-to-query InfoNCE + 0.3 * query-to-value InfoNCE "
            f"+ {MARGIN_WEIGHT} * absolute-cosine hinge "
            f"(m_pos={M_POS}, m_neg={M_NEG})"
        ),
        "steps": steps,
        "batch_facts": batch_facts,
        "temperature": temperature,
        "train_facts": len(train_facts),
        "train_templates": TRAIN_TEMPLATES,
        "heldout_templates": HELDOUT_TEMPLATES,
        "final_loss": float(total_loss.detach()),
        "calibration": calib,
        "calibration_positive_slot_accuracy": correct_slot / n_pos,
        "log": log_entries,
    }
    training_log["key_encoder_hidden"] = KEY_ENCODER_HIDDEN
    atomic_json_write(training_log, OUTPUT_DIR / f"training-log{OUTPUT_SUFFIX}.json")
    print("MEMORY_TRAINING_V2_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
