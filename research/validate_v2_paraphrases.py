import os, gc, json, torch
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from dataclasses import replace
from pathlib import Path
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.cognition import CognitiveForwardMetadata
from transformers import AutoTokenizer
from f51_darwin.transplant_16b.cli import DEFAULT_SOURCE_ROOT

ROOT = Path("workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1")
ckpt = ROOT / "organism_cycle_000.pt"
ADAPTERS = Path("workspace/runtime/memory-training/trained-adapters-v2.pt")

print("Loading model with v2 adapters...")
payload = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
config = DarwinXConfig.from_mapping(payload["config"])
cog = replace(config, cognitive_architecture_version="three_organs_v1",
              cognitive_shadow_enabled=True, cognitive_pulse_enabled=True)
prev = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
model = DarwinXModel(cog)
torch.set_default_dtype(prev)
model.load_state_dict(payload["model_state_dict"], strict=False)

trained = torch.load(ADAPTERS, map_location="cpu", weights_only=False)
model.cognitive_runtime.memory.key_encoder.load_state_dict(trained["key_encoder"])
model.cognitive_runtime.memory.value_encoder.load_state_dict(trained["value_encoder"])
model.cognitive_runtime.memory.readout.load_state_dict(trained["readout"])
model.eval()

tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_SOURCE_ROOT), local_files_only=True)
runtime = model.cognitive_runtime
runtime.set_active()
memory = runtime.memory

SYSTEM = "Voce e um assistente preciso. Responda com o codigo de seis digitos."
_step = 0
def meta():
    global _step; _step += 1
    return CognitiveForwardMetadata(step_id=_step, checkpoint_id="sha256:"+"a"*64, context_digest="sha256:"+"b"*64)

# 10 facts with 10 paraphrases each
FACTS = [
    ("arquivo-baleia", "457291"), ("chave-golfinho", "185034"),
    ("documento-falcao", "628410"), ("arquivo-tartaruga", "372819"),
    ("documento-lontra", "914756"), ("arquivo-gaviao", "246813"),
    ("chave-morcego", "509317"), ("registro-coruja", "731592"),
    ("cartao-pelicano", "862045"), ("pasta-albatros", "193478"),
]

PARAPHRASES = [
    "Qual e o codigo atribuido a {e}?",
    "Informe somente o codigo cadastrado para {e}.",
    "No catalogo secreto, que numero identifica {e}?",
    "Nao consigo lembrar o codigo de {e}. Qual e?",
    "Me diga o codigo secreto do {e}.",
    "Qual o numero de identificacao do {e}?",
    "Preciso do codigo associado ao {e}. Informe.",
    "O {e} esta registrado sob qual codigo?",
    "Consulte o codigo do {e} no sistema.",
    "Que codigo foi designado para o {e}?",
]

def encode_q(entity):
    chat = [{"role":"system","content":SYSTEM},{"role":"user","content":f"Qual e o codigo atribuido a {entity}?"}]
    fmt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(fmt, add_special_tokens=False, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, heartbeat=False, cognitive_metadata=meta())
    return out.hidden_states

# Teach all 10 facts (literal)
print("\nTeaching 10 facts...")
for entity, code in FACTS:
    q_hidden = encode_q(entity)
    a_text = f"O codigo de {entity} e {code}."
    a_chat = [{"role":"system","content":SYSTEM},{"role":"assistant","content":a_text}]
    a_fmt = tokenizer.apply_chat_template(a_chat, tokenize=False, add_generation_prompt=False)
    a_ids = tokenizer(a_fmt, add_special_tokens=False, return_tensors="pt").input_ids
    with torch.no_grad():
        a_out = model(a_ids, heartbeat=False, cognitive_metadata=meta())
    memory.teach(q_hidden, a_out.hidden_states, event_type="explicit_teaching", provenance="human", tags=(entity,))
print(f"  {memory.slot_count} facts stored")

# Test: literal vs paraphrases
print("\n=== LITERAL vs PARAPHRASE RECALL ===")
for entity, code in FACTS:
    literal_q = encode_q(entity)
    literal_r = memory.recall(literal_q, top_k=1, require_verified=True)[0]
    lit_score = literal_r.score
    lit_hit = not literal_r.abstained and entity in literal_r.record.tags

    para_hits = 0
    para_scores = []
    for p_template in PARAPHRASES[1:]:  # skip first (literal)
        p_text = p_template.format(e=entity)
        p_chat = [{"role":"system","content":SYSTEM},{"role":"user","content":p_text}]
        p_fmt = tokenizer.apply_chat_template(p_chat, tokenize=False, add_generation_prompt=True)
        p_ids = tokenizer(p_fmt, add_special_tokens=False, return_tensors="pt").input_ids
        with torch.no_grad():
            p_out = model(p_ids, heartbeat=False, cognitive_metadata=meta())
        p_r = memory.recall(p_out.hidden_states, top_k=1, require_verified=True)[0]
        para_scores.append(p_r.score)
        if not p_r.abstained and entity in p_r.record.tags:
            para_hits += 1

    min_para = min(para_scores) if para_scores else 0
    print(f"  {entity}: literal={lit_score:.3f} ({'HIT' if lit_hit else 'MISS'})  para_hits={para_hits}/9  min_para_score={min_para:.3f}")

# Summary
total_literal = 0
total_para = 0
below_threshold = 0
for entity, code in FACTS:
    literal_q = encode_q(entity)
    literal_r = memory.recall(literal_q, top_k=1, require_verified=True)[0]
    if not literal_r.abstained and entity in literal_r.record.tags:
        total_literal += 1
    for p_template in PARAPHRASES[1:]:
        p_text = p_template.format(e=entity)
        p_chat = [{"role":"system","content":SYSTEM},{"role":"user","content":p_text}]
        p_fmt = tokenizer.apply_chat_template(p_chat, tokenize=False, add_generation_prompt=True)
        p_ids = tokenizer(p_fmt, add_special_tokens=False, return_tensors="pt").input_ids
        with torch.no_grad():
            p_out = model(p_ids, heartbeat=False, cognitive_metadata=meta())
        p_r = memory.recall(p_out.hidden_states, top_k=1, require_verified=True)[0]
        if not p_r.abstained and entity in p_r.record.tags:
            total_para += 1
        elif p_r.abstained:
            below_threshold += 1

print(f"\n=== SUMMARY ===")
print(f"  Literal: {total_literal}/{len(FACTS)}")
print(f"  Paraphrase: {total_para}/{len(FACTS)*9}")
print(f"  Abstained (below threshold): {below_threshold}/{len(FACTS)*9}")
