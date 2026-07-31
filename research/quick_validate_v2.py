import os, torch
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from dataclasses import replace
from pathlib import Path
from f51_darwin.darwin_x_core.config import DarwinXConfig
from f51_darwin.darwin_x_core.model import DarwinXModel
from f51_darwin.cognition import CognitiveForwardMetadata
from transformers import AutoTokenizer
from f51_darwin.transplant_16b.cli import DEFAULT_SOURCE_ROOT

ckpt = Path("workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt")
ADAPTERS = Path("workspace/runtime/memory-training/trained-adapters-v2.pt")

print("Building model with v2 adapters on GPU...")
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
model.to(dtype=torch.bfloat16)
model.enable_dual_gpu(gpu0=0, gpu1=1)
dev = next(model.parameters()).device
print(f"  Device: {dev}")

tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_SOURCE_ROOT), local_files_only=True)
runtime = model.cognitive_runtime
runtime.set_active()
memory = runtime.memory

SYSTEM = "Voce e um assistente preciso."
_step = 0
def meta():
    global _step; _step += 1
    return CognitiveForwardMetadata(step_id=_step, checkpoint_id="sha256:"+"a"*64, context_digest="sha256:"+"b"*64)

FACTS = [
    ("arquivo-baleia", "457291"),
    ("chave-golfinho", "185034"),
    ("documento-falcao", "628410"),
]

PARAS = [
    "Qual e o codigo atribuido a {e}?",
    "Informe somente o codigo cadastrado para {e}.",
    "Nao consigo lembrar o codigo de {e}. Qual e?",
    "Me diga o codigo secreto do {e}.",
]

print("\nTeaching 3 facts...")
for entity, code in FACTS:
    q_chat = [{"role":"system","content":SYSTEM},{"role":"user","content":f"Qual e o codigo atribuido a {entity}?"}]
    q_fmt = tokenizer.apply_chat_template(q_chat, tokenize=False, add_generation_prompt=True)
    q_ids = tokenizer(q_fmt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        q_out = model(q_ids, heartbeat=False, cognitive_metadata=meta())
    a_chat = [{"role":"system","content":SYSTEM},{"role":"assistant","content":f"O codigo de {entity} e {code}."}]
    a_fmt = tokenizer.apply_chat_template(a_chat, tokenize=False, add_generation_prompt=False)
    a_ids = tokenizer(a_fmt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        a_out = model(a_ids, heartbeat=False, cognitive_metadata=meta())
    memory.teach(q_out.hidden_states, a_out.hidden_states,
                 event_type="explicit_teaching", provenance="human", tags=(entity,))
print(f"  {memory.slot_count} facts stored")

print("\n=== RECALL: Literal + 3 Paraphrases ===")
lit_hits = 0
para_hits = 0
para_total = 0

for entity, code in FACTS:
    for pi, p_template in enumerate(PARAS):
        txt = p_template.format(e=entity)
        chat = [{"role":"system","content":SYSTEM},{"role":"user","content":txt}]
        fmt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(fmt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out = model(ids, heartbeat=False, cognitive_metadata=meta())
        r = memory.recall(out.hidden_states, top_k=1, require_verified=True)[0]
        hit = not r.abstained and entity in r.record.tags
        kind = "LIT" if pi == 0 else "PARA"
        if pi == 0:
            if hit:
                lit_hits += 1
        else:
            para_total += 1
            if hit:
                para_hits += 1
        tag = "HIT" if hit else "MISS"
        print(f"  [{kind}] {entity}: score={r.score:.3f}  abst={r.abstained}  {tag}")

print(f"\nLiteral: {lit_hits}/{len(FACTS)}  Paraphrase: {para_hits}/{para_total}")
if para_hits >= para_total * 0.8:
    print("PARAPHRASE_V2_OK")
else:
    print(f"PARAPHRASE_V2_PARTIAL ({para_hits}/{para_total})")
