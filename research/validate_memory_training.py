import os, gc, torch
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

print("Loading checkpoint...")
payload = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
config = DarwinXConfig.from_mapping(payload["config"])

cognitive_config = replace(config,
    cognitive_architecture_version="three_organs_v1",
    cognitive_shadow_enabled=True, cognitive_pulse_enabled=True)

prev = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
model = DarwinXModel(cognitive_config)
torch.set_default_dtype(prev)

model.load_state_dict(payload["model_state_dict"], strict=False)

# Load trained adapters
adapter_path = Path("workspace/runtime/memory-training/trained-adapters.pt")
trained = torch.load(adapter_path, map_location="cpu", weights_only=False)
model.cognitive_runtime.memory.key_encoder.load_state_dict(trained["key_encoder"])
model.cognitive_runtime.memory.value_encoder.load_state_dict(trained["value_encoder"])
model.cognitive_runtime.memory.readout.load_state_dict(trained["readout"])
model.eval()

tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_SOURCE_ROOT), local_files_only=True)
runtime = model.cognitive_runtime
runtime.set_active()
memory = runtime.memory

# Teach 5 facts
print("Encoding and teaching facts...")
_step = 0
def meta():
    global _step; _step += 1
    return CognitiveForwardMetadata(step_id=_step, checkpoint_id="sha256:"+"a"*64, context_digest="sha256:"+"b"*64)

SYSTEM = "Voce e um assistente preciso. Responda com o codigo de seis digitos."
facts_taught = []

for i in range(5):
    entity = f"entidade-teste-{i:02d}"
    code = f"{200000 + i*12345:06d}"
    q_text = f"Qual e o codigo atribuido a {entity}?"
    a_text = f"O codigo de {entity} e {code}."

    q_chat = [{"role":"system","content":SYSTEM},{"role":"user","content":q_text}]
    q_formatted = tokenizer.apply_chat_template(q_chat, tokenize=False, add_generation_prompt=True)
    q_ids = tokenizer(q_formatted, add_special_tokens=False, return_tensors="pt").input_ids

    a_chat = [{"role":"system","content":SYSTEM},{"role":"assistant","content":a_text}]
    a_formatted = tokenizer.apply_chat_template(a_chat, tokenize=False, add_generation_prompt=False)
    a_ids = tokenizer(a_formatted, add_special_tokens=False, return_tensors="pt").input_ids

    with torch.inference_mode():
        q_out = model(q_ids, heartbeat=False, cognitive_metadata=meta())
        q_hidden = q_out.hidden_states
        a_out = model(a_ids, heartbeat=False, cognitive_metadata=meta())
        a_hidden = a_out.hidden_states

    rid = memory.teach(q_hidden, a_hidden, event_type="explicit_teaching", provenance="human", tags=(entity,))
    facts_taught.append({"entity": entity, "code": code, "q_hidden": q_hidden, "memory_id": rid})
    print(f"  taught: {entity} -> {code}  (id={rid[:16]}...)")

print(f"Store: {memory.slot_count} slots, {memory.verified_count} verified")

# Test recall
print()
print("=== RECALL TEST ===")
hits = 0
for fact in facts_taught:
    recalls = memory.recall(fact["q_hidden"], top_k=1, require_verified=True)
    r = recalls[0]
    hit = not r.abstained and fact["memory_id"] == r.record.memory_id
    if hit:
        hits += 1
    status = "HIT" if hit else "MISS"
    print(f"  {fact['entity']}: score={r.score:.4f}  prob={r.calibrated_probability:.4f}  abstained={r.abstained}  [{status}]")

print(f"\nRecall: {hits}/{len(facts_taught)}")
print("MEMORY_VALIDATION_OK" if hits == len(facts_taught) else f"MEMORY_VALIDATION_PARTIAL ({hits}/{len(facts_taught)})")
