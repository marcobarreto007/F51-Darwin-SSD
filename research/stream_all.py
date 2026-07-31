
import sys, struct, os
# HF_TOKEN via env — NUNCA hardcoded
_token = os.environ.get('HF_TOKEN', '')
if _token:
    from huggingface_hub import login; login(token=_token)
else:
    print('⚠️ HF_TOKEN nao configurado. Stream desativado.')
from f51_darwin.tokenizer import F51BPETokenizer
from datasets import load_dataset

tok = F51BPETokenizer.load('tokenizer/f51_bpe_80k')
CHUNK = 200000

sources = [
    ('wiki_pt', 'wikimedia/wikipedia', '20231101.pt', 50000),
    ('wiki_en', 'wikimedia/wikipedia', '20231101.en', 100000),
    ('fineweb_edu', 'HuggingFaceFW/fineweb-edu', 'sample-10BT', 100000),
    ('metamath', 'meta-math/MetaMathQA', None, 100000),
]

for name, ds_name, config, limit in sources:
    out = f'data/tokens_{name}.bin'
    print(f'{name}: sugando...', flush=True)
    buf = []; written = 0
    try:
        if config:
            ds = load_dataset(ds_name, config, streaming=True, split='train')
        else:
            ds = load_dataset(ds_name, streaming=True, split='train')
        with open(out, 'wb') as f:
            for i, row in enumerate(ds):
                if i >= limit: break
                if name == 'metamath':
                    text = f"Q: {row.get('query','')}\nA: {row.get('response','')}"
                else:
                    text = row.get('text', '')
                text = str(text).strip()
                if len(text) < 300: continue
                ids = tok.encode(text, add_eos=True)
                buf.extend(ids)
                if len(buf) >= CHUNK:
                    f.write(struct.pack(f'{len(buf)}i', *buf))
                    written += len(buf); buf = []
            if buf:
                f.write(struct.pack(f'{len(buf)}i', *buf))
                written += len(buf)
        print(f'  {name}: {written//4/1e6:.0f}M tokens', flush=True)
    except Exception as e:
        print(f'  {name}: ERRO {e}', flush=True)
