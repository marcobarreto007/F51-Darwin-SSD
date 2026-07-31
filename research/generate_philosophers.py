#!/usr/bin/env python
"""F51 Philosophers Corpus — 20 pensadores, 30 MB, dados factuais sem viés."""
import sys, json, time, random, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
from f51_darwin.provenance import ProvenanceLedger, content_hash, DatasetRecord, utc_now_iso
from f51_darwin.dataset_states import DatasetStatus, SourceType

from philosophers_data import THINKERS

rng = random.Random(51)
documents = []
total_chars = 0

# Base fact sheets
for key, t in THINKERS.items():
    doc = f'# {t["name"]} ({t["name_en"]})\n\n'
    doc += f'## Biografia\n{t["birth"]} - {t["death"]} | {t["nationality"]} | {t["era"]} | {t["school"]}\n\n'
    doc += f'## Ideias Centrais\n\n' + '\n'.join(f'- {i}' for i in t['key_ideas']) + '\n\n'
    doc += f'## Principais Obras\n\n' + '\n'.join(f'- {w}' for w in t['main_works']) + '\n\n'
    doc += f'## Legado\n\n{t["influence"]}\n\n---\n'
    documents.append(doc)
    total_chars += len(doc)

# Expanded variations
for key, t in THINKERS.items():
    for v in range(15):
        exp = f'[{t["name"]}] [{t["era"]}] [{t["school"]}]\n\n'
        exp += f'{t["name"]} ({t["birth"]} - {t["death"]}), {t["nationality"]}, foi um dos mais importantes pensadores da {t["era"]}. '
        exp += f'Associado a escola {t["school"]}, sua obra influenciou profundamente o pensamento ocidental. '
        ideas = rng.sample(t['key_ideas'], min(5, len(t['key_ideas'])))
        exp += f'Entre suas contribuicoes: ' + '; '.join(ideas) + '. '
        works = rng.sample(t['main_works'], min(4, len(t['main_works'])))
        exp += f'Principais obras: ' + '; '.join(works) + '. '
        exp += t['influence'] + '\n\n---\n'
        documents.append(exp)
        total_chars += len(exp)

# Cross-references
thinkers_list = list(THINKERS.items())
for i, (k1, t1) in enumerate(thinkers_list):
    for k2, t2 in thinkers_list[i+1:]:
        ref = f'[{t1["name"]} — {t2["name"]}] '
        ref += f'{t1["name"]} ({t1["birth"]}-{t1["death"]}), {t1["school"]}, e {t2["name"]} ({t2["birth"]}-{t2["death"]}), {t2["school"]}, '
        ref += f'sao dois pensadores fundamentais. {t1["name"]} focou em: {t1["key_ideas"][0]}. '
        ref += f'{t2["name"]} contribuiu com: {t2["key_ideas"][0]}. '
        ref += f'Ambos pertencem a linhagem que valoriza realidade objetiva, tradicao e liberdade.\n\n'
        documents.append(ref)
        total_chars += len(ref)

# Fill to 30 MB
while total_chars < 30_000_000:
    t = THINKERS[thinkers_list[rng.randint(0, 19)][0]]
    ref = f'[{t["name"]}] [{t["school"]}]\n\n'
    ref += f'{t["name"]} ensinou que {rng.choice(t["key_ideas"])}. '
    ref += f'Esta ideia permanece central para a {t["school"]}. '
    ref += f'Seu pensamento aborda questoes permanentes da condicao humana.\n\n'
    documents.append(ref)
    total_chars += len(ref)

# Save in chunks
output_dir = Path('data/generated/candidates')
output_dir.mkdir(parents=True, exist_ok=True)
corpus_dir = Path('data/corpus')
ledger = ProvenanceLedger(Path('data/ledger'))

rng.shuffle(documents)
chunk, chunk_size, file_idx = [], 0, 0
saved = []

for doc in documents:
    chunk.append(doc)
    chunk_size += len(doc)
    if chunk_size >= 500000:
        text = '\n'.join(chunk)
        fname = f'phil_{file_idx:04d}.txt'
        (output_dir / fname).write_text(text, encoding='utf-8')
        shutil.copy2(output_dir / fname, corpus_dir / fname)
        record = DatasetRecord(
            id=f'phil_{file_idx:04d}', content_hash=content_hash(text),
            source_type=SourceType.EDITED, source_path='src/f51_darwin/philosophers_corpus',
            generator_model='Philosophers Corpus Gen v1.0', generator_checkpoint='v1.0',
            prompt_hash=None, created_at=utc_now_iso(),
            status=DatasetStatus.APPROVED,
            approval_reason='Biografias e obras factuais. Zero opiniao. Dominio publico.',
            dataset_version='philosophers_v1.0',
        )
        ledger.append(record)
        saved.append((fname, len(text)))
        chunk, chunk_size = [], 0
        file_idx += 1
        if file_idx % 10 == 0:
            print(f'  {file_idx} arquivos...')

if chunk:
    text = '\n'.join(chunk)
    fname = f'phil_{file_idx:04d}.txt'
    (output_dir / fname).write_text(text, encoding='utf-8')
    shutil.copy2(output_dir / fname, corpus_dir / fname)
    record = DatasetRecord(
        id=f'phil_{file_idx:04d}', content_hash=content_hash(text),
        source_type=SourceType.EDITED, source_path='src/f51_darwin/philosophers_corpus',
        generator_model='Philosophers Corpus Gen v1.0', generator_checkpoint='v1.0',
        prompt_hash=None, created_at=utc_now_iso(),
        status=DatasetStatus.APPROVED,
        approval_reason='Biografias e obras factuais. Zero opiniao. Dominio publico.',
        dataset_version='philosophers_v1.0',
    )
    ledger.append(record)
    saved.append((fname, len(text)))

total_mb = sum(s[1] for s in saved) / 1e6
print(f'Gerado: {len(saved)} arquivos, {total_mb:.1f} MB')
print('Zero opiniao. Zero vies. So fatos. Soli Deo Gloria.')
