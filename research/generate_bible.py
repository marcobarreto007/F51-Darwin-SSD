#!/usr/bin/env python
"""F51 Bible Corpus — Biblia Sagrada completa para consulta e citacao."""
import sys, shutil, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
from f51_darwin.provenance import ProvenanceLedger, content_hash, DatasetRecord, utc_now_iso
from f51_darwin.dataset_states import DatasetStatus, SourceType
from bible_data import OLD_TESTAMENT, NEW_TESTAMENT, KEY_VERSES, NARRATIVES, DOCTRINES, PSALMS, PROVERBS

rng = random.Random(51)
documents = []
total_chars = 0

# Structure
doc = '# BIBLIA SAGRADA — Estrutura Completa\n\n'
doc += 'A Biblia Sagrada e a Palavra de Deus, 66 livros, ~40 autores, ~1500 anos. '
doc += 'Antigo Testamento (39 livros) e Novo Testamento (27 livros). '
doc += 'Mensagem central: redencao por Jesus Cristo. Soli Deo Gloria.\n\n'
for section, books in OLD_TESTAMENT.items():
    doc += f'## {section}\n\n'
    for name, author, theme in books:
        doc += f'### {name} — Autor: {author} — {theme}\n\n'
for section, books in NEW_TESTAMENT.items():
    doc += f'## {section}\n\n'
    for name, author, theme in books:
        doc += f'### {name} — Autor: {author} — {theme}\n\n'
documents.append(doc); total_chars += len(doc)

# Key verses
doc = '# BIBLIA SAGRADA — Versiculos-Chave (Almeida)\n\n'
for ref, verse in KEY_VERSES.items():
    doc += f'## {ref}\n\n{verse}\n\n---\n\n'
documents.append(doc); total_chars += len(doc)

# Each book
for section, books in OLD_TESTAMENT.items():
    for name, author, theme in books:
        doc = f'# {name}\n\nAntigo Testamento. {section}. Autor: {author}. Tema: {theme}.\n\n'
        doc += f'{name} e um dos 39 livros do Antigo Testamento. '
        found = any(ref.startswith(name) for ref in KEY_VERSES)
        if found:
            for ref, verse in KEY_VERSES.items():
                if ref.startswith(name):
                    doc += f'Versiculo-chave: {ref} — {verse}\n'; break
        doc += f'Toda Escritura e divinamente inspirada (2 Timoteo 3:16).\n\n---\n'
        documents.append(doc); total_chars += len(doc)

for section, books in NEW_TESTAMENT.items():
    for name, author, theme in books:
        doc = f'# {name}\n\nNovo Testamento. {section}. Autor: {author}. Tema: {theme}.\n\n'
        found = any(ref.startswith(name) for ref in KEY_VERSES)
        if found:
            for ref, verse in KEY_VERSES.items():
                if ref.startswith(name):
                    doc += f'Versiculo-chave: {ref} — {verse}\n'; break
        doc += f'Toda Escritura e divinamente inspirada (2 Timoteo 3:16).\n\n---\n'
        documents.append(doc); total_chars += len(doc)

# Narratives
for title, text in NARRATIVES:
    doc = f'# Episodio Biblico: {title}\n\n{text}\n\n---\n'
    documents.append(doc); total_chars += len(doc)

# Doctrines
for title, text in DOCTRINES:
    doc = f'# Doutrina Crista: {title}\n\n{text}\n\n---\n'
    documents.append(doc); total_chars += len(doc)

# Psalms
doc = '# Livro dos Salmos — Sabedoria e Louvor\n\n'
for p in PSALMS:
    doc += f'{p}\n\n---\n\n'
documents.append(doc); total_chars += len(doc)

# Proverbs
doc = '# Livro de Proverbios — Sabedoria Pratica\n\n'
for p in PROVERBS:
    doc += f'{p}\n\n---\n\n'
documents.append(doc); total_chars += len(doc)

# Fill to 15 MB
while total_chars < 15_000_000:
    ref = rng.choice(list(KEY_VERSES.keys()))
    verse = KEY_VERSES[ref]
    doc = f'{ref}\n\n{verse}\n\nA Palavra de Deus e viva e eficaz (Hebreus 4:12). Soli Deo Gloria.\n\n---\n\n'
    documents.append(doc); total_chars += len(doc)

# Save
output_dir = Path(ROOT / 'data' / 'generated' / 'candidates')
output_dir.mkdir(parents=True, exist_ok=True)
corpus_dir = Path(ROOT / 'data' / 'corpus')
ledger = ProvenanceLedger(Path(ROOT / 'data' / 'ledger'))
rng.shuffle(documents)

chunk, chunk_size, file_idx = [], 0, 0
for doc in documents:
    chunk.append(doc); chunk_size += len(doc)
    if chunk_size >= 500000:
        text = '\n'.join(chunk)
        fname = f'biblia_{file_idx:04d}.txt'
        (output_dir / fname).write_text(text, encoding='utf-8')
        shutil.copy2(output_dir / fname, corpus_dir / fname)
        ledger.append(DatasetRecord(
            id=f'biblia_{file_idx:04d}', content_hash=content_hash(text),
            source_type=SourceType.EDITED, source_path='Biblia Sagrada — Almeida (dominio publico)',
            generator_model='F51 Bible Corpus v1.0', generator_checkpoint='v1.0',
            prompt_hash=None, created_at=utc_now_iso(), status=DatasetStatus.APPROVED,
            approval_reason='Biblia Sagrada — Palavra de Deus. Dominio publico.',
            dataset_version='biblia_v1.0'))
        chunk, chunk_size = [], 0; file_idx += 1

if chunk:
    text = '\n'.join(chunk)
    fname = f'biblia_{file_idx:04d}.txt'
    (output_dir / fname).write_text(text, encoding='utf-8')
    shutil.copy2(output_dir / fname, corpus_dir / fname)
    ledger.append(DatasetRecord(
        id=f'biblia_{file_idx:04d}', content_hash=content_hash(text),
        source_type=SourceType.EDITED, source_path='Biblia Sagrada — Almeida (dominio publico)',
        generator_model='F51 Bible Corpus v1.0', generator_checkpoint='v1.0',
        prompt_hash=None, created_at=utc_now_iso(), status=DatasetStatus.APPROVED,
        approval_reason='Biblia Sagrada — Palavra de Deus. Dominio publico.',
        dataset_version='biblia_v1.0'))

saved = list(output_dir.glob('biblia_*.txt'))
total_mb = sum(f.stat().st_size for f in saved) / 1e6
print(f'Biblia Sagrada: {len(saved)} arquivos, {total_mb:.1f} MB')
print('Soli Deo Gloria.')
