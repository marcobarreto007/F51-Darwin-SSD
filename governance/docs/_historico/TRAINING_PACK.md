# F51 Darwin-SSD — Training Pack para Nuvem

## Visão Geral

O `prepare_training_pack.py` orquestra a criação completa do material de treino pronto para upload:

1. **Gera Corpus Clássico** — Gutenberg PT/EN/FR, Bíblia, Constituição
2. **Builda Pack Aprovado** — Filtra por policy, cria manifesto
3. **Encapsula** — Cria training pack final com tokenizer
4. **Comprime** — Gera tar.gz pronto para nuvem

## Uso Básico

```powershell
# Pipeline completo
.\src\\scripts\\prepare_training_pack.ps1

# Ou Python direto
python src/scripts/prepare_training_pack.py
```

## Opções

```powershell
# Pula download (usa corpus existente)
.\src\\scripts\\prepare_training_pack.ps1 --skip-download

# Apenas baixa corpus, não builda pack
.\src\\scripts\\prepare_training_pack.ps1 --raw-only

# Não inclui tokenizer no pack final
.\src\\scripts\\prepare_training_pack.ps1 --no-tokenizer

# Saída em JSON
.\src\\scripts\\prepare_training_pack.ps1 --json

# Batch com nome específico
.\src\\scripts\\prepare_training_pack.ps1 --batch-name "darwin_25b_baseline"
```

## Estrutura do Training Pack

```
f51_training_pack_YYYYMMDD_HHMMSS.tar.gz
├── approved_corpus_pack.tar.gz      # Corpus filtrado + manifesto
├── tokenizer/                         # Tokenizer F51 BPE 80k
│   └── f51_bpe_80k/
│       ├── vocab.json
│       ├── merges.txt
│       ├── config.json
│       └── tokenizer.json
└── manifest.json                      # Metadados completos
```

## Upload para Nuvem

Após criar o pack, use o script de upload:

```powershell
python src/scripts/upload_corpus_pack.py \
    "staging/f51_training_pack_YYYYMMDD_HHMMSS.tar.gz" \
    --target master \
    --batch-name "darwin_25b_baseline"
```

## Configuração de Nuvem

Edita `src/configs/corpus_factory.yaml`:

```yaml
cloud:
  master:
    role: corpus_master
    ssh: ssh -p 56013 root@70.30.158.46
    workspace: /workspace/f51_corpus_factory
```

## Fluxo de Trabalho

```mermaid
graph LR
    A[Fontes Externas] --> B[build_classical_corpus.py]
    B --> C[Corpus Raw]
    C --> D[build_corpus_pack]
    D --> E[Approved Pack]
    E --> F[prepare_training_pack.py]
    F --> G[Training Pack Final]
    G --> H[upload_corpus_pack.py]
    H --> I[Nuvem]
```

## Verificação

Após upload, verifique na nuvem:

```bash
ssh -p 56013 root@70.30.158.46 \
    "ls -lh /workspace/f51_corpus_factory/incoming/"
```

## Troubleshooting

**Erro: "Corpus não encontrado"**
- Execute sem `--skip-download` primeiro

**Erro: "Pack não foi criado"**
- Verifique se há arquivos .txt no corpus raw
- Cheque logs em `packs/manifest.jsonl`

**Erro SSH no upload**
- Verifique configuração em `src/configs/corpus_factory.yaml`
- Teste conexão: `ssh -p PORT root@HOST`
