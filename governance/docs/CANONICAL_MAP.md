# Mapa canonico Darwin-X

Este mapa descreve componentes duraveis. Estado datado pertence a
`governance/docs/operacao/STATUS_ATUAL.md`.

## Arvore de autoridade

```text
F51-Darwin-SSD/
|-- src/f51_darwin/                produto: modelo, organismo, servico e ingestao
|-- src/configs/                   configuracoes locais classificadas
|-- src/scripts/                   adaptadores suportados (lista em src/scripts/README.md)
|-- src/tools/                     manutencao e verificadores classificados
|-- research/                      experimentos fora da superficie operacional
|-- src/tests/                  contratos e regressoes CPU-safe
|-- governance/docs/            operacao, arquitetura, pesquisa e historia
|-- governance/audit/           politicas, schemas, SBOM e proveniencia
|-- governance/archive/         material historico nao operacional
`-- workspace/                  dados e runtime pesados, ignorados pelo Git
```

`workspace/` contem `01_TOKENIZADOS/`, `02_CORPUS/`,
`03_CHECKPOINTS_100M_FULL_V9/` (unica raiz ativa), `04_MANIFESTOS/`,
`tokenizer/` e `runtime/`. As raizes historicas nao estao presentes no clone
limpo; metadados selecionados ficam em `governance/archive/legacy-state/f51/`.
O resolver oficial esta em `src/f51_darwin/dataset_layout.py`: override explicito
`F51_DATASET_ROOT` ou, por padrao, `<project_root>/workspace`.

## Modelo e linhagem

- modelo: fachada `src/f51_darwin/darwin_x.py`, nucleo `src/f51_darwin/darwin_x_core/`;
- runtime evolutivo: `src/scripts/darwin_organism.py`;
- implementacao do runtime: `src/f51_darwin/organism/`;
- servico Davi: `src/f51_darwin/serving/`;
- ingestao: `src/f51_darwin/ingestion/`;
- configs: `src/configs/darwin_x_100m.yaml` (100M), `src/configs/darwin_x_600m.yaml` (600M),
  `src/configs/darwin_x_1.6b_nitro.yaml` (1.6B);
- corpus: `00_CORPUS_PRINCIPAL_tokens_feast_v2.bin` (18.5 bilhoes de tokens);
- checkpoints isolados por linhagem, ancorados em `lineage_root.json`;
- gold 1.6B-Nitro (cycle 71, step 40751) nao acessivel localmente;
- verificadores: `src/scripts/inspect_organism_checkpoint.py` e
  `src/tools/verify_gold_lineage.py`.

O checkpoint posterior preservado na raiz ativa nao e automaticamente
canonico. Promocao exige verificacao integral e decisao explicita; numeracao
maior nao e evidencia. Os metadados historicos em
`governance/archive/legacy-state/f51/` nao constituem autoridade operacional.

## Superficie operacional

| Entrada | Responsabilidade |
|---|---|
| `src/scripts/start_overnight_16b.ps1` | readiness, canary e launch autorizado 1.6B |
| `src/scripts/start_100m_65b.ps1` | canarios causais e treino finito 100M/65B |
| `src/scripts/start_100m_auto.ps1` | 100M FULL V9 — treino auto-resiliente com resume |
| `src/scripts/darwin_organism.py` | organismo e servico evolutivo |
| `src/scripts/serve_davi.py` | UI/API de loopback |
| `src/scripts/inspect_organism_checkpoint.py` | inspecao estrita de checkpoint |
| `src/scripts/darwin_inventory.py` | inventario local |
| `src/scripts/ingest_pipeline.py` | quarentena, aprovacao e consolidacao |

A lista machine-readable e a classificacao de todo executavel estao em
`governance/audit/policy/operational-surface.json`. Nenhum outro script e entrypoint de
producao.

## Distribuicao local auditavel

O wheel/sdist nao inclui `workspace/`, auditoria, pesquisa, ferramentas,
testes, datasets ou checkpoints. O build parte de commit imutavel e arvore
limpa, trabalha offline, gera wheel a partir do sdist e conclui com instalacao
em venv vazia. Politica: `governance/audit/policy/distribution.json`; comando:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv_nitro\Scripts\python.exe -B -m tools.build_distribution --commit HEAD
```

## Governanca e prova

- duplicidade e autoridade: `governance/audit/policy/duplicates.json`;
- dependencias CPU e Nitro: `governance/audit/policy/dependencies.json`;
- seguranca: `SECURITY.md`;
- evidencia e reproducao: `governance/audit/README.md`;
- rollback e launch: `governance/docs/operacao/OPERACAO_SEGURA.md`;
- arquitetura code-backed: `governance/docs/arquitetura/IMPLEMENTACAO_ATUAL.md`.

A hierarquia de prova e artefato/runtime, manifest/hash, codigo/teste,
documento canonico e, por ultimo, historico. Arquivos em `governance/archive/` e
`governance/docs/_historico/` preservam proveniencia, mas nao autorizam operacao.
