# Implementacao atual

## Modelo

`src/f51_darwin/darwin_x.py` e a fachada publica estavel da linhagem local
`F51-Darwin-X`, com tres escalas ativas: 100M (512d, 12L), 600M (1408d, 12L)
e 1.6B (1920d, 16L). A implementacao esta separada em
`src/f51_darwin/darwin_x_core/`: configuracao e
saida, camadas, neuroendocrino, MoE, bloco, modelo, estado/migracao e estimativa
de parametros. Essa divisao preserva os nomes publicos, a ordem de
`named_parameters`, o `state_dict`, os pesos compartilhados e o carregamento
estrito, cobertos por contrato hashado CPU-safe.

A config canonica para a escala experimental e `src/configs/darwin_x_100m.yaml`
(100M FULL V9, 16 orgaos ativos + ghost desabilitado). As tres linhagens possuem
configs isoladas: `src/configs/darwin_x_100m.yaml`, `src/configs/darwin_x_600m.yaml` e
`src/configs/darwin_x_1.6b_nitro.yaml`. Apenas a 100M possui checkpoint real.

## Dados e caminhos

`src/f51_darwin/dataset_layout.py` resolve `F51_DATASET_ROOT` quando explicitamente
fornecido e, caso contrario, `<project_root>/workspace`. O default nao depende
de sibling nem junction. O corpus canonico e feast_v2 e o lifecycle de ingestao
em `src/scripts/ingest_pipeline.py` separa quarentena, aprovacao e consolidacao. O
script e um adaptador fino para `src/f51_darwin/ingestion/`.

## Checkpoint e retomada

`src/f51_darwin/organism/checkpoint.py` e a autoridade unica para persistencia e
carregamento de checkpoint do organismo. `src/scripts/inspect_organism_checkpoint.py`
valida checkpoint version 7/8, config,
shapes, Topology Manifest, estado AdamW, identidade integral, hash e
`base_checkpoint_id`. Cada linhagem possui raiz de checkpoint isolada ancorada em `lineage_root.json`.
A unica raiz ativa e `workspace/03_CHECKPOINTS_100M_FULL_V9/` (100M FULL V9).
As raizes historicas nao estao presentes no clone limpo; metadados selecionados
ficam em `governance/archive/legacy-state/f51/`.
O gold 1.6B-Nitro (cycle 71, step 40751) nao esta acessivel localmente.

## Execucao

`src/scripts/start_overnight_16b.ps1` e o adaptador operacional para o gate mantido
em `src/f51_darwin/operations/`; ele calcula readiness, valida as duas GPUs e
controla canary/launch. O model parallel usa um unico processo e split dinamico.
`src/scripts/serve_davi.py` e o adaptador para `src/f51_darwin/serving/`, que publica
UI/API apenas em loopback e resolve o checkpoint pelo pointer validado, sem
promover o maior cycle. O HTML e package-data versionado, nao string embutida.

Os seis entrypoints suportados sao:

- `src/scripts/start_overnight_16b.ps1`;
- `src/scripts/darwin_organism.py`;
- `src/scripts/serve_davi.py`;
- `src/scripts/inspect_organism_checkpoint.py`;
- `src/scripts/darwin_inventory.py`;
- `src/scripts/ingest_pipeline.py`.

## Ambientes e auditoria

O ambiente CPU-audit usa Python 3.12 e locks hash-checked separados. O runtime
Nitro-CUDA e descrito por `governance/audit/provenance/nitro-runtime.json` e seu SBOM; um
ambiente nao e substituto do outro. CI cobre os gates CPU-safe. GPU, hashes de
artefatos pesados, readiness e canary exigem verificacao local.

O pacote Python usa PEP 517 com backend e versao fixados. A base importa sem
Torch, o CLI `f51-darwin version` funciona sem dependencias e o build auditavel
parte de `git archive`, gera wheel a partir do sdist, valida `RECORD` e instala
o wheel offline em venv limpa. A politica e
`governance/audit/policy/distribution.json`; o executor e `src/tools/build_distribution.py`.

Os limites machine-readable de tamanho, ciclos, defaults e carregamento
permissivo estao em `governance/audit/policy/operational-surface.json` e sao aplicados por
`src/tools/check_architecture_boundaries.py`. A organizacao fisica da raiz e
validada por `src/tools/check_physical_hygiene.py`.

Rollback de workspace e regras de launch estao em
`governance/docs/operacao/OPERACAO_SEGURA.md`. A hierarquia de prova e deliberadamente
fail-closed.
