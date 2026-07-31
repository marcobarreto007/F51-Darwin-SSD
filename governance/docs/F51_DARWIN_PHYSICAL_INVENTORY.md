# F51 Darwin-X — Inventário Físico da Raiz

**Snapshot:** 2026-07-31 (America/Toronto)
**Raiz:** `C:\Users\marco\Desktop\F51-Darwin-SSD`  
**Branch:** `feat/darwin-16b-smol-transplant`  
**HEAD observado:** `590d08d` mais reorganização da sessão no working tree
**Método:** `Get-ChildItem -File -Recurse -Force`, sem importar Python, sem
carregar checkpoints e sem calcular SHA-256 dos artefatos pesados.

Este documento complementa o índice Git. O índice responde “o caminho é
rastreado?”; este inventário responde “o que existe fisicamente na raiz, quanto
ocupa e qual é a fronteira operacional?”. Diretórios gerados, caches, ambiente
virtual, VCS, worktrees, arquivo morto e artefatos pesados permanecem visíveis
mesmo quando são ignorados pelo Git.

## Resumo físico

| Escopo | Arquivos | Bytes | Classificação |
|---|---:|---:|---|
| Raiz inteira, incluindo `.git` | 128.089 | 308.710.673.600 | estado físico observado; `.git` é volátil |
| `workspace/` | 49.359 | 191.731.309.446 | dados, corpus, checkpoints e runtime |
| `governance/archive/` | 2.062 | 104.724.366.426 | histórico, checkpoints e material aposentado |
| `.git/` | 5.993 | 6.398.194.141 | metadados VCS; volátil; não é runtime |
| `.venv_nitro/` | 47.067 | 5.306.572.997 | ambiente Python local |
| `.worktrees/` | 7.824 | 163.904.077 | checkout/worktree auxiliar |
| `F51-JEPA-100M-2.0/` | 14.957 | 354.429.719 | checkout aninhado separado |

O total é uma soma física no instante da leitura; ele inclui caches e arquivos
de controle e não deve ser interpretado como tamanho do produto distribuível.

## Inventário de primeiro nível

| Entrada | Arquivos | Bytes | Função / classificação |
|---|---:|---:|---|
| `.agent_bus/` | 1 | 1.546 | estado de comunicação local |
| `.audit-tools/` | 1 | 22.575.104 | ferramentas de auditoria local |
| `.claude/` | 31 | 60.206 | estado/configuração de agente |
| `.git/` | 5.993 | 6.398.194.141 | metadados e objetos Git |
| `.gitattributes` | 1 | 63 | política de atributos Git |
| `.github/` | 1 | 2.238 | workflow CI |
| `.gitignore` | 1 | 1.441 | exclusões Git |
| `.gitleaks.toml` | 1 | 728 | configuração de secret scan |
| `.superpowers/` | 65 | 1.218.536 | planos/estado auxiliar |
| `.venv_nitro/` | 47.067 | 5.306.572.997 | ambiente virtual |
| `.worktrees/` | 7.824 | 163.904.077 | worktree auxiliar |
| `AGENTS.md` | 1 | 4.724 | contrato operacional |
| `governance/archive/` | 2.062 | 104.724.366.426 | material histórico/aposentado |
| `governance/audit/` | 47 | 524.545 | políticas, schemas e relatórios |
| `CLAUDE.md` | 1 | 4.724 | contrato duplicado para outro agente |
| `src/configs/` | 17 | 26.970 | configurações de modelo e execução |
| `DIARIO_DE_BORDO.md` | 1 | 115.730 | diário histórico |
| `governance/docs/` | 88 | 1.640.490 | documentação de autoridade variável |
| `src/f51_darwin/` | 191 | 2.415.531 | pacote principal e runtime |
| `F51-JEPA-100M-2.0/` | 14.957 | 354.429.719 | checkout aninhado |
| `LICENSE` | 1 | 830 | licença |
| `MANIFEST.in` | 1 | 236 | empacotamento |
| `NOTICE` | 1 | 469 | avisos legais |
| `pyproject.toml` | 1 | 1.227 | metadados/dependências do pacote |
| `README.md` | 1 | 4.068 | documentação de entrada |
| `research/` | 124 | 1.452.872 | experimentos, benchmarks e simuladores |
| `requirements-*.in/lock` | 4 | 44.736 | dependências fixadas |
| `src/scripts/` | 13 | 49.153 | somente os entrypoints operacionais suportados |
| `SECURITY.md` | 1 | 1.280 | política de segurança |
| `src/tests/` | 160 | 923.190 | testes e fixtures |
| `THIRD_PARTY_NOTICES.md` | 1 | 2.047 | avisos de terceiros |
| `src/tools/` | 71 | 824.110 | verificadores, manutenção e assets do dashboard |
| `workspace/` | 49.359 | 191.731.309.446 | estado pesado local |

## Subárvores pesadas

### `workspace/`

| Subárvore | Arquivos | Bytes | Leitura operacional |
|---|---:|---:|---|
| `00_BRUTOS/` | 2.620 | 1.121.310.171 | corpus bruto |
| `00_DONORS/` | 72 | 16.326.274.700 | doadores/artefatos de transferência |
| `01_TOKENIZADOS/` | 57 | 122.561.072.115 | token bins e partes tokenizadas |
| `01_TOKENIZER/` | 3 | 2.108.975 | tokenizer |
| `02_CORPUS/` | 5.842 | 1.158.708.619 | corpus preparado |
| `03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/` | 4 | 4.091.865.697 | candidato Dense |
| `03_CHECKPOINTS_1.7B_SMOL_EXACT_V2/` | 4 | 3.882.461.549 | candidato Exact |
| `03_CHECKPOINTS_1.7B_SMOL_EXACT_V3/` | 4 | 4.092.299.314 | candidato Exact posterior |
| `03_CHECKPOINTS_100M_BASELINE_PURO/` | 5 | 1.831.027.805 | baseline 100M |
| `03_CHECKPOINTS_100M_DENSE_TRANSFORMER_BASELINE/` | 6 | 1.038.934.071 | baseline Dense |
| `03_CHECKPOINTS_100M_FULL_ORGANISM_V1/` | 8 | 1.985.921.379 | organismo V1 |
| `03_CHECKPOINTS_100M_FULL_ORGANISM_V1_pre_5M_holdout_20260726_092433/` | 5 | 1.991.451.303 | snapshot V1/holdout |
| `03_CHECKPOINTS_100M_FULL_ORGANISM_V2_MOTORFIX/` | 6 | 1.960.421.697 | organismo V2 |
| `03_CHECKPOINTS_100M_FULL_ORGANISM_V3/` | 11 | 11.753.054.163 | organismo V3 |
| `03_CHECKPOINTS_DARWIN_TRANSFER_DISTILGPT2_V1/` | 38 | 10.162.555.237 | transferência |
| `03_CHECKPOINTS_DARWIN_TWO_DONOR_V1/` | 3 | 1.222.271.089 | dois doadores |
| `04_MANIFESTOS/` | 17 | 3.996.373 | manifests e estado declarado |
| `05_BACKUPS/` | 2 | 1.035 | backups auxiliares |
| `corpus_transfer/` | 16 | 731.292.789 | transferência de corpus |
| `diagnostics/` | 4 | 57.155 | diagnósticos |
| `runtime/` | 40.628 | 5.800.838.329 | logs, relatórios e execuções |
| `tokenizer/` | 4 | 13.385.881 | material do tokenizer |

As roots canônicas declaradas em `AGENTS.md` continuam ausentes nesta leitura;
as roots acima são as que existem fisicamente e não podem ser promovidas por
nome, mtime ou tamanho.

### `governance/archive/`

| Subárvore | Arquivos | Bytes | Classificação |
|---|---:|---:|---|
| `checkpoints/` | 129 | 103.558.383.037 | checkpoints históricos; não autoridade atual |
| `cloud-local/` | 1.917 | 1.155.571.172 | scripts/experimentos antigos |
| `cloud-releases/` | 6 | 11.355.713 | pacotes antigos |
| `configs/` | 2 | 1.995 | configs antigas |
| `dead_code/` | 8 | 65.460 | código aposentado |
| `historical-docs/` | 10 | 63.762 | documentação histórica |
| `historical-evaluations/` | 9 | 40.678 | avaliações históricas |
| `legacy-scratch/` | 40 | 104.937 | rascunhos |
| `legacy-state/` | 8 | 103.701 | estado legado |
| `scripts/` | 4 | 10.431 | scripts antigos |
| `agent-bus/` | 14 | 29.409 | mensagens históricas |

### Ambientes e worktrees

| Local | Arquivos | Bytes | Regra |
|---|---:|---:|---|
| `.venv_nitro/Lib/` | 46.933 | 5.296.289.972 | dependência local, não produto |
| `.venv_nitro/Scripts/` | 59 | 4.213.236 | executáveis do ambiente |
| `.venv_nitro/share/` | 9 | 4.856.644 | dados do ambiente |
| `.worktrees/three-organ-cognition-foundation/` | 8.447 | 173.486.518 | branch auxiliar separado |
| `F51-JEPA-100M-2.0/` | 15.050 | 356.065.105 | checkout aninhado; autoridade pendente |

## Fronteiras e arquivos fora do índice

- `git ls-files` contém 860 caminhos no HEAD observado; os destinos de
  movimentação ainda estão no working tree até o próximo commit.
- `src/tests/test_provenance_certificate.py` está rastreado e aparece no
  manifesto de trabalho; não foi tratado como artefato descartável.
- `workspace/`, `.venv_nitro/`, `.worktrees/`, `governance/archive/` e caches não podem ser
  considerados cobertos por Git limpo. Para afirmar identidade de artefato é
  necessário usar manifest, pointer, shape, config e hash do artefato exato.
- Os gates desta sessão podem criar caches e diretórios temporários; eles são
  artefatos de teste, não parte do produto, e são removidos no fechamento da
  sessão quando a ACL local permite.
- O inventário mede presença e volume. Não afirma que um checkpoint é válido,
  carregável, dourado ou operacional.

## Limitações honestas

Esta leitura não recalculou SHA-256 dos blobs de centenas de gigabytes, não
abriu cada checkpoint em GPU e não iniciou treino/serving. O inventário físico
é completo no nível de contagem/tamanho observado; a validade científica e
operacional permanece nos documentos de domínio e nos gates da Bíblia.

Os números de `.git` podem mudar apenas pela criação de objetos, reflogs ou
commits automáticos durante a sessão; por isso a contagem física deve sempre
ser lida com o timestamp acima, não como uma constante do produto.
