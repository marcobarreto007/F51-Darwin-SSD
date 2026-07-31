# F51 Darwin-X — Bíblia do Sistema e Dicionário Técnico

**Estado:** consolidação inicial da auditoria; ainda não é o veredito final  
**Snapshot:** 2026-07-31
**Branch:** `feat/darwin-16b-smol-transplant`  
**HEAD:** `590d08d` mais reorganização da sessão no working tree
**Raiz:** `C:\Users\marco\Desktop\F51-Darwin-SSD`

> Regra de verdade: nenhum nome de módulo, documento histórico, teste estreito,
> loss, ciclo, mtime ou hash isolado prova qualidade, prontidão, causalidade ou
> linhagem. Cada entrada final receberá `OPERACIONAL`, `IMPLEMENTADO E TESTADO`,
> `EM INTEGRAÇÃO`, `PESQUISA`, `HISTÓRICO`, `AUSENTE` ou `CONTRADITÓRIO`.

## 1. Identidade do sistema

F51 Darwin-X é uma plataforma local de pesquisa e engenharia para treinar,
retomar, avaliar, servir e evoluir modelos híbridos SSD + Attention + MoE,
com corpus governado, checkpoints isolados, runtime de organismo, ingestão,
serving e experimentos de circuitos/memória/transplante.

Os nomes biológicos — órgãos, soul, heartbeat, ghost e consciência — são
metáforas de engenharia. O repositório não autoriza concluir consciência,
biologia artificial, inteligência geral ou aprendizado por inferência sem prova
causal específica.

## 2. Estado real da raiz

- `git ls-files`: 860 caminhos rastreados.
- `git status --short --untracked-files=all`: a reorganização estrutural e os
  manifestos/documentos derivados estão modificados no working tree.
- `F51-JEPA-100M-2.0` é um gitlink/check-out aninhado capturado no HEAD; seu
  conteúdo continua uma fronteira de autoridade separada.
- `src/tests/test_provenance_certificate.py` agora pertence ao commit `df9f250` e
  precisa ser incluído na leitura de proveniência/transplante.
- `workspace/` é pesado e ignorado pelo Git; Git limpo não prova a coerência
  física de corpus, tokenizer, manifests, pointers ou checkpoints.
- O status canônico é `governance/docs/operacao/STATUS_ATUAL.md`, mas seu snapshot é de
  2026-07-29 e o HEAD atual é de 2026-07-30.
- A evidência de source audit preservada está ligada a commit antigo e mantém
  `NOT_GOLD` por falha histórica de secret scan.

### Contagem rastreada

| Área | Arquivos |
|---|---:|
| `src/f51_darwin/` | 191 |
| `src/scripts/` | 13 |
| `research/` | 124 |
| `src/tools/` | 71 |
| `src/tests/` | 160 |
| `governance/docs/` | 87 |
| `governance/archive/` | 144 |
| `governance/audit/` | 32 |
| `src/configs/` | 17 |
| `F51-JEPA-100M-2.0` | 1 gitlink |

O índice mecânico completo dos arquivos rastreados está em
[`F51_DARWIN_SYSTEM_FILE_INDEX.md`](F51_DARWIN_SYSTEM_FILE_INDEX.md).

O dicionário semântico arquivo-a-arquivo está em
[`F51_DARWIN_FILE_DICTIONARY.md`](F51_DARWIN_FILE_DICTIONARY.md).
Ele cobre cada caminho rastreado com classificação, tamanho, linhas e primeiro
sinal textual observável.

O dicionário estático de classes, funções e métodos Python está em
[`F51_DARWIN_SYSTEM_SYMBOL_DICTIONARY.md`](F51_DARWIN_SYSTEM_SYMBOL_DICTIONARY.md).
Ele cobre os 589 arquivos Python rastreados; dois arquivos históricos têm erro
de parsing real. Quatro arquivos com BOM UTF-8 foram lidos corretamente com
`utf-8-sig`.

O inventário físico de todos os diretórios, caches, worktrees, ambiente,
arquivo morto e `workspace/` está em
[`F51_DARWIN_PHYSICAL_INVENTORY.md`](F51_DARWIN_PHYSICAL_INVENTORY.md).
Ele registra 128.871 arquivos e 308.727.412.118 bytes no snapshot físico;
presença e tamanho não equivalem a validade de checkpoint.

O commit `df9f250` acrescentou um certificado de herança por bijeção ao caminho
`src/f51_darwin/transplant_16b/`: operações que apenas selecionam/reordenam valores
podem carregar `provenance_sha256` e drift zero medido, enquanto projeções que
fabricam valores exigem uma métrica de drift. Isso é uma melhoria de
proveniência/integridade, não prova de transferência de capacidade nem uma nova
descoberta científica por si só.

O commit `52acfdd` acrescentou `governance/docs/BRAINSTORM_2026-07-30.md`, um backlog de
40 ideias distribuídas entre hoje, próxima semana, este mês e sonhos de 3–6
meses. Ele é `PLANO / NÃO-AUTORIDADE`: nenhuma das 40 ideias é evidência de
execução, publicação, aquisição de hardware ou descoberta.

## 3. Hierarquia de autoridade

1. runtime vivo e artefato carregado;
2. manifests, pointers, identidades, shapes e hashes verificados;
3. código e testes que cobrem o comportamento;
4. `governance/docs/operacao/STATUS_ATUAL.md`;
5. documentos arquiteturais canônicos;
6. histórico, planos, specs e diário.

Fontes estruturais: `AGENTS.md`, `governance/docs/CANONICAL_MAP.md`,
`governance/docs/operacao/OPERACAO_SEGURA.md`, `governance/docs/arquitetura/IMPLEMENTACAO_ATUAL.md`,
`governance/audit/policy/operational-surface.json` e `governance/docs/_historico/INDEX.md`.

## 4. Dicionário da raiz

| Diretório | Conteúdo | Classificação |
|---|---|---|
| `src/f51_darwin/` | modelo, organismo, dados, checkpoint, serving, ingestão, circuitos e transplantes | produto/runtime |
| `src/configs/` | configurações de linhagem e pesquisa | configuração |
| `src/scripts/` | os 11 entrypoints operacionais suportados | superfície oficial |
| `src/tools/` | auditoria, manutenção, distribuição e verificadores | manutenção |
| `research/` | protótipos, simuladores, geradores e benchmarks | pesquisa |
| `src/tests/` | contratos, regressões, invariantes e smoke tests | validação |
| `governance/docs/` | operação, arquitetura, pesquisa, planos e histórico | autoridade variável |
| `governance/audit/` | políticas, schemas, SBOM, proveniência e relatórios | governança |
| `governance/archive/` | material aposentado | histórico |
| `workspace/` | dados, corpus, tokenizer, checkpoints e runtime pesado | artefatos locais |
| `F51-JEPA-100M-2.0/` | checkout/worktree paralelo | fronteira não resolvida |

## 5. Modelo e runtime

- `src/f51_darwin/darwin_x.py`: fachada pública.
- `src/f51_darwin/darwin_x_core/`: config, blocos, layers, MoE, losses, estado e
  estimativa de parâmetros.
- `src/f51_darwin/organism/`: bootstrap, lifecycle, treinamento, causal bus,
  órgãos, checkpoint, runtime e CLI.
- `src/f51_darwin/cognition/`: contratos, adapters, memória, executivo e world
  model.
- `src/f51_darwin/serving/`: Davi em loopback.
- `src/f51_darwin/ingestion/`: ingestão governada.
- `src/f51_darwin/circuits/`: identidade, ablação, manifest, ledger, transação,
  package, pointer e rollback.
- `src/f51_darwin/transfer/`, `transplant/` e `transplant_16b/`: conversões,
  assemblies, seleção, órgãos e verificação.

Órgãos que exigem entrada própria no dicionário: GABA, Ghost, Curiosity, Soul,
Decision Engine, Ghost Brain, JEPA, Spider-Sense, Heartbeat/TTM, controle
neuroendócrino/MoE, Unified Mesh, Inter-Hemispheric, Sleep e evolução
estrutural. “Conectado” não significa “útil” ou “aprende”.

### Bloqueios do runtime confirmados

- `src/f51_darwin/inference_engine.py` chama `block.moe(...)`; em Dense SwiGLU
  `block.moe` é `None` e o caminho correto é `block.ffn`.
- O engine manual não replica integralmente GABA, TTM, Spider, Ghost,
  Inter-Hemispheric, causal runtime e heartbeat do forward completo.
- Quando o engine compilado está disponível,
  `InferenceLearner.generate_response` pode retornar antes de aplicar o
  `OnlineAdapter`.
- YAML e `DarwinOrganismConfig` são planos separados; flags podem ser
  sobrescritas ou não chegar ao organismo.
- O serving usa `context_length` de treino onde alguns YAMLs declaram
  `inference_context_length` diferente.

Classificação: **núcleo implementado; runtime parcialmente integrado e sem
prova end-to-end para Dense/adapter**.

## 6. Linhagens e artefatos

O contrato declara 100M, 600M e 1.6B com roots isoladas em `workspace/`, mas as
três roots canônicas declaradas não foram encontradas fisicamente nesta leitura.
Existem roots experimentais Smol, transfer, 100M antigas e two-donor.

O corpus principal `feast_v2` tem manifesto declarado de 74.195.890.756 bytes,
18.548.972.689 tokens, `int32`, tokenizer `f51_bpe_80k` e SHA-256
`9677e9f22f4d78efa7b25c77b2da3cbdb5fb2a499926ac641a75c13b65e25cff`. O hash
integral não foi recalculado nesta rodada.

Conflitos encontrados:

- README usa 100M V1/contexto 4096; AGENTS/config usam V9/contexto 5120.
- manifests de gold/readiness apontam para root legada ausente.
- status e manifests Smol divergem sobre candidato e rollback.
- registros aprovados de corpus apontam para `data/`, ausente no snapshot.

## 7. SHA-256 e rollback

SHA-256 é padrão criptográfico, não novidade do projeto. A aplicação local é
proveniência e integridade:

- `src/f51_darwin/hashing.py`: arquivo, tensor e JSON canônico;
- `research/scan_ffn_neurons.py`: checkpoint e canal FFN;
- `research/rome_proper_identity.py`: hash antes/depois e rollback;
- `research/transplant_circuit_atomic.py`: hash do estado do modelo;
- `src/f51_darwin/circuits/ledger.py`: cadeia de eventos;
- `src/f51_darwin/circuits/rollback.py`: snapshots e package hashes.

O hash detecta alteração do material hashado; não prova autoria, causalidade,
qualidade ou utilidade do circuito.

## 8. Pesquisa: claim ledger provisório

| Linha | Veredito atual |
|---|---|
| catálogo FFN com hashes | engenharia/proveniência observacional |
| ablação causal em probes | resultado causal específico |
| direção vence neurônio | refutado no protocolo auditado |
| SAE separa domínios | teste provisoriamente negativo e fraco |
| ROME edita identidade | demonstrado em suite estreita |
| ROME preserva tudo | não demonstrado |
| UniversalMemory é memória comportamental | não demonstrado |
| transplante transfere domínio | resultado negativo/inconclusivo |
| memória externa é melhor universalmente | hipótese, não lei |

Dados-chave: direction-vs-neuron em SmolLM cross `0,9715` versus self/noise
`0,9762`; TinyLlama `0,0711` versus `0,0609`; direction rank-8 collateral
`66,6%` versus neurônio `64,7%`. ROME alcançou 6/6 em identidade controlada,
com rollback, mas não prova preservação universal. UniversalMemory teve
armazenamento literal/persistência, mas o relatório auditado mostra 9/20 de
aceitação de paráfrases.

## 9. Learning-Gain Simulation

Arquivos: `research/learning_gain/`, `research/run_learning_gain_sim.py`,
`src/tests/test_learning_gain_sim.py` e `workspace/runtime/learning_gain_sim/`.

O código implementa estado em três escalas, quatro ações, cinco políticas e
dez cenários. A hipótese **não está validada**:

- run salvo com 1.500 linhas e 167 erros;
- S03 falha e S05 tem 120 erros de broadcast;
- `y` é entregue à política antes da decisão;
- S07 descarta o retorno de consequências atrasadas;
- S10 é substituído por S01;
- gates 4, 6 e 7 são placeholders;
- CI não é efeito pareado conforme o spec;
- provenance incompleta e exceções convertidas em rows sintéticas.

Classificação: `IMPLEMENTAÇÃO PARCIAL / HIPÓTESE NÃO VALIDADA`.

## 10. Testes, operação e blockers

A suíte possui 159 arquivos de teste, mas smoke de entrypoints não cobre todos
os PowerShell, CLI instalada, dashboard ou checkpoint real. Há mocks,
checkpoints sintéticos, skips de plataforma e source audit antigo.

Blockers P0:

- checkout aninhado não resolvido;
- linhagem V1/V9 e roots declaradas contraditórias;
- `start_100m_auto.ps1` pode lançar `run247` apesar da descrição de gate;
- source audit não vinculado ao HEAD e histórico de segredos mantém NOT GOLD;
- Learning-Gain sem run científico válido.

Blockers P1/P2: pointers/manifests antigos, dashboard sem prova ponta a ponta,
proveniência apontando para caminhos ausentes, lista de entrypoints divergente e
claims de pesquisa acima da evidência. Os dicionários arquivo-a-arquivo, de
símbolos e físico já foram gerados; eles não eliminam os blockers operacionais.

O teste `src/tests/test_provenance_certificate.py` cobre essa fronteira com 13
testes, mas teste e implementação ainda não equivalem a descoberta validada do
projeto.

Verificação focada de 2026-07-30: `python -B -m pytest -q
src/tests/test_provenance_certificate.py src/tests/test_transplant_16b_ledger.py
-p no:cacheprovider` executou 16 testes com PASS e encontrou 5 erros de
`PermissionError: [WinError 5]` ao preparar/limpar `tmp_path`. Os cinco casos
dependentes de diretório foram então chamados diretamente com diretórios
controlados e passaram. Resultado honesto: 21 comportamentos funcionais
passaram, mas o comando pytest não é verde por causa da infraestrutura
temporária.

## 11. Cobertura dos seis subagentes

| Domínio | Agente | Estado |
|---|---|---|
| topologia/governança | Erdos | recebido |
| modelo/treino/runtime | Zeno | recebido |
| dados/checkpoints/segurança | Confucius | recebido |
| pesquisa/experimentos | Chandrasekhar | recebido |
| Learning-Gain | Lovelace | recebido |
| testes/governance/docs/operação | Halley | recebido |

Os seis relatórios foram obtidos em modo read-only. Nenhum subagente editou
arquivos, lançou treino, canary, servidor, rede ou experimento pesado.

## 12. Critério de fechamento

Esta Bíblia só será marcada como auditada quando o dicionário arquivo-a-arquivo
dos 859 caminhos rastreados, do checkout aninhado e da fronteira física estiver
anexado, cada área pesada tiver manifest/hash/estado atual, cada claim tiver
artefato e cada divergência estiver resolvida ou explicitamente marcada como
`CONTRADITÓRIA`.

Comandos read-only usados: `git status --short`, `git branch --show-current`,
`git log --oneline -12`, `git ls-files`, `Get-ChildItem -File -Recurse -Force`
e análise AST estática.
