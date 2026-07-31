# Native TTM Memory Recall — desenho experimental

## Objetivo final

Provar causalmente se o órgão TTM do Darwin-Smol Native Dense V1 consegue
guardar cinco fatos sintéticos que o cérebro-base desconhece e fazer o mesmo
cérebro responder corretamente em uma sessão nova, sem alterar os pesos do
cérebro.

O experimento é para Marco Barreto/Fuch F51 Labs e produz um relatório local
reproduzível, não um novo checkpoint publicado.

## Fluxo principal

1. Carregar e verificar o checkpoint aprovado
   `workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/organism_cycle_000.pt`.
2. Gerar candidatos sintéticos determinísticos e perguntar ao cérebro-base.
3. Selecionar os primeiros cinco fatos cujas respostas secretas não apareçam
   na resposta-base.
4. Executar a mesma fase de ensino nos dois braços, sem backward:
   - braço A processa pergunta e correção, mas não grava TTM;
   - braço B processa a mesma pergunta e correção e grava o hidden final no
     TTM.
5. Serializar somente o estado Heartbeat/TTM do braço B.
6. Destruir os modelos de ensino, recarregar o checkpoint original e criar uma
   sessão nova.
7. Avaliar os controles causais com cérebro congelado:
   - `no_memory`: checkpoint limpo, sem slots;
   - `correct_memory`: slots persistidos e gate TTM aberto somente na RAM;
   - `memory_disabled`: mesmos slots, escala residual zero;
   - `shuffled_memory`: mesmas chaves, valores permutados entre fatos;
   - `restored_memory`: estado correto restaurado após o controle embaralhado.
8. Persistir JSON, Markdown e estado TTM isolado em
   `workspace/runtime/darwin_17b_smol_dense_v1/memory-recall/`.

## Dados sintéticos

Os fatos usam identificadores, entidades e códigos gerados por seed fixa. Eles
não representam conhecimento público e não podem ser inferidos semanticamente.
Cada item possui:

- identificador estável;
- frase de ensino;
- pergunta literal;
- duas paráfrases;
- pergunta com distração;
- resposta secreta exata.

O pool deve ser maior que cinco. Um item só entra no conjunto final quando a
resposta-base não contém a resposta secreta normalizada. O conjunto final e o
hash do protocolo entram no relatório.

## Anatomia testada

O teste usa `Heartbeat.tt_memory` e o caminho residual nativo de
`DarwinXModel`. Não vale:

- anexar a resposta recuperada ao prompt;
- consultar um dicionário de respostas;
- alterar logits diretamente;
- usar RAG externo;
- executar backward ou optimizer;
- alterar o checkpoint publicado.

Como a linhagem publicada fixa `ttm_residual_max_scale: 0.0`, o braço
`correct_memory` usa uma cópia de config somente em RAM com escala máxima
predeclarada. O benchmark registra a escala efetiva. Uma curva de doses fixa
`0.00, 0.01, 0.03, 0.10, 0.30` é reportada integralmente; nenhum valor é
escolhido ou escondido depois de observar as respostas.

## Isolamento e persistência

- As avaliações usam decodificação greedy e o mesmo template do QA aprovado.
- Todos os parâmetros do cérebro permanecem `requires_grad=False`.
- O benchmark calcula SHA-256 tensorial do cérebro antes do ensino e depois de
  cada braço.
- A fase de teste recarrega o checkpoint do disco; não reutiliza o objeto de
  ensino.
- O único estado transferido para a sessão nova é o estado Heartbeat/TTM
  serializado.
- Cada braço começa da mesma sessão nova para impedir vazamento de slots ou
  contadores.
- Nenhum trainer concorrente pode estar ativo.

## Métricas

As métricas primárias são:

- `exact_recall`: acertos nas cinco perguntas literais;
- `paraphrase_recall`: acertos nas dez paráfrases;
- `distractor_recall`: acertos nas cinco perguntas com distração;
- `causal_drop`: queda entre `correct_memory` e os controles
  `memory_disabled`/`shuffled_memory`;
- `restoration_recovery`: recuperação ao restaurar a memória correta.

Métricas mecânicas obrigatórias:

- slots escritos e restaurados;
- slot/top-k recuperado por pergunta;
- similaridade da recuperação;
- escala residual efetiva;
- hash do cérebro antes/depois;
- latência por pergunta;
- ambiente de hardware/software;
- seed, checkpoint SHA-256 e protocolo SHA-256.

## Gate de aprovação

O resultado só é `causal_memory_recall_pass` se todos os requisitos abaixo
forem verdadeiros:

1. O baseline erra os cinco fatos selecionados.
2. `correct_memory` alcança pelo menos 4/5 literal, 6/10 paráfrases e 3/5 com
   distração em pelo menos uma dose predeclarada.
3. `memory_disabled` e `shuffled_memory` ficam pelo menos dois acertos literais
   abaixo de `correct_memory` na mesma dose.
4. `restored_memory` recupera pelo menos os acertos literais de
   `correct_memory` menos um.
5. Pelo menos quatro das cinco perguntas recuperam uma memória e o controle
   embaralhado preserva as decisões de chave enquanto troca os valores.
6. O hash tensorial do cérebro é idêntico em todos os braços.
7. Nenhum NaN/Inf, trainer concorrente ou mutação do checkpoint é observado.

Qualquer resultado intermediário recebe um rótulo explícito:

- `retrieval_only`: a chave certa é recuperada, mas a resposta não melhora;
- `literal_only`: melhora literal sem generalização para paráfrases;
- `noncausal_gain`: melhora que não desaparece nos controles;
- `no_memory_effect`: não há ganho comportamental;
- `invalid_protocol`: isolamento, hashes ou artefatos falharam.

## Arquivos

- Criar `src/scripts/benchmark_native_ttm_recall.py`: operador do experimento e
  geração dos artefatos.
- Criar `src/tests/test_native_ttm_recall.py`: testes determinísticos da seleção de
  fatos, permutação de valores, classificação e hash cerebral.
- Criar `governance/docs/operacao/NATIVE_TTM_RECALL.md`: comando de reprodução e leitura
  dos rótulos.
- Produzir, sem versionar, os artefatos pesados em
  `workspace/runtime/darwin_17b_smol_dense_v1/memory-recall/`.

## Comando operacional

```powershell
python src/scripts/benchmark_native_ttm_recall.py
```

O processo termina com `NATIVE_TTM_RECALL_OK classification=<rotulo>` e código
zero quando o protocolo foi executado integralmente. O rótulo pode ser uma
reprovação científica; código zero não significa que a memória passou.
