# Operação — prova causal de memória TTM nativa

## Objetivo

Este ensaio responde a uma pergunta estreita: depois de receber cinco
correções que o cérebro-base errou, o Darwin-Smol Native Dense V1 consegue
lembrá-las numa sessão nova usando somente o estado nativo do TTM?

O cérebro fica congelado. Não há backward, fine-tuning, RAG, cache de respostas,
edição de logits ou alteração do checkpoint. O grupo-controle recebe as mesmas
sequências de ensino, mas não grava slots.

## Pré-condições

- checkpoint e manifest em
  `workspace/03_CHECKPOINTS_1.7B_SMOL_DENSE_V1/`;
- duas GPUs CUDA locais disponíveis;
- nenhum treino Darwin concorrente;
- tokenizer Smol disponível no cache local;
- execução offline.

O executor valida o manifest, calcula o SHA-256 do checkpoint antes e depois,
carrega uma cópia fresca por dose e aborta se qualquer peso do cérebro mudar.

## Comandos

Ensaio completo:

```powershell
python research/benchmark_native_ttm_recall.py
```

Verificação somente leitura dos artefatos existentes:

```powershell
python research/benchmark_native_ttm_recall.py --verify-report
```

O verificador não carrega o modelo de 1,7B. Ele recalcula os hashes do
protocolo, da memória e do checkpoint e reclassifica o resultado.

## Protocolo congelado

- seed: `51`;
- cinco fatos sintéticos escolhidos entre erros reais do baseline;
- ensino associativo: chave no hidden bruto de `T-1` da pergunta literal e
  valor na média dos seis hidden states teacher-forced da resposta;
- doses efetivas em RAM: `0.00`, `0.01`, `0.03`, `0.10`, `0.30`;
- braços: sem memória, memória correta, memória desligada, valores
  embaralhados com as mesmas chaves e memória restaurada do disco;
- avaliações: pergunta literal, duas paráfrases e uma distração por fato;
- decodificação greedy e acerto apenas quando a saída inteira é o código de
  seis dígitos esperado.

O gate científico exige, na mesma dose:

- `5/5` erros no baseline;
- memória correta com pelo menos `4/5` literais, `6/10` paráfrases e `3/5`
  distrações;
- queda de pelo menos dois literais ao desligar ou embaralhar a memória;
- memória restaurada no máximo um literal abaixo da memória correta;
- recuperação aceita em pelo menos `4/5` perguntas literais;
- chaves preservadas no shuffle;
- checkpoint, hashes e versões do cérebro invariáveis.

## Artefatos

Diretório isolado:
`workspace/runtime/darwin_17b_smol_dense_v1/memory-recall/`

- `memory-state.pt`: cinco slots nativos serializados;
- `native-ttm-recall.json`: linhas completas, métricas, hashes e ambiente;
- `native-ttm-recall.md`: resumo legível;
- `partial.json`: último estágio concluído para diagnóstico.

## Classificações

- `causal_memory_recall_pass`: passou o gate causal completo;
- `retrieval_only`: o trace manual encontrou o domínio correto, mas ainda não
  há prova comportamental;
- `literal_only`: acertou repetição literal sem generalizar;
- `noncausal_gain`: o aparente ganho sobreviveu aos controles;
- `no_memory_effect`: não produziu ganho causal de resposta; o campo
  `failure_mode` registra se falhou na recuperação, no canal ou no valor;
- `invalid_protocol`: isolamento, integridade ou artefatos falharam.

O diagnóstico pode acrescentar uma causa mecânica sem mudar essa taxonomia:
`native_retrieval_absent`, `native_injection_misses_decoder`,
`native_channel_reaches_decoder`, `value_not_behaviorally_decodable` ou
`inconclusive_decoder_floor`.

Um processo completo pode retornar zero e ainda reprovar cientificamente. O
sucesso operacional significa que o protocolo terminou e os artefatos são
válidos; sucesso de memória exige exclusivamente
`causal_memory_recall_pass`.

## Resultado medido após a cirurgia de canal em 2026-07-29

Classificação científica: `no_memory_effect`; causa:
`value_not_behaviorally_decodable`; gate estrito falso.

A cirurgia sem treino corrigiu os dois bloqueios do caminho nativo:

- escrita e leitura agora usam vetores `[1,D]` do mesmo `T-1`;
- a recuperação top-1 injeta somente em `T-1`, com residual limitado à norma
  do hidden e dose máxima `0,30`.

O controle de piso repetiu `5/5` quando o fato foi declarado no prompt. As cinco
consultas nativas foram aceitas, o residual foi aplicado nas cinco e a diferença
máxima nos logits ficou entre `5,0` e `5,5`; a razão efetiva entre residual
escalado e hidden ficou entre `0,1864` e `0,1888`. O canal, portanto, alcança o
decoder.

Isso ainda não é lembrança. Em todas as doses, a memória correta acertou
`0/5` perguntas literais. Em dose `0,30`, também acertou `0/10` paráfrases e
`0/5` distrações. O top-1 escolheu o domínio correto em somente `1/5` perguntas:
as cinco similaridades BF16 foram exatamente `1,0`, mostrando colapso/empate
das chaves sob esse template. Além disso, mesmo no único top-1 correto, o valor
injetado não mudou o argmax do primeiro token para a resposta esperada.

Logo, a incompatibilidade pooled-versus-token e a injeção fora de `T-1` foram
removidas. Restam dois bloqueios medidos: chaves sem discriminação entre fatos e
valor hidden não decodificável como a resposta ensinada. Nenhum peso do cérebro
ou checkpoint mudou.

Hashes atuais:

- checkpoint:
  `9329b8d25fc23acde021161da7a3421e6e17ceea3e940e0298162f39b62bc01e`;
- protocolo:
  `559f60912687bc4437ed5a73abb4dd4ec6e77a3c6f4889f6804eb3e7de5e3bf0`;
- memória:
  `0da6938da286e3e61935a0ac70988af8dc9bd7e6b456a1b97668c06534b037fe`;
- diagnóstico:
  `66f61b68681a1e1e04b4f300de6fec21b288e1501b6b54ca9e76c1b6e4494943`.

Artefatos adicionais: `native-ttm-diagnostic.json` e
`native-ttm-diagnostic.md` no mesmo diretório.

## Rollback

O ensaio não publica checkpoint. Para remover apenas seus resultados, apague
exclusivamente
`workspace/runtime/darwin_17b_smol_dense_v1/memory-recall/`.
O checkpoint de 4,09 GB não faz parte do rollback.
