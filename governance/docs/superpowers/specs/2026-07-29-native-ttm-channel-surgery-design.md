# Design — cirurgia do canal TTM nativo

Data: 2026-07-29. Escopo: Darwin-Smol Native Dense V1, checkpoint
`organism_cycle_000.pt`.

## Objetivo

Corrigir os três bloqueios medidos no TTM sem treinar ou alterar o cérebro:

1. escrita pooled e leitura por token vivem em espaços incompatíveis;
2. a injeção esparsa não alcança a posição que produz o próximo token;
3. as chaves atuais não discriminam os cinco fatos.

O mesmo protocolo causal de cinco fatos continua sendo o gate. O resultado só
passa se memória correta melhorar e os braços desligado e embaralhado perderem
o ganho.

## Alternativas consideradas

### A. Associação vetorial alinhada — escolhida

Escrever explicitamente uma associação com fontes separadas:

- chave: hidden bruto da última posição do prompt antes da resposta;
- valor: representação do trecho de resposta observado na sequência de ensino.

Na leitura, consultar com o hidden bruto de `T-1`, usar top-1, limitar a norma do
valor à norma do hidden de referência e injetar somente em `T-1`.

Essa opção preserva a anatomia TTM: slots vetoriais, cérebro congelado e nenhum
texto ou token de resposta armazenado.

### B. Memória de sequência de tokens — rejeitada

Guardar os seis tokens e reproduzi-los por cursor teria alta chance de acerto,
mas seria cache de resposta. Isso violaria o protocolo sem RAG/cache e não
provaria uma memória neural residual.

### C. Fast weights durante inferência — adiada

Otimizar um adaptador TTM contra a resposta é adaptação controlada legítima, mas
introduz backward e outra hipótese. Só será considerado se a associação
vetorial chegar ao decoder e ainda não carregar informação útil.

## Arquitetura

### Escrita

`TestTimeMemory` recebe uma operação explícita de associação entre `key_source`
e `value_source`. As duas entradas já estão pooled no formato `[B, D]`.

- `key_source` passa por `proj_key`;
- `value_source` é armazenado no espaço residual de dimensão `D`, sem passar
  pelo `proj_value` aleatório do cycle 0;
- cada fato produz exatamente um slot;
- shapes, domínio, surpresa, timestamp e persistência continuam compatíveis.

O benchmark captura o hidden bruto na saída de `model.norm`:

1. forward do prompt sem resposta e sem memória para obter a chave em `T-1`;
2. forward da sequência de ensino completa para obter o valor médio somente no
   trecho da resposta;
3. gravação da associação depois dos dois forwards.

### Leitura e injeção

Quando `ttm_entity_addressing=true`, o TTM deixa de consultar quatro posições
Spider para o readout da linguagem. Ele consulta uma vez com
`hidden[:, -1, :]`, usando `already_pooled=True` e `top_k=1`.

Se houver match:

1. o valor é convertido para device/dtype do hidden;
2. sua norma é limitada à norma do hidden de `T-1`;
3. `residual_scale * bounded_value` é somado somente em `T-1`;
4. `ttm_memory_retrieved` e `ttm_residual_applied` refletem o caminho real.

O Spider continua disponível para observação/escrita geral, mas não decide mais
se a memória alcança o token de resposta.

## Isolamento e compatibilidade

- nenhum parâmetro novo;
- nenhuma mudança de shape no checkpoint;
- gates existentes continuam controlando a dose;
- slots antigos ainda carregam;
- modo sem `ttm_entity_addressing` permanece inalterado;
- alterações JEPA não commitadas em `config.py` e `model.py` não pertencem a
  esta cirurgia e serão preservadas fora dos commits TTM.

## Segurança numérica

O valor recuperado é limitado por amostra:

`bounded = value * min(1, ||hidden_T-1|| / (||value|| + eps))`

Com dose máxima `0,30`, a perturbação não excede aproximadamente 30% da norma
do hidden de decisão. NaN, Inf, shape incompatível ou norma não finita abortam
o braço.

## Testes

### Unitários

- associação usa fontes de chave e valor separadas;
- top-1 retorna o slot semanticamente correspondente;
- bounding nunca excede a norma de referência;
- injeção altera somente `T-1`;
- gate zero preserva logits;
- serialização/restauração mantém as associações.

### Diagnóstico real

O probe deve mostrar, nos cinco prompts:

- recuperação nativa em pelo menos 4/5;
- `T-1` como posição efetivamente injetada em 5/5;
- diferença de logits não zero quando a dose é positiva;
- cérebro invariável.

### Gate comportamental

Reexecutar `benchmark_native_ttm_recall.py` sem relaxar:

- baseline: cinco erros;
- memória correta: pelo menos 4/5 literal, 6/10 paráfrase e 3/5 distração;
- braços desligado e embaralhado: queda causal mínima de dois literais;
- memória restaurada: no máximo um literal abaixo da correta;
- hashes e versões do cérebro invariáveis.

Se o canal passar e o comportamento não melhorar, a classificação permanece
`no_memory_effect` com causa `value_not_behaviorally_decodable`; não haverá
promoção por mera mudança de logits.

## Rollback

Os commits da cirurgia serão isolados. O checkpoint publicado nunca é
reescrito. O rollback de código reverte somente esses commits; os artefatos
novos ficam no diretório isolado de memory-recall e podem ser removidos sem
afetar a linhagem.
