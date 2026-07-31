# Darwin-X 100M Cloud V2: Motor Corrigido e Dopamina por Progresso

## Objetivo final

Construir e treinar uma nova linhagem Darwin-X 100M na A100 da Vast sem
alterar o modelo, os processos ou os checkpoints locais administrados por
Marco.

O sistema é um modelo de linguagem causal híbrido SSD/GQA com MoE. Ele é
destinado ao laboratório F51 para testar se representação preditiva JEPA e
regulação por progresso de aprendizagem melhoram aquisição e retenção sem
corromper a função principal de linguagem.

O fluxo principal é:

1. preparar uma fonte V2 isolada e reproduzível;
2. provar as correções do motor em CPU e CUDA;
3. executar canários pareados com o mesmo corpus, ordem, seed e orçamento;
4. promover o controlador de `shadow` para `enforce` somente após os gates;
5. iniciar um `train-budget` finito numa raiz de checkpoint inédita;
6. acompanhar PID, GPU, métricas, ledger, holdout e checkpoints por identidade.

O comando final será um launcher Linux dedicado, que invoca
`python -u -m scripts.darwin_organism train-budget` com a configuração
`src/configs/darwin_x_100m_cloud_v2.yaml`.

O treino só é considerado iniciado quando houver, ao mesmo tempo, PID remoto
vivo, uso de VRAM na A100, log avançando, `run_id` único, manifesto de
implantação e `lineage_root.json` compatível. Iniciar um processo não é prova
de qualidade nem de superioridade.

## Limites de autoridade e isolamento

- O checkout ativo `C:\Users\marco\Desktop\F51-Darwin-SSD`, seu treino local e
  `workspace\03_CHECKPOINTS_100M_FULL_ORGANISM_V1` não serão interrompidos,
  modificados ou reutilizados.
- A implementação ocorrerá numa branch/worktree isolada, derivada do commit
  auditado. O arquivo local preexistente `src/scripts/weight_sim.py` não entra no
  commit nem no pacote.
- A instância alvo é a Vast `45934537`, A100 SXM4 40 GB, acessada por
  `ssh6.vast.ai:14536`.
- O código será publicado em `/root/f51-cloud-v2/source`, nunca sobre
  `/root/f51`.
- O corpus remoto existente será montado somente para leitura a partir de
  `/root/f51/workspace/01_TOKENIZADOS/00_CORPUS_PRINCIPAL_tokens_feast_v2.bin`.
- A linhagem usará uma raiz nova:
  `/root/f51/workspace/03_CHECKPOINTS_100M_CLOUD_V2`.
- Canários usam raízes descartáveis separadas da raiz longa.
- Nenhum checkpoint de ablação antigo será promovido por nome ou data.
- O launcher longo recusa processo Darwin concorrente. Como Marco controla o
  treino local, o agente apenas verifica o gate; não encerra processos locais.
- Segredos e chaves SSH não entram no pacote, manifesto ou Git.

## Plano de controle tipado

O modelo é dividido em quatro planos. Cada módulo tem uma autoridade explícita.

### 1. Cognição

Participa do `forward`, produz logits ou losses differentiáveis e pode receber
gradiente:

- embedding e LM head;
- blocos SSD e atenção GQA;
- MoE com especialistas finos e compartilhados;
- LM loss;
- JEPA online predictor e seu loss auxiliar;
- projeções necessárias para os objetivos cognitivos.

Cognição não lê arquivos operacionais, não escolhe checkpoints e não altera
parâmetros fora do otimizador.

### 2. Regulação

Observa sinais destacados do grafo e só intervém por interfaces limitadas:

- controlador de progresso de aprendizagem;
- traços de elegibilidade por camada/especialista;
- bias externo do router, limitado e de soma zero;
- prioridade de replay;
- máscara de slots ativos e transições do ciclo de vida dos especialistas;
- modo causal `disabled`, `control`, `shadow` ou `enforce`.

Regulação não adiciona um escalar de “dopamina” à loss global. Ela influencia
qual especialista e qual experiência serão escolhidos numa atualização futura.

### 3. Telemetria

É somente leitura em relação à aprendizagem:

- métricas JSONL;
- atribuição de gradiente;
- estatísticas de routing e load balance;
- probes de aprendizagem;
- holdout oficial;
- ledger causal;
- processos, GPU, throughput, disco e checkpoints.

Telemetria recebe tensores destacados ou escalares serializados. Nenhuma
métrica volta ao grafo por acidente.

### 4. Decoração

Nomes biológicos, dashboard, narrativas, blockchain e sinais sem intervenção
causal ficam fora da função de treinamento. Eles podem exibir telemetria, mas
não podem alterar loss, optimizer, router, replay, memória ou checkpoint.

## Gate 0: correções obrigatórias do motor

Essas correções são pré-condições. Elas não serão anunciadas como ganho
cognitivo.

### Inicialização SSD

`dt_proj.weight` conservará a inicialização canônica definida pelo SSD.
`_init_weights` respeitará uma marca de não reinicialização tanto no módulo
quanto no peso. Um teste medirá o intervalo/desvio esperado e falhará se a
inicialização genérica de `0.02` sobrescrever o tensor.

### RoPE

Treino, holdout e inferência dentro do contexto treinado usarão a mesma base
RoPE. Uma base estendida só poderá ser escolhida por uma política explícita de
long-context, nunca pelo simples estado `model.training`.

A V2 fixa `rope_base=10000` para contexto de treino e holdout. Avaliações com
base diferente serão rotuladas `long_context_eval` e não poderão ser
comparadas diretamente ao holdout oficial.

### Inferência SSD

O engine atual não possui estado recorrente suficiente para um decode SSD
token a token equivalente. A V2 não fingirá que um token isolado contém o
histórico.

Para o primeiro lançamento:

- blocos de atenção continuam usando KV cache;
- se houver bloco SSD e não existir estado SSD validado, o engine usa
  recomputação da janela completa;
- um caminho recorrente só pode ser ativado quando armazenar estado do scan e
  histórico da convolução e passar equivalência de logits token a token.

Essa política privilegia correção sobre velocidade e não bloqueia o treino.

### Limite de contexto em `live_generate`

A janela será truncada ou deslizada em toda iteração, antes do `forward`, e
nunca apenas no prompt inicial. O teste gera além do limite e prova ausência de
overflow com política determinística.

### Router e `expert_bias`

O bias regulatório deixa de ser `nn.Parameter`. Ele será buffer persistente
gerenciado por uma API que aplica:

- atualização sem gradiente;
- centralização para soma zero;
- clipping configurável;
- decaimento;
- registro causal da origem e do modo.

O router treinável continua sendo otimizado por AdamW. Mutação aleatória
direta do bias durante `forward` é removida. Isso evita que os momentos do
otimizador descrevam um tensor diferente do usado pelo router.

### Loss auxiliar MoE

O compositor expõe a loss auxiliar por camada e uma redução explícita
`mean_layers`. A V2 usa média, de modo que aumentar o número de camadas MoE não
multiplique silenciosamente a força do regularizador. Os pesos de load balance
e router-z permanecem configuráveis e aparecem separados na telemetria.

### Especialistas compartilhados

`mean` e `sum` serão políticas explícitas, testáveis e gravadas na identidade
da linhagem. A correção do motor não presume que uma delas melhora perplexity.

A V2 começa com `mean`, preservando a escala residual conhecida. `sum` entra
somente como ablação pareada; se vencer em LM, estabilidade e balanceamento,
ganha uma nova identidade de config. Não haverá mudança silenciosa de escala
na linhagem longa.

## JEPA V2

JEPA continua auxiliar à LM; JEPA-only é proibido nesta linhagem.

### Representações

- A fonte online é o estado cognitivo antes de qualquer memória TTM residual.
- O predictor online estima a representação futura escolhida pela máscara
  causal.
- O alvo usa uma cópia EMA não treinada por gradiente.
- O alvo é destacado antes do cálculo da distância.
- O estado EMA é atualizado após o optimizer step e salvo no checkpoint.
- Ghost e outros órgãos não compartilham nem otimizam o predictor JEPA.
- Não existe alias duplicado `jepa_ref` registrado no `state_dict`.

Para limitar custo, o primeiro desenho usa um target projector EMA sobre
representações limpas do backbone. Ele será descrito honestamente como
`EMA-target JEPA auxiliary`, não como uma reprodução integral de I-JEPA. Uma
cópia EMA do backbone inteiro exige ablação separada por dobrar o forward do
encoder.

### Loss e conflito de gradiente

O loss reporta separadamente:

- distância preditiva;
- variância;
- covariância;
- peso efetivo;
- norma e cosseno de gradiente LM/JEPA em probes.

O heartbeat/TTM não recebe o composto JEPA como “surpresa”. Quando um sinal
preditivo for necessário, usa apenas a distância destacada, normalizada por
baseline móvel. PCGrad ou GradNorm não entram por padrão; serão ativados apenas
se o canário medir conflito persistente e benefício sobre o controle.

O multiplicador de LR JEPA volta a `1.0` na V2 inicial. Multiplicadores `3.0` e
`10.0` são hipóteses de ablação, não defaults herdados.

## Dopamina por progresso real de aprendizagem

“Dopamina” é definida operacionalmente como erro de previsão do progresso de
aprendizagem, não como loss, surpresa bruta, ruído ou humor global:

`delta_e = progresso_observado_e - progresso_esperado_e`

para cada camada MoE e especialista `e`.

### Banco de probes

- Probes vêm de uma partição de treino versionada, separada do holdout oficial.
- O banco é rotativo e determinístico por seed.
- Nenhum exemplo do holdout pode entrar no controlador, replay ou estimador de
  progresso.
- A cada intervalo, o modelo avalia o mesmo microprobe antes de reutilizá-lo
  como referência temporal.
- O progresso observado combina queda de LM ponderada pelo gate, melhora da
  distância JEPA e penalidade de esquecimento.

Uma forma inicial normalizada é:

`P_e = z(queda_lm_e) + w_jepa*z(queda_jepa_e)
       - w_forget*z(esquecimento_e) - w_load*z(desequilibrio_e)`

Os pesos, janelas e versões fazem parte da identidade da configuração.

### Elegibilidade

O forward expõe massa de gate destacada por token, camada e especialista.
Cada especialista mantém um EMA de elegibilidade que representa participação
real nas experiências vistas desde o probe anterior.

O crédito é:

`credito_e = clip(delta_e * elegibilidade_e, -c, c)`

Especialista que não participou recebe crédito zero. O crédito não altera
pesos diretamente.

### Ação regulatória

Em `shadow`, a ação é calculada e registrada, mas não aplicada.

Em `enforce`, a ação futura pode:

1. ajustar o bias externo do router com soma zero e clipping;
2. priorizar experiências learnable no replay;
3. reduzir prioridade de fontes com surpresa alta e progresso não positivo.

O controlador nunca multiplica a loss global por dopamina e nunca chama
`optimizer.step`.

### Curiosidade e proteção contra noisy TV

Curiosidade propõe candidatos por novidade e incerteza. Dopamina confirma se
eles foram aprendíveis. Uma fonte que mantém alta surpresa sem progresso por
`K` probes recebe decaimento de prioridade e entra em quarentena. Isso impede
que ruído imprevisível seja confundido com conhecimento valioso.

### Controle negativo

O canário inclui `shuffled_dopamine`: os deltas corretos são permutados entre
especialistas mantendo a mesma distribuição. Se o enforce real não superar
esse controle em progresso, LM e retenção, a hipótese causal falha e o treino
longo permanece bloqueado.

## Ciclo de vida dos especialistas

Parâmetros não serão criados ou destruídos no meio de um optimizer step.
Cada camada possui capacidade prealocada com slots `active`, `probation`,
`dormant` e `archived`.

### Nascimento

Um novo especialista só entra em `probation` quando coexistirem:

- sobrecarga persistente de roteamento;
- progresso positivo num domínio/probe;
- capacidade ociosa;
- ausência de regressão no holdout;
- orçamento de nascimentos disponível.

O slot nasce por clone controlado de um especialista elegível mais pequena
perturbação, com optimizer state reinicializado apenas para o novo slot. O
evento registra origem, seed, métricas e hash.

### Promoção

O slot em probation recebe tráfego limitado. Ele vira `active` somente se
melhorar progresso e balanceamento sem piorar LM/holdout além dos limites.

### Morte sem destruição

Especialista sem uso, sem progresso ou com dano persistente é roteado para
`archived`. Seus tensores permanecem no checkpoint para rollback e auditoria;
o active mask impede tráfego. Um slot arquivado só pode ser reciclado numa
fronteira de ciclo, com evento explícito e optimizer state reinicializado.

Na primeira V2 longa, o lifecycle fica `shadow` até haver evidência de canário.
Dopamina em `enforce` não implica automaticamente nascimento ou morte.

## Checkpoint e identidade

O checkpoint causal V2 inclui:

- pesos cognitivos;
- optimizer e scheduler;
- target EMA JEPA;
- expected progress e histórico dos probes;
- elegibilidade por especialista;
- bias regulatório externo;
- replay priorities;
- active/probation/dormant/archive masks;
- ledger causal e versão do controlador;
- contador real de tokens;
- hash da config, corpus, tokenizer, fonte e pacote implantado.

Resume rejeita ausência ou incompatibilidade desses campos. Migração de
checkpoint antigo só pode produzir uma raiz de experimento separada; a V2
longa começa fresh.

## Telemetria mínima

Cada optimizer step registra LM, JEPA por componente, MoE aux por componente,
grad norm before/after, update applied e throughput.

Cada probe registra:

- `progress_observed`, `progress_expected` e `delta` por especialista;
- elegibilidade;
- bias proposto/aplicado;
- modo causal;
- replay priority;
- noisy-TV score;
- mudanças de lifecycle;
- LM/JEPA/forgetting/load separados.

Cada checkpoint registra identidade e hash. O dashboard apenas lê esses
eventos.

## Matriz de prova

Todos os canários usam o mesmo snapshot de fonte, config cognitiva, corpus,
ordem de blocos, batch, seed, número de updates e orçamento de parede:

1. `lm_control`: LM, motor corrigido, sem JEPA/dopamina;
2. `lm_jepa`: LM + EMA-target JEPA, dopamina disabled;
3. `dopamine_shadow`: calcula controller sem intervir;
4. `dopamine_enforce`: aplica bias/replay limitados;
5. `shuffled_dopamine`: mesma magnitude, crédito permutado.

São necessárias pelo menos três seeds para uma alegação de ganho. Um canário
de integração de uma seed pode autorizar apenas o início cauteloso de treino,
não uma conclusão científica.

### Gates obrigatórios

- testes de inicialização SSD, RoPE e contexto;
- equivalência de logits do engine; recomputação correta é aceita;
- ausência de `expert_bias` no optimizer;
- soma zero, clipping e determinismo do bias regulatório;
- nenhum vazamento do holdout para probes/replay;
- checkpoint round-trip integral do controller e EMA;
- gradientes finitos e nenhum `applied=False`;
- identidade estrita de config/corpus/tokenizer/source;
- GPU e disco suficientes;
- nenhum treinador Darwin concorrente no gate final;
- fonte remota igual ao manifesto local por SHA-256.

## Implantação e lançamento

Antes da implantação, a versão cloud antiga será congelada num arquivo local
de comparação. O arquivo inclui o source remoto, configs, logs, métricas,
ledgers, pointers, lineage roots e somente os checkpoints finais resolvidos
pelos pointers. Checkpoints intermediários não serão baixados. Cada arquivo
material terá tamanho e SHA-256 verificados no remoto e novamente no local.
Braços incompletos serão preservados e rotulados `incomplete`, nunca
comparados como se tivessem atingido o mesmo orçamento.

O pacote é criado a partir de um commit limpo da worktree V2. Um manifesto
lista todos os arquivos e SHA-256. A transferência vai para um diretório
temporário remoto; hashes são verificados antes de uma troca atômica para
`/root/f51-cloud-v2/source`.

O launcher:

- valida instância/GPU;
- valida corpus, tokenizer e manifest por tamanho e hash;
- valida raiz fresh vazia;
- recusa treinador concorrente;
- executa o source gate e os canários;
- grava comando, PID, commit, hashes, config identity, CUDA/PyTorch e logs;
- inicia `train-budget` com `nohup` e PID file;
- espera evidência de vários optimizer steps e de um checkpoint válido.

Falha em qualquer gate mantém a A100 ociosa e preserva logs para diagnóstico.
O launcher não sobrescreve `/root/f51`, não usa `run247` e não apaga
checkpoints antigos.

## Critérios de pronto

A implementação está pronta para treino quando:

- todos os testes focados e o source audit passam;
- os cinco canários produzem losses finitas e checkpoints inspecionáveis;
- `dopamine_enforce` respeita todos os invariantes;
- o controle embaralhado não explica sozinho o efeito observado;
- o bundle remoto corresponde ao commit/manifesto;
- a raiz longa é inédita e o treino único está garantido.

O lançamento está concluído quando PID, VRAM, log, run metadata,
`lineage_root.json` e primeiro checkpoint concordam. Qualidade é avaliada
depois por holdout e provas multi-seed; não é inferida do início do processo.

## Base científica

- I-JEPA, arXiv:2301.08243.
- data2vec, arXiv:2202.03555.
- ProteinJEPA, arXiv:2605.07554.
- LLM-JEPA, arXiv:2509.14252.
- Curiosity-driven Exploration by Self-supervised Prediction,
  arXiv:1705.05363.
- Curiosity in Hindsight, arXiv:1810.06284.
- DeepSeek-V3 auxiliary-loss-free load balancing, arXiv:2412.19437.
- Expert Choice Routing, arXiv:2202.09368.
- PCGrad, arXiv:2001.06782.
- GradNorm, arXiv:1711.02257.

Esses trabalhos justificam hipóteses e controles; nenhum deles prova que a
combinação F51 melhora o modelo antes dos canários e da avaliação multi-seed.
