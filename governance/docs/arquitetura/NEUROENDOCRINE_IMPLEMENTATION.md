# Darwin-X neuroendócrino — estado científico verificável

Atualizado em 2026-07-13. Este documento descreve o código em disco; o treino
dual-GPU iniciado às 05:27 carregou uma versão anterior em memória.

## Objetivo do sistema

O Darwin-X é um modelo causal híbrido SSD/atenção com camadas MoE. A extensão
neuroendócrina investiga se sinais locais e temporais podem modular experts e
propor mudanças estruturais sem depender diretamente da loss global.

Fluxo real de treino:

`tokens -> forward/router -> loss -> backward/sinais locais -> optimizer.step -> decisões seguras -> checkpoint/eval`

O comando operacional e `python -m scripts.darwin_organism run247`. A
prova de funcionamento é loss finita, gradientes e updates reais, checkpoints
retomáveis, métricas por expert e testes causais das intervenções.

## Invariante de segurança

O backward hook observa gradientes, uso e entropia e atualiza o estado hormonal.
Ele não pode criar, remover, redimensionar ou mover parâmetros. Mudanças desse
tipo dentro do backward quebram o grafo, o estado do optimizer e o contrato do
checkpoint.

Depois de `optimizer.step()`, `apply_pending_autonomic_actions()` consome as
propostas. Na implementação atual somente ações que preservam shapes podem ser
aplicadas. Neurogenese, expansão e apoptose são registradas como
`proposed_not_applied`; ainda não são plasticidade estrutural válida.

## Mecanismos auditados

| Mecanismo | Sinal implementado | Ponto forte | Limite/correção |
|---|---|---|---|
| Dopamina | gradiente do expert menos EMA local | modulação rápida por expert | gate antigo começava em 0,5 e só atenuava; agora é neutro em 1,0 e limitado a 0,5–1,5 |
| Norepinefrina | surpresa positiva da entropia contra EMA da própria camada | distingue mudança de regime | entropia absoluta alta não é novidade; agora uma distribuição estável de alta entropia produz sinal zero |
| Cortisol | gradiente upstream e dispersão entre experts | sinal local de instabilidade | tendência de loss tinha sinal invertido; corrigida. No caminho estritamente local não há alegação de medir estagnação global |
| BDNF | uso e gradiente suavizados | candidato a detectar pressão de capacidade | só propõe expansão; falta cooldown, budget e prova de ganho marginal |
| Pressão de sono | processo homeostático limitado em [0,1) | limiar agora é matematicamente alcançável | o nome de buffer `ach_pressure` é legado; a analogia correta é adenosina-like, não acúmulo de acetilcolina |
| Vínculo vertical | uso do batch anterior da camada precedente | cria hipótese de continuidade entre routers | antes alterava permanentemente `expert_bias`; agora o viés é centrado, transitório e cruza a fronteira dual-GPU |
| Traços de roteamento | índices top-k amostrados deterministicamente | telemetria reprodutível | não contêm input/target e portanto não são replay de treino |
| Poda sináptica | menor magnitude absoluta | operação determinística testável | bug `num_pr` corrigido; continua experimental e fora do ciclo automático porque zeros podem voltar no optimizer |
| Fusão | comparação experimental de saídas em inputs comuns | critério funcional é melhor que BDNF escalar | fusão destrutiva está desativada; faltam conjunto held-out, teste de equivalência e proteção contra shapes distintos |
| Neurogenese/expansão/apoptose | limiares hormonais | gera hipóteses estruturais auditáveis | substitui Parameters, invalida AdamW e muda shapes de checkpoint; não é executada automaticamente |

## O que não deve ser afirmado

- O sistema não provou neurogenese útil.
- Traços de roteamento não são replay ou consolidação de memória.
- Uma analogia com hormônios não demonstra plausibilidade neurobiológica.
- “Activation Signature Matching (Liu 2025, rejected ICML)” não foi localizado
  em busca acadêmica e não é uma referência aceitável.
- O conjunto não é inédito só por combinar Mamba/SSD, atenção e MoE.

## Hipótese de ineditismo

Componentes amplos têm antecedentes: arquiteturas híbridas Mamba/MoE, MoE com
número adaptativo de experts e plasticidade neuromodulada já foram publicados.
A hipótese estreita ainda defensável é:

> Uma camada MoE localmente autorregulada em que amplitude dos experts e
> propostas de ciclo de vida estrutural são controladas conjuntamente por
> estados neuromoduladores multiescala derivados de surpresa de roteamento,
> uso e gradientes locais.

Isso é uma hipótese de composição/mecanismo, não prova de novidade mundial. Ela
precisa de busca bibliográfica/patentes mais ampla e de resultados de ablação.

Antecedentes mínimos:

- Jamba: híbrido Mamba/Transformer com MoE — https://arxiv.org/abs/2403.19887
- DynMoE: ajuste adaptativo do número de experts — https://arxiv.org/abs/2405.14297
- Backpropamine: plasticidade diferenciável neuromodulada — https://arxiv.org/abs/2002.10585
- Evolved Mixture Model: expansão por mudança de distribuição — https://arxiv.org/abs/2207.05080

## Evidência atual

`src/tests/test_neuroendocrine_moe.py` cobre causalmente:

- gate neutro e modulação bidirecional;
- surpresa de entropia versus entropia alta constante;
- pressão de sono alcançável e limitada;
- loss melhorando não produzindo cortisol;
- backward propondo sem mudar topologia/identidade de parâmetros;
- viés vertical sem deriva do parâmetro aprendido;
- fração exata de poda;
- janela de sono sem reescrever pesos.

Em 2026-07-13, a suíte completa teve 188 testes aprovados e duas falhas de
contrato histórico do corpus clássico, sem relação com esta implementação.

## Próximo experimento obrigatório

Comparar, com seeds e compute pareados:

1. MoE base sem modulação;
2. apenas gate dopamina/cortisol;
3. apenas surpresa de entropia;
4. combinação completa sem ações estruturais;
5. controlador estrutural futuro com optimizer/checkpoint topology-aware.

Métricas: loss/val perplexity, especialização e balanceamento dos experts,
forgetting, estabilidade de gradientes, tokens/s, VRAM e custo por melhoria.
Sem esse experimento, há mecanismo novo no código, mas não resultado científico.
