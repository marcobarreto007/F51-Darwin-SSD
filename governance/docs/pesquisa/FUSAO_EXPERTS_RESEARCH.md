# Fusão de experts — protocolo de pesquisa

Atualizado em 2026-07-13.

## Estado atual

A fusão destrutiva está desativada no Darwin-X. O código pode armazenar entradas
e calcular assinaturas experimentais, mas não há evidência suficiente para
alterar pesos ou remover experts durante treino.

A implementação anterior usava BDNF escalar como proxy de redundância. Isso é
inválido: dois experts ativos no mesmo período não necessariamente realizam a
mesma função. Uma segunda versão comparava a média das saídas, mas citava um
paper não verificável e fundia pesos sem teste de equivalência ou rollback.

## Pontos fortes da hipótese funcional

- Compara comportamento, não somente distância entre pesos.
- Pode usar um conjunto held-out comum a todos os experts.
- Permite medir erro antes/depois e rejeitar uma fusão ruim.

## Pontos fracos que bloqueiam ativação

- A média de saída pode esconder diferenças importantes por token.
- Similaridade cosseno alta não implica substituibilidade na loss.
- Experts com hidden dimensions diferentes não podem ter pesos promediados.
- Média de pesos ignora simetrias/permutação dos neurônios.
- Remoção muda router, buffers, optimizer e formato do checkpoint.
- Inputs recentes do treino não são um conjunto independente de validação.

## Protocolo causal mínimo

Para um par candidato `(A, B)`:

1. Congelar um conjunto held-out estratificado por domínio.
2. Medir similaridade por token das saídas normalizadas de A e B.
3. Medir a contribuição marginal removendo A e B separadamente, sem update.
4. Construir a fusão em uma cópia isolada do modelo.
5. Comparar loss, router balance, especialização e forgetting antes/depois.
6. Aceitar somente dentro de uma margem pré-registrada.
7. Reconstruir optimizer e checkpoint de forma topology-aware.
8. Manter rollback integral e validar retomada do checkpoint.

## Experimentos

| ID | Intervenção | Controle | Critério primário |
|---|---|---|---|
| F0 | nenhuma fusão | MoE atual | baseline |
| F1 | selecionar candidatos por saída | mesmos forwards sem fundir | precisão do detector |
| F2 | ablação de um expert | expert intacto | delta de validation loss |
| F3 | fusão em sandbox | checkpoint original | não inferioridade + retomada |

## Ineditismo

“Comparar ativações” e “fundir modelos” não são novos isoladamente. O possível
espaço novo é a fusão online de experts integrada ao ciclo neuroendócrino local,
mas essa frase é somente uma direção de pesquisa. Não há busca exaustiva de
literatura/patentes nem resultado causal que autorize reivindicação forte.

## Referências verificáveis relacionadas

- TIES-Merging: https://arxiv.org/abs/2306.01708
- Git Re-Basin: https://arxiv.org/abs/2209.04836
- Fisher-weighted model merging: https://arxiv.org/abs/2111.09832
- DynMoE: https://arxiv.org/abs/2405.14297

Próxima decisão: implementar apenas o detector F1 e medir falsos positivos. A
fusão F3 permanece fora do runtime até F1/F2 passarem.
