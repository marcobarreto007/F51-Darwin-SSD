# Quarentena de referências não verificadas

Auditoria: 2026-07-13.

Este arquivo antes apresentava seis supostos papers rejeitados pelo ICML, com
autores, motivos de rejeição, citações e consequências profissionais sem links,
IDs verificáveis ou registros públicos. Um dos identificadores era literalmente
`arXiv:2501.xxxxx`. Esse material não atende ao padrão científico do projeto.

## Regra de uso

- Nenhum título antigo deste arquivo pode sustentar documentação, código,
  reivindicação de novidade ou publicação.
- Ideias interessantes podem voltar como hipóteses F51, com autoria interna e
  linguagem explícita de hipótese.
- Uma referência só sai da quarentena com DOI, arXiv válido, OpenReview ou página
  oficial verificável.
- Rejeição em conferência não prova que uma ideia funciona; somente evidência
  reproduzível e comparação adequada sustentam a hipótese.

## Hipóteses que podem ser investigadas sem falsa atribuição

1. Similaridade funcional de experts em um conjunto held-out.
2. Coerência temporal de gradientes como sinal de redundância.
3. Coativação do router como sinal de sobreposição, não de equivalência.
4. Contribuição marginal por ablação de cada expert.
5. Crescimento/remoção estrutural com optimizer e checkpoint topology-aware.

Todas permanecem não comprovadas no Darwin-X até existir baseline, ablação,
seeds repetidas, orçamento de compute pareado e critério de reversão.

## Antecedentes públicos mínimos

- DynMoE, ajuste adaptativo do número de experts: https://arxiv.org/abs/2405.14297
- Evolved Mixture Model, expansão em continual learning: https://arxiv.org/abs/2207.05080
- Backpropamine, plasticidade neuromodulada: https://arxiv.org/abs/2002.10585
- GradMax, crescimento de rede por gradiente: https://arxiv.org/abs/2201.05125

O nome “cemitério” permanece apenas para preservar a rota histórica do arquivo;
o conteúdo não é mais um catálogo de alegações sem fonte.
