# Decisao Arquitetural: live_generate ↔ InferenceLearner
Data: 2026-07-12
Status: DECIDIDO

## Contexto

Davi tem dois caminhos de geracao:
1. DarwinXModel.live_generate() — gera tokens com heartbeat/memoria/thoughts
2. InferenceLearner.interact() — pesquisa web, aprende, responde

Eles nao estao integrados. Precisamos decidir como unifica-los.

## Decisao

**InferenceLearner e o caminho principal. live_generate e o fallback.**

Fluxo unificado:
```
Usuario pergunta
    ↓
InferenceLearner.interact(prompt)
    ├─ think(): Quiet-STaR-like, decide se precisa pesquisar
    ├─ research(): DuckDuckGo se necessario
    ├─ retrieve(): EpisodicMemory busca conhecimento previo
    ├─ learn(): OnlineAdapter + Forward-Forward se novidade
    ├─ respond(): live_generate com contexto enriquecido
    └─ persist(): salva na EpisodicMemory + replay buffer
```

live_generate() continua existindo como fallback:
- Quando nao ha internet (DuckDuckGo offline)
- Quando EpisodicMemory esta vazia
- Para geracao simples sem pesquisa

## Implementacao

1. InferenceLearner.interact() chama model.live_generate() internamente
   com prompt enriquecido (contexto de pesquisa + memorias)
2. live_generate() ganha parametro opcional `context: str = ""`
3. Se context vazio → caminho antigo (fallback)
4. Se context preenchido → usa no prompt

## Nao fazer

- NAO injetar web direto nos pesos do modelo
- NAO substituir live_generate (ele e o fallback seguro)
- NAO adicionar complexidade desnecessaria
