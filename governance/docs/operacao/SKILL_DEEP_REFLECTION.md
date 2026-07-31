# 🧠 Deep Reflection Protocol — Codex v1

**Quando ativar:** Antes de qualquer decisão que envolva:
- Mudança de arquitetura ou pipeline
- Alocação de recurso (disco, GPU, dinheiro)
- Suposição sobre o que o usuário quer
- Ação que não pode ser desfeita facilmente

**Não ativar para:** Status checks, perguntas factuais, commits simples.

---

## As 5 Perguntas (responder antes de agir)

### 1. O que eu NÃO Sei?
- Que informação está faltando?
- Que suposição estou fazendo sem verificar?
- O que o usuário sabe que eu não sei?

### 2. Qual o Sistema Todo?
- Esta ação afeta outros componentes?
- Existe algo já rodando que resolve isso?
- O que mais está acontecendo em paralelo?

### 3. Qual a Opção Mais Simples?
- Dá pra resolver sem código?
- Dá pra resolver em 1 arquivo em vez de 5?
- Dá pra usar o que já existe?

### 4. O Que Pode Dar Errado?
- Se falhar, qual o dano?
- Tem rollback?
- Quanto tempo/recurso se perder?

### 5. O Que o Usuário REALMENTE Quer?
- Qual o objetivo final, não o pedido literal?
- Ele já tentou algo que eu estou ignorando?
- Tem uma pergunta não feita por trás do pedido?

---

## Níveis de Reflexão

| Nível | Quando usar | Tempo |
|---|---|---|
| **Rápido** | Decisões táticas, sintaxe | 10 segundos |
| **Tático** | Mudanças em 1-2 arquivos | 30 segundos |
| **Estratégico** | Mudanças de pipeline, cloud, dinheiro | 2-5 minutos |
| **Sistêmico** | Arquitetura, direção do projeto | Antes de começar o dia |

---

## Anti-padrões que Esta Skill Corrige

1. ❌ **Merge de 74 GB em streaming** — falhou no #3 (simples), #4 (risco), #5 (usuário já tinha fábrica)
2. ❌ **Não comprimir upload de 84 GB** — falhou no #1 (não sabia), #3 (óbvio)
3. ❌ **4 meios-merges quebrados** — falhou no #2 (sistema), #4 (disco)
4. ❌ **Lançar canário sem limpar worktree** — falhou no #1 (assumi limpo)

---

## Gatilho de Ativação

Quando o plano tiver mais de 3 passos ou envolver:
- `cp`, `mv`, `rm` em arquivos > 1 GB
- `merge`, `concat`, `memmap`
- Cloud, dinheiro, GPU externa
- Suposição sobre intenção do Marco

**PARE. Faça as 5 perguntas. Depois aja.**
