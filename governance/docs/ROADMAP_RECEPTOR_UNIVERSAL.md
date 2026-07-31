# Roadmap — Receptor Universal

**Objetivo declarado:** provar que pesos marcados podem ser copiados de
doadores independentes para um receptor, transformando um modelo fraco em um
forte, com proveniência por parâmetro e sem destilação.

**Critério de sucesso:** um receptor que, depois de receber enxertos, mede
melhor que antes numa capacidade específica, sem degradar o resto, com
certificado de herança em cada peso copiado e deriva medida em cada peso
fabricado.

**Critério de fracasso, declarado antes:** se o enxerto não superar o
controle de ruído de mesma norma, a hipótese cai. Isso é resultado, não
derrota.

---

## Fase 0 — Base a 100% (bloqueante)

Nenhum agente é despachado antes disto. Base torta multiplica erro por
quatro.

| # | Item | Estado | Dono |
|---|---|---|---|
| 0.1 | Suíte de testes verde | **3 falhas** | ver 0.4–0.6 |
| 0.2 | Checker de documentação | ✅ 0 achados | — |
| 0.3 | SBOM CycloneDX reconciliado | ✅ 41 pacotes registrados | feito |
| 0.4 | `python:executable_hash_mismatch` | ❌ aberto | **decisão humana** |
| 0.5 | `torch:installed_file_mismatch` (METADATA, RECORD) | ❌ aberto | **decisão humana** |
| 0.6 | `torch` órfão em `%APPDATA%` (31 MB, sem `__init__.py`) | ❌ aberto | **decisão humana** |

**0.4 e 0.5** não devem ser re-baselineados sem explicação. Descasamento de
hash de executável é o sinal mais forte que o controle de cadeia de
suprimentos consegue emitir; zerá-lo por conveniência é desligar o alarme.
Investigar primeiro: o interpretador foi atualizado? O venv foi reconstruído?
Se houver causa legítima, re-baselinear registrando a causa.

**0.6** é poluição de máquina fora do repositório. Remover resolve o terceiro
teste. Está fora da árvore, então é decisão do dono da máquina.

---

## Fase 1 — Primeira carga real pelos portões

O sistema tem instrumentos completos e nunca rodou um experimento dentro
deles. Esta fase existe para mudar isso.

### 1.1 Remontagem certificada (baseline de confiança)

Remontar o SmolLM2-1.7B a partir dele mesmo, tensor por tensor, com
`verify_inheritance` em cada um.

- **Prova:** KL exatamente 0 contra o original, e certificado válido em 100%
  dos tensores.
- **Por que primeiro:** calibra o instrumento contra uma resposta conhecida.
  Se a remontagem idêntica não der KL 0, o problema é o aparato, não a
  hipótese. Nunca fizemos isso.
- **Custo:** horas.

### 1.2 Enxerto entre linhagens independentes

SmolLM2-1.7B como receptor e base de coordenadas. FFN do TinyLlama-1.1B como
expert adicional. Ambos d_model 2048; o expert nunca vê token, então
vocabulário divergente não importa.

- **Braços obrigatórios:** doador real / ruído de mesma norma / sem enxerto.
- **Prova:** ganho no domínio do doador acima do braço de ruído, sem perda
  fora dele.
- **Risco conhecido:** base residual descorrelacionada (0.175 contra 0.178 de
  acaso). A literatura de merge documenta colapso quando as distribuições de
  hidden-state divergem.
- **Custo:** dias.

### 1.3 Representações relativas como ponte

Se 1.2 falhar por incompatibilidade de base, reexpressar ativações por
similaridade a âncoras fixas e remedir a correlação.

- **Prova:** correlação de base sobe acima do acaso após reexpressão.
- **Por que importa:** é a diferença entre "receptor universal" e "receptor
  que só aceita parentes".
- **Custo:** dias.

---

## Fase 2 — Transferência de conhecimento, não de peso

### 2.1 Fatos como unidade de transplante

Extrair N associações que o doador forte acerta e o receptor fraco erra, e
instalar via ROME com `v*` otimizado. Cada instalação emite registro de
proveniência e rollback verificado.

- **Prova:** acurácia do receptor sobe nos N fatos, conhecimento geral
  preservado, rollback restaura o hash original.
- **Por que promissor:** é a única coisa do repositório que comprovadamente
  funciona (6/6 identidade, conhecimento intacto).
- **Não é destilação:** sem gradiente sobre saídas, sem alvo suave. É escrita
  rank-1 por fato, reversível e auditável.
- **Custo:** dias.

### 2.2 Curva de saturação

Instalar fatos incrementalmente e medir onde para de melhorar e onde começa a
degradar.

- **Prova:** a curva é resultado publicável sozinha.

---

## Fase 3 — Consolidação

### 3.1 Ligar os órgãos

Hoje os três órgãos têm 0 chaves no checkpoint, portões em zero e
`max_abs_logit_error: 0.0`. O controle negativo é exemplar e nada passa por
ele. Conectar ao forward e medir com braços casados.

### 3.2 Remedir na arquitetura certa

Catálogo de 196K, ablação de domínio e direção-contra-neurônio foram medidos
no `1.7B_SMOL_DENSE_V1`, que é a única configuração em que o DarwinX vira um
transformer comum. Refazer no 100M MoE+SSD, que é a arquitetura real.

### 3.3 Paper e depósito

Depositar antes de publicar. Com o resultado da Fase 1 ou 2, o paper deixa de
ser sobre instrumentos e passa a ter carga que os instrumentos sustentam.

---

## Regras para agentes despachados

1. Nenhum agente commita. O commit é revisado.
2. Cada agente é dono exclusivo de arquivos declarados. Sem sobreposição.
3. Proibido relaxar asserção, marcar skip/xfail ou apagar teste. Se a única
   forma de passar for enfraquecer a verificação, **parar e reportar**.
4. Nenhum agente inicia treino, canary ou servidor.
5. Todo resultado cita arquivo, comando, teste ou hash.
6. Controle negativo obrigatório em qualquer medição de efeito.

---

## O que este roadmap NÃO promete

Que o receptor universal funciona. Quatro buscas dirigidas dizem que ninguém
tentou nesses termos, e a literatura de merge documenta colapso quando as
bases divergem. As Fases 1 e 2 existem para descobrir, não para confirmar.

Um resultado negativo em 1.2, com controle de ruído e braços casados, é
entregável e publicável. Um resultado positivo sem controle não é.
