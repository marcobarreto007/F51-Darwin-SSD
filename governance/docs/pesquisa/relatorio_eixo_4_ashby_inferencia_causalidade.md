# Eixo 4 — Ashby, Homeostase, Inferência, Causalidade, Memória & Irreversibilidade (1940–1951)

**Agente:** ae36c5c352be5ac89
**Data:** 2026-07-27
**Fontes primárias analisadas:** 12

---

## 1. W. ROSS ASHBY — HOMEOSTASE, VARIEDADE E ADAPTAÇÃO

**Contexto:** Ashby (1903–1972), psiquiatra no Barnwood House Hospital, Gloucester. Isolado do circuito Macy-Wiener-McCulloch. Correspondência com Wiener só começou em 1951. Partiu da psiquiatria biológica e da física, não da engenharia elétrica.

### Adaptiveness and Equilibrium (1940)
- **Veículo:** Journal of Mental Science, 86:478–483
- **Digitalização:** Wiley Online Library (British J. Psychiatry archives)
- Adaptação = equilíbrio dinâmico ativo. Precursor do conceito de "variáveis essenciais".

### The Nervous System as Physical Machine (1947)
- **Veículo:** Circulação restrita. Fragmentos em Conant (1981)
- Sistema nervoso como máquina física determinística. Mente = comportamento de sistema material complexo. Diferença crucial com McCulloch-Pitts: modelo DINÂMICO (equações diferenciais), não lógico (tabelas-verdade).

### The Physical Origin of Adaptation (1947, não publicado)
- **Status:** Manuscrito, British Library Add MS 89153
- Adaptação como consequência necessária de sistemas físicos com certas propriedades organizacionais.

### O Homeostato (1948–1949)
- **Apresentação:** Ratio Club, Londres, 1949
- 4 unidades eletromecânicas interconectadas. Cada unidade: ímã móvel em bobina, potenciômetros, uniselector (step aleatório).

**Ultra-estabilidade (ultrastability):**
- Estabilidade comum: retorna ao equilíbrio após perturbação pequena
- Ultra-estabilidade: se perturbação excede limiar crítico, o sistema MUDA SUA PRÓPRIA ORGANIZAÇÃO INTERNA (potenciômetros) até encontrar nova configuração estável
- Ultra-estabilidade = definição operacional de adaptação

### Lei da Variedade Requerida (1949–1951, pub. 1956)
- V(R) >= V(D) / V(S_aceitavel)
- "Only variety can destroy variety"
- **Variedade:** número de estados distinguíveis do sistema
- **Restrição (constraint):** variedade real < variedade máxima possível. Essência da organização.

### Publicações de Ashby no período (1940–1951)

| Ano | Título | Periódico |
|-----|--------|-----------|
| 1940 | "Adaptiveness and Equilibrium" | J. Mental Science, 86:478–483 |
| 1945 | "Effect of control on stability" | Nature, 155:242–243 |
| 1947 | "The Nervous System as Physical Machine" | Circulação restrita |
| 1947 | "The Physical Origin of Adaptation" | Manuscrito |
| 1949 | "The Homeostat" (Ratio Club) | Ata de reunião |

---

## 2. INFERÊNCIA, CAUSALIDADE E TEMPO

### Hans Reichenbach — The Direction of Time (1947–1953, pub. post. 1956)
- **Veículo:** University of California Press, 1956. Reimpresso Dover, 1991
- **Digitalização:** Archive.org
- Baseado em palestras na UCLA (1947) e manuscritos até sua morte (1953).

**Common Cause Principle:**
Se P(A&B) > P(A)·P(B) e nenhum causa o outro → existe C anterior que é causa comum de ambos.
Ex: barômetro cai (A) e chuva (B) são correlacionados; causa comum C = queda da pressão atmosférica.
Fundamento das redes bayesianas causais (Pearl, 2000).

**Mark Method:**
Se uma pequena variação em A se propaga para B, mas não o inverso → A causa B.
Definição operacional de causalidade independente de intervenção humana.
Precursor do do-calculus de Pearl.

**Screening Off:**
P(A&B | C) = P(A | C) · P(B | C). C "bloqueia" a correlação entre A e B.
Precursor da d-separação em redes bayesianas.

**Ramo de Reichenbach (Branch Hypothesis):**
A seta do tempo termodinâmica emerge de condição de contorno cosmológica — o universo começou em estado de baixa entropia. Não é lei fundamental; é contingência histórica.

### Wiener — Predição como Inferência Causal (1948)
- Feedback = mecanismo causal circular (efeito retroage sobre causa)
- Predição = estimação estatística do futuro dado o passado
- Sistema nervoso = máquina de predição causal

### Shannon — Equivocation e Inferência (1949)
- Equivocation H(X|Y) = incerteza sobre a causa dado o efeito
- Unicity distance = volume mínimo de evidência para inferência única
- Ressoa com Ashby (variedade) e Reichenbach (screen off)

---

## 3. MEMÓRIA E IRREVERSIBILIDADE

### von Neumann — Memória Ativa vs. Estrutural (1948, Hixon)
- **Memória ativa (curto prazo):** Estados elétricos/neurais ativos. Volátil.
- **Memória estrutural (longo prazo):** Mudanças físicas persistentes (sinapses, fitas, cartões). Sobrevive a ciclos de energia.
- No cérebro: sem separação entre CPU e RAM. Sinapses são simultaneamente armazenamento e processamento.

### McCulloch — Memória como Reverberação (1948)
- Circuitos reverberantes = memória de curto prazo
- Evidência anatômica: circuitos fechados de Lorente de Nó (1938)
- Limitação: não explica transição para memória de longo prazo

### Karl Lashley — In Search of the Engram (1950)
- **Veículo:** Symposia Soc. Exp. Biology, 4:454–482
- **Lei da Ação de Massa:** Déficit ∝ quantidade de cortex removido, não localização
- **Equipotencialidade:** Qualquer parte do cortex pode assumir função de outra
- **Conclusão:** Memória é distribuída. Refuta engrama localizado e circuito reverberante fixo.
- Antecipa memórias associativas distribuídas (Hopfield, 1982; PDP, 1986)

### Donald Hebb — The Organization of Behavior (1949)
- **Veículo:** Wiley. Reimpresso Lawrence Erlbaum, 2002
- **Digitalização:** Amplamente disponível

**Regra de Hebb:**
"When an axon of cell A is near enough to excite cell B and repeatedly or persistently takes part in firing it, some growth process or metabolic change takes place in one or both cells such that A's efficiency, as one of the cells firing B, is increased."
→ "Neurons that fire together, wire together" (S. Löwel, 1992)

**Cell Assembly:**
Grupo de neurônios interconectados que se excitam mutuamente e funcionam como sistema fechado. Unidade básica de processamento. Distribuída (explica Lashley). Dinâmica (cresce, encolhe, reorganiza).

**Phase Sequence:**
Assembleias ligadas em sequências temporais. Correspondem a pensamentos complexos, memórias episódicas, planejamento motor.

**Hebb e McCulloch-Pitts:**
Hebb cita McCulloch-Pitts (1943) como fundamento lógico, mas aponta 3 limitações:
1. Sem aprendizagem (pesos fixos)
2. Sem plasticidade (sem mecanismo para experiência mudar conexões)
3. Sem desenvolvimento (não explica ontogênese)

Hebb adiciona ao modelo exatamente o que faltava: mecanismo pelo qual experiência modifica permanentemente a estrutura da rede.

---

## 4. SÍNTESES CRUZADAS

### 4.1 Ashby + Hebb = Aprendizado por Estabilização Estrutural
Ultra-estabilidade (Ashby) + plasticidade hebbiana (Hebb) = mecanismo completo. Cell assembly é um "homeostato distribuído" que busca estabilidade via reorganização sináptica.

### 4.2 Reichenbach + Shannon = Inferência Causal
Common Cause Principle (Reichenbach) + equivocation (Shannon) = inferência causal como redução de incerteza sobre o grafo causal.

### 4.3 Wiener + Reichenbach = Predição como Inferência Causal
Predição ótima (Wiener) + assimetria temporal das marcas (Reichenbach) = predição como inferência causal futuro-dirigida.

### 4.4 von Neumann + Lashley + Hebb = Memória Distribuída Confiável
Redundância para confiabilidade (von Neumann) + distribuição da memória (Lashley) + mecanismo de plasticidade (Hebb). Memória distribuída não é acidente evolutivo; é necessidade de engenharia — sistemas irreversíveis longe do equilíbrio precisam de redundância.

### 4.5 A Seta do Tempo como Denominador Comum

| Domínio | Autor | Manifestação da Irreversibilidade |
|---|---|---|
| Termodinâmica | Reichenbach, Wiener | 2ª lei; ramo cosmológico |
| Informação | Shannon | Equivocation só diminui; canal ruidoso é irreversível |
| Aprendizado | Hebb | Mudanças sinápticas estruturais e permanentes |
| Memória | von Neumann, Lashley, McCulloch | Engrama sobrevive ao tempo; consolidação unidirecional |
| Causalidade | Reichenbach, Wiener | Assimetria temporal como marca da causação |
| Adaptação | Ashby | Ultra-estabilidade muda organização permanentemente |
| Vida/Morte | von Neumann | Informação genética transmitida em uma direção; entropia vence |

**Intuição compartilhada:** A seta do tempo é a condição de possibilidade para qualquer sistema que processe informação, aprenda ou se adapte. Sem irreversibilidade, não há aprendizado, memória, causalidade ou vida.

### O Que Faltava em 1951
1. Mecanismo biofísico da regra de Hebb (LTP/LTD, NMDA — Bliss & Lømo 1973)
2. Integração formal Shannon-Wiener-Ashby (só com Ashby 1956, Introduction to Cybernetics)
3. Simulação computacional (redes neurais, modelos de aprendizado)
