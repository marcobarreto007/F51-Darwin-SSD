# Eixo 3 — McCulloch-Pitts, Turing & von Neumann (1940–1952)

**Agente:** abca97d35a2a34bb7
**Data:** 2026-07-27
**Fontes primárias analisadas:** 10

---

## 1. McCULLOCH E PITTS — NEURÔNIOS COMO LÓGICA

### A Logical Calculus of the Ideas Immanent in Nervous Activity (1943)
- **Veículo:** Bulletin of Mathematical Biophysics, 5:115–133
- **Digitalização:** Springer archives

**Neurônio formal:**
- All-or-none (binário, threshold θ)
- Sinapses excitatórias (soma) + inibitórias (veto absoluto)
- Tempo discreto (1 tick = 1 atraso sináptico)
- AND (θ=2), OR (θ=1), NOT (via inibição)

**Teorema central:** Toda função Turing-computável pode ser realizada por uma rede de neurônios formais com ciclos.
- Redes sem ciclos = funções booleanas finitas (sem memória)
- Redes com ciclos = Turing-completas (ciclo = estado persistente = memória)

**Psychons:** Entidades psíquicas mínimas que emergem de atividade circular. Conexão com fenomenologia (Husserl: retenção).

**Jerome Lettvin:** O "quarto homem". Apresentou Pitts a McCulloch. Tradutor entre neurofisiologia (Lorente de Nó) e lógica formal.

**Limitações:** Pesos fixos (sem aprendizado), timing ignorado, inibição binária irrealista.

### How We Know Universals (1947)
- **Veículo:** Bulletin of Mathematical Biophysics, 9(3):127–147
- **Digitalização:** Springer archives

**Arquitetura de scanning neural:**
- Camadas aplicam transformações geométricas (translação, rotação, dilatação) a cada tick do ciclo alpha
- Camada superior integra atividade temporalmente → invariância
- **Precursor direto de CNNs** (convolução + pooling)

### Why the Mind Is in the Head (1948)
- **Veículo:** Dialectica, 2(3-4):375–385
- **Tese:** A mente está na cabeça porque só lá existem circuitos com topologia de ciclos fechados capazes de sustentar reverberação. Se o fígado tivesse essa topologia, a mente estaria no fígado.

---

## 2. ALAN TURING

### On Computable Numbers (1936–37)
- **Veículo:** Proc. LMS, 42:230–265
- Máquina de Turing (a-machine), máquina universal (u-machine), tese de Church-Turing. Fundação citada por McCulloch-Pitts (1943) e von Neumann (1945).

### Intelligent Machinery (1948, engavetado pelo NPL)
- **Veículo:** NPL Report. Pub. post. em Machine Intelligence 5 (1969)
- **Digitalização:** Turing Digital Archive

**Unorganized Machines:**
- **Tipo A:** Rede aleatória de NANDs. Determinística.
- **Tipo B:** Conexões modificáveis (ligar/desligar). "Treinamento" por poda.
- **Tipo P (Pleasure-Pain):** Sinal externo de prazer/dor modula modificação. Precursor de reinforcement learning.

**Aprendizado:** "Instead of trying to produce a programme to simulate the adult mind, why not rather try to produce one which simulates the child's?"

**Censura:** Sir Charles Darwin (diretor do NPL) rejeitou como "a schoolboy's essay". Permaneceu inédito por 21 anos.

### Computing Machinery and Intelligence (1950)
- **Veículo:** Mind, LIX(236):433–460. DOI: 10.1093/mind/LIX.236.433

**Turing Test (Jogo da Imitação):** 9 objeções refutadas — teológica, "cabeças na areia", matemática (Gödel), consciência, incapacidades, Lady Lovelace, continuidade do sistema nervoso, informalidade do comportamento, ESP.

**Refutação de Lady Lovelace:** Máquinas que aprendem podem nos surpreender — fazem coisas não programadas explicitamente.

### The Chemical Basis of Morphogenesis (1952)
- **Veículo:** Phil. Trans. R. Soc. B, 237(641):37–72. DOI: 10.1098/rstb.1952.0012
- **Instabilidade de Turing:** Reaction-diffusion. ∂u/∂t = f(u,v) + D_u ∇²u. Ativador (difusão lenta) + inibidor (difusão rápida) → padrões espaciais. Validado experimentalmente décadas depois.

---

## 3. JOHN VON NEUMANN

### First Draft of a Report on the EDVAC (1945)
- **Veículo:** Moore School, U. Penn. 101 pp.
- **Digitalização:** U. Penn digital archives; Smithsonian

**Arquitetura stored-program:** 5 unidades — CA (ALU), CC (controle), M (memória), I (input), O (output). Instruções e dados na mesma memória. Cita McCulloch-Pitts (1943).

### The General and Logical Theory of Automata (1948, Hixon Symposium)
- **Veículo:** Em Jeffress (ed.), Cerebral Mechanisms in Behavior, Wiley, 1951

**Confiabilidade via redundância:** Cérebro: 10^10 neurônios operando por décadas. ENIAC: ~18.000 válvulas, uma falha = parada total. Diferença é arquitetônica.

**Complexity threshold:** Limiar abaixo do qual autômatos degeneram; acima do qual podem evoluir. ~200.000 elementos no modelo celular.

**Digital vs. analógico:** Micro = analógico/probabilístico. Macro = digital (disparo). Cérebro explora ambos.

### Theory of Self-Reproducing Automata (1948–49, pub. post. 1966)
- **Veículo:** Editado por A.W. Burks, Univ. of Illinois Press

**Construtor universal + fita de instruções:**
- Separação entre "o quê" (informação, fita) e "como" (hardware, construtor)
- Auto-reprodução: duplicar fita → construir cópia → inserir fita duplicada
- **Antecipou a lógica do DNA ANTES de Watson-Crick (1953).** Sydney Brenner escreveu a von Neumann apontando a correspondência.

### Probabilistic Logics... (1952, pub. 1956)
- **Veículo:** Em Shannon & McCarthy (eds.), Automata Studies, Princeton, 1956

**Multiplexação:** Cada sinal = feixe de M cópias. Órgão restaurador (majority gate). Se ε < ε_c ≈ 0.007, confiabilidade arbitrária é possível.

---

## 4. SÍNTESE: A CONVERSA A TRÊS VOZES

| Tema | McCulloch-Pitts | Turing | von Neumann |
|---|---|---|---|
| Unidade mínima | Neurônio formal | NAND gate (Tipo A) | Neurônio simplificado |
| Poder computacional | Turing-completo (com ciclos) | Máquina universal | Construtor universal |
| Memória | Ciclos reverberantes | Fita infinita | RAM (stored-program) |
| Aprendizado | Não abordado | Modificação + pleasure-pain | Multiplexação + restauração |
| Auto-organização | Ciclos fechados | Reaction-diffusion | Autômatos auto-reprodutores |

**Intuição comum:** Ordem (pensamento, padrão, complexidade) emerge de regras locais simples operando em substrato com armazenamento de estado e realimentação.

**Conexão com Darwin-X:** O barramento causal é a implementação moderna da separação von Neumann entre controle e memória. Os órgãos (GABA, TTM, Curiosity, etc.) são implementações das "unorganized machines Tipo P" de Turing com sinal de reforço (dopamina).
