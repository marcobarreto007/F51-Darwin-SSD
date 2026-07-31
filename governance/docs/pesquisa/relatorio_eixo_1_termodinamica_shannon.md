# Eixo 1 — Termodinâmica, Entropia, Mecânica Estatística & Teoria da Informação (1940–1951)

**Agente:** a4223149ed21d97dc
**Data:** 2026-07-27
**Fontes primárias analisadas:** 16

---

## 1. TERMODINÂMICA, ENTROPIA E MECÂNICA ESTATÍSTICA

### Erwin Schrödinger — What is Life? (1944)
- **Veículo:** Cambridge University Press (palestras no Trinity College, Dublin, fev/1943)
- **Digitalização:** Archive.org (edição 1948 com epílogo)
- **Argumento central:** O gene como "cristal aperiódico" contendo um "code-script". Negentropia: "o organismo se alimenta de entropia negativa". Demônio de Maxwell como dispositivo que precisaria de informação molecular — antecipando a conexão entropia-informação. Catalisou a migração de físicos (Crick, Watson, Wilkins, Franklin) para biologia molecular.

### Lars Onsager — Reciprocal Relations (1931, maturação nos 1940s)
- **Veículo:** Physical Review, 37:405–426; 38:2265–2279
- **Argumento central:** Relações de reciprocidade L_{ij} = L_{ji} para sistemas próximos ao equilíbrio. Emergem da reversibilidade microscópica. Confirmadas experimentalmente nos anos 1940. Base para Prigogine.

### Ilya Prigogine — Étude thermodynamique des phénomènes irréversibles (1947)
- **Veículo:** Tese de doutoramento, Éditions Desoer, Liège
- **Digitalização:** Rara. Edição inglesa de 1955 no Archive.org
- **Argumento central:** Princípio da produção mínima de entropia para estados estacionários. dS = d_eS + d_iS (entropia trocada + entropia produzida). Primeira demonstração de princípio variacional para sistemas irreversíveis lineares.

### Leo Szilard — On the Decrease of Entropy... (1929, redescoberto nos 1940s)
- **Veículo:** Zeitschrift für Physik, 53:840–856
- **Tradução:** Behavioral Science, 9(4):301–310, 1964
- **Argumento central:** O demônio de Maxwell precisa MEDIR a posição da molécula. 1 bit de medição custa k ln 2 entropia. Informação e entropia são conversíveis. Redescoberto pós-guerra por Brillouin, Gabor e Shannon.

---

## 2. TEORIA DA INFORMAÇÃO — CLAUDE SHANNON

### A Mathematical Theory of Communication (1948)
- **Veículo:** BSTJ, 27(3):379–423 (Parte I, julho); 27(4):623–656 (Parte II, outubro)
- **Parte I:** Arquitetura fonte→transmissor→canal→receptor→destino. H = -Σ p_i log p_i. Teorema da codificação de fonte (limite H). Capacidade de canal C. Teorema da codificação de canal (R < C → comunicação confiável).
- **Parte II:** Canal AWGN: C = W log_2(1 + S/N). Rate distortion. Três níveis de comunicação (técnico, semântico, efetividade).

### Communication Theory of Secrecy Systems (1949)
- **Veículo:** BSTJ, 28(4):656–715
- **Argumento central:** Equivocation H(M|C) e H(K|C). Distância de unicidade. One-time pad como sistema de sigilo perfeito: H(M|C) = H(M).

### Warren Weaver — Ensaio introdutório (1949)
- **Veículo:** The Mathematical Theory of Communication, Univ. of Illinois Press, pp. 1–28
- **Argumento central:** Três níveis (A: técnico, B: semântico, C: efetividade). Popularizou e expandiu Shannon para audiências não-matemáticas.

### A anedota von Neumann / "entropia"
- **Evidência:** Oral de 2ª mão. Fonte mais antiga: Myron Tribus, Thermostatics and Thermodynamics (1961), pp. 177–178. Corroborada por Robert Fano (1961). Shannon confirmou tardiamente em entrevista a Robert Price (1984).
- **Status:** "Folclore científico bem-fundamentado", não fato documentado.

---

## 3. PRECURSORES E CONEXÕES

### Dennis Gabor — Theory of Communication (1946)
- **Veículo:** J. IEE, 93(III):429–441
- **Argumento central:** Logon (célula tempo-frequência). Δt · Δf ≥ 1/(4π). Comunicação como processo físico fundamental.

### Harry Nyquist — Certain Topics in Telegraph Transmission Theory (1928)
- V = 2W log_2 m — precursor do teorema da amostragem

### Ralph Hartley — Transmission of Information (1928)
- H = n log m — primeira definição quantitativa de informação

### Léon Brillouin — Maxwell's Demon Cannot Operate (1950–51)
- **Veículo:** J. Appl. Phys., 22:334–343
- **Argumento central:** Observação física tem custo entrópico irredutível. Cada bit = kT ln 2 joules. Informação livre vs. informação ligada (bound information).

### Genealogia completa
```
Boltzmann (1872-1877): H-theorem, S = k log W
    ↓
Gibbs (1902): Σ p_i log p_i para ensemble canônico
    ↓
Szilard (1929): 1 bit = k log 2 entropia
    ↓
    ├── Wiener (1948): Informação é negentropia (via Gibbs)
    ├── Shannon (1948): H = -Σ p_i log p_i por axiomas
    └── Brillouin (1950-51): Síntese termodinâmica
```
