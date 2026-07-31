# Eixo 2 — Cibernética, Feedback, Previsão & Comportamento Teleológico (1940–1951)

**Agente:** a3bdbcfbfc32d85bf
**Data:** 2026-07-27
**Fontes primárias analisadas:** 12 + Conferências Macy

---

## 1. NORBERT WIENER — CIBERNÉTICA, FEEDBACK E PREVISÃO

### Cybernetics (1948, MIT Press / Wiley; 2ª ed. 1961)
**Digitalização:** Archive.org

- **Cap. I — Newtonian and Bergsonian Time:** Distinção entre tempo reversível (mecânica clássica) e irreversível (termodinâmica, biologia). A cibernética é a ciência de sistemas cujo comportamento é intrinsecamente dirigido no tempo. Crítica ao vitalismo de Bergson: irreversibilidade não requer élan vital, requer matemática de feedback.
- **Cap. II — Groups and Statistical Mechanics:** Conexão Gibbs-Wiener. Predição de séries temporais ≡ mecânica estatística (processo parcialmente observável, ergodicidade, médias de ensemble vs. temporais).
- **Cap. III — Time Series, Information, and Communication:** Informação como entropia negativa. Predição linear (filtro de Wiener). "Information is information, not matter or energy."
- **Cap. IV — Feedback and Oscillation:** Feedback negativo (estabilizador) vs. positivo (amplificador). Tremor cerebelar = oscilação por feedback negativo insuficiente. Condição de estabilidade: |GH| < 1.
- **Cap. V — Computing Machines and the Nervous System:** Analogia neurônio↔válvula, sinapse↔porta lógica, memória↔reverberação. Cérebro opera com lógica estatística, não determinística.
- **Cap. VIII — Information, Language, and Society:** "The automatic machine is the precise economic equivalent of slave labor." Sociedade como sistema de comunicação. Entropia social = ruído, desinformação.

### Extrapolation, Interpolation, and Smoothing... ("Yellow Peril", 1949)
- **Veículo:** MIT Press / Technology Press
- **Argumento central:** Filtro de Wiener-Hopf. Equação integral. Fatoração espectral (causal vs. anticausal). Predição pura vs. filtragem vs. smoothing. Conexão com controle de fogo AA.

### Relatório classificado NDRC (1942)
- OSRD Report 370, MIT. Circulação restrita. Resolvia predição de trajetórias de aeronaves para controle de fogo antiaéreo. Shannon leu o relatório (comunicação oral).

### Time, Communication, and the Nervous System (1948)
- **Veículo:** Annals of the NYAS, 50:197–220
- Codificação temporal no sistema nervoso. EEG como sinal de comunicação. Auto-correlação como ferramenta de análise neural.

---

## 2. COMPORTAMENTO TELEOLÓGICO

### Rosenblueth, Wiener & Bigelow — Behavior, Purpose and Teleology (1943)
- **Veículo:** Philosophy of Science, 10(1):18–24. 7 páginas.
- **Digitalização:** JSTOR / Univ. Chicago Press

**Classificação do comportamento:**
```
Ativo (fonte de energia)
├── Sem propósito (random, tropismo)
└── Com propósito (dirigido a meta)
    ├── Sem feedback (teleologia intrínseca) — ex: míssil balístico
    └── Com feedback (teleologia extrínseca)
        ├── Preditivo (extrapolativo) — ex: cão antecipando presa
        └── Não-preditivo — ex: termostato
```

**Tese central:** Propósito = feedback negativo dirigido a meta. Nenhum ingrediente misterioso. O artigo elimina o vitalismo mantendo a noção de propósito. Servomecanismos preditivos como nova classe de máquinas.

**Impacto:** Citado por Shannon, McCulloch, von Neumann, Bateson, Skinner. Funda a cibernética como campo.

### Rosenblueth & Wiener — The Role of Models in Science (1945)
- **Veículo:** Philosophy of Science, 12(4):316–321
- Modelos como analogias formais (isomorfismo, não identidade material). Cérebro ↔ máquina como modelos recíprocos. Fundamento filosófico do funcionalismo.

---

## 3. AS CONFERÊNCIAS MACY (1946–1951)

**Chairman:** Warren S. McCulloch
**Secretário editorial:** Heinz von Foerster
**Sede:** Beekman Hotel, NYC

| # | Data | Tema/Foco |
|---|---|---|
| 1ª | Mar/1946 | Feedback e sistemas circulares; McCulloch-Pitts (1943); von Neumann EDVAC |
| 2ª | Out/1947 | Comunicação e linguagem; Shannon apresenta teoria antes da publicação |
| 3ª | Out/1948 | Cybernetics (livro recém-publicado); tensão Wiener-von Neumann |
| 4ª | Mar/1949 | Aprendizagem e memória; Ashby apresenta homeostato |
| 5ª | Mar/1950 | Linguagem e cognição; Bateson (double bind); Licklider |
| 6ª | 1951 | Tensões no ápice; Wiener abandona; fragmentação do grupo |

**Participantes do núcleo:** McCulloch, Wiener, von Neumann, Rosenblueth, Bigelow, Shannon, Pitts, Bateson, Mead, Lewin, Frank, Klüver, Gerard, Lorente de Nó, von Foerster, Ashby, MacKay, Licklider.

**Tensões:** Wiener vs. von Neumann (digital vs. analógico; militar vs. pacifista). Wiener vs. Bateson (extrapolação social).

**Digitalização:** Transactions editadas por von Foerster (5+ vols). Reedição: Pias (2016, diaphanes).

---

## 4. PRECURSORES DE ENGENHARIA

| Autor | Ano | Contribuição |
|---|---|---|
| Harold Black | 1927 | Amplificador de feedback negativo (patente US 2,102,671) |
| Harry Nyquist | 1932 | Critério de estabilidade (diagrama de Nyquist) |
| Hendrik Bode | 1945 | Diagrama de Bode, relação ganho-fase, sensibilidade integral ("waterbed effect") |

---

## SÍNTESE DO ARCO 1927–1951

1. **1927–1940:** Black, Nyquist, Bode — feedback como disciplina de engenharia
2. **1942–1943:** Problema militar (controle AA) força encontro engenharia + fisiologia + matemática → artigo de 1943
3. **1946–1948:** Conferências Macy + Cybernetics (1948) = campo nomeado e articulado
4. **1949–1951:** Formalização matemática (Yellow Peril) + fragmentação social (Macy se dissolve)
