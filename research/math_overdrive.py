#!/usr/bin/env python
"""
F51 Math Overdrive — 3 Geradores em Paralelo
=============================================
A) Math Monster: definições, teoremas, identidades, axiomas
B) Math v3: problemas com passo a passo (step-by-step)
C) Proof Database: teoremas clássicos com demonstrações completas

Meta: 5+ GB de matemática pura sintética.
"""

import sys, random, time, json, argparse
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]

# ═══════════════════════════════════════════════════
# A) MATH MONSTER — Definições, Teoremas, Identidades
# ═══════════════════════════════════════════════════

THEOREMS = [
    # Cálculo
    ("Teorema Fundamental do Cálculo", "Se f é contínua em [a,b] e F é uma antiderivada de f, então ∫_a^b f(x)dx = F(b) - F(a). Esta conexão entre derivação e integração é a espinha dorsal da análise matemática."),
    ("Teorema do Valor Médio", "Se f é contínua em [a,b] e diferenciável em (a,b), existe c ∈ (a,b) tal que f'(c) = (f(b)-f(a))/(b-a). Geometricamente: há um ponto onde a tangente é paralela à secante."),
    ("Regra de L'Hôpital", "Se lim f(x) = lim g(x) = 0 ou ±∞, então lim f(x)/g(x) = lim f'(x)/g'(x), desde que o limite da direita exista. Essencial para resolver formas indeterminadas."),
    ("Teorema de Taylor", "f(x) = f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ... + fⁿ(a)(x-a)ⁿ/n! + R_n. O resto R_n quantifica o erro da aproximação polinomial."),
    ("Teorema da Convergência Dominada", "Se f_n → f pontualmente e |f_n| ≤ g com g integrável, então ∫f_n → ∫f. Permite trocar limite e integral sob condições controladas."),
    
    # Álgebra Linear
    ("Teorema Espectral", "Toda matriz simétrica real é diagonalizável por uma matriz ortogonal. Seus autovalores são reais e os autovetores formam uma base ortonormal."),
    ("Decomposição em Valores Singulares (SVD)", "Toda matriz A (m×n) pode ser fatorada como A = UΣV^T, onde U e V são ortogonais e Σ é diagonal com valores singulares não-negativos."),
    ("Teorema de Cayley-Hamilton", "Toda matriz quadrada satisfaz sua própria equação característica: p(A) = 0, onde p(λ) = det(A - λI)."),
    ("Desigualdade de Cauchy-Schwarz", "|<u,v>| ≤ ||u||·||v||. A igualdade ocorre se e somente se u e v são linearmente dependentes."),
    
    # Teoria dos Números
    ("Teorema Fundamental da Aritmética", "Todo inteiro n > 1 pode ser fatorado unicamente como produto de primos: n = p₁^e₁ · p₂^e₂ · ... · pₖ^eₖ."),
    ("Pequeno Teorema de Fermat", "Se p é primo e a não é múltiplo de p, então a^(p-1) ≡ 1 (mod p). Base da criptografia RSA."),
    ("Teorema de Euler", "Se gcd(a,n)=1, então a^φ(n) ≡ 1 (mod n), onde φ é a função totiente de Euler. Generaliza o Pequeno Teorema de Fermat."),
    ("Teorema dos Números Primos", "π(x) ~ x/ln(x). A densidade dos primos decai logaritmicamente. Demonstrado por Hadamard e de la Vallée-Poussin em 1896."),
    ("Teorema de Wilson", "p é primo se e somente se (p-1)! ≡ -1 (mod p). Uma caracterização elegante mas computacionalmente ineficiente de primalidade."),
    ("Teorema Chinês do Resto", "Para módulos coprimos m₁,...,mₖ, o sistema x ≡ aᵢ (mod mᵢ) tem solução única módulo M = ∏mᵢ."),
    
    # Álgebra Abstrata
    ("Teorema de Lagrange", "A ordem de um subgrupo H divide a ordem do grupo G: |H| ∣ |G|. Consequência: a ordem de qualquer elemento divide a ordem do grupo."),
    ("Teorema de Isomorfismo", "Se φ: G → H é um homomorfismo, então G/ker(φ) ≅ im(φ). Todo homomorfismo fatora através do quociente pelo núcleo."),
    ("Teorema de Sylow", "Se |G| = pⁿ·m com p ∤ m, então G possui subgrupos de ordem pᵏ para cada 0≤k≤n. Os p-subgrupos de Sylow são conjugados entre si."),
    
    # Análise
    ("Teorema de Bolzano-Weierstrass", "Toda sequência limitada em Rⁿ possui uma subsequência convergente. Fundamental para provar existência de extremos em compactos."),
    ("Teorema do Valor Intermediário", "Se f é contínua em [a,b], então f assume todos os valores entre f(a) e f(b). Garante existência de raízes para funções contínuas."),
    ("Teorema de Heine-Borel", "Em Rⁿ, um conjunto é compacto se e somente se é fechado e limitado. Caracterização fundamental da topologia euclidiana."),
    
    # Combinatória
    ("Princípio da Casa dos Pombos", "Se n+1 objetos são colocados em n caixas, pelo menos uma caixa contém ≥2 objetos. Simples mas poderoso em provas de existência."),
    ("Teorema Binomial", "(x+y)ⁿ = Σ_{k=0}ⁿ C(n,k) x^(n-k) y^k. Os coeficientes binomiais aparecem em probabilidade, combinatória e álgebra."),
    ("Fórmula de Stirling", "n! ~ √(2πn)·(n/e)ⁿ. Aproximação assintótica fundamental para fatoriais, usada em probabilidade e análise de algoritmos."),
]

DEFINITIONS = [
    ("Limite de uma função", "lim_{x→a} f(x) = L significa que para todo ε > 0, existe δ > 0 tal que 0 < |x-a| < δ implica |f(x)-L| < ε. Esta definição epsilon-delta, devida a Weierstrass, formaliza o conceito intuitivo de 'aproximação arbitrária'."),
    ("Derivada", "f'(x) = lim_{h→0} [f(x+h)-f(x)]/h. A derivada mede a taxa de variação instantânea. Geometricamente, é a inclinação da reta tangente ao gráfico de f no ponto x."),
    ("Integral de Riemann", "∫_a^b f(x)dx = lim_{n→∞} Σ_{i=1}ⁿ f(x_i*)Δx. A integral mede a área sob a curva. A partição do intervalo [a,b] é refinada até o limite."),
    ("Espaço Vetorial", "Conjunto V com operações de adição e multiplicação por escalar satisfazendo 8 axiomas: associatividade, comutatividade, elemento neutro, inverso aditivo, distributividade, etc."),
    ("Grupo", "Conjunto G com operação binária associativa, elemento identidade e inversos para todos elementos. Os grupos capturam a essência da simetria em matemática."),
    ("Anel", "Conjunto R com duas operações (adição e multiplicação) onde (R,+) é grupo abeliano, a multiplicação é associativa e distributiva sobre a adição."),
    ("Corpo", "Anel comutativo onde todo elemento não-nulo tem inverso multiplicativo. Exemplos: Q, R, C, Z_p (p primo). Estrutura fundamental da álgebra."),
    ("Espaço Topológico", "Conjunto X com uma coleção τ de subconjuntos (abertos) fechada sob uniões arbitrárias e interseções finitas. Generaliza noções de continuidade e convergência."),
    ("Variedade", "Espaço topológico localmente homeomorfo a Rⁿ. As variedades são os objetos fundamentais da geometria diferencial e da topologia."),
    ("Homotopia", "Duas funções contínuas f,g: X→Y são homotópicas se existe H: X×[0,1]→Y contínua com H(x,0)=f(x) e H(x,1)=g(x). Captura a ideia de deformação contínua."),
    ("Número Primo", "Inteiro p > 1 cujos únicos divisores positivos são 1 e p. Os primos são os 'átomos' da aritmética — todo inteiro se fatora unicamente neles."),
    ("Congruência", "a ≡ b (mod m) significa que m divide a-b. A aritmética modular é a base da criptografia moderna e da teoria dos números computacional."),
    ("Probabilidade Condicional", "P(A|B) = P(A∩B)/P(B). A probabilidade de A dado que B ocorreu. Fundamental para inferência bayesiana e aprendizado de máquina."),
    ("Variável Aleatória", "Função X: Ω → R que associa um número real a cada resultado do espaço amostral. Permite quantificar incerteza e modelar fenômenos aleatórios."),
    ("Autovalor e Autovetor", "Para matriz A, se Av = λv com v ≠ 0, então λ é autovalor e v é autovetor. Autovetores indicam direções invariantes sob a transformação A."),
]

IDENTITIES = [
    ("Identidade de Euler", "e^(iπ) + 1 = 0. Considerada a mais bela equação da matemática, conecta cinco constantes fundamentais: e, i, π, 1, 0."),
    ("Fórmula de Euler", "e^(ix) = cos(x) + i·sin(x). Unifica exponenciais complexas com trigonometria. Base da análise de Fourier e processamento de sinais."),
    ("Identidade Trigonométrica Fundamental", "sin²(x) + cos²(x) = 1. Decorre do teorema de Pitágoras aplicado ao círculo unitário."),
    ("Fórmula do Binômio de Newton", "(x+y)ⁿ = Σ_{k=0}ⁿ C(n,k)x^(n-k)y^k. Generaliza produtos notáveis para qualquer expoente natural."),
    ("Soma de Gauss", "Σ_{k=1}ⁿ k = n(n+1)/2. A soma dos primeiros n números naturais. Descoberta por Gauss aos 7 anos."),
    ("Soma dos Quadrados", "Σ_{k=1}ⁿ k² = n(n+1)(2n+1)/6. Aparece em cálculo de volumes, momentos de inércia e análise de algoritmos."),
    ("Série Geométrica", "Σ_{k=0}ⁿ ar^k = a(1-r^(n+1))/(1-r) para r≠1. Se |r|<1, a série infinita converge para a/(1-r)."),
    ("Série Harmônica", "Σ_{n=1}∞ 1/n diverge, mas Σ_{n=1}∞ 1/n² = π²/6 (problema de Basel, resolvido por Euler em 1734)."),
    ("Identidade de Parseval", "∫|f(x)|²dx = Σ|c_n|². A energia de um sinal é preservada na transformada de Fourier. Fundamental em processamento de sinais."),
    ("Fórmula de Stirling", "n! ~ √(2πn)(n/e)ⁿ. Aproximação assintótica do fatorial. O erro relativo tende a zero quando n→∞."),
    ("Identidade de Vandermonde", "Σ_{k} C(r,k)·C(s,n-k) = C(r+s,n). Convolução de coeficientes binomiais."),
    ("Teorema de Pitágoras", "a² + b² = c² para um triângulo retângulo de catetos a,b e hipotenusa c. Conhecido desde a Babilônia antiga."),
]

def generate_monster_doc():
    """Gera um documento denso de matemática pura."""
    category = random.random()
    if category < 0.4:
        name, text = random.choice(THEOREMS)
        header = f"Teorema: {name}"
    elif category < 0.75:
        name, text = random.choice(DEFINITIONS)
        header = f"Definição: {name}"
    else:
        name, text = random.choice(IDENTITIES)
        header = f"Identidade: {name}"
    
    # Expandir com contexto
    contexts = [
        f"\n\nContexto histórico: Este resultado foi desenvolvido ao longo de séculos de investigação matemática. Sua importância transcende a matemática pura, encontrando aplicações em física, engenharia, ciência da computação e economia.",
        f"\n\nDemonstração (esboço): A prova segue por {random.choice(['indução', 'redução ao absurdo', 'construção direta', 'argumento de ponto fixo', 'diagonalização'])}. O caso base é verificado diretamente. O passo indutivo utiliza a hipótese para estender o resultado a estruturas mais complexas.",
        f"\n\nAplicações: Este teorema é fundamental em {random.choice(['criptografia', 'compressão de dados', 'aprendizado de máquina', 'processamento de sinais', 'otimização', 'mecânica quântica', 'relatividade geral', 'teoria dos jogos'])}.",
        f"\n\nGeneralizações: O resultado pode ser estendido para {random.choice(['espaços de Banach', 'variedades Riemannianas', 'categorias abelianas', 'espaços de Sobolev', 'grupos topológicos'])}.",
        f"\n\nCorolários importantes: {random.choice(['O teorema da função implícita', 'A existência de bases ortogonais', 'A unicidade da fatoração', 'A convergência de algoritmos iterativos'])} seguem diretamente deste resultado.",
    ]
    
    return header + "\n" + text + random.choice(contexts)

# ═══════════════════════════════════════════════════
# B) MATH V3 — Problemas com Solução Passo a Passo
# ═══════════════════════════════════════════════════

PROBLEM_TEMPLATES = [
    # Cálculo
    {
        "domain": "Cálculo Diferencial",
        "problem": "Calcule a derivada de f(x) = {a}x^{n} + {b}x^{m} + {c}.",
        "steps": [
            "Passo 1 — Regra da potência: d/dx(x^k) = k·x^(k-1)",
            "Passo 2 — Aplicar a cada termo: d/dx({a}x^{n}) = {a}·{n}·x^{{{n-1}}}",
            "Passo 3 — d/dx({b}x^{m}) = {b}·{m}·x^{{{m-1}}}",
            "Passo 4 — d/dx({c}) = 0 (constante)",
            "Passo 5 — Somar os termos: f'(x) = {a*n}·x^{{{n-1}}} + {b*m}·x^{{{m-1}}}",
        ],
        "answer": "f'(x) = {a*n}x^{{{n-1}}} + {b*m}x^{{{m-1}}}",
    },
    {
        "domain": "Integrais",
        "problem": "Calcule ∫ ({a}x^{n} + {b}x^{m}) dx.",
        "steps": [
            "Passo 1 — Regra da potência para integração: ∫x^k dx = x^(k+1)/(k+1) + C",
            "Passo 2 — ∫{a}x^{n}dx = {a}·x^{{{n+1}}}/{n+1}",
            "Passo 3 — ∫{b}x^{m}dx = {b}·x^{{{m+1}}}/{m+1}",
            "Passo 4 — Resultado: ({a}/{n+1})x^{{{n+1}}} + ({b}/{m+1})x^{{{m+1}}} + C",
        ],
        "answer": "({a}/{n+1})x^{{{n+1}}} + ({b}/{m+1})x^{{{m+1}}} + C",
    },
    {
        "domain": "Integrais Definidas",
        "problem": "Calcule ∫_{{{lower}}}^{{{upper}}} ({a}x^{n} + {b}) dx.",
        "steps": [
            "Passo 1 — Antiderivada: F(x) = ({a}/{n+1})x^{{{n+1}}} + {b}x",
            "Passo 2 — Teorema Fundamental: ∫_a^b f(x)dx = F(b) - F(a)",
            "Passo 3 — F({upper}) = ({a}/{n+1})·{upper}^{{{n+1}}} + {b}·{upper} = {a*upper**(n+1)/(n+1) + b*upper:.1f}",
            "Passo 4 — F({lower}) = ({a}/{n+1})·{lower}^{{{n+1}}} + {b}·{lower} = {a*lower**(n+1)/(n+1) + b*lower:.1f}",
            "Passo 5 — Resultado: F({upper}) - F({lower})",
        ],
        "answer": "A integral definida vale aproximadamente {abs(a*upper**(n+1)/(n+1) + b*upper - a*lower**(n+1)/(n+1) - b*lower):.2f}",
    },
    # Álgebra
    {
        "domain": "Sistemas Lineares",
        "problem": "Resolva o sistema: {a1}x + {b1}y = {c1}, {a2}x + {b2}y = {c2}.",
        "steps": [
            "Passo 1 — Método da eliminação: multiplicar equações para igualar coeficientes",
            "Passo 2 — Multiplicar primeira por {a2}: {a1*a2}x + {b1*a2}y = {c1*a2}",
            "Passo 3 — Multiplicar segunda por {a1}: {a2*a1}x + {b2*a1}y = {c2*a1}",
            "Passo 4 — Subtrair: ({b1*a2} - {b2*a1})y = {c1*a2} - {c2*a1}",
            "Passo 5 — y = ({c1*a2 - c2*a1})/({b1*a2 - b2*a1}) = {(c1*a2 - c2*a1)/(b1*a2 - b2*a1):.2f}",
            "Passo 6 — Substituir y na primeira: x = ({c1} - {b1}·{(c1*a2 - c2*a1)/(b1*a2 - b2*a1):.2f})/{a1}",
        ],
        "answer": "x = {(a1*b2*c1 - a1*b1*c2 - a2*b1*c1 + a1*b1*c2)/(a1*(a1*b2 - a2*b1)) if (a1*b2 - a2*b1) != 0 else 0:.2f}, y = {(c1*a2 - c2*a1)/(b1*a2 - b2*a1) if (b1*a2 - b2*a1) != 0 else 0:.2f}",
    },
    # Teoria dos Números
    {
        "domain": "Teoria dos Números",
        "problem": "Determine se {n} é primo e encontre sua fatoração.",
        "steps": [
            "Passo 1 — Verificar divisibilidade pelos primeiros primos",
            "Passo 2 — Testar 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31...",
            "Passo 3 — Encontrar divisores através de divisão tentada até √{n}",
            "Passo 4 — Construir a fatoração completa",
        ],
        "answer": "A fatoração de {n} será determinada testando divisibilidade.",
    },
    # Combinatória
    {
        "domain": "Combinatória",
        "problem": "De quantas maneiras podemos escolher {k} elementos de um conjunto de {n} elementos?",
        "steps": [
            "Passo 1 — Identificar: é um problema de combinação (ordem não importa)",
            "Passo 2 — Fórmula: C({n},{k}) = {n}!/({k}!·({n}-{k})!)",
            "Passo 3 — Simplificar: C({n},{k}) = {n}·{n-1}·...·{n-k+1}/{k}!",
            "Passo 4 — Calcular numerador: {n}·{n-1} = {n*(n-1)} (simplificado)",
            "Passo 5 — Dividir por {k}! = {k} = ...",
        ],
        "answer": "C({n},{k}) = {n*(n-1)//2 if k==2 else n} maneiras",
    },
    # Probabilidade
    {
        "domain": "Probabilidade",
        "problem": "Ao lançar {n} moedas justas, qual a probabilidade de obter exatamente {k} caras?",
        "steps": [
            "Passo 1 — Distribuição binomial: X ~ Bin({n}, 0.5)",
            "Passo 2 — P(X={k}) = C({n},{k}) · (0.5)^{k} · (0.5)^{{{n-k}}}",
            "Passo 3 — C({n},{k}) = {n}!/({k}!·{n-k}!)",
            "Passo 4 — P(X={k}) = C({n},{k}) · (0.5)^{n}",
            "Passo 5 — Resultado numérico aproximado",
        ],
        "answer": "P({k} caras) = C({n},{k})/2^{n}",
    },
]

def generate_problem_doc():
    """Gera um problema com solução passo a passo."""
    tmpl = random.choice(PROBLEM_TEMPLATES)
    
    # Fill parameters
    a = random.randint(1, 9)
    b = random.randint(1, 9)
    c = random.randint(1, 9)
    n = random.randint(2, 8)
    m = random.randint(1, n-1) if n > 1 else 1
    k = random.randint(1, min(3, n))
    lower = random.randint(0, 3)
    upper = random.randint(4, 10)
    a1, b1, c1 = random.randint(1,5), random.randint(1,5), random.randint(1,20)
    a2, b2, c2 = random.randint(1,5), random.randint(1,5), random.randint(1,20)
    
    text = f"Domínio: {tmpl['domain']}\n\n"
    text += f"Problema: {tmpl['problem']}\n\n"
    text += "Solução passo a passo:\n\n"
    
    for step in tmpl["steps"]:
        text += step + "\n\n"
    
    text += f"Resposta: {tmpl['answer']}\n"
    
    # Fill in the actual numbers
    replacements = {
        "{a}": str(a), "{b}": str(b), "{c}": str(c),
        "{n}": str(n), "{m}": str(m), "{k}": str(k),
        "{lower}": str(lower), "{upper}": str(upper),
        "{a1}": str(a1), "{b1}": str(b1), "{c1}": str(c1),
        "{a2}": str(a2), "{b2}": str(b2), "{c2}": str(c2),
        "{a*n}": str(a*n), "{b*m}": str(b*m),
        "{n-1}": str(n-1), "{m-1}": str(m-1),
        "{n+1}": str(n+1), "{m+1}": str(m+1),
        "{n-k}": str(n-k),
        "{a1*a2}": str(a1*a2), "{b1*a2}": str(b1*a2), "{c1*a2}": str(c1*a2),
        "{a2*a1}": str(a2*a1), "{b2*a1}": str(b2*a1), "{c2*a1}": str(c2*a1),
    }
    
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # Also replace double-braced numbers (from string formatting)
    import re
    text = re.sub(r'\{\{(\d+)\}\}', r'\1', text)
    
    return text

# ═══════════════════════════════════════════════════
# C) PROOF DATABASE — Teoremas Clássicos com Demonstrações
# ═══════════════════════════════════════════════════

PROOFS = [
    {
        "theorem": "A raiz quadrada de 2 é irracional",
        "domain": "Teoria dos Números",
        "proof": """Prova por contradição (reductio ad absurdum):

Suponha que √2 é racional. Então existem inteiros p, q coprimos (gcd(p,q)=1) tais que √2 = p/q.

Elevando ao quadrado: 2 = p²/q² → p² = 2q².

Portanto p² é par. Se p² é par, então p é par (pois ímpar² = ímpar). Escreva p = 2k.

Substituindo: (2k)² = 2q² → 4k² = 2q² → 2k² = q².

Logo q² é par, e portanto q é par. Mas isso contradiz gcd(p,q) = 1.

A contradição prova que √2 não pode ser racional. ∎"""
    },
    {
        "theorem": "Existem infinitos números primos",
        "domain": "Teoria dos Números",
        "proof": """Prova por contradição (Euclides, ~300 a.C.):

Suponha que existe apenas um número finito de primos: p₁, p₂, ..., pₙ.

Considere N = p₁ · p₂ · ... · pₙ + 1.

N não é divisível por nenhum pᵢ (pois deixa resto 1). Portanto N é primo ou tem um fator primo diferente de todos os pᵢ.

Em qualquer caso, encontramos um primo fora da lista finita — contradição.

Logo, existem infinitos números primos. ∎"""
    },
    {
        "theorem": "A série harmônica diverge",
        "domain": "Análise Real",
        "proof": """Prova por agrupamento (Oresme, ~1350):

Considere a série harmônica H = 1 + 1/2 + 1/3 + 1/4 + ...

Agrupe os termos em blocos de potências de 2:

1/3 + 1/4 > 1/4 + 1/4 = 1/2
1/5 + 1/6 + 1/7 + 1/8 > 4·(1/8) = 1/2
1/9 + ... + 1/16 > 8·(1/16) = 1/2

Cada bloco de 2ᵏ termos é maior que 1/2. Como há infinitos blocos, a soma diverge para infinito. ∎"""
    },
    {
        "theorem": "e é irracional",
        "domain": "Análise Real",
        "proof": """Prova por contradição usando a série de Taylor.

Suponha e = a/b com a,b inteiros positivos.

e = Σ_{n=0}∞ 1/n! = 1 + 1/1! + 1/2! + ... + 1/b! + R, onde R = Σ_{n=b+1}∞ 1/n!

Multiplique por b!: b!·e = b!·(1 + 1/1! + ... + 1/b!) + b!·R

b!·R = 1/(b+1) + 1/((b+1)(b+2)) + ... < 1/(b+1) + 1/(b+1)² + ... = 1/b ≤ 1

Portanto b!·R está estritamente entre 0 e 1, mas b!·e deve ser inteiro — contradição. ∎"""
    },
    {
        "theorem": "O conjunto dos números reais é não-enumerável",
        "domain": "Teoria dos Conjuntos",
        "proof": """Prova por diagonalização (Cantor, 1891):

Suponha que [0,1] é enumerável. Liste todos os números em [0,1]:

x₁ = 0.a₁₁ a₁₂ a₁₃ ...
x₂ = 0.a₂₁ a₂₂ a₂₃ ...
x₃ = 0.a₃₁ a₃₂ a₃₃ ...
...

Construa y = 0.b₁b₂b₃... onde bᵢ = 5 se aᵢᵢ ≠ 5, e bᵢ = 7 se aᵢᵢ = 5.

y difere de cada xᵢ na i-ésima casa decimal → y não está na lista.

Portanto [0,1] é não-enumerável. ∎"""
    },
]

def generate_proof_doc():
    """Gera uma demonstração completa de um teorema clássico."""
    p = random.choice(PROOFS)
    return f"Teorema: {p['theorem']}\nDomínio: {p['domain']}\n\nDemonstração:\n{p['proof']}"

# ═══════════════════════════════════════════════════
# PARALLEL GENERATION ENGINE
# ═══════════════════════════════════════════════════

GENERATORS = [
    ("monster", generate_monster_doc, 0.40),   # 40% teoremas/definições
    ("problem", generate_problem_doc, 0.40),    # 40% problemas
    ("proof", generate_proof_doc, 0.20),         # 20% demonstrações
]

def generate_chunk(args):
    chunk_id, chunk_size, output_dir, seed = args
    random.seed(seed)
    
    names = [g[0] for g in GENERATORS]
    weights = [g[2] for g in GENERATORS]
    funcs = [g[1] for g in GENERATORS]
    
    docs = []
    for _ in range(chunk_size):
        gen = random.choices(funcs, weights=weights, k=1)[0]
        docs.append(gen())
    
    path = Path(output_dir) / f"math_overdrive_{chunk_id:06d}.txt"
    path.write_text("\n\n---\n\n".join(docs), encoding="utf-8")
    return chunk_id, len(docs), path.stat().st_size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=1_000_000, help="Total documents")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--output", default="data/corpus/math_overdrive")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_chunks = (args.docs + args.chunk_size - 1) // args.chunk_size
    
    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  F51 MATH OVERDRIVE — 3 GERADORES               ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  Docs:    {args.docs:,}                                   ║")
    print(f"║  Chunks:  {num_chunks:,} × {args.chunk_size:,}                          ║")
    print(f"║  Workers: {args.workers}                                          ║")
    print(f"║  A) Teoremas/Definições (40%)                    ║")
    print(f"║  B) Problemas passo-a-passo (40%)                ║")
    print(f"║  C) Demonstrações clássicas (20%)                ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    tasks = [(i, min(args.chunk_size, args.docs - i * args.chunk_size),
              str(output_dir), hash(f"math_overdrive_{i}") % (2**31))
             for i in range(num_chunks)]

    t0 = time.time()
    total_size = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(generate_chunk, t): t for t in tasks}
        for future in as_completed(futures):
            _, n, size = future.result()
            total_size += size
            completed += 1
            if completed % max(1, num_chunks // 50) == 0:
                pct = completed / num_chunks * 100
                bar = "█" * int(pct/2) + "░" * (50 - int(pct/2))
                elapsed = time.time() - t0
                rate = (completed * args.chunk_size) / elapsed if elapsed > 0 else 0
                print(f"\r[{bar}] {completed}/{num_chunks} | {total_size/1e9:.1f} GB | {rate/1e6:.1f}M governance/docs/s", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n\n✅ {args.docs:,} docs em {elapsed:.0f}s ({args.governance/docs/elapsed/1e6:.1f}M governance/docs/s)")
    print(f"   Tamanho: {total_size/1e9:.1f} GB")
    print(f"   Dir:     {output_dir}")

if __name__ == "__main__":
    main()
