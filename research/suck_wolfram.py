#!/usr/bin/env python
"""
F51 Wolfram Mass Extractor — Suga Tudo Sem Pena
================================================
Extrai TODO conhecimento matemático da API Wolfram Alpha.
Matemática pura é a base de tudo — lógica, teoremas, provas.

Estratégia:
  1. Catálogo de 5.000+ queries cobrindo toda matemática pura
  2. Rate limiting respeitoso (~2 queries/segundo)
  3. Cache agressivo — nunca repete query
  4. Salva cada resposta como documento de treino
  5. Roda até acabar a cota da API

Uso:
  python research/suck_wolfram.py
  python research/suck_wolfram.py --domain number_theory --count 500
"""

import sys, time, json, random
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.wolfram_bridge import WolframBridge, WolframConfig, WolframResult

# ═══════════════════════════════════════════════════════════
# CATÁLOGO DE QUERIES — Matemática Pura sem Pena
# ═══════════════════════════════════════════════════════════

DOMAINS = {
    "number_theory": {
        "label": "Teoria dos Números",
        "weight": 5,
        "queries": [
            # Primes
            "prime factors of {n}",
            "is {n} prime",
            "next prime after {n}",
            "{n}th prime",
            "primes between {a} and {b}",
            "Mersenne prime exponents",
            "prime counting function pi({n})",
            "twin primes less than {n}",
            "Goldbach partitions of {n}",
            "Riemann zeta function zeta({x})",
            # Divisibility
            "gcd({a}, {b})",
            "lcm({a}, {b})",
            "divisors of {n}",
            "sigma({n}) divisor function",
            "phi({n}) Euler totient",
            "mu({n}) Moebius function",
            "is {n} a perfect number",
            "Carmichael numbers less than {n}",
            # Congruences
            "{a}^({b}) mod {m}",
            "solve x^2 = {a} mod {p}",
            "primitive roots modulo {p}",
            "legendre symbol ({a}/{p})",
            "quadratic residues modulo {p}",
            "Chinese remainder theorem {a} mod {m1}, {b} mod {m2}",
            # Diophantine
            "integer solutions to x^{e1} + y^{e2} = {c}",
            "Pythagorean triples with hypotenuse {c}",
            "continued fraction of sqrt({n})",
            "convergents of sqrt({n})",
            "Pell equation x^2 - {d}*y^2 = 1",
            # Special numbers
            "{n}th Fibonacci number",
            "{n}th triangular number",
            "{n}th Bell number",
            "{n}th Bernoulli number",
            "{n}th partition number p({n})",
            "pi digits {n}",
            "e digits {n}",
            "golden ratio digits {n}",
            "Euler-Mascheroni constant digits",
        ],
    },

    "calculus": {
        "label": "Cálculo e Análise",
        "weight": 5,
        "queries": [
            # Derivatives
            "derivative of {expr}",
            "{n}th derivative of {expr}",
            "partial derivative d/dx of {expr}",
            "gradient of {expr}",
            "Hessian of {expr}",
            "Jacobian of [{f1},{f2}]",
            "Laplacian of {expr}",
            # Integrals
            "integrate {expr}",
            "definite integral of {expr} from {a} to {b}",
            "double integral of {expr}",
            "triple integral of {expr}",
            "line integral of {expr}",
            "surface integral of {expr}",
            # Limits
            "limit of {expr} as x->{a}",
            "limit of {expr} as x->infinity",
            "limit of {expr} as x->0",
            "one-sided limit of {expr}",
            # Series
            "sum of {series_expr} from n={a} to infinity",
            "Taylor series of {expr} at x={a}",
            "Maclaurin series of {expr}",
            "Fourier series of {expr}",
            "Laurent series of {expr}",
            "radius of convergence of sum {series_expr}",
            # Transforms
            "Laplace transform of {expr}",
            "inverse Laplace transform of {expr}",
            "Fourier transform of {expr}",
            "Z-transform of {expr}",
            # ODEs
            "solve y'' + {a}*y' + {b}*y = {c}",
            "solve y' = {expr}",
            "general solution of {ode}",
            "particular solution of {ode}, y(0)={y0}",
            "phase portrait of {system}",
            # PDEs
            "solve wave equation u_tt = c^2 u_xx",
            "solve heat equation u_t = alpha u_xx",
            "solve Laplace equation u_xx + u_yy = 0",
        ],
    },

    "linear_algebra": {
        "label": "Álgebra Linear",
        "weight": 4,
        "queries": [
            # Matrices
            "eigenvalues of {{ {rows} }}",
            "eigenvectors of {{ {rows} }}",
            "determinant of {{ {rows} }}",
            "inverse of {{ {rows} }}",
            "trace of {{ {rows} }}",
            "rank of {{ {rows} }}",
            "null space of {{ {rows} }}",
            "characteristic polynomial of {{ {rows} }}",
            "matrix exponential of {{ {rows} }}",
            "SVD of {{ {rows} }}",
            "QR decomposition of {{ {rows} }}",
            "LU decomposition of {{ {rows} }}",
            "Jordan canonical form of {{ {rows} }}",
            "diagonalize {{ {rows} }}",
            # Vectors
            "norm of ({v})",
            "dot product ({v1}) . ({v2})",
            "cross product ({v1}) x ({v2})",
            "angle between ({v1}) and ({v2})",
            "projection of ({v1}) onto ({v2})",
            "Gram-Schmidt ({v1}), ({v2}), ({v3})",
            # Systems
            "solve linear system {eq1}, {eq2}, {eq3}",
            "row reduce {{ {rows} }}",
        ],
    },

    "abstract_algebra": {
        "label": "Álgebra Abstrata",
        "weight": 4,
        "queries": [
            # Groups
            "group order of {group}",
            "subgroups of {group}",
            "is {group} abelian",
            "center of {group}",
            "symmetric group S{n}",
            "alternating group A{n}",
            "cyclic group C{n}",
            "dihedral group D{n}",
            # Rings & Fields
            "polynomial ring Z[{x}]",
            "Galois group of x^{n} - {a}",
            "splitting field of x^{n} - {a}",
            "finite field GF({p}^{n})",
            "algebraic number {expr}",
            "minimal polynomial of {expr}",
            # Polynomials
            "factor {polynomial}",
            "roots of {polynomial}",
            "discriminant of {polynomial}",
            "resultant of {p1} and {p2}",
            "Groebner basis of {system}",
            "symmetric polynomials of {vars}",
            "cyclotomic polynomial Phi_{n}(x)",
        ],
    },

    "combinatorics": {
        "label": "Combinatória e Grafos",
        "weight": 3,
        "queries": [
            # Counting
            "{n} choose {k}",
            "{n} multichoose {k}",
            "permutations of {n} elements",
            "derangements of {n}",
            "Stirling numbers of the second kind S({n},{k})",
            "Catalan number C({n})",
            "integer partitions of {n}",
            "compositions of {n}",
            # Graph theory
            "complete graph K{n}",
            "complete bipartite graph K({a},{b})",
            "Petersen graph",
            "chromatic number of {graph}",
            "chromatic polynomial of {graph}",
            "matching number of {graph}",
            "Hamiltonian cycle in {graph}",
            "Eulerian path in {graph}",
            "adjacency matrix of {graph}",
            "graph diameter of {graph}",
            "minimum spanning tree of {graph}",
            "shortest path {a} to {b} in {graph}",
            "max flow {s} to {t} in {graph}",
        ],
    },

    "geometry": {
        "label": "Geometria e Topologia",
        "weight": 4,
        "queries": [
            # Euclidean
            "area of triangle with sides {a},{b},{c}",
            "area of circle radius {r}",
            "volume of sphere radius {r}",
            "surface area of torus radii {R},{r}",
            "distance between ({x1},{y1}) and ({x2},{y2})",
            "intersection of line {line} and circle {circle}",
            "ellipse with axes {a},{b}",
            "parabola y = {a}x^2 + {b}x + {c}",
            "hyperbola x^2/{a}^2 - y^2/{b}^2 = 1",
            # Transformations
            "rotate ({x},{y}) by {theta} degrees",
            "reflect ({x},{y}) across line y={m}x+{b}",
            "translation of ({x},{y}) by ({dx},{dy})",
            # Topology
            "fundamental group of {space}",
            "homology of {space}",
            "Euler characteristic of {space}",
            "genus of {surface}",
            "knot polynomial of {knot}",
            # Differential geometry
            "curvature of {curve}",
            "torsion of {curve}",
            "Gaussian curvature of {surface}",
            "geodesics of {surface}",
            "metric tensor of {space}",
        ],
    },

    "statistics": {
        "label": "Estatística e Probabilidade",
        "weight": 3,
        "queries": [
            # Distributions
            "normal distribution mean {mu} sd {sigma}",
            "binomial distribution n={n} p={p}",
            "Poisson distribution lambda={l}",
            "exponential distribution lambda={l}",
            "gamma distribution alpha={a} beta={b}",
            "beta distribution alpha={a} beta={b}",
            "chi-squared distribution df={df}",
            "t-distribution df={df}",
            "F-distribution df1={d1} df2={d2}",
            # Probability
            "probability of {event}",
            "expected value of {expr}",
            "variance of {expr}",
            "standard deviation of {expr}",
            "moment generating function of {dist}",
            "Bayes theorem P(A|B) given {data}",
            # Inference
            "confidence interval for mean sample={data}",
            "hypothesis test {test} with data {data}",
            "linear regression {xdata} vs {ydata}",
            "correlation coefficient of {data}",
            "ANOVA of {groups}",
        ],
    },
}

# ═══════════════════════════════════════════════════════════
# GERADOR DE QUERIES — preenche placeholders com números reais
# ═══════════════════════════════════════════════════════════

def random_int(a: int = 2, b: int = 9999) -> int:
    return random.randint(a, b)

def random_prime() -> int:
    return random.choice([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,
                          101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199])

def random_float() -> float:
    return round(random.uniform(0.1, 10.0), 3)

def random_matrix(rows: int = 3) -> str:
    return ", ".join("{" + ",".join(str(random_int(1, 9)) for _ in range(rows)) + "}" for _ in range(rows))

def random_vec(dim: int = 3) -> str:
    return ",".join(str(random_int(0, 9)) for _ in range(dim))

def random_poly() -> str:
    degree = random.randint(2, 5)
    terms = []
    for i in range(degree, -1, -1):
        coef = random.randint(-9, 9)
        if coef == 0:
            continue
        if i == 0:
            terms.append(f"{coef:+d}")
        elif i == 1:
            terms.append(f"{coef:+d}x")
        else:
            terms.append(f"{coef:+d}x^{i}")
    return "".join(terms).lstrip("+") or "0"

def random_expr() -> str:
    funcs = ["sin", "cos", "tan", "exp", "log", "sqrt", "abs"]
    f = random.choice(funcs)
    a = random_int(1, 5)
    b = random_int(0, 3)
    c = random_int(1, 5)
    return f"({a}*{f}({c}*x^{b}))"

def random_ode() -> str:
    a, b, c = random_int(1, 5), random_int(-5, 5), random_int(-5, 5)
    return f"y'' + {a}*y' + {b}*y = sin({c}*x)"

def random_series() -> str:
    a, b = random_int(1, 5), random_int(1, 3)
    return f"({a}*x^{b})/(n^{b}+1)"

def fill_placeholders(template: str) -> str:
    """Fill template placeholders with random values."""
    result = template
    for _ in range(20):  # safety limit
        modified = False
        for old, new in [
            ("{n}", str(random_int(2, 5000))),
            ("{a}", str(random_int(1, 20))),
            ("{b}", str(random_int(1, 20))),
            ("{c}", str(random_int(1, 20))),
            ("{d}", str(random_int(2, 10))),
            ("{e1}", str(random.choice([2,3,4,5]))),
            ("{e2}", str(random.choice([2,3,4,5]))),
            ("{p}", str(random_prime())),
            ("{m}", str(random_prime())),
            ("{m1}", str(random_prime())),
            ("{m2}", str(random_prime())),
            ("{x}", str(round(random.uniform(0.1, 5.0), 3))),
            ("{y}", str(round(random.uniform(0.1, 5.0), 3))),
            ("{R}", str(random_int(1, 10))),
            ("{r}", str(random_int(1, 10))),
            ("{theta}", str(random_int(0, 360))),
            ("{mu}", str(round(random.uniform(-5, 5), 1))),
            ("{sigma}", str(round(random.uniform(0.5, 5), 2))),
            ("{l}", str(round(random.uniform(0.5, 10), 1))),
            ("{df}", str(random_int(1, 30))),
            ("{expr}", random_expr()),
            ("{ode}", random_ode()),
            ("{series_expr}", random_series()),
            ("{rows}", random_matrix(random.choice([2,3,4]))),
            ("{v}", random_vec(random.choice([2,3,4]))),
            ("{v1}", random_vec()),
            ("{v2}", random_vec()),
            ("{v3}", random_vec()),
            ("{polynomial}", random_poly()),
            ("{graph}", random.choice(["K5","PetersenGraph","WheelGraph[6]","GridGraph[{3,3}]"])),
            ("{group}", random.choice(["S4","A5","D8","C12","Z5","SL(2,3)"])),
            ("{space}", random.choice(["circle","sphere","torus","Klein bottle","RP^2","Moebius strip"])),
            ("{surface}", random.choice(["sphere","torus","hyperboloid","catenoid","helicoid"])),
            ("{curve}", random.choice(["circle","helix","catenary","cycloid","tractrix"])),
            ("{knot}", random.choice(["trefoil","figure eight","cinquefoil","stevedore"])),
            ("{eq1}", f"{random_int(1,5)}x+{random_int(1,5)}y+{random_int(1,5)}z={random_int(1,10)}"),
            ("{eq2}", f"{random_int(1,5)}x+{random_int(1,5)}y+{random_int(1,5)}z={random_int(1,10)}"),
            ("{eq3}", f"{random_int(1,5)}x+{random_int(1,5)}y+{random_int(1,5)}z={random_int(1,10)}"),
            ("{system}", f"{random_poly()}, {random_poly()}"),
            ("{p1}", random_poly()),
            ("{p2}", random_poly()),
            ("{event}", random.choice(["rolling 7 with 2 dice","drawing ace from 52 cards","3 heads in 5 coin flips"])),
            ("{dist}", random.choice(["normal","binomial","Poisson","exponential"])),
            ("{test}", random.choice(["t-test","z-test","chi-squared","F-test"])),
            ("{data}", "{" + ",".join(str(random_int(1,100)) for _ in range(10)) + "}"),
            ("{xdata}", "{" + ",".join(str(random_int(1,50)) for _ in range(10)) + "}"),
            ("{ydata}", "{" + ",".join(str(random_int(1,50)) for _ in range(10)) + "}"),
            ("{groups}", "{" + ",".join("{" + ",".join(str(random_int(1,100)) for _ in range(5)) + "}" for _ in range(3)) + "}"),
            ("{vars}", random.choice(["x,y,z","a,b,c","u,v,w"])),
        ]:
            if old in result:
                result = result.replace(old, new, 1)
                modified = True
        if not modified:
            break
    return result

# ═══════════════════════════════════════════════════════════
# EXTRACTOR PRINCIPAL
# ═══════════════════════════════════════════════════════════

def extract_domain(bridge: WolframBridge, domain_key: str, count: int, output_dir: Path) -> int:
    """Extract math data from one domain."""
    domain = DOMAINS[domain_key]
    templates = domain["queries"]
    weight = domain["weight"]
    
    successful = 0
    documents = []
    
    for i in range(count):
        template = random.choice(templates)
        query = fill_placeholders(template)
        
        result = bridge.query(query)
        
        if result.success:
            doc = f"Query: {query}\n"
            doc += f"Domain: {domain['label']}\n"
            doc += f"Result: {result.result_text}\n"
            for pod in result.pods:
                if pod.get('title') and pod.get('text'):
                    doc += f"[{pod['title']}] {pod['text']}\n"
            
            documents.append(doc)
            successful += 1
        
        # Rate limiting: 2 queries/sec for free tier
        time.sleep(0.5)
        
        if (i + 1) % 50 == 0:
            print(f"\r  {domain['label']}: {i+1}/{count} queries, {successful} successful", end="", flush=True)
    
    # Save domain results
    if documents:
        output_path = output_dir / f"wolfram_{domain_key}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt"
        output_path.write_text("\n\n---\n\n".join(documents), encoding="utf-8")
        print(f"\r  {domain['label']}: {successful}/{count} successful → {output_path.name} ({output_path.stat().st_size/1e6:.0f} MB)")
    
    return successful

def main():
    import argparse
    parser = argparse.ArgumentParser(description="F51 Wolfram Mass Extractor")
    parser.add_argument("--domain", default="all", help="Domain to extract (all, number_theory, calculus, etc.)")
    parser.add_argument("--count", type=int, default=500, help="Queries per domain")
    parser.add_argument("--workers", type=int, default=4, help="Parallel domains")
    parser.add_argument("--output", default="data/corpus/wolfram_mass", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config Wolfram
    config = WolframConfig(
        app_id="W4E3KY-LEV3Q82QP9",
        cache_enabled=True,
        cache_ttl_hours=72,
        timeout_sec=20,
        max_retries=2,
    )
    bridge = WolframBridge(config)

    domains_to_extract = list(DOMAINS.keys()) if args.domain == "all" else [args.domain]

    total_queries = len(domains_to_extract) * args.count
    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  F51 WOLFRAM MASS EXTRACTOR — SEM PENA           ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  Domínios:     {len(domains_to_extract)}                                  ║")
    print(f"║  Queries/dom:  {args.count}                                ║")
    print(f"║  Total:        {total_queries:,} queries                      ║")
    print(f"║  Rate:         ~2 queries/s (free tier)         ║")
    print(f"║  Est. time:    {total_queries * 0.5 / 60:.0f} min                            ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    t0 = time.time()
    total_success = 0

    # Process domains sequentially to respect rate limits
    for domain_key in domains_to_extract:
        n = extract_domain(bridge, domain_key, args.count, output_dir)
        total_success += n

    elapsed = time.time() - t0
    print(f"\n✅ {total_success:,} respostas extraídas em {elapsed/60:.0f} min")
    print(f"   Taxa de sucesso: {total_success/total_queries*100:.0f}%")
    print(f"   Diretório: {output_dir}")

    # Merge all into one file
    print("\n📦 Consolidando...")
    all_docs = []
    for f in sorted(output_dir.glob("wolfram_*.txt")):
        text = f.read_text(encoding="utf-8")
        all_docs.append(text)
    
    merged = output_dir / "wolfram_math_corpus.txt"
    merged.write_text("\n\n---\n\n".join(all_docs), encoding="utf-8")
    print(f"   Corpus unificado: {merged.stat().st_size/1e6:.0f} MB")

if __name__ == "__main__":
    main()
