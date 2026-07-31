#!/usr/bin/env python
"""
F51 Math Corpus Generator — Usa Wolfram Alpha API para gerar dataset matematico.

Gera milhares de fatos matematicos verificados, organizados por dominio:
- algebra, calculus, number_theory, linear_algebra, statistics
- physics, chemistry, geometry, trigonometry, differential_equations

Cada fato eh verificado pelo Wolfram antes de entrar no corpus.
O modelo aprende matematica VERDADEIRA, nao alucinada.

Uso:
    python research/generate_math_corpus.py
    python research/generate_math_corpus.py --queries 500 --domain calculus
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.wolfram_bridge import WolframBridge, WolframConfig

# ═══════════════════════════════════════════════════════════
# QUERY TEMPLATES — Organizado por dominio
# ═══════════════════════════════════════════════════════════

MATH_QUERIES = {
    "algebra": [
        "solve x^{n} + {a}x + {b} = 0",
        "factor x^{n} + {a}x^{n-1} + {b}",
        "expand (x + {a})^{n}",
        "simplify (x^{n} - y^{n}) / (x - y)",
        "roots of x^{n} - {a}",
        "discriminant of x^{n} + {a}x^{n-1} + {b}x + {c}",
        "{a}x + {b}y = {c}, {d}x - {e}y = {f}",
        "gcd({a}, {b})",
        "lcm({a}, {b}, {c})",
        "is {n} prime",
        "prime factors of {n}",
        "phi({n})",
        "{n} mod {m}",
        "modular inverse of {a} mod {n}",
        "solve x^{a} mod {n} = {b}",
        "binomial({n}, {k})",
        "{n}!",
        "sum k^{a} for k=1 to {n}",
        "partial fractions of 1/(x^{a} + {b}x + {c})",
        "eigenvalues of [[{a},{b}],[{c},{d}]]",
        "determinant of [[{a},{b},{c}],[{d},{e},{f}],[{g},{h},{i}]]",
        "inverse of [[{a},{b}],[{c},{d}]]",
        "characteristic polynomial of [[{a},{b}],[{c},{d}]]",
        "transpose of [[{a},{b},{c}],[{d},{e},{f}]]",
        "rank of [[{a},{b},{c}],[{d},{e},{f}],[{g},{h},{i}]]",
        "null space of [[{a},{b}],[{c},{d}]]",
        "diagonalize [[{a},{b}],[{c},{d}]]",
        "SVD of [[{a},{b}],[{c},{d}]]",
        "dot product of ({a},{b},{c}) and ({d},{e},{f})",
        "cross product of ({a},{b},{c}) and ({d},{e},{f})",
        "angle between vectors ({a},{b}) and ({c},{d})",
    ],
    "calculus": [
        "derivative of x^{n}",
        "derivative of sin({a}x)",
        "derivative of e^({a}x)",
        "derivative of ln({a}x)",
        "derivative of tan(x^{a})",
        "derivative of x^{a} sin({b}x)",
        "second derivative of x^{n} e^{x}",
        "nth derivative of x^{a} for n={n}",
        "integrate x^{n}",
        "integrate sin({a}x)",
        "integrate e^({a}x)",
        "integrate 1/(x^{a} + {b})",
        "integrate x e^{x}",
        "integrate ln(x)",
        "integrate sec(x)",
        "definite integral of x^{a} from 0 to {b}",
        "definite integral of e^{-x} from 0 to infinity",
        "limit of sin(x)/x as x->0",
        "limit of (1 + 1/n)^{n} as n->infinity",
        "limit of (x^{n} - {a})/(x - {a}) as x->{a}",
        "taylor series of sin(x) at x=0",
        "taylor series of e^{x} at x=0",
        "taylor series of ln(1+x) at x=0",
        "radius of convergence of sum x^{n}/{n}!",
        "gradient of x^{a} + y^{b} + z^{c}",
        "partial derivative of x^{a}y^{b} with respect to x",
        "laplacian of 1/sqrt(x^{a}+y^{a}+z^{a})",
        "jacobian of (x*y, x+y, x/y)",
        "hessian of x^{a} + {b}xy + y^{a}",
    ],
    "number_theory": [
        "prime numbers between {a} and {b}",
        "is {n} a perfect square",
        "is {n} a triangular number",
        "is {n} a Fibonacci number",
        "goldbach partitions of {n}",
        "partitions of {n}",
        "sigma({n})",
        "tau({n})",
        "mobius({n})",
        "primitive root of {n}",
        "is {n} a perfect number",
        "mersenne prime exponents up to {n}",
        "continued fraction of sqrt({n})",
        "continued fraction of e",
        "diophantine equation {a}x + {b}y = {c}",
        "pythagorean triples with a={a}",
        "collatz sequence for {n}",
        "digits of pi to {n} places",
        "digits of e to {n} places",
        "golden ratio to {n} decimal places",
    ],
    "statistics": [
        "mean of {a}, {b}, {c}, {d}, {e}",
        "median of {a}, {b}, {c}, {d}, {e}",
        "standard deviation of {a}, {b}, {c}, {d}, {e}",
        "variance of {a}, {b}, {c}, {d}, {e}",
        "normal distribution with mean {a} and standard deviation {b}",
        "probability of z > {a} for standard normal",
        "binomial distribution n={a} p={b}",
        "poisson distribution lambda={a}",
        "chi-squared distribution df={a}",
        "t-distribution df={a}",
        "confidence interval 95% for mean={a} sd={b} n={c}",
        "correlation coefficient of ({a},{b},{c},{d},{e}) and ({e},{d},{c},{b},{a})",
        "linear regression {a}x + {b} through points (1,{c}), (2,{d}), (3,{e})",
        "bayes theorem P(A)={a} P(B|A)={b} P(B)={c}",
        "expected value of geometric distribution p={a}",
    ],
    "geometry": [
        "area of circle radius {a}",
        "circumference of circle radius {a}",
        "volume of sphere radius {a}",
        "surface area of sphere radius {a}",
        "volume of cylinder radius {a} height {b}",
        "volume of cone radius {a} height {b}",
        "area of triangle sides {a}, {b}, {c}",
        "pythagorean theorem a={a} b={b}",
        "distance between ({a},{b}) and ({c},{d})",
        "midpoint of ({a},{b}) and ({c},{d})",
        "area of regular {n}-gon side {a}",
        "interior angle of regular {n}-gon",
        "equation of circle center ({a},{b}) radius {c}",
        "intersection of line y={a}x+{b} and circle x^2+y^2={c}^2",
    ],
    "physics": [
        "kinetic energy mass={a} kg velocity={b} m/s",
        "potential energy mass={a} kg height={b} m gravity 9.8",
        "momentum mass={a} kg velocity={b} m/s",
        "force mass={a} kg acceleration={b} m/s^2",
        "work force={a} N distance={b} m",
        "power work={a} J time={b} s",
        "period of pendulum length={a} m gravity 9.8",
        "frequency of wave speed={a} m/s wavelength={b} m",
        "energy of photon wavelength={a} nm",
        "orbital velocity radius={a} m mass={b} kg",
        "escape velocity planet mass={a} kg radius={b} m",
        "relativistic mass m0={a} v={b}c",
        "time dilation v={a}c t0={b}s",
        "length contraction L0={a} v={b}c",
        "ohms law V={a} I={b}",
        "power electric V={a} I={b}",
        "resistance series R1={a} R2={b} R3={c}",
        "resistance parallel R1={a} R2={b}",
        "capacitance series C1={a} C2={b}",
        "inductance series L1={a} L2={b}",
    ],
    "trigonometry": [
        "sin({a} degrees)",
        "cos({a} degrees)",
        "tan({a} degrees)",
        "sin(pi/{a})",
        "cos(pi/{a})",
        "arcsin({a})",
        "arccos({a})",
        "arctan({a})",
        "sin({a}x) + cos({b}x) simplify",
        "sin^2({a}) + cos^2({a})",
        "tan^2({a}) + 1 = sec^2({a})",
        "sin({a}x)sin({b}x) product-to-sum",
        "sin({a}x) + sin({b}x) sum-to-product",
    ],
    "differential_equations": [
        "solve dy/dx = {a}y",
        "solve dy/dx + {a}y = {b}",
        "solve y'' + {a}y' + {b}y = 0",
        "solve y'' + {a}y = sin({b}x)",
        "solve y''' + {a}y'' + {b}y' = 0",
        "laplace transform of e^({a}t)",
        "laplace transform of t^{a}",
        "laplace transform of sin({a}t)",
        "inverse laplace of 1/(s+{a})",
        "inverse laplace of 1/(s^{a}+{b})",
        "fourier series of square wave",
        "fourier transform of e^{-{a}|t|}",
        "fourier transform of rect(t/{a})",
    ],
}


def generate_math_corpus(
    app_id: str = "W4E3KY-LEV3Q82QP9",
    queries_per_domain: int = 50,
    output_dir: str = "data/corpus",
):
    """Gera corpus matematico usando Wolfram Alpha."""

    import random
    rng = random.Random(42)

    bridge = WolframBridge(WolframConfig(
        app_id=app_id,
        cache_enabled=True,
        cache_ttl_hours=720,
        log_queries=True,
    ))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_documents = []
    stats = {}

    total_queries = queries_per_domain * len(MATH_QUERIES)
    print(f"F51 Math Corpus Generator")
    print(f"  Dominios: {len(MATH_QUERIES)}")
    print(f"  Queries por dominio: {queries_per_domain}")
    print(f"  Total queries: {total_queries}")
    print(f"  Wolfram: {'OK' if app_id else 'SEM API KEY'}")
    print()

    query_count = 0
    success_count = 0

    for domain, templates in sorted(MATH_QUERIES.items()):
        domain_docs = []
        domain_success = 0

        for i in range(queries_per_domain):
            template = rng.choice(templates)

            # Fill template with random values
            a = rng.randint(2, 20) if rng.random() > 0.3 else rng.randint(-10, -2)
            b = rng.randint(2, 20)
            c = rng.randint(2, 15)
            d = rng.randint(2, 15)
            e = rng.randint(2, 15)
            f = rng.randint(2, 15)
            g = rng.randint(2, 15)
            h = rng.randint(2, 15)
            i = rng.randint(2, 15) if rng.random() > 0.5 else rng.randint(-10, 10)
            n = rng.randint(2, 12)
            m = rng.randint(2, 20)
            k = rng.randint(1, n - 1) if n > 2 else 1
            p = round(rng.uniform(0.1, 0.9), 2)

            try:
                query = template.format(
                    a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h, i=i,
                    n=n, m=m, k=k, p=p,
                )
            except (KeyError, ValueError):
                continue

            query_count += 1

            try:
                result = bridge.query(query)
                time.sleep(0.05)

                if result.success and result.result_text:
                    doc = (
                        f"[MATH] [DOMAIN: {domain}]\n"
                        f"Query: {query}\n"
                        f"Result: {result.result_text}\n"
                        f"Verified: True\n"
                        f"---\n"
                    )
                    domain_docs.append(doc)
                    domain_success += 1
                    success_count += 1
                else:
                    # Even failed queries teach — log what we asked
                    doc = (
                        f"[MATH EXPLORATION] [DOMAIN: {domain}]\n"
                        f"Question: {query}\n"
                        f"---\n"
                    )
                    domain_docs.append(doc)
            except Exception as exc:
                print(f"  Erro [{domain}]: {query[:60]}... → {exc}")

            if query_count % 10 == 0:
                print(f"  Progresso: {query_count}/{total_queries} "
                      f"({success_count} verificados, {domain_success} neste dominio)")

        # Save per domain
        domain_file = output_path / f"math_{domain}_wolfram.txt"
        domain_content = "\n".join(domain_docs)
        domain_file.write_text(domain_content, encoding="utf-8")
        stats[domain] = {
            "queries": queries_per_domain,
            "verified": domain_success,
            "size_mb": round(len(domain_content) / 1e6, 2),
        }
        print(f"  {domain}: {domain_success}/{queries_per_domain} verificados "
              f"→ {domain_file.name} ({stats[domain]['size_mb']} MB)")

    # ── Summary ──
    total_size = sum(s["size_mb"] for s in stats.values())
    print(f"\n{'='*60}")
    print(f"  MATH CORPUS GERADO: {total_size:.1f} MB")
    print(f"  Total queries: {query_count}")
    print(f"  Verificados: {success_count} ({100*success_count/max(1,query_count):.0f}%)")
    print(f"{'='*60}")

    # Save summary
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wolfram_app_id": app_id[:8] + "..." if app_id else "none",
        "total_queries": query_count,
        "verified": success_count,
        "total_size_mb": total_size,
        "domains": stats,
    }
    (output_path / "math_corpus_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="F51 Math Corpus Generator")
    parser.add_argument("--queries", type=int, default=30,
                       help="Queries por dominio (default: 30)")
    parser.add_argument("--domain", default=None,
                       help="Dominio especifico (ou todos se omitido)")
    parser.add_argument("--app-id", default="W4E3KY-LEV3Q82QP9")
    parser.add_argument("--output", default="data/corpus")
    args = parser.parse_args()

    if args.domain:
        # Single domain
        MATH_QUERIES = {args.domain: MATH_QUERIES.get(args.domain, [])}
        if not MATH_QUERIES[args.domain]:
            print(f"Dominio '{args.domain}' nao encontrado.")
            print(f"Disponiveis: {', '.join(MATH_QUERIES.keys())}")
            sys.exit(1)

    generate_math_corpus(
        app_id=args.app_id,
        queries_per_domain=args.queries,
        output_dir=args.output,
    )
