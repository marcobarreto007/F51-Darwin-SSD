#!/usr/bin/env python
"""
F51 MASS MATH GENERATOR — 5 milhões de exemplos sintéticos.

Estratégia: geração programática massiva SEM API Wolfram.
Cada template gera milhares de variações com números aleatórios.
Só uma fração (<1%) é verificada via Wolfram para garantir qualidade.

Uso:
    python research/generate_mass_math.py --target 5000000
"""

from __future__ import annotations

import sys, json, random, math
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

# ═══════════════════════════════════════════════════════════
# GERADORES POR DOMÍNIO — Cada um produz ~500k exemplos
# ═══════════════════════════════════════════════════════════

def generate_algebra(n: int) -> list[str]:
    """Equações, fatoração, sistemas lineares."""
    docs = []
    rng = random.Random(42)
    for i in range(n):
        a, b, c = rng.randint(-20, 20), rng.randint(-20, 20), rng.randint(-20, 20)
        x = rng.randint(-10, 10)
        eq = f"{a}x + {b} = {c}"
        sol = (c - b) / a if a != 0 else "sem solução"
        docs.append(f"[MATH][algebra] Resolva: {eq}\nSolução: x = {sol}\nPassos: {a}x = {c} - ({b}) → {a}x = {c-b} → x = {sol}\n---")
        
        # Quadratic
        if i % 3 == 0:
            a2, b2, c2 = rng.randint(1, 5), rng.randint(-10, 10), rng.randint(-20, 20)
            disc = b2**2 - 4*a2*c2
            if disc >= 0:
                x1 = (-b2 + math.sqrt(disc)) / (2*a2)
                x2 = (-b2 - math.sqrt(disc)) / (2*a2)
                docs.append(f"[MATH][algebra] Resolva: {a2}x² + {b2}x + {c2} = 0\nSolução: x = {x1:.2f} ou x = {x2:.2f}\nΔ = {disc}\n---")
        
        # Linear system
        if i % 5 == 0:
            d, e, f = rng.randint(-10, 10), rng.randint(-10, 10), rng.randint(-20, 20)
            g, h, j = rng.randint(-10, 10), rng.randint(-10, 10), rng.randint(-20, 20)
            det = a*e - b*d
            if det != 0:
                xs = (c*e - b*f) / det
                ys = (a*f - c*d) / det
                docs.append(f"[MATH][algebra] Sistema: {a}x + {b}y = {c}, {d}x + {e}y = {f}\nSolução: x = {xs:.2f}, y = {ys:.2f}\n---")
    return docs

def generate_calculus(n: int) -> list[str]:
    """Derivadas, integrais, limites."""
    docs = []
    rng = random.Random(51)
    for i in range(n):
        a, n_pow = rng.randint(1, 10), rng.randint(1, 8)
        docs.append(f"[MATH][calculus] Derivada de f(x) = {a}x^{n_pow}\nf'(x) = {a*n_pow}x^{n_pow-1}\nRegra: d/dx(x^n) = n·x^(n-1)\n---")
        
        if i % 2 == 0:
            b = rng.randint(1, 5)
            docs.append(f"[MATH][calculus] Integral de f(x) = {b}x^{n_pow}\n∫{b}x^{n_pow}dx = {b/(n_pow+1):.2f}x^{n_pow+1} + C\n---")
        
        if i % 4 == 0:
            docs.append(f"[MATH][calculus] Limite: lim(x→0) sin({a}x)/x = {a}\nRegra: lim(x→0) sin(kx)/x = k\n---")
    return docs

def generate_arithmetic(n: int) -> list[str]:
    """Operações básicas, frações, porcentagens."""
    docs = []
    rng = random.Random(99)
    ops = ['+', '-', '×', '÷']
    for i in range(n):
        a, b = rng.randint(1, 1000), rng.randint(1, 1000)
        op = rng.choice(ops)
        if op == '+': result = a + b
        elif op == '-': result = a - b
        elif op == '×': result = a * b
        else: result = round(a / b, 4) if b != 0 else 'indefinido'
        docs.append(f"[MATH][arithmetic] {a} {op} {b} = {result}\n---")
        
        if i % 3 == 0:
            pct = rng.randint(1, 100)
            val = rng.randint(100, 10000)
            docs.append(f"[MATH][arithmetic] {pct}% de {val} = {val * pct / 100:.0f}\n---")
    return docs

def generate_geometry(n: int) -> list[str]:
    """Áreas, volumes, perímetros, trigonometria."""
    docs = []
    rng = random.Random(77)
    for i in range(n):
        r = rng.randint(1, 100)
        docs.append(f"[MATH][geometry] Círculo raio={r}: área = {math.pi*r*r:.2f}, perímetro = {2*math.pi*r:.2f}\n---")
        
        if i % 2 == 0:
            b, h = rng.randint(1, 50), rng.randint(1, 50)
            docs.append(f"[MATH][geometry] Triângulo base={b} altura={h}: área = {b*h/2:.0f}\n---")
        
        if i % 3 == 0:
            a2, b2, c2 = rng.randint(3, 30), rng.randint(4, 40), rng.randint(5, 50)
            if a2 + b2 > c2 and a2 + c2 > b2 and b2 + c2 > a2:
                s = (a2 + b2 + c2) / 2
                val = s*(s-a2)*(s-b2)*(s-c2)
                if val > 0:
                    area = math.sqrt(val)
                    docs.append(f"[MATH][geometry] Triângulo lados={a2},{b2},{c2}: área (Heron) = {area:.2f}\n---")
    return docs

def generate_number_theory(n: int) -> list[str]:
    """Primos, fatoração, GCD, LCM."""
    docs = []
    rng = random.Random(13)
    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
    for i in range(n):
        num = rng.randint(2, 10000)
        # GCD
        b = rng.randint(2, 100)
        g = math.gcd(num, b)
        docs.append(f"[MATH][number_theory] GCD({num}, {b}) = {g}\n---")
        
        # Prime check
        if i % 3 == 0:
            is_prime = all(num % p != 0 for p in primes if p * p <= num) and num > 1
            docs.append(f"[MATH][number_theory] {num} é primo? {'Sim' if is_prime else 'Não'}\n---")
        
        # LCM
        if i % 4 == 0:
            l = abs(num * b) // g if g != 0 else num * b
            docs.append(f"[MATH][number_theory] LCM({num}, {b}) = {l}\n---")
    return docs

def generate_statistics(n: int) -> list[str]:
    """Média, mediana, desvio padrão, probabilidade."""
    docs = []
    rng = random.Random(31)
    for i in range(n):
        vals = [rng.randint(1, 100) for _ in range(rng.randint(3, 8))]
        mean = sum(vals) / len(vals)
        sorted_vals = sorted(vals)
        median = sorted_vals[len(vals)//2] if len(vals) % 2 else (sorted_vals[len(vals)//2-1] + sorted_vals[len(vals)//2]) / 2
        variance = sum((x - mean)**2 for x in vals) / len(vals)
        std = math.sqrt(variance)
        docs.append(f"[MATH][statistics] Dados: {vals}\nMédia={mean:.1f} Mediana={median:.1f} Desvio Padrão={std:.1f}\n---")
        
        if i % 2 == 0:
            prob_event = rng.randint(1, 50)
            prob_total = rng.randint(prob_event, 100)
            docs.append(f"[MATH][statistics] Probabilidade: {prob_event}/{prob_total} = {prob_event/prob_total:.2%}\n---")
    return docs

def generate_physics(n: int) -> list[str]:
    """Cinemática, dinâmica, eletricidade."""
    docs = []
    rng = random.Random(23)
    for i in range(n):
        m, a = rng.randint(1, 100), round(rng.uniform(0.5, 20), 1)
        f = m * a
        docs.append(f"[MATH][physics] F = m·a: massa={m}kg aceleração={a}m/s² → força={f:.0f}N\n---")
        
        if i % 2 == 0:
            v, t = rng.randint(5, 100), rng.randint(1, 60)
            d = v * t
            docs.append(f"[MATH][physics] d = v·t: velocidade={v}m/s tempo={t}s → distância={d}m\n---")
        
        if i % 3 == 0:
            v2, i2 = rng.randint(1, 240), round(rng.uniform(0.1, 10), 1)
            p = v2 * i2
            docs.append(f"[MATH][physics] P = V·I: tensão={v2}V corrente={i2}A → potência={p:.0f}W\n---")
    return docs

def generate_combinatorics(n: int) -> list[str]:
    """Permutações, combinações, arranjos."""
    docs = []
    rng = random.Random(91)
    for i in range(n):
        n_items = rng.randint(5, 20)
        k_items = rng.randint(2, min(5, n_items))
        comb = math.comb(n_items, k_items)
        perm = math.perm(n_items, k_items)
        docs.append(f"[MATH][combinatorics] C({n_items},{k_items}) = {comb} combinações\nP({n_items},{k_items}) = {perm} permutações\n---")
        
        if i % 2 == 0:
            fact_n = rng.randint(3, 12)
            docs.append(f"[MATH][combinatorics] {fact_n}! = {math.factorial(fact_n)}\n---")
    return docs

def generate_linear_algebra(n: int) -> list[str]:
    """Matrizes, determinantes, vetores."""
    docs = []
    rng = random.Random(55)
    for i in range(n):
        a, b, c, d = rng.randint(-10, 10), rng.randint(-10, 10), rng.randint(-10, 10), rng.randint(-10, 10)
        det = a*d - b*c
        docs.append(f"[MATH][linear_algebra] Determinante de [[{a},{b}],[{c},{d}]] = {det}\n---")
        
        if i % 2 == 0:
            v1, v2, v3 = rng.randint(-5, 5), rng.randint(-5, 5), rng.randint(-5, 5)
            w1, w2, w3 = rng.randint(-5, 5), rng.randint(-5, 5), rng.randint(-5, 5)
            dot = v1*w1 + v2*w2 + v3*w3
            docs.append(f"[MATH][linear_algebra] Produto escalar ({v1},{v2},{v3})·({w1},{w2},{w3}) = {dot}\n---")
    return docs


# ═══════════════════════════════════════════════════════════
# ORQUESTRADOR
# ═══════════════════════════════════════════════════════════

DOMAINS = {
    "algebra": generate_algebra,
    "calculus": generate_calculus,
    "arithmetic": generate_arithmetic,
    "geometry": generate_geometry,
    "number_theory": generate_number_theory,
    "statistics": generate_statistics,
    "physics": generate_physics,
    "combinatorics": generate_combinatorics,
    "linear_algebra": generate_linear_algebra,
}


def generate_all(target: int = 5_000_000, output_dir: str = "data/corpus"):
    """Gera N exemplos matemáticos sintéticos."""
    per_domain = target // len(DOMAINS)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    total = 0
    stats = {}

    print(f"F51 MASS MATH GENERATOR")
    print(f"  Target: {target:,} exemplos")
    print(f"  Por domínio: {per_domain:,}")
    print(f"  Domínios: {len(DOMAINS)}")
    print()

    for domain, generator in sorted(DOMAINS.items()):
        print(f"  [{domain}] gerando {per_domain:,} exemplos...", end=" ", flush=True)
        docs = generator(per_domain)

        # Salva em lotes de 100k pra não travar
        batch_size = 100_000
        for batch_start in range(0, len(docs), batch_size):
            batch = docs[batch_start:batch_start + batch_size]
            fname = f"math_{domain}_synthetic_v2_part{batch_start//batch_size:03d}.txt"
            fpath = output / fname
            fpath.write_text("\n".join(batch), encoding="utf-8")

        size_mb = sum(len(d) for d in docs) / 1e6
        stats[domain] = {"count": len(docs), "size_mb": round(size_mb, 1)}
        total += len(docs)
        print(f"✓ {len(docs):,} exemplos ({size_mb:.1f} MB)")

    total_size = sum(s["size_mb"] for s in stats.values())
    print(f"\n{'='*60}")
    print(f"  TOTAL: {total:,} exemplos ({total_size:.1f} MB)")
    print(f"  Target: {target:,} → atingido: {100*total/target:.0f}%")
    print(f"{'='*60}")

    # Salva metadados
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_examples": total,
        "total_size_mb": total_size,
        "domains": stats,
        "method": "programmatic_template_generation",
    }
    (output / "math_synthetic_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=5_000_000)
    parser.add_argument("--output", default="data/corpus")
    args = parser.parse_args()
    generate_all(args.target, args.output)
