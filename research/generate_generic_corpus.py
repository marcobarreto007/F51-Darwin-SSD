#!/usr/bin/env python
"""
F51 Generic Document Generator — 2 Milhões de Documentos
=========================================================

Gera documentos sintéticos diversos para combater overfitting.
Cada doc tem 100-500 tokens. 2M docs ≈ 400M-1B tokens.

Domínios cobertos:
  - Matemática pura e aplicada
  - Ciências naturais (física, química, biologia)
  - Programação e algoritmos
  - História e geografia
  - Filosofia e lógica
  - Literatura e poesia
  - Conversação cotidiana
  - Documentação técnica
  - Fatos aleatórios e trivia
  - Português brasileiro coloquial

Uso:
    python research/generate_generic_corpus.py
    python research/generate_generic_corpus.py --docs 2000000 --workers 8
"""

import sys
import os
import random
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

ROOT = Path(__file__).resolve().parents[1]

# ═══════════════════════════════════════════════════════════
# DOMAIN GENERATORS — cada um gera texto de um domínio
# ═══════════════════════════════════════════════════════════

# ── Matemática ──
MATH_TOPICS = [
    "álgebra linear", "cálculo diferencial", "teoria dos números",
    "geometria analítica", "combinatória", "probabilidade",
    "estatística inferencial", "topologia", "análise real",
    "equações diferenciais", "álgebra abstrata", "teoria dos grupos",
    "anéis e corpos", "espaços vetoriais", "transformações lineares",
    "autovalores e autovetores", "séries de Taylor", "integrais",
    "limites e continuidade", "derivadas parciais", "otimização",
    "teoria dos grafos", "criptografia", "teoria da informação",
]

MATH_TEMPLATES = [
    "Seja {topic} uma área fundamental da matemática. Considere a definição formal: {termo} é definido como {definicao}. Esta noção generaliza conceitos elementares e permite tratar problemas de forma abstrata e rigorosa. A demonstração segue por indução sobre a dimensão do espaço. O caso base é trivial. Para o passo indutivo, aplicamos a hipótese e simplificamos usando as propriedades algébricas do operador. O teorema resultante estabelece uma conexão profunda entre estruturas aparentemente distintas.",

    "Em {topic}, um resultado clássico afirma que {termo} satisfaz a equação fundamental: f(x) = integral de g(t) dt de 0 a x. Esta relação é consequência direta do teorema fundamental do cálculo e das propriedades de continuidade uniforme. Para verificar, considere uma partição do intervalo e aplique o teorema do valor médio. A convergência é garantida pela hipótese de Lipschitz sobre a função integranda. Este resultado tem aplicações em física matemática, engenharia e ciência da computação teórica.",

    "O estudo de {topic} revela padrões surpreendentes. Por exemplo, a sequência definida por recorrência a_n = f(a sub n-1) exibe comportamento caótico para certos valores do parâmetro. A análise do expoente de Lyapunov confirma a sensibilidade às condições iniciais. Este fenômeno está relacionado ao teorema de Sharkovsky sobre órbitas periódicas. A demonstração utiliza argumentos de ponto fixo e o teorema da função implícita em espaços de Banach.",

    "Problema: Dado {termo}, determine o valor ótimo de x que minimiza a função custo J(x) = norma de Ax - b ao quadrado mais lambda vezes norma L1 de x. A solução envolve o operador proximal e o algoritmo de gradiente descendente. A convergência é linear para funções fortemente convexas. Em dimensão finita, o método dos mínimos quadrados com regularização L1 produz soluções esparsas, fundamentais em compressed sensing e seleção de variáveis.",

    "A beleza da {topic} está na sua universalidade. Os mesmos princípios que governam {termo} aparecem em contextos completamente diferentes: da teoria dos números à mecânica quântica, da criptografia à biologia molecular. Esta unidade subjacente da matemática é o que a torna a linguagem fundamental da natureza. Como disse Galileu, o livro da natureza está escrito em caracteres matemáticos.",
]

MATH_TERMS = [
    "o operador laplaciano", "a matriz jacobiana", "o grupo fundamental",
    "a sequência espectral", "o fibrado tangente", "a cohomologia de de Rham",
    "o espectro do operador", "a medida de Haar", "o teorema de Pitágoras",
    "a desigualdade de Cauchy-Schwarz", "o teorema espectral",
    "a decomposição em valores singulares", "o método de Newton",
    "a transformada de Fourier", "a função zeta de Riemann",
    "o teorema de Green", "a fórmula de Stirling",
    "o princípio do máximo", "a equação de Euler-Lagrange",
]

MATH_DEFS = [
    "o limite uniforme de uma sequência de funções contínuas",
    "a generalização natural do conceito de distância em espaços abstratos",
    "o invariante topológico que conta os buracos de cada dimensão",
    "a classe de equivalência de caminhos sob homotopia",
    "a melhor aproximação linear de uma função num ponto",
    "a taxa de variação instantânea de uma grandeza",
    "o valor que uma função assume quando o argumento tende a um ponto",
    "a soma infinita dos termos de uma progressão bem comportada",
]

def generate_math_doc() -> str:
    template = random.choice(MATH_TEMPLATES)
    topic = random.choice(MATH_TOPICS)
    termo = random.choice(MATH_TERMS)
    definicao = random.choice(MATH_DEFS)
    text = template.format(topic=topic, termo=termo, definicao=definicao)
    # Add random equation
    if random.random() < 0.3:
        eqs = [
            "\n\nEquação: derivada parcial de u no tempo = alpha vezes laplaciano de u mais f(x,t)",
            "\n\nFórmula: e elevado a i pi mais 1 igual a 0",
            "\n\nTeorema: integral de contorno de f(z) dz = 2 pi i vezes soma dos resíduos",
            "\n\nIdentidade: determinante de AB = determinante de A vezes determinante de B",
            "\n\nDesigualdade: norma de x mais y menor ou igual a norma de x mais norma de y",
        ]
        text += random.choice(eqs)
    return text

# ── Ciências ──
SCIENCE_TOPICS = [
    "mecânica quântica", "relatividade geral", "termodinâmica",
    "eletromagnetismo", "física de partículas", "astrofísica",
    "química orgânica", "bioquímica", "genética molecular",
    "evolução darwiniana", "ecologia de populações", "neurociência",
    "geologia estrutural", "climatologia", "oceanografia",
]

SCIENCE_TEMPLATES = [
    "A {topic} é um campo fascinante da ciência moderna. Experimentos recentes demonstraram que as interações entre partículas elementares obedecem a princípios de simetria profundos. O modelo padrão da física de partículas descreve três das quatro forças fundamentais: eletromagnética, nuclear forte e nuclear fraca. A gravidade, descrita pela relatividade geral, permanece como o grande desafio de unificação. Pesquisadores do CERN e outros laboratórios continuam investigando fenômenos além do modelo padrão.",
    
    "No estudo da {topic}, observamos que sistemas complexos emergem de regras simples. As formigas, por exemplo, criam colônias sofisticadas sem nenhum controle central. O cérebro humano, com bilhões de neurônios, produz consciência. Este fenômeno de emergência é ubíquo na natureza e sugere que a complexidade não requer design inteligente, mas sim iterações repetidas de processos simples sob restrições ambientais.",
    
    "A {topic} moderna utiliza métodos computacionais avançados. Simulações de dinâmica molecular permitem estudar o enovelamento de proteínas. Modelos climáticos acoplam atmosfera, oceanos e biosfera. Telescópios como o James Webb revelam galáxias formadas logo após o Big Bang. A ciência avança pela combinação de teoria, experimento e computação — o tripé do método científico contemporâneo.",
    
    "Um princípio fundamental da {topic} é a conservação de energia. Em qualquer sistema isolado, a energia total permanece constante — ela apenas muda de forma. Esta lei, descoberta no século XIX por Joule, Mayer e Helmholtz, unificou fenômenos antes considerados distintos: calor, movimento, eletricidade e reações químicas são manifestações da mesma grandeza física fundamental.",
]

def generate_science_doc() -> str:
    template = random.choice(SCIENCE_TEMPLATES)
    topic = random.choice(SCIENCE_TOPICS)
    text = template.format(topic=topic)
    if random.random() < 0.2:
        facts = [
            "\n\nFato: A velocidade da luz no vácuo é exatamente 299.792.458 m/s.",
            "\n\nFato: O DNA humano tem aproximadamente 3 bilhões de pares de bases.",
            "\n\nFato: A Terra orbita o Sol a uma velocidade média de 107.000 km/h.",
            "\n\nFato: Existem mais estrelas no universo que grãos de areia na Terra.",
        ]
        text += random.choice(facts)
    return text

# ── Programação ──
CODE_TOPICS = [
    "Python", "Rust", "algoritmos de ordenação", "estruturas de dados",
    "programação funcional", "orientação a objetos", "APIs REST",
    "bancos de dados", "machine learning", "redes neurais",
    "processamento de linguagem natural", "visão computacional",
    "sistemas distribuídos", "containers Docker", "Git e versionamento",
]

CODE_TEMPLATES = [
    "Ao desenvolver software em {topic}, é importante seguir boas práticas de engenharia. O código deve ser legível, bem documentado e testado. Utilize nomes descritivos para variáveis e funções. Evite duplicação — cada conhecimento deve ter uma representação única e autoritativa no sistema. Prefira composição a herança. Mantenha as funções pequenas e com responsabilidade única. Escreva testes antes do código de produção quando possível.",
    
    "A implementação eficiente de {topic} requer compreensão profunda das estruturas subjacentes. A complexidade assintótica — notação Big O — é a ferramenta principal para analisar algoritmos. Um algoritmo O(n log n) é aceitável para a maioria das aplicações. Algoritmos O(n²) só funcionam para entradas pequenas. Busque sempre a menor complexidade possível sem sacrificar a clareza do código. Otimização prematura é a raiz de todos os males.",
    
    "Em {topic}, o tratamento de erros é crítico. Use exceções para condições excepcionais, não para controle de fluxo normal. Valide entradas nas fronteiras do sistema. Logs detalhados são essenciais para debugging em produção. Implemente estratégias de retry com backoff exponencial para operações de rede. Sistemas distribuídos devem ser projetados para falhar graciosamente — a queda de um nó não pode derrubar o sistema inteiro.",
]

def generate_code_doc() -> str:
    template = random.choice(CODE_TEMPLATES)
    topic = random.choice(CODE_TOPICS)
    text = template.format(topic=topic)
    if random.random() < 0.3:
        snippets = [
            "\n\n```python\ndef fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```",
            "\n\n```python\nfrom collections import defaultdict\n\ngraph = defaultdict(list)\nfor u, v in edges:\n    graph[u].append(v)\n```",
            "\n\n```python\ntry:\n    result = risky_operation()\nexcept ValueError as e:\n    logger.error('Invalid value: ' + str(e))\n    raise\n```",
        ]
        text += random.choice(snippets)
    return text

# ── História / Geografia ──
HISTORY_FACTS = [
    ("Império Romano", "dominou o Mediterrâneo por mais de 500 anos, deixando um legado de direito, engenharia e língua que perdura até hoje."),
    ("Revolução Industrial", "começou na Inglaterra no século XVIII, transformando a produção artesanal em fabril e alterando profundamente a sociedade."),
    ("Renascimento", "foi um movimento cultural dos séculos XIV-XVII que resgatou valores clássicos e impulsionou as artes e as ciências."),
    ("Segunda Guerra Mundial", "envolveu a maioria das nações do mundo entre 1939 e 1945, resultando em dezenas de milhões de mortes."),
    ("Império Inca", "construiu uma rede de estradas de mais de 30.000 km pelos Andes, conectando um território que ia da Colômbia ao Chile."),
    ("Revolução Francesa", "derrubou a monarquia absolutista em 1789 e estabeleceu princípios de liberdade, igualdade e fraternidade."),
    ("Dinastia Ming", "governou a China de 1368 a 1644, período de esplendor cultural e explorações marítimas lideradas pelo almirante Zheng He."),
    ("Guerra Fria", "foi o conflito ideológico entre EUA e URSS que dividiu o mundo em dois blocos de 1947 a 1991."),
]

GEOGRAPHY_FACTS = [
    "O Rio Amazonas é o maior rio do mundo em volume de água, descarregando 209.000 m³/s no Oceano Atlântico.",
    "O Monte Everest, na fronteira entre Nepal e China, é o ponto mais alto da Terra com 8.848 metros de altitude.",
    "O Deserto do Saara cobre 9,2 milhões de km², sendo o maior deserto quente do mundo.",
    "A Indonésia é o maior arquipélago do mundo, com mais de 17.000 ilhas.",
    "O Lago Baikal, na Sibéria, contém 20% da água doce não congelada do mundo.",
    "A Austrália é o único país que é também um continente inteiro.",
    "O Japão é formado por mais de 6.800 ilhas, sendo as quatro principais Honshu, Hokkaido, Kyushu e Shikoku.",
    "A Islândia é o único país do mundo sem mosquitos.",
]

def generate_history_doc() -> str:
    styles = [
        "O {name} {fact}. Este evento marcou profundamente a trajetória da humanidade. Historiadores debatem até hoje suas causas e consequências de longo prazo. O que é indiscutível é que o mundo nunca mais foi o mesmo depois disso.",
        "Poucos eventos históricos tiveram impacto tão duradouro quanto {name}, que {fact}. As repercussões deste período reverberam até os dias atuais, influenciando instituições políticas, fronteiras nacionais e identidades culturais em escala global.",
        "A história registra que {name} {fact}. Este capítulo da experiência humana ilustra como forças sociais, econômicas e tecnológicas convergem para produzir transformações radicais na organização da sociedade.",
    ]
    event = random.choice(HISTORY_FACTS)
    style = random.choice(styles)
    text = style.format(name=event[0], fact=event[1])
    return text

def generate_geography_doc() -> str:
    fact = random.choice(GEOGRAPHY_FACTS)
    extras = [
        "A geografia física do nosso planeta é resultado de bilhões de anos de processos geológicos.",
        "Compreender a distribuição dos recursos naturais é essencial para a geopolítica contemporânea.",
        "As características geográficas de uma região influenciam profundamente sua cultura e economia.",
    ]
    return f"{fact} {random.choice(extras)}"

# ── Filosofia ──
PHILOSOPHY_TOPICS = [
    "ética", "epistemologia", "metafísica", "lógica", "filosofia da mente",
    "filosofia da ciência", "filosofia política", "estética",
    "existencialismo", "estoicismo", "utilitarismo", "racionalismo",
    "empirismo", "idealismo", "materialismo", "fenomenologia",
]

PHILOSOPHY_TEMPLATES = [
    "A {topic} investiga questões fundamentais sobre a natureza da realidade e do conhecimento. Desde os pré-socráticos até os filósofos contemporâneos, esta disciplina tem desafiado pressupostos e expandido os horizontes do pensamento humano. A pergunta central permanece: o que podemos conhecer com certeza? As respostas variam do ceticismo radical ao realismo científico, passando por posições intermediárias que reconhecem os limites do conhecimento humano sem renunciar à busca pela verdade.",
    
    "No campo da {topic}, o debate entre racionalistas e empiristas moldou a filosofia moderna. Descartes afirmava que a razão pura podia alcançar verdades fundamentais. Locke respondia que toda ideia deriva da experiência sensorial. Kant propôs uma síntese: a estrutura da mente organiza os dados dos sentidos, produzindo conhecimento que é ao mesmo tempo empírico na origem e racional na forma. Esta visão influenciou profundamente a psicologia cognitiva e a neurociência contemporâneas.",
    
    "A {topic} nos convida a examinar nossos valores mais profundos. O que é uma vida boa? Como devemos tratar os outros? Estas perguntas não têm respostas simples, mas o processo de reflexão já é transformador. Sócrates dizia que uma vida não examinada não vale a pena ser vivida. O exercício filosófico não oferece conforto fácil, mas proporciona clareza e autonomia intelectual — ferramentas essenciais para navegar um mundo complexo e em rápida transformação.",
]

def generate_philosophy_doc() -> str:
    return random.choice(PHILOSOPHY_TEMPLATES).format(topic=random.choice(PHILOSOPHY_TOPICS))

# ── Conversação / Português coloquial ──
DIALOGUE_TEMPLATES = [
    "— Bom dia! Como você está hoje?\n— Estou bem, obrigado! E você, como vai?\n— Vou indo. O trânsito estava horrível hoje de manhã.\n— Nem me fale. A cidade está cada vez mais congestionada. Precisamos de mais transporte público de qualidade.\n— Concordo totalmente. Metrô, ciclovias, ônibus elétricos... tantas opções melhores que carro particular.\n— Pois é. Mas enquanto não melhorar, a gente vai se virando. Quer um café?\n— Aceito! Café é essencial para começar bem o dia.",
    
    "— Você viu o jogo ontem?\n— Vi sim! Que partida emocionante. Achei que iam perder no segundo tempo.\n— Também achei. Mas a virada foi épica. O terceiro gol foi de placa.\n— O goleiro deles falhou feio. Mas fazer o quê, faz parte.\n— Sem dúvida. Futebol é isso: imprevisível e apaixonante.\n— E o campeonato ainda está completamente aberto. Qualquer um dos quatro primeiros pode levar.\n— Vai ser emocionante até a última rodada.",
    
    "— O que você acha dessa história de inteligência artificial?\n— Olha, é uma faca de dois gumes. Tem um potencial incrível para o bem: medicina, educação, ciência.\n— Mas também tem os riscos: desemprego, desinformação, viés algorítmico.\n— Exatamente. Acho que o segredo está na regulação inteligente. Nem proibir, nem liberar totalmente.\n— Concordo. Precisamos de transparência nos algoritmos e responsabilização pelos resultados.\n— E educação digital para todo mundo. As pessoas precisam entender o básico para não serem enganadas.\n— Isso é fundamental. Alfabetização digital deveria ser matéria obrigatória nas escolas.",
]

PORTUGUESE_TEXTS = [
    "O Brasil é um país de dimensões continentais, com uma diversidade cultural imensa. De norte a sul, encontramos tradições, sotaques e culinárias completamente diferentes. O acarajé baiano, o churrasco gaúcho, o tacacá paraense, o pão de queijo mineiro — cada região tem suas iguarias. A música também reflete essa diversidade: samba, forró, frevo, maracatu, sertanejo, funk. O português brasileiro é uma língua viva, em constante evolução, que absorve influências indígenas, africanas e de imigrantes de todo o mundo.",
    
    "A culinária brasileira é um patrimônio cultural. A feijoada, originada nas senzalas, tornou-se prato nacional. A caipirinha, com cachaça, limão e açúcar, conquistou o mundo. O açaí paraense virou moda internacional. Mas a verdadeira riqueza está nos ingredientes: mandioca, milho, amendoim, frutas tropicais como cupuaçu, jabuticaba, cajá. Cada bioma — Amazônia, Cerrado, Caatinga, Mata Atlântica, Pantanal, Pampa — oferece produtos únicos que definem a gastronomia regional.",
    
    "O futebol é mais que um esporte no Brasil — é parte da identidade nacional. Das peladas de rua aos estádios lotados, a paixão pela bola une pessoas de todas as classes sociais. Grandes craques como Pelé, Garrincha, Zico, Romário, Ronaldo e Ronaldinho encantaram o mundo com seu talento. A seleção brasileira é a única a ter participado de todas as Copas do Mundo e a maior vencedora, com cinco títulos. O Maracanã, templo do futebol, já recebeu mais de 200 mil pessoas em uma única partida.",
]

# ── Trivia / Fatos aleatórios ──
TRIVIA_FACTS = [
    "O coração de um camarão está localizado em sua cabeça.",
    "As bananas são tecnicamente bagas, mas os morangos não são.",
    "O mel nunca estraga. Já foi encontrado mel comestível em tumbas egípcias de 3.000 anos.",
    "Os golfinhos dormem com um olho aberto — metade do cérebro descansa enquanto a outra vigia.",
    "O olho humano pode distinguir aproximadamente 10 milhões de cores diferentes.",
    "A Torre Eiffel cresce até 15 cm no verão devido à expansão térmica do ferro.",
    "Os coalas têm impressões digitais quase idênticas às humanas.",
    "O sistema solar tem 8 planetas reconhecidos, mas estima-se que existam bilhões de exoplanetas na Via Láctea.",
    "A água quente congela mais rápido que a água fria em certas condições — é o Efeito Mpemba.",
    "As formigas não dormem. Em vez disso, tiram centenas de pequenos cochilos de 1 minuto.",
    "O som viaja 4 vezes mais rápido na água que no ar.",
    "Um raio atinge a Terra aproximadamente 8 milhões de vezes por dia.",
    "O cérebro humano gera cerca de 20 watts de potência elétrica — suficiente para acender uma lâmpada LED.",
    "A Grande Muralha da China não é visível da Lua a olho nu, ao contrário do mito popular.",
    "Existem mais átomos em um copo d'água que copos d'água em todos os oceanos da Terra.",
]

# ═══════════════════════════════════════════════════════════
# MAIN GENERATOR
# ═══════════════════════════════════════════════════════════

GENERATORS = [
    ("math", generate_math_doc, 0.20),        # 20% matemática
    ("science", generate_science_doc, 0.15),   # 15% ciências
    ("code", generate_code_doc, 0.10),          # 10% programação
    ("philosophy", generate_philosophy_doc, 0.10), # 10% filosofia
    ("history", generate_history_doc, 0.05),    # 5% história
    ("geography", generate_geography_doc, 0.05), # 5% geografia
    ("dialogue", lambda: random.choice(DIALOGUE_TEMPLATES), 0.15), # 15% conversas
    ("portuguese", lambda: random.choice(PORTUGUESE_TEXTS), 0.10), # 10% português
    ("trivia", lambda: "Fato interessante: " + random.choice(TRIVIA_FACTS), 0.10), # 10% curiosidades
]

def generate_document(doc_id: int) -> str:
    """Gera um documento aleatório usando os geradores ponderados."""
    names = [g[0] for g in GENERATORS]
    weights = [g[2] for g in GENERATORS]
    generators = [g[1] for g in GENERATORS]
    
    gen = random.choices(generators, weights=weights, k=1)[0]
    text = gen()
    
    # Garantir tamanho razoável
    words = text.split()
    target_words = random.randint(40, 200)
    
    while len(words) < target_words:
        # Adiciona mais texto se muito curto
        extra_gen = random.choices(generators, weights=weights, k=1)[0]
        extra = extra_gen()
        words.extend(extra.split())
    
    # Truncar se muito longo
    if len(words) > 300:
        words = words[:random.randint(180, 300)]
    
    return " ".join(words)

def generate_chunk(args):
    """Gera um chunk de documentos (para paralelismo)."""
    chunk_id, num_docs, output_dir = args
    output_path = Path(output_dir) / f"generic_chunk_{chunk_id:06d}.txt"
    
    docs = []
    start_id = chunk_id * num_docs
    for i in range(num_docs):
        doc = generate_document(start_id + i)
        docs.append(doc)
    
    output_path.write_text("\n\n---\n\n".join(docs), encoding="utf-8")
    return chunk_id, len(docs), output_path.stat().st_size

def main():
    parser = argparse.ArgumentParser(description="F51 Generic Corpus Generator")
    parser.add_argument("--docs", type=int, default=2_000_000, help="Número de documentos")
    parser.add_argument("--workers", type=int, default=8, help="Threads paralelas")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Docs por chunk")
    parser.add_argument("--output", default="data/corpus/generic", help="Diretório de saída")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_docs = args.docs
    chunk_size = args.chunk_size
    num_chunks = (total_docs + chunk_size - 1) // chunk_size
    
    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  F51 GENERIC CORPUS GENERATOR               ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║  Documentos: {total_docs:,}                           ║")
    print(f"║  Chunks:     {num_chunks:,} (${chunk_size:,} docs cada)       ║")
    print(f"║  Workers:    {args.workers}                               ║")
    print(f"║  Output:     {output_dir}  ║")
    print(f"╚══════════════════════════════════════════════╝")
    print()
    
    t0 = time.time()
    total_size = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for chunk_id in range(num_chunks):
            n = min(chunk_size, total_docs - chunk_id * chunk_size)
            futures.append(executor.submit(generate_chunk, (chunk_id, n, str(output_dir))))
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            chunk_id, n_docs, size = future.result()
            total_size += size
            
            elapsed = time.time() - t0
            docs_done = completed * chunk_size
            rate = docs_done / elapsed if elapsed > 0 else 0
            
            bar_len = 30
            pct = completed / num_chunks
            bar = "█" * int(pct * bar_len) + "░" * (bar_len - int(pct * bar_len))
            
            print(f"\r[{bar}] {completed}/{num_chunks} chunks | "
                  f"{docs_done:,} docs | {rate:,.0f} governance/docs/s | "
                  f"{total_size/1e6:.0f} MB", end="", flush=True)
    
    elapsed = time.time() - t0
    print(f"\n\n✅ {total_docs:,} documentos gerados em {elapsed:.0f}s "
          f"({total_governance/docs/elapsed:,.0f} governance/docs/s)")
    print(f"   Tamanho total: {total_size/1e6:.0f} MB ({total_size/1e9:.2f} GB)")
    print(f"   Diretório: {output_dir}")
    
    # Salvar metadados
    meta = {
        "total_docs": total_docs,
        "size_bytes": total_size,
        "elapsed_sec": elapsed,
        "docs_per_sec": total_docs / elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "domains": {g[0]: g[2] for g in GENERATORS},
    }
    (output_dir / "generic_meta.json").write_text(json.dumps(meta, indent=2))
    
    # Resumo
    print(f"\n📊 Distribuição de domínios:")
    for name, _, weight in GENERATORS:
        bar = "█" * int(weight * 50)
        print(f"   {name:<15s} {weight:.0%}  {bar}")

if __name__ == "__main__":
    main()
