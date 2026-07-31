#!/usr/bin/env python
"""F51 Final Corpus Push — 10 domínios, 3 GB cada, 30 GB total"""
import sys, random, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

DOMAINS = {
    "logic": ("Lógica & Pensamento Crítico", [
        "Silogismos: Todo A é B, Todo B é C, logo Todo A é C. Falácias formais: afirmação do consequente, negação do antecedente. Falácias informais: ad hominem, strawman, falsa dicotomia, apelo à autoridade, slippery slope, petição de princípio.",
        "Lógica proposicional: conectivos (∧, ∨, →, ¬, ↔), tabelas-verdade, tautologias, contradições. Leis de De Morgan: ¬(P∧Q)≡¬P∨¬Q. Modus ponens, modus tollens, silogismo disjuntivo, dilema construtivo.",
        "Lógica de predicados: quantificadores ∀ (para todo) e ∃ (existe). Domínio de discurso. Negação de quantificadores: ¬∀x P(x) ≡ ∃x ¬P(x). Skolemização. Resolução para lógica de primeira ordem.",
        "Lógica modal: necessidade (□) e possibilidade (◇). Mundos possíveis (Kripke). Sistemas: K, T, S4, S5. Lógica epistêmica, deôntica (obrigação/permissão), temporal (sempre, eventualmente).",
        "Teoria da argumentação: Toulmin (dado, garantia, backing, qualificador, refutação, conclusão). Perelman (nova retórica, auditório universal). Van Eemeren (pragma-dialética, estágios da discussão crítica).",
        "Vieses cognitivos na lógica: viés de confirmação, viés de disponibilidade, ancoragem, framing effect, hindsight bias, Dunning-Kruger. Como a lógica formal combate vieses.",
    ]),
    "law": ("Direito & Sistemas Legais", [
        "Common Law vs Civil Law: precedente (stare decisis) vs código. Ratio decidendi vs obiter dictum. Hierarquia de tribunais. Distinguishing e overruling. Sistemas mistos: Escócia, Louisiana, Quebec, África do Sul.",
        "Direito Constitucional: separação de poderes (Montesquieu), checks and balances, direitos fundamentais. Controle de constitucionalidade: difuso (EUA) vs concentrado (Europa). Judicial review. Marbury v. Madison (1803).",
        "Direito Penal: actus reus + mens rea. Dolo vs culpa. Excludentes: legítima defesa, estado de necessidade, estrito cumprimento do dever. Teorias da pena: retribuição, prevenção geral/especial, ressocialização.",
        "Direito Contratual: formação (oferta, aceitação, consideração), vícios (erro, dolo, coação), inadimplemento, resolução. Boa-fé objetiva. Teoria da imprevisão. Contratos relacionais (Macneil). Análise econômica do direito contratual.",
        "Direito Internacional: fontes (tratados, costume, princípios gerais, doutrina, jurisprudência). Soberania, não-intervenção, autodeterminação. Jus ad bellum vs jus in bello. Tribunal Penal Internacional (Estatuto de Roma).",
        "Filosofia do Direito: jusnaturalismo (Direito deriva da moral/Deus), positivismo (Direito é norma posta, Kelsen), realismo jurídico (Direito é o que juízes decidem), critical legal studies, direito como integridade (Dworkin).",
    ]),
    "engineering": ("Engenharia & Tecnologia", [
        "Engenharia Civil: análise estrutural (treliças, vigas, pórticos), mecânica dos solos (fundações, taludes), hidráulica (canais abertos, tubulações), concreto armado (flexão, cisalhamento), aço estrutural.",
        "Engenharia Elétrica: circuitos (Leis de Kirchhoff, Thevenin, Norton), eletromagnetismo aplicado, sistemas de potência (geração, transmissão, distribuição), máquinas elétricas (motores, geradores, transformadores).",
        "Engenharia Mecânica: termodinâmica (ciclos Rankine, Brayton, Otto, Diesel, refrigeração), mecânica dos fluidos (Navier-Stokes, Bernoulli, camada limite), transferência de calor (condução, convecção, radiação).",
        "Engenharia de Software: arquitetura (monolítica, microsserviços, event-driven, CQRS, hexagonal), padrões de design (GoF: singleton, factory, observer, strategy), DevOps (CI/CD, Docker, Kubernetes).",
        "Engenharia Química: balanços de massa e energia, operações unitárias (destilação, absorção, extração, secagem), reatores químicos (batelada, CSTR, PFR), catálise, controle de processos (PID, MPC).",
        "Engenharia Aeroespacial: aerodinâmica (arrasto, sustentação, escoamento compressível), propulsão (foguete: Tsiolkovsky, jato: ciclo Brayton), estruturas aeroespaciais, controle de atitude, órbitas (Kepler, Hohmann).",
    ]),
    "medicine": ("Medicina & Saúde", [
        "Fisiologia Humana: sistema cardiovascular (ciclo cardíaco, pressão arterial, ECG), respiratório (ventilação, troca gasosa, controle), renal (filtração glomerular, clearance, balanço hídrico), endócrino (eixos hormonais).",
        "Farmacologia: farmacocinética (absorção, distribuição, metabolismo, excreção), farmacodinâmica (receptores, agonistas/antagonistas, curva dose-resposta), classes terapêuticas, interações medicamentosas.",
        "Patologia: inflamação aguda/crônica, necrose vs apoptose, neoplasia (benigna vs maligna, metástase), carcinogênese (iniciação, promoção, progressão), doenças autoimunes, doenças infecciosas.",
        "Imunologia: imunidade inata (barreiras, fagócitos, NK, complemento) vs adaptativa (linfócitos T/B, anticorpos, memória). MHC I/II, apresentação de antígenos. Hipersensibilidade I-IV. Vacinas.",
        "Neurociência clínica: AVC isquêmico/hemorrágico, Alzheimer (placas amiloides, tau), Parkinson (degeneração dopaminérgica), esclerose múltipla (desmielinização), epilepsia, enxaqueca, neuropatias.",
        "Epidemiologia: medidas (incidência, prevalência, risco relativo, odds ratio), desenhos de estudo (coorte, caso-controle, ensaio clínico randomizado, transversal), vieses (seleção, informação, confundimento).",
    ]),
    "psychology": ("Psicologia & Comportamento", [
        "Psicologia Cognitiva: atenção (seletiva, dividida, sustentada), percepção (Gestalt, bottom-up/top-down), memória (sensorial, curto prazo/working, longo prazo: episódica, semântica, procedural), tomada de decisão (heurísticas, vieses).",
        "Psicologia Social: conformidade (Asch), obediência (Milgram), papéis sociais (Zimbardo), dissonância cognitiva (Festinger), atribuição (fundamental attribution error), atitudes, estereótipos, preconceito.",
        "Psicologia do Desenvolvimento: Piaget (sensório-motor, pré-operatório, operatório concreto, formal), Vygotsky (zona de desenvolvimento proximal), Bowlby (apego: seguro, ansioso, evitativo), Erikson (8 estágios psicossociais).",
        "Psicologia Clínica: TCC (Beck: pensamentos automáticos, distorções cognitivas, reestruturação), psicanálise (Freud: id/ego/superego, mecanismos de defesa), humanista (Rogers: congruência, empatia, consideração positiva incondicional).",
        "Neurociência Cognitiva: redes neurais (default mode, saliência, executiva central), plasticidade sináptica (LTP, LTD), consciência (teorias: espaço de trabalho global, informação integrada, processamento preditivo).",
        "Psicologia Organizacional: motivação (Maslow, Herzberg, Deci & Ryan: autodeterminação), liderança (transformacional, transacional, situacional), cultura organizacional (Schein), seleção, avaliação de desempenho.",
    ]),
    "military": ("Estratégia & História Militar", [
        "Sun Tzu — A Arte da Guerra: 'A suprema arte da guerra é derrotar o inimigo sem lutar.' Engano, terreno, espionagem, moral. Influência em estratégia militar, negócios e esportes até hoje.",
        "Clausewitz — Da Guerra: 'A guerra é a continuação da política por outros meios.' Névoa da guerra, fricção, centro de gravidade, trindade (governo, exército, povo). Guerra absoluta vs guerra real.",
        "Batalhas decisivas: Canas (216 a.C., Aníbal, duplo envolvimento), Waterloo (1815, Napoleão, Wellington), Stalingrado (1942-43, virada da 2ª Guerra), Midway (1942, porta-aviões, código quebrado), D-Day (1944, Overlord).",
        "Estratégia nuclear: MAD (destruição mútua assegurada), tríade nuclear, first strike vs second strike. Crise dos mísseis de Cuba (1962). Doutrinas: massive retaliation, flexible response, counterforce vs countervalue.",
        "Guerra assimétrica: guerrilha (Mao: 'o guerrilheiro é o peixe, o povo é a água'), contrainsurgência (corações e mentes, clear-hold-build), terrorismo, cyberwarfare, proxy wars, PMCs (Blackwater/Wagner).",
        "Geopolítica: Mackinder (heartland theory), Spykman (rimland), Mahan (poder naval). Containment (Kennan), détente, rollback. Século XXI: China (Belt and Road, mar do Sul), Rússia, Oriente Médio, Ártico.",
    ]),
    "chemistry": ("Química & Materiais", [
        "Química Geral: estrutura atômica (Bohr, quântico, orbitais s/p/d/f), tabela periódica (tendências: eletronegatividade, raio atômico, energia de ionização), ligações (iônica, covalente, metálica), estequiometria.",
        "Química Orgânica: grupos funcionais (álcool, aldeído, cetona, ácido carboxílico, éster, amina, amida), mecanismos (SN1, SN2, E1, E2), aromáticos (benzeno, substituição eletrofílica), polímeros.",
        "Físico-Química: termodinâmica química (entalpia, entropia, energia livre de Gibbs), cinética química (ordem de reação, equação de Arrhenius, catálise), eletroquímica (Nernst, pilhas, eletrólise).",
        "Química Inorgânica: compostos de coordenação (número de coordenação, isomeria, teoria do campo cristalino), metais de transição, lantanídeos, actinídeos, química bioinorgânica (hemoglobina, clorofila).",
        "Química Analítica: métodos clássicos (titulação, gravimetria), instrumentais (espectrofotometria UV-Vis, IR, RMN, massa, cromatografia HPLC/GC), validação de métodos (precisão, exatidão, LOD, LOQ).",
        "Ciência dos Materiais: metais (diagrama de fases Fe-C, aços, ligas de alumínio), cerâmicas, polímeros, compósitos. Nanomateriais (fulerenos, nanotubos, grafeno). Semicondutores (banda de valência/condução, dopagem).",
    ]),
    "brasil": ("Brasil — Cultura & Sociedade", [
        "Formação do povo brasileiro: miscigenação (indígena, europeu, africano). Casa-grande & senzala (Gilberto Freyre). O povo brasileiro (Darcy Ribeiro). Raízes do Brasil (Sérgio Buarque de Holanda: homem cordial).",
        "Literatura brasileira: Machado de Assis (Dom Casmurro, Memórias Póstumas, realismo psicológico), Guimarães Rosa (Grande Sertão: Veredas, linguagem reinventada), Clarice Lispector (introspecção, epifania), Carlos Drummond de Andrade (poesia).",
        "Música brasileira: samba (Cartola, Noel Rosa), bossa nova (Tom Jobim, João Gilberto, Garota de Ipanema), MPB (Chico Buarque, Caetano Veloso, Gilberto Gil, Elis Regina), tropicália, forró, funk carioca.",
        "História política: Império (1822-1889), República Velha (1889-1930, café-com-leite), Era Vargas (1930-1945, CLT, industrialização), Ditadura Militar (1964-1985, AI-5, milagre econômico), Nova República (1985-presente).",
        "Economia brasileira: ciclos (pau-brasil, açúcar, ouro, café, borracha). Plano Real (1994, URV, âncora cambial). Commodities, agronegócio, pré-sal. Desafios: desigualdade, produtividade, complexidade tributária.",
        "Cultura regional: Nordeste (sertão, cangaço, cordel, frevo, maracatu), Norte (Amazônia, lendas, carimbó), Centro-Oeste (sertanejo, pantanal), Sudeste (metrópole, diversidade), Sul (imigração europeia, chimarrão, CTG).",
    ]),
    "religion": ("Religiões & Filosofia Comparada", [
        "Cristianismo: Trindade, encarnação, crucificação, ressurreição. Catolicismo (papado, sacramentos, Tradição), Ortodoxia (ícones, theosis), Protestantismo (sola scriptura, sola fide, sola gratia). Agostinho, Tomás de Aquino, Lutero, Calvino.",
        "Islamismo: Cinco pilares (shahada, salat, zakat, sawm, hajj). Alcorão, Hadith, Sunna. Xiismo vs Sunismo. Sufismo (misticismo islâmico: Rumi, Al-Ghazali). Sharia. Idade de Ouro Islâmica (Averróis, Avicena, álgebra).",
        "Judaísmo: Torá (Pentateuco), Talmud, Midrash. Aliança, êxodo, exílio. Ortodoxo, Conservador, Reformista. Cabala (misticismo judaico: Zohar, Árvore da Vida). Contribuições: ética monoteísta, pensamento jurídico.",
        "Budismo: Quatro Nobres Verdades, Caminho Óctuplo. Nirvana, samsara, karma. Theravada (sudeste asiático), Mahayana (China, Japão, Coreia), Vajrayana (Tibet). Zen (meditação, koans). Dalai Lama.",
        "Hinduísmo: Brahman (absoluto), Atman (alma), karma, dharma, moksha (libertação). Trimurti: Brahma (criador), Vishnu (preservador), Shiva (destruidor). Vedas, Upanishads, Bhagavad Gita. Yoga. Sistema de castas.",
        "Filosofia da religião: argumentos para existência de Deus (ontológico, cosmológico, teleológico, moral). Problema do mal (teodiceia: Agostinho, Leibniz, Plantinga). Fé e razão (fideísmo, evidencialismo, pragmatismo). Pascal, Kierkegaard, Tillich.",
    ]),
}

def generate_doc(domain_key):
    domain_name, topics = DOMAINS[domain_key]
    topic = random.choice(topics)
    return f"DOMÍNIO: {domain_name}\n\n{topic}"

def generate_chunk(args):
    chunk_id, size, out_dir, seed, domain_key = args
    random.seed(seed)
    docs = [generate_doc(domain_key) for _ in range(size)]
    p = Path(out_dir) / f"{domain_key}_{chunk_id:06d}.txt"
    p.write_text("\n\n---\n\n".join(docs), encoding="utf-8")
    return chunk_id, size, p.stat().st_size

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs-per", type=int, default=2_000_000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--output", default="data/corpus/final_push")
    args = p.parse_args()

    total = args.docs_per * len(DOMAINS)
    print(f"╔═══════════════════════════════════════╗")
    print(f"║  FINAL PUSH — {len(DOMAINS)} DOMÍNIOS × {args.docs_per//1_000_000}M docs    ║")
    print(f"║  Total: {total:,} docs                       ║")
    print(f"╚═══════════════════════════════════════╝")
    print()

    all_tasks = []
    for dk, (name, _) in DOMAINS.items():
        out = Path(args.output) / dk
        out.mkdir(parents=True, exist_ok=True)
        chunks = [(i, min(50000, args.docs_per - i*50000), str(out),
                   hash(f"fp_{dk}_{i}")%(2**31), dk)
                  for i in range((args.docs_per + 50000 - 1)//50000)]
        all_tasks.extend(chunks)
        print(f"  {name:<35s}  {args.docs_per:,} docs")

    print(f"\n  {len(all_tasks)} chunks, {args.workers} workers\n")
    
    t0 = time.time()
    total_size = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed({ex.submit(generate_chunk, t): t for t in all_tasks}):
            _, n, s = f.result(); total_size += s

    elapsed = time.time() - t0
    print(f"\n✅ {total:,} docs em {elapsed:.0f}s")
    print(f"   Tamanho: {total_size/1e9:.1f} GB")
    
    for dk in DOMAINS:
        sz = sum(f.stat().st_size for f in Path(args.output, dk).glob('*.txt')) / 1e9
        print(f"   {DOMAINS[dk][0]:<35s} {sz:.1f} GB")

if __name__ == "__main__":
    main()
