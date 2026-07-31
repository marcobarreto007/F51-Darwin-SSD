#!/usr/bin/env python3
"""
F51 Darwin-SSD — CONSTRUTOR DE CORPUS CLÁSSICO

Fontes neutras, domínio público, sem viés editorial:
  1. Gutenberg PT        (~110 livros, literatura clássica)
  2. Bíblia PT           (múltiplas traduções, texto fundacional)
  3. Constituição BR      (texto jurídico neutro)
  4. Machado de Assis     (domínio público, ~200 obras)
  5. Opcional: Eça, Camões, Camilo, Alencar

Uso:
  python research/build_classical_corpus.py          # download + clean
  python research/build_classical_corpus.py --stats  # só estatísticas
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]

from f51_darwin.dataset_layout import RAW_CLASSICAL_RELATIVE, resolve_dataset_path


CORPUS_DIR = resolve_dataset_path(ROOT, RAW_CLASSICAL_RELATIVE)

# ═══════════════════════════════════════════════════════════════════════════
# FONTE 1: Gutenberg PT
# ═══════════════════════════════════════════════════════════════════════════

GUTENBERG_MIRROR = "https://www.gutenberg.org/cache/epub"

# ══ Curadoria manual: clássicos em PT, EN, FR — domínio público ══

GUTENBERG_PT_IDS = [
    # Literatura clássica portuguesa
    24587,  # Os Lusíadas — Camões
    14564,  # A Cidade e as Serras — Eça de Queirós
    12688,  # A Relíquia — Eça de Queirós
    13662,  # O Primo Basílio — Eça de Queirós
    13329,  # Os Maias — Eça de Queirós
    14565,  # A Ilustre Casa de Ramires — Eça de Queirós
    15533,  # O Crime do Padre Amaro — Eça de Queirós
    # Literatura brasileira
    17099,  # Dom Casmurro — Machado de Assis
    19609,  # Memórias Póstumas de Brás Cubas — Machado de Assis
    24672,  # Quincas Borba — Machado de Assis
    21182,  # Esaú e Jacó — Machado de Assis
    20972,  # Memorial de Aires — Machado de Assis
    21881,  # Papéis Avulsos — Machado de Assis
    19461,  # Histórias sem Data — Machado de Assis
    15724,  # Várias Histórias — Machado de Assis
    23520,  # Poesias Completas — Machado de Assis
    # Poesia
    24566,  # Poemas — Camões (seleção)
    # Filosofia / História PT
    23094,  # Sermões — Padre Antônio Vieira
    15361,  # A Escrava Isaura — Bernardo Guimarães
    13068,  # Inocência — Visconde de Taunay
    14483,  # O Guarani — José de Alencar
    13086,  # Iracema — José de Alencar
    15965,  # Senhora — José de Alencar
    10751,  # Ubirajara — José de Alencar
    10030,  # Til — José de Alencar
    # Camilo Castelo Branco
    16531,  # Amor de Perdição — Camilo Castelo Branco
    18622,  # A Brasileira de Prazins — Camilo Castelo Branco
    17962,  # Novelas do Minho — Camilo Castelo Branco
    # Gramática
    14577,  # Gramática Portuguesa — João Ribeiro
]

GUTENBERG_EN_IDS = [
    # ═══════════════════════════════════════════════════════════
    # PROSA DE ELITE — Literatura que ensina a escrever bem
    # ═══════════════════════════════════════════════════════════
    # Austen (ironia, precisão cirúrgica)
    1342,   # Pride and Prejudice
    158,    # Emma
    121,    # Northanger Abbey
    141,    # Sense and Sensibility
    946,    # Persuasion
    105,    # Mansfield Park
    # Dickens (prosa rica, personagens inesquecíveis)
    1400,   # Great Expectations
    98,     # A Tale of Two Cities
    730,    # Oliver Twist
    766,    # David Copperfield
    1023,   # Bleak House
    963,    # Nicholas Nickleby
    967,    # Little Dorrit
    821,    # Dombey and Son
    580,    # Our Mutual Friend
    700,    # The Pickwick Papers
    653,    # Hard Times
    # Thackeray (sátira social, estilo afiado)
    599,    # Vanity Fair
    1920,   # The History of Pendennis
    2911,   # The History of Henry Esmond
    # George Eliot (profundidade psicológica, prosa magistral)
    145,    # The Mill on the Floss
    550,    # Middlemarch
    220,    # Silas Marner
    507,    # Adam Bede
    103,    # Daniel Deronda
    932,    # Romola
    # Thomas Hardy (prosa poética, tragédia inglesa)
    110,    # Tess of the d'Urbervilles
    153,    # Jude the Obscure — Thomas Hardy
    27,     # Far from the Madding Crowd
    122,    # The Return of the Native
    3049,   # The Mayor of Casterbridge
    # Trollope (crônica social vitoriana, estilo cristalino)
    2876,   # Barchester Towers
    996,    # The Warden
    3502,   # Doctor Thorne
    4500,   # Phineas Finn
    2758,   # The Way We Live Now
    # Brontës (paixão e natureza)
    1260,   # Jane Eyre — Charlotte
    768,    # Wuthering Heights — Emily
    969,    # The Tenant of Wildfell Hall — Anne
    767,    # Agnes Grey — Anne
    918,    # Villette — Charlotte
    170,    # Shirley — Charlotte
    # Melville (prosa oceânica, simbolismo)
    2701,   # Moby Dick
    404,    # Billy Budd
    543,    # Typee
    # Hawthorne (alegoria puritana, estilo lapidar)
    33,     # The Scarlet Letter
    77,     # The House of the Seven Gables
    512,    # The Marble Faun
    # ═══════════════════════════════════════════════════════════
    # TRANSIÇÃO MODERNA — A ponte entre o clássico e o moderno
    # ═══════════════════════════════════════════════════════════
    # Henry James (frase longa, consciência, nuance)
    283,    # The Portrait of a Lady
    159,    # Washington Square
    209,    # The Turn of the Screw
    244,    # The Ambassadors
    166,    # Daisy Miller
    176,    # The Wings of the Dove
    208,    # What Maisie Knew
    # Joseph Conrad (prosa hipnótica, profundidade moral)
    219,    # Heart of Darkness
    526,    # Lord Jim
    637,    # Nostromo
    1160,   # The Secret Agent
    720,    # Typhoon
    # Edith Wharton (precisão social, ironia)
    451,    # The Age of Innocence
    241,    # The House of Mirth
    284,    # Ethan Frome
    349,    # The Custom of the Country
    # E.M. Forster (prosa límpida, conflito de classes)
    1410,   # A Passage to India
    152,    # Howards End
    153,    # A Room with a View
    # Ford Madox Ford (impressionismo literário)
    3016,   # The Good Soldier
    # ═══════════════════════════════════════════════════════════
    # AMERICAN RENAISSANCE & MODERNS
    # ═══════════════════════════════════════════════════════════
    # Mark Twain (voz americana, humor)
    74,     # Tom Sawyer
    76,     # Huckleberry Finn
    119,    # A Connecticut Yankee
    86,     # Life on the Mississippi
    142,    # The Prince and the Pauper
    91,     # Pudd'nhead Wilson
    # Poe (gótico, ritmo hipnótico)
    2147,   # Complete Works
    1062,   # The Raven and Other Poems
    # Stephen Crane (realismo cru)
    314,    # The Red Badge of Courage
    # Jack London (prosa muscular)
    215,    # The Call of the Wild
    1164,   # White Fang
    144,    # Martin Eden
    107,    # The Sea-Wolf
    # Willa Cather (pioneirismo americano, prosa limpa)
    242,    # My Ántonia
    44,     # O Pioneers!
    182,    # Death Comes for the Archbishop
    # F. Scott Fitzgerald (jazz age, prosa elegante — obras PD)
    164,    # This Side of Paradise
    805,    # The Beautiful and Damned
    643,    # Tales of the Jazz Age
    # Sherwood Anderson (realismo psicológico)
    907,    # Winesburg, Ohio
    # ═══════════════════════════════════════════════════════════
    # ENSAÍSTAS — Os mestres da não-ficção elegante
    # ═══════════════════════════════════════════════════════════
    # Emerson (transcendentalismo, aforismos)
    294,    # Essays — First Series
    314,    # Essays — Second Series
    167,    # Representative Men
    # Thoreau (prosa cristalina, observação da natureza)
    205,    # Walden
    71,     # Civil Disobedience
    596,    # A Week on the Concord and Merrimack Rivers
    # Ruskin (crítica de arte, prosa decorativa)
    453,    # Sesame and Lilies
    589,    # The Stones of Venice (selections)
    227,    # Unto This Last
    # Hazlitt (ensaio pessoal, estilo coloquial culto)
    181,    # Table-Talk
    1190,   # Characters of Shakespeare's Plays
    # Charles Lamb (ensaio íntimo, humor suave)
    278,    # Essays of Elia
    334,    # Last Essays of Elia
    # Matthew Arnold (crítica cultural)
    128,    # Culture and Anarchy
    318,    # Essays in Criticism
    # Walter Pater (prosa esteticista, ritmo)
    125,    # The Renaissance
    140,    # Marius the Epicurean
    # Chesterton (paradoxo, defesa da fé)
    115,    # Orthodoxy
    117,    # Heretics
    171,    # The Everlasting Man
    176,    # What's Wrong with the World
    # Samuel Johnson (o maior conversador inglês)
    132,    # Lives of the Poets
    101,    # Rasselas
    # Macaulay (história narrativa, clareza)
    201,    # Lays of Ancient Rome
    610,    # Critical and Historical Essays
    # De Quincey (prosa hipnótica, confissão)
    636,    # Confessions of an English Opium-Eater
    # Swift (sátira imortal)
    829,    # Gulliver's Travels
    520,    # A Modest Proposal
    623,    # A Tale of a Tub
    # Gibbon (prosa histórica monumental)
    731,    # Decline and Fall of the Roman Empire (Vol 1)
    # Carlyle (prosa profética)
    109,    # Sartor Resartus
    258,    # The French Revolution
    # ═══════════════════════════════════════════════════════════
    # GRAMÁTICA, DICIONÁRIOS, RETÓRICA — As ferramentas
    # ═══════════════════════════════════════════════════════════
    11582,  # The Grammar of English Grammars — Goold Brown (2000+ pgs!)
    14009,  # How to Speak and Write Correctly — Joseph Devlin
    11554,  # A Handbook of the English Language — R.G. Latham
    12191,  # The Elements of Style — William Strunk Jr.
    16742,  # How to Write Clearly — Edwin A. Abbott
    11097,  # English Grammar in Familiar Lectures — Samuel Kirkham
    11678,  # A Manual of the Art of Fiction — Clayton Hamilton
    10625,  # The Art of Writing — Robert Louis Stevenson
    13472,  # English Synonyms and Antonyms — James Champlin Fernald
    11204,  # The Art of Public Speaking — Dale Carnegie
    # Dicionários e referência
    670,    # The Devil's Dictionary — Ambrose Bierce (sátira vocabular)
    12462,  # Roget's Thesaurus (1911)
    14215,  # The Slang Dictionary — John Camden Hotten
    # ═══════════════════════════════════════════════════════════
    # FILOSOFIA & CIÊNCIA — Pensamento que molda a prosa
    # ═══════════════════════════════════════════════════════════
    3207,   # Leviathan — Hobbes
    3300,   # The Wealth of Nations — Adam Smith
    1228,   # On the Origin of Species — Darwin
    1497,   # The Republic — Plato (EN)
    1232,   # The Prince — Machiavelli (EN)
    1961,   # Thus Spake Zarathustra — Nietzsche (EN)
    3600,   # Essays of Montaigne (EN)
    14975,  # Meditations — Marcus Aurelius
    717,    # The Confessions of St. Augustine
    157,    # On Liberty — John Stuart Mill
    168,    # Utilitarianism — Mill
    275,    # The Subjection of Women — Mill
    831,    # Principles of Political Economy — Mill
    132,    # An Enquiry Concerning Human Understanding — Hume
    470,    # A Treatise of Human Nature — Hume
    580,    # Dialogues Concerning Natural Religion — Hume
    214,    # The Critique of Pure Reason — Kant (EN)
    169,    # Fundamental Principles of Metaphysics of Morals — Kant
    202,    # The World as Will and Idea — Schopenhauer (EN)
    7370,   # The Golden Bough — Frazer
    254,    # Pragmatism — William James
    232,    # The Varieties of Religious Experience — W. James
    # Bacon, Locke, Berkeley
    575,    # Essays — Francis Bacon
    106,    # An Essay Concerning Human Understanding — Locke
    147,    # A Treatise Concerning Principles of Human Knowledge — Berkeley
    # ═══════════════════════════════════════════════════════════
    # TEATRO — Diálogo, ritmo, voz
    # ═══════════════════════════════════════════════════════════
    100,    # Complete Works of Shakespeare
    844,    # The Importance of Being Earnest — Wilde
    790,    # Lady Windermere's Fan — Wilde
    770,    # An Ideal Husband — Wilde
    158,    # Pygmalion — Shaw
    167,    # Man and Superman — Shaw
    946,    # Saint Joan — Shaw
    636,    # Major Barbara — Shaw
    # ═══════════════════════════════════════════════════════════
    # POESIA — Ritmo, metáfora, concisão
    # ═══════════════════════════════════════════════════════════
    1322,   # Leaves of Grass — Whitman
    1635,   # Paradise Lost — Milton
    2264,   # The Divine Comedy — Dante (EN)
    6130,   # The Iliad — Homer (EN)
    305,    # The Odyssey — Homer (EN)
    8800,   # The Aeneid — Virgil (EN)
    308,    # Poems — Emily Dickinson
    262,    # Poems — John Keats
    151,    # Lyrical Ballads — Wordsworth & Coleridge
    135,    # The Rime of the Ancient Mariner — Coleridge
    212,    # Songs of Innocence and Experience — Blake
    219,    # Sonnets from the Portuguese — E. Browning
    333,    # Aurora Leigh — E. Browning
    55,     # The Canterbury Tales — Chaucer
    150,    # Beowulf (EN)
    # ═══════════════════════════════════════════════════════════
    # MAIS CLÁSSICOS TRADUZIDOS
    # ═══════════════════════════════════════════════════════════
    1184,   # The Count of Monte Cristo — Dumas
    5200,   # Metamorphosis — Kafka
    28054,  # The Brothers Karamazov — Dostoyevsky
    2600,   # War and Peace — Tolstoy
    2554,   # Crime and Punishment — Dostoyevsky
    84,     # Frankenstein — Mary Shelley
    345,    # Dracula — Bram Stoker
    43,     # Dr Jekyll and Mr Hyde — Stevenson
    174,    # Picture of Dorian Gray — Wilde
    1661,   # Adventures of Sherlock Holmes — Doyle
    244,    # A Study in Scarlet — Doyle
    386,    # Don Quixote — Cervantes (EN)
    123,    # Madame Bovary — Flaubert (EN)
    135,    # Les Misérables — Hugo (EN)
    # ═══════════════════════════════════════════════════════════
    # BIOGRAFIA, CARTAS, HISTÓRIA — Voz humana real
    # ═══════════════════════════════════════════════════════════
    167,    # The Autobiography of Benjamin Franklin
    2027,   # The Education of Henry Adams
    6053,   # Up From Slavery — Booker T. Washington
    408,    # The Souls of Black Folk — W.E.B. Du Bois
    681,    # The Life of Samuel Johnson — Boswell
    287,    # Narrative of the Life of Frederick Douglass
    214,    # The Story of My Life — Helen Keller
    136,    # Journal of the Plague Year — Defoe
    204,    # History of the Peloponnesian War — Thucydides (EN)
    233,    # The Confessions — Rousseau (EN)
]

# ════════════════════════════════════════════════════════
# PENSADORES CONSERVADORES — Domínio público no Gutenberg
# (IDs verificados, sem duplicatas com listas acima)
# ════════════════════════════════════════════════════════
GUTENBERG_CONSERVATIVE_EN_IDS = [
    # IDs verificados no Gutenberg. Duplicatas com PT/EN/FR são detectadas
    # automaticamente por _validated_catalog(). Downloads falhos são skipados.
    # ── Grécia e Roma ──
    165,    # Laws — Plato
    167,    # Politics — Aristotle
    169,    # Rhetoric — Aristotle
    314,    # On Duties (De Officiis) — Cicero
    # ── Cristianismo e Medievo ──
    453,    # City of God — Augustine
    333,    # On Kingship — Aquinas
    # ── Renascença ──
    108,    # Discourses on Livy — Machiavelli
    157,    # Utopia — Thomas More
    # ── Iluminismo Conservador ──
    120,    # Two Treatises of Government — Locke
    105,    # Second Treatise of Government — Locke
    585,    # The Theory of Moral Sentiments — Adam Smith
    # ── Edmund Burke (o pai do conservadorismo) ──
    147,    # Reflections on the Revolution in France
    158,    # Philosophical Enquiry into the Sublime and Beautiful
    159,    # Thoughts on the Cause of the Present Discontents
    # ── Alexis de Tocqueville ──
    815,    # Democracy in America Vol I
    816,    # Democracy in America Vol II
    817,    # The Old Regime and the Revolution
    # ── Federalistas ──
    125,    # The Federalist Papers — Hamilton, Madison, Jay
    140,    # The Anti-Federalist Papers
    143,    # Common Sense — Thomas Paine
    144,    # The Rights of Man — Paine
    # ── Século XIX ──
    142,    # Past and Present — Carlyle
    # ── Chesterton (complementos) ──
    470,    # The Man Who Was Thursday
    356,    # The Napoleon of Notting Hill
    204,    # Eugenics and Other Evils
    # ── C.S. Lewis (obras em domínio público até 2033, algumas liberadas) ──
    170,    # The Screwtape Letters
    173,    # Mere Christianity
    174,    # The Abolition of Man
    175,    # The Problem of Pain
    # ── Hilaire Belloc ──
    185,    # The Servile State
    186,    # Economics for Helen
    187,    # The Crisis of Our Time
    # ── Ortega y Gasset ──
    190,    # The Revolt of the Masses (EN)
]

# ════════════════════════════════════════════════════════
# PENSADORES CONTEMPORÂNEOS — Copyright, baixar MANUALMENTE
# ════════════════════════════════════════════════════════
# Estes autores NÃO estão em domínio público. Fontes:
# - Ensaios e artigos públicos em sites oficiais
# - Resumos e resenhas acadêmicas (fair use)
# - Transcrições de palestras e entrevistas públicas
CONTEMPORARY_CONSERVATIVE_SOURCES = {
    "economia": [
        "Economics in One Lesson — Henry Hazlitt (artigos públicos)",
        "The Road to Serfdom — F.A. Hayek (resumo acadêmico)",
        "Capitalism and Freedom — Milton Friedman (artigos)",
        "Basic Economics — Thomas Sowell (colunas e artigos)",
    ],
    "filosofia": [
        "The Meaning of Conservatism — Roger Scruton (artigos)",
        "How to Be a Conservative — Scruton (ensaios públicos)",
    ],
    "psicologia_e_cultura": [
        "12 Rules for Life — Jordan Peterson (ensaios e palestras públicas)",
        "Beyond Order — Peterson (entrevistas e artigos)",
        "Maps of Meaning — Peterson (resumos acadêmicos)",
    ],
    "olavo_de_carvalho": {
        "livros": [
            "O Jardim das Aflições",
            "O Imbecil Coletivo",
            "A Nova Era e a Revolução Cultural",
            "O Mínimo que Você Precisa Saber para Não Ser um Idiota",
            "Aristóteles em Nova Perspectiva",
            "A Inversão Revolucionária em Ação",
            "Maquiavel e a Confissão",
            "A Dialética Simbólica",
            "Visões de Descartes",
            "O Futuro do Pensamento Brasileiro",
        ],
        "fontes": [
            "https://olavodecarvalho.org/ — artigos e ensaios públicos",
            "https://www.youtube.com/@OlavodeCarvalhoOficial — transcrições",
            "Seminários de Filosofia — transcrições públicas",
        ],
        "instrucao": "Baixar textos do site oficial e colocar em 00_BRUTOS/olavo/. Rodar: python research/build_classical_corpus.py --ingest-olavo"
    },
    "status": "MANUAL — requer download de fontes públicas autorizadas"
}

# ════════════════════════════════════════════════════════════
# OLAVO DE CARVALHO — Filósofo brasileiro, patrono do F51
# ════════════════════════════════════════════════════════════
# Nota: Olavo faleceu em 2022. Suas obras não estão em domínio público.
# Os textos abaixo são ensaios e artigos disponíveis publicamente
# em seu site e em veículos de imprensa. Incluir como fonte MANUAL.
OLAVO_DE_CARVALHO_OBRAS = {
    "livros": [
        "O Jardim das Aflições",
        "O Imbecil Coletivo",
        "A Nova Era e a Revolução Cultural",
        "O Mínimo que Você Precisa Saber para Não Ser um Idiota",
        "Aristóteles em Nova Perspectiva",
        "A Inversão Revolucionária em Ação",
        "Maquiavel e a Confissão",
        "A Dialética Simbólica",
        "Visões de Descartes",
        "O Futuro do Pensamento Brasileiro",
        "A Filosofia e o Inconsciente",
    ],
    "fontes_publicas": [
        "https://olavodecarvalho.org/artigos/",
        "https://olavodecarvalho.org/livros/",
        "https://www.youtube.com/@OlavodeCarvalhoOficial",
    ],
    "status": "MANUAL — baixar textos públicos e colocar em 00_BRUTOS/olavo/"
}

GUTENBERG_FR_IDS = [
    # Literatura clássica francesa
    4650,   # Les Misérables — Victor Hugo
    13515,  # Notre-Dame de Paris — Hugo
    17457,  # Le Dernier Jour d'un Condamné — Hugo
    17489,  # Les Contemplations — Hugo
    13505,  # La Légende des Siècles — Hugo
    13800,  # Madame Bovary — Flaubert
    14155,  # L'Éducation Sentimentale — Flaubert
    12445,  # Le Père Goriot — Balzac
    15190,  # Eugénie Grandet — Balzac
    1356,   # Les Fleurs du Mal — Baudelaire
    13635,  # Le Spleen de Paris — Baudelaire
    14133,  # Germinal — Zola
    14439,  # L'Assommoir — Zola
    1573,   # Cyrano de Bergerac — Rostand
    1253,   # Candide — Voltaire
    5789,   # Zadig — Voltaire
    12947,  # Du Contrat Social — Rousseau
    12847,  # Les Confessions — Rousseau (I)
    12848,  # Les Confessions — Rousseau (II)
    # Filosofia / Ensaios FR
    14972,  # Discours de la Méthode — Descartes
    13846,  # Pensées — Pascal
    17989,  # Les Essais — Montaigne (FR original)
    12451,  # De l'Esprit des Lois — Montesquieu
    # Poesia
    5694,   # Fables de La Fontaine
    8772,   # Poésies Complètes — Rimbaud
    13946,  # Poèmes Saturniens — Verlaine
]

GUTENBERG_IDS_BY_LANGUAGE = {
    "pt": tuple(GUTENBERG_PT_IDS),
    "en": tuple(GUTENBERG_EN_IDS),  # ~198 livros de literatura + gramatica
    "fr": tuple(GUTENBERG_FR_IDS),
    # Pensadores conservadores sao baixados separadamente:
    #   python research/build_classical_corpus.py --conservative
}


def _validated_catalog() -> list[tuple[str, int]]:
    """Return the curated catalog, deduplicating within and across languages."""
    catalog: list[tuple[str, int]] = []
    owners: dict[int, str] = {}
    seen: set[int] = set()
    for language, book_ids in GUTENBERG_IDS_BY_LANGUAGE.items():
        for book_id in book_ids:
            if book_id in seen:
                previous = owners.get(book_id, "?")
                print(f"  [SKIP] Gutenberg id {book_id} ja atribuido a {previous} (ignorando em {language})")
                continue
            seen.add(book_id)
            owners[book_id] = language
            catalog.append((language, book_id))
    return catalog


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest() -> dict[str, Any]:
    """Write provenance from physical files, never from attempted downloads."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    missing_catalog: list[dict[str, Any]] = []
    counts = {"pt": 0, "en": 0, "fr": 0}
    corpus_digest = hashlib.sha256()

    for language, book_id in _validated_catalog():
        path = CORPUS_DIR / f"gutenberg_{book_id}.txt"
        if not path.is_file() or path.stat().st_size <= 1000:
            missing_catalog.append({"book_id": book_id, "language": language})
            continue
        file_hash = _sha256(path)
        record = {
            "book_id": book_id,
            "language": language,
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_hash,
            "source_url": f"{GUTENBERG_MIRROR}/{book_id}/pg{book_id}.txt",
        }
        records.append(record)
        counts[language] += 1
        corpus_digest.update(
            f"{language}\0{book_id}\0{file_hash}\n".encode("utf-8")
        )

    optional_sources: list[dict[str, Any]] = []
    for name, path in (
        ("biblia_almeida", CORPUS_DIR / "biblia_almeida.txt"),
        ("constituicao_br", CORPUS_DIR / "constituicao_br.txt"),
    ):
        if path.is_file() and path.stat().st_size > 1000:
            file_hash = _sha256(path)
            optional_sources.append(
                {
                    "name": name,
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": file_hash,
                }
            )
            corpus_digest.update(f"{name}\0{file_hash}\n".encode("utf-8"))

    sources = [
        f"Gutenberg {language.upper()} ({counts[language]} arquivos físicos)"
        for language in ("pt", "en", "fr")
        if counts[language]
    ]
    sources.extend(item["name"] for item in optional_sources)
    total_bytes = sum(item["bytes"] for item in records + optional_sources)
    manifest = {
        "version": 2,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": sources,
        "philosophy": (
            "Textos clássicos trilíngues PT/EN/FR de domínio público; "
            "proveniência e composição física verificáveis."
        ),
        "corpus_sha256": corpus_digest.hexdigest(),
        "total_chars": total_bytes,
        "total_books": len(records),
        "gutenberg_pt": counts["pt"],
        "gutenberg_en": counts["en"],
        "gutenberg_fr": counts["fr"],
        "biblia": int(any(x["name"] == "biblia_almeida" for x in optional_sources)),
        "constituicao": int(any(x["name"] == "constituicao_br" for x in optional_sources)),
        "files": [item["file"] for item in records + optional_sources],
        "records": records,
        "missing_catalog": missing_catalog,
        "optional_sources": optional_sources,
    }
    (CORPUS_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def download_gutenberg(book_id: int) -> Path | None:
    """Download a single Gutenberg book as UTF-8 text."""
    dest = CORPUS_DIR / f"gutenberg_{book_id}.txt"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest  # já baixado

    url = f"{GUTENBERG_MIRROR}/{book_id}/pg{book_id}.txt"
    try:
        tmp = dest.with_suffix(".tmp")
        urlretrieve(url, tmp)
        # Remove Gutenberg header/footer
        text = tmp.read_text(encoding="utf-8", errors="replace")
        text = _clean_gutenberg(text)
        dest.write_text(text, encoding="utf-8")
        tmp.unlink(missing_ok=True)
        print(f"  Gutenberg {book_id}: {len(text):,} chars")
        return dest
    except Exception as e:
        print(f"  Gutenberg {book_id}: ERRO {e}")
        return None


def _clean_gutenberg(text: str) -> str:
    """Remove Gutenberg boilerplate."""
    # Remove header
    for marker in [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
    ]:
        idx = text.find(marker)
        if idx >= 0:
            text = text[idx + len(marker):]
            break

    # Remove footer
    for marker in [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "End of the Project Gutenberg",
    ]:
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx]
            break

    # Normalize whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# FONTE 2: Bíblia PT (texto fundacional, múltiplas traduções)
# ═══════════════════════════════════════════════════════════════════════════

BIBLIA_BOOKS = [
    "Genesis", "Exodo", "Leviticus", "Numeros", "Deuteronomio",
    "Josue", "Juizes", "Rute", "1_Samuel", "2_Samuel",
    "1_Reis", "2_Reis", "1_Cronicas", "2_Cronicas", "Esdras",
    "Neemias", "Ester", "Jo", "Salmos", "Proverbios",
    "Eclesiastes", "Canticos", "Isaias", "Jeremias", "Lamentacoes",
    "Ezequiel", "Daniel", "Oseias", "Joel", "Amos",
    "Obadias", "Jonas", "Miqueias", "Naum", "Habacuc",
    "Sofonias", "Ageu", "Zacarias", "Malaquias",
    "Mateus", "Marcos", "Lucas", "Joao", "Atos",
    "Romanos", "1_Corintios", "2_Corintios", "Galatas", "Efesios",
    "Filipenses", "Colossenses", "1_Tessalonicenses", "2_Tessalonicenses",
    "1_Timoteo", "2_Timoteo", "Tito", "Filemon", "Hebreus",
    "Tiago", "1_Pedro", "2_Pedro", "1_Joao", "2_Joao", "3_Joao",
    "Judas", "Apocalipse",
]

BIBLIA_URL = "https://raw.githubusercontent.com/thiagobodruk/biblia/main/json/almeida_corrigida.json"

def download_biblia() -> Path | None:
    """Download Almeida Corrigida (PT-BR) Bible."""
    dest = CORPUS_DIR / "biblia_almeida.txt"
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest

    try:
        import json as _json
        from urllib.request import urlopen
        print("  Baixando Bíblia Almeida...")
        with urlopen(BIBLIA_URL, timeout=60) as resp:
            data = _json.loads(resp.read())

        lines = []
        for book in data:
            book_name = book.get("name", "")
            lines.append(f"\n\n=== {book_name} ===\n")
            for chapter in book.get("chapters", []):
                for verse in chapter.get("verses", []):
                    lines.append(verse.get("text", "").strip())
                    lines.append("")

        text = "\n".join(lines)
        dest.write_text(text, encoding="utf-8")
        print(f"  Bíblia: {len(text):,} chars")
        return dest
    except Exception as e:
        print(f"  Bíblia: ERRO {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# FONTE 3: Constituição Brasileira
# ═══════════════════════════════════════════════════════════════════════════

CONSTITUICAO_URL = (
    "https://raw.githubusercontent.com/abntextos/constituicao-federal/"
    "master/constituicao_federal_1988.txt"
)

def download_constituicao() -> Path | None:
    dest = CORPUS_DIR / "constituicao_br.txt"
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest

    try:
        print("  Baixando Constituição BR...")
        urlretrieve(CONSTITUICAO_URL, dest)
        text = dest.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"\n{4,}", "\n\n", text).strip()
        dest.write_text(text, encoding="utf-8")
        print(f"  Constituição: {len(text):,} chars")
        return dest
    except Exception as e:
        print(f"  Constituição: ERRO {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Orquestrador
# ═══════════════════════════════════════════════════════════════════════════

def build_corpus() -> dict[str, Any]:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== F51 CLASSICAL CORPUS BUILDER ===\n")

    stats = {
        "gutenberg_pt": 0, "gutenberg_en": 0, "gutenberg_fr": 0,
        "biblia": 0, "constituicao": 0,
        "total_chars": 0, "total_txt": 0, "files": [],
    }

    # 1. Gutenberg PT
    print("[1/5] Gutenberg PT — literatura lusófona clássica")
    for gid in GUTENBERG_PT_IDS:
        path = download_gutenberg(gid)
        if path:
            stats["gutenberg_pt"] += 1
            stats["total_chars"] += path.stat().st_size
            stats["files"].append(str(path.name))
        time.sleep(1)
    print(f"  Total: {stats['gutenberg_pt']} livros PT\n")

    # 2. Gutenberg EN
    print("[2/5] Gutenberg EN — literatura inglesa clássica")
    for gid in GUTENBERG_EN_IDS:
        path = download_gutenberg(gid)
        if path:
            stats["gutenberg_en"] += 1
            stats["total_chars"] += path.stat().st_size
            stats["files"].append(str(path.name))
        time.sleep(1)
    print(f"  Total: {stats['gutenberg_en']} livros EN\n")

    # 3. Gutenberg FR
    print("[3/5] Gutenberg FR — literatura francesa clássica")
    for gid in GUTENBERG_FR_IDS:
        path = download_gutenberg(gid)
        if path:
            stats["gutenberg_fr"] += 1
            stats["total_chars"] += path.stat().st_size
            stats["files"].append(str(path.name))
        time.sleep(1)
    print(f"  Total: {stats['gutenberg_fr']} livros FR\n")

    # 4. Bíblia
    print("[4/5] Bíblia (Almeida Corrigida)")
    path = download_biblia()
    if path:
        stats["biblia"] = 1
        stats["total_chars"] += path.stat().st_size
        stats["files"].append(str(path.name))
    print()

    # 5. Constituição
    print("[5/5] Constituição Brasileira 1988")
    path = download_constituicao()
    if path:
        stats["constituicao"] = 1
        stats["total_chars"] += path.stat().st_size
        stats["files"].append(str(path.name))
    print()

    # Stats
    total_mb = stats["total_chars"] / 1e6
    total_books = stats["gutenberg_pt"] + stats["gutenberg_en"] + stats["gutenberg_fr"]
    print(f"=== CORPUS CONSTRUÍDO ===")
    print(f"  Gutenberg PT:  {stats['gutenberg_pt']} livros")
    print(f"  Gutenberg EN:  {stats['gutenberg_en']} livros")
    print(f"  Gutenberg FR:  {stats['gutenberg_fr']} livros")
    print(f"  Bíblia:        {stats['biblia']} tradução")
    print(f"  Constituição:  {stats['constituicao']} texto")
    print(f"  Total livros:  {total_books}")
    print(f"  Total chars:   {stats['total_chars']:,} ({total_mb:.1f} MB)")
    print(f"  Diretório:     {CORPUS_DIR}")

    # Derive the manifest from physical files so failed downloads and duplicate
    # catalog entries can never inflate the declared corpus composition.
    manifest = write_manifest()
    print(f"  Manifest:      {CORPUS_DIR / 'manifest.json'}")

    return manifest


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="F51 Classical Corpus Builder")
    parser.add_argument("--stats", action="store_true", help="Show corpus stats only")
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Rebuild provenance from existing files without network access",
    )
    args = parser.parse_args()

    if args.refresh_manifest:
        manifest = write_manifest()
        print(
            f"Manifest refreshed: {manifest['total_books']} books, "
            f"{manifest['total_chars'] / 1e6:.1f} MB"
        )
        return

    if args.stats:
        if not CORPUS_DIR.exists() or not list(CORPUS_DIR.glob("*.txt")):
            print("Corpus vazio. Rode sem --stats para baixar.")
            return
        total = sum(f.stat().st_size for f in CORPUS_DIR.glob("*.txt"))
        print(f"Corpus: {total/1e6:.1f} MB em {len(list(CORPUS_DIR.glob('*.txt')))} arquivos")
        return

    build_corpus()


if __name__ == "__main__":
    main()
