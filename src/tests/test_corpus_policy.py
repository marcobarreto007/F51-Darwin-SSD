from f51_darwin.corpus_policy import CorpusDecision, evaluate_corpus_source


def test_approves_mathematics_public_domain_source() -> None:
    text = (
        "This theorem gives a proof of convergence for a sequence in calculus. "
        "The lemma follows from algebra, probability, and a differential equation. "
        "The proof is repeated with clear assumptions and no political advocacy. "
    ) * 8

    result = evaluate_corpus_source(
        text=text,
        source_url="https://www.gutenberg.org/ebooks/123",
        license_label="PD-US",
    )

    assert result.decision == CorpusDecision.APPROVE
    assert "science:mathematics" in result.tags


def test_quarantines_noncommercial_license_even_for_good_science() -> None:
    text = (
        "A clinical randomized trial measured a biomarker in epidemiology. "
        "The diagnosis section reports the pathophysiology and statistics. "
    ) * 10

    result = evaluate_corpus_source(
        text=text,
        source_url="https://example.edu/open-course",
        license_label="CC-BY-NC 4.0",
    )

    assert result.decision == CorpusDecision.QUARANTINE
    assert "license" in result.reason


def test_does_not_block_protected_class_terms_inside_science_context() -> None:
    text = (
        "The epidemiology study measured diagnosis outcomes by age, race, and sex. "
        "The clinical protocol reports biomarkers, statistics, and confidence intervals. "
    ) * 10

    result = evaluate_corpus_source(
        text=text,
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/example",
        license_label="CC-BY 4.0",
    )

    assert result.decision == CorpusDecision.APPROVE
    assert "science:medicine" in result.tags


def test_quarantines_high_activism_without_science_signal() -> None:
    text = (
        "Activists demand that everyone must dismantle the system and join the movement. "
        "Organizers call for pressure, action, praxis, and coalition mobilization. "
    ) * 10

    result = evaluate_corpus_source(
        text=text,
        source_url="https://example.org/editorial",
        license_label="CC-BY 4.0",
    )

    assert result.decision == CorpusDecision.QUARANTINE
    assert "activism" in result.reason


def test_tags_right_thought_without_confusing_it_with_science() -> None:
    text = (
        "Mises and Hayek describe economic calculation, private property, "
        "limited government, rule of law, and spontaneous order. "
    ) * 12

    result = evaluate_corpus_source(
        text=text,
        source_url="https://mises.org/library/example",
        license_label="CC-BY",
        domain_hint="political_economy",
    )

    assert result.decision == CorpusDecision.APPROVE
    assert "right_thought:austrian_economics" in result.tags
    assert "hint:political_economy" in result.tags


def test_rejects_pirate_prone_hosts() -> None:
    text = (
        "This theorem and proof discuss algebra, geometry, and probability in detail. "
    ) * 12

    result = evaluate_corpus_source(
        text=text,
        source_url="https://pdfcoffee.com/random-book",
        license_label="public domain",
    )

    assert result.decision == CorpusDecision.REJECT
    assert "host" in result.reason
