"""The refusal gates.

The distractor cases here are the ones that matter: a question naming a customer
who does not exist retrieves perfectly reasonable customer rows with high scores
and healthy plain lexical overlap. Only the weighted-overlap and entity-coverage
gates can tell it apart from the answerable version of the same question, and
`test_plain_overlap_alone_would_not_discriminate` pins exactly that.
"""

from __future__ import annotations

import pytest

from anyrag.core.config import GenerationConfig
from anyrag.core.types import Chunk, ChunkKind, Hit, RowRef
from anyrag.generate.refusal import (
    RefusalPolicy,
    assess,
    entity_terms,
    term_weights,
    terms,
    weighted_overlap,
)

CUSTOMER_ROWS = [
    "customer_id: 7 | name: Ahmed Al-Sayed | email: ahmed@example.com | region: Cairo",
    "customer_id: 8 | name: Fatima Nasser | email: fatima@example.com | region: Giza",
    "customer_id: 9 | name: John Smith | email: john@example.com | region: Cairo",
    "customer_id: 10 | name: Maria Lopez | email: maria@example.com | region: Lima",
    "customer_id: 11 | name: Wei Chen | email: wei@example.com | region: Beijing",
]


def make_hit(chunk_id: str, text: str, score: float) -> Hit:
    return Hit(
        chunk=Chunk(
            chunk_id=chunk_id,
            source_id="src",
            kind=ChunkKind.ROW,
            text=text,
            row_refs=(RowRef(table="customers", pk=chunk_id),),
            meta={"table": "customers"},
        ),
        score=score,
    )


def customer_hits() -> list[Hit]:
    return [
        make_hit(f"c{i}", row, score=0.9 - i * 0.1)
        for i, row in enumerate(CUSTOMER_ROWS)
    ]


CONFIG = GenerationConfig()


# --------------------------------------------------------------------------
# Lexical machinery
# --------------------------------------------------------------------------


def test_terms_drop_stopwords_and_normalise() -> None:
    got = terms("What is the EMAIL of the customers?")
    assert "email" in got
    assert "customer" in got  # singularised
    assert "the" not in got and "what" not in got


def test_terms_handle_arabic_variants_and_cjk() -> None:
    assert set(terms("أحمد")) == set(terms("احمد"))
    cjk = terms("客户购买")
    assert len(cjk) == 4, "CJK runs split per character"


def test_entity_terms_pick_out_values_not_grammar() -> None:
    ents = set(entity_terms("What is the email of Ahmed Al-Sayed in 2023?"))
    assert {"ahmed", "sayed", "2023"} <= ents
    assert "email" not in ents


def test_entity_terms_include_a_sentence_initial_name() -> None:
    assert "zanzibar" in entity_terms("Zanzibar Petrov's email address?")


def test_weighted_overlap_downweights_terms_common_to_every_chunk() -> None:
    weights = term_weights(CUSTOMER_ROWS)
    assert weights.weight("email") < weights.weight("ahmed")
    assert weights.weight("ahmed") < weights.weight("zanzibar")  # df == 0


# --------------------------------------------------------------------------
# The gates
# --------------------------------------------------------------------------


def test_no_hits_refuses() -> None:
    decision = assess("anything at all?", [], CONFIG)
    assert decision.refuse
    assert decision.code == "no_hits"


def test_low_support_score_refuses() -> None:
    hits = [make_hit("c0", CUSTOMER_ROWS[0], 0.001)]
    decision = assess("What is the email of Ahmed Al-Sayed?", hits, CONFIG)
    assert decision.refuse
    assert decision.code == "low_support_score"


def test_insufficient_support_chunks_refuses() -> None:
    hits = [make_hit("c0", CUSTOMER_ROWS[0], 0.9),
            make_hit("c1", CUSTOMER_ROWS[1], 0.001)]
    config = GenerationConfig(min_support_chunks=2)
    decision = assess("What is the email of Ahmed Al-Sayed?", hits, config)
    assert decision.refuse
    assert decision.code == "insufficient_support_chunks"


def test_strong_supporting_context_is_accepted() -> None:
    decision = assess(
        "What is the email of customer Ahmed Al-Sayed?", customer_hits(), CONFIG
    )
    assert not decision.refuse
    assert decision.code == "supported"
    assert decision.entity_coverage == 1.0
    assert all(decision.gates.values())


def test_question_naming_an_absent_entity_refuses() -> None:
    decision = assess(
        "What is the email of customer Zanzibar Petrov?", customer_hits(), CONFIG
    )
    assert decision.refuse
    assert decision.code == "ungrounded_entities"
    assert set(decision.ungrounded_entities) == {"zanzibar", "petrov"}


def test_plain_overlap_alone_would_not_discriminate() -> None:
    """The distractor clears the plain-overlap bar; only G4 catches it.

    This is the whole justification for the weighted metric and the entity
    gate. If it ever fails because the distractor's plain overlap fell below
    the threshold, the discrimination happened by luck, not by design.
    """
    hits = customer_hits()
    answerable = assess(
        "What is the email of customer Ahmed Al-Sayed?", hits, CONFIG
    )
    distractor = assess(
        "What is the email of customer Zanzibar Petrov?", hits, CONFIG
    )

    assert distractor.plain_overlap >= CONFIG.min_overlap
    assert distractor.best_overlap < answerable.best_overlap
    assert distractor.gates["lexical_overlap"] is True
    assert distractor.gates["entity_coverage"] is False


def test_unrelated_question_refuses_on_overlap() -> None:
    decision = assess(
        "quarterly photosynthesis rates for tundra lichen",
        customer_hits(),
        CONFIG,
    )
    assert decision.refuse
    assert decision.code in {"low_lexical_overlap", "ungrounded_entities"}


def test_absent_year_refuses() -> None:
    hits = [make_hit("o1", "order_id: 1 | order_date: 2023-04-02 | total: 90.0", 0.9),
            make_hit("o2", "order_id: 2 | order_date: 2024-01-11 | total: 55.5", 0.8)]
    assert assess("How many orders were placed in 2023?", hits, CONFIG).refuse is False
    absent = assess("How many orders were placed in 1998?", hits, CONFIG)
    assert absent.refuse
    assert absent.code == "ungrounded_entities"


def test_near_duplicate_arabic_spelling_still_counts_as_grounded() -> None:
    hits = [make_hit("c0", "name: أحمد السيد | region: القاهرة", 0.9)]
    decision = assess("ما هو سجل احمد السيد؟", hits, CONFIG)
    assert decision.entity_coverage == 1.0


# --------------------------------------------------------------------------
# Tunability
# --------------------------------------------------------------------------


def test_policy_is_derived_from_generation_config() -> None:
    config = GenerationConfig(min_support_score=0.5, min_support_chunks=3,
                              min_overlap=0.4)
    policy = RefusalPolicy.from_config(config)
    assert policy.min_support_score == 0.5
    assert policy.min_support_chunks == 3
    assert policy.min_overlap == 0.4


def test_entity_gate_can_be_disabled_for_ablation() -> None:
    hits = customer_hits()
    policy = RefusalPolicy.from_config(CONFIG, require_entity_coverage=False)
    decision = assess(
        "What is the email of customer Zanzibar Petrov?", hits, policy=policy
    )
    assert not decision.refuse


def test_raising_min_overlap_refuses_the_distractor_without_the_entity_gate() -> None:
    hits = customer_hits()
    policy = RefusalPolicy.from_config(
        CONFIG, require_entity_coverage=False, min_overlap=0.5
    )
    decision = assess(
        "What is the email of customer Zanzibar Petrov?", hits, policy=policy
    )
    assert decision.refuse
    assert decision.code == "low_lexical_overlap"


def test_entity_gate_tolerates_a_minority_of_ungrounded_terms() -> None:
    """One unknown name among four grounded ones is not enough to refuse."""
    hits = customer_hits()
    # Grounded: ahmed, al, sayed, cairo. Ungrounded: zanzibar -> 0.8.
    decision = assess("Is Ahmed Al-Sayed from Cairo or Zanzibar?", hits, CONFIG)
    assert pytest.approx(0.8, abs=1e-9) == decision.entity_coverage
    assert decision.ungrounded_entities == ("zanzibar",)
    assert not decision.refuse


def test_caseless_function_words_are_not_treated_as_entities() -> None:
    """Arabic grammar must not trip the entity gate on every Arabic question."""
    ents = set(entity_terms("ما هو سجل احمد السيد؟"))
    assert "احمد" in ents and "السيد" in ents
    assert "هو" not in ents and "سجل" not in ents


def test_decision_trace_is_json_friendly() -> None:
    import json

    decision = assess("What is Ahmed Al-Sayed's email?", customer_hits(), CONFIG)
    json.dumps(decision.as_trace())
