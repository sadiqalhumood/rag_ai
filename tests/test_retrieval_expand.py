"""Deterministic query expansion.

Every rule here is a bet that the corpus spells something differently from the
question. The tests pin the bets: hyphenated vs spaced vs joined surnames,
Arabic orthography, ISO dates, normalised decimals, transliterated names. They
also pin the conservatism -- the original query first, a hard cap on variants,
and no expansion at all for a query no rule applies to.
"""

from __future__ import annotations

import pytest

from anyrag.core.interfaces import QueryExpander
from anyrag.retrieval.expand import (
    DeterministicExpander,
    NoopExpander,
    get_expander,
)


def expand(query: str, **kw) -> list[str]:
    return DeterministicExpander(**kw).expand(query)


def rules_for(query: str, **kw) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for rule, variant in DeterministicExpander(max_variants=20, **kw).explain(query):
        out.setdefault(rule, []).append(variant)
    return out


def test_conforms_to_the_protocol() -> None:
    assert isinstance(DeterministicExpander(), QueryExpander)
    assert isinstance(NoopExpander(), QueryExpander)


def test_original_query_is_always_first_and_kept() -> None:
    query = "which customers are in Cairo"
    for variants in (expand(query), NoopExpander().expand(query)):
        assert variants[0] == query


def test_a_query_no_rule_applies_to_is_not_expanded() -> None:
    assert expand("which customers are in Cairo") == ["which customers are in Cairo"]


def test_empty_and_whitespace_queries_expand_to_themselves() -> None:
    assert expand("") == [""]
    assert expand("   ") == ["   "]


def test_hyphenated_name_gets_the_spaced_and_joined_spellings() -> None:
    variants = expand("email of Ahmed Al-Sayed", max_variants=6)
    assert "email of Ahmed Al Sayed" in variants
    assert "email of Ahmed AlSayed" in variants


def test_spaced_name_gets_the_hyphenated_and_joined_spellings() -> None:
    variants = expand("email of Ahmad Al Sayed", max_variants=6)
    assert "email of Ahmad Al-Sayed" in variants
    assert "email of Ahmad AlSayed" in variants


def test_transliteration_variants_are_generated_from_the_closed_table() -> None:
    assert "email of Ahmad Hassan" in expand("email of Ahmed Hassan", max_variants=8)
    assert "orders of Mohammed" in expand("orders of Mohamed", max_variants=8)


def test_transliteration_does_not_collapse_distinct_names() -> None:
    # Hassan and Hussein are different people and must not expand into each other.
    variants = rules_for("Hassan").get("translit", [])
    assert "Hasan" in variants
    assert not any("Hussein" in v or "Hussain" in v for v in variants)


def test_arabic_orthography_is_normalised() -> None:
    variants = rules_for("عميل أحمد السيّد")["arabic"]
    assert variants == ["عميل احمد السيد"]


def test_arabic_rule_does_not_fire_on_latin_text() -> None:
    assert "arabic" not in rules_for("customer Ahmed")


def test_dates_are_rewritten_into_the_iso_form_the_corpus_uses() -> None:
    assert rules_for("orders placed on 5 January 2023")["date"] == [
        "orders placed on 2023-01-05"
    ]
    assert rules_for("orders placed on January 5, 2023")["date"] == [
        "orders placed on 2023-01-05"
    ]
    assert rules_for("signups in January 2023")["date"] == ["signups in 2023-01"]


def test_a_bare_year_needs_no_date_variant() -> None:
    # "2023" already tokenises identically against "2023-04-02".
    assert "date" not in rules_for("orders in 2023")


def test_numbers_are_normalised_the_way_row_text_renders_them() -> None:
    assert rules_for("orders of 1,234.50")["number"] == ["orders of 1234.5"]
    assert rules_for("amount 2,000")["number"] == ["amount 2000"]
    assert "number" not in rules_for("order 1001")


def test_acronym_needs_three_capitalised_words() -> None:
    assert rules_for("orders from Gulf National Bank")["acronym"] == [
        "orders from GNB"
    ]
    # A two-word personal name is not an initialism.
    assert "acronym" not in rules_for("email of Mona Hassan")


def test_variants_are_capped_and_the_cap_keeps_the_safest_rules() -> None:
    query = "email of Ahmed Al-Sayed on 5 January 2023"
    capped = DeterministicExpander(max_variants=2).expand(query)
    assert len(capped) == 3  # original + 2
    ordered = [r for r, _v in DeterministicExpander(max_variants=20).explain(query)]
    assert ordered.index("punctuation") < ordered.index("translit")


def test_expansion_is_deterministic() -> None:
    query = "email of Ahmed Al-Sayed on 5 January 2023"
    assert expand(query, max_variants=8) == expand(query, max_variants=8)


def test_variants_are_deduplicated_case_insensitively() -> None:
    variants = expand("Ahmed Al-Sayed", max_variants=10)
    assert len(variants) == len({v.casefold() for v in variants})


def test_rules_can_be_restricted_and_unknown_rules_raise() -> None:
    only_punct = DeterministicExpander(rules=("punctuation",), max_variants=9)
    assert only_punct.expand("Ahmed Al-Sayed") == ["Ahmed Al-Sayed", "Ahmed Al Sayed"]
    with pytest.raises(ValueError):
        DeterministicExpander(rules=("nope",)).expand("x")


def test_get_expander_builds_by_name_and_rejects_unknown() -> None:
    assert isinstance(get_expander("noop"), NoopExpander)
    assert isinstance(get_expander(), DeterministicExpander)
    assert get_expander("deterministic", max_variants=1).max_variants == 1
    with pytest.raises(ValueError):
        get_expander("telepathy")
