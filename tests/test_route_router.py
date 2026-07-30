"""Router and NL->SQL tests against a hand-built fake source.

The fixture here is written by hand rather than borrowed from the eval harness,
deliberately: the router must be exercised on a schema the harness knows nothing
about, so that passing these tests cannot be confused with fitting the harness.
"""

from __future__ import annotations

import pytest

from anyrag.core.errors import SqlGenerationError, UnsafeQueryError
from anyrag.core.lint import lint_readonly
from anyrag.core.types import (
    ColumnProfile,
    ColumnRole,
    ColumnSchema,
    QueryRoute,
    TableProfile,
    TableRef,
    TableSchema,
)
from anyrag.route.router import classify
from anyrag.route.schema_lexicon import SchemaLexicon
from anyrag.route.sqlgen import HeuristicSqlGenerator

SHIPS = TableRef("shipments")
PORTS = TableRef("ports")


def _profile(name: str, role: ColumnRole, categories=()) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        role=role,
        total_count=100,
        distinct_count=len(categories) or 90,
        categories=tuple(categories),
    )


@pytest.fixture()
def lexicon() -> SchemaLexicon:
    schemas = {
        "shipments": TableSchema(
            table=SHIPS,
            columns=(
                ColumnSchema("shipment_id", "INTEGER", is_primary_key=True),
                ColumnSchema("carrier", "TEXT"),
                ColumnSchema("weight_kg", "REAL"),
                ColumnSchema("dispatched_on", "DATE"),
                ColumnSchema("port_id", "INTEGER", references="ports.port_id"),
                ColumnSchema("remarks", "TEXT"),
            ),
            primary_key=("shipment_id",),
        ),
        "ports": TableSchema(
            table=PORTS,
            columns=(
                ColumnSchema("port_id", "INTEGER", is_primary_key=True),
                ColumnSchema("country", "TEXT"),
            ),
            primary_key=("port_id",),
        ),
    }
    profiles = {
        "shipments": TableProfile(
            table=SHIPS,
            row_count=100,
            columns={
                "shipment_id": _profile("shipment_id", ColumnRole.ID),
                "carrier": _profile(
                    "carrier", ColumnRole.CATEGORICAL, ["Maersk", "Evergreen", "CMA CGM"]
                ),
                "weight_kg": _profile("weight_kg", ColumnRole.NUMERIC),
                "dispatched_on": _profile("dispatched_on", ColumnRole.DATE),
                "port_id": _profile("port_id", ColumnRole.ID),
                "remarks": _profile("remarks", ColumnRole.FREE_TEXT),
            },
        ),
        "ports": TableProfile(
            table=PORTS,
            row_count=12,
            columns={
                "port_id": _profile("port_id", ColumnRole.ID),
                "country": _profile(
                    "country", ColumnRole.CATEGORICAL, ["Oman", "Egypt", "Spain"]
                ),
            },
        ),
    }
    lex = SchemaLexicon(schemas=schemas, profiles=profiles)
    lex.build()
    return lex


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "How many shipments were dispatched in 2024?",
        "What is the average weight_kg of shipments?",
        "Total weight of shipments per carrier",
        "Count of shipments by carrier",
    ],
)
def test_aggregate_questions_route_to_aggregate(question, lexicon) -> None:
    assert classify(question, lexicon).route is QueryRoute.AGGREGATE


@pytest.mark.parametrize(
    "question",
    [
        "Show me the shipment with remarks about a delayed crane",
        "Which shipment mentions damaged cargo?",
    ],
)
def test_entity_questions_route_to_lookup(question, lexicon) -> None:
    assert classify(question, lexicon).route is QueryRoute.LOOKUP


@pytest.mark.parametrize(
    "question",
    [
        "What columns does the shipments table have?",
        "Describe the ports table",
        "What fields are in shipments?",
    ],
)
def test_schema_questions_route_to_lookup_on_schema_cards(question, lexicon) -> None:
    from anyrag.core.types import ChunkKind

    decision = classify(question, lexicon)
    assert decision.route is QueryRoute.LOOKUP
    assert decision.preferred_kinds == frozenset({ChunkKind.SCHEMA_CARD})


def test_multi_table_question_routes_to_hybrid(lexicon) -> None:
    decision = classify("Which carrier shipped to ports in Oman?", lexicon)
    assert decision.route is QueryRoute.HYBRID
    assert len(decision.tables) >= 2


def test_superlative_without_agg_word_still_needs_sql(lexicon) -> None:
    assert classify("Which carrier is the largest?", lexicon).route is QueryRoute.AGGREGATE


def test_empty_question_is_safe(lexicon) -> None:
    assert classify("", lexicon).route is QueryRoute.LOOKUP
    assert classify("   ", lexicon).route is QueryRoute.LOOKUP


def test_classify_without_lexicon_still_works() -> None:
    assert classify("How many shipments are there?").route is QueryRoute.AGGREGATE


# --------------------------------------------------------------------------
# Schema lexicon
# --------------------------------------------------------------------------


def test_lexicon_matches_table_and_column_names(lexicon) -> None:
    assert SHIPS in lexicon.match_tables("how many shipments")
    cols = {m.column for m in lexicon.match_columns("average weight_kg")}
    assert "weight_kg" in cols


def test_lexicon_matches_categorical_values_only(lexicon) -> None:
    values = lexicon.match_values("shipments carried by Maersk")
    assert any(v.value == "Maersk" and v.column == "carrier" for v in values)
    # Free-text content must never become an equality predicate.
    assert lexicon.match_values("a delayed crane at the dock") == []


def test_lexicon_strips_id_suffix_for_prose(lexicon) -> None:
    cols = {m.column for m in lexicon.match_columns("which port")}
    assert "port_id" in cols


def test_foreign_key_path_found(lexicon) -> None:
    assert lexicon.foreign_key_path(SHIPS, PORTS) == ("port_id", "port_id")


# --------------------------------------------------------------------------
# SQL generation
# --------------------------------------------------------------------------


def test_count_with_categorical_filter(lexicon) -> None:
    sql = HeuristicSqlGenerator(lexicon=lexicon).generate(
        "How many shipments were carried by Maersk?"
    )
    assert "COUNT(*)" in sql
    assert '"carrier" = \'Maersk\'' in sql
    lint_readonly(sql)


def test_avg_picks_named_numeric_column(lexicon) -> None:
    sql = HeuristicSqlGenerator(lexicon=lexicon).generate(
        "What is the average weight_kg of shipments?"
    )
    assert "AVG(" in sql and "weight_kg" in sql


def test_group_by_is_detected(lexicon) -> None:
    sql = HeuristicSqlGenerator(lexicon=lexicon).generate(
        "How many shipments per carrier?"
    )
    assert "GROUP BY" in sql and "carrier" in sql


def test_year_filter_becomes_a_date_range(lexicon) -> None:
    sql = HeuristicSqlGenerator(lexicon=lexicon).generate(
        "How many shipments were there in 2024?"
    )
    assert "dispatched_on" in sql
    assert "2024-01-01" in sql and "2024-12-31" in sql


def test_iso_date_range(lexicon) -> None:
    sql = HeuristicSqlGenerator(lexicon=lexicon).generate(
        "How many shipments between 2023-03-01 and 2023-06-30?"
    )
    assert "BETWEEN '2023-03-01' AND '2023-06-30'" in sql


def test_generated_sql_always_passes_the_linter(lexicon) -> None:
    gen = HeuristicSqlGenerator(lexicon=lexicon)
    for question in [
        "How many shipments by carrier?",
        "Average weight_kg per carrier",
        "Total weight_kg for Evergreen in 2024",
        "How many shipments were carried by CMA CGM?",
    ]:
        lint_readonly(gen.generate(question))


def test_out_of_coverage_raises_rather_than_guessing(lexicon) -> None:
    """The failure mode that matters: refuse, never invent a number."""
    gen = HeuristicSqlGenerator(lexicon=lexicon)
    with pytest.raises(SqlGenerationError):
        gen.generate("Show me the shipment with the funniest remarks")
    with pytest.raises(SqlGenerationError):
        gen.generate("How many unicorns are there?")


def test_ambiguous_numeric_target_raises(lexicon) -> None:
    """Two summable columns and no hint: refuse instead of picking one."""
    lexicon.schemas["shipments"] = TableSchema(
        table=SHIPS,
        columns=lexicon.schemas["shipments"].columns
        + (ColumnSchema("declared_value", "REAL"),),
        primary_key=("shipment_id",),
    )
    lexicon.profiles["shipments"].columns["declared_value"] = _profile(  # type: ignore[index]
        "declared_value", ColumnRole.NUMERIC
    )
    lexicon.build()
    with pytest.raises(SqlGenerationError):
        HeuristicSqlGenerator(lexicon=lexicon).generate("What is the total for shipments?")


def test_can_generate_reports_coverage(lexicon) -> None:
    gen = HeuristicSqlGenerator(lexicon=lexicon)
    assert gen.can_generate("How many shipments by carrier?") is True
    assert gen.can_generate("What is the meaning of life?") is False


def test_literal_with_quote_is_escaped(lexicon) -> None:
    lexicon.profiles["ports"].columns["country"] = _profile(  # type: ignore[index]
        "country", ColumnRole.CATEGORICAL, ["Cote d'Ivoire"]
    )
    lexicon.build()
    sql = HeuristicSqlGenerator(lexicon=lexicon).generate(
        "How many ports in Cote d'Ivoire?"
    )
    assert "''" in sql
    lint_readonly(sql)


def test_identifier_injection_is_rejected() -> None:
    from anyrag.route.sqlgen import _quote_ident

    with pytest.raises(SqlGenerationError):
        _quote_ident('x"; DROP TABLE t; --')
