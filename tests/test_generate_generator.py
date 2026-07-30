"""End-to-end generator behaviour: refuse, answer-and-cite, stay deterministic.

`AnthropicGenerator` is exercised through an injected stub client. That keeps
the tests offline while still covering the parts that actually differ from the
extractive path: response unwrapping, the refusal sentinel, provider refusals,
and -- most importantly -- that a marker the model invented is rejected by the
same citation validator.
"""

from __future__ import annotations

import sys

import pytest

from anyrag.core.config import GenerationConfig
from anyrag.core.errors import ConfigError
from anyrag.core.tokenizer import get_tokenizer
from anyrag.core.types import Answer, Chunk, ChunkKind, Hit, QueryRoute, RowRef
from anyrag.generate import (
    AnthropicGenerator,
    ExtractiveGenerator,
    available_generators,
    get_generator,
)
from anyrag.generate.generator import _pack_for_prompt
from anyrag.generate.refusal import RefusalPolicy


def make_hit(chunk_id: str, text: str, score: float, pk: str = "1") -> Hit:
    return Hit(
        chunk=Chunk(
            chunk_id=chunk_id,
            source_id="src",
            kind=ChunkKind.ROW,
            text=text,
            row_refs=(RowRef(table="customers", pk=pk),),
            meta={"table": "customers"},
        ),
        score=score,
    )


def customer_hits() -> list[Hit]:
    rows = [
        ("c1", "customer_id: 7 | name: Ahmed Al-Sayed | email: ahmed@example.com "
               "| region: Cairo", 0.91, "7"),
        ("c2", "customer_id: 8 | name: Fatima Nasser | email: fatima@example.com "
               "| region: Giza", 0.62, "8"),
        ("c3", "customer_id: 9 | name: John Smith | email: john@example.com "
               "| region: Cairo", 0.41, "9"),
    ]
    return [make_hit(cid, text, score, pk) for cid, text, score, pk in rows]


ANSWERABLE = "What is the email of customer Ahmed Al-Sayed?"
DISTRACTOR = "What is the email of customer Zanzibar Petrov?"
CONFIG = GenerationConfig()


# --------------------------------------------------------------------------
# ExtractiveGenerator
# --------------------------------------------------------------------------


def test_strong_supporting_context_answers_and_cites() -> None:
    answer = ExtractiveGenerator().generate(ANSWERABLE, customer_hits(), CONFIG,
                                            route=QueryRoute.LOOKUP)

    assert answer.refused is False
    assert answer.citations
    assert "ahmed@example.com" in answer.text
    assert "[1]" in answer.text
    assert answer.cited_chunk_ids == ("c1",)
    assert RowRef(table="customers", pk="7") in answer.cited_row_refs
    assert answer.route is QueryRoute.LOOKUP


def test_question_with_no_supporting_context_refuses() -> None:
    answer = ExtractiveGenerator().generate(DISTRACTOR, customer_hits(), CONFIG)

    assert answer.refused is True
    assert answer.citations == ()
    # Which gate reports first is evaluation-order detail; that both lexical
    # gates rejected it is the behaviour worth pinning.
    gates = answer.trace["refusal"]["gates"]
    assert gates["lexical_overlap"] is False
    assert gates["entity_coverage"] is False


def test_empty_retrieval_refuses() -> None:
    answer = ExtractiveGenerator().generate(ANSWERABLE, [], CONFIG)
    assert answer.refused is True
    assert answer.reason.startswith("no_hits")


def test_weak_scores_refuse() -> None:
    weak = [make_hit(h.chunk.chunk_id, h.chunk.text, 0.0001) for h in customer_hits()]
    answer = ExtractiveGenerator().generate(ANSWERABLE, weak, CONFIG)
    assert answer.refused is True
    assert answer.reason.startswith("low_support_score")


def test_extractive_output_is_byte_identical_across_runs() -> None:
    hits = customer_hits()
    gen = ExtractiveGenerator()

    first = gen.generate(ANSWERABLE, hits, CONFIG)
    second = ExtractiveGenerator().generate(ANSWERABLE, hits, CONFIG)

    assert first.text.encode("utf-8") == second.text.encode("utf-8")
    assert first.cited_chunk_ids == second.cited_chunk_ids
    assert [c.quoted_span for c in first.citations] == \
           [c.quoted_span for c in second.citations]


def test_extractive_output_is_deterministic_on_tied_scores() -> None:
    hits = [make_hit(f"c{i}", f"customer_id: {i} | name: Ahmed Al-Sayed "
                              f"| email: ahmed{i}@example.com", 0.5, str(i))
            for i in range(6)]
    gen = ExtractiveGenerator()
    assert gen.generate(ANSWERABLE, hits, CONFIG).text == \
           gen.generate(ANSWERABLE, hits, CONFIG).text


def test_extractive_answer_only_contains_retrieved_values() -> None:
    """The composer selects; it never writes a value the context did not have."""
    hits = customer_hits()
    answer = ExtractiveGenerator().generate(ANSWERABLE, hits, CONFIG)

    corpus = " ".join(h.chunk.text for h in hits)
    for line in answer.text.splitlines():
        if not line.startswith("- "):
            continue
        body = line[2:].rsplit(" [", 1)[0]
        assert body in corpus


def test_answer_stays_within_the_answer_token_budget() -> None:
    tok = get_tokenizer()
    config = GenerationConfig(max_answer_tokens=24)
    answer = ExtractiveGenerator().generate(
        "Which customers are in the Cairo region?", customer_hits(), config
    )
    assert not answer.refused
    assert tok.count(answer.text) <= config.max_answer_tokens


def test_full_prompt_respects_the_measured_budget() -> None:
    tok = get_tokenizer()
    config = GenerationConfig(max_prompt_tokens=600, max_chunk_tokens=64)
    hits = [make_hit(f"c{i}", f"customer_id: {i} | name: Person {i} "
                              f"| email: p{i}@example.com | region: Cairo",
                     0.9 - i / 100.0, str(i))
            for i in range(60)]

    context, prompt = _pack_for_prompt("Who is in Cairo?", hits, config, tok)

    assert not context.is_empty
    assert len(context.blocks) < len(hits), "the budget must actually bind"
    assert prompt.token_count(tok) <= config.max_prompt_tokens


def test_prompt_scaffold_alone_can_exhaust_a_tiny_budget() -> None:
    """A budget smaller than the instructions is a refusal, not an overflow."""
    tok = get_tokenizer()
    config = GenerationConfig(max_prompt_tokens=50, max_chunk_tokens=32)

    context, _prompt = _pack_for_prompt("Who is in Cairo?", customer_hits(),
                                        config, tok)

    assert context.is_empty
    assert context.reason == "context_budget_exhausted"


def test_budget_too_small_for_any_chunk_refuses_rather_than_crashing() -> None:
    """Isolates the packing path: the lexical gates are relaxed on purpose.

    Without that, the refusal gates would reject this question first and the
    budget-exhaustion branch -- the thing under test -- would never run.
    """
    config = GenerationConfig(max_prompt_tokens=120, max_chunk_tokens=100_000)
    permissive = RefusalPolicy.from_config(
        config, min_overlap=0.0, require_entity_coverage=False
    )
    hits = [make_hit("giant", "value " * 20_000, 0.95)]

    answer = ExtractiveGenerator().generate(
        "What is the value in the giant record?", hits, config, policy=permissive
    )

    assert answer.refused is True
    assert answer.reason.startswith("context_budget_exhausted")


def test_context_with_no_citable_values_refuses() -> None:
    """Passing the gates is not enough; there must be something to quote."""
    permissive = RefusalPolicy.from_config(
        CONFIG, require_entity_coverage=False, min_overlap=0.0
    )
    hits = [make_hit("blank", "   \n  \n", 0.9)]

    answer = ExtractiveGenerator().generate(ANSWERABLE, hits, CONFIG,
                                            policy=permissive)

    assert answer.refused is True
    assert answer.reason.startswith("no_extractable_answer")


def test_refusal_trace_exposes_every_gate() -> None:
    answer = ExtractiveGenerator().generate(DISTRACTOR, customer_hits(), CONFIG)
    gates = answer.trace["refusal"]["gates"]
    assert set(gates) == {"support_score", "support_count", "lexical_overlap",
                          "entity_coverage"}
    assert answer.trace["refusal"]["ungrounded_entities"]


def test_policy_overrides_reach_the_generator() -> None:
    """A policy passed at call time must actually drive the gates."""
    hits = customer_hits()
    assert ExtractiveGenerator().generate(DISTRACTOR, hits, CONFIG).refused is True

    permissive = RefusalPolicy.from_config(
        CONFIG, min_overlap=0.0, require_entity_coverage=False
    )
    relaxed = ExtractiveGenerator().generate(
        DISTRACTOR, hits, CONFIG, policy=permissive
    )
    assert relaxed.refused is False  # the gates really were the thing refusing


def test_generation_config_drives_every_gate() -> None:
    """One config object configures the whole refusal path."""
    strict = GenerationConfig(min_overlap=0.99, min_entity_coverage=1.0)
    policy = RefusalPolicy.from_config(strict)
    assert policy.min_overlap == 0.99
    assert policy.min_entity_coverage == 1.0
    assert policy.overlap_top_k == strict.overlap_top_k
    assert policy.caseless_entity_min_length == strict.caseless_entity_min_length

    # ...and reaches the generator without an explicit policy argument.
    # One ungrounded entity of five terms: coverage 0.8, so the default 0.75
    # answers and a raised threshold refuses, driven purely by the config.
    borderline = "Is Ahmed Al-Sayed from Cairo or Zanzibar?"
    hits = customer_hits()
    assert ExtractiveGenerator().generate(
        borderline, hits, GenerationConfig()
    ).refused is False
    assert ExtractiveGenerator().generate(
        borderline, hits, GenerationConfig(min_entity_coverage=0.9)
    ).refused is True


def test_arabic_and_cjk_context_answers_and_cites() -> None:
    hits = [
        make_hit("ar", "name: أحمد السيد | region: القاهرة | orders: 3", 0.9, "7"),
        make_hit("cjk", "name: 田中 | region: 東京 | orders: 5", 0.8, "8"),
    ]
    answer = ExtractiveGenerator().generate("كم طلب لدى أحمد السيد؟", hits, CONFIG)
    assert answer.refused is False
    assert answer.citations


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def test_get_generator_defaults_to_extractive() -> None:
    assert isinstance(get_generator(), ExtractiveGenerator)
    assert isinstance(get_generator(None), ExtractiveGenerator)
    assert isinstance(get_generator("extractive"), ExtractiveGenerator)
    assert get_generator().name == "extractive"


def test_get_generator_rejects_unknown_names() -> None:
    with pytest.raises(ConfigError):
        get_generator("gpt")


def test_available_generators_reports_the_offline_default() -> None:
    assert available_generators()["extractive"] is True


def test_generator_conforms_to_the_answer_generator_protocol() -> None:
    from anyrag.core.interfaces import AnswerGenerator

    assert isinstance(ExtractiveGenerator(), AnswerGenerator)


# --------------------------------------------------------------------------
# AnthropicGenerator (env-gated, lazily imported, stub-driven)
# --------------------------------------------------------------------------


class _Block:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.content = [_Block(text)]
        self.stop_reason = stop_reason


class _Messages:
    def __init__(self, response: _Response) -> None:
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(kwargs)
        return self._response


class _StubClient:
    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.messages = _Messages(_Response(text, stop_reason))


def anthropic_gen(text: str, stop_reason: str = "end_turn") -> AnthropicGenerator:
    return AnthropicGenerator(client=_StubClient(text, stop_reason))


def test_importing_the_package_does_not_import_the_anthropic_sdk() -> None:
    """An offline import must never depend on the optional dependency."""
    sys.modules.pop("anthropic", None)
    import importlib

    import anyrag.generate as pkg

    importlib.reload(pkg)
    assert "anthropic" not in sys.modules


def test_constructing_without_a_key_or_client_raises_config_error(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        AnthropicGenerator()
    with pytest.raises(ConfigError):
        get_generator("anthropic")
    assert AnthropicGenerator.available() is False


def test_model_comes_from_the_environment(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("ANYRAG_ANTHROPIC_MODEL", "claude-opus-5")
    assert AnthropicGenerator(client=_StubClient("x")).model == "claude-opus-5"
    monkeypatch.delenv("ANYRAG_ANTHROPIC_MODEL")
    assert AnthropicGenerator(client=_StubClient("x")).model == "claude-sonnet-5"


def test_anthropic_answer_is_cited() -> None:
    gen = anthropic_gen("The email is ahmed@example.com [1].")
    answer = gen.generate(ANSWERABLE, customer_hits(), CONFIG)

    assert answer.refused is False
    assert answer.cited_chunk_ids == ("c1",)
    assert answer.trace["generator"] == "anthropic"


def test_anthropic_hallucinated_marker_is_rejected() -> None:
    gen = anthropic_gen("Ahmed is in Cairo [1] and also in Tokyo [9].")
    answer = gen.generate(ANSWERABLE, customer_hits(), CONFIG)

    assert answer.refused is False
    assert answer.cited_chunk_ids == ("c1",)  # [9] did not become a citation
    assert answer.trace["citations"]["invalid_markers"] == [9]
    assert "[9]" not in answer.text


def test_anthropic_answer_with_only_invalid_markers_refuses() -> None:
    gen = anthropic_gen("It is in the archive [4][5].")
    answer = gen.generate(ANSWERABLE, customer_hits(), CONFIG)

    assert answer.refused is True
    assert answer.reason.startswith("no_valid_citations")


def test_anthropic_uncited_answer_refuses() -> None:
    gen = anthropic_gen("The email is ahmed@example.com.")
    answer = gen.generate(ANSWERABLE, customer_hits(), CONFIG)
    assert answer.refused is True
    assert isinstance(answer, Answer)


def test_anthropic_refusal_sentinel_is_honoured() -> None:
    from anyrag.generate.prompt import REFUSAL_SENTINEL

    gen = anthropic_gen(REFUSAL_SENTINEL)
    answer = gen.generate(ANSWERABLE, customer_hits(), CONFIG)

    assert answer.refused is True
    assert answer.reason.startswith("model_declared_insufficient_context")


def test_anthropic_provider_refusal_stop_reason_is_honoured() -> None:
    gen = anthropic_gen("", stop_reason="refusal")
    answer = gen.generate(ANSWERABLE, customer_hits(), CONFIG)

    assert answer.refused is True
    assert answer.reason.startswith("provider_refusal")


def test_anthropic_refuses_before_spending_a_call() -> None:
    """The refusal gates run first; a distractor never reaches the API."""
    client = _StubClient("should not be reached [1]")
    gen = AnthropicGenerator(client=client)

    answer = gen.generate(DISTRACTOR, customer_hits(), CONFIG)

    assert answer.refused is True
    assert client.messages.calls == []


def test_anthropic_sends_the_assembled_prompt() -> None:
    client = _StubClient("email: ahmed@example.com [1]")
    gen = AnthropicGenerator(client=client, tokenizer=get_tokenizer())
    gen.generate(ANSWERABLE, customer_hits(), CONFIG)

    (call,) = client.messages.calls
    assert call["model"] == gen.model
    assert call["max_tokens"] == CONFIG.max_answer_tokens
    assert "citation marker" in call["system"].lower()
    assert "[1]" in call["messages"][0]["content"]
    # Sampling parameters are rejected on current Claude models.
    assert "temperature" not in call and "top_p" not in call
