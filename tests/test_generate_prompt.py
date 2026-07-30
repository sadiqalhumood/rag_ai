"""Prompt assembly: visible markers, a mapping kept as data, refusal wording."""

from __future__ import annotations

from anyrag.core.config import GenerationConfig
from anyrag.core.tokenizer import get_tokenizer
from anyrag.core.types import Chunk, ChunkKind, Hit, RowRef
from anyrag.generate.packer import PackedContext, pack
from anyrag.generate.prompt import (
    REFUSAL_SENTINEL,
    SYSTEM_PROMPT,
    build_prompt,
    overhead_tokens,
)


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


HITS = [
    make_hit("c1", "customer_id: 7 | name: Ahmed Al-Sayed | region: Cairo", 0.9),
    make_hit("c2", "customer_id: 8 | name: Fatima Nasser | region: Giza", 0.7),
]


def test_prompt_carries_visible_stable_markers() -> None:
    packed = pack(HITS, GenerationConfig())
    prompt = build_prompt("Where is Ahmed Al-Sayed?", packed)

    assert "[1]" in prompt.user and "[2]" in prompt.user
    assert "Ahmed Al-Sayed" in prompt.user
    assert "Where is Ahmed Al-Sayed?" in prompt.user


def test_marker_map_is_data_not_reparsed_prose() -> None:
    packed = pack(HITS, GenerationConfig())
    prompt = build_prompt("q", packed)

    assert prompt.marker_map == {1: "c1", 2: "c2"}
    assert packed.by_marker(1).chunk_id == "c1"


def test_instructions_demand_citations_and_refusal() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "citation marker" in lowered
    assert "never invent a" in lowered
    assert REFUSAL_SENTINEL in SYSTEM_PROMPT
    assert REFUSAL_SENTINEL in build_prompt("q", pack(HITS, GenerationConfig())).user


def test_overhead_is_measured_with_the_tokenizer() -> None:
    tok = get_tokenizer()
    question = "What is the email of customer Ahmed Al-Sayed?"

    overhead = overhead_tokens(question, tokenizer=tok)
    skeleton = build_prompt(question, PackedContext())

    assert overhead >= tok.count(skeleton.text)
    # A longer question costs more; the value is not a constant.
    assert overhead_tokens(question * 5, tokenizer=tok) > overhead


def test_prompt_token_count_accounts_for_context() -> None:
    tok = get_tokenizer()
    empty = build_prompt("q", PackedContext())
    full = build_prompt("q", pack(HITS, GenerationConfig()))

    assert full.token_count(tok) > empty.token_count(tok)
