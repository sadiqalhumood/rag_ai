"""BM25Index: hand-written Okapi scoring with genuinely incremental statistics.

Two things are being defended here. First, that the ranking is real BM25 --
term-frequency saturation, length normalisation, IDF -- and not a bag-of-words
overlap count. Second, that upsert and delete maintain `df`, `|d|` and `avgdl`
in place, which is why this is not a `rank_bm25` wrapper.
"""

from __future__ import annotations

import json
import math
import os
import pathlib

import pytest

from anyrag.core.config import MetadataFilter
from anyrag.core.errors import IndexError_
from anyrag.core.interfaces import LexicalIndex
from anyrag.core.types import ChunkKind
from anyrag.index import MANIFEST_FILE, BM25Index, tokenize
from anyrag.index.bm25 import STATS_FILE
from test_index_fixtures import ids, make_chunk

AR_ORDERS = "طلب رقم ٤٥ من العميل شركة الخليج للتجارة بمبلغ ألف ريال"
AR_INVOICE = "فاتورة صادرة لشركة النيل للمقاولات في القاهرة"
EN_ORDERS = "order number 45 from customer gulf trading company for one thousand"


def build(*docs: tuple[str, str], **kw) -> BM25Index:
    idx = BM25Index(**kw)
    idx.upsert([make_chunk(cid, text) for cid, text in docs])
    return idx


# -- protocol and independence --------------------------------------------


def test_conforms_to_lexical_index_protocol() -> None:
    assert isinstance(BM25Index(), LexicalIndex)


def test_does_not_depend_on_rank_bm25() -> None:
    """The whole point of writing this by hand is the incremental update."""
    import anyrag.index.bm25 as mod

    src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    assert "import rank_bm25" not in src
    assert "from rank_bm25" not in src


# -- tokenization ----------------------------------------------------------


def test_tokenizer_is_unicode_aware_and_case_folding() -> None:
    assert tokenize("Hello, WORLD!") == ["hello", "world"]
    assert tokenize("مرحبا بالعالم") == ["مرحبا", "بالعالم"]
    assert tokenize("طلب 45 order_id") == ["طلب", "45", "order_id"]
    assert tokenize("") == []


def test_tokenizer_strips_arabic_diacritics_but_not_letters() -> None:
    """Harakat vary between rows describing the same thing; letters do not."""
    assert tokenize("مَكْتَبٌ") == tokenize("مكتب")
    # No stemming: a prefixed form stays a distinct term.
    assert tokenize("بالمكتب") != tokenize("مكتب")


# -- ranking ---------------------------------------------------------------


def test_exact_term_match_outranks_a_partial_one() -> None:
    idx = build(
        ("all", "northern region revenue summary"),
        ("partial", "revenue summary for the year"),
        ("other", "shipping costs and packaging"),
    )
    hits = idx.search("northern region revenue", k=3)
    assert hits[0].chunk_id == "all"
    assert hits[1].chunk_id == "partial"
    assert hits[0].score > hits[1].score
    assert "other" not in ids(hits)


def test_terms_match_whole_words_not_substrings() -> None:
    """No stemming means `bookkeeping` is simply a different term to `book`."""
    idx = build(("exact", "the book was returned"), ("sub", "bookkeeping ledger"))
    assert ids(idx.search("book", k=5)) == ["exact"]


def test_arabic_query_matches_an_arabic_document() -> None:
    idx = build(
        ("ar-order", AR_ORDERS),
        ("ar-invoice", AR_INVOICE),
        ("en-order", EN_ORDERS),
    )
    hits = idx.search("شركة الخليج للتجارة", k=3)
    assert hits[0].chunk_id == "ar-order"
    assert hits[0].score > 0
    assert idx.search("القاهرة", k=1)[0].chunk_id == "ar-invoice"
    # Mixed-script corpora must not cross-match on script alone.
    assert ids(idx.search("gulf trading", k=3)) == ["en-order"]


def test_rare_terms_outweigh_common_ones() -> None:
    idx = BM25Index()
    idx.upsert([make_chunk(f"c{i}", "invoice total amount") for i in range(20)])
    idx.upsert([make_chunk("rare", "invoice total amount kwacha")])
    hits = idx.search("invoice kwacha", k=3)
    assert hits[0].chunk_id == "rare"


def test_length_normalisation_prefers_the_shorter_document() -> None:
    padding = " ".join(f"filler{i}" for i in range(200))
    idx = build(("short", "revenue report"), ("long", f"revenue report {padding}"))
    hits = idx.search("revenue report", k=2)
    assert hits[0].chunk_id == "short"


def test_k1_and_b_are_configurable() -> None:
    docs = (("a", "alpha alpha alpha beta"), ("b", "alpha gamma delta epsilon zeta"))
    saturating = build(*docs, k1=0.001, b=0.0).search("alpha", k=2)
    linear = build(*docs, k1=1000.0, b=0.0).search("alpha", k=2)
    # With k1 -> 0 term frequency stops mattering; with a huge k1 it dominates.
    assert saturating[0].score == pytest.approx(saturating[1].score, rel=1e-3)
    assert linear[0].chunk_id == "a"
    assert linear[0].score > linear[1].score * 2
    assert BM25Index(k1=1.5, b=0.75).k1 == 1.5


def test_rank_is_one_based_contiguous_and_labelled_bm25() -> None:
    idx = build(*[(f"c{i}", f"widget {i} report revenue") for i in range(6)])
    hits = idx.search("report revenue widget", k=4)
    assert [h.rank for h in hits] == [1, 2, 3, 4]
    assert all(h.retriever == "bm25" for h in hits)
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:]))


def test_score_ties_are_broken_deterministically() -> None:
    idx = build(("z", "same text"), ("a", "same text"), ("m", "same text"))
    assert ids(idx.search("same text", k=3)) == ["a", "m", "z"]


# -- the incremental-update contract ---------------------------------------


def test_reupserting_identical_chunks_leaves_count_unchanged() -> None:
    chunks = [make_chunk(f"c{i}", f"row {i} revenue") for i in range(5)]
    idx = BM25Index()
    idx.upsert(chunks)
    before_df = idx.doc_freq("revenue")
    idx.upsert(chunks)
    idx.upsert(chunks)
    assert idx.count() == 5
    assert idx.doc_freq("revenue") == before_df == 5
    assert idx.avgdl == pytest.approx(3.0)


def test_upsert_with_changed_text_updates_scoring_not_count() -> None:
    idx = build(("c1", "revenue for the northern region"), ("c2", "shipping costs"))
    assert ids(idx.search("northern", k=5)) == ["c1"]
    idx.upsert([make_chunk("c1", "revenue for the southern region")])
    assert idx.count() == 2
    assert idx.get("c1").text == "revenue for the southern region"
    assert idx.search("northern", k=5) == []  # the old term is really gone
    assert ids(idx.search("southern", k=5)) == ["c1"]
    assert idx.doc_freq("northern") == 0


def test_statistics_are_maintained_incrementally_across_delete() -> None:
    idx = BM25Index()
    idx.upsert([make_chunk("a", "one two three"), make_chunk("b", "one two")])
    assert idx.avgdl == pytest.approx(2.5)
    assert idx.doc_freq("one") == 2 and idx.doc_freq("three") == 1
    assert idx.delete_by_ids(["a"]) == 1
    assert idx.avgdl == pytest.approx(2.0)
    assert idx.doc_freq("one") == 1
    assert idx.doc_freq("three") == 0
    # Vocabulary must shrink too, or the index leaks terms forever.
    assert idx.vocabulary_size == 2


def test_delete_by_source_removes_exactly_the_right_chunks() -> None:
    idx = BM25Index()
    idx.upsert(
        [make_chunk(f"a{i}", "shared text", source_id="sqlite:A") for i in range(4)]
        + [make_chunk(f"b{i}", "shared text", source_id="sqlite:B") for i in range(3)]
    )
    assert idx.delete_by_source("sqlite:A") == 4
    assert idx.count() == 3
    assert sorted(c.chunk_id for c in idx.iter_chunks()) == ["b0", "b1", "b2"]
    assert idx.doc_freq("shared") == 3
    assert idx.delete_by_source("sqlite:A") == 0


def test_delete_by_ids_counts_only_ids_that_were_present() -> None:
    idx = build(*[(f"c{i}", f"row {i}") for i in range(5)])
    assert idx.delete_by_ids(["c0", "ghost", "c0", "c4"]) == 2
    assert idx.count() == 3
    assert idx.delete_by_ids([]) == 0


def test_deleting_everything_leaves_a_usable_empty_index() -> None:
    idx = build(("a", "hello"), ("b", "world"))
    assert idx.delete_by_ids(["a", "b"]) == 2
    assert idx.count() == 0
    assert idx.avgdl == 0.0
    assert idx.search("hello", k=3) == []
    idx.upsert([make_chunk("c", "hello again")])
    assert ids(idx.search("hello", k=3)) == ["c"]


# -- filtering -------------------------------------------------------------


def test_filtered_search_returns_k_results_when_k_are_available() -> None:
    idx = BM25Index()
    chunks = [
        make_chunk(f"cust{i}", "revenue " * (i + 1), table="customers")
        for i in range(30)
    ] + [make_chunk(f"ord{i}", "revenue report", table="orders") for i in range(5)]
    idx.upsert(chunks)
    hits = idx.search("revenue", k=5, flt=MetadataFilter(tables=frozenset({"orders"})))
    assert len(hits) == 5
    assert all(h.chunk.table == "orders" for h in hits)
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]


def test_filter_by_kind_source_and_equals() -> None:
    idx = BM25Index()
    idx.upsert(
        [
            make_chunk("row-a", "revenue", source_id="sqlite:A", lang="en"),
            make_chunk("row-b", "revenue", source_id="sqlite:B", lang="ar"),
            make_chunk(
                "card-a", "revenue", source_id="sqlite:A",
                kind=ChunkKind.SCHEMA_CARD, lang="en",
            ),
        ]
    )
    assert ids(idx.search("revenue", k=5, flt=MetadataFilter(
        kinds=frozenset({ChunkKind.SCHEMA_CARD})))) == ["card-a"]
    assert ids(idx.search("revenue", k=5, flt=MetadataFilter(
        source_ids=frozenset({"sqlite:B"})))) == ["row-b"]
    assert ids(idx.search("revenue", k=5, flt=MetadataFilter(
        equals={"lang": "ar"}))) == ["row-b"]
    assert idx.search("revenue", k=5, flt=MetadataFilter(
        tables=frozenset({"nope"}))) == []


# -- empty / degenerate inputs --------------------------------------------


def test_empty_index_and_empty_query_return_empty_lists() -> None:
    assert BM25Index().search("anything", k=5) == []
    idx = build(("a", "hello world"))
    assert idx.search("", k=5) == []
    assert idx.search("   ,,, ", k=5) == []
    assert idx.search("hello", k=0) == []
    assert idx.search("nonexistentterm", k=5) == []
    assert idx.get("nope") is None


# -- persistence -----------------------------------------------------------


def test_persist_then_fresh_load_gives_identical_results(tmp_path) -> None:
    path = str(tmp_path / "lex")
    idx = BM25Index(path=path)
    idx.upsert(
        [make_chunk(f"c{i}", f"{AR_ORDERS} {i}" if i % 2 else f"{EN_ORDERS} {i}")
         for i in range(20)]
    )
    queries = ["gulf trading company", "شركة الخليج", "order 45", "١٧"]
    before = [idx.search(q, k=10) for q in queries]
    idx.persist()

    fresh = BM25Index()
    fresh.load(path)
    assert fresh.count() == 20
    assert fresh.k1 == idx.k1 and fresh.b == idx.b
    assert fresh.avgdl == pytest.approx(idx.avgdl)
    for q, expected in zip(queries, before):
        got = fresh.search(q, k=10)
        assert ids(got) == ids(expected)
        assert [h.rank for h in got] == [h.rank for h in expected]
        for a, b in zip(got, expected):
            assert a.score == pytest.approx(b.score, abs=1e-9)


def test_deletions_survive_a_persist_load_cycle(tmp_path) -> None:
    path = str(tmp_path / "lex")
    idx = BM25Index(path=path)
    idx.upsert(
        [make_chunk(f"a{i}", "alpha document", source_id="sqlite:A") for i in range(5)]
        + [make_chunk(f"b{i}", "beta document", source_id="sqlite:B") for i in range(5)]
    )
    idx.delete_by_source("sqlite:A")
    idx.delete_by_ids(["b0"])
    idx.persist()

    fresh = BM25Index()
    fresh.load(path)
    assert sorted(c.chunk_id for c in fresh.iter_chunks()) == ["b1", "b2", "b3", "b4"]
    assert fresh.search("alpha", k=5) == []
    assert fresh.doc_freq("alpha") == 0
    assert fresh.doc_freq("beta") == 4
    # And the reloaded index is still incrementally updatable.
    fresh.upsert([make_chunk("b1", "beta document revised")])
    assert fresh.count() == 4
    assert ids(fresh.search("revised", k=5)) == ["b1"]


def test_persisted_layout_is_plain_files(tmp_path) -> None:
    path = str(tmp_path / "lex")
    idx = BM25Index(path=path, k1=1.2, b=0.5)
    idx.upsert([make_chunk("a", "hello world"), make_chunk("b", "مرحبا بالعالم")])
    idx.persist()
    assert MANIFEST_FILE in os.listdir(path) and STATS_FILE in os.listdir(path)
    stats = json.loads(open(os.path.join(path, STATS_FILE), encoding="utf-8").read())
    assert stats["k1"] == 1.2 and stats["b"] == 0.5
    assert stats["postings"]["hello"] == {"a": 1}
    assert stats["doc_len"] == {"a": 2, "b": 2}


def test_autoload_reopens_an_existing_index(tmp_path) -> None:
    path = str(tmp_path / "lex")
    idx = BM25Index(path=path)
    idx.upsert([make_chunk("a", "hello world")])
    idx.persist()
    assert BM25Index(path=path, autoload=True).count() == 1
    assert BM25Index(path=str(tmp_path / "elsewhere"), autoload=True).count() == 0


def test_persist_and_load_error_cases(tmp_path) -> None:
    with pytest.raises(IndexError_, match="no path"):
        BM25Index().persist()
    with pytest.raises(IndexError_, match="no persisted index"):
        BM25Index().load(str(tmp_path / "absent"))


def test_load_rejects_a_dense_index_directory(tmp_path) -> None:
    from anyrag.index import PersistentVectorIndex

    path = str(tmp_path / "dense")
    dense = PersistentVectorIndex(path, dim=3)
    dense.upsert([make_chunk("a", "hello")], [[1.0, 0.0, 0.0]])
    dense.persist()
    with pytest.raises(IndexError_, match="not 'bm25'"):
        BM25Index().load(path)


# -- scoring sanity against the formula -----------------------------------


def test_score_matches_the_documented_okapi_formula() -> None:
    """A regression guard: the docstring formula is the implemented one."""
    idx = BM25Index(k1=1.5, b=0.75)
    idx.upsert([make_chunk("a", "alpha beta"), make_chunk("b", "alpha gamma delta")])
    hit = idx.search("alpha beta", k=1)[0]
    n, avgdl = 2, 2.5
    expected = 0.0
    for term, df, tf, dl in (("alpha", 2, 1, 2), ("beta", 1, 1, 2)):
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        expected += idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))
    assert hit.chunk_id == "a"
    assert hit.score == pytest.approx(expected, rel=1e-9)
