"""Shared builders and spies for the retrieval tests, plus their self-checks.

The retrieval tests have to prove *negatives* -- that a disabled stage did not
run -- so the components here are spies over the real implementations rather
than mocks that replace them: `SpyVectorIndex` records every `search` call and
then delegates to a genuine `MemoryVectorIndex`, so a test can assert both
"dense retrieval was never called" and "the results are what the real index
returns".

`BagEmbedder` is a deterministic bag-of-vocabulary embedder. It is not a
stand-in for MiniLM's semantics; it exists so a test can *predict* which chunk
the dense retriever will rank first, which no learned model lets you do.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from anyrag.core.types import Chunk, ChunkKind, Hit, RowRef
from anyrag.index import BM25Index, MemoryVectorIndex, tokenize

# --------------------------------------------------------------------------
# Chunks
# --------------------------------------------------------------------------


def make_chunk(
    chunk_id: str,
    text: str = "",
    *,
    source_id: str = "sqlite:test",
    kind: ChunkKind = ChunkKind.ROW,
    table: str | None = "customers",
    pk: str | None = None,
    **meta: Any,
) -> Chunk:
    full_meta: dict[str, Any] = {"content_hash": f"h-{chunk_id}"}
    if table is not None:
        full_meta["table"] = table
    full_meta.update(meta)
    refs: tuple[RowRef, ...] = ()
    if table is not None:
        refs = (RowRef(table=table, pk=pk if pk is not None else chunk_id),)
    return Chunk(
        chunk_id=chunk_id,
        source_id=source_id,
        kind=kind,
        text=text or f"row {chunk_id}",
        row_refs=refs,
        meta=full_meta,
    )


def make_hit(
    chunk_id: str,
    rank: int,
    score: float = 1.0,
    retriever: str = "dense",
    text: str = "",
    **chunk_kw: Any,
) -> Hit:
    return Hit(
        chunk=make_chunk(chunk_id, text, **chunk_kw),
        score=score,
        rank=rank,
        retriever=retriever,
    )


def ranked(
    ids: Sequence[str], retriever: str = "dense", start_score: float = 1.0
) -> list[Hit]:
    """A ranked list of hits for `ids`, ranks 1..n, descending scores."""
    return [
        make_hit(cid, rank=i + 1, score=start_score - 0.1 * i, retriever=retriever)
        for i, cid in enumerate(ids)
    ]


def ids(hits: Sequence[Hit]) -> list[str]:
    return [h.chunk_id for h in hits]


# --------------------------------------------------------------------------
# A predictable embedder
# --------------------------------------------------------------------------


class BagEmbedder:
    """Unit-norm bag-of-vocabulary vectors, so dense hits are predictable.

    Shares `anyrag.index.tokenize` with BM25 so the two retrievers see the same
    tokens; they still disagree, because BM25 weights by IDF and length while
    this is plain cosine over term presence.
    """

    def __init__(self, vocab: Sequence[str]) -> None:
        self.vocab = list(dict.fromkeys(vocab))
        self.dim = len(self.vocab)
        self._at = {term: i for i, term in enumerate(self.vocab)}
        self.query_calls: list[str] = []
        self.doc_calls = 0

    @classmethod
    def over(cls, *texts: str) -> "BagEmbedder":
        vocab: list[str] = []
        for text in texts:
            vocab.extend(tokenize(text))
        return cls(vocab or ["_"])

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for term in tokenize(text):
            i = self._at.get(term)
            if i is not None:
                vec[i] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.doc_calls += 1
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        self.query_calls.append(text)
        return self._vector(text)


# --------------------------------------------------------------------------
# Spies
# --------------------------------------------------------------------------


class SpyVectorIndex:
    """`MemoryVectorIndex` that records its `search` calls."""

    retriever = "dense"

    def __init__(self, inner: MemoryVectorIndex) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    def search(self, vector: Any, k: int = 10, flt: Any = None) -> list[Hit]:
        self.calls.append({"k": k, "flt": flt})
        return self.inner.search(vector, k=k, flt=flt)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class SpyLexicalIndex:
    """`BM25Index` that records its `search` calls."""

    retriever = "bm25"

    def __init__(self, inner: BM25Index) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def queries(self) -> list[str]:
        return [c["query"] for c in self.calls]

    def search(self, query: str, k: int = 10, flt: Any = None) -> list[Hit]:
        self.calls.append({"query": query, "k": k, "flt": flt})
        return self.inner.search(query, k=k, flt=flt)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class SpyReranker:
    """Records calls; by default reverses, so "it ran" is visible in the order."""

    name = "spy"

    def __init__(self, reverse: bool = True) -> None:
        self.calls: list[tuple[str, int, int]] = []
        self.reverse = reverse

    def rerank(self, query: str, hits: Sequence[Hit], k: int = 10) -> list[Hit]:
        self.calls.append((query, len(hits), k))
        out = list(reversed(hits)) if self.reverse else list(hits)
        return out[:k] if k > 0 else out


class SpyExpander:
    """Records calls and appends one fixed variant."""

    name = "spy"

    def __init__(self, variant: str = "extra variant") -> None:
        self.calls: list[str] = []
        self.variant = variant

    def expand(self, query: str) -> list[str]:
        self.calls.append(query)
        return [query, self.variant]


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------

#: Row chunks and schema cards written so that the two retrievers disagree:
#: the near-duplicate customer names differ by one token, which BM25 sees and a
#: bag-of-words cosine mostly does not.
CORPUS_TEXTS: tuple[tuple[str, str, ChunkKind, str], ...] = (
    ("cust:1", "customer Ahmed Al-Sayed email ahmed@example.com region Cairo "
               "signup_date is 2023-01-05", ChunkKind.ROW, "customers"),
    ("cust:2", "customer Ahmad Al Sayed email ahmad@example.com region Giza "
               "signup_date is 2023-04-02", ChunkKind.ROW, "customers"),
    ("cust:3", "customer Mona Hassan email mona@example.com region Cairo "
               "signup_date is 2024-02-11", ChunkKind.ROW, "customers"),
    ("cust:4", "customer Omar Farouk email omar@example.com region Alexandria "
               "signup_date is 2024-07-30", ChunkKind.ROW, "customers"),
    ("ord:1", "order 1001 customer Ahmed Al-Sayed amount is 1234.5 status shipped",
     ChunkKind.ROW, "orders"),
    ("ord:2", "order 1002 customer Mona Hassan amount is 42 status pending",
     ChunkKind.ROW, "orders"),
    ("card:customers", "table customers columns customer_id name email region "
                       "signup_date describing customers of the shop",
     ChunkKind.SCHEMA_CARD, "customers"),
    ("card:orders", "table orders columns order_id customer_id amount status "
                    "describing orders placed by customers",
     ChunkKind.SCHEMA_CARD, "orders"),
)


def corpus() -> list[Chunk]:
    return [
        make_chunk(cid, text, kind=kind, table=table, pk=cid.split(":")[-1])
        for cid, text, kind, table in CORPUS_TEXTS
    ]


def build_indexes(
    chunks: Sequence[Chunk] | None = None, *, extra_vocab: Sequence[str] = ()
) -> tuple[SpyVectorIndex, SpyLexicalIndex, BagEmbedder]:
    """Real indexes behind spies, loaded with `chunks` and a matching embedder."""
    chunks = list(chunks) if chunks is not None else corpus()
    embedder = BagEmbedder.over(*[c.text for c in chunks], *extra_vocab)
    vector = MemoryVectorIndex(dim=embedder.dim)
    if chunks:
        vector.upsert(chunks, embedder.embed([c.text for c in chunks]))
    lexical = BM25Index()
    lexical.upsert(chunks)
    return SpyVectorIndex(vector), SpyLexicalIndex(lexical), embedder


def many_chunks(
    n: int, *, kind: ChunkKind = ChunkKind.ROW, prefix: str = "r", table: str = "customers"
) -> list[Chunk]:
    """`n` chunks that all match a common query term, for k-completeness tests."""
    return [
        make_chunk(
            f"{prefix}{i:03d}",
            f"customer number {i} of the shop with email user{i}@example.com",
            kind=kind,
            table=table,
            pk=str(i),
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# Self-checks
# --------------------------------------------------------------------------


def test_make_chunk_defaults_are_coherent() -> None:
    chunk = make_chunk("c1")
    assert chunk.table == "customers"
    assert chunk.kind is ChunkKind.ROW
    assert chunk.row_refs == (RowRef(table="customers", pk="c1"),)


def test_bag_embedder_is_deterministic_and_unit_norm() -> None:
    emb = BagEmbedder.over("customer Ahmed Cairo", "order amount")
    a = emb.embed_query("customer Ahmed")
    b = emb.embed_query("customer Ahmed")
    assert np.array_equal(a, b)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-6


def test_bag_embedder_ignores_unknown_terms() -> None:
    emb = BagEmbedder.over("customer Ahmed")
    assert float(np.linalg.norm(emb.embed_query("zzz nothing here"))) == 0.0


def test_spies_delegate_to_the_real_index() -> None:
    vector, lexical, embedder = build_indexes()
    assert vector.count() == len(CORPUS_TEXTS)
    assert lexical.count() == len(CORPUS_TEXTS)
    hits = lexical.search("Ahmed", k=3)
    assert lexical.n_calls == 1
    assert hits and hits[0].retriever == "bm25"
    dense = vector.search(embedder.embed_query("customer Ahmed"), k=3)
    assert vector.n_calls == 1
    assert dense and dense[0].retriever == "dense"
