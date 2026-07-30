"""Okapi BM25, written out rather than imported.

`rank_bm25` is the obvious dependency and the wrong one here: its API takes the
whole tokenized corpus in the constructor and precomputes IDF over it, so adding
one document means rebuilding the entire index. Incremental upsert and
delete-by-source are part of the `LexicalIndex` contract, so the scoring is
implemented directly against an inverted index whose statistics -- document
frequencies, document lengths, average length -- are maintained per document.
Adding or removing a chunk touches only that chunk's terms; the corpus is never
re-tokenized.

Scoring is the standard Okapi form with the non-negative (Lucene) IDF::

    idf(t)   = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score(d) = sum_t qtf(t) * idf(t) * tf(t,d) * (k1 + 1)
                / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

`k1=1.5`, `b=0.75` by default, both constructor arguments. The `+1` inside the
log keeps IDF positive, so a term present in every document contributes ~0
instead of pushing scores negative -- with a small table corpus, where a column
name really can appear in every row chunk, negative IDF otherwise makes the
documents that *contain* the query term rank below the ones that do not.

Tokenization
------------
The corpus is bilingual, so tokenization must not be ASCII-shaped: `\\w+` under
`re.UNICODE` keeps Arabic script as words, and the only normalisation applied is
case folding, NFKC, and stripping Arabic diacritics/tatweel -- orthographic
noise that varies between rows describing the same thing. No stemming, no stop
words, no alef/ya folding: those change what a term *is*, and the eval compares
retrievers, not stemmers.

Rank convention
---------------
**`Hit.rank` is 1-based**, identical to `anyrag.index.memory`: best hit has
``rank == 1``. `retriever` is ``"bm25"``.
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter
from typing import Any, Iterator, Sequence

from anyrag.core.config import MetadataFilter
from anyrag.core.errors import IndexError_
from anyrag.core.types import Chunk, Hit

from .filters import PostingLists, chunk_to_dict
from .memory import (
    CHUNKS_FILE,
    FORMAT,
    MANIFEST_FILE,
    read_chunks,
    read_manifest,
)

__all__ = ["BM25Index", "tokenize"]

#: Extra BM25 state beside the shared chunk sidecar.
STATS_FILE = "bm25.json"

_WORD_RE = re.compile(r"\w+", re.UNICODE)

#: Arabic combining marks (harakat, shadda, sukun, superscript alef) and the
#: tatweel elongation character. Removing them is normalisation, not stemming:
#: the letters of the word are untouched.
_ARABIC_MARKS = re.compile(
    "[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]"
)


def tokenize(text: str) -> list[str]:
    """Lowercased, Unicode-aware word tokens. Works for Arabic and English."""
    if not text:
        return []
    text = unicodedata.normalize("NFKC", text).lower()
    text = _ARABIC_MARKS.sub("", text)
    return _WORD_RE.findall(text)


class BM25Index:
    """Incremental Okapi BM25 `LexicalIndex`.

    State kept per document: its unique terms and length. State kept per term:
    a posting map ``{chunk_id: term_frequency}``, from which document frequency
    is just ``len(postings[term])``. Both are updated per-document, so `upsert`
    and `delete_*` cost time proportional to the documents they touch.
    """

    #: Value of `Hit.retriever` on every hit this index produces.
    retriever = "bm25"

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        path: str | None = None,
        *,
        autoload: bool = False,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self._path = path
        self._chunks: dict[str, Chunk] = {}
        self._doc_terms: dict[str, tuple[str, ...]] = {}
        self._doc_len: dict[str, int] = {}
        self._postings: dict[str, dict[str, int]] = {}
        self._total_len = 0
        self._meta = PostingLists()
        if autoload and path and self.exists(path):
            self.load(path)

    # -- introspection ----------------------------------------------------

    @property
    def path(self) -> str | None:
        return self._path

    def count(self) -> int:
        return len(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._chunks

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def iter_chunks(self) -> Iterator[Chunk]:
        return iter(list(self._chunks.values()))

    def sources(self) -> tuple[str, ...]:
        return self._meta.sources

    @property
    def avgdl(self) -> float:
        n = len(self._chunks)
        return (self._total_len / n) if n else 0.0

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    def doc_freq(self, term: str) -> int:
        return len(self._postings.get(term, ()))

    @staticmethod
    def exists(path: str) -> bool:
        return os.path.isfile(os.path.join(path, MANIFEST_FILE))

    # -- writes -----------------------------------------------------------

    def upsert(self, chunks: Sequence[Chunk]) -> int:
        """Insert new chunk ids and replace existing ones.

        Returns the number of chunks written (inserted plus updated); `count()`
        is unchanged when every id was already present. Only the chunks passed in
        are tokenized -- the rest of the corpus is untouched, which is the whole
        reason this class exists instead of a `rank_bm25` wrapper.
        """
        written = 0
        for chunk in chunks:
            if chunk.chunk_id in self._chunks:
                self._remove(chunk.chunk_id)
            self._add(chunk)
            written += 1
        return written

    def _add(self, chunk: Chunk) -> None:
        tokens = tokenize(chunk.text)
        tf = Counter(tokens)
        cid = chunk.chunk_id
        for term, count in tf.items():
            self._postings.setdefault(term, {})[cid] = count
        self._chunks[cid] = chunk
        self._doc_terms[cid] = tuple(tf)
        self._doc_len[cid] = len(tokens)
        self._total_len += len(tokens)
        self._meta.add(chunk)

    def _remove(self, chunk_id: str) -> bool:
        chunk = self._chunks.pop(chunk_id, None)
        if chunk is None:
            return False
        for term in self._doc_terms.pop(chunk_id, ()):
            bucket = self._postings.get(term)
            if bucket is not None:
                bucket.pop(chunk_id, None)
                if not bucket:
                    # Drop the term entirely: an index that only ever grows its
                    # vocabulary is a memory leak with extra steps.
                    del self._postings[term]
        self._total_len -= self._doc_len.pop(chunk_id, 0)
        self._meta.discard(chunk)
        return True

    def delete_by_ids(self, chunk_ids: Sequence[str]) -> int:
        """Remove the given ids; returns how many were actually present."""
        removed = 0
        seen: set[str] = set()
        for cid in chunk_ids:
            if cid in seen:
                continue
            seen.add(cid)
            if self._remove(cid):
                removed += 1
        return removed

    def delete_by_source(self, source_id: str) -> int:
        """Remove every chunk from `source_id` via its posting list."""
        return self.delete_by_ids(sorted(self._meta.ids_for_source(source_id)))

    def clear(self) -> None:
        self._chunks.clear()
        self._doc_terms.clear()
        self._doc_len.clear()
        self._postings.clear()
        self._total_len = 0
        self._meta.clear()

    # -- search -----------------------------------------------------------

    def search(
        self, query: str, k: int = 10, flt: MetadataFilter | None = None
    ) -> list[Hit]:
        """Top-`k` chunks by BM25 score, filtered *before* ranking.

        The `MetadataFilter` restricts the candidate set rather than the result
        list, so a filtered search returns `k` hits whenever `k` matching chunks
        contain query terms. Hits are ordered by descending score with `rank`
        1..k (ties broken by `chunk_id`) and ``retriever == "bm25"``. An empty
        index, an empty query, or a query whose terms are all unknown returns
        `[]` rather than raising.
        """
        if k <= 0 or not self._chunks:
            return []
        q_tf = Counter(tokenize(query))
        if not q_tf:
            return []

        allowed = self._allowed_ids(flt)
        if allowed is not None and not allowed:
            return []

        n = len(self._chunks)
        avgdl = self.avgdl or 1.0
        k1, b = self.k1, self.b
        scores: dict[str, float] = {}
        for term, qtf in q_tf.items():
            bucket = self._postings.get(term)
            if not bucket:
                continue
            df = len(bucket)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            if idf <= 0.0:
                continue
            weight = qtf * idf
            # Walk whichever side is smaller: a common term's posting list can
            # be the whole corpus, while a table pre-filter is often a handful
            # of chunks. Iterating the larger one is the difference between a
            # filtered ablation cell being fast and being pointless.
            if allowed is not None and len(allowed) < len(bucket):
                pairs = (
                    (cid, bucket[cid]) for cid in allowed if cid in bucket
                )
            elif allowed is None:
                pairs = iter(bucket.items())
            else:
                pairs = ((cid, tf) for cid, tf in bucket.items() if cid in allowed)
            for cid, tf in pairs:
                dl = self._doc_len.get(cid, 0)
                denom = tf + k1 * (1.0 - b + b * dl / avgdl)
                scores[cid] = scores.get(cid, 0.0) + weight * tf * (k1 + 1.0) / denom

        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [
            Hit(
                chunk=self._chunks[cid],
                score=float(score),
                rank=rank,
                retriever=self.retriever,
            )
            for rank, (cid, score) in enumerate(ranked, start=1)
        ]

    def _allowed_ids(self, flt: MetadataFilter | None) -> set[str] | None:
        """Ids surviving `flt`, or None meaning "every indexed chunk"."""
        ids, needs_verify = self._meta.candidates(flt)
        if ids is None:
            if not needs_verify:
                return None
            assert flt is not None
            return {cid for cid, c in self._chunks.items() if flt.matches(c)}
        if not needs_verify:
            return ids
        assert flt is not None
        return {cid for cid in ids if flt.matches(self._chunks[cid])}

    # -- persistence ------------------------------------------------------

    def persist(self, path: str | None = None) -> None:
        """Write chunks plus the inverted index to a directory.

        The postings are written out rather than recomputed on load, so a reload
        never re-tokenizes the corpus and the restored index is bit-identical to
        the one that was saved.
        """
        target = self._resolve(path)
        os.makedirs(target, exist_ok=True)
        order = sorted(self._chunks)
        with open(os.path.join(target, CHUNKS_FILE), "w", encoding="utf-8") as fh:
            for cid in order:
                fh.write(
                    json.dumps(chunk_to_dict(self._chunks[cid]), ensure_ascii=False)
                    + "\n"
                )
        stats = {
            "k1": self.k1,
            "b": self.b,
            "doc_len": {cid: self._doc_len[cid] for cid in order},
            "postings": {
                term: bucket for term, bucket in sorted(self._postings.items())
            },
        }
        with open(os.path.join(target, STATS_FILE), "w", encoding="utf-8") as fh:
            json.dump(stats, fh, ensure_ascii=False)
        manifest = {
            "format": FORMAT,
            "kind": "bm25",
            "count": len(self._chunks),
            "vocabulary": len(self._postings),
        }
        with open(os.path.join(target, MANIFEST_FILE), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
        self._path = target

    def load(self, path: str | None = None) -> None:
        """Replace this index's contents with the ones persisted at `path`."""
        target = self._resolve(path)
        read_manifest(target, expect_kind="bm25")
        stats_path = os.path.join(target, STATS_FILE)
        if not os.path.exists(stats_path):
            raise IndexError_(f"no {STATS_FILE} in index directory {target!r}")
        with open(stats_path, encoding="utf-8") as fh:
            stats: dict[str, Any] = json.load(fh)
        chunks = read_chunks(os.path.join(target, CHUNKS_FILE))

        self.clear()
        self.k1 = float(stats.get("k1", self.k1))
        self.b = float(stats.get("b", self.b))
        doc_len: dict[str, int] = {
            cid: int(v) for cid, v in (stats.get("doc_len") or {}).items()
        }
        postings: dict[str, dict[str, int]] = {
            term: {cid: int(tf) for cid, tf in bucket.items()}
            for term, bucket in (stats.get("postings") or {}).items()
        }
        doc_terms: dict[str, list[str]] = {}
        for term, bucket in postings.items():
            for cid in bucket:
                doc_terms.setdefault(cid, []).append(term)

        for chunk in chunks:
            cid = chunk.chunk_id
            if cid not in doc_len:
                raise IndexError_(
                    f"index at {target!r} is inconsistent: no length recorded "
                    f"for chunk {cid!r}"
                )
            self._chunks[cid] = chunk
            self._doc_terms[cid] = tuple(doc_terms.get(cid, ()))
            self._doc_len[cid] = doc_len[cid]
            self._total_len += doc_len[cid]
            self._meta.add(chunk)
        self._postings = postings
        self._path = target

    def _resolve(self, path: str | None) -> str:
        target = path or self._path
        if not target:
            raise IndexError_(
                "BM25Index has no path: pass one to persist()/load() or set it "
                "on the constructor"
            )
        return target

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"BM25Index(count={self.count()}, vocab={self.vocabulary_size}, "
            f"k1={self.k1}, b={self.b})"
        )
