"""Row -> natural-language sentence.

A row is a mapping; an embedder wants prose. The gap between them is where most
table-RAG systems quietly lose their recall, so the choices here are explicit:

* **Column names stay in the text.** `signup_date is 2023-04-02`, not
  `Signup Date: 2023-04-02`. The question "when did Ahmed sign up" has to match
  lexically as well as densely, and the schema vocabulary is what the user
  borrows when they ask.
* **Nulls are stated, not omitted.** "region is not recorded" is a retrievable
  fact; a missing clause is not. "Which customers have no region?" is only
  answerable if absence was written down.
* **Non-ASCII text is passed through untouched.** No transliteration, no
  normalisation, no `ensure_ascii`. Arabic values must survive byte-for-byte or
  every Arabic question in the eval fails for an encoding reason rather than a
  retrieval one.
* **ID columns are dropped from the prose but kept in `meta`.** A surrogate key
  contributes nothing to a sentence's meaning and dilutes its embedding, but it
  is still needed for joins and filters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Mapping, Sequence

from ..core.tokenizer import Tokenizer, get_tokenizer
from ..core.types import (
    Chunk,
    ChunkKind,
    ColumnRole,
    Row,
    RowRef,
    TableProfile,
    TableSchema,
)
from .ids import content_hash, pk_for_row, row_chunk_id
from .truncate import split_with_overlap, truncate_field

#: Value rendered when a column is NULL. Phrased as a statement so it embeds and
#: matches like one.
NULL_PHRASE = "not recorded"
EMPTY_PHRASE = "empty"

_ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))


@dataclass(frozen=True)
class RowSerializerConfig:
    """Knobs for row verbalisation. Defaults are the eval's operating point."""

    #: Ceiling for a whole serialized row before it is split into parts.
    max_chunk_tokens: int = 512
    #: Ceiling for any single field value inside a row.
    max_field_tokens: int = 96
    #: Tokens shared between consecutive parts of a split row.
    overlap_tokens: int = 48
    #: Drop ColumnRole.ID columns from the prose (they stay in meta).
    skip_id_columns: bool = True
    #: Cap on how many columns get verbalised, widest tables first. None = all.
    max_columns: int | None = None
    null_phrase: str = NULL_PHRASE


@dataclass(frozen=True)
class RowText:
    """The verbalised row, before it is split into chunks."""

    text: str
    pk: str
    columns: tuple[str, ...]
    truncated_fields: tuple[str, ...] = ()
    id_columns: Mapping[str, str] = field(default_factory=dict)
    date_min: str | None = None
    date_max: str | None = None
    lang: str = "en"


def _has_arabic(text: str) -> bool:
    return any(
        any(lo <= ord(ch) <= hi for lo, hi in _ARABIC_RANGES) for ch in text
    )


def _detect_lang(text: str) -> str:
    """Crude script tag, not language identification.

    It exists so a downstream filter can ask for "the Arabic rows" without
    pulling in a language-id dependency; it claims nothing more than which
    scripts are present.
    """
    arabic = _has_arabic(text)
    latin = any("a" <= ch.lower() <= "z" for ch in text)
    if arabic and latin:
        return "mixed"
    if arabic:
        return "ar"
    return "en"


def format_value(value: Any, role: ColumnRole = ColumnRole.UNKNOWN) -> str:
    """Render one cell for prose. Never returns the literal 'None'."""
    if value is None:
        return NULL_PHRASE
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value)
    if isinstance(value, bytes):
        return f"<{len(value)} bytes of binary data>"
    if isinstance(value, (list, tuple, set)):
        inner = ", ".join(format_value(v, role) for v in value)
        return inner if inner else EMPTY_PHRASE
    if isinstance(value, Mapping):
        return ", ".join(f"{k} {format_value(v, role)}" for k, v in value.items())
    text = str(value)
    if not text.strip():
        return EMPTY_PHRASE
    return text


def _iso_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    text = str(value).strip()
    return text or None


class RowSerializer:
    """Verbalises rows of one table into `Chunk`s.

    Constructed once per table so schema lookups, role lookups and the pk column
    list are resolved once rather than per row -- ingestion walks millions of
    rows and this is the inner loop.
    """

    def __init__(
        self,
        *,
        source_id: str,
        schema: TableSchema,
        profile: TableProfile | None = None,
        config: RowSerializerConfig | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.source_id = source_id
        self.schema = schema
        self.profile = profile
        self.config = config or RowSerializerConfig()
        self.tokenizer = tokenizer or get_tokenizer()
        self.table = schema.table
        self.table_name = schema.table.qualified
        self.pk_columns = self._resolve_pk_columns()
        self._roles = {c.name: self._role_of(c.name) for c in schema.columns}
        self._date_columns = tuple(
            c.name for c in schema.columns if self._roles[c.name] is ColumnRole.DATE
        )

    # -- setup ------------------------------------------------------------

    def _resolve_pk_columns(self) -> tuple[str, ...]:
        if self.schema.primary_key:
            return tuple(self.schema.primary_key)
        declared = tuple(c.name for c in self.schema.columns if c.is_primary_key)
        if declared:
            return declared
        for candidate in ("id", "rowid", f"{self.schema.table.name}_id"):
            if candidate in self.schema.column_names:
                return (candidate,)
        return ()

    def _role_of(self, column: str) -> ColumnRole:
        if self.profile is not None:
            role = self.profile.role(column)
            if role is not ColumnRole.UNKNOWN:
                return role
        # No profile: fall back to the declared type. Coarse, but it is only
        # used for phrasing and for date min/max, never for correctness.
        try:
            type_name = self.schema.column(column).type_name.lower()
        except Exception:  # column not in schema (extra key in the row)
            return ColumnRole.UNKNOWN
        if "date" in type_name or "time" in type_name:
            return ColumnRole.DATE
        if "bool" in type_name:
            return ColumnRole.BOOLEAN
        if any(t in type_name for t in ("int", "float", "double", "numeric", "real", "decimal")):
            return ColumnRole.NUMERIC
        return ColumnRole.UNKNOWN

    def _is_foreign_key(self, column: str) -> bool:
        try:
            return bool(self.schema.column(column).references)
        except Exception:
            return False

    def _is_skipped(self, column: str) -> bool:
        if not self.config.skip_id_columns:
            return False
        if column in self.pk_columns:
            # The pk is already in the header ("customers record 41"); repeating
            # it as a clause adds tokens and no information.
            return True
        if self._roles.get(column, ColumnRole.UNKNOWN) is not ColumnRole.ID:
            return False
        # A declared foreign key is not surrogate noise -- it is the only thing
        # in the row that says which customer this order belongs to, and
        # dropping it makes every join question unanswerable from row chunks.
        return not self._is_foreign_key(column)

    # -- verbalisation ----------------------------------------------------

    def pk_of(self, row: Row, ordinal: int | None = None) -> str:
        if self.pk_columns:
            return pk_for_row(row, self.pk_columns)
        # No declared key anywhere: fall back to position. Stable only for a
        # source that iterates deterministically, which is why it is last.
        return "" if ordinal is None else str(ordinal)

    def row_text(self, row: Row, *, ordinal: int | None = None) -> RowText:
        cfg = self.config
        pk = self.pk_of(row, ordinal)

        # Schema order first (stable, meaningful), then any extra keys the row
        # carries that the schema did not declare.
        ordered = [c for c in self.schema.column_names if c in row]
        ordered += [c for c in row if c not in self.schema.column_names]

        clauses: list[str] = []
        used: list[str] = []
        truncated: list[str] = []
        id_columns: dict[str, str] = {}
        dates: list[str] = []

        for column in ordered:
            value = row[column]
            role = self._roles.get(column, ColumnRole.UNKNOWN)

            if role is ColumnRole.DATE or column in self._date_columns:
                iso = _iso_date(value)
                if iso:
                    dates.append(iso)

            if self._is_skipped(column):
                if value is not None:
                    id_columns[column] = format_value(value, role)
                continue

            if cfg.max_columns is not None and len(used) >= cfg.max_columns:
                continue

            if value is None:
                clauses.append(f"{column} is {cfg.null_phrase}")
                used.append(column)
                continue

            rendered = format_value(value, role)
            cut = truncate_field(
                rendered, cfg.max_field_tokens, tokenizer=self.tokenizer
            )
            if cut.truncated:
                truncated.append(column)
            clauses.append(f"{column} is {cut.text}")
            used.append(column)

        header = f"{self.table_name} record {pk}" if pk else f"{self.table_name} record"
        text = f"{header}: " + "; ".join(clauses) + "."
        return RowText(
            text=text,
            pk=pk,
            columns=tuple(used),
            truncated_fields=tuple(truncated),
            id_columns=id_columns,
            date_min=min(dates) if dates else None,
            date_max=max(dates) if dates else None,
            lang=_detect_lang(text),
        )

    # -- chunking ---------------------------------------------------------

    def chunks(self, row: Row, *, ordinal: int | None = None) -> list[Chunk]:
        """One row -> one chunk, or several overlapping parts if oversized.

        All parts share the same `row_refs`, so retrieving any part of a row
        credits the same gold row in the eval.
        """
        rendered = self.row_text(row, ordinal=ordinal)
        parts = split_with_overlap(
            rendered.text,
            self.config.max_chunk_tokens,
            self.config.overlap_tokens,
            tokenizer=self.tokenizer,
        )
        refs = (RowRef(table=self.table_name, pk=rendered.pk),)
        n_parts = len(parts)
        out: list[Chunk] = []
        for index, part in enumerate(parts):
            meta: dict[str, Any] = {
                "table": self.table_name,
                "source_id": self.source_id,
                "kind": ChunkKind.ROW.value,
                "pk": rendered.pk,
                "pk_columns": tuple(self.pk_columns),
                "columns": rendered.columns,
                "part_index": index,
                "n_parts": n_parts,
                "content_hash": content_hash(part),
                "truncated_fields": rendered.truncated_fields,
                "lang": rendered.lang,
            }
            if rendered.id_columns:
                meta["id_columns"] = dict(rendered.id_columns)
            if rendered.date_min is not None:
                meta["date_min"] = rendered.date_min
                meta["date_max"] = rendered.date_max
            out.append(
                Chunk(
                    chunk_id=row_chunk_id(
                        source_id=self.source_id,
                        table=self.table,
                        pk=rendered.pk,
                        part_index=index,
                    ),
                    source_id=self.source_id,
                    kind=ChunkKind.ROW,
                    text=part,
                    row_refs=refs,
                    meta=meta,
                )
            )
        return out

    def chunk_rows(
        self, rows: Sequence[Row], *, start_ordinal: int = 0
    ) -> list[Chunk]:
        out: list[Chunk] = []
        for offset, row in enumerate(rows):
            out.extend(self.chunks(row, ordinal=start_ordinal + offset))
        return out
