"""Table -> schema card.

Row chunks answer "what did Ahmed buy". They cannot answer "what columns does
the customers table have", because no row ever states its own schema -- the
column names appear in every row and therefore discriminate between none of
them. That class of question is why a schema card exists: one chunk per table,
describing the table itself.

A card is written as prose rather than as a DDL dump because it has to be
retrieved by an embedder and a BM25 index, not parsed by a compiler. It states
column names, declared types, inferred roles, cardinality, null fractions,
example values and foreign keys -- the same facts a DDL dump has, plus the
statistical ones it does not.

Schema cards carry `row_refs=()`. They describe a table, not rows, so crediting
them with row provenance would corrupt the eval's retrieval scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.tokenizer import Tokenizer, get_tokenizer
from ..core.types import (
    Chunk,
    ChunkKind,
    ColumnProfile,
    ColumnRole,
    TableProfile,
    TableSchema,
)
from .ids import content_hash, schema_card_id
from .row_serializer import format_value
from .truncate import split_with_overlap, truncate_field


@dataclass(frozen=True)
class SchemaCardConfig:
    max_chunk_tokens: int = 512
    overlap_tokens: int = 48
    #: Example values quoted per column.
    max_samples: int = 5
    #: Categorical vocabularies longer than this are elided with a count.
    max_categories: int = 12
    #: Ceiling for one rendered sample value (a free-text column's samples can
    #: be arbitrarily long).
    max_sample_tokens: int = 24


def _pct(fraction: float) -> str:
    if fraction <= 0:
        return "0%"
    if fraction < 0.001:
        return "<0.1%"
    return f"{fraction * 100:.1f}%".replace(".0%", "%")


def _describe_column(
    name: str,
    type_name: str,
    *,
    is_pk: bool,
    nullable: bool,
    references: str | None,
    prof: ColumnProfile | None,
    cfg: SchemaCardConfig,
    tokenizer: Tokenizer,
) -> str:
    bits: list[str] = [f"type {type_name}"]
    role = prof.role if prof is not None else ColumnRole.UNKNOWN
    if role is not ColumnRole.UNKNOWN:
        bits.append(f"role {role.value}")
    if is_pk:
        bits.append("primary key")
    bits.append("nullable" if nullable else "not nullable")
    if references:
        bits.append(f"foreign key to {references}")

    if prof is not None:
        if prof.total_count:
            bits.append(f"{prof.distinct_count} distinct values")
            bits.append(f"{_pct(prof.null_fraction)} missing")
        if prof.min_value is not None or prof.max_value is not None:
            lo = format_value(prof.min_value, role)
            hi = format_value(prof.max_value, role)
            bits.append(f"ranges from {lo} to {hi}")
        if prof.mean_length is not None:
            bits.append(f"average length {prof.mean_length:.0f} characters")

    line = f"- {name}: " + ", ".join(bits) + "."

    if prof is not None and prof.categories:
        cats = list(prof.categories)
        shown = cats[: cfg.max_categories]
        rendered = ", ".join(str(c) for c in shown)
        if len(cats) > len(shown):
            rendered += f", and {len(cats) - len(shown)} more"
        line += f" Values: {rendered}."
    if prof is not None and prof.samples:
        samples = []
        for raw in prof.samples[: cfg.max_samples]:
            text = format_value(raw, role)
            samples.append(
                truncate_field(text, cfg.max_sample_tokens, tokenizer=tokenizer).text
            )
        if samples:
            line += " Examples: " + "; ".join(samples) + "."
    return line


def schema_card_text(
    schema: TableSchema,
    profile: TableProfile | None = None,
    *,
    config: SchemaCardConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> str:
    cfg = config or SchemaCardConfig()
    tok = tokenizer or get_tokenizer()
    table = schema.table.qualified
    names = schema.column_names

    lines: list[str] = []
    row_count = profile.row_count if profile is not None else None
    if row_count is not None:
        lines.append(
            f"Schema card for table {table}. "
            f"The {table} table has {len(names)} columns and {row_count} rows."
        )
    else:
        lines.append(
            f"Schema card for table {table}. "
            f"The {table} table has {len(names)} columns."
        )
    # A single flat list of names, so "what columns does X have" matches
    # lexically without the reader parsing the per-column detail below.
    lines.append(f"Columns of {table}: " + ", ".join(names) + ".")

    pk = schema.primary_key or tuple(c.name for c in schema.columns if c.is_primary_key)
    if pk:
        lines.append(f"Primary key: {', '.join(pk)}.")
    else:
        lines.append("Primary key: none declared.")

    lines.append("Column details:")
    prof_cols = profile.columns if profile is not None else {}
    for col in schema.columns:
        lines.append(
            _describe_column(
                col.name,
                col.type_name,
                is_pk=col.is_primary_key or col.name in pk,
                nullable=col.nullable,
                references=col.references,
                prof=prof_cols.get(col.name),
                cfg=cfg,
                tokenizer=tok,
            )
        )

    fks = [
        f"{table}.{c.name} references {c.references}"
        for c in schema.columns
        if c.references
    ]
    lines.append(
        "Foreign keys: " + ("; ".join(fks) + "." if fks else "none.")
    )
    return "\n".join(lines)


def build_schema_card(
    *,
    source_id: str,
    schema: TableSchema,
    profile: TableProfile | None = None,
    config: SchemaCardConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> list[Chunk]:
    """One card per table, split into parts only if a very wide table demands it."""
    cfg = config or SchemaCardConfig()
    tok = tokenizer or get_tokenizer()
    text = schema_card_text(schema, profile, config=cfg, tokenizer=tok)
    parts = split_with_overlap(
        text, cfg.max_chunk_tokens, cfg.overlap_tokens, tokenizer=tok
    )
    table = schema.table
    pk = schema.primary_key or tuple(c.name for c in schema.columns if c.is_primary_key)
    prof_cols = profile.columns if profile is not None else {}

    chunks: list[Chunk] = []
    for index, part in enumerate(parts):
        meta: dict[str, Any] = {
            "table": table.qualified,
            "source_id": source_id,
            "kind": ChunkKind.SCHEMA_CARD.value,
            "columns": schema.column_names,
            "column_types": {c.name: c.type_name for c in schema.columns},
            "column_roles": {
                name: (prof.role.value if prof else ColumnRole.UNKNOWN.value)
                for name, prof in (
                    (c.name, prof_cols.get(c.name)) for c in schema.columns
                )
            },
            "primary_key": tuple(pk),
            "foreign_keys": tuple(
                f"{c.name}->{c.references}" for c in schema.columns if c.references
            ),
            "part_index": index,
            "n_parts": len(parts),
            "content_hash": content_hash(part),
        }
        if profile is not None:
            meta["row_count"] = profile.row_count
        chunks.append(
            Chunk(
                chunk_id=schema_card_id(
                    source_id=source_id, table=table, part_index=index
                ),
                source_id=source_id,
                kind=ChunkKind.SCHEMA_CARD,
                text=part,
                row_refs=(),
                meta=meta,
            )
        )
    return chunks
