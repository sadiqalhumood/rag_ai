"""A fake `DataSource` for ingest tests.

`anyrag.sources` is written by another agent and may be absent or incomplete.
Ingest must not depend on it: the pipeline talks to the `DataSource` Protocol,
so a dozen lines of dict-backed fake exercise exactly the same code path a real
adapter will. This module holds no tests of its own; it is imported by the
`test_ingest_*` files.
"""

from __future__ import annotations

from typing import Any, Iterator

from anyrag.core.errors import SchemaError
from anyrag.core.types import (
    ColumnProfile,
    ColumnRole,
    ColumnSchema,
    QueryResult,
    Row,
    TableProfile,
    TableRef,
    TableSchema,
)

CUSTOMERS = TableRef("customers")
ORDERS = TableRef("orders")

#: Deliberately mixed: Arabic free text, an explicit NULL, a boolean, a date.
CUSTOMER_ROWS: list[dict[str, Any]] = [
    {
        "customer_id": 41,
        "name": "Ahmed Al-Sayed",
        "region": "EMEA",
        "signup_date": "2023-04-02",
        "notes": "Prefers email contact.",
        "active": True,
    },
    {
        "customer_id": 42,
        "name": "سارة عبد الله",
        "region": None,
        "signup_date": "2021-11-30",
        "notes": "ملاحظات باللغة العربية عن العميل.",
        "active": False,
    },
    {
        "customer_id": 43,
        "name": "Mei Chen",
        "region": "APAC",
        "signup_date": "2024-01-15",
        "notes": None,
        "active": True,
    },
]

#: Composite primary key, to pin the joined-pk behaviour.
ORDER_ROWS: list[dict[str, Any]] = [
    {"order_id": 900, "line_no": 1, "customer_id": 41, "amount": 120.0, "sku": "A-1"},
    {"order_id": 900, "line_no": 2, "customer_id": 41, "amount": 80.5, "sku": "B-2"},
    {"order_id": 901, "line_no": 1, "customer_id": 43, "amount": 12.0, "sku": "A-1"},
]

CUSTOMER_SCHEMA = TableSchema(
    table=CUSTOMERS,
    columns=(
        ColumnSchema("customer_id", "INTEGER", nullable=False, is_primary_key=True),
        ColumnSchema("name", "TEXT"),
        ColumnSchema("region", "TEXT"),
        ColumnSchema("signup_date", "DATE"),
        ColumnSchema("notes", "TEXT"),
        ColumnSchema("active", "BOOLEAN", nullable=False),
    ),
    primary_key=("customer_id",),
)

ORDER_SCHEMA = TableSchema(
    table=ORDERS,
    columns=(
        ColumnSchema("order_id", "INTEGER", nullable=False, is_primary_key=True),
        ColumnSchema("line_no", "INTEGER", nullable=False, is_primary_key=True),
        ColumnSchema(
            "customer_id", "INTEGER", nullable=False, references="customers.customer_id"
        ),
        ColumnSchema("amount", "NUMERIC"),
        ColumnSchema("sku", "TEXT"),
    ),
    primary_key=("order_id", "line_no"),
)

CUSTOMER_PROFILE = TableProfile(
    table=CUSTOMERS,
    row_count=3,
    columns={
        "customer_id": ColumnProfile(
            "customer_id", ColumnRole.ID, 3, 0, 3, samples=(41, 42, 43),
            min_value=41, max_value=43,
        ),
        "name": ColumnProfile(
            "name", ColumnRole.FREE_TEXT, 3, 0, 3,
            samples=("Ahmed Al-Sayed", "سارة عبد الله"), mean_length=13.0,
        ),
        "region": ColumnProfile(
            "region", ColumnRole.CATEGORICAL, 3, 1, 2,
            samples=("EMEA", "APAC"), categories=("APAC", "EMEA"),
        ),
        "signup_date": ColumnProfile(
            "signup_date", ColumnRole.DATE, 3, 0, 3,
            min_value="2021-11-30", max_value="2024-01-15",
        ),
        "notes": ColumnProfile("notes", ColumnRole.FREE_TEXT, 3, 1, 2),
        "active": ColumnProfile(
            "active", ColumnRole.BOOLEAN, 3, 0, 2, samples=(True, False)
        ),
    },
)

ORDER_PROFILE = TableProfile(
    table=ORDERS,
    row_count=3,
    columns={
        "order_id": ColumnProfile("order_id", ColumnRole.ID, 3, 0, 2),
        "line_no": ColumnProfile("line_no", ColumnRole.NUMERIC, 3, 0, 2),
        "customer_id": ColumnProfile("customer_id", ColumnRole.ID, 3, 0, 2),
        "amount": ColumnProfile(
            "amount", ColumnRole.NUMERIC, 3, 0, 3, min_value=12.0, max_value=120.0
        ),
        "sku": ColumnProfile(
            "sku", ColumnRole.CATEGORICAL, 3, 0, 2, categories=("A-1", "B-2")
        ),
    },
)


class FakeSource:
    """Dict-backed `DataSource`. Read-only, deterministic, no I/O."""

    def __init__(
        self,
        source_id: str = "fake://demo",
        *,
        rows: dict[str, list[dict[str, Any]]] | None = None,
        profiles: bool = True,
    ) -> None:
        self.source_id = source_id
        self._rows = rows if rows is not None else {
            "customers": [dict(r) for r in CUSTOMER_ROWS],
            "orders": [dict(r) for r in ORDER_ROWS],
        }
        self._schemas = {"customers": CUSTOMER_SCHEMA, "orders": ORDER_SCHEMA}
        self._profiles = {"customers": CUSTOMER_PROFILE, "orders": ORDER_PROFILE}
        self.has_profiles = profiles
        self.closed = False

    def tables(self) -> list[TableRef]:
        return [self._schemas[name].table for name in sorted(self._rows)]

    def schema(self, table: TableRef) -> TableSchema:
        try:
            return self._schemas[table.name]
        except KeyError:
            raise SchemaError(f"no table {table.name!r}") from None

    def profile(self, table: TableRef) -> TableProfile:
        if not self.has_profiles:
            raise SchemaError("profiling disabled on this fake source")
        return self._profiles[table.name]

    def iter_rows(self, table: TableRef, batch_size: int = 1000) -> Iterator[list[Row]]:
        rows = self._rows[table.name]
        for start in range(0, len(rows), batch_size):
            yield [dict(r) for r in rows[start : start + batch_size]]

    def execute_readonly(self, sql: str, max_rows: int = 1000) -> QueryResult:
        return QueryResult(columns=(), rows=(), sql=sql)

    def close(self) -> None:
        self.closed = True

    # -- test helpers -----------------------------------------------------

    def edit(self, table: str, pk_column: str, pk_value: Any, **changes: Any) -> None:
        """Mutate a row in place, to test content-hash change detection."""
        for row in self._rows[table]:
            if row[pk_column] == pk_value:
                row.update(changes)
                return
        raise AssertionError(f"no row {pk_value!r} in {table}")
