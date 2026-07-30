"""Shared fixture data for the source-adapter tests.

Contains no tests. It lives under `tests/` as a `test_sources_*` module so it
stays inside source-eng's file ownership, and so `pytest tests/test_sources_*`
picks up everything the adapters need in one glob.

The same logical dataset is materialised into SQLite (dynamic types, dates as
ISO strings, booleans as 0/1) and into PostgreSQL (native DATE, BOOLEAN,
NUMERIC, BYTEA). That is the point: role inference must reach the *same*
conclusions from two very different type systems, which is the only real test
of whether the abstraction holds.

Covered on purpose: composite primary key, composite foreign key, nullable
columns, a categorical column, a free-text column, a date column, Arabic text,
near-duplicate names, a binary column (the honest UNKNOWN case), and a table
with no primary key at all.
"""

from __future__ import annotations

import random
from typing import Any

N_CUSTOMERS = 120
LINES_PER_ORDER = 2

COUNTRIES = ["EG", "SA", "AE", "MA", "JO"]
STATUSES = ["pending", "shipped", "delivered", "cancelled"]

#: Deliberately includes near-duplicate spellings and Arabic script.
BASE_NAMES = [
    "Ahmad Al-Sayed",
    "Ahmed Al-Sayed",
    "Ahmad Al Sayed",
    "Fatima Al-Zahra",
    "Fatimah Al-Zahra",
    "أحمد السيد",
    "فاطمة الزهراء",
    "محمد عبد الله",
    "Mohammed Abdullah",
    "Mohamed Abdullah",
    "Layla Haddad",
    "Laila Haddad",
    "Omar Khalil",
    "Umar Khalil",
    "Sara Mansour",
    "Sarah Mansour",
    "Youssef Nassar",
    "Yusuf Nassar",
    "Nour El-Din",
    "Noor El-Din",
    "Karim Boutros",
    "Rania Saleh",
    "Tariq Bin Zayed",
    "Hala Darwish",
]

NOTE_TEMPLATES = [
    "Long-standing wholesale account; prefers consolidated monthly invoices and "
    "has asked twice for delivery windows to avoid the afternoon heat.",
    "Escalated a damaged-shipment claim in the last quarter. Resolution was a "
    "partial refund plus expedited replacement on the following order.",
    "عميل منذ فترة طويلة ويفضل التواصل باللغة العربية، وقد طلب تغيير عنوان "
    "الشحن إلى المستودع الجديد في المنطقة الصناعية.",
    "Switched from monthly to quarterly ordering after a procurement review; "
    "volumes are lumpier but the annual total is roughly unchanged.",
    "Requested that all correspondence go through the finance contact rather "
    "than the original buyer, following an internal reorganisation.",
]


def _rng() -> random.Random:
    # Fixed seed: every assertion about counts and cardinality below depends
    # on this data being byte-identical on every run and on every machine.
    return random.Random(20240517)


def customer_rows() -> list[dict[str, Any]]:
    rng = _rng()
    rows: list[dict[str, Any]] = []
    for i in range(N_CUSTOMERS):
        base = BASE_NAMES[i % len(BASE_NAMES)]
        name = base if i < len(BASE_NAMES) else f"{base} {i // len(BASE_NAMES) + 1}"
        # Every 12th customer has no country: gives the null-fraction test
        # something real to measure.
        country = None if i % 12 == 0 else COUNTRIES[i % len(COUNTRIES)]
        day = 1 + (i % 27)
        month = 1 + (i % 12)
        rows.append(
            {
                "customer_id": i + 1,
                "full_name": name,
                "country": country,
                "signup_date": f"2023-{month:02d}-{day:02d}",
                "is_active": 1 if i % 3 else 0,
                "notes": f"{NOTE_TEMPLATES[i % len(NOTE_TEMPLATES)]} (ref {i + 1})",
                "avatar": bytes([i % 256, (i * 7) % 256]) if i % 40 == 0 else None,
            }
        )
    rng.shuffle(rows)
    rows.sort(key=lambda r: r["customer_id"])
    return rows


def order_rows() -> list[dict[str, Any]]:
    rng = _rng()
    rows: list[dict[str, Any]] = []
    for i in range(N_CUSTOMERS):
        order_id = 1000 + i
        for line in range(1, LINES_PER_ORDER + 1):
            rows.append(
                {
                    "order_id": order_id,
                    "line_no": line,
                    "customer_id": i + 1,
                    "status": STATUSES[(i + line) % len(STATUSES)],
                    "amount": round(rng.uniform(5.0, 4000.0), 2),
                    "ordered_at": f"2024-{1 + (i % 12):02d}-{1 + (i % 27):02d}"
                    f"T{(i % 24):02d}:{(i * 7) % 60:02d}:00",
                }
            )
    return rows


def shipment_rows() -> list[dict[str, Any]]:
    """Rows whose FK into `orders` is composite -- the case that breaks the
    naive information_schema foreign-key join."""
    return [
        {
            "shipment_id": n + 1,
            "order_id": 1000 + n,
            "line_no": 1,
            "carrier": ["aramex", "dhl", "smsa"][n % 3],
        }
        for n in range(30)
    ]


def event_rows() -> list[dict[str, Any]]:
    """A table with no primary key, to exercise the surrogate-key path."""
    return [
        {"kind": ["login", "view", "purchase"][n % 3], "at": f"2024-06-{1 + n % 28:02d}"}
        for n in range(15)
    ]


# --------------------------------------------------------------------------
# SQLite materialisation
# --------------------------------------------------------------------------

SQLITE_DDL = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    full_name   TEXT NOT NULL,
    country     TEXT,
    signup_date TEXT,
    is_active   INTEGER,
    notes       TEXT,
    avatar      BLOB
);

CREATE TABLE orders (
    order_id    INTEGER NOT NULL,
    line_no     INTEGER NOT NULL,
    -- Deliberately column-less: `REFERENCES customers` targets the referenced
    -- table's primary key positionally, which adapters have to resolve.
    customer_id INTEGER REFERENCES customers,
    status      TEXT,
    amount      REAL,
    ordered_at  TEXT,
    PRIMARY KEY (order_id, line_no)
);

CREATE TABLE shipments (
    shipment_id INTEGER PRIMARY KEY,
    order_id    INTEGER,
    line_no     INTEGER,
    carrier     TEXT,
    FOREIGN KEY (order_id, line_no) REFERENCES orders(order_id, line_no)
);

CREATE TABLE events (
    kind TEXT,
    at   TEXT
);
"""


def build_sqlite_db(path: str) -> str:
    """Create the fixture database at `path` and return the path."""
    import sqlite3

    conn = sqlite3.connect(path)
    try:
        conn.executescript(SQLITE_DDL)
        conn.executemany(
            "INSERT INTO customers (customer_id, full_name, country, signup_date,"
            " is_active, notes, avatar) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r["customer_id"],
                    r["full_name"],
                    r["country"],
                    r["signup_date"],
                    r["is_active"],
                    r["notes"],
                    r["avatar"],
                )
                for r in customer_rows()
            ],
        )
        conn.executemany(
            "INSERT INTO orders (order_id, line_no, customer_id, status, amount,"
            " ordered_at) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    r["order_id"],
                    r["line_no"],
                    r["customer_id"],
                    r["status"],
                    r["amount"],
                    r["ordered_at"],
                )
                for r in order_rows()
            ],
        )
        conn.executemany(
            "INSERT INTO shipments (shipment_id, order_id, line_no, carrier)"
            " VALUES (?, ?, ?, ?)",
            [
                (r["shipment_id"], r["order_id"], r["line_no"], r["carrier"])
                for r in shipment_rows()
            ],
        )
        conn.executemany(
            "INSERT INTO events (kind, at) VALUES (?, ?)",
            [(r["kind"], r["at"]) for r in event_rows()],
        )
        conn.commit()
    finally:
        conn.close()
    return path


# --------------------------------------------------------------------------
# PostgreSQL materialisation
# --------------------------------------------------------------------------

POSTGRES_DDL = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    full_name   TEXT NOT NULL,
    country     TEXT,
    signup_date DATE,
    is_active   BOOLEAN,
    notes       TEXT,
    avatar      BYTEA
);

CREATE TABLE orders (
    order_id    INTEGER NOT NULL,
    line_no     INTEGER NOT NULL,
    -- Deliberately column-less: `REFERENCES customers` targets the referenced
    -- table's primary key positionally, which adapters have to resolve.
    customer_id INTEGER REFERENCES customers,
    status      TEXT,
    amount      NUMERIC(10, 2),
    ordered_at  TIMESTAMP,
    PRIMARY KEY (order_id, line_no)
);

CREATE TABLE shipments (
    shipment_id INTEGER PRIMARY KEY,
    order_id    INTEGER,
    line_no     INTEGER,
    carrier     TEXT,
    FOREIGN KEY (order_id, line_no) REFERENCES orders(order_id, line_no)
);

CREATE TABLE events (
    kind TEXT,
    at   TEXT
);
"""


def populate_postgres(conn: Any) -> None:
    """Create and fill the fixture schema on an open psycopg connection."""
    import datetime as dt

    with conn.cursor() as cur:
        cur.execute(POSTGRES_DDL)
        cur.executemany(
            "INSERT INTO customers (customer_id, full_name, country, signup_date,"
            " is_active, notes, avatar) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    r["customer_id"],
                    r["full_name"],
                    r["country"],
                    dt.date.fromisoformat(r["signup_date"]),
                    bool(r["is_active"]),
                    r["notes"],
                    r["avatar"],
                )
                for r in customer_rows()
            ],
        )
        cur.executemany(
            "INSERT INTO orders (order_id, line_no, customer_id, status, amount,"
            " ordered_at) VALUES (%s, %s, %s, %s, %s, %s)",
            [
                (
                    r["order_id"],
                    r["line_no"],
                    r["customer_id"],
                    r["status"],
                    r["amount"],
                    dt.datetime.fromisoformat(r["ordered_at"]),
                )
                for r in order_rows()
            ],
        )
        cur.executemany(
            "INSERT INTO shipments (shipment_id, order_id, line_no, carrier)"
            " VALUES (%s, %s, %s, %s)",
            [
                (r["shipment_id"], r["order_id"], r["line_no"], r["carrier"])
                for r in shipment_rows()
            ],
        )
        cur.executemany(
            "INSERT INTO events (kind, at) VALUES (%s, %s)",
            [(r["kind"], r["at"]) for r in event_rows()],
        )
    conn.commit()


#: Roles every adapter must agree on, regardless of how the backend stores the
#: values. Asserted against both SQLite and Postgres.
EXPECTED_CUSTOMER_ROLES = {
    "customer_id": "id",
    "full_name": "free_text",
    "country": "categorical",
    "signup_date": "date",
    "is_active": "boolean",
    "notes": "free_text",
    "avatar": "unknown",
}

EXPECTED_ORDER_ROLES = {
    "order_id": "id",
    "line_no": "id",
    "customer_id": "id",
    "status": "categorical",
    "amount": "numeric",
    "ordered_at": "date",
}
