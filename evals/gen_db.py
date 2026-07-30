"""Seeded synthetic source with fully known ground truth.

The whole eval rests on one property: *we generated the data, so we know the
answers*. Gold answers are therefore computed by direct computation over
`manifest.json` -- the generator's own record of every row it emitted -- and
never by asking a model and never by querying through the system under test.

The database is deliberately awkward in five specific ways, because a clean
synthetic table would flatter the retriever:

* **Nulls** at varying rates, so "the email of X" is sometimes genuinely absent
  even for a customer who exists.
* **Near-identical names** ("Ahmed Al-Sayed" / "Ahmad Al Sayed" / "Ahmed
  Alsayed") as *distinct* customers. Lexical and dense retrieval disagree about
  these, which is the point: a harness where both agree measures nothing.
* **Arabic and English free text**, both populated, so tokenisation and byte-vs-
  token assumptions are exercised.
* **Dates spanning two full years** (2023-01-01 .. 2024-12-31), so date-range
  questions have interior windows and distractors have exterior ones.
* **Real foreign keys**, so join questions require more than one table.

Everything is a pure function of `--seed`. Two runs with the same seed produce
byte-identical manifests; `manifest["meta"]["content_hash"]` proves it.

Usage::

    python -m evals.gen_db --seed 7
    python -m evals.gen_db --seed 7 --export-dir evals/exports   # CSV + Parquet
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_SEED = 7
EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = EVALS_DIR / "data" / "eval.sqlite"
DEFAULT_MANIFEST_PATH = EVALS_DIR / "data" / "manifest.json"

#: The data's real date span. Distractor questions deliberately sit outside it.
DATE_MIN = date(2023, 1, 1)
DATE_MAX = date(2024, 12, 31)

MANIFEST_VERSION = 3


# --------------------------------------------------------------------------
# Schema, declared once and used for both DDL and the manifest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Col:
    name: str
    type_name: str
    nullable: bool = True
    is_primary_key: bool = False
    references: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type_name,
            "nullable": self.nullable,
            "primary_key": self.is_primary_key,
            "references": self.references,
        }


SCHEMA: dict[str, tuple[Col, ...]] = {
    "regions": (
        Col("region_id", "INTEGER", nullable=False, is_primary_key=True),
        Col("name", "TEXT", nullable=False),
        Col("name_ar", "TEXT", nullable=False),
        Col("country", "TEXT", nullable=False),
        Col("timezone", "TEXT", nullable=True),
    ),
    "customers": (
        Col("customer_id", "INTEGER", nullable=False, is_primary_key=True),
        Col("full_name", "TEXT", nullable=False),
        Col("full_name_ar", "TEXT", nullable=False),
        Col("email", "TEXT", nullable=True),
        Col("region_id", "INTEGER", nullable=False, references="regions.region_id"),
        Col("segment", "TEXT", nullable=False),
        Col("loyalty_tier", "TEXT", nullable=True),
        Col("signup_date", "TEXT", nullable=False),
        Col("credit_limit", "REAL", nullable=True),
        Col("notes", "TEXT", nullable=True),
        Col("notes_ar", "TEXT", nullable=True),
    ),
    "products": (
        Col("product_id", "INTEGER", nullable=False, is_primary_key=True),
        Col("name", "TEXT", nullable=False),
        Col("name_ar", "TEXT", nullable=False),
        Col("category", "TEXT", nullable=False),
        Col("subcategory", "TEXT", nullable=True),
        Col("unit_price", "REAL", nullable=False),
        Col("launch_date", "TEXT", nullable=False),
        Col("in_stock", "INTEGER", nullable=False),
        Col("description", "TEXT", nullable=True),
        Col("description_ar", "TEXT", nullable=True),
    ),
    "orders": (
        Col("order_id", "INTEGER", nullable=False, is_primary_key=True),
        Col("customer_id", "INTEGER", nullable=False, references="customers.customer_id"),
        Col("order_date", "TEXT", nullable=False),
        Col("status", "TEXT", nullable=False),
        Col("channel", "TEXT", nullable=False),
        Col("discount_pct", "REAL", nullable=True),
        Col("shipping_note", "TEXT", nullable=True),
    ),
    "order_items": (
        Col("item_id", "INTEGER", nullable=False, is_primary_key=True),
        Col("order_id", "INTEGER", nullable=False, references="orders.order_id"),
        Col("product_id", "INTEGER", nullable=False, references="products.product_id"),
        Col("quantity", "INTEGER", nullable=False),
        Col("unit_price", "REAL", nullable=False),
        Col("line_total", "REAL", nullable=False),
    ),
}

PRIMARY_KEYS: dict[str, str] = {
    t: next(c.name for c in cols if c.is_primary_key) for t, cols in SCHEMA.items()
}

TABLE_ORDER: tuple[str, ...] = (
    "regions",
    "customers",
    "products",
    "orders",
    "order_items",
)


# --------------------------------------------------------------------------
# Value vocabularies
# --------------------------------------------------------------------------

REGION_ROWS: tuple[tuple[str, str, str, str | None], ...] = (
    ("Riyadh", "الرياض", "Saudi Arabia", "Asia/Riyadh"),
    ("Jeddah", "جدة", "Saudi Arabia", "Asia/Riyadh"),
    ("Dammam", "الدمام", "Saudi Arabia", "Asia/Riyadh"),
    ("Abha", "أبها", "Saudi Arabia", None),
    ("Dubai", "دبي", "United Arab Emirates", "Asia/Dubai"),
    ("Doha", "الدوحة", "Qatar", "Asia/Qatar"),
    ("Cairo", "القاهرة", "Egypt", "Africa/Cairo"),
    ("Amman", "عمّان", "Jordan", None),
)

SEGMENTS: tuple[str, ...] = ("Retail", "Wholesale", "Enterprise", "Government")
LOYALTY_TIERS: tuple[str, ...] = ("Bronze", "Silver", "Gold", "Platinum")
ORDER_STATUSES: tuple[str, ...] = (
    "placed",
    "shipped",
    "delivered",
    "cancelled",
    "returned",
)
CHANNELS: tuple[str, ...] = ("web", "mobile", "store", "phone")

CATEGORIES: tuple[str, ...] = (
    "Electronics",
    "Furniture",
    "Stationery",
    "Grocery",
    "Apparel",
    "Hardware",
)

SUBCATEGORIES: dict[str, tuple[str, ...]] = {
    "Electronics": ("Peripherals", "Audio", "Networking", "Displays"),
    "Furniture": ("Seating", "Desks", "Storage"),
    "Stationery": ("Paper", "Writing", "Filing"),
    "Grocery": ("Beverages", "Snacks", "Staples"),
    "Apparel": ("Uniforms", "Outerwear", "Footwear"),
    "Hardware": ("Tools", "Fasteners", "Safety"),
}

CATEGORY_AR: dict[str, str] = {
    "Electronics": "إلكترونيات",
    "Furniture": "أثاث",
    "Stationery": "قرطاسية",
    "Grocery": "بقالة",
    "Apparel": "ملابس",
    "Hardware": "أدوات",
}

#: Deliberately confusable customers. Each base yields three *distinct* rows
#: whose names differ only in transliteration, so exact-string gold answers stay
#: unambiguous while retrieval has to actually discriminate.
CONFUSABLE_BASES: tuple[tuple[str, str, str, str], ...] = (
    # (first, alt_first, particle, last)
    ("Ahmed", "Ahmad", "Al", "Sayed"),
    ("Mohammed", "Muhammad", "Al", "Otaibi"),
    ("Fatima", "Fatimah", "Al", "Zahrani"),
    ("Youssef", "Yusuf", "Al", "Ibrahim"),
    ("Noura", "Nourah", "Al", "Qahtani"),
    ("Khaled", "Khalid", "Al", "Salem"),
    ("Sara", "Sarah", "Al", "Rahman"),
    ("Omar", "Umar", "Al", "Harbi"),
    ("Layla", "Laila", "Al", "Mutairi"),
    ("Hassan", "Hasan", "Al", "Dosari"),
    ("Maryam", "Mariam", "Al", "Shammari"),
    ("Abdullah", "Abdallah", "Al", "Ghamdi"),
)

FIRST_NAMES: tuple[str, ...] = (
    "Tariq", "Huda", "Rami", "Dana", "Salim", "Amina", "Bilal", "Rana",
    "Faisal", "Lina", "Nabil", "Zeina", "Hadi", "Manal", "Sami", "Reem",
    "Jamal", "Aisha", "Karim", "Salma", "Basel", "Dalia", "Waleed", "Hanan",
    "Ziad", "Rasha", "Adel", "Nadia", "Fadi", "Iman", "Marwan", "Ghada",
)

LAST_NAMES: tuple[str, ...] = (
    "Haddad", "Nasser", "Farouk", "Mansour", "Khoury", "Saab", "Barakat",
    "Zaher", "Ajami", "Sabbagh", "Darwish", "Fakhoury", "Rahal", "Toubia",
    "Chalhoub", "Maalouf", "Rizk", "Sleiman", "Younes", "Azar",
)

FIRST_NAMES_AR: tuple[str, ...] = (
    "طارق", "هدى", "رامي", "دانا", "سليم", "أمينة", "بلال", "رنا",
    "فيصل", "لينا", "نبيل", "زينة", "هادي", "منال", "سامي", "ريم",
)

LAST_NAMES_AR: tuple[str, ...] = (
    "حداد", "ناصر", "فاروق", "منصور", "خوري", "صعب", "بركات", "زاهر",
)

NOTES_EN: tuple[str, ...] = (
    "Prefers email contact before shipment.",
    "Long-standing account with quarterly review.",
    "Requires a purchase order number on every invoice.",
    "Delivery address updated after the last audit.",
    "Escalate any delay to the account manager.",
    "Tax exemption certificate on file.",
    "Consolidated billing across sites.",
    "Requested Arabic-language documentation.",
)

NOTES_AR: tuple[str, ...] = (
    "عميل منتظم يفضل التواصل عبر البريد الإلكتروني.",
    "حساب مؤسسي مع مراجعة ربع سنوية.",
    "يتطلب رقم أمر شراء على كل فاتورة.",
    "تم تحديث عنوان التوصيل بعد المراجعة الأخيرة.",
    "يرجى تصعيد أي تأخير إلى مدير الحساب.",
    "شهادة إعفاء ضريبي محفوظة في الملف.",
    "فوترة موحدة لجميع الفروع.",
    "طلب وثائق باللغة العربية.",
)

SHIPPING_NOTES: tuple[str, ...] = (
    "Leave with the reception desk.",
    "Call thirty minutes before arrival.",
    "Fragile: handle the display panel with care.",
    "يرجى الاتصال قبل التوصيل.",
    "التسليم بعد الساعة الرابعة عصراً.",
    "Signature required on delivery.",
)

PRODUCT_ADJ: tuple[str, ...] = (
    "Aurora", "Nimbus", "Falcon", "Cedar", "Orion", "Marina", "Vertex",
    "Halcyon", "Zephyr", "Quartz", "Basalt", "Lumen", "Cobalt", "Sable",
)

PRODUCT_NOUN: dict[str, tuple[str, ...]] = {
    "Electronics": ("Wireless Keyboard", "Noise-Cancelling Headset", "Mesh Router", "27-inch Monitor"),
    "Furniture": ("Office Chair", "Standing Desk", "Filing Cabinet"),
    "Stationery": ("A4 Paper Ream", "Gel Pen Pack", "Lever Arch File"),
    "Grocery": ("Arabic Coffee Blend", "Date Assortment", "Sparkling Water Case"),
    "Apparel": ("Field Uniform", "Rain Shell", "Safety Boots"),
    "Hardware": ("Torque Wrench", "Hex Bolt Set", "Safety Goggles"),
}

DESCRIPTIONS_EN: tuple[str, ...] = (
    "Stocked in the regional warehouse and shipped within two business days.",
    "Bulk pricing applies above twenty units.",
    "Replaces the previous generation and keeps the same mounting pattern.",
    "Certified for use in government procurement contracts.",
)

DESCRIPTIONS_AR: tuple[str, ...] = (
    "متوفر في المستودع الإقليمي ويشحن خلال يومي عمل.",
    "يطبق سعر الجملة عند طلب أكثر من عشرين وحدة.",
    "يحل محل الجيل السابق ويحافظ على نفس نمط التركيب.",
    "معتمد للاستخدام في عقود المشتريات الحكومية.",
)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Sizes:
    customers: int = 400
    products: int = 150
    orders: int = 1500
    max_items_per_order: int = 4


def _rand_date(rng: random.Random, lo: date = DATE_MIN, hi: date = DATE_MAX) -> date:
    return lo + timedelta(days=rng.randint(0, (hi - lo).days))


def _maybe(rng: random.Random, value: Any, null_rate: float) -> Any:
    """Return `value`, or None at the given rate. Nulls are load-bearing here."""
    return None if rng.random() < null_rate else value


def _confusable_names(first: str, alt_first: str, particle: str, last: str) -> tuple[str, str, str]:
    """Three transliterations of one underlying name, as distinct strings."""
    return (
        f"{first} {particle}-{last}",
        f"{alt_first} {particle} {last}",
        f"{first} {particle}{last.lower()}",
    )


def generate_rows(seed: int = DEFAULT_SEED, sizes: Sizes | None = None) -> dict[str, list[dict[str, Any]]]:
    """Produce every row of every table. Pure function of (seed, sizes)."""
    sizes = sizes or Sizes()
    rng = random.Random(seed)

    tables: dict[str, list[dict[str, Any]]] = {t: [] for t in TABLE_ORDER}

    # -- regions ---------------------------------------------------------
    for i, (name, name_ar, country, tz) in enumerate(REGION_ROWS, start=1):
        tables["regions"].append(
            {
                "region_id": i,
                "name": name,
                "name_ar": name_ar,
                "country": country,
                "timezone": tz,
            }
        )
    region_ids = [r["region_id"] for r in tables["regions"]]

    # -- customers -------------------------------------------------------
    used_names: set[str] = set()
    name_pool: list[str] = []
    for base in CONFUSABLE_BASES:
        for variant in _confusable_names(*base):
            if variant not in used_names:
                used_names.add(variant)
                name_pool.append(variant)

    # Fill the rest with unique first+last combinations. Uniqueness matters:
    # entity-lookup gold is "every row with exactly this name", and an accidental
    # duplicate would turn a single-row question into a silently multi-row one.
    for last in LAST_NAMES:
        for first in FIRST_NAMES:
            candidate = f"{first} {last}"
            if candidate not in used_names:
                used_names.add(candidate)
                name_pool.append(candidate)
    if len(name_pool) < sizes.customers:  # pragma: no cover - vocabulary is ample
        raise RuntimeError("name vocabulary too small for requested customer count")

    chosen = name_pool[: sizes.customers]
    rng.shuffle(chosen)

    for cid, full_name in enumerate(chosen, start=1):
        first_ar = FIRST_NAMES_AR[rng.randrange(len(FIRST_NAMES_AR))]
        last_ar = LAST_NAMES_AR[rng.randrange(len(LAST_NAMES_AR))]
        slug = full_name.lower().replace(" ", ".").replace("-", "")
        tables["customers"].append(
            {
                "customer_id": cid,
                "full_name": full_name,
                "full_name_ar": f"{first_ar} {last_ar}",
                # ~12% missing: "what is the email of X" must sometimes be a
                # legitimate "the row exists but the field is null".
                "email": _maybe(rng, f"{slug}@example.com", 0.12),
                "region_id": rng.choice(region_ids),
                "segment": rng.choice(SEGMENTS),
                "loyalty_tier": _maybe(rng, rng.choice(LOYALTY_TIERS), 0.08),
                "signup_date": _rand_date(rng).isoformat(),
                "credit_limit": _maybe(rng, round(rng.uniform(1000, 90000), 2), 0.18),
                "notes": _maybe(rng, rng.choice(NOTES_EN), 0.22),
                "notes_ar": _maybe(rng, rng.choice(NOTES_AR), 0.31),
            }
        )

    # -- products --------------------------------------------------------
    used_products: set[str] = set()
    pid = 0
    while len(tables["products"]) < sizes.products:
        category = CATEGORIES[len(tables["products"]) % len(CATEGORIES)]
        adj = rng.choice(PRODUCT_ADJ)
        noun = rng.choice(PRODUCT_NOUN[category])
        base = f"{adj} {noun}"
        # A "Pro" sibling makes near-duplicate product names as well as names.
        name = base if base not in used_products else f"{base} Pro"
        if name in used_products:
            name = f"{base} Mk{len(tables['products'])}"
        used_products.add(name)
        pid += 1
        price = round(rng.uniform(8.0, 1450.0), 2)
        tables["products"].append(
            {
                "product_id": pid,
                "name": name,
                "name_ar": f"{CATEGORY_AR[category]} {adj}",
                "category": category,
                "subcategory": _maybe(rng, rng.choice(SUBCATEGORIES[category]), 0.07),
                "unit_price": price,
                "launch_date": _rand_date(rng, DATE_MIN, date(2024, 6, 30)).isoformat(),
                "in_stock": 1 if rng.random() > 0.25 else 0,
                "description": _maybe(rng, rng.choice(DESCRIPTIONS_EN), 0.15),
                "description_ar": _maybe(rng, rng.choice(DESCRIPTIONS_AR), 0.26),
            }
        )

    customer_ids = [c["customer_id"] for c in tables["customers"]]
    product_rows = tables["products"]

    # -- orders + order_items -------------------------------------------
    item_id = 0
    for oid in range(1, sizes.orders + 1):
        tables["orders"].append(
            {
                "order_id": oid,
                "customer_id": rng.choice(customer_ids),
                "order_date": _rand_date(rng).isoformat(),
                "status": rng.choice(ORDER_STATUSES),
                "channel": rng.choice(CHANNELS),
                "discount_pct": _maybe(rng, round(rng.uniform(0.0, 35.0), 1), 0.42),
                "shipping_note": _maybe(rng, rng.choice(SHIPPING_NOTES), 0.55),
            }
        )
        for _ in range(rng.randint(1, sizes.max_items_per_order)):
            item_id += 1
            product = rng.choice(product_rows)
            qty = rng.randint(1, 12)
            unit_price = product["unit_price"]
            tables["order_items"].append(
                {
                    "item_id": item_id,
                    "order_id": oid,
                    "product_id": product["product_id"],
                    "quantity": qty,
                    "unit_price": unit_price,
                    # Rounded at generation time so gold sums and SQL sums agree
                    # to within float noise rather than by luck.
                    "line_total": round(qty * unit_price, 2),
                }
            )

    return tables


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_manifest(
    seed: int = DEFAULT_SEED,
    sizes: Sizes | None = None,
    *,
    source_id: str = "sqlite:evaldb",
) -> dict[str, Any]:
    """Schema + every generated row + a content hash proving reproducibility."""
    sizes = sizes or Sizes()
    tables = generate_rows(seed, sizes)

    schema = {
        "tables": [
            {
                "name": t,
                "primary_key": PRIMARY_KEYS[t],
                "columns": [c.as_dict() for c in SCHEMA[t]],
            }
            for t in TABLE_ORDER
        ]
    }

    payload = {"schema": schema, "tables": {t: tables[t] for t in TABLE_ORDER}}
    content_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    return {
        "meta": {
            "manifest_version": MANIFEST_VERSION,
            "generator": "evals.gen_db",
            "seed": seed,
            "source_id": source_id,
            "date_min": DATE_MIN.isoformat(),
            "date_max": DATE_MAX.isoformat(),
            "row_counts": {t: len(tables[t]) for t in TABLE_ORDER},
            "total_rows": sum(len(tables[t]) for t in TABLE_ORDER),
            "content_hash": content_hash,
            # No wall-clock timestamp: it would make the manifest non-reproducible
            # for no benefit, and reproducibility is the point of the file.
        },
        "schema": schema,
        "tables": payload["tables"],
    }


class Manifest:
    """Read-only accessor over `manifest.json`.

    Every gold answer in `questions.py` is computed from this object, in plain
    Python, with no SQL and no model. `tests/test_questions.py` recomputes a
    sample of the same answers with SQL against the generated SQLite file, so
    the two independent paths have to agree.
    """

    def __init__(self, data: Mapping[str, Any]) -> None:
        self.data = data
        self._by_pk: dict[str, dict[str, dict[str, Any]]] = {}

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MANIFEST_PATH) -> "Manifest":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    @classmethod
    def generate(cls, seed: int = DEFAULT_SEED, sizes: Sizes | None = None) -> "Manifest":
        return cls(build_manifest(seed, sizes))

    # -- accessors -------------------------------------------------------

    @property
    def meta(self) -> Mapping[str, Any]:
        return self.data["meta"]

    @property
    def seed(self) -> int:
        return int(self.data["meta"]["seed"])

    @property
    def source_id(self) -> str:
        return str(self.data["meta"].get("source_id", "sqlite:evaldb"))

    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(t["name"] for t in self.data["schema"]["tables"])

    def rows(self, table: str) -> list[dict[str, Any]]:
        return self.data["tables"][table]

    def pk_col(self, table: str) -> str:
        for spec in self.data["schema"]["tables"]:
            if spec["name"] == table:
                return spec["primary_key"]
        raise KeyError(table)

    def columns(self, table: str) -> tuple[str, ...]:
        for spec in self.data["schema"]["tables"]:
            if spec["name"] == table:
                return tuple(c["name"] for c in spec["columns"])
        raise KeyError(table)

    def column_specs(self, table: str) -> list[dict[str, Any]]:
        for spec in self.data["schema"]["tables"]:
            if spec["name"] == table:
                return list(spec["columns"])
        raise KeyError(table)

    def pk_of(self, table: str, row: Mapping[str, Any]) -> str:
        return str(row[self.pk_col(table)])

    def by_pk(self, table: str) -> dict[str, dict[str, Any]]:
        cached = self._by_pk.get(table)
        if cached is None:
            pk = self.pk_col(table)
            cached = {str(r[pk]): r for r in self.rows(table)}
            self._by_pk[table] = cached
        return cached

    def distinct(self, table: str, column: str) -> list[Any]:
        seen: list[Any] = []
        marker: set[Any] = set()
        for row in self.rows(table):
            val = row.get(column)
            if val is None or val in marker:
                continue
            marker.add(val)
            seen.append(val)
        return seen

    def values(self, table: str, column: str) -> list[Any]:
        return [r.get(column) for r in self.rows(table)]


# --------------------------------------------------------------------------
# Emission: SQLite, JSON, and optional flat-file exports
# --------------------------------------------------------------------------


def _ddl(table: str) -> str:
    cols = SCHEMA[table]
    parts: list[str] = []
    for c in cols:
        frag = f'  "{c.name}" {c.type_name}'
        if c.is_primary_key:
            frag += " PRIMARY KEY"
        if not c.nullable:
            frag += " NOT NULL"
        parts.append(frag)
    for c in cols:
        if c.references:
            ref_table, ref_col = c.references.split(".")
            parts.append(f'  FOREIGN KEY ("{c.name}") REFERENCES "{ref_table}"("{ref_col}")')
    body = ",\n".join(parts)
    return f'CREATE TABLE "{table}" (\n{body}\n)'


def write_sqlite(manifest: Mapping[str, Any], db_path: str | Path) -> Path:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        for table in TABLE_ORDER:
            conn.execute(_ddl(table))
            cols = [c.name for c in SCHEMA[table]]
            placeholders = ",".join("?" for _ in cols)
            quoted = ",".join(f'"{c}"' for c in cols)
            conn.executemany(
                f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})',
                [tuple(row[c] for c in cols) for row in manifest["tables"][table]],
            )
        # Indexes the retriever never sees but the SQL path benefits from.
        conn.execute('CREATE INDEX idx_orders_customer ON "orders"("customer_id")')
        conn.execute('CREATE INDEX idx_orders_date ON "orders"("order_date")')
        conn.execute('CREATE INDEX idx_items_order ON "order_items"("order_id")')
        conn.execute('CREATE INDEX idx_items_product ON "order_items"("product_id")')
        conn.commit()
    finally:
        conn.close()
    return db_path


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return path


def write_exports(manifest: Mapping[str, Any], export_dir: str | Path) -> list[Path]:
    """CSV (+ Parquet when pyarrow is present) for the source-agnosticism run."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for table in TABLE_ORDER:
        cols = [c.name for c in SCHEMA[table]]
        rows = manifest["tables"][table]
        csv_path = export_dir / f"{table}.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row[c] for c in cols})
        written.append(csv_path)
        try:
            import pyarrow as pa  # noqa: PLC0415
            import pyarrow.parquet as pq  # noqa: PLC0415
        except Exception:  # pragma: no cover - pyarrow is in requirements
            continue
        table_arrow = pa.table({c: [row[c] for row in rows] for c in cols})
        pq_path = export_dir / f"{table}.parquet"
        pq.write_table(table_arrow, pq_path)
        written.append(pq_path)
    return written


def generate(
    seed: int = DEFAULT_SEED,
    db_path: str | Path = DEFAULT_DB_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    sizes: Sizes | None = None,
    export_dir: str | Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    manifest = build_manifest(seed, sizes, source_id=f"sqlite:{Path(db_path).stem}")
    db = write_sqlite(manifest, db_path)
    mf = write_manifest(manifest, manifest_path)
    if export_dir:
        write_exports(manifest, export_dir)
    return db, mf, manifest


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the synthetic eval database.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    ap.add_argument("--export-dir", default=None, help="also write CSV/Parquet here")
    ap.add_argument("--customers", type=int, default=Sizes().customers)
    ap.add_argument("--products", type=int, default=Sizes().products)
    ap.add_argument("--orders", type=int, default=Sizes().orders)
    args = ap.parse_args(argv)

    sizes = Sizes(customers=args.customers, products=args.products, orders=args.orders)
    db, mf, manifest = generate(
        seed=args.seed,
        db_path=args.db,
        manifest_path=args.manifest,
        sizes=sizes,
        export_dir=args.export_dir,
    )
    counts = manifest["meta"]["row_counts"]
    print(f"database : {db}")
    print(f"manifest : {mf}")
    print(f"seed     : {args.seed}")
    print(f"hash     : {manifest['meta']['content_hash'][:16]}")
    for table in TABLE_ORDER:
        print(f"  {table:<12} {counts[table]:>6} rows")
    print(f"  {'TOTAL':<12} {manifest['meta']['total_rows']:>6} rows")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
