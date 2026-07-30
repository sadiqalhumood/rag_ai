"""Programmatically generated questions with programmatically computed gold.

Every gold answer here is computed by plain Python over `manifest.json` -- the
generator's own record of the rows it emitted. No model is asked what the answer
is, and nothing is queried through the system under test. That is the only
reason the harness can grade itself.

Eight question types
--------------------

===========================  =====================================  ==========
type                         what it exercises                      route
===========================  =====================================  ==========
``entity_lookup``            name -> row, including confusable      LOOKUP
                             transliterations
``multi_filter_lookup``      conjunctive attribute filters          LOOKUP
``count_by_category``        COUNT over one categorical column      AGGREGATE
``numeric_aggregate``        SUM/AVG/MIN/MAX by group, with nulls   AGGREGATE
``date_range_aggregate``     windows inside the two-year span       AGGREGATE
``join_relationship``        two or three tables                    HYBRID
``schema_question``          schema cards, not rows                 LOOKUP
``distractor``               answer genuinely absent -> refuse      (varies)
===========================  =====================================  ==========

Distractors
-----------

The false-answer rate is the headline number, so a distractor set that is
trivially refusable would make the whole report meaningless. These are built to
be hard on purpose:

* names one letter from a real customer, checked against a *normalised* form of
  every real name so a perturbation cannot accidentally land on a real person or
  on one of the deliberate transliteration variants;
* categories one letter off ("Stationary" for "Stationery", "Electronic" for
  "Electronics") -- the retriever will happily surface near-matching rows, and
  the system has to notice that the value does not exist;
* attributes that do not exist on entities that do ("the phone number of
  <a real customer>");
* whole tables that do not exist (suppliers, warehouses);
* date windows entirely outside the data, phrased so that the empty result is
  genuinely undefined rather than legitimately zero.

That last point matters. "How many orders were placed in March 2025" has a
correct answer -- zero -- so it is *not* a distractor and is deliberately
excluded. Only questions whose empty result is undefined (the average over an
empty set) or which ask to enumerate and cite rows that do not exist are used.

Dev vs held-out
---------------

`DEV_TEMPLATES` is built now. `HELDOUT_TEMPLATES` is deliberately empty until
the router and SQL generator are frozen: a held-out set written against a router
that does not exist yet cannot measure overfitting to dev phrasings. Selecting
``--templates heldout`` fails loudly rather than silently returning nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Iterable, Mapping, Sequence

from anyrag.core.types import QueryRoute, RowRef

from .gen_db import DATE_MAX, DATE_MIN, DEFAULT_SEED, Manifest
from .metrics import GoldTarget

TEMPLATE_SET_NAMES: tuple[str, ...] = ("dev", "heldout")

QTYPES: tuple[str, ...] = (
    "entity_lookup",
    "multi_filter_lookup",
    "count_by_category",
    "numeric_aggregate",
    "date_range_aggregate",
    "join_relationship",
    "schema_question",
    "distractor",
)


# --------------------------------------------------------------------------
# Question record
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalQuestion:
    qid: str
    text: str
    qtype: str
    template_key: str
    template_set: str
    route: QueryRoute
    gold: GoldTarget = field(default_factory=GoldTarget)
    gold_scalar: Any = None
    #: "count" | "sum" | "avg" | "min" | "max" | "" -- drives the tolerance rule.
    gold_scalar_kind: str = ""
    answerable: bool = True
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def gold_row_refs(self) -> tuple[RowRef, ...]:
        return tuple(sorted(self.gold.row_refs))

    def as_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "text": self.text,
            "qtype": self.qtype,
            "template_key": self.template_key,
            "template_set": self.template_set,
            "route": self.route.value,
            "answerable": self.answerable,
            "gold_row_refs": [f"{r.table}#{r.pk}" for r in self.gold_row_refs],
            "gold_schema_tables": sorted(self.gold.schema_tables),
            "gold_scalar": self.gold_scalar,
            "gold_scalar_kind": self.gold_scalar_kind,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class Template:
    key: str
    qtype: str
    template_set: str
    n: int
    fn: Callable[[Manifest, random.Random, int], list[dict[str, Any]]]


DEV_TEMPLATES: list[Template] = []
HELDOUT_TEMPLATES: list[Template] = []

#: Flipped to True in the same commit that adds the held-out templates.
HELDOUT_AVAILABLE = True

#: The router freeze this held-out set was written against (amendment 3).
#:
#: The freeze commit was rewritten by a `--reset-author` rebase after it was
#: first communicated. That is exactly the situation where a quietly-corrected
#: hash is indistinguishable from one chosen after the fact, so both SHAs are
#: recorded along with the *blob* hashes of the three frozen files, which are
#: what actually matter: they are byte-identical at both commits and identical
#: to the files these templates were written against. Verified, not accepted --
#: `tests/test_questions.py::test_the_router_freeze_is_verifiable` re-checks the
#: blob hashes against the working tree on every run, so a later edit to any
#: frozen file fails the suite instead of silently invalidating the held-out
#: numbers.
ROUTER_FREEZE = {
    "commit": "60adca7543fda82daf90cb17db61513abace64b7",
    "superseded_commit": "0439893552d38469ea65c9a48ee8ed04c156bbe7",
    "supersession": (
        "--reset-author rebase rewrote the commit object; "
        "git diff <old>..<new> -- anyrag/route/ is empty"
    ),
    "frozen_blobs": {
        "anyrag/route/router.py": "3be95d29da9586d2549576ae7d3ea6d1a145d788",
        "anyrag/route/sqlgen.py": "9e338594cacd1663fa5c0e116f0ca81f06b37fc0",
        "anyrag/route/schema_lexicon.py": "a4cf6d4d8795d30240aafae11a36daf719c6b79a",
    },
}


def _register(bucket: list[Template], key: str, qtype: str, tset: str, n: int):
    def deco(fn):
        if any(t.key == key for t in bucket):
            raise ValueError(f"duplicate template key {key!r}")
        if qtype not in QTYPES:
            raise ValueError(f"unknown qtype {qtype!r}")
        bucket.append(Template(key=key, qtype=qtype, template_set=tset, n=n, fn=fn))
        return fn

    return deco


def dev_template(key: str, qtype: str, n: int):
    return _register(DEV_TEMPLATES, key, qtype, "dev", n)


def heldout_template(key: str, qtype: str, n: int):  # pragma: no cover - not yet used
    return _register(HELDOUT_TEMPLATES, key, qtype, "heldout", n)


# --------------------------------------------------------------------------
# Gold-computation helpers (direct computation over the manifest)
# --------------------------------------------------------------------------


def _ref(m: Manifest, table: str, row: Mapping[str, Any]) -> RowRef:
    return RowRef(table=table, pk=m.pk_of(table, row))


def _refs(m: Manifest, table: str, rows: Iterable[Mapping[str, Any]]) -> list[RowRef]:
    return [_ref(m, table, r) for r in rows]


def _where(rows: Sequence[Mapping[str, Any]], **eq: Any) -> list[Mapping[str, Any]]:
    return [r for r in rows if all(r.get(k) == v for k, v in eq.items())]


def _non_null(values: Iterable[Any]) -> list[Any]:
    return [v for v in values if v is not None]


def _agg(values: Sequence[Any], fn: str) -> Any:
    """SQL aggregate semantics: nulls already excluded, empty -> None (except COUNT)."""
    if fn == "count":
        return len(values)
    if not values:
        return None
    if fn == "sum":
        return float(sum(values))
    if fn == "avg":
        return float(sum(values)) / len(values)
    if fn == "min":
        return min(values)
    if fn == "max":
        return max(values)
    raise ValueError(fn)


def _norm_name(value: Any) -> str:
    """Casefold and strip non-alphanumerics.

    Used to check that a synthetic distractor name really is absent. Plain
    equality is not enough: the data deliberately contains "Ahmad Al Sayed" and
    "Ahmed Al-Sayed", so a perturbation that differs only in punctuation would
    otherwise be scored as unanswerable while a reasonable system finds it.
    """
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _assert_absent(candidate: str, pool: Iterable[Any], what: str) -> str:
    normalised = {_norm_name(v) for v in pool}
    if _norm_name(candidate) in normalised:
        raise AssertionError(
            f"distractor {what} {candidate!r} actually exists in the data; "
            "a distractor whose answer exists would corrupt the false-answer rate"
        )
    return candidate


def _iso(d: date) -> str:
    return d.isoformat()


_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    end = (date(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1))
    return start, end


def _sample(rng: random.Random, population: Sequence[Any], n: int) -> list[Any]:
    """Deterministic sample that tolerates n > len(population) by cycling."""
    pop = list(population)
    if not pop:
        return []
    if n <= len(pop):
        return rng.sample(pop, n)
    out: list[Any] = []
    while len(out) < n:
        chunk = pop[:]
        rng.shuffle(chunk)
        out.extend(chunk)
    return out[:n]


# ==========================================================================
# DEV TEMPLATES
# ==========================================================================

# -- 1. entity lookup by name ---------------------------------------------

_CUSTOMER_ATTRS: tuple[tuple[str, str], ...] = (
    ("email", "What is the email address of {name}?"),
    ("segment", "Which customer segment is {name} in?"),
    ("loyalty_tier", "What loyalty tier does {name} hold?"),
    ("signup_date", "On what date did {name} sign up?"),
    ("credit_limit", "What credit limit is recorded for {name}?"),
)


@dev_template("ent_customer_attr", "entity_lookup", 24)
def _t_ent_customer_attr(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    customers = m.rows("customers")
    # Half the questions deliberately name a customer from a confusable
    # transliteration cluster, where lexical and dense retrieval disagree.
    clusters: dict[str, list[Mapping[str, Any]]] = {}
    for row in customers:
        clusters.setdefault(_norm_name(row["full_name"])[-8:], []).append(row)
    confusable = [r for r in customers if len(clusters[_norm_name(r["full_name"])[-8:]]) > 1]
    plain = [r for r in customers if r not in confusable]

    out: list[dict[str, Any]] = []
    half = n // 2
    pool = _sample(rng, confusable, half) + _sample(rng, plain, n - half)
    for i, row in enumerate(pool):
        attr, phrasing = _CUSTOMER_ATTRS[i % len(_CUSTOMER_ATTRS)]
        if row.get(attr) is None:
            # Asking for a null field makes "answerable" ambiguous, so pick a
            # populated attribute instead. Nulls are stressed elsewhere.
            populated = [a for a, _ in _CUSTOMER_ATTRS if row.get(a) is not None]
            if not populated:
                continue
            attr = populated[0]
            phrasing = dict(_CUSTOMER_ATTRS)[attr]
        out.append(
            {
                "text": phrasing.format(name=row["full_name"]),
                "route": QueryRoute.LOOKUP,
                "gold": GoldTarget.of_rows([_ref(m, "customers", row)]),
                "meta": {
                    "table": "customers",
                    "attribute": attr,
                    "expected_value": row[attr],
                    "confusable": row in confusable,
                },
            }
        )
    return out


_PRODUCT_ATTRS: tuple[tuple[str, str], ...] = (
    ("category", "Which category does the product '{name}' belong to?"),
    ("unit_price", "What is the unit price of the product '{name}'?"),
    ("launch_date", "When was the product '{name}' launched?"),
    ("subcategory", "What subcategory is the product '{name}' filed under?"),
)


@dev_template("ent_product_attr", "entity_lookup", 20)
def _t_ent_product_attr(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    products = m.rows("products")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(_sample(rng, products, n)):
        attr, phrasing = _PRODUCT_ATTRS[i % len(_PRODUCT_ATTRS)]
        if row.get(attr) is None:
            attr, phrasing = _PRODUCT_ATTRS[0]
        out.append(
            {
                "text": phrasing.format(name=row["name"]),
                "route": QueryRoute.LOOKUP,
                "gold": GoldTarget.of_rows([_ref(m, "products", row)]),
                "meta": {"table": "products", "attribute": attr, "expected_value": row[attr]},
            }
        )
    return out


@dev_template("ent_region_attr", "entity_lookup", 12)
def _t_ent_region_attr(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    regions = m.rows("regions")
    phrasings = (
        ("country", "Which country is the {name} region in?"),
        ("timezone", "What timezone is used in the {name} region?"),
        ("name_ar", "What is the Arabic name of the {name} region?"),
    )
    out: list[dict[str, Any]] = []
    for i, row in enumerate(_sample(rng, regions, n)):
        attr, phrasing = phrasings[i % len(phrasings)]
        if row.get(attr) is None:
            attr, phrasing = phrasings[0]
        out.append(
            {
                "text": phrasing.format(name=row["name"]),
                "route": QueryRoute.LOOKUP,
                "gold": GoldTarget.of_rows([_ref(m, "regions", row)]),
                "meta": {"table": "regions", "attribute": attr, "expected_value": row[attr]},
            }
        )
    return out


# -- 2. multi-attribute filter lookup -------------------------------------


@dev_template("filt_customers", "multi_filter_lookup", 20)
def _t_filt_customers(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    customers = m.rows("customers")
    regions = {r["region_id"]: r for r in m.rows("regions")}
    segments = sorted({c["segment"] for c in customers})
    tiers = sorted({c["loyalty_tier"] for c in customers if c["loyalty_tier"]})

    combos: list[tuple[int, str, str]] = [
        (rid, seg, tier)
        for rid in sorted(regions)
        for seg in segments
        for tier in tiers
    ]
    rng.shuffle(combos)

    out: list[dict[str, Any]] = []
    for rid, seg, tier in combos:
        matches = _where(customers, region_id=rid, segment=seg, loyalty_tier=tier)
        # Keep it a lookup, not a disguised aggregate: a gold set of 50 rows
        # cannot be answered by citing rows, so those combinations are dropped.
        if not (1 <= len(matches) <= 8):
            continue
        region = regions[rid]["name"]
        out.append(
            {
                "text": (
                    f"Which customers in the {region} region are in the {seg} "
                    f"segment with a {tier} loyalty tier?"
                ),
                "route": QueryRoute.LOOKUP,
                "gold": GoldTarget.of_rows(_refs(m, "customers", matches)),
                "meta": {
                    "table": "customers",
                    "filters": {"region": region, "segment": seg, "loyalty_tier": tier},
                    "n_matches": len(matches),
                },
            }
        )
        if len(out) >= n:
            break
    return out


@dev_template("filt_products", "multi_filter_lookup", 12)
def _t_filt_products(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    products = m.rows("products")
    categories = sorted({p["category"] for p in products})
    thresholds = (50.0, 120.0, 300.0, 600.0, 900.0)
    combos = [(c, t, s) for c in categories for t in thresholds for s in (0, 1)]
    rng.shuffle(combos)

    out: list[dict[str, Any]] = []
    for cat, thresh, stock in combos:
        matches = [
            p
            for p in products
            if p["category"] == cat and p["unit_price"] < thresh and p["in_stock"] == stock
        ]
        if not (1 <= len(matches) <= 8):
            continue
        stock_phrase = "in stock" if stock else "out of stock"
        out.append(
            {
                "text": (
                    f"Which {cat} products priced under {thresh:.0f} are {stock_phrase}?"
                ),
                "route": QueryRoute.LOOKUP,
                "gold": GoldTarget.of_rows(_refs(m, "products", matches)),
                "meta": {
                    "table": "products",
                    "filters": {"category": cat, "unit_price_lt": thresh, "in_stock": stock},
                    "n_matches": len(matches),
                },
            }
        )
        if len(out) >= n:
            break
    return out


@dev_template("filt_orders", "multi_filter_lookup", 8)
def _t_filt_orders(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    orders = m.rows("orders")
    statuses = sorted({o["status"] for o in orders})
    channels = sorted({o["channel"] for o in orders})
    cutoffs = (25.0, 28.0, 30.0, 32.0)
    combos = [(s, c, d) for s in statuses for c in channels for d in cutoffs]
    rng.shuffle(combos)

    out: list[dict[str, Any]] = []
    for status, channel, cutoff in combos:
        matches = [
            o
            for o in orders
            if o["status"] == status
            and o["channel"] == channel
            and o["discount_pct"] is not None
            and o["discount_pct"] > cutoff
        ]
        if not (1 <= len(matches) <= 8):
            continue
        out.append(
            {
                "text": (
                    f"Which {channel} orders have status {status} and a discount "
                    f"above {cutoff:.0f} percent?"
                ),
                "route": QueryRoute.LOOKUP,
                "gold": GoldTarget.of_rows(_refs(m, "orders", matches)),
                "meta": {
                    "table": "orders",
                    "filters": {"status": status, "channel": channel, "discount_gt": cutoff},
                    "n_matches": len(matches),
                },
            }
        )
        if len(out) >= n:
            break
    return out


# -- 3. count by category --------------------------------------------------

_COUNT_SPECS: tuple[tuple[str, str, str], ...] = (
    ("orders", "status", "How many orders have the status '{value}'?"),
    ("orders", "status", "Count the orders whose status is '{value}'."),
    ("orders", "channel", "How many orders came through the {value} channel?"),
    ("orders", "channel", "What is the number of orders placed via {value}?"),
    ("customers", "segment", "How many customers belong to the {value} segment?"),
    ("customers", "segment", "Count the customers in the {value} segment."),
    ("customers", "loyalty_tier", "How many customers hold a {value} loyalty tier?"),
    ("products", "category", "How many products are in the {value} category?"),
    ("products", "category", "Count the products filed under {value}."),
    ("products", "subcategory", "How many products are in the {value} subcategory?"),
)


@dev_template("cnt_by_category", "count_by_category", 40)
def _t_count_by_category(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    candidates: list[tuple[str, str, str, Any]] = []
    for table, column, phrasing in _COUNT_SPECS:
        for value in sorted(str(v) for v in m.distinct(table, column)):
            candidates.append((table, column, phrasing, value))
    rng.shuffle(candidates)

    out: list[dict[str, Any]] = []
    for table, column, phrasing, value in candidates[:n]:
        matches = [r for r in m.rows(table) if str(r.get(column)) == value]
        out.append(
            {
                "text": phrasing.format(value=value),
                "route": QueryRoute.AGGREGATE,
                # The gold *evidence* for a count is every contributing row, but
                # the gold *answer* is the scalar. Retrieval is not graded on
                # counts: no top-10 list can cover 300 rows, and pretending
                # otherwise would report a fake retrieval failure.
                "gold": GoldTarget(),
                "gold_scalar": len(matches),
                "gold_scalar_kind": "count",
                "meta": {
                    "table": table,
                    "column": column,
                    "value": value,
                    "sql": f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" = \'{value}\'',
                },
            }
        )
    return out


# -- 4. numeric aggregate by group ----------------------------------------

_AGG_PHRASINGS: dict[str, str] = {
    "avg": "What is the average {label} of {scope}?",
    "sum": "What is the total {label} of {scope}?",
    "min": "What is the lowest {label} of {scope}?",
    "max": "What is the highest {label} of {scope}?",
}


@dev_template("num_products_by_category", "numeric_aggregate", 24)
def _t_num_products(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    products = m.rows("products")
    categories = sorted({p["category"] for p in products})
    combos = [(c, f) for c in categories for f in ("avg", "sum", "min", "max")]
    rng.shuffle(combos)

    out: list[dict[str, Any]] = []
    for cat, fn in combos[:n]:
        vals = _non_null([p["unit_price"] for p in products if p["category"] == cat])
        gold = _agg(vals, fn)
        if gold is None:
            continue
        out.append(
            {
                "text": _AGG_PHRASINGS[fn].format(
                    label="unit price", scope=f"products in the {cat} category"
                ),
                "route": QueryRoute.AGGREGATE,
                "gold": GoldTarget(),
                "gold_scalar": float(gold),
                "gold_scalar_kind": fn,
                "meta": {
                    "table": "products",
                    "column": "unit_price",
                    "group": {"category": cat},
                    "n_contributing_rows": len(vals),
                    "sql": (
                        f'SELECT {fn.upper()}("unit_price") FROM "products" '
                        f"WHERE \"category\" = '{cat}'"
                    ),
                },
            }
        )
    return out


@dev_template("num_customers_by_segment", "numeric_aggregate", 12)
def _t_num_customers(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    customers = m.rows("customers")
    segments = sorted({c["segment"] for c in customers})
    combos = [(s, f) for s in segments for f in ("avg", "max", "min", "sum")]
    rng.shuffle(combos)

    out: list[dict[str, Any]] = []
    for seg, fn in combos[:n]:
        # credit_limit is ~18% null. SQL aggregates skip nulls, and so does the
        # gold: this is one of the places the null rate is load-bearing.
        vals = _non_null([c["credit_limit"] for c in customers if c["segment"] == seg])
        gold = _agg(vals, fn)
        if gold is None:
            continue
        out.append(
            {
                "text": _AGG_PHRASINGS[fn].format(
                    label="credit limit", scope=f"customers in the {seg} segment"
                ),
                "route": QueryRoute.AGGREGATE,
                "gold": GoldTarget(),
                "gold_scalar": float(gold),
                "gold_scalar_kind": fn,
                "meta": {
                    "table": "customers",
                    "column": "credit_limit",
                    "group": {"segment": seg},
                    "n_contributing_rows": len(vals),
                    "null_sensitive": True,
                    "sql": (
                        f'SELECT {fn.upper()}("credit_limit") FROM "customers" '
                        f"WHERE \"segment\" = '{seg}'"
                    ),
                },
            }
        )
    return out


@dev_template("num_orders_and_items", "numeric_aggregate", 8)
def _t_num_orders_items(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    orders = m.rows("orders")
    items = m.rows("order_items")
    specs: list[tuple[str, str, str, str, list[Any]]] = []
    for status in sorted({o["status"] for o in orders}):
        vals = _non_null([o["discount_pct"] for o in orders if o["status"] == status])
        specs.append(("avg", "discount percentage", f"orders with status {status}", "orders", vals))
    for fn in ("avg", "max", "min", "sum"):
        specs.append((fn, "line total", "all order items", "order_items",
                      _non_null([i["line_total"] for i in items])))
    for fn in ("avg", "max"):
        specs.append((fn, "quantity", "all order items", "order_items",
                      _non_null([i["quantity"] for i in items])))
    rng.shuffle(specs)

    out: list[dict[str, Any]] = []
    for fn, label, scope, table, vals in specs[:n]:
        gold = _agg(vals, fn)
        if gold is None:
            continue
        out.append(
            {
                "text": _AGG_PHRASINGS[fn].format(label=label, scope=scope),
                "route": QueryRoute.AGGREGATE,
                "gold": GoldTarget(),
                "gold_scalar": float(gold),
                "gold_scalar_kind": fn,
                "meta": {"table": table, "n_contributing_rows": len(vals)},
            }
        )
    return out


# -- 5. date-range aggregate ----------------------------------------------


@dev_template("date_orders_window", "date_range_aggregate", 18)
def _t_date_orders_window(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    orders = m.rows("orders")
    span = (DATE_MAX - DATE_MIN).days
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    guard = 0
    while len(out) < n and guard < n * 40:
        guard += 1
        start = DATE_MIN + timedelta(days=rng.randint(0, span - 30))
        length = rng.choice((30, 45, 60, 90, 120, 180))
        end = min(start + timedelta(days=length), DATE_MAX)
        key = (_iso(start), _iso(end))
        if key in seen:
            continue
        seen.add(key)
        matches = [o for o in orders if _iso(start) <= o["order_date"] <= _iso(end)]
        if not matches:
            continue
        out.append(
            {
                "text": (
                    f"How many orders were placed between {_iso(start)} and {_iso(end)}?"
                ),
                "route": QueryRoute.AGGREGATE,
                "gold": GoldTarget(),
                "gold_scalar": len(matches),
                "gold_scalar_kind": "count",
                "meta": {
                    "table": "orders",
                    "column": "order_date",
                    "window": [_iso(start), _iso(end)],
                    "inclusive": True,
                },
            }
        )
    return out


@dev_template("date_month_windows", "date_range_aggregate", 12)
def _t_date_month(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    orders = m.rows("orders")
    months = [(y, mo) for y in (2023, 2024) for mo in range(1, 13)]
    rng.shuffle(months)
    out: list[dict[str, Any]] = []
    for year, month in months[:n]:
        start, end = _month_bounds(year, month)
        matches = [o for o in orders if _iso(start) <= o["order_date"] <= _iso(end)]
        out.append(
            {
                "text": (
                    f"How many orders were placed in {_MONTH_NAMES[month - 1]} {year}?"
                ),
                "route": QueryRoute.AGGREGATE,
                "gold": GoldTarget(),
                "gold_scalar": len(matches),
                "gold_scalar_kind": "count",
                "meta": {
                    "table": "orders",
                    "column": "order_date",
                    "window": [_iso(start), _iso(end)],
                },
            }
        )
    return out


@dev_template("date_signup_launch", "date_range_aggregate", 6)
def _t_date_signup_launch(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    specs: list[tuple[str, str, str, date, date]] = []
    for year in (2023, 2024):
        specs.append(
            ("customers", "signup_date", f"How many customers signed up in {year}?",
             date(year, 1, 1), date(year, 12, 31))
        )
        specs.append(
            ("products", "launch_date", f"How many products were launched in {year}?",
             date(year, 1, 1), date(year, 12, 31))
        )
    specs.append(
        ("customers", "signup_date",
         "How many customers signed up between 2023-07-01 and 2024-06-30?",
         date(2023, 7, 1), date(2024, 6, 30))
    )
    specs.append(
        ("products", "launch_date",
         "How many products were launched between 2023-03-01 and 2023-12-31?",
         date(2023, 3, 1), date(2023, 12, 31))
    )
    rng.shuffle(specs)
    out: list[dict[str, Any]] = []
    for table, column, text, start, end in specs[:n]:
        matches = [r for r in m.rows(table) if _iso(start) <= r[column] <= _iso(end)]
        out.append(
            {
                "text": text,
                "route": QueryRoute.AGGREGATE,
                "gold": GoldTarget(),
                "gold_scalar": len(matches),
                "gold_scalar_kind": "count",
                "meta": {"table": table, "column": column, "window": [_iso(start), _iso(end)]},
            }
        )
    return out


# -- 6. join / relationship ------------------------------------------------


@dev_template("join_order_customer", "join_relationship", 14)
def _t_join_order_customer(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    orders = m.rows("orders")
    customers = m.by_pk("customers")
    out: list[dict[str, Any]] = []
    phrasings = (
        "Which customer placed order {oid}?",
        "What is the email address of the customer who placed order {oid}?",
        "Which customer segment does the buyer on order {oid} belong to?",
    )
    for i, order in enumerate(_sample(rng, orders, n)):
        cust = customers[str(order["customer_id"])]
        out.append(
            {
                "text": phrasings[i % len(phrasings)].format(oid=order["order_id"]),
                "route": QueryRoute.HYBRID,
                "gold": GoldTarget.of_rows(
                    [_ref(m, "orders", order), _ref(m, "customers", cust)]
                ),
                "meta": {
                    "tables": ["orders", "customers"],
                    "order_id": order["order_id"],
                    "customer": cust["full_name"],
                },
            }
        )
    return out


@dev_template("join_order_region", "join_relationship", 10)
def _t_join_order_region(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    orders = m.rows("orders")
    customers = m.by_pk("customers")
    regions = m.by_pk("regions")
    out: list[dict[str, Any]] = []
    for order in _sample(rng, orders, n):
        cust = customers[str(order["customer_id"])]
        region = regions[str(cust["region_id"])]
        out.append(
            {
                "text": f"Which region is the customer who placed order {order['order_id']} based in?",
                "route": QueryRoute.HYBRID,
                "gold": GoldTarget.of_rows(
                    [
                        _ref(m, "orders", order),
                        _ref(m, "customers", cust),
                        _ref(m, "regions", region),
                    ]
                ),
                "meta": {
                    "tables": ["orders", "customers", "regions"],
                    "expected_value": region["name"],
                },
            }
        )
    return out


@dev_template("join_count_by_region", "join_relationship", 8)
def _t_join_count_region(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    regions = m.rows("regions")
    customers = m.rows("customers")
    orders = m.rows("orders")
    by_region: dict[int, set[int]] = {}
    for c in customers:
        by_region.setdefault(c["region_id"], set()).add(c["customer_id"])

    out: list[dict[str, Any]] = []
    for region in _sample(rng, regions, n):
        cids = by_region.get(region["region_id"], set())
        count = sum(1 for o in orders if o["customer_id"] in cids)
        out.append(
            {
                "text": f"How many orders were placed by customers in the {region['name']} region?",
                "route": QueryRoute.HYBRID,
                # Retrieval gold is the *dimension* row the question names: the
                # contributing orders number in the hundreds and cannot be a
                # top-10 target, but the region row is real supporting evidence.
                "gold": GoldTarget.of_rows([_ref(m, "regions", region)]),
                "gold_scalar": count,
                "gold_scalar_kind": "count",
                "meta": {
                    "tables": ["orders", "customers", "regions"],
                    "gold_basis": "dimension_rows",
                    "region": region["name"],
                },
            }
        )
    return out


@dev_template("join_product_quantity", "join_relationship", 8)
def _t_join_product_quantity(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    products = m.rows("products")
    items = m.rows("order_items")
    qty: dict[int, int] = {}
    for it in items:
        qty[it["product_id"]] = qty.get(it["product_id"], 0) + it["quantity"]
    ordered = [p for p in products if qty.get(p["product_id"])]

    out: list[dict[str, Any]] = []
    for product in _sample(rng, ordered, n):
        out.append(
            {
                "text": (
                    f"What is the total quantity ordered of the product "
                    f"'{product['name']}'?"
                ),
                "route": QueryRoute.HYBRID,
                "gold": GoldTarget.of_rows([_ref(m, "products", product)]),
                "gold_scalar": qty[product["product_id"]],
                "gold_scalar_kind": "count",
                "meta": {
                    "tables": ["order_items", "products"],
                    "gold_basis": "dimension_rows",
                    "product": product["name"],
                },
            }
        )
    return out


# -- 7. schema questions ---------------------------------------------------


@dev_template("schema_columns", "schema_question", 24)
def _t_schema(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    tables = list(m.table_names)
    phrasings: tuple[tuple[str, str], ...] = (
        ("columns", "What columns does the {table} table have?"),
        ("columns", "Which fields are stored for {table}?"),
        ("primary_key", "What is the primary key column of the {table} table?"),
        ("nullable", "Which columns in the {table} table can contain missing values?"),
        ("foreign_keys", "Which columns in {table} reference another table?"),
    )
    combos = [(t, p) for t in tables for p in phrasings]
    rng.shuffle(combos)

    out: list[dict[str, Any]] = []
    for table, (aspect, phrasing) in combos:
        if len(out) >= n:
            break
        specs = m.column_specs(table)
        expected: Any
        if aspect == "columns":
            expected = [c["name"] for c in specs]
        elif aspect == "primary_key":
            expected = m.pk_col(table)
        elif aspect == "nullable":
            expected = [c["name"] for c in specs if c["nullable"]]
        else:
            expected = [c["name"] for c in specs if c["references"]]
            if not expected:
                continue
        out.append(
            {
                "text": phrasing.format(table=table),
                "route": QueryRoute.LOOKUP,
                # Gold is the schema *card*, not any row: this is the one type
                # that would score zero on a row-chunks-only ablation cell, which
                # is exactly what the chunk-kind axis is there to expose.
                "gold": GoldTarget.of_schema([table]),
                "meta": {"table": table, "aspect": aspect, "expected_value": expected},
            }
        )
    return out


# -- 8. distractors / unanswerable ----------------------------------------


#: Edits that produce a *plausible* alternative spelling rather than a typo.
#: A random letter substitution gives "Faisal Mdnsour", which any system can
#: reject on sight; "Faisal Mansor" is the near-miss that actually tests whether
#: near-match retrieval is being confused for an exact-match answer.
_PLAUSIBLE_EDITS: tuple[tuple[str, str], ...] = (
    ("ou", "u"), ("u", "ou"), ("ss", "s"), ("dd", "d"), ("ll", "l"),
    ("tt", "t"), ("rr", "r"), ("nn", "n"), ("mm", "m"), ("kh", "k"),
    ("ph", "f"), ("ie", "ee"), ("ee", "ea"), ("oo", "o"), ("o", "oo"),
    ("ei", "ai"), ("ai", "ei"), ("y", "i"), ("i", "y"), ("z", "s"),
    ("c", "k"), ("q", "k"), ("a", "e"), ("e", "a"),
)


def _plausible_variants(word: str) -> list[str]:
    """Alternative spellings of one word, in a deterministic order."""
    lowered = word.lower()
    out: list[str] = []
    for src, dst in _PLAUSIBLE_EDITS:
        idx = lowered.find(src, 1)
        if idx < 0:
            continue
        candidate = word[:idx] + dst + word[idx + len(src) :]
        if candidate != word and candidate not in out:
            out.append(candidate)
    # Transliteration endings: "Ajami" -> "Ajamy" is covered above; dropping or
    # adding a trailing vowel is the other common one.
    if len(word) > 3 and word[-1].lower() in "aeiou":
        out.append(word[:-1])
    elif len(word) > 3:
        out.append(word + "i")
    return out


def _perturb_name(rng: random.Random, name: str, pool: Sequence[str]) -> str | None:
    """A plausible near-miss spelling of `name` that is absent from `pool`.

    Returns None when every variant collides with something real -- which is a
    correct outcome, not a failure: the caller simply picks another base name.
    """
    words = name.split(" ")
    order = list(range(len(words)))
    # Prefer editing the last word (the surname / head noun), then the others.
    order.sort(key=lambda i: (i != len(words) - 1, i))
    normalised_pool = {_norm_name(p) for p in pool}
    for i in order:
        variants = _plausible_variants(words[i])
        rng.shuffle(variants)
        for variant in variants:
            candidate = " ".join(words[:i] + [variant] + words[i + 1 :])
            if _norm_name(candidate) not in normalised_pool:
                return candidate
    return None


@dev_template("dist_near_customer", "distractor", 12)
def _t_dist_near_customer(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    names = [c["full_name"] for c in m.rows("customers")]
    phrasings = (
        "What is the email address of {name}?",
        "Which customer segment is {name} in?",
        "What loyalty tier does {name} hold?",
    )
    out: list[dict[str, Any]] = []
    used: set[str] = set()
    guard = 0
    while len(out) < n and guard < n * 60:
        guard += 1
        base = rng.choice(names)
        fake = _perturb_name(rng, base, names)
        if fake is None or fake in used:
            continue
        used.add(fake)
        _assert_absent(fake, names, "customer name")
        out.append(
            {
                "text": phrasings[len(out) % len(phrasings)].format(name=fake),
                "route": QueryRoute.LOOKUP,
                "answerable": False,
                "gold": GoldTarget(),
                "meta": {
                    "kind": "near_miss_entity",
                    "fake_value": fake,
                    "nearest_real": base,
                    "table": "customers",
                },
            }
        )
    return out


@dev_template("dist_near_product", "distractor", 8)
def _t_dist_near_product(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    names = [p["name"] for p in m.rows("products")]
    phrasings = (
        "Which category does the product '{name}' belong to?",
        "What is the unit price of the product '{name}'?",
    )
    out: list[dict[str, Any]] = []
    used: set[str] = set()
    guard = 0
    while len(out) < n and guard < n * 60:
        guard += 1
        base = rng.choice(names)
        fake = _perturb_name(rng, base, names)
        if fake is None or fake in used:
            continue
        used.add(fake)
        _assert_absent(fake, names, "product name")
        out.append(
            {
                "text": phrasings[len(out) % len(phrasings)].format(name=fake),
                "route": QueryRoute.LOOKUP,
                "answerable": False,
                "gold": GoldTarget(),
                "meta": {
                    "kind": "near_miss_entity",
                    "fake_value": fake,
                    "nearest_real": base,
                    "table": "products",
                },
            }
        )
    return out


#: Categories one letter (or one plural) off. "Stationary"/"Stationery" is the
#: sharpest: it is a real English word, so lexical retrieval will not flag it.
_FAKE_CATEGORIES: tuple[str, ...] = (
    "Stationary", "Electronic", "Groceries", "Apparels",
    "Furnishings", "Hardwares", "Elektronics", "Furnitur",
)


@dev_template("dist_near_category", "distractor", 8)
def _t_dist_near_category(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    real = [str(v) for v in m.distinct("products", "category")]
    phrasings = (
        "How many products are in the {value} category?",
        "What is the average unit price of products in the {value} category?",
    )
    out: list[dict[str, Any]] = []
    for i, fake in enumerate(_sample(rng, _FAKE_CATEGORIES, n)):
        _assert_absent(fake, real, "product category")
        out.append(
            {
                "text": phrasings[i % len(phrasings)].format(value=fake),
                "route": QueryRoute.AGGREGATE,
                "answerable": False,
                "gold": GoldTarget(),
                "meta": {
                    "kind": "near_miss_category",
                    "fake_value": fake,
                    "table": "products",
                    "column": "category",
                },
            }
        )
    return out


_FAKE_ENUMS: tuple[tuple[str, str, str, str], ...] = (
    ("orders", "status", "refunded", "How many orders have the status 'refunded'?"),
    ("orders", "status", "backordered", "How many orders are currently backordered?"),
    ("orders", "channel", "kiosk", "How many orders came through the kiosk channel?"),
    ("orders", "channel", "fax", "How many orders were placed by fax?"),
    ("customers", "loyalty_tier", "Diamond", "How many customers hold a Diamond loyalty tier?"),
    ("customers", "segment", "Nonprofit", "How many customers belong to the Nonprofit segment?"),
)


@dev_template("dist_near_enum", "distractor", 6)
def _t_dist_near_enum(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for table, column, fake, text in _sample(rng, _FAKE_ENUMS, n):
        _assert_absent(fake, [str(v) for v in m.distinct(table, column)], f"{table}.{column}")
        out.append(
            {
                "text": text,
                "route": QueryRoute.AGGREGATE,
                "answerable": False,
                "gold": GoldTarget(),
                "meta": {
                    "kind": "near_miss_enum",
                    "fake_value": fake,
                    "table": table,
                    "column": column,
                },
            }
        )
    return out


_FAKE_REGIONS: tuple[str, ...] = ("Riyad", "Jedda", "Dubay", "Kairo", "Damman", "Ammon")


@dev_template("dist_near_region", "distractor", 4)
def _t_dist_near_region(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    real = [str(v) for v in m.distinct("regions", "name")]
    phrasings = (
        "Which country is the {value} region in?",
        "How many customers are based in the {value} region?",
    )
    out: list[dict[str, Any]] = []
    for i, fake in enumerate(_sample(rng, _FAKE_REGIONS, n)):
        _assert_absent(fake, real, "region name")
        out.append(
            {
                "text": phrasings[i % len(phrasings)].format(value=fake),
                "route": QueryRoute.LOOKUP if i % 2 == 0 else QueryRoute.HYBRID,
                "answerable": False,
                "gold": GoldTarget(),
                "meta": {"kind": "near_miss_region", "fake_value": fake, "table": "regions"},
            }
        )
    return out


_FAKE_ATTRIBUTES: tuple[tuple[str, str, str], ...] = (
    ("customers", "phone number", "What is the phone number of {entity}?"),
    ("customers", "VAT registration number", "What is the VAT registration number of {entity}?"),
    ("customers", "date of birth", "What is the date of birth of {entity}?"),
    ("products", "shipping weight", "What is the shipping weight of the product '{entity}'?"),
    ("products", "warranty period", "How long is the warranty period on the product '{entity}'?"),
    ("products", "country of manufacture", "In which country is the product '{entity}' manufactured?"),
)


@dev_template("dist_missing_attribute", "distractor", 10)
def _t_dist_missing_attribute(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    """Real entity, attribute that does not exist anywhere in the schema.

    The hardest class in the set: every retriever will return exactly the right
    row, and the system still has to refuse because the column is not there.
    """
    customers = [c["full_name"] for c in m.rows("customers")]
    products = [p["name"] for p in m.rows("products")]
    out: list[dict[str, Any]] = []
    for i in range(n):
        table, attribute, phrasing = _FAKE_ATTRIBUTES[i % len(_FAKE_ATTRIBUTES)]
        entity = rng.choice(customers if table == "customers" else products)
        if attribute in m.columns(table):  # pragma: no cover - guards a schema edit
            raise AssertionError(f"{attribute!r} exists on {table}; not a distractor")
        out.append(
            {
                "text": phrasing.format(entity=entity),
                "route": QueryRoute.LOOKUP,
                "answerable": False,
                "gold": GoldTarget(),
                "meta": {
                    "kind": "missing_attribute",
                    "table": table,
                    "attribute": attribute,
                    "real_entity": entity,
                },
            }
        )
    return out


_FAKE_TABLES: tuple[tuple[str, str, QueryRoute], ...] = (
    ("suppliers", "How many suppliers deliver to the {region} region?", QueryRoute.AGGREGATE),
    ("warehouses", "Which warehouse stores the product '{product}'?", QueryRoute.LOOKUP),
    ("employees", "Which employee is the account manager for {customer}?", QueryRoute.LOOKUP),
    ("invoices", "What is the total value of invoices issued in 2024?", QueryRoute.AGGREGATE),
    ("support_tickets", "How many support tickets were opened by {customer}?", QueryRoute.AGGREGATE),
    ("campaigns", "Which marketing campaign brought in the {segment} segment?", QueryRoute.LOOKUP),
)


@dev_template("dist_missing_table", "distractor", 6)
def _t_dist_missing_table(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    regions = [r["name"] for r in m.rows("regions")]
    products = [p["name"] for p in m.rows("products")]
    customers = [c["full_name"] for c in m.rows("customers")]
    segments = [str(v) for v in m.distinct("customers", "segment")]
    out: list[dict[str, Any]] = []
    for table, phrasing, route in _sample(rng, _FAKE_TABLES, n):
        if table in m.table_names:  # pragma: no cover - guards a schema edit
            raise AssertionError(f"table {table!r} exists; not a distractor")
        out.append(
            {
                "text": phrasing.format(
                    region=rng.choice(regions),
                    product=rng.choice(products),
                    customer=rng.choice(customers),
                    segment=rng.choice(segments),
                ),
                "route": route,
                "answerable": False,
                "gold": GoldTarget(),
                "meta": {"kind": "missing_table", "table": table},
            }
        )
    return out


@dev_template("dist_out_of_range_date", "distractor", 10)
def _t_dist_out_of_range_date(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    """Windows entirely outside the data.

    Phrased so the empty result is *undefined*, never legitimately zero. "How
    many orders were placed in March 2025" is excluded on purpose: zero is the
    correct answer to it, so answering would not be a false answer.
    """
    orders = m.rows("orders")
    windows: list[tuple[date, date]] = [
        (date(2025, 1, 1), date(2025, 3, 31)),
        (date(2025, 4, 1), date(2025, 9, 30)),
        (date(2021, 1, 1), date(2021, 12, 31)),
        (date(2022, 6, 1), date(2022, 12, 31)),
        (date(2026, 1, 1), date(2026, 6, 30)),
        (date(2020, 5, 1), date(2020, 11, 30)),
    ]
    phrasings = (
        ("Which orders were placed between {a} and {b}?", QueryRoute.LOOKUP),
        ("What was the average order discount between {a} and {b}?", QueryRoute.AGGREGATE),
        ("What was the highest order discount between {a} and {b}?", QueryRoute.AGGREGATE),
    )
    out: list[dict[str, Any]] = []
    for i in range(n):
        start, end = windows[i % len(windows)]
        phrasing, route = phrasings[i % len(phrasings)]
        matches = [o for o in orders if _iso(start) <= o["order_date"] <= _iso(end)]
        if matches:  # pragma: no cover - guards a change to the date span
            raise AssertionError(
                f"window {start}..{end} contains {len(matches)} orders; not a distractor"
            )
        out.append(
            {
                "text": phrasing.format(a=_iso(start), b=_iso(end)),
                "route": route,
                "answerable": False,
                "gold": GoldTarget(),
                "meta": {
                    "kind": "out_of_range_date",
                    "window": [_iso(start), _iso(end)],
                    "data_span": [DATE_MIN.isoformat(), DATE_MAX.isoformat()],
                },
            }
        )
    return out


# ==========================================================================
# HELD-OUT TEMPLATES
# ==========================================================================
#
# Written after the router freeze (see ROUTER_FREEZE above), against the same
# eight question types, with materially different sentence shapes -- not
# paraphrases of the dev wording.
#
# What changes, deliberately, on the answerable types:
#   * different question openings (imperative, possessive, inverted, trailing)
#   * different constraint *ordering* inside multi-attribute filters
#   * date windows described in prose ("the first half of 2024") as well as ISO
#   * relationships traversed from the other end ("Order 12 was placed by whom")
#
# What changes on the distractors: every one of them is a form a real user would
# type, chosen to separate two hypotheses about the refusal guards --
#
#   H1 (the claim): the guards key on the *schema* -- a value that is not in a
#       column's vocabulary, an attribute no table has, an entity type that is
#       not a table -- and therefore transfer to any phrasing.
#   H2 (the risk):  the guards key on the *surface patterns* that happened to
#       appear in the dev set, and a different phrasing of the same semantic
#       question walks straight past them.
#
# Several templates below deliberately contain both a form the guard's pattern
# covers and a form it does not, so the per-template breakdown separates "the
# guard is wrong" from "the guard never fired". Nothing here is contorted to
# break the router: every question is one a user could plausibly ask, and a
# question that is only unanswerable because it is written strangely would prove
# nothing about generalisation.


# -- 1. entity lookup, re-phrased -----------------------------------------


@heldout_template("ho_ent_customer", "entity_lookup", 24)
def _h_ent_customer(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    """Possessive, imperative and inverted forms of the dev attribute lookups."""
    customers = m.rows("customers")
    phrasings: tuple[tuple[str, str], ...] = (
        ("email", "What is {name}'s email address?"),
        ("segment", "{name} belongs to which customer segment?"),
        ("loyalty_tier", "Tell me the loyalty tier held by {name}."),
        ("signup_date", "On which date was {name} registered as a customer?"),
        ("credit_limit", "What credit limit applies to {name}?"),
        ("full_name_ar", "How is {name} written in Arabic?"),
    )
    clusters: dict[str, list[Mapping[str, Any]]] = {}
    for row in customers:
        clusters.setdefault(_norm_name(row["full_name"])[-8:], []).append(row)
    confusable = [r for r in customers if len(clusters[_norm_name(r["full_name"])[-8:]]) > 1]
    plain = [r for r in customers if r not in confusable]

    out: list[dict[str, Any]] = []
    half = n // 2
    for i, row in enumerate(_sample(rng, confusable, half) + _sample(rng, plain, n - half)):
        attr, phrasing = phrasings[i % len(phrasings)]
        if row.get(attr) is None:
            populated = [a for a, _ in phrasings if row.get(a) is not None]
            if not populated:
                continue
            attr = populated[0]
            phrasing = dict(phrasings)[attr]
        out.append(
            {
                "text": phrasing.format(name=row["full_name"]),
                "route": QueryRoute.LOOKUP,
                "gold": GoldTarget.of_rows([_ref(m, "customers", row)]),
                "meta": {
                    "table": "customers",
                    "attribute": attr,
                    "expected_value": row[attr],
                    "confusable": row in confusable,
                },
            }
        )
    return out


@heldout_template("ho_ent_product", "entity_lookup", 20)
def _h_ent_product(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    products = m.rows("products")
    phrasings: tuple[tuple[str, str], ...] = (
        ("category", "Under which category is '{name}' sold?"),
        ("unit_price", "What does one unit of '{name}' sell for?"),
        ("launch_date", "Give me the launch date for '{name}'."),
        ("subcategory", "'{name}' sits in which subcategory?"),
    )
    out: list[dict[str, Any]] = []
    for i, row in enumerate(_sample(rng, products, n)):
        attr, phrasing = phrasings[i % len(phrasings)]
        if row.get(attr) is None:
            attr, phrasing = phrasings[0]
        out.append(
            {
                "text": phrasing.format(name=row["name"]),
                "route": QueryRoute.LOOKUP,
                "gold": GoldTarget.of_rows([_ref(m, "products", row)]),
                "meta": {"table": "products", "attribute": attr, "expected_value": row[attr]},
            }
        )
    return out


@heldout_template("ho_ent_region", "entity_lookup", 12)
def _h_ent_region(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    regions = m.rows("regions")
    phrasings: tuple[tuple[str, str], ...] = (
        ("country", "The {name} region sits in which country?"),
        ("timezone", "Which timezone applies in {name}?"),
        ("name_ar", "Give me the Arabic spelling of {name}."),
    )
    out: list[dict[str, Any]] = []
    for i, row in enumerate(_sample(rng, regions, n)):
        attr, phrasing = phrasings[i % len(phrasings)]
        if row.get(attr) is None:
            attr, phrasing = phrasings[0]
        out.append(
            {
                "text": phrasing.format(name=row["name"]),
                "route": QueryRoute.LOOKUP,
                "gold": GoldTarget.of_rows([_ref(m, "regions", row)]),
                "meta": {"table": "regions", "attribute": attr, "expected_value": row[attr]},
            }
        )
    return out


# -- 2. multi-attribute filters, re-ordered -------------------------------


@heldout_template("ho_filt_customers", "multi_filter_lookup", 20)
def _h_filt_customers(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    """Same three constraints as dev, named in the opposite order.

    Dev says region -> segment -> tier. If constraint extraction is positional
    rather than lexical, tier -> region -> segment breaks it.
    """
    customers = m.rows("customers")
    regions = {r["region_id"]: r for r in m.rows("regions")}
    segments = sorted({c["segment"] for c in customers})
    tiers = sorted({c["loyalty_tier"] for c in customers if c["loyalty_tier"]})
    combos = [(rid, seg, tier) for rid in sorted(regions) for seg in segments for tier in tiers]
    rng.shuffle(combos)

    phrasings = (
        "Show me {tier}-tier customers who are based in {region} and classified as {seg}.",
        "Which {seg}-segment accounts in {region} hold {tier} status?",
    )
    out: list[dict[str, Any]] = []
    for rid, seg, tier in combos:
        matches = _where(customers, region_id=rid, segment=seg, loyalty_tier=tier)
        if not (1 <= len(matches) <= 8):
            continue
        out.append(
            {
                "text": phrasings[len(out) % len(phrasings)].format(
                    tier=tier, region=regions[rid]["name"], seg=seg
                ),
                "route": QueryRoute.LOOKUP,
                "gold": GoldTarget.of_rows(_refs(m, "customers", matches)),
                "meta": {
                    "table": "customers",
                    "filters": {
                        "region": regions[rid]["name"],
                        "segment": seg,
                        "loyalty_tier": tier,
                    },
                    "n_matches": len(matches),
                },
            }
        )
        if len(out) >= n:
            break
    return out


@heldout_template("ho_filt_products", "multi_filter_lookup", 12)
def _h_filt_products(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    products = m.rows("products")
    categories = sorted({p["category"] for p in products})
    # Different thresholds from dev, so the underlying row sets differ too.
    thresholds = (75.0, 150.0, 400.0, 750.0)
    combos = [(c, t, s) for c in categories for t in thresholds for s in (0, 1)]
    rng.shuffle(combos)

    out: list[dict[str, Any]] = []
    for cat, thresh, stock in combos:
        matches = [
            p for p in products
            if p["category"] == cat and p["unit_price"] < thresh and p["in_stock"] == stock
        ]
        if not (1 <= len(matches) <= 8):
            continue
        availability = "still available" if stock else "no longer available"
        out.append(
            {
                "text": (
                    f"List the items from the {cat} range that sell for less than "
                    f"{thresh:.0f} and are {availability}."
                ),
                "route": QueryRoute.LOOKUP,
                "gold": GoldTarget.of_rows(_refs(m, "products", matches)),
                "meta": {
                    "table": "products",
                    "filters": {"category": cat, "unit_price_lt": thresh, "in_stock": stock},
                    "n_matches": len(matches),
                },
            }
        )
        if len(out) >= n:
            break
    return out


@heldout_template("ho_filt_orders", "multi_filter_lookup", 8)
def _h_filt_orders(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    orders = m.rows("orders")
    statuses = sorted({o["status"] for o in orders})
    channels = sorted({o["channel"] for o in orders})
    cutoffs = (20.0, 26.0, 29.0, 33.0)
    combos = [(s, c, d) for s in statuses for c in channels for d in cutoffs]
    rng.shuffle(combos)

    out: list[dict[str, Any]] = []
    for status, channel, cutoff in combos:
        matches = [
            o for o in orders
            if o["status"] == status and o["channel"] == channel
            and o["discount_pct"] is not None and o["discount_pct"] > cutoff
        ]
        if not (1 <= len(matches) <= 8):
            continue
        out.append(
            {
                "text": (
                    f"Find the orders where the discount exceeded {cutoff:.0f} percent, "
                    f"the {channel} channel was used, and the status is {status}."
                ),
                "route": QueryRoute.LOOKUP,
                "gold": GoldTarget.of_rows(_refs(m, "orders", matches)),
                "meta": {
                    "table": "orders",
                    "filters": {"status": status, "channel": channel, "discount_gt": cutoff},
                    "n_matches": len(matches),
                },
            }
        )
        if len(out) >= n:
            break
    return out


# -- 3. count by category, re-phrased -------------------------------------

_HO_COUNT_SPECS: tuple[tuple[str, str, str], ...] = (
    ("orders", "status", "Orders with status '{value}' — how many are there?"),
    ("orders", "status", "What is the count of orders sitting at '{value}'?"),
    ("orders", "channel", "Give me a tally of orders that used the {value} channel."),
    ("orders", "channel", "Of all the orders, how many went through {value}?"),
    ("customers", "segment", "What is the count of customers in the {value} segment?"),
    ("customers", "segment", "In the {value} segment, how many customers are there?"),
    ("customers", "loyalty_tier", "How many customers sit at the {value} tier?"),
    ("products", "category", "Within the {value} category, how many products are listed?"),
    ("products", "category", "What is the size of the {value} category?"),
    ("products", "subcategory", "How many products fall under the {value} subcategory?"),
)


@heldout_template("ho_cnt_by_category", "count_by_category", 40)
def _h_count_by_category(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    candidates: list[tuple[str, str, str, Any]] = []
    for table, column, phrasing in _HO_COUNT_SPECS:
        for value in sorted(str(v) for v in m.distinct(table, column)):
            candidates.append((table, column, phrasing, value))
    rng.shuffle(candidates)

    out: list[dict[str, Any]] = []
    for table, column, phrasing, value in candidates[:n]:
        matches = [r for r in m.rows(table) if str(r.get(column)) == value]
        out.append(
            {
                "text": phrasing.format(value=value),
                "route": QueryRoute.AGGREGATE,
                "gold": GoldTarget(),
                "gold_scalar": len(matches),
                "gold_scalar_kind": "count",
                "meta": {"table": table, "column": column, "value": value},
            }
        )
    return out


# -- 4. numeric aggregates, re-phrased ------------------------------------

_HO_AGG_PHRASINGS: dict[str, str] = {
    "avg": "Across {scope}, what does the {label} average out to?",
    "sum": "Summed across {scope}, what do the {label}s come to?",
    "min": "Among {scope}, which {label} is the smallest?",
    "max": "Among {scope}, which {label} is the largest?",
}


@heldout_template("ho_num_products", "numeric_aggregate", 24)
def _h_num_products(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    products = m.rows("products")
    categories = sorted({p["category"] for p in products})
    combos = [(c, f) for c in categories for f in ("avg", "sum", "min", "max")]
    rng.shuffle(combos)

    out: list[dict[str, Any]] = []
    for cat, fn in combos[:n]:
        vals = _non_null([p["unit_price"] for p in products if p["category"] == cat])
        gold = _agg(vals, fn)
        if gold is None:
            continue
        out.append(
            {
                "text": _HO_AGG_PHRASINGS[fn].format(
                    label="unit price", scope=f"the {cat} range"
                ),
                "route": QueryRoute.AGGREGATE,
                "gold": GoldTarget(),
                "gold_scalar": float(gold),
                "gold_scalar_kind": fn,
                "meta": {
                    "table": "products",
                    "column": "unit_price",
                    "group": {"category": cat},
                    "n_contributing_rows": len(vals),
                },
            }
        )
    return out


@heldout_template("ho_num_customers", "numeric_aggregate", 12)
def _h_num_customers(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    customers = m.rows("customers")
    segments = sorted({c["segment"] for c in customers})
    combos = [(s, f) for s in segments for f in ("avg", "max", "min", "sum")]
    rng.shuffle(combos)

    out: list[dict[str, Any]] = []
    for seg, fn in combos[:n]:
        vals = _non_null([c["credit_limit"] for c in customers if c["segment"] == seg])
        gold = _agg(vals, fn)
        if gold is None:
            continue
        out.append(
            {
                "text": _HO_AGG_PHRASINGS[fn].format(
                    label="credit limit", scope=f"{seg} accounts"
                ),
                "route": QueryRoute.AGGREGATE,
                "gold": GoldTarget(),
                "gold_scalar": float(gold),
                "gold_scalar_kind": fn,
                "meta": {
                    "table": "customers",
                    "column": "credit_limit",
                    "group": {"segment": seg},
                    "n_contributing_rows": len(vals),
                    "null_sensitive": True,
                },
            }
        )
    return out


@heldout_template("ho_num_items", "numeric_aggregate", 8)
def _h_num_items(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    orders = m.rows("orders")
    items = m.rows("order_items")
    specs: list[tuple[str, str, str, str, list[Any]]] = []
    for status in sorted({o["status"] for o in orders}):
        specs.append(
            ("avg", "discount percentage", f"{status} orders", "orders",
             _non_null([o["discount_pct"] for o in orders if o["status"] == status]))
        )
    for fn in ("avg", "max", "min", "sum"):
        specs.append((fn, "line total", "every order line", "order_items",
                      _non_null([i["line_total"] for i in items])))
    for fn in ("avg", "max"):
        specs.append((fn, "quantity", "every order line", "order_items",
                      _non_null([i["quantity"] for i in items])))
    rng.shuffle(specs)

    out: list[dict[str, Any]] = []
    for fn, label, scope, table, vals in specs[:n]:
        gold = _agg(vals, fn)
        if gold is None:
            continue
        out.append(
            {
                "text": _HO_AGG_PHRASINGS[fn].format(label=label, scope=scope),
                "route": QueryRoute.AGGREGATE,
                "gold": GoldTarget(),
                "gold_scalar": float(gold),
                "gold_scalar_kind": fn,
                "meta": {"table": table, "n_contributing_rows": len(vals)},
            }
        )
    return out


# -- 5. date-range aggregates, described in prose -------------------------


def _period_windows(year: int) -> list[tuple[str, date, date]]:
    """Calendar periods as a human would name them, not as an ISO pair."""
    return [
        ("the first half of", date(year, 1, 1), date(year, 6, 30)),
        ("the second half of", date(year, 7, 1), date(year, 12, 31)),
        ("the first quarter of", date(year, 1, 1), date(year, 3, 31)),
        ("the second quarter of", date(year, 4, 1), date(year, 6, 30)),
        ("the third quarter of", date(year, 7, 1), date(year, 9, 30)),
        ("the fourth quarter of", date(year, 10, 1), date(year, 12, 31)),
    ]


@heldout_template("ho_date_periods", "date_range_aggregate", 18)
def _h_date_periods(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    """Windows named as halves and quarters rather than as ISO endpoints.

    Dev used ISO pairs and "in <Month> <Year>", both of which the date slot
    parses. "The first half of 2024" is an equally ordinary way to ask, and
    whether it is covered is a genuine coverage question rather than a trick.
    """
    orders = m.rows("orders")
    specs = [
        (label, start, end)
        for year in (2023, 2024)
        for label, start, end in _period_windows(year)
    ]
    rng.shuffle(specs)
    phrasings = (
        "How many orders came in during {label} {year}?",
        "Counting only {label} {year}, how many orders were there?",
        "During {label} {year}, how many orders were logged?",
    )
    out: list[dict[str, Any]] = []
    for i, (label, start, end) in enumerate(_sample(rng, specs, n)):
        matches = [o for o in orders if _iso(start) <= o["order_date"] <= _iso(end)]
        out.append(
            {
                "text": phrasings[i % len(phrasings)].format(label=label, year=start.year),
                "route": QueryRoute.AGGREGATE,
                "gold": GoldTarget(),
                "gold_scalar": len(matches),
                "gold_scalar_kind": "count",
                "meta": {
                    "table": "orders",
                    "column": "order_date",
                    "window": [_iso(start), _iso(end)],
                    "window_form": "named_period",
                },
            }
        )
    return out


@heldout_template("ho_date_spans", "date_range_aggregate", 12)
def _h_date_spans(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    """ISO endpoints, but in a leading subordinate clause rather than trailing."""
    orders = m.rows("orders")
    span = (DATE_MAX - DATE_MIN).days
    phrasings = (
        "Over the period from {a} to {b}, how many orders were recorded?",
        "Restricting to {a} through {b}, what is the order count?",
        "Between {a} and {b} inclusive, how many orders were logged?",
    )
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    guard = 0
    while len(out) < n and guard < n * 40:
        guard += 1
        start = DATE_MIN + timedelta(days=rng.randint(0, span - 30))
        end = min(start + timedelta(days=rng.choice((40, 75, 110, 150, 200))), DATE_MAX)
        key = (_iso(start), _iso(end))
        if key in seen:
            continue
        seen.add(key)
        matches = [o for o in orders if _iso(start) <= o["order_date"] <= _iso(end)]
        if not matches:
            continue
        out.append(
            {
                "text": phrasings[len(out) % len(phrasings)].format(a=_iso(start), b=_iso(end)),
                "route": QueryRoute.AGGREGATE,
                "gold": GoldTarget(),
                "gold_scalar": len(matches),
                "gold_scalar_kind": "count",
                "meta": {
                    "table": "orders",
                    "column": "order_date",
                    "window": [_iso(start), _iso(end)],
                    "window_form": "iso_pair",
                },
            }
        )
    return out


@heldout_template("ho_date_joined", "date_range_aggregate", 6)
def _h_date_joined(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    specs: list[tuple[str, str, str, date, date]] = []
    for year in (2023, 2024):
        specs.append(
            ("customers", "signup_date",
             f"How many customers joined during {year}?", date(year, 1, 1), date(year, 12, 31))
        )
        specs.append(
            ("products", "launch_date",
             f"How many products came to market in {year}?",
             date(year, 1, 1), date(year, 12, 31))
        )
    specs.append(
        ("customers", "signup_date",
         "Counting only the second half of 2023, how many customers joined?",
         date(2023, 7, 1), date(2023, 12, 31))
    )
    specs.append(
        ("products", "launch_date",
         "Over the period from 2024-01-01 to 2024-06-30, how many products launched?",
         date(2024, 1, 1), date(2024, 6, 30))
    )
    rng.shuffle(specs)
    out: list[dict[str, Any]] = []
    for table, column, text, start, end in specs[:n]:
        matches = [r for r in m.rows(table) if _iso(start) <= r[column] <= _iso(end)]
        out.append(
            {
                "text": text,
                "route": QueryRoute.AGGREGATE,
                "gold": GoldTarget(),
                "gold_scalar": len(matches),
                "gold_scalar_kind": "count",
                "meta": {"table": table, "column": column, "window": [_iso(start), _iso(end)]},
            }
        )
    return out


# -- 6. joins, traversed from the other end -------------------------------


@heldout_template("ho_join_order_customer", "join_relationship", 14)
def _h_join_order_customer(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    orders = m.rows("orders")
    customers = m.by_pk("customers")
    phrasings = (
        "Order {oid} was placed by which customer?",
        "Who is the buyer on order {oid}, and what is their email address?",
        "The account behind order {oid} sits in which segment?",
    )
    out: list[dict[str, Any]] = []
    for i, order in enumerate(_sample(rng, orders, n)):
        cust = customers[str(order["customer_id"])]
        out.append(
            {
                "text": phrasings[i % len(phrasings)].format(oid=order["order_id"]),
                "route": QueryRoute.HYBRID,
                "gold": GoldTarget.of_rows(
                    [_ref(m, "orders", order), _ref(m, "customers", cust)]
                ),
                "meta": {
                    "tables": ["orders", "customers"],
                    "order_id": order["order_id"],
                    "customer": cust["full_name"],
                },
            }
        )
    return out


@heldout_template("ho_join_order_region", "join_relationship", 10)
def _h_join_order_region(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    orders = m.rows("orders")
    customers = m.by_pk("customers")
    regions = m.by_pk("regions")
    out: list[dict[str, Any]] = []
    for order in _sample(rng, orders, n):
        cust = customers[str(order["customer_id"])]
        region = regions[str(cust["region_id"])]
        out.append(
            {
                "text": f"The customer behind order {order['order_id']} operates out of which region?",
                "route": QueryRoute.HYBRID,
                "gold": GoldTarget.of_rows(
                    [
                        _ref(m, "orders", order),
                        _ref(m, "customers", cust),
                        _ref(m, "regions", region),
                    ]
                ),
                "meta": {
                    "tables": ["orders", "customers", "regions"],
                    "expected_value": region["name"],
                },
            }
        )
    return out


@heldout_template("ho_join_count_region", "join_relationship", 8)
def _h_join_count_region(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    regions = m.rows("regions")
    customers = m.rows("customers")
    orders = m.rows("orders")
    by_region: dict[int, set[int]] = {}
    for c in customers:
        by_region.setdefault(c["region_id"], set()).add(c["customer_id"])

    out: list[dict[str, Any]] = []
    for region in _sample(rng, regions, n):
        cids = by_region.get(region["region_id"], set())
        count = sum(1 for o in orders if o["customer_id"] in cids)
        out.append(
            {
                "text": (
                    f"Across the {region['name']} region, how many orders have "
                    f"customers placed altogether?"
                ),
                "route": QueryRoute.HYBRID,
                "gold": GoldTarget.of_rows([_ref(m, "regions", region)]),
                "gold_scalar": count,
                "gold_scalar_kind": "count",
                "meta": {
                    "tables": ["orders", "customers", "regions"],
                    "gold_basis": "dimension_rows",
                    "region": region["name"],
                },
            }
        )
    return out


@heldout_template("ho_join_product_units", "join_relationship", 8)
def _h_join_product_units(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    products = m.rows("products")
    items = m.rows("order_items")
    qty: dict[int, int] = {}
    for it in items:
        qty[it["product_id"]] = qty.get(it["product_id"], 0) + it["quantity"]
    ordered = [p for p in products if qty.get(p["product_id"])]

    out: list[dict[str, Any]] = []
    for product in _sample(rng, ordered, n):
        out.append(
            {
                "text": f"Summed over every order, how many units of '{product['name']}' were sold?",
                "route": QueryRoute.HYBRID,
                "gold": GoldTarget.of_rows([_ref(m, "products", product)]),
                "gold_scalar": qty[product["product_id"]],
                "gold_scalar_kind": "count",
                "meta": {
                    "tables": ["order_items", "products"],
                    "gold_basis": "dimension_rows",
                    "product": product["name"],
                },
            }
        )
    return out


# -- 7. schema questions, re-phrased --------------------------------------


@heldout_template("ho_schema", "schema_question", 23)
def _h_schema(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    """Some of these hit the router's schema cues, some deliberately do not.

    "Describe the structure of X" is squarely inside `_SCHEMA_CUES`; "List every
    field defined on X" is an equally natural request that is not. Both are
    schema questions to a user, so the split shows whether schema handling is
    driven by the question's *intent* or by a phrase list.
    """
    tables = list(m.table_names)
    phrasings: tuple[tuple[str, str], ...] = (
        ("columns", "Describe the structure of the {table} table."),
        ("columns", "List every field defined on {table}."),
        ("primary_key", "Which column serves as the identifier for {table}?"),
        ("nullable", "In the {table} table, which columns are allowed to be empty?"),
        ("foreign_keys", "Which parts of {table} point at another table?"),
    )
    combos = [(t, p) for t in tables for p in phrasings]
    rng.shuffle(combos)

    out: list[dict[str, Any]] = []
    for table, (aspect, phrasing) in combos:
        if len(out) >= n:
            break
        specs = m.column_specs(table)
        if aspect == "columns":
            expected: Any = [c["name"] for c in specs]
        elif aspect == "primary_key":
            expected = m.pk_col(table)
        elif aspect == "nullable":
            expected = [c["name"] for c in specs if c["nullable"]]
        else:
            expected = [c["name"] for c in specs if c["references"]]
            if not expected:
                continue
        out.append(
            {
                "text": phrasing.format(table=table),
                "route": QueryRoute.LOOKUP,
                "gold": GoldTarget.of_schema([table]),
                "meta": {"table": table, "aspect": aspect, "expected_value": expected},
            }
        )
    return out


# -- 8. distractors that probe the guards' mechanism ----------------------


_HO_POSSESSIVE_ATTRS: tuple[tuple[str, str, str, bool], ...] = (
    # (table, attribute, phrasing, whether the guard's regex frame covers it)
    ("customers", "phone number", "What is {entity}'s phone number?", False),
    ("customers", "date of birth", "Tell me {entity}'s date of birth.", False),
    ("customers", "credit rating", "What's the credit rating for {entity}?", False),
    ("customers", "VAT registration number",
     "What is the VAT registration number for {entity}?", True),
    ("products", "shipping weight", "Give me the shipping weight of '{entity}'.", False),
    ("products", "warranty period", "What's the warranty on '{entity}'?", False),
    ("products", "country of manufacture",
     "What is the country of manufacture of '{entity}'?", True),
)


@heldout_template("ho_dist_possessive_attribute", "distractor", 12)
def _h_dist_possessive_attribute(
    m: Manifest, rng: random.Random, n: int
) -> list[dict[str, Any]]:
    """A real entity, an attribute nothing has, asked possessively.

    The dev set always used "What is the <attr> of <entity>?". These ask the
    same semantic question as a possessive, a contraction, or an imperative.
    Two of the seven keep the dev-style frame as controls, so a uniform failure
    can be told apart from a frame-specific one.
    """
    customers = [c["full_name"] for c in m.rows("customers")]
    products = [p["name"] for p in m.rows("products")]
    out: list[dict[str, Any]] = []
    for i in range(n):
        table, attribute, phrasing, covered = _HO_POSSESSIVE_ATTRS[
            i % len(_HO_POSSESSIVE_ATTRS)
        ]
        entity = rng.choice(customers if table == "customers" else products)
        if attribute in m.columns(table):  # pragma: no cover - guards a schema edit
            raise AssertionError(f"{attribute!r} exists on {table}; not a distractor")
        out.append(
            {
                "text": phrasing.format(entity=entity),
                "route": QueryRoute.LOOKUP,
                "answerable": False,
                "gold": GoldTarget(),
                "meta": {
                    "kind": "missing_attribute",
                    "probe": "possessive_or_imperative_frame",
                    "dev_style_frame": covered,
                    "table": table,
                    "attribute": attribute,
                    "real_entity": entity,
                },
            }
        )
    return out


#: Values absent from the data, named with no column or table word beside them.
#: `unresolved_constraints` anchors on an adjacent column name, and reads only
#: *backwards* from a table name, so none of these has an anchor to hang on.
#: Every one is a question a user would actually type, and each names a state the
#: data does not record -- so "0" is not a safe answer either: `returned` exists,
#: which makes a confident "no orders were refunded" actively misleading.
#: The last entry is the control: "Elektronics **products**" places the unknown
#: value immediately before a *table* name, which `unresolved_constraints` does
#: anchor on (it reads backwards from a table name). It is kept deliberately --
#: within one template, the anchored item and the unanchored ones should behave
#: differently if the guard is pattern-driven, and identically if it is not.
_HO_UNANCHORED: tuple[tuple[str, str, str, bool], ...] = (
    ("orders", "refunded", "How many orders were refunded?", False),
    ("orders", "backordered", "How many orders were backordered?", False),
    ("orders", "expedited", "How many orders were expedited?", False),
    ("orders", "fax", "How many orders arrived by fax?", False),
    ("orders", "kiosk", "How many orders went out through a kiosk?", False),
    ("customers", "Diamond", "How many customers were upgraded to Diamond?", False),
    ("customers", "Copper", "How many customers were downgraded to Copper?", False),
    ("customers", "Nonprofit", "How many customers signed up as Nonprofit?", False),
    ("products", "discontinued", "How many products have been discontinued?", False),
    ("products", "Elektronics", "How many Elektronics products do we carry?", True),
)


@heldout_template("ho_dist_unanchored_value", "distractor", 10)
def _h_dist_unanchored_value(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    """The same semantic distractors as dev, with the column word removed.

    Dev asks "How many orders have the status 'refunded'?" -- the word `status`
    anchors the value to a column whose vocabulary can be checked. Here the same
    question is asked the way people actually ask it. If the guard is checking
    the schema, both refuse; if it is checking for an adjacent column word, only
    the dev form does.
    """
    out: list[dict[str, Any]] = []
    for table, value, text, anchored in _sample(rng, _HO_UNANCHORED, n):
        for column in m.columns(table):
            values = [str(v) for v in m.distinct(table, column)]
            if values:
                _assert_absent(value, values, f"{table}.{column}")
        out.append(
            {
                "text": text,
                "route": QueryRoute.AGGREGATE,
                "answerable": False,
                "gold": GoldTarget(),
                "meta": {
                    "kind": "near_miss_enum",
                    "probe": "value_without_adjacent_column_word",
                    "fake_value": value,
                    "table": table,
                    # True = the value sits next to a table name, which the
                    # guard does anchor on. The in-template control.
                    "anchored_on_table_name": anchored,
                },
            }
        )
    return out


@heldout_template("ho_dist_prose_date", "distractor", 12)
def _h_dist_prose_date(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    """Out-of-range windows written as years, months and quarters, not ISO pairs.

    Same rule as the dev set: nothing here is a plain count, because zero is the
    correct answer to "how many orders in 2026" and scoring that as a
    hallucination would flatter the headline number in our favour. These either
    enumerate rows that do not exist or take an aggregate that is undefined over
    an empty set.
    """
    orders = m.rows("orders")
    specs: tuple[tuple[str, date, date, QueryRoute], ...] = (
        ("Which orders were placed during March 2025?", date(2025, 3, 1), date(2025, 3, 31), QueryRoute.LOOKUP),
        ("What was the average discount on orders from the first quarter of 2026?",
         date(2026, 1, 1), date(2026, 3, 31), QueryRoute.AGGREGATE),
        ("List the orders recorded in 2021.", date(2021, 1, 1), date(2021, 12, 31), QueryRoute.LOOKUP),
        ("What was the highest discount applied during the second half of 2025?",
         date(2025, 7, 1), date(2025, 12, 31), QueryRoute.AGGREGATE),
        ("Which orders came in during November 2022?", date(2022, 11, 1), date(2022, 11, 30), QueryRoute.LOOKUP),
        ("Across 2027, what did the order discount average out to?",
         date(2027, 1, 1), date(2027, 12, 31), QueryRoute.AGGREGATE),
        ("Show me the orders logged in December 2020.", date(2020, 12, 1), date(2020, 12, 31), QueryRoute.LOOKUP),
        ("What was the lowest discount on orders during the fourth quarter of 2026?",
         date(2026, 10, 1), date(2026, 12, 31), QueryRoute.AGGREGATE),
    )
    out: list[dict[str, Any]] = []
    for i in range(n):
        text, start, end, route = specs[i % len(specs)]
        matches = [o for o in orders if _iso(start) <= o["order_date"] <= _iso(end)]
        if matches:  # pragma: no cover - guards a change to the date span
            raise AssertionError(f"window {start}..{end} contains {len(matches)} orders")
        out.append(
            {
                "text": text,
                "route": route,
                "answerable": False,
                "gold": GoldTarget(),
                "meta": {
                    "kind": "out_of_range_date",
                    "probe": "prose_date_not_iso_pair",
                    "window": [_iso(start), _iso(end)],
                    "data_span": [DATE_MIN.isoformat(), DATE_MAX.isoformat()],
                },
            }
        )
    return out


#: Entity types that are not tables. The verb after the entity decides whether
#: `_ENTITY_REQUEST` matches at all, so both covered and uncovered verbs appear.
_HO_FAKE_ENTITIES: tuple[tuple[str, str, bool], ...] = (
    ("shipments", "How many shipments were sent to the {region} region?", True),
    ("suppliers", "How many suppliers serve the {region} region?", False),
    ("vendors", "How many vendors cover the {region} region?", False),
    ("depots", "Which depot stores the product '{product}'?", True),
    ("distributors", "Which distributor handles '{product}'?", True),
    ("carriers", "Which carrier delivered order {oid}?", False),
    ("refunds", "How many refunds exceeded 100 in value?", False),
    ("subscriptions", "How many subscriptions are active right now?", True),
    ("accounts", "How many customer accounts lapsed last year?", False),
    ("resellers", "How many resellers operate in {region}?", False),
)


@heldout_template("ho_dist_fake_entity", "distractor", 10)
def _h_dist_fake_entity(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    """Entity types that do not exist, some near-misses of real table names.

    Dev used "How many suppliers *deliver to* the X region" -- a verb the guard's
    pattern lists. "How many suppliers *serve* the X region" is the same question
    with an ordinary synonym. The `pattern_covered` flag records which is which.
    """
    regions = [r["name"] for r in m.rows("regions")]
    products = [p["name"] for p in m.rows("products")]
    orders = [o["order_id"] for o in m.rows("orders")]
    out: list[dict[str, Any]] = []
    for entity, phrasing, covered in _sample(rng, _HO_FAKE_ENTITIES, n):
        if entity in m.table_names:  # pragma: no cover - guards a schema edit
            raise AssertionError(f"table {entity!r} exists; not a distractor")
        out.append(
            {
                "text": phrasing.format(
                    region=rng.choice(regions),
                    product=rng.choice(products),
                    oid=rng.choice(orders),
                ),
                "route": QueryRoute.AGGREGATE if "how many" in phrasing.lower() else QueryRoute.LOOKUP,
                "answerable": False,
                "gold": GoldTarget(),
                "meta": {
                    "kind": "missing_table",
                    "probe": "entity_verb_outside_pattern",
                    "pattern_covered": covered,
                    "table": entity,
                },
            }
        )
    return out


#: Near-misses of values that dev did not touch: subcategories, countries and
#: timezones rather than categories, statuses and tiers.
_HO_FAKE_VALUES: tuple[tuple[str, str, str, str], ...] = (
    ("products", "subcategory", "Beverage",
     "How many products sit in the Beverage subcategory?"),
    ("products", "subcategory", "Snack", "What is the size of the Snack subcategory?"),
    ("products", "subcategory", "Fastener",
     "Within the Fastener subcategory, how many products are listed?"),
    ("products", "subcategory", "Writings",
     "How many products fall under the Writings subcategory?"),
    ("regions", "country", "Saudia Arabia",
     "How many regions are in the country Saudia Arabia?"),
    ("regions", "country", "Emirates", "Which regions sit in the country Emirates?"),
    ("regions", "timezone", "Asia/Mecca",
     "Which region uses the Asia/Mecca timezone?"),
    ("orders", "channel", "wholesale portal",
     "How many orders used the wholesale portal channel?"),
)


@heldout_template("ho_dist_near_value", "distractor", 12)
def _h_dist_near_value(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    """Near-miss literals in columns the dev distractors never probed."""
    out: list[dict[str, Any]] = []
    pool = _sample(rng, _HO_FAKE_VALUES, min(n, len(_HO_FAKE_VALUES)))
    for table, column, value, text in pool:
        _assert_absent(value, [str(v) for v in m.distinct(table, column)], f"{table}.{column}")
        out.append(
            {
                "text": text,
                "route": QueryRoute.AGGREGATE if "how many" in text.lower() else QueryRoute.LOOKUP,
                "answerable": False,
                "gold": GoldTarget(),
                "meta": {
                    "kind": "near_miss_category",
                    "probe": "untouched_column_vocabulary",
                    "fake_value": value,
                    "table": table,
                    "column": column,
                },
            }
        )

    # Top up with fresh near-miss entity names, perturbed from bases the dev
    # templates did not draw.
    names = [c["full_name"] for c in m.rows("customers")]
    used: set[str] = set()
    guard = 0
    while len(out) < n and guard < n * 60:
        guard += 1
        base = rng.choice(names)
        fake = _perturb_name(rng, base, names)
        if fake is None or fake in used:
            continue
        used.add(fake)
        _assert_absent(fake, names, "customer name")
        out.append(
            {
                "text": f"Which segment does {fake} belong to?",
                "route": QueryRoute.LOOKUP,
                "answerable": False,
                "gold": GoldTarget(),
                "meta": {
                    "kind": "near_miss_entity",
                    "probe": "possessive_free_near_miss",
                    "fake_value": fake,
                    "nearest_real": base,
                    "table": "customers",
                },
            }
        )
    return out


_HO_FAKE_RELATIONSHIPS: tuple[tuple[str, str], ...] = (
    ("region_manufactures_product", "Which region manufactures the product '{product}'?"),
    ("customer_supplies_product", "Which customer supplies the product '{product}'?"),
    ("region_ships_product", "Which region does the product '{product}' ship from?"),
    ("product_belongs_to_region", "Which region is the product '{product}' assigned to?"),
    ("order_has_manager", "Who manages the account behind order {oid}?"),
    ("customer_owns_region", "Which region does {customer} own?"),
    ("product_has_customer", "Which customer is the product '{product}' reserved for?"),
    ("region_has_discount", "What discount rate applies to the {region} region?"),
)


@heldout_template("ho_dist_missing_relationship", "distractor", 8)
def _h_dist_missing_relationship(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    """Two real tables, a relationship the schema does not contain.

    Nothing is misspelled and nothing is missing from the vocabulary: every
    noun in these questions is real. The only thing that is absent is the edge
    between them, which is a different failure from an unknown value and is not
    covered by any of the five guards.
    """
    products = [p["name"] for p in m.rows("products")]
    customers = [c["full_name"] for c in m.rows("customers")]
    regions = [r["name"] for r in m.rows("regions")]
    orders = [o["order_id"] for o in m.rows("orders")]
    out: list[dict[str, Any]] = []
    for key, phrasing in _sample(rng, _HO_FAKE_RELATIONSHIPS, n):
        out.append(
            {
                "text": phrasing.format(
                    product=rng.choice(products),
                    customer=rng.choice(customers),
                    region=rng.choice(regions),
                    oid=rng.choice(orders),
                ),
                "route": QueryRoute.HYBRID,
                "answerable": False,
                "gold": GoldTarget(),
                "meta": {
                    "kind": "missing_relationship",
                    "probe": "real_nouns_absent_edge",
                    "relationship": key,
                },
            }
        )
    return out


_HO_FAKE_TABLES: tuple[tuple[str, str, QueryRoute], ...] = (
    ("payments", "What is the total value of payments received in 2024?", QueryRoute.AGGREGATE),
    ("contracts", "Which contract covers {customer}?", QueryRoute.LOOKUP),
    ("reviews", "What is the average review score for '{product}'?", QueryRoute.AGGREGATE),
    ("returns_log", "Which entry in the returns log matches order {oid}?", QueryRoute.LOOKUP),
    ("budgets", "What budget is allocated to the {region} region?", QueryRoute.LOOKUP),
    ("forecasts", "What is the sales forecast for the {region} region?", QueryRoute.LOOKUP),
)


@heldout_template("ho_dist_missing_table", "distractor", 6)
def _h_dist_missing_table(m: Manifest, rng: random.Random, n: int) -> list[dict[str, Any]]:
    """Fake tables disjoint from the dev set's (suppliers/warehouses/...)."""
    regions = [r["name"] for r in m.rows("regions")]
    products = [p["name"] for p in m.rows("products")]
    customers = [c["full_name"] for c in m.rows("customers")]
    orders = [o["order_id"] for o in m.rows("orders")]
    out: list[dict[str, Any]] = []
    for table, phrasing, route in _sample(rng, _HO_FAKE_TABLES, n):
        if table in m.table_names:  # pragma: no cover - guards a schema edit
            raise AssertionError(f"table {table!r} exists; not a distractor")
        out.append(
            {
                "text": phrasing.format(
                    region=rng.choice(regions),
                    product=rng.choice(products),
                    customer=rng.choice(customers),
                    oid=rng.choice(orders),
                ),
                "route": route,
                "answerable": False,
                "gold": GoldTarget(),
                "meta": {"kind": "missing_table", "probe": "fresh_fake_tables", "table": table},
            }
        )
    return out


# ==========================================================================
# Building
# ==========================================================================


def _derive_seed(seed: int, key: str) -> int:
    """Per-template seed, so adding a template does not reshuffle the others."""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def templates_for(sets: Sequence[str]) -> list[Template]:
    out: list[Template] = []
    for name in sets:
        if name == "dev":
            out.extend(DEV_TEMPLATES)
        elif name == "heldout":
            if not HELDOUT_AVAILABLE:
                raise RuntimeError(
                    "the held-out template set has not been written yet. It is "
                    "written only after route/router.py and route/sqlgen.py are "
                    "frozen (freeze SHA recorded in DECISIONS.md); a held-out set "
                    "written earlier cannot measure overfitting to dev phrasings."
                )
            out.extend(HELDOUT_TEMPLATES)  # pragma: no cover - until the freeze
        else:
            raise ValueError(f"unknown template set {name!r}")
    return out


def resolve_sets(selector: str) -> tuple[str, ...]:
    if selector == "all":
        return ("dev", "heldout") if HELDOUT_AVAILABLE else ("dev",)
    if selector not in TEMPLATE_SET_NAMES:
        raise ValueError(f"unknown template selector {selector!r}")
    return (selector,)


def build_questions(
    manifest: Manifest,
    templates: str | Sequence[str] = "dev",
    seed: int = DEFAULT_SEED,
    limit: int | None = None,
    qtypes: Sequence[str] | None = None,
) -> list[EvalQuestion]:
    """Build the question set. Deterministic in (manifest, templates, seed)."""
    sets = resolve_sets(templates) if isinstance(templates, str) else tuple(templates)
    selected = templates_for(sets)
    if qtypes:
        wanted = set(qtypes)
        selected = [t for t in selected if t.qtype in wanted]

    questions: list[EvalQuestion] = []
    # A template that cycles a short spec list to reach its target n can emit
    # the same sentence twice. A duplicate question is not a second measurement:
    # it double-weights one item and inflates n without adding signal, so the
    # first occurrence wins and the rest are dropped.
    seen_text: set[tuple[str, str]] = set()
    # Counted per (template set, type), not per type: a shared counter makes
    # `heldout-distractor-0001` mean a different question in a heldout-only run
    # than in a `--templates all` run, so a qid in a results file would not
    # identify a question. It has to be stable under what else was built.
    counters: Counter[tuple[str, str]] = Counter()
    for tpl in selected:
        rng = random.Random(_derive_seed(seed, tpl.key))
        for draft in tpl.fn(manifest, rng, tpl.n):
            marker = (tpl.template_set, draft["text"])
            if marker in seen_text:
                continue
            seen_text.add(marker)
            counters[(tpl.template_set, tpl.qtype)] += 1
            index = counters[(tpl.template_set, tpl.qtype)]
            qid = f"{tpl.template_set}-{tpl.qtype}-{index:04d}"
            questions.append(
                EvalQuestion(
                    qid=qid,
                    text=draft["text"],
                    qtype=tpl.qtype,
                    template_key=tpl.key,
                    template_set=tpl.template_set,
                    route=draft["route"],
                    gold=draft.get("gold") or GoldTarget(),
                    gold_scalar=draft.get("gold_scalar"),
                    gold_scalar_kind=draft.get("gold_scalar_kind", ""),
                    answerable=draft.get("answerable", True),
                    meta=draft.get("meta", {}),
                )
            )

    if limit is not None and limit < len(questions):
        questions = stratified_sample(questions, limit, seed)
    return questions


def stratified_sample(
    questions: Sequence[EvalQuestion], size: int, seed: int = DEFAULT_SEED
) -> list[EvalQuestion]:
    """One seeded sample, stratified by question type.

    The ablation reuses this identically across all 18 cells. An inconsistent
    sample would make cells incomparable, which is worse than a smaller n.
    """
    by_type: dict[str, list[EvalQuestion]] = {}
    for q in questions:
        by_type.setdefault(q.qtype, []).append(q)

    rng = random.Random(_derive_seed(seed, f"stratified:{size}"))
    total = len(questions)
    if size >= total:
        return sorted(questions, key=lambda q: q.qid)

    chosen: dict[str, list[EvalQuestion]] = {}
    spare: dict[str, list[EvalQuestion]] = {}
    for qtype in sorted(by_type):
        bucket = sorted(by_type[qtype], key=lambda q: q.qid)
        rng.shuffle(bucket)
        # Proportional allocation, but never fewer than one per type: a stratum
        # rounded down to zero would silently delete a whole question type.
        take = min(len(bucket), max(1, int(size * len(bucket) / total)))
        chosen[qtype] = bucket[:take]
        spare[qtype] = bucket[take:]

    # Rounding down loses a handful of slots; hand them back to the strata with
    # the most left over, so `size` is met exactly and the shape stays close to
    # proportional.
    order = sorted(spare, key=lambda t: (-len(spare[t]), t))
    while sum(len(v) for v in chosen.values()) < size:
        progressed = False
        for qtype in order:
            if not spare[qtype]:
                continue
            chosen[qtype].append(spare[qtype].pop(0))
            progressed = True
            if sum(len(v) for v in chosen.values()) >= size:
                break
        if not progressed:  # pragma: no cover - size < total guarantees spares
            break

    # Trim the largest strata first, never below one per type.
    while sum(len(v) for v in chosen.values()) > size:
        biggest = max(chosen, key=lambda t: (len(chosen[t]), t))
        if len(chosen[biggest]) <= 1:  # pragma: no cover - size >= n_types
            break
        chosen[biggest].pop()

    picked = [q for qtype in sorted(chosen) for q in chosen[qtype]]
    picked.sort(key=lambda q: q.qid)
    return picked


def type_counts(questions: Sequence[EvalQuestion]) -> dict[str, int]:
    return dict(sorted(Counter(q.qtype for q in questions).items()))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build and inspect the eval questions.")
    ap.add_argument("--templates", choices=("dev", "heldout", "all"), default="dev")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--dump", default=None, help="write the question set to this JSON file")
    ap.add_argument("--show", type=int, default=0, help="print this many sample questions")
    args = ap.parse_args(argv)

    manifest = Manifest.load(args.manifest) if args.manifest else Manifest.load()
    questions = build_questions(manifest, args.templates, args.seed, args.limit)

    counts = type_counts(questions)
    print(f"template set : {args.templates}")
    print(f"seed         : {args.seed}")
    print(f"questions    : {len(questions)}")
    for qtype, count in counts.items():
        n_unans = sum(1 for q in questions if q.qtype == qtype and not q.answerable)
        suffix = f"  ({n_unans} unanswerable)" if n_unans else ""
        print(f"  {qtype:<24} {count:>4}{suffix}")

    if args.show:
        print("\nsamples:")
        for q in questions[: args.show]:
            print(f"  [{q.qid}] ({q.route.value}) {q.text}")

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump([q.as_dict() for q in questions], fh, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.dump}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
