"""Shared fixtures.

The database is generated once per session into a tmp dir, never into the repo,
so the tests never depend on `evals/data/` having been built by hand.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from evals.gen_db import Manifest, Sizes, generate
from evals.questions import build_questions

# A smaller world than the real run: the properties under test (determinism,
# null presence, FK integrity, gold agreement) are all scale-free, and 5.7k rows
# per test session is pure latency.
TEST_SIZES = Sizes(customers=160, products=60, orders=400, max_items_per_order=3)
TEST_SEED = 7


@pytest.fixture(scope="session")
def built(tmp_path_factory) -> tuple[Path, Manifest]:
    out = tmp_path_factory.mktemp("evaldb")
    db, _mf, manifest = generate(
        seed=TEST_SEED,
        db_path=out / "eval.sqlite",
        manifest_path=out / "manifest.json",
        sizes=TEST_SIZES,
    )
    return db, Manifest(manifest)


@pytest.fixture(scope="session")
def manifest(built) -> Manifest:
    return built[1]


@pytest.fixture(scope="session")
def db_path(built) -> Path:
    return built[0]


@pytest.fixture(scope="session")
def conn(db_path):
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    yield connection
    connection.close()


@pytest.fixture(scope="session")
def questions(manifest):
    return build_questions(manifest, "dev", TEST_SEED)
