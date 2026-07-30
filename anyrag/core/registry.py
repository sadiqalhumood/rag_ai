"""Plugin registry with auto-discovery.

Source adapters self-register with a decorator, and `anyrag.sources.__init__`
imports every module in its own package at import time. The consequence is the
property we actually want to prove: adding a new backend means adding one file
and editing nothing -- not even a registry table or an __init__ import list.

Source URIs look like `<scheme>:<locator>`, e.g.
    sqlite:/path/to/db.sqlite
    postgres:postgresql://host/db      (or postgres:$ENV_VAR_NAME)
    files:/path/to/directory
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Callable, TypeVar

from .errors import ConfigError

_SOURCES: dict[str, type] = {}
_EMBEDDERS: dict[str, Callable[..., object]] = {}

T = TypeVar("T", bound=type)


def register_source(scheme: str) -> Callable[[T], T]:
    """Class decorator: bind a DataSource implementation to a URI scheme."""

    def deco(cls: T) -> T:
        key = scheme.lower()
        existing = _SOURCES.get(key)
        if existing is not None and existing is not cls:
            raise ConfigError(
                f"source scheme {scheme!r} already registered to {existing.__name__}"
            )
        _SOURCES[key] = cls
        return cls

    return deco


def register_embedder(name: str) -> Callable[[T], T]:
    def deco(cls: T) -> T:
        _EMBEDDERS[name.lower()] = cls
        return cls

    return deco


def discover(package: str) -> None:
    """Import every submodule of `package` so decorators run.

    Import failures are swallowed per-module: an adapter whose optional driver
    is missing (psycopg, pyarrow) must not prevent the others from loading.
    The failure is recorded so callers can report it rather than lose it.
    """
    try:
        pkg = importlib.import_module(package)
    except Exception as exc:  # pragma: no cover - package must exist
        raise ConfigError(f"cannot import {package}: {exc}") from exc
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name.startswith("_"):
            continue
        full = f"{package}.{mod.name}"
        try:
            importlib.import_module(full)
        except Exception as exc:
            DISCOVERY_ERRORS[full] = f"{type(exc).__name__}: {exc}"


#: module path -> import error, for adapters whose optional deps are absent.
DISCOVERY_ERRORS: dict[str, str] = {}


def available_sources() -> dict[str, type]:
    if not _SOURCES:
        discover("anyrag.sources")
    return dict(_SOURCES)


def get_source_class(scheme: str) -> type:
    sources = available_sources()
    cls = sources.get(scheme.lower())
    if cls is None:
        known = ", ".join(sorted(sources)) or "(none)"
        extra = ""
        if DISCOVERY_ERRORS:
            extra = f"; adapters that failed to import: {DISCOVERY_ERRORS}"
        raise ConfigError(
            f"unknown source scheme {scheme!r}; known schemes: {known}{extra}"
        )
    return cls


def open_source(uri: str, **kwargs: object):
    """Instantiate a DataSource from a `scheme:locator` URI."""
    if ":" not in uri:
        raise ConfigError(
            f"source URI must look like 'scheme:locator', got {uri!r}"
        )
    scheme, locator = uri.split(":", 1)
    cls = get_source_class(scheme)
    return cls(locator, **kwargs)  # type: ignore[call-arg]
