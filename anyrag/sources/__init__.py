"""Source adapters.

This module deliberately contains no adapter list. Every module in the package
is imported on first use, and adapters bind themselves to a URI scheme with the
`@register_source` decorator. That is what makes adding a backend a one-file
change: there is no table here to update.

Orchestrator-owned (discovery infrastructure). The adapter modules themselves
are owned by source-eng.
"""

from __future__ import annotations

from ..core.registry import (  # noqa: F401
    DISCOVERY_ERRORS,
    available_sources,
    discover,
    get_source_class,
    open_source,
    register_source,
)

__all__ = [
    "DISCOVERY_ERRORS",
    "available_sources",
    "discover",
    "get_source_class",
    "open_source",
    "register_source",
]
