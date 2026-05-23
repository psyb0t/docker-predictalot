"""Model backends — one module per supported model slug.

Each module exposes:
    SLUG: str
    SUPPORTED_TYPES: frozenset[str]  — see predictalot.types
    async get_model() -> object       — lazy loader, idempotent
    async unload() -> None
    def  loaded() -> bool
    def  last_used_secs_ago() -> float | None
    plus a `predict_<type>` async function for each SUPPORTED_TYPES member
    (see predictalot.types for the matrix).

Returned as `Any` from `get()` because each backend exposes a different set of
per-type predict functions; the dispatcher knows which one to call based on the
type slug.
"""

from __future__ import annotations

from typing import Any

from . import chronos2, moirai2, sundial, timesfm25, toto1

BACKENDS: dict[str, Any] = {
    "chronos-2": chronos2,
    "timesfm-2.5": timesfm25,
    "moirai-2": moirai2,
    "toto-1": toto1,
    "sundial-base-128m": sundial,
}


def get(slug: str) -> Any:
    """Return the backend module for a model slug. KeyError if unknown."""
    return BACKENDS[slug]
