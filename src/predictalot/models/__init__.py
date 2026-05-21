"""Model backends — one module per supported model slug.

Each module exposes:
    async def predict(context, horizon, quantile_levels, context_length, unload) -> dict
    async def get_model() -> object  # lazy loader, idempotent
    async def unload() -> None
    def loaded() -> bool
    def last_used_secs_ago() -> float | None

The forecast dispatcher imports them by slug.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ModelBackend

from . import chronos2, moirai2, sundial, timesfm25, toto1

BACKENDS: dict[str, "ModelBackend"] = {
    "chronos-2": chronos2,
    "timesfm-2.5": timesfm25,
    "moirai-2": moirai2,
    "toto-1": toto1,
    "sundial-base-128m": sundial,
}


def get(slug: str) -> "ModelBackend":
    """Return the backend module for a model slug. KeyError if unknown."""
    return BACKENDS[slug]
