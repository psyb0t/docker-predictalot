"""Shared types + interface protocol for the per-model backend modules."""

from __future__ import annotations

from typing import Protocol


class ModelBackend(Protocol):
    """Each model module is treated as an instance of this protocol.

    All methods are module-level functions, not class methods.
    """

    async def predict(
        self,
        context: list[list[float]],
        horizon: int,
        quantile_levels: list[float],
        context_length: int,
    ) -> dict: ...

    async def get_model(self) -> object: ...

    async def unload(self) -> None: ...

    def loaded(self) -> bool: ...

    def last_used_secs_ago(self) -> float | None: ...
