"""Shared types + interface protocol for the per-model backend modules.

Each backend module exposes a set of `predict_<type>` functions matching its
declared `SUPPORTED_TYPES`. The dispatch layer picks the function by type.

Backends that don't support a given type simply don't define the corresponding
`predict_<type>` function — `dispatch_<type>(model, ...)` checks the type
registry before calling.
"""

from __future__ import annotations

from typing import Protocol


class ModelBackend(Protocol):
    """Each model module is treated as an instance of this protocol.

    All methods are module-level functions, not class methods. The Protocol
    documents the *minimum* surface; per-type `predict_*` functions are
    declared optionally per backend (see TYPE_MEMBERS for the matrix).
    """

    SLUG: str
    SUPPORTED_TYPES: frozenset[str]

    async def get_model(self) -> object: ...

    async def unload(self) -> None: ...

    def loaded(self) -> bool: ...

    def last_used_secs_ago(self) -> float | None: ...
