"""Coordinate safe model use, explicit unloads, and idle memory cleanup."""

from __future__ import annotations

import asyncio
import gc
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from . import config, models
from .errors import ModelBusyError, ModelUnloadError

log = logging.getLogger("predictalot.model_lifecycle")

_UNLOADED_STATUS = "unloaded"


@dataclass(frozen=True)
class ModelUnloadResult:
    """One foundation model's idempotent unload result."""

    slug: str
    was_loaded: bool

    def to_dict(self) -> dict[str, object]:
        return {"slug": self.slug, "wasLoaded": self.was_loaded}


class ModelLifecycle:
    """Own in-process model state transitions without interrupting inference."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._active_requests = {slug: 0 for slug in config.MODEL_SLUGS}
        self._unloading: set[str] = set()
        self._pending_unload: set[str] = set()

    async def invoke(
        self,
        slug: str,
        operation: Callable[[], Awaitable[dict[str, Any]]],
        *,
        unload_after: bool,
    ) -> dict[str, Any]:
        """Run one inference while preventing concurrent teardown of its model."""
        await self._acquire(slug)
        try:
            return await operation()
        finally:
            await self._release(slug, unload_after=unload_after)

    async def unload_all(self) -> dict[str, object]:
        """Unload every foundation model, rejecting the whole operation if one is busy."""
        async with self._condition:
            await self._wait_for_pending_unloads()
            if any(self._active_requests.values()):
                raise ModelBusyError("a foundation model is processing a forecast")
            self._unloading.update(config.MODEL_SLUGS)

        results = await self._unload_claimed(config.MODEL_SLUGS, reason="explicit")
        return {
            "status": _UNLOADED_STATUS,
            "models": [result.to_dict() for result in results],
        }

    async def unload_idle_models(self) -> tuple[ModelUnloadResult, ...]:
        """Unload resident foundation models whose configured idle timeout elapsed."""
        results: list[ModelUnloadResult] = []
        for slug in config.MODEL_SLUGS:
            if not await self._claim_idle_unload(slug):
                continue
            try:
                results.extend(await self._unload_claimed((slug,), reason="idle_timeout"))
            except ModelUnloadError:
                log.exception(
                    "idle foundation-model unload failed",
                    extra={"model": slug},
                )
        return tuple(results)

    async def _acquire(self, slug: str) -> None:
        async with self._condition:
            self._require_slug(slug)
            while slug in self._unloading:
                await self._condition.wait()
            self._active_requests[slug] += 1

    async def _release(self, slug: str, *, unload_after: bool) -> None:
        should_unload = False
        async with self._condition:
            self._active_requests[slug] -= 1
            if unload_after:
                self._pending_unload.add(slug)
            if self._active_requests[slug] == 0 and slug in self._pending_unload:
                self._pending_unload.remove(slug)
                self._unloading.add(slug)
                should_unload = True
            self._condition.notify_all()

        if should_unload:
            await self._unload_claimed((slug,), reason="request")

    async def _claim_idle_unload(self, slug: str) -> bool:
        timeout = config.idle_timeout_for(slug)
        if timeout == 0:
            return False

        async with self._condition:
            if self._active_requests[slug] > 0 or slug in self._unloading:
                return False
            backend = models.get(slug)
            if not backend.loaded():
                return False
            last_used = backend.last_used_secs_ago()
            if last_used is None or last_used < timeout:
                return False
            self._unloading.add(slug)
            return True

    async def _unload_claimed(
        self,
        slugs: tuple[str, ...],
        *,
        reason: str,
    ) -> tuple[ModelUnloadResult, ...]:
        results: list[ModelUnloadResult] = []
        failures: list[tuple[str, Exception]] = []
        try:
            log.info(
                "unloading foundation models",
                extra={"reason": reason, "models": list(slugs)},
            )
            for slug in slugs:
                backend = models.get(slug)
                was_loaded = backend.loaded()
                if was_loaded:
                    try:
                        await backend.unload()
                    except Exception as exc:  # noqa: BLE001
                        log.exception(
                            "foundation-model unload failed",
                            extra={"model": slug, "reason": reason},
                        )
                        failures.append((slug, exc))
                results.append(ModelUnloadResult(slug=slug, was_loaded=was_loaded))
            should_release_torch_runtime = reason == "explicit" or any(
                result.was_loaded for result in results
            )
            if should_release_torch_runtime:
                async with self._condition:
                    no_forecasts_active = not any(self._active_requests.values())
                if no_forecasts_active:
                    try:
                        await asyncio.to_thread(_release_torch_runtime)
                    except Exception as exc:  # noqa: BLE001
                        log.exception("foundation-model Torch runtime release failed")
                        failures.append(("torch-runtime", exc))
            if failures:
                failed = ", ".join(slug for slug, _ in failures)
                raise ModelUnloadError(f"failed to fully unload: {failed}") from failures[0][1]
            return tuple(results)
        finally:
            async with self._condition:
                for slug in slugs:
                    self._unloading.discard(slug)
                self._condition.notify_all()

    async def _wait_for_pending_unloads(self) -> None:
        while self._unloading:
            await self._condition.wait()

    def _require_slug(self, slug: str) -> None:
        if slug not in self._active_requests:
            raise KeyError(f"unknown foundation model: {slug}")


def _release_torch_runtime() -> None:
    """Release global Torch caches after every foundation forecast has finished."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    compiler = getattr(torch, "compiler", None)
    compiler_reset = getattr(compiler, "reset", None)
    if callable(compiler_reset):
        compiler_reset()

    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


model_lifecycle = ModelLifecycle()
