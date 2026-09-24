"""Behavioral coverage for safe foundation-model teardown."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from predictalot import config, models
from predictalot.errors import ModelBusyError, ModelUnloadError
from predictalot.model_lifecycle import ModelLifecycle

_PRIMARY_SLUG = "chronos-2"
_IDLE_TIMEOUT_SECONDS = 30.0


@dataclass
class FakeBackend:
    is_loaded: bool = False
    last_used_seconds: float | None = None
    unload_calls: int = 0
    unload_error: Exception | None = None

    def loaded(self) -> bool:
        return self.is_loaded

    def last_used_secs_ago(self) -> float | None:
        return self.last_used_seconds

    async def unload(self) -> None:
        self.unload_calls += 1
        if self.unload_error is not None:
            raise self.unload_error
        self.is_loaded = False
        self.last_used_seconds = None


@pytest.fixture
def fake_backends(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeBackend]:
    backend_by_slug = {slug: FakeBackend() for slug in config.MODEL_SLUGS}
    monkeypatch.setattr(models, "get", backend_by_slug.__getitem__)
    return backend_by_slug


async def test_explicit_unload_releases_every_loaded_backend_and_runtime(
    fake_backends: dict[str, FakeBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from predictalot import model_lifecycle as lifecycle_module

    for backend in fake_backends.values():
        backend.is_loaded = True
    runtime_release_calls = 0

    def release_runtime() -> None:
        nonlocal runtime_release_calls
        runtime_release_calls += 1

    monkeypatch.setattr(lifecycle_module, "_release_torch_runtime", release_runtime)

    result = await ModelLifecycle().unload_all()

    assert result["status"] == "unloaded"
    assert result["models"] == [{"slug": slug, "wasLoaded": True} for slug in config.MODEL_SLUGS]
    assert [backend.unload_calls for backend in fake_backends.values()] == [1] * len(
        config.MODEL_SLUGS
    )
    assert runtime_release_calls == 1


async def test_explicit_unload_releases_torch_runtime_when_models_are_already_unloaded(
    fake_backends: dict[str, FakeBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from predictalot import model_lifecycle as lifecycle_module

    runtime_release_calls = 0

    def release_runtime() -> None:
        nonlocal runtime_release_calls
        runtime_release_calls += 1

    monkeypatch.setattr(lifecycle_module, "_release_torch_runtime", release_runtime)

    result = await ModelLifecycle().unload_all()

    expected_models = [{"slug": slug, "wasLoaded": False} for slug in config.MODEL_SLUGS]
    assert result["models"] == expected_models
    assert all(backend.unload_calls == 0 for backend in fake_backends.values())
    assert runtime_release_calls == 1


async def test_explicit_unload_rejects_without_touching_models_during_forecast(
    fake_backends: dict[str, FakeBackend],
) -> None:
    lifecycle = ModelLifecycle()
    started = asyncio.Event()
    finish = asyncio.Event()

    async def forecast() -> dict[str, Any]:
        started.set()
        await finish.wait()
        return {"ok": True}

    running = asyncio.create_task(lifecycle.invoke(_PRIMARY_SLUG, forecast, unload_after=False))
    await started.wait()

    with pytest.raises(ModelBusyError):
        await lifecycle.unload_all()

    assert all(backend.unload_calls == 0 for backend in fake_backends.values())
    finish.set()
    assert await running == {"ok": True}


async def test_explicit_unload_keeps_releasing_after_one_backend_fails(
    fake_backends: dict[str, FakeBackend],
) -> None:
    for backend in fake_backends.values():
        backend.is_loaded = True
    failed = fake_backends[_PRIMARY_SLUG]
    failed.unload_error = RuntimeError("worker unavailable")

    with pytest.raises(ModelUnloadError, match="chronos-2"):
        await ModelLifecycle().unload_all()

    assert all(backend.unload_calls == 1 for backend in fake_backends.values())


async def test_request_unload_waits_for_the_last_same_model_forecast(
    fake_backends: dict[str, FakeBackend],
) -> None:
    lifecycle = ModelLifecycle()
    backend = fake_backends[_PRIMARY_SLUG]
    backend.is_loaded = True
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    first_finish = asyncio.Event()
    second_finish = asyncio.Event()

    async def first_forecast() -> dict[str, Any]:
        first_started.set()
        await first_finish.wait()
        return {"request": "first"}

    async def second_forecast() -> dict[str, Any]:
        second_started.set()
        await second_finish.wait()
        return {"request": "second"}

    first = asyncio.create_task(lifecycle.invoke(_PRIMARY_SLUG, first_forecast, unload_after=True))
    second = asyncio.create_task(
        lifecycle.invoke(_PRIMARY_SLUG, second_forecast, unload_after=False)
    )
    await asyncio.gather(first_started.wait(), second_started.wait())

    first_finish.set()
    assert await first == {"request": "first"}
    assert backend.unload_calls == 0

    second_finish.set()
    assert await second == {"request": "second"}
    assert backend.unload_calls == 1


async def test_idle_sweep_unloads_only_models_at_or_beyond_the_timeout(
    fake_backends: dict[str, FakeBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from predictalot import config as config_module

    due = fake_backends[_PRIMARY_SLUG]
    due.is_loaded = True
    due.last_used_seconds = _IDLE_TIMEOUT_SECONDS
    too_soon = fake_backends["toto-1"]
    too_soon.is_loaded = True
    too_soon.last_used_seconds = _IDLE_TIMEOUT_SECONDS - 1
    monkeypatch.setattr(
        config_module,
        "idle_timeout_for",
        lambda _slug: _IDLE_TIMEOUT_SECONDS,
    )

    result = await ModelLifecycle().unload_idle_models()

    assert [item.to_dict() for item in result] == [{"slug": _PRIMARY_SLUG, "wasLoaded": True}]
    assert due.unload_calls == 1
    assert too_soon.unload_calls == 0
