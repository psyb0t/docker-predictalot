"""Sundial bridge lifecycle behavior."""

from __future__ import annotations

import pytest

from predictalot.models import sundial

_UNLOAD_PATH = "/unload"


class FakeResponse:
    status_code = 200
    text = ""


class FakeWorkerClient:
    def __init__(self) -> None:
        self.requested_paths: list[str] = []

    async def post(self, path: str) -> FakeResponse:
        self.requested_paths.append(path)
        return FakeResponse()


async def test_unload_releases_the_real_sundial_worker_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_client = FakeWorkerClient()
    monkeypatch.setattr(sundial, "_client", worker_client)
    monkeypatch.setattr(sundial, "_loaded", True)
    monkeypatch.setattr(sundial, "_last_used", 1.0)

    await sundial.unload()

    assert worker_client.requested_paths == [_UNLOAD_PATH]
    assert sundial.loaded() is False
    assert sundial.last_used_secs_ago() is None


async def test_unload_keeps_sundial_marked_loaded_when_worker_rejects_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedResponse:
        status_code = 503
        text = "worker unavailable"

    class FailedWorkerClient(FakeWorkerClient):
        async def post(self, path: str) -> FailedResponse:
            self.requested_paths.append(path)
            return FailedResponse()

    worker_client = FailedWorkerClient()
    monkeypatch.setattr(sundial, "_client", worker_client)
    monkeypatch.setattr(sundial, "_loaded", True)
    monkeypatch.setattr(sundial, "_last_used", 1.0)

    with pytest.raises(RuntimeError, match="503"):
        await sundial.unload()

    assert worker_client.requested_paths == [_UNLOAD_PATH]
    assert sundial.loaded() is True
