"""Shared pytest fixtures.

Crucially: we patch each model backend's `predict` / `get_model` / `loaded`
methods to never touch the real ML libs or HuggingFace. Tests run anywhere,
including hosts without torch installed.
"""

from __future__ import annotations

import os
from typing import Any

# Set required env defaults BEFORE importing the app, otherwise config.py
# might choke at import time. The defaults match production-like values.
os.environ.setdefault("PREDICTALOT_AUTH_TOKENS", "testtoken")
os.environ.setdefault("PREDICTALOT_MODEL_DIR", "/tmp/predictalot-test-models")

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """FastAPI TestClient with all model backends stubbed."""
    from predictalot import models
    from predictalot.server import app

    _stub_all_backends(monkeypatch, models)

    return TestClient(app)


@pytest.fixture
def open_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Like `client` but with auth disabled (PREDICTALOT_AUTH_TOKENS empty)."""
    from predictalot import config, models
    from predictalot.server import app

    monkeypatch.setattr(config, "AUTH_TOKENS", [])
    monkeypatch.setattr(config, "ALLOW_NO_AUTH", True)
    _stub_all_backends(monkeypatch, models)
    return TestClient(app)


def _stub_all_backends(monkeypatch: pytest.MonkeyPatch, models: Any) -> None:
    """Replace each backend's predict/get_model/loaded with deterministic fakes."""
    for slug in ("chronos-2", "timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m"):
        backend = models.get(slug)
        monkeypatch.setattr(backend, "predict", _make_fake_predict(slug))
        monkeypatch.setattr(backend, "get_model", _make_fake_get_model())
        monkeypatch.setattr(backend, "loaded", lambda: False)
        monkeypatch.setattr(backend, "last_used_secs_ago", lambda: None)
        monkeypatch.setattr(backend, "unload", _make_fake_unload())


def _make_fake_predict(slug: str):
    async def fake_predict(
        context: list[list[float]],
        horizon: int,
        quantile_levels: list[float],
        _context_length: int,
    ) -> dict[str, Any]:
        # Deterministic: each output element is the mean of the input series
        # plus a horizon-step offset. Just enough to be testable.
        median = [
            [sum(s) / max(len(s), 1) + step for step in range(horizon)] for s in context
        ]
        quantiles = {
            f"{q:.1f}": [
                [sum(s) / max(len(s), 1) + step + (q - 0.5) for step in range(horizon)]
                for s in context
            ]
            for q in quantile_levels
        }
        return {
            "model": slug,
            "horizon": horizon,
            "quantile_levels": list(quantile_levels),
            "median": median,
            "quantiles": quantiles,
        }

    return fake_predict


def _make_fake_get_model():
    async def fake_get_model():
        return object()

    return fake_get_model


def _make_fake_unload():
    async def fake_unload() -> None:
        return None

    return fake_unload
