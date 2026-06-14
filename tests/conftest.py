"""Shared pytest fixtures.

Each backend module gets the per-type ``predict_<type>`` function it advertises
in ``SUPPORTED_TYPES`` replaced with a deterministic stub so tests can run on
any host (no torch / no HuggingFace weights / no sidecar process). The
runtime-state helpers (``loaded``, ``unload``, ``get_model``,
``last_used_secs_ago``) are also stubbed.

Two clients are exposed:
  * ``client`` — auth required (token ``testtoken``).
  * ``open_client`` — auth disabled (PREDICTALOT_AUTH_TOKENS empty + ALLOW_NO_AUTH).
"""

from __future__ import annotations

import os
from typing import Any

# Set required env defaults BEFORE importing the app; config.py reads these
# at import time and will fail-fast on missing required values.
os.environ.setdefault("PREDICTALOT_AUTH_TOKENS", "testtoken")
os.environ.setdefault("PREDICTALOT_MODEL_DIR", "/tmp/predictalot-test-models")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """FastAPI TestClient with all model backends stubbed."""
    from predictalot import models
    from predictalot.server import app

    _stub_all_backends(monkeypatch, models)

    return TestClient(app)


@pytest.fixture
def open_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Like ``client`` but with auth disabled."""
    from predictalot import config, models
    from predictalot.server import app

    monkeypatch.setattr(config, "AUTH_TOKENS", [])
    monkeypatch.setattr(config, "ALLOW_NO_AUTH", True)
    _stub_all_backends(monkeypatch, models)
    return TestClient(app)


# ─── stub installation ───────────────────────────────────────────────────────

_ATTR_FOR_TYPE = {
    "univariate": "predict_univariate",
    "multivariate": "predict_multivariate",
    "covariates-past": "predict_covariates_past",
    "covariates-future": "predict_covariates_future",
    "covariates-both": "predict_covariates_both",
    "samples": "predict_samples",
}


def _stub_all_backends(monkeypatch: pytest.MonkeyPatch, models: Any) -> None:
    factories = {
        "univariate": _make_fake_univariate,
        "multivariate": _make_fake_multivariate,
        "covariates-past": _make_fake_covariates_past,
        "covariates-future": _make_fake_covariates_future,
        "covariates-both": _make_fake_covariates_both,
        "samples": _make_fake_samples,
    }
    for slug in ("chronos-2", "timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m"):
        backend = models.get(slug)
        for t in backend.SUPPORTED_TYPES:
            attr = _ATTR_FOR_TYPE[t]
            monkeypatch.setattr(backend, attr, factories[t](slug))
        monkeypatch.setattr(backend, "get_model", _make_fake_get_model())
        monkeypatch.setattr(backend, "loaded", lambda: False)
        monkeypatch.setattr(backend, "last_used_secs_ago", lambda: None)
        monkeypatch.setattr(backend, "unload", _make_fake_unload())


# ─── per-type stub factories ─────────────────────────────────────────────────


def _mean(series: list[float]) -> float:
    return sum(series) / max(len(series), 1)


def _univariate_payload(
    slug: str, context: list[list[float]], horizon: int, q_levels: list[float]
) -> dict[str, Any]:
    median = [[_mean(s) + t for t in range(horizon)] for s in context]
    quantiles = {
        f"{q:.1f}": [[_mean(s) + t + (q - 0.5) for t in range(horizon)] for s in context]
        for q in q_levels
    }
    return {
        "model": slug,
        "horizon": horizon,
        "quantile_levels": list(q_levels),
        "median": median,
        "quantiles": quantiles,
    }


def _make_fake_univariate(slug: str):
    async def f(
        context: list[list[float]],
        horizon: int,
        quantile_levels: list[float],
        _context_length: int,
        extra: dict | None = None,  # noqa: ARG001
    ) -> dict[str, Any]:
        return _univariate_payload(slug, context, horizon, quantile_levels)

    return f


def _make_fake_multivariate(slug: str):
    async def f(
        context: list[list[list[float]]],
        horizon: int,
        quantile_levels: list[float],
        _context_length: int,
        extra: dict | None = None,  # noqa: ARG001
    ) -> dict[str, Any]:
        median = [
            [[_mean(channel) + t for t in range(horizon)] for channel in series]
            for series in context
        ]
        quantiles = {
            f"{q:.1f}": [
                [[_mean(channel) + t + (q - 0.5) for t in range(horizon)] for channel in series]
                for series in context
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

    return f


def _make_fake_covariates_past(slug: str):
    async def f(
        context: list[list[float]],
        _past_covariates: list[dict[str, list[float]]],
        horizon: int,
        quantile_levels: list[float],
        _context_length: int,
        extra: dict | None = None,  # noqa: ARG001
    ) -> dict[str, Any]:
        return _univariate_payload(slug, context, horizon, quantile_levels)

    return f


def _make_fake_covariates_future(slug: str):
    async def f(
        context: list[list[float]],
        _future_covariates: list[dict[str, list[float]]],
        horizon: int,
        quantile_levels: list[float],
        _context_length: int,
        extra: dict | None = None,  # noqa: ARG001
    ) -> dict[str, Any]:
        return _univariate_payload(slug, context, horizon, quantile_levels)

    return f


def _make_fake_covariates_both(slug: str):
    async def f(
        context: list[list[float]],
        _past_covariates: list[dict[str, list[float]]],
        _future_covariates: list[dict[str, list[float]]],
        horizon: int,
        quantile_levels: list[float],
        _context_length: int,
        extra: dict | None = None,  # noqa: ARG001
    ) -> dict[str, Any]:
        return _univariate_payload(slug, context, horizon, quantile_levels)

    return f


def _make_fake_samples(slug: str):
    async def f(
        context: list[list[float]],
        horizon: int,
        num_samples: int,
        _context_length: int,
        extra: dict | None = None,  # noqa: ARG001
    ) -> dict[str, Any]:
        # Per-sample deterministic offset so individual draws are distinct.
        samples = [
            [
                [_mean(s) + t + k * 0.01 for t in range(horizon)]
                for k in range(num_samples)
            ]
            for s in context
        ]
        median = [[_mean(s) + t for t in range(horizon)] for s in context]
        return {
            "model": slug,
            "horizon": horizon,
            "num_samples": num_samples,
            "samples": samples,
            "median": median,
        }

    return f


def _make_fake_get_model():
    async def fake_get_model() -> object:
        return object()

    return fake_get_model


def _make_fake_unload():
    async def fake_unload() -> None:
        return None

    return fake_unload
