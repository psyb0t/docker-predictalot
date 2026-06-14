"""Verify the FM `extra` and `member_overrides` plumbing.

These tests intercept each backend's ``predict_<type>`` to capture the
``extra`` kwarg the dispatch passes through, so we don't need any real
model loading. The dispatch + router + per-member override merge logic
is what's being exercised.
"""

from __future__ import annotations

from typing import Any

import pytest


AUTH = {"Authorization": "Bearer testtoken"}


def _capture_univariate(captured: dict[str, Any]):
    """Build a fake predict_univariate that records its kwargs."""
    async def f(
        context: list[list[float]],
        horizon: int,
        quantile_levels: list[float],
        _context_length: int,
        extra: dict | None = None,
    ) -> dict[str, Any]:
        captured.setdefault("calls", []).append({"extra": extra})
        # Minimal valid univariate response.
        median = [[0.0] * horizon for _ in context]
        quantiles = {f"{q:.1f}": [[0.0] * horizon for _ in context]
                     for q in quantile_levels}
        return {
            "model": "stub",
            "horizon": horizon,
            "quantile_levels": list(quantile_levels),
            "median": median,
            "quantiles": quantiles,
        }
    return f


# ─── single-forecast: body.config.extra → backend extra= ───────────────────


def test_extra_passes_to_single_forecast(client, monkeypatch) -> None:
    from predictalot import models
    cap: dict[str, Any] = {}
    monkeypatch.setattr(
        models.get("chronos-2"), "predict_univariate", _capture_univariate(cap),
    )
    body = {
        "model": "chronos-2",
        "context": [[1.0, 2.0, 3.0, 4.0, 5.0]],
        "config": {
            "horizon": 2,
            "extra": {"batchSize": 64, "crossLearning": True},
        },
    }
    r = client.post(
        "/v1/timeseries/univariate/forecast", json=body, headers=AUTH,
    )
    assert r.status_code == 200, r.text
    assert cap["calls"][0]["extra"] == {
        "batchSize": 64, "crossLearning": True,
    }


def test_extra_defaults_to_empty_dict_when_omitted(client, monkeypatch) -> None:
    from predictalot import models
    cap: dict[str, Any] = {}
    monkeypatch.setattr(
        models.get("chronos-2"), "predict_univariate", _capture_univariate(cap),
    )
    body = {
        "model": "chronos-2",
        "context": [[1.0, 2.0, 3.0, 4.0, 5.0]],
        "config": {"horizon": 2},  # no extra
    }
    r = client.post(
        "/v1/timeseries/univariate/forecast", json=body, headers=AUTH,
    )
    assert r.status_code == 200
    assert cap["calls"][0]["extra"] == {}


# ─── ensemble: global extra propagates to every member ──────────────────────


def test_ensemble_global_extra_propagates_to_all_members(
    client, monkeypatch,
) -> None:
    from predictalot import models
    cap_by_slug: dict[str, dict[str, Any]] = {}
    for slug in ("chronos-2", "timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m"):
        cap = cap_by_slug.setdefault(slug, {})
        monkeypatch.setattr(
            models.get(slug), "predict_univariate", _capture_univariate(cap),
        )
    body = {
        "context": [[1.0, 2.0, 3.0, 4.0]],
        "config": {
            "horizon": 2,
            "extra": {"globalKnob": "yes"},
        },
    }
    r = client.post(
        "/v1/timeseries/univariate/forecast/ensemble",
        json=body, headers=AUTH,
    )
    assert r.status_code == 200, r.text
    for slug, cap in cap_by_slug.items():
        assert cap["calls"][0]["extra"] == {"globalKnob": "yes"}, slug


# ─── member_overrides: per-slug merging ────────────────────────────────────


def test_member_overrides_override_extra_per_member(
    client, monkeypatch,
) -> None:
    from predictalot import models
    cap_by_slug: dict[str, dict[str, Any]] = {}
    for slug in ("chronos-2", "timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m"):
        cap = cap_by_slug.setdefault(slug, {})
        monkeypatch.setattr(
            models.get(slug), "predict_univariate", _capture_univariate(cap),
        )
    body = {
        "context": [[1.0, 2.0, 3.0, 4.0]],
        "config": {
            "horizon": 2,
            "extra": {"k": "global"},
        },
        "memberOverrides": {
            "chronos-2":   {"extra": {"k": "chronos-special"}},
            "moirai-2":    {"extra": {"extraK": 42}},
        },
    }
    r = client.post(
        "/v1/timeseries/univariate/forecast/ensemble",
        json=body, headers=AUTH,
    )
    assert r.status_code == 200, r.text
    # chronos-2: override completely replaces k for that member
    assert cap_by_slug["chronos-2"]["calls"][0]["extra"] == {
        "k": "chronos-special",
    }
    # moirai-2: keeps global k, adds extraK
    assert cap_by_slug["moirai-2"]["calls"][0]["extra"] == {
        "k": "global", "extraK": 42,
    }
    # Others: untouched, get the global
    for slug in ("timesfm-2.5", "toto-1", "sundial-base-128m"):
        assert cap_by_slug[slug]["calls"][0]["extra"] == {"k": "global"}


def test_member_overrides_can_set_per_member_context_length(
    client, monkeypatch,
) -> None:
    """Context-length overrides flow through the per-member resolver."""
    from predictalot import models, dispatch
    captured: dict[str, int] = {}
    orig = dispatch._resolve_ctx_len  # noqa: SLF001

    def spy(slug: str, ctx: int | None) -> int:
        v = orig(slug, ctx)
        captured.setdefault(slug, v)
        return v

    monkeypatch.setattr(dispatch, "_resolve_ctx_len", spy)
    for slug in ("chronos-2", "timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m"):
        monkeypatch.setattr(
            models.get(slug), "predict_univariate", _capture_univariate({}),
        )

    body = {
        "context": [[float(i) for i in range(20)]],
        "config": {"horizon": 2, "contextLength": 1024},
        "memberOverrides": {
            "chronos-2": {"contextLength": 4096},
            "moirai-2":  {"contextLength": 256},
        },
    }
    r = client.post(
        "/v1/timeseries/univariate/forecast/ensemble",
        json=body, headers=AUTH,
    )
    assert r.status_code == 200, r.text
    # chronos-2 saw 4096; moirai-2 saw 256; others saw 1024.
    assert captured["chronos-2"] == 4096
    assert captured["moirai-2"] == 256
    assert captured["timesfm-2.5"] == 1024
    assert captured["toto-1"] == 1024
    assert captured["sundial-base-128m"] == 1024


def test_member_overrides_unknown_slug_is_ignored(
    client, monkeypatch,
) -> None:
    """Unknown slugs in member_overrides don't break the request — they
    just have no effect (no member with that slug to apply to)."""
    from predictalot import models
    cap_by_slug: dict[str, dict[str, Any]] = {}
    for slug in ("chronos-2", "timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m"):
        cap = cap_by_slug.setdefault(slug, {})
        monkeypatch.setattr(
            models.get(slug), "predict_univariate", _capture_univariate(cap),
        )
    body = {
        "context": [[1.0, 2.0, 3.0, 4.0]],
        "config": {"horizon": 2, "extra": {"k": "global"}},
        "memberOverrides": {"nonexistent-model": {"extra": {"k": "nope"}}},
    }
    r = client.post(
        "/v1/timeseries/univariate/forecast/ensemble",
        json=body, headers=AUTH,
    )
    assert r.status_code == 200
    for slug, cap in cap_by_slug.items():
        assert cap["calls"][0]["extra"] == {"k": "global"}
