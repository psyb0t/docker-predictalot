"""End-to-end HTTP tests for /v1/tabular/forecast/ensemble.

Covers weight resolution, mode/horizon/feature-name validation,
combination math, and edge cases.
"""

from __future__ import annotations

import random

import pytest

# Tabular HTTP roundtrip needs the real ML backends installed.
pytest.importorskip("lightgbm")
pytest.importorskip("xgboost")
pytest.importorskip("sklearn")


AUTH = {"Authorization": "Bearer testtoken"}


def _train(client, model_id: str, backend: str = "lightgbm",
           mode: str = "direction", horizon: int = 1, n: int = 200) -> None:
    random.seed(0)
    a = [random.random() for _ in range(n)]
    b = [random.random() for _ in range(n)]
    c = [random.random() for _ in range(n)]
    # Random-walk target driven by 'b' so direction mode sees a real
    # up/down mix (strictly monotone targets break classification).
    target = [0.0] * n
    target[0] = 50.0
    for i in range(1, n):
        step = (b[i] - 0.5) * 4.0 + (a[i] - 0.5) * 0.5
        target[i] = target[i - 1] + step
    body = {
        "modelId": model_id,
        "backend": backend,
        "target": [target],
        "features": [{"a": a, "b": b, "c": c}],
        "config": {
            "mode": mode,
            "horizon": horizon,
            "nEstimators": 50,
            "learningRate": 0.1,
            "randomState": 0,
        },
        "overwrite": True,
    }
    r = client.post("/v1/tabular/train", json=body, headers=AUTH)
    assert r.status_code == 200, r.text


def _features() -> list[dict[str, list[float]]]:
    return [{"a": [0.5], "b": [0.95], "c": [0.5]}]


# ─── happy paths ────────────────────────────────────────────────────────────


def test_ensemble_equal_weights_default(client) -> None:
    _train(client, "ens-lgbm", backend="lightgbm")
    _train(client, "ens-xgb",  backend="xgboost")
    _train(client, "ens-log",  backend="logistic")

    body = {
        "modelIds": ["ens-lgbm", "ens-xgb", "ens-log"],
        "features": _features(),
    }
    r = client.post(
        "/v1/tabular/forecast/ensemble", json=body, headers=AUTH,
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["mode"] == "direction"
    assert data["horizon"] == 1
    for w in data["weights"].values():
        assert w == pytest.approx(1.0 / 3.0)
    assert set(data["individual"]) == {"ens-lgbm", "ens-xgb", "ens-log"}
    # Combined prob_up = mean of the three members.
    members = data["individual"]
    expected = sum(
        members[mid]["probUp"][0] / 3.0 for mid in ("ens-lgbm", "ens-xgb", "ens-log")
    )
    assert data["probUp"][0] == pytest.approx(expected)


def test_ensemble_explicit_weights(client) -> None:
    _train(client, "ens-lgbm", backend="lightgbm")
    _train(client, "ens-log",  backend="logistic")

    body = {
        "modelIds": ["ens-lgbm", "ens-log"],
        "weights": {"ens-lgbm": 0.3, "ens-log": 0.7},
        "features": _features(),
    }
    r = client.post(
        "/v1/tabular/forecast/ensemble", json=body, headers=AUTH,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["weights"]["ens-lgbm"] == pytest.approx(0.3)
    assert data["weights"]["ens-log"] == pytest.approx(0.7)
    expected = (
        0.3 * data["individual"]["ens-lgbm"]["probUp"][0]
        + 0.7 * data["individual"]["ens-log"]["probUp"][0]
    )
    assert data["probUp"][0] == pytest.approx(expected)


def test_ensemble_zero_weight_drops_member(client) -> None:
    _train(client, "ens-lgbm", backend="lightgbm")
    _train(client, "ens-xgb",  backend="xgboost")
    _train(client, "ens-log",  backend="logistic")

    body = {
        "modelIds": ["ens-lgbm", "ens-xgb", "ens-log"],
        "weights": {"ens-lgbm": 0.5, "ens-xgb": 0.0, "ens-log": 0.5},
        "features": _features(),
    }
    r = client.post(
        "/v1/tabular/forecast/ensemble", json=body, headers=AUTH,
    )
    assert r.status_code == 200
    data = r.json()
    assert set(data["ensembleMembers"]) == {"ens-lgbm", "ens-log"}
    assert "ens-xgb" not in data["individual"]


def test_ensemble_confidence_is_distance_from_half(client) -> None:
    _train(client, "ens-lgbm", backend="lightgbm")
    _train(client, "ens-log",  backend="logistic")
    body = {
        "modelIds": ["ens-lgbm", "ens-log"],
        "features": _features(),
    }
    data = client.post(
        "/v1/tabular/forecast/ensemble", json=body, headers=AUTH,
    ).json()
    p = data["probUp"][0]
    assert data["confidence"][0] == pytest.approx(abs(p - 0.5) * 2.0)


def test_ensemble_value_mode_takes_weighted_mean(client) -> None:
    _train(client, "ens-lgbm-v", backend="lightgbm", mode="value")
    _train(client, "ens-log-v",  backend="logistic", mode="value")
    body = {
        "modelIds": ["ens-lgbm-v", "ens-log-v"],
        "weights": {"ens-lgbm-v": 0.25, "ens-log-v": 0.75},
        "features": _features(),
    }
    data = client.post(
        "/v1/tabular/forecast/ensemble", json=body, headers=AUTH,
    ).json()
    assert data["mode"] == "value"
    expected = (
        0.25 * data["individual"]["ens-lgbm-v"]["predicted"][0]
        + 0.75 * data["individual"]["ens-log-v"]["predicted"][0]
    )
    assert data["predicted"][0] == pytest.approx(expected)


# ─── validation paths ──────────────────────────────────────────────────────


def test_ensemble_empty_model_ids_returns_400(client) -> None:
    r = client.post(
        "/v1/tabular/forecast/ensemble",
        json={"modelIds": [], "features": _features()},
        headers=AUTH,
    )
    assert r.status_code == 400


def test_ensemble_unknown_member_returns_404(client) -> None:
    _train(client, "ens-lgbm", backend="lightgbm")
    body = {
        "modelIds": ["ens-lgbm", "never-trained"],
        "features": _features(),
    }
    r = client.post(
        "/v1/tabular/forecast/ensemble", json=body, headers=AUTH,
    )
    assert r.status_code == 404


def test_ensemble_mode_mismatch_returns_400(client) -> None:
    _train(client, "ens-dir",   backend="lightgbm", mode="direction")
    _train(client, "ens-value", backend="lightgbm", mode="value")
    body = {
        "modelIds": ["ens-dir", "ens-value"],
        "features": _features(),
    }
    r = client.post(
        "/v1/tabular/forecast/ensemble", json=body, headers=AUTH,
    )
    assert r.status_code == 400
    assert "mode" in r.json()["detail"]


def test_ensemble_horizon_mismatch_returns_400(client) -> None:
    _train(client, "ens-h1", backend="lightgbm", horizon=1)
    _train(client, "ens-h3", backend="lightgbm", horizon=3)
    body = {
        "modelIds": ["ens-h1", "ens-h3"],
        "features": _features(),
    }
    r = client.post(
        "/v1/tabular/forecast/ensemble", json=body, headers=AUTH,
    )
    assert r.status_code == 400
    assert "horizon" in r.json()["detail"]


def test_ensemble_unknown_weight_key_returns_400(client) -> None:
    _train(client, "ens-lgbm", backend="lightgbm")
    body = {
        "modelIds": ["ens-lgbm"],
        "weights": {"some-other-id": 1.0},
        "features": _features(),
    }
    r = client.post(
        "/v1/tabular/forecast/ensemble", json=body, headers=AUTH,
    )
    assert r.status_code == 400


def test_ensemble_negative_weight_returns_400(client) -> None:
    _train(client, "ens-lgbm", backend="lightgbm")
    body = {
        "modelIds": ["ens-lgbm"],
        "weights": {"ens-lgbm": -1.0},
        "features": _features(),
    }
    r = client.post(
        "/v1/tabular/forecast/ensemble", json=body, headers=AUTH,
    )
    assert r.status_code == 400


def test_ensemble_all_zero_weights_returns_400(client) -> None:
    _train(client, "ens-lgbm", backend="lightgbm")
    _train(client, "ens-log",  backend="logistic")
    body = {
        "modelIds": ["ens-lgbm", "ens-log"],
        "weights": {"ens-lgbm": 0.0, "ens-log": 0.0},
        "features": _features(),
    }
    r = client.post(
        "/v1/tabular/forecast/ensemble", json=body, headers=AUTH,
    )
    assert r.status_code == 400


def test_ensemble_requires_auth(client) -> None:
    r = client.post(
        "/v1/tabular/forecast/ensemble",
        json={"modelIds": ["x"], "features": []},
    )
    assert r.status_code == 401
