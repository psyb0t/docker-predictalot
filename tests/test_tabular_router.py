"""End-to-end HTTP tests for the /v1/tabular/* router (single-model
train + forecast + models + delete + backends listing).

Uses the shared ``client`` TestClient fixture from conftest.py, which
stubs the FM backends but leaves the real tabular libraries in place.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

# Tabular HTTP roundtrip needs the real ML backends installed.
pytest.importorskip("lightgbm")
pytest.importorskip("xgboost")
pytest.importorskip("sklearn")


AUTH = {"Authorization": "Bearer testtoken"}


def _synth_train_body(
    backend: str = "lightgbm",
    model_id: str = "test-direction-h1",
    mode: str = "direction",
    horizon: int = 1,
    n: int = 200,
    overwrite: bool = True,
) -> dict[str, Any]:
    random.seed(0)
    a = [random.random() for _ in range(n)]
    b = [random.random() for _ in range(n)]
    c = [random.random() for _ in range(n)]
    # Build a random-walk target driven by 'b' so that direction-mode
    # produces a real mix of up/down moves AND value-mode predictions
    # correlate with 'b'. Strictly monotone targets break direction
    # classification (only one class).
    target = [0.0] * n
    target[0] = 50.0
    for i in range(1, n):
        step = (b[i] - 0.5) * 4.0 + (a[i] - 0.5) * 0.5
        target[i] = target[i - 1] + step
    return {
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
        "overwrite": overwrite,
    }


# ─── /v1/tabular/backends ───────────────────────────────────────────────────


def test_backends_lists_all_nine_known_slugs(client) -> None:
    r = client.get("/v1/tabular/backends", headers=AUTH)
    assert r.status_code == 200
    slugs = {b["slug"] for b in r.json()["backends"]}
    assert slugs == {
        "lightgbm", "xgboost", "hist-gbt", "random-forest", "logistic",
        "mlp", "svm-rbf", "knn", "naive-bayes",
    }


def test_backends_each_lists_supported_modes(client) -> None:
    r = client.get("/v1/tabular/backends", headers=AUTH)
    for be in r.json()["backends"]:
        assert set(be["supportedModes"]) >= {"direction", "value", "quantile"}


def test_backends_each_has_category(client) -> None:
    r = client.get("/v1/tabular/backends", headers=AUTH)
    expected = {
        "lightgbm": "boosting",
        "xgboost": "boosting",
        "hist-gbt": "boosting",
        "random-forest": "bagging",
        "logistic": "linear",
        "mlp": "neural",
        "svm-rbf": "kernel",
        "knn": "distance",
        "naive-bayes": "independence",
    }
    by_slug = {b["slug"]: b for b in r.json()["backends"]}
    for slug, cat in expected.items():
        assert by_slug[slug]["category"] == cat


def test_backends_requires_auth(client) -> None:
    r = client.get("/v1/tabular/backends")
    assert r.status_code == 401


# ─── /v1/tabular/train ──────────────────────────────────────────────────────


def test_train_lightgbm_direction_returns_metadata(client) -> None:
    body = _synth_train_body()
    r = client.post("/v1/tabular/train", json=body, headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert data["modelId"] == "test-direction-h1"
    assert data["backend"] == "lightgbm"
    assert data["mode"] == "direction"
    assert data["horizon"] == 1
    assert data["nTrainingRows"] > 100
    assert set(data["featureNames"]) == {"a", "b", "c"}
    # 'b' is the dominant feature; 'a' and 'c' are mostly noise. Don't
    # over-constrain (lightgbm sometimes assigns close importances when
    # signal is split across features); just check 'b' is at least
    # competitive with the noise features.
    imp = data["featureImportance"]
    assert imp["b"] >= 0.2  # nontrivial signal
    assert imp["b"] + imp["a"] + imp["c"] == pytest.approx(1.0, rel=1e-3)


def test_train_unknown_backend_returns_404(client) -> None:
    body = _synth_train_body(backend="nonexistent")
    r = client.post("/v1/tabular/train", json=body, headers=AUTH)
    assert r.status_code == 404


def test_train_target_features_length_mismatch_returns_400(client) -> None:
    body = _synth_train_body()
    body["features"].append({"a": [0.0], "b": [0.0], "c": [0.0]})
    r = client.post("/v1/tabular/train", json=body, headers=AUTH)
    assert r.status_code == 400


def test_train_empty_target_returns_400(client) -> None:
    body = _synth_train_body()
    body["target"] = []
    body["features"] = []
    r = client.post("/v1/tabular/train", json=body, headers=AUTH)
    assert r.status_code == 400


def test_train_duplicate_without_overwrite_returns_409(client) -> None:
    client.post("/v1/tabular/train", json=_synth_train_body(), headers=AUTH)
    body = _synth_train_body(overwrite=False)
    r = client.post("/v1/tabular/train", json=body, headers=AUTH)
    assert r.status_code == 409


def test_train_duplicate_with_overwrite_succeeds(client) -> None:
    client.post("/v1/tabular/train", json=_synth_train_body(), headers=AUTH)
    body = _synth_train_body(overwrite=True)
    r = client.post("/v1/tabular/train", json=body, headers=AUTH)
    assert r.status_code == 200


# ─── /v1/tabular/forecast ───────────────────────────────────────────────────


def test_forecast_returns_prob_up_and_confidence(client) -> None:
    client.post(
        "/v1/tabular/train", json=_synth_train_body(), headers=AUTH,
    )
    body = {
        "modelId": "test-direction-h1",
        "features": [{"a": [0.5], "b": [0.95], "c": [0.5]}],
    }
    r = client.post("/v1/tabular/forecast", json=body, headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "direction"
    assert len(data["probUp"]) == 1
    assert 0.0 <= data["probUp"][0] <= 1.0
    assert data["confidence"][0] == pytest.approx(
        abs(data["probUp"][0] - 0.5) * 2.0
    )


def test_forecast_unknown_model_returns_404(client) -> None:
    body = {
        "modelId": "never-trained",
        "features": [{"a": [0.5], "b": [0.5], "c": [0.5]}],
    }
    r = client.post("/v1/tabular/forecast", json=body, headers=AUTH)
    assert r.status_code == 404


def test_forecast_with_missing_features_returns_400(client) -> None:
    client.post(
        "/v1/tabular/train", json=_synth_train_body(), headers=AUTH,
    )
    body = {
        "modelId": "test-direction-h1",
        "features": [{"a": [0.5]}],  # missing 'b' and 'c'
    }
    r = client.post("/v1/tabular/forecast", json=body, headers=AUTH)
    assert r.status_code == 400


def test_forecast_value_mode_returns_predicted(client) -> None:
    body = _synth_train_body(model_id="test-value", mode="value", horizon=1)
    client.post("/v1/tabular/train", json=body, headers=AUTH)
    forecast = {
        "modelId": "test-value",
        "features": [{"a": [0.5], "b": [0.9], "c": [0.5]}],
    }
    r = client.post("/v1/tabular/forecast", json=forecast, headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "value"
    assert len(data["predicted"]) == 1
    assert data["probUp"] is None


def test_forecast_quantile_mode_returns_median_and_quantiles(client) -> None:
    body = _synth_train_body(model_id="test-quantile", mode="value")
    body["config"]["mode"] = "quantile"
    body["config"]["quantileLevels"] = [0.1, 0.5, 0.9]
    client.post("/v1/tabular/train", json=body, headers=AUTH)
    forecast = {
        "modelId": "test-quantile",
        "features": [{"a": [0.5], "b": [0.9], "c": [0.5]}],
    }
    r = client.post("/v1/tabular/forecast", json=forecast, headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "quantile"
    assert set(data["quantiles"]) == {"0.1", "0.5", "0.9"}
    assert len(data["median"]) == 1
    assert len(data["median"][0]) == 1


# ─── /v1/tabular/models + DELETE ────────────────────────────────────────────


def test_list_models_includes_trained(client) -> None:
    client.post(
        "/v1/tabular/train", json=_synth_train_body(), headers=AUTH,
    )
    r = client.get("/v1/tabular/models", headers=AUTH)
    assert r.status_code == 200
    ids = [m["modelId"] for m in r.json()["models"]]
    assert "test-direction-h1" in ids


def test_delete_known_model_removes_it(client) -> None:
    client.post(
        "/v1/tabular/train", json=_synth_train_body(), headers=AUTH,
    )
    r = client.delete("/v1/tabular/models/test-direction-h1", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["removed"] is True
    # Subsequent forecast 404s.
    fc = client.post(
        "/v1/tabular/forecast",
        json={
            "modelId": "test-direction-h1",
            "features": [{"a": [0.5], "b": [0.5], "c": [0.5]}],
        },
        headers=AUTH,
    )
    assert fc.status_code == 404


def test_delete_unknown_returns_404(client) -> None:
    r = client.delete("/v1/tabular/models/nope", headers=AUTH)
    assert r.status_code == 404
