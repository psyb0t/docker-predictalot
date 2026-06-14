"""HTTP tests for /v1/tabular/{train,forecast}/{calibrated,stacking,
diversified}.

Uses the shared ``client`` TestClient fixture from conftest.py. Tabular
backends run the REAL ML libs on synthetic data.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("xgboost")
pytest.importorskip("sklearn")


AUTH = {"Authorization": "Bearer testtoken"}


def _synth_payload(n: int = 250, seed: int = 0) -> dict[str, Any]:
    random.seed(seed)
    a = [random.random() for _ in range(n)]
    b = [random.random() for _ in range(n)]
    c = [random.random() for _ in range(n)]
    target = [0.0] * n
    target[0] = 50.0
    for i in range(1, n):
        step = (b[i] - 0.5) * 4.0 + (a[i] - 0.5) * 0.5
        target[i] = target[i - 1] + step
    return {"target": [target], "features": [{"a": a, "b": b, "c": c}]}


# ─── calibrated ────────────────────────────────────────────────────────


def test_calibrated_train_then_forecast(client) -> None:
    p = _synth_payload()
    body = {
        "modelId": "cal-test",
        "baseBackend": "lightgbm",
        "target": p["target"],
        "features": p["features"],
        "config": {
            "mode": "direction", "horizon": 3,
            "nEstimators": 50, "learningRate": 0.1, "randomState": 0,
        },
        "calibrationMethod": "sigmoid",
        "calibrationFraction": 0.2,
        "overwrite": True,
    }
    r = client.post(
        "/v1/tabular/train/calibrated", json=body, headers=AUTH,
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["kind"] == "calibrated"
    assert data["mode"] == "direction"
    assert data["membersUsed"] == ["lightgbm"]

    fr = client.post(
        "/v1/tabular/forecast/calibrated",
        json={"modelId": "cal-test", "features": p["features"]},
        headers=AUTH,
    )
    assert fr.status_code == 200, fr.text
    fdata = fr.json()
    assert fdata["kind"] == "calibrated"
    assert fdata["mode"] == "direction"
    assert len(fdata["probUp"]) == 1
    assert 0.0 <= fdata["probUp"][0] <= 1.0
    # Both calibrated + raw should be in the unit range.
    assert 0.0 <= fdata["members"]["lightgbm"]["probUp"][0] <= 1.0


def test_calibrated_rejects_non_direction_mode(client) -> None:
    p = _synth_payload()
    body = {
        "modelId": "cal-bad", "baseBackend": "lightgbm",
        "target": p["target"], "features": p["features"],
        "config": {"mode": "value", "horizon": 1},
        "calibrationFraction": 0.2,
        "overwrite": True,
    }
    r = client.post(
        "/v1/tabular/train/calibrated", json=body, headers=AUTH,
    )
    assert r.status_code == 400
    assert "direction" in r.json()["detail"]


def test_calibrated_isotonic_method_also_works(client) -> None:
    p = _synth_payload()
    body = {
        "modelId": "cal-iso", "baseBackend": "logistic",
        "target": p["target"], "features": p["features"],
        "config": {
            "mode": "direction", "horizon": 3, "randomState": 0,
        },
        "calibrationMethod": "isotonic",
        "calibrationFraction": 0.25,
        "overwrite": True,
    }
    r = client.post(
        "/v1/tabular/train/calibrated", json=body, headers=AUTH,
    )
    assert r.status_code == 200, r.text


def test_calibrated_forecast_with_wrong_model_kind_returns_400(client) -> None:
    # Train a non-meta model + try to use the calibrated forecast on it.
    body = {
        "modelId": "regular-not-meta", "backend": "lightgbm",
        **_synth_payload(),
        "config": {
            "mode": "direction", "horizon": 1, "nEstimators": 30,
            "randomState": 0,
        },
        "overwrite": True,
    }
    r = client.post("/v1/tabular/train", json=body, headers=AUTH)
    assert r.status_code == 200, r.text
    fr = client.post(
        "/v1/tabular/forecast/calibrated",
        json={"modelId": "regular-not-meta", "features": body["features"]},
        headers=AUTH,
    )
    assert fr.status_code == 400
    assert "meta:calibrated" in fr.json()["detail"]


# ─── stacking ──────────────────────────────────────────────────────────


def test_stacking_train_then_forecast(client) -> None:
    p = _synth_payload()
    body = {
        "modelId": "stk-test",
        "members": [
            {
                "backend": "lightgbm",
                "config": {
                    "mode": "direction", "horizon": 3,
                    "nEstimators": 30, "randomState": 0,
                },
            },
            {
                "backend": "logistic",
                "config": {"mode": "direction", "horizon": 3, "randomState": 0},
            },
        ],
        "metaBackend": "logistic",
        "target": p["target"], "features": p["features"],
        "horizon": 3,
        "nFolds": 3,
        "overwrite": True,
    }
    r = client.post(
        "/v1/tabular/train/stacking", json=body, headers=AUTH,
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["kind"] == "stacking"
    assert set(data["membersUsed"]) == {"lightgbm", "logistic"}
    # oofScore is meta-learner AUC on OOF — must be a float in [0,1] or NaN
    assert data["oofScore"] is None or 0.0 <= data["oofScore"] <= 1.0

    fr = client.post(
        "/v1/tabular/forecast/stacking",
        json={"modelId": "stk-test", "features": p["features"]},
        headers=AUTH,
    )
    assert fr.status_code == 200, fr.text
    fdata = fr.json()
    assert fdata["kind"] == "stacking"
    assert set(fdata["members"].keys()) == {"lightgbm", "logistic"}
    assert 0.0 <= fdata["probUp"][0] <= 1.0


def test_stacking_rejects_mixed_modes(client) -> None:
    p = _synth_payload()
    body = {
        "modelId": "stk-mix",
        "members": [
            {
                "backend": "lightgbm",
                "config": {"mode": "value", "horizon": 1},
            },
            {
                "backend": "logistic",
                "config": {"mode": "direction", "horizon": 1},
            },
        ],
        "target": p["target"], "features": p["features"], "horizon": 1,
        "overwrite": True,
    }
    r = client.post(
        "/v1/tabular/train/stacking", json=body, headers=AUTH,
    )
    assert r.status_code == 400


# ─── diversified ───────────────────────────────────────────────────────


def test_diversified_train_direction(client) -> None:
    p = _synth_payload(n=300)
    body = {
        "modelId": "div-test",
        "candidates": [
            {
                "backend": "lightgbm",
                "config": {
                    "mode": "direction", "horizon": 3,
                    "nEstimators": 30, "randomState": 0,
                },
            },
            {
                "backend": "xgboost",
                "config": {
                    "mode": "direction", "horizon": 3,
                    "nEstimators": 30, "randomState": 0,
                },
            },
            {
                "backend": "logistic",
                "config": {"mode": "direction", "horizon": 3, "randomState": 0},
            },
        ],
        "target": p["target"], "features": p["features"],
        "horizon": 3, "mode": "direction",
        "nFolds": 3,
        "maxPairwiseCorr": 0.95,
        "minMembers": 1, "maxMembers": 3,
        "overwrite": True,
    }
    r = client.post(
        "/v1/tabular/train/diversified", json=body, headers=AUTH,
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["kind"] == "diversified"
    assert 1 <= len(data["membersUsed"]) <= 3
    # candidateCorr is a 3x3 (or n x n) matrix
    assert set(data["candidateCorr"].keys()) == {"lightgbm", "xgboost", "logistic"}

    fr = client.post(
        "/v1/tabular/forecast/diversified",
        json={"modelId": "div-test", "features": p["features"]},
        headers=AUTH,
    )
    assert fr.status_code == 200, fr.text
    fdata = fr.json()
    assert fdata["kind"] == "diversified"
    assert fdata["mode"] == "direction"
    assert 0.0 <= fdata["probUp"][0] <= 1.0


def test_diversified_value_mode(client) -> None:
    p = _synth_payload(n=300)
    body = {
        "modelId": "div-value",
        "candidates": [
            {
                "backend": "lightgbm",
                "config": {
                    "mode": "value", "horizon": 1,
                    "nEstimators": 30, "randomState": 0,
                },
            },
            {
                "backend": "logistic",
                "config": {"mode": "value", "horizon": 1, "randomState": 0},
            },
        ],
        "target": p["target"], "features": p["features"],
        "horizon": 1, "mode": "value",
        "nFolds": 3,
        "maxPairwiseCorr": 0.99,
        "minMembers": 1, "maxMembers": 2,
        "overwrite": True,
    }
    r = client.post(
        "/v1/tabular/train/diversified", json=body, headers=AUTH,
    )
    assert r.status_code == 200, r.text
    fr = client.post(
        "/v1/tabular/forecast/diversified",
        json={"modelId": "div-value", "features": p["features"]},
        headers=AUTH,
    )
    assert fr.status_code == 200, fr.text
    assert isinstance(fr.json()["predicted"][0], float)


def test_diversified_quantile_mode_requires_levels(client) -> None:
    p = _synth_payload(n=300)
    body = {
        "modelId": "div-q-missing",
        "candidates": [
            {
                "backend": "lightgbm",
                "config": {"mode": "quantile", "horizon": 1, "quantileLevels": [0.1, 0.5, 0.9]},
            },
            {
                "backend": "logistic",
                "config": {"mode": "quantile", "horizon": 1, "quantileLevels": [0.1, 0.5, 0.9]},
            },
        ],
        "target": p["target"], "features": p["features"],
        "horizon": 1, "mode": "quantile",
        "nFolds": 3,
        "overwrite": True,
    }
    r = client.post(
        "/v1/tabular/train/diversified", json=body, headers=AUTH,
    )
    assert r.status_code == 400
    assert "quantile_levels" in r.json()["detail"]


def test_diversified_rejects_mismatched_modes(client) -> None:
    p = _synth_payload()
    body = {
        "modelId": "div-bad",
        "candidates": [
            {
                "backend": "lightgbm",
                "config": {"mode": "direction", "horizon": 1},
            },
            {
                "backend": "logistic",
                "config": {"mode": "value", "horizon": 1},
            },
        ],
        "target": p["target"], "features": p["features"],
        "horizon": 1, "mode": "direction",
        "overwrite": True,
    }
    r = client.post(
        "/v1/tabular/train/diversified", json=body, headers=AUTH,
    )
    assert r.status_code == 400


# ─── auth ──────────────────────────────────────────────────────────────


def test_calibrated_train_requires_auth(client) -> None:
    r = client.post("/v1/tabular/train/calibrated", json={"modelId": "x"})
    assert r.status_code == 401


def test_stacking_train_requires_auth(client) -> None:
    r = client.post("/v1/tabular/train/stacking", json={"modelId": "x"})
    assert r.status_code == 401


def test_diversified_train_requires_auth(client) -> None:
    r = client.post("/v1/tabular/train/diversified", json={"modelId": "x"})
    assert r.status_code == 401
