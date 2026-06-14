"""End-to-end tabular tests against a real container.

Walks every registered tabular backend through a train + forecast cycle
in each supported mode. Uses synthetic data so we're testing the wiring
+ each backend's library integration, not model quality.
"""

from __future__ import annotations

import math
import random
from typing import Any

import httpx
import pytest


pytestmark = pytest.mark.integration


BACKENDS_URL = "/v1/tabular/backends"
TRAIN_URL = "/v1/tabular/train"
FORECAST_URL = "/v1/tabular/forecast"
ENSEMBLE_URL = "/v1/tabular/forecast/ensemble"
MODELS_URL = "/v1/tabular/models"

# All registered backends — must match the registry in
# predictalot.models.__init__.
ALL_BACKENDS = [
    "lightgbm", "xgboost", "hist-gbt", "random-forest", "logistic",
    "mlp", "svm-rbf", "knn", "naive-bayes",
]


def _synth(n: int = 250, seed: int = 0) -> dict[str, Any]:
    """Random-walk target driven by 'b' so direction labels are mixed
    and value/quantile have monotone-in-b ground truth."""
    rng = random.Random(seed)
    a = [rng.random() for _ in range(n)]
    b = [rng.random() for _ in range(n)]
    c = [rng.random() for _ in range(n)]
    target = [0.0] * n
    target[0] = 50.0
    for i in range(1, n):
        step = (b[i] - 0.5) * 4.0 + (a[i] - 0.5) * 0.5
        target[i] = target[i - 1] + step
    return {"target": [target], "features": [{"a": a, "b": b, "c": c}]}


def _sane(vals: list[float]) -> None:
    for v in vals:
        assert isinstance(v, (int, float))
        assert math.isfinite(v), f"non-finite: {v}"


# ─── backends listing ──────────────────────────────────────────────────


class TestBackendsListing:
    def test_lists_all_nine_backends(self, http_client: httpx.Client) -> None:
        r = http_client.get(BACKENDS_URL)
        assert r.status_code == 200, r.text
        slugs = {b["slug"] for b in r.json()["backends"]}
        assert slugs == set(ALL_BACKENDS)

    def test_each_has_category(self, http_client: httpx.Client) -> None:
        r = http_client.get(BACKENDS_URL)
        for b in r.json()["backends"]:
            assert b["category"] in {
                "boosting", "bagging", "linear", "neural",
                "kernel", "distance", "independence", "other",
            }


# ─── per-backend train+forecast across the 3 modes ─────────────────────


class TestBackendDirectionLive:
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_direction_train_and_forecast(
        self, http_client: httpx.Client, backend: str,
    ) -> None:
        p = _synth(n=250)
        train_body = {
            "modelId": f"int-dir-{backend}",
            "backend": backend,
            "target": p["target"], "features": p["features"],
            "config": {
                "mode": "direction", "horizon": 3,
                "nEstimators": 30, "learningRate": 0.1, "randomState": 0,
            },
            "overwrite": True,
        }
        r = http_client.post(TRAIN_URL, json=train_body)
        assert r.status_code == 200, f"{backend}: {r.text}"
        data = r.json()
        assert data["backend"] == backend
        assert data["mode"] == "direction"
        assert data["nTrainingRows"] > 0
        assert set(data["featureImportance"]) == {"a", "b", "c"}

        fr = http_client.post(
            FORECAST_URL,
            json={"modelId": train_body["modelId"], "features": p["features"]},
        )
        assert fr.status_code == 200, f"{backend}: {fr.text}"
        fdata = fr.json()
        assert fdata["mode"] == "direction"
        assert len(fdata["probUp"]) == 1
        prob = fdata["probUp"][0]
        assert 0.0 <= prob <= 1.0


class TestBackendValueLive:
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_value_train_and_forecast(
        self, http_client: httpx.Client, backend: str,
    ) -> None:
        p = _synth(n=250)
        train_body = {
            "modelId": f"int-val-{backend}",
            "backend": backend,
            "target": p["target"], "features": p["features"],
            "config": {
                "mode": "value", "horizon": 1,
                "nEstimators": 30, "randomState": 0,
            },
            "overwrite": True,
        }
        r = http_client.post(TRAIN_URL, json=train_body)
        assert r.status_code == 200, f"{backend}: {r.text}"

        fr = http_client.post(
            FORECAST_URL,
            json={"modelId": train_body["modelId"], "features": p["features"]},
        )
        assert fr.status_code == 200, f"{backend}: {fr.text}"
        fdata = fr.json()
        assert fdata["mode"] == "value"
        assert len(fdata["predicted"]) == 1
        _sane(fdata["predicted"])


class TestBackendQuantileLive:
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_quantile_train_and_forecast(
        self, http_client: httpx.Client, backend: str,
    ) -> None:
        p = _synth(n=400)
        train_body = {
            "modelId": f"int-q-{backend}",
            "backend": backend,
            "target": p["target"], "features": p["features"],
            "config": {
                "mode": "quantile", "horizon": 1,
                "quantileLevels": [0.1, 0.5, 0.9],
                "nEstimators": 30, "randomState": 0,
            },
            "overwrite": True,
        }
        r = http_client.post(TRAIN_URL, json=train_body)
        assert r.status_code == 200, f"{backend}: {r.text}"

        fr = http_client.post(
            FORECAST_URL,
            json={"modelId": train_body["modelId"], "features": p["features"]},
        )
        assert fr.status_code == 200, f"{backend}: {fr.text}"
        fdata = fr.json()
        assert fdata["mode"] == "quantile"
        assert set(fdata["quantiles"]) == {"0.1", "0.5", "0.9"}
        assert len(fdata["median"]) == 1
        assert len(fdata["median"][0]) == 1
        for q in ("0.1", "0.5", "0.9"):
            _sane([fdata["quantiles"][q][0][0]])


# ─── cross-backend ensemble ────────────────────────────────────────────


class TestEnsembleLive:
    def test_three_backend_ensemble(self, http_client: httpx.Client) -> None:
        """Train 3 backends on the same data, ensemble them with custom
        weights, verify the weighted-mean is in the unit interval +
        individual responses are returned."""
        p = _synth(n=250)
        member_ids = []
        for backend in ("lightgbm", "logistic", "random-forest"):
            mid = f"ens-{backend}"
            r = http_client.post(
                TRAIN_URL,
                json={
                    "modelId": mid, "backend": backend,
                    "target": p["target"], "features": p["features"],
                    "config": {
                        "mode": "direction", "horizon": 3,
                        "nEstimators": 30, "randomState": 0,
                    },
                    "overwrite": True,
                },
            )
            assert r.status_code == 200, f"{backend}: {r.text}"
            member_ids.append(mid)

        er = http_client.post(
            ENSEMBLE_URL,
            json={
                "modelIds": member_ids,
                "weights": {member_ids[0]: 2.0, member_ids[1]: 1.0, member_ids[2]: 1.0},
                "features": p["features"],
            },
        )
        assert er.status_code == 200, er.text
        edata = er.json()
        assert set(edata["individual"]) == set(member_ids)
        prob = edata["probUp"][0]
        assert 0.0 <= prob <= 1.0
        # Weights normalize to sum 1
        wsum = sum(edata["weights"].values())
        assert abs(wsum - 1.0) < 1e-6
