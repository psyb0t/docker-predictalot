"""End-to-end tests for /v1/tabular/train/{calibrated,stacking,
diversified} + matching forecast endpoints against a real container.
"""

from __future__ import annotations

import random
from typing import Any

import httpx
import pytest


pytestmark = pytest.mark.integration


CAL_TRAIN_URL = "/v1/tabular/train/calibrated"
CAL_FCAST_URL = "/v1/tabular/forecast/calibrated"
STK_TRAIN_URL = "/v1/tabular/train/stacking"
STK_FCAST_URL = "/v1/tabular/forecast/stacking"
DIV_TRAIN_URL = "/v1/tabular/train/diversified"
DIV_FCAST_URL = "/v1/tabular/forecast/diversified"


def _synth(n: int = 400, seed: int = 0) -> dict[str, Any]:
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


class TestCalibratedLive:
    def test_lightgbm_sigmoid(self, http_client: httpx.Client) -> None:
        p = _synth()
        body = {
            "modelId": "int-cal-lgbm-sigmoid",
            "baseBackend": "lightgbm",
            "target": p["target"], "features": p["features"],
            "config": {
                "mode": "direction", "horizon": 3,
                "nEstimators": 30, "randomState": 0,
            },
            "calibrationMethod": "sigmoid",
            "calibrationFraction": 0.2,
            "overwrite": True,
        }
        r = http_client.post(CAL_TRAIN_URL, json=body)
        assert r.status_code == 200, r.text
        assert r.json()["kind"] == "calibrated"

        fr = http_client.post(
            CAL_FCAST_URL,
            json={"modelId": body["modelId"], "features": p["features"]},
        )
        assert fr.status_code == 200, fr.text
        fdata = fr.json()
        assert 0.0 <= fdata["probUp"][0] <= 1.0
        assert "lightgbm" in fdata["members"]

    def test_logistic_isotonic(self, http_client: httpx.Client) -> None:
        p = _synth()
        body = {
            "modelId": "int-cal-log-isotonic",
            "baseBackend": "logistic",
            "target": p["target"], "features": p["features"],
            "config": {"mode": "direction", "horizon": 3, "randomState": 0},
            "calibrationMethod": "isotonic",
            "calibrationFraction": 0.25,
            "overwrite": True,
        }
        r = http_client.post(CAL_TRAIN_URL, json=body)
        assert r.status_code == 200, r.text

        fr = http_client.post(
            CAL_FCAST_URL,
            json={"modelId": body["modelId"], "features": p["features"]},
        )
        assert fr.status_code == 200, fr.text


class TestStackingLive:
    def test_three_member_stacking(self, http_client: httpx.Client) -> None:
        p = _synth()
        body = {
            "modelId": "int-stk-3",
            "members": [
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
                    "config": {
                        "mode": "direction", "horizon": 3,
                        "randomState": 0,
                    },
                },
            ],
            "metaBackend": "logistic",
            "target": p["target"], "features": p["features"],
            "horizon": 3,
            "nFolds": 3,
            "overwrite": True,
        }
        r = http_client.post(STK_TRAIN_URL, json=body)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["kind"] == "stacking"
        assert set(data["membersUsed"]) == {"lightgbm", "xgboost", "logistic"}

        fr = http_client.post(
            STK_FCAST_URL,
            json={"modelId": body["modelId"], "features": p["features"]},
        )
        assert fr.status_code == 200, fr.text
        fdata = fr.json()
        assert 0.0 <= fdata["probUp"][0] <= 1.0
        assert set(fdata["members"]) == {"lightgbm", "xgboost", "logistic"}


class TestDiversifiedLive:
    def test_direction_picks_diverse_subset(
        self, http_client: httpx.Client,
    ) -> None:
        p = _synth(n=500)
        body = {
            "modelId": "int-div-direction",
            "candidates": [
                {
                    "backend": be,
                    "config": {
                        "mode": "direction", "horizon": 3,
                        "nEstimators": 30, "randomState": 0,
                    },
                }
                for be in ("lightgbm", "xgboost", "random-forest", "logistic", "naive-bayes")
            ],
            "target": p["target"], "features": p["features"],
            "horizon": 3, "mode": "direction",
            "nFolds": 3,
            "maxPairwiseCorr": 0.85,
            "minMembers": 2, "maxMembers": 4,
            "overwrite": True,
        }
        r = http_client.post(DIV_TRAIN_URL, json=body)
        assert r.status_code == 200, r.text
        data = r.json()
        assert 2 <= len(data["membersUsed"]) <= 4
        # Correlation map keys = full candidate set
        assert set(data["candidateCorr"]) == {
            "lightgbm", "xgboost", "random-forest", "logistic", "naive-bayes",
        }

        fr = http_client.post(
            DIV_FCAST_URL,
            json={"modelId": body["modelId"], "features": p["features"]},
        )
        assert fr.status_code == 200, fr.text
        fdata = fr.json()
        assert 0.0 <= fdata["probUp"][0] <= 1.0
        assert fdata["mode"] == "direction"
        assert set(fdata["selectedMembers"]) == set(data["membersUsed"])

    def test_value_mode_uses_negative_mae_ranking(
        self, http_client: httpx.Client,
    ) -> None:
        p = _synth(n=500)
        body = {
            "modelId": "int-div-value",
            "candidates": [
                {
                    "backend": be,
                    "config": {
                        "mode": "value", "horizon": 1,
                        "nEstimators": 30, "randomState": 0,
                    },
                }
                for be in ("lightgbm", "random-forest", "logistic")
            ],
            "target": p["target"], "features": p["features"],
            "horizon": 1, "mode": "value",
            "nFolds": 3,
            "maxPairwiseCorr": 0.99,
            "minMembers": 1, "maxMembers": 3,
            "overwrite": True,
        }
        r = http_client.post(DIV_TRAIN_URL, json=body)
        assert r.status_code == 200, r.text

        fr = http_client.post(
            DIV_FCAST_URL,
            json={"modelId": body["modelId"], "features": p["features"]},
        )
        assert fr.status_code == 200, fr.text
        assert isinstance(fr.json()["predicted"][0], float)

    def test_quantile_mode_combines_per_quantile(
        self, http_client: httpx.Client,
    ) -> None:
        p = _synth(n=500)
        body = {
            "modelId": "int-div-quantile",
            "candidates": [
                {
                    "backend": be,
                    "config": {
                        "mode": "quantile", "horizon": 1,
                        "quantileLevels": [0.1, 0.5, 0.9],
                        "nEstimators": 30, "randomState": 0,
                    },
                }
                for be in ("lightgbm", "random-forest")
            ],
            "target": p["target"], "features": p["features"],
            "horizon": 1, "mode": "quantile",
            "quantileLevels": [0.1, 0.5, 0.9],
            "nFolds": 3,
            "maxPairwiseCorr": 0.99,
            "minMembers": 1, "maxMembers": 2,
            "overwrite": True,
        }
        r = http_client.post(DIV_TRAIN_URL, json=body)
        assert r.status_code == 200, r.text

        fr = http_client.post(
            DIV_FCAST_URL,
            json={"modelId": body["modelId"], "features": p["features"]},
        )
        assert fr.status_code == 200, fr.text
        fdata = fr.json()
        assert fdata["mode"] == "quantile"
        assert set(fdata["quantiles"]) == {"0.1", "0.5", "0.9"}
