"""POST /v1/covariates/{forecast,forecast/ensemble} — past + future combined.

Member of TYPE_COVARIATES_BOTH: chronos-2 only (in v0.2).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


HEADERS = {"Authorization": "Bearer testtoken"}
URL = "/v1/timeseries/covariates/forecast"
ENSEMBLE_URL = "/v1/timeseries/covariates/forecast/ensemble"


def _payload(model: str | None = None) -> dict:
    body = {
        "context": [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]],
        "pastCovariates": [
            {"price": [9.0, 9.0, 9.5, 9.5], "promo": [0.0, 0.0, 1.0, 1.0]},
            {"price": [11.0, 11.5, 12.0, 12.0], "promo": [1.0, 1.0, 0.0, 0.0]},
        ],
        "futureCovariates": [
            {"price": [9.5, 9.5], "promo": [1.0, 1.0]},
            {"price": [12.0, 12.5], "promo": [0.0, 0.0]},
        ],
        "config": {"horizon": 2},
    }
    if model is not None:
        body["model"] = model
    return body


class TestCovariatesBothHappyPath:
    def test_chronos2_shapes(self, client: TestClient) -> None:
        resp = client.post(URL, headers=HEADERS, json=_payload("chronos-2"))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model"] == "chronos-2"
        assert body["horizon"] == 2
        assert len(body["median"]) == 2
        assert len(body["median"][0]) == 2
        assert set(body["quantiles"].keys()) == {"0.1", "0.5", "0.9"}


class TestCovariatesBothValidation:
    @pytest.mark.parametrize(
        "unsupported", ["timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m"]
    )
    def test_capability_rejection(self, client: TestClient, unsupported: str) -> None:
        resp = client.post(URL, headers=HEADERS, json=_payload(unsupported))
        assert resp.status_code == 400

    def test_unknown_model_404(self, client: TestClient) -> None:
        resp = client.post(URL, headers=HEADERS, json=_payload("nonexistent"))
        assert resp.status_code == 404

    def test_missing_past_covariates_field_rejected(self, client: TestClient) -> None:
        resp = client.post(
            URL,
            headers=HEADERS,
            json={
                "model": "chronos-2",
                "context": [[1.0, 2.0]],
                "futureCovariates": [{}],
                "config": {"horizon": 1},
            },
        )
        assert resp.status_code == 422

    def test_missing_future_covariates_field_rejected(self, client: TestClient) -> None:
        resp = client.post(
            URL,
            headers=HEADERS,
            json={
                "model": "chronos-2",
                "context": [[1.0, 2.0]],
                "pastCovariates": [{}],
                "config": {"horizon": 1},
            },
        )
        assert resp.status_code == 422

    def test_auth_required(self, client: TestClient) -> None:
        resp = client.post(URL, json=_payload("chronos-2"))
        assert resp.status_code == 401


class TestCovariatesBothEnsemble:
    def test_single_member_ensemble_works(self, client: TestClient) -> None:
        resp = client.post(ENSEMBLE_URL, headers=HEADERS, json=_payload())
        assert resp.status_code == 200, resp.text
        out = resp.json()
        assert out["model"] == "ensemble"
        assert out["ensembleMembers"] == ["chronos-2"]
        assert out["weights"] == {"chronos-2": pytest.approx(1.0)}
