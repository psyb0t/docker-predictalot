"""POST /v1/covariates/past/{forecast,forecast/ensemble}.

Members of TYPE_COVARIATES_PAST: chronos-2, moirai-2.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


HEADERS = {"Authorization": "Bearer testtoken"}
URL = "/v1/timeseries/covariates/past/forecast"
ENSEMBLE_URL = "/v1/timeseries/covariates/past/forecast/ensemble"
MEMBERS = ["chronos-2", "moirai-2"]


def _payload(model: str | None = None) -> dict:
    body = {
        "context": [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]],
        "pastCovariates": [
            {"temp": [0.1, 0.2, 0.3, 0.4], "promo": [0.0, 1.0, 0.0, 1.0]},
            {"temp": [1.0, 1.1, 1.2, 1.3], "promo": [1.0, 1.0, 0.0, 0.0]},
        ],
        "config": {"horizon": 2},
    }
    if model is not None:
        body["model"] = model
    return body


@pytest.mark.parametrize("model_slug", MEMBERS)
class TestCovariatesPastHappyPath:
    def test_shapes(self, client: TestClient, model_slug: str) -> None:
        resp = client.post(URL, headers=HEADERS, json=_payload(model_slug))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model"] == model_slug
        assert body["horizon"] == 2
        # Univariate-target output: [series][time]
        assert len(body["median"]) == 2
        assert len(body["median"][0]) == 2
        assert set(body["quantiles"].keys()) == {"0.1", "0.5", "0.9"}


class TestCovariatesPastValidation:
    def test_capability_rejection_for_timesfm(self, client: TestClient) -> None:
        resp = client.post(URL, headers=HEADERS, json=_payload("timesfm-2.5"))
        assert resp.status_code == 400

    def test_capability_rejection_for_toto(self, client: TestClient) -> None:
        resp = client.post(URL, headers=HEADERS, json=_payload("toto-1"))
        assert resp.status_code == 400

    def test_unknown_model_404(self, client: TestClient) -> None:
        resp = client.post(URL, headers=HEADERS, json=_payload("nonexistent"))
        assert resp.status_code == 404

    def test_empty_context_rejected(self, client: TestClient) -> None:
        resp = client.post(
            URL,
            headers=HEADERS,
            json={
                "model": "chronos-2",
                "context": [],
                "pastCovariates": [],
                "config": {"horizon": 1},
            },
        )
        assert resp.status_code == 400

    def test_missing_past_covariates_field_rejected(self, client: TestClient) -> None:
        resp = client.post(
            URL,
            headers=HEADERS,
            json={
                "model": "chronos-2",
                "context": [[1.0, 2.0]],
                "config": {"horizon": 1},
            },
        )
        assert resp.status_code == 422

    def test_covariates_length_must_match_context(self, client: TestClient) -> None:
        body = _payload("chronos-2")
        # context has 2 series, supply only 1 covariate entry
        body["pastCovariates"] = body["pastCovariates"][:1]
        resp = client.post(URL, headers=HEADERS, json=body)
        assert resp.status_code == 400
        assert "context" in resp.text.lower()

    def test_covariates_keys_must_match_across_series(self, client: TestClient) -> None:
        body = _payload("chronos-2")
        body["pastCovariates"] = [
            {"temp": [0.1, 0.2, 0.3, 0.4], "promo": [0.0, 1.0, 0.0, 1.0]},
            {"temp": [1.0, 1.1, 1.2, 1.3]},  # missing 'promo'
        ]
        resp = client.post(URL, headers=HEADERS, json=body)
        assert resp.status_code == 400
        assert "covariate" in resp.text.lower()

    def test_auth_required(self, client: TestClient) -> None:
        resp = client.post(URL, json=_payload("chronos-2"))
        assert resp.status_code == 401


class TestCovariatesPastEnsemble:
    def test_happy_path(self, client: TestClient) -> None:
        body = _payload()
        resp = client.post(ENSEMBLE_URL, headers=HEADERS, json=body)
        assert resp.status_code == 200, resp.text
        out = resp.json()
        assert out["model"] == "ensemble"
        assert sorted(out["ensembleMembers"]) == sorted(MEMBERS)
        for slug in MEMBERS:
            assert out["weights"][slug] == pytest.approx(0.5)

    def test_weight_zero_drops_member(self, client: TestClient) -> None:
        body = _payload()
        body["weights"] = {"chronos-2": 1.0, "moirai-2": 0.0}
        resp = client.post(ENSEMBLE_URL, headers=HEADERS, json=body)
        assert resp.status_code == 200
        out = resp.json()
        assert out["ensembleMembers"] == ["chronos-2"]
        assert out["weights"] == {"chronos-2": pytest.approx(1.0)}
