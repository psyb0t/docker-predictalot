"""POST /v1/samples/{forecast,forecast/ensemble}.

Members of TYPE_SAMPLES: toto-1, sundial-base-128m.

Returns raw Monte-Carlo sample paths instead of quantiles. Shape:
``samples[series][sample][time]``; convenience ``median[series][time]``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


HEADERS = {"Authorization": "Bearer testtoken"}
URL = "/v1/samples/forecast"
ENSEMBLE_URL = "/v1/samples/forecast/ensemble"
MEMBERS = ["toto-1", "sundial-base-128m"]


@pytest.mark.parametrize("model_slug", MEMBERS)
class TestSamplesHappyPath:
    def test_default_sample_count(self, client: TestClient, model_slug: str) -> None:
        resp = client.post(
            URL,
            headers=HEADERS,
            json={
                "model": model_slug,
                "context": [[1.0, 2.0, 3.0, 4.0]],
                "config": {"horizon": 3},
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model"] == model_slug
        assert body["horizon"] == 3
        # Default DEFAULT_SAMPLES = 64 in dispatch.
        assert body["numSamples"] == 64
        assert len(body["samples"]) == 1
        assert len(body["samples"][0]) == 64
        assert len(body["samples"][0][0]) == 3
        assert len(body["median"]) == 1
        assert len(body["median"][0]) == 3

    def test_explicit_num_samples(self, client: TestClient, model_slug: str) -> None:
        resp = client.post(
            URL,
            headers=HEADERS,
            json={
                "model": model_slug,
                "context": [[1.0, 2.0, 3.0]],
                "config": {"horizon": 2, "numSamples": 16},
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["numSamples"] == 16
        assert len(body["samples"][0]) == 16


class TestSamplesValidation:
    @pytest.mark.parametrize(
        "unsupported", ["chronos-2", "timesfm-2.5", "moirai-2"]
    )
    def test_capability_rejection(self, client: TestClient, unsupported: str) -> None:
        resp = client.post(
            URL,
            headers=HEADERS,
            json={
                "model": unsupported,
                "context": [[1.0, 2.0]],
                "config": {"horizon": 1},
            },
        )
        assert resp.status_code == 400

    def test_unknown_model_404(self, client: TestClient) -> None:
        resp = client.post(
            URL,
            headers=HEADERS,
            json={
                "model": "nonexistent",
                "context": [[1.0, 2.0]],
                "config": {"horizon": 1},
            },
        )
        assert resp.status_code == 404

    def test_zero_num_samples_rejected_by_schema(self, client: TestClient) -> None:
        resp = client.post(
            URL,
            headers=HEADERS,
            json={
                "model": "toto-1",
                "context": [[1.0, 2.0]],
                "config": {"horizon": 1, "numSamples": 0},
            },
        )
        # SamplesForecastConfig.num_samples uses Field(gt=0) → 422
        assert resp.status_code == 422

    def test_auth_required(self, client: TestClient) -> None:
        resp = client.post(
            URL,
            json={
                "model": "toto-1",
                "context": [[1.0, 2.0]],
                "config": {"horizon": 1},
            },
        )
        assert resp.status_code == 401


class TestSamplesEnsemble:
    def test_happy_path(self, client: TestClient) -> None:
        resp = client.post(
            ENSEMBLE_URL,
            headers=HEADERS,
            json={
                "context": [[1.0, 2.0, 3.0]],
                "config": {"horizon": 2, "numSamples": 32},
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model"] == "ensemble"
        assert sorted(body["ensembleMembers"]) == sorted(MEMBERS)
        # 32 total → 16 per member (uniform weights). Concat = 32 paths.
        assert body["numSamples"] == 32
        assert len(body["samples"]) == 1
        assert len(body["samples"][0]) == 32
        assert len(body["samples"][0][0]) == 2
        for slug in MEMBERS:
            assert body["weights"][slug] == pytest.approx(0.5)
            ind = body["individual"][slug]
            assert ind["numSamples"] == 16

    def test_weight_zero_drops_member(self, client: TestClient) -> None:
        resp = client.post(
            ENSEMBLE_URL,
            headers=HEADERS,
            json={
                "context": [[1.0, 2.0]],
                "config": {"horizon": 1, "numSamples": 4},
                "weights": {"toto-1": 1.0, "sundial-base-128m": 0.0},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ensembleMembers"] == ["toto-1"]
        assert body["weights"] == {"toto-1": pytest.approx(1.0)}
        assert body["numSamples"] == 4

    def test_minimum_one_per_member_when_rounded_to_zero(self, client: TestClient) -> None:
        """Per-member share = max(1, round(weight*total)). A 99/1 split with
        small total still gives the minority member 1 sample, not 0."""
        resp = client.post(
            ENSEMBLE_URL,
            headers=HEADERS,
            json={
                "context": [[1.0, 2.0]],
                "config": {"horizon": 1, "numSamples": 2},
                "weights": {"toto-1": 99.0, "sundial-base-128m": 1.0},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["individual"]["sundial-base-128m"]["numSamples"] >= 1
        # Total = sum of per-member shares.
        total = sum(body["individual"][s]["numSamples"] for s in body["ensembleMembers"])
        assert body["numSamples"] == total
