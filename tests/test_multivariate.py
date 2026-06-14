"""POST /v1/multivariate/{forecast,forecast/ensemble} + capability rejection.

Members of TYPE_MULTIVARIATE: chronos-2, moirai-2, toto-1. Forecasts are
shaped [series][channel][time] — both ``median`` and ``quantiles[level]``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


HEADERS = {"Authorization": "Bearer testtoken"}
URL = "/v1/timeseries/multivariate/forecast"
ENSEMBLE_URL = "/v1/timeseries/multivariate/forecast/ensemble"
MEMBERS = ["chronos-2", "moirai-2", "toto-1"]


def _example_context() -> list[list[list[float]]]:
    # Two series, three channels per series, four observations.
    return [
        [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], [100.0, 200.0, 300.0, 400.0]],
        [[5.0, 6.0, 7.0, 8.0], [50.0, 60.0, 70.0, 80.0], [500.0, 600.0, 700.0, 800.0]],
    ]


@pytest.mark.parametrize("model_slug", MEMBERS)
class TestMultivariateHappyPath:
    def test_shapes(self, client: TestClient, model_slug: str) -> None:
        ctx = _example_context()
        resp = client.post(
            URL,
            headers=HEADERS,
            json={
                "model": model_slug,
                "context": ctx,
                "config": {"horizon": 2},
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model"] == model_slug
        assert body["horizon"] == 2
        assert body["quantileLevels"] == [0.1, 0.5, 0.9]
        # [series][channel][time]
        assert len(body["median"]) == 2
        assert len(body["median"][0]) == 3
        assert len(body["median"][0][0]) == 2
        for q_level, q_payload in body["quantiles"].items():
            assert q_level in {"0.1", "0.5", "0.9"}
            assert len(q_payload) == 2
            assert len(q_payload[0]) == 3
            assert len(q_payload[0][0]) == 2


class TestMultivariateValidation:
    def test_capability_rejection_for_timesfm(self, client: TestClient) -> None:
        """timesfm-2.5 does not support multivariate → 400."""
        resp = client.post(
            URL,
            headers=HEADERS,
            json={
                "model": "timesfm-2.5",
                "context": _example_context(),
                "config": {"horizon": 1},
            },
        )
        assert resp.status_code == 400

    def test_capability_rejection_for_sundial(self, client: TestClient) -> None:
        resp = client.post(
            URL,
            headers=HEADERS,
            json={
                "model": "sundial-base-128m",
                "context": _example_context(),
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
                "context": _example_context(),
                "config": {"horizon": 1},
            },
        )
        assert resp.status_code == 404

    def test_empty_context_rejected(self, client: TestClient) -> None:
        resp = client.post(
            URL,
            headers=HEADERS,
            json={"model": "chronos-2", "context": [], "config": {"horizon": 1}},
        )
        assert resp.status_code == 400

    def test_empty_channel_rejected(self, client: TestClient) -> None:
        resp = client.post(
            URL,
            headers=HEADERS,
            json={
                "model": "chronos-2",
                "context": [[[]]],
                "config": {"horizon": 1},
            },
        )
        assert resp.status_code == 400

    def test_jagged_channel_counts_rejected(self, client: TestClient) -> None:
        """Series with mismatched channel counts must be rejected at dispatch
        so behavior is uniform regardless of which backend the request lands
        on (some backends silently accept the mismatch, some don't)."""
        resp = client.post(
            URL,
            headers=HEADERS,
            json={
                "model": "chronos-2",
                "context": [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],          # 2 channels
                    [[7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # 3 channels
                ],
                "config": {"horizon": 1},
            },
        )
        assert resp.status_code == 400
        assert "channel" in resp.text.lower()

    def test_auth_required(self, client: TestClient) -> None:
        resp = client.post(
            URL,
            json={
                "model": "chronos-2",
                "context": _example_context(),
                "config": {"horizon": 1},
            },
        )
        assert resp.status_code == 401


class TestMultivariateEnsemble:
    def test_happy_path(self, client: TestClient) -> None:
        resp = client.post(
            ENSEMBLE_URL,
            headers=HEADERS,
            json={
                "context": _example_context(),
                "config": {"horizon": 2},
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model"] == "ensemble"
        assert sorted(body["ensembleMembers"]) == sorted(MEMBERS)
        # MV ensemble carries the [series][channel][time] shape
        assert len(body["median"]) == 2
        assert len(body["median"][0]) == 3
        assert len(body["median"][0][0]) == 2
        for slug in MEMBERS:
            assert body["weights"][slug] == pytest.approx(1 / 3)
            ind = body["individual"][slug]
            assert ind["model"] == slug
            assert ind["weight"] == pytest.approx(1 / 3)

    def test_weight_excludes_unsupported_model(self, client: TestClient) -> None:
        """Weights for a slug that doesn't support multivariate → 400."""
        resp = client.post(
            ENSEMBLE_URL,
            headers=HEADERS,
            json={
                "context": _example_context(),
                "config": {"horizon": 1},
                "weights": {"chronos-2": 1.0, "timesfm-2.5": 1.0},
            },
        )
        assert resp.status_code == 400
