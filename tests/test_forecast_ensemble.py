from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


HEADERS = {"Authorization": "Bearer testtoken"}


class TestForecastEnsemble:
    def test_happy_path(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast/ensemble",
            headers=HEADERS,
            json={
                "context": [[1.0, 2.0, 3.0, 4.0, 5.0]],
                "config": {"horizon": 3},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "ensemble"
        assert body["horizon"] == 3
        assert body["quantileLevels"] == [0.1, 0.5, 0.9]
        assert sorted(body["ensembleMembers"]) == sorted(
            ["chronos-2", "timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m"]
        )
        assert len(body["median"]) == 1
        assert len(body["median"][0]) == 3
        assert set(body["quantiles"].keys()) == {"0.1", "0.5", "0.9"}

        # individual results present, each with its own forecast + weight
        assert set(body["individual"].keys()) == {
            "chronos-2", "timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m"
        }
        for slug, ind in body["individual"].items():
            assert ind["model"] == slug
            assert ind["horizon"] == 3
            assert len(ind["median"]) == 1
            assert len(ind["median"][0]) == 3
            assert set(ind["quantiles"].keys()) == {"0.1", "0.5", "0.9"}
            assert ind["weight"] == pytest.approx(1 / 5)
        assert body["weights"] == {
            "chronos-2": pytest.approx(1 / 5),
            "timesfm-2.5": pytest.approx(1 / 5),
            "moirai-2": pytest.approx(1 / 5),
            "toto-1": pytest.approx(1 / 5),
            "sundial-base-128m": pytest.approx(1 / 5),
        }

    def test_custom_weights_normalize(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast/ensemble",
            headers=HEADERS,
            json={
                "context": [[2.0, 4.0]],
                "config": {"horizon": 1},
                "weights": {
                    "chronos-2": 2.0,
                    "timesfm-2.5": 1.0,
                    "moirai-2": 1.0,
                    "toto-1": 0.0,
                    "sundial-base-128m": 0.0,  # drop sidecar models too for testable math
                },
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        # Active: 2+1+1=4 → chronos=0.5, timesfm=0.25, moirai=0.25; toto skipped
        assert body["weights"]["chronos-2"] == pytest.approx(0.5)
        assert body["weights"]["timesfm-2.5"] == pytest.approx(0.25)
        assert body["weights"]["moirai-2"] == pytest.approx(0.25)
        assert "toto-1" not in body["weights"]
        for slug, expected in body["weights"].items():
            assert body["individual"][slug]["weight"] == pytest.approx(expected)

    def test_weight_zero_skips_model(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast/ensemble",
            headers=HEADERS,
            json={
                "context": [[1.0, 2.0]],
                "config": {"horizon": 1},
                "weights": {
                    "chronos-2": 1.0,
                    "timesfm-2.5": 0.0,
                    "moirai-2": 1.0,
                    "toto-1": 0.0,
                    "sundial-base-128m": 0.0,
                },
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "timesfm-2.5" not in body["individual"]
        assert "timesfm-2.5" not in body["weights"]
        assert "toto-1" not in body["individual"]
        assert sorted(body["ensembleMembers"]) == ["chronos-2", "moirai-2"]
        assert body["weights"]["chronos-2"] == pytest.approx(0.5)
        assert body["weights"]["moirai-2"] == pytest.approx(0.5)

    def test_all_zero_weights_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast/ensemble",
            headers=HEADERS,
            json={
                "context": [[1.0]],
                "config": {"horizon": 1},
                "weights": {
                    "chronos-2": 0,
                    "timesfm-2.5": 0,
                    "moirai-2": 0,
                    "toto-1": 0,
                    "sundial-base-128m": 0,
                },
            },
        )
        assert resp.status_code == 400

    def test_unknown_model_in_weights_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast/ensemble",
            headers=HEADERS,
            json={
                "context": [[1.0, 2.0]],
                "config": {"horizon": 1},
                "weights": {"chronos-2": 1.0, "made-up-model": 1.0},
            },
        )
        assert resp.status_code == 400

    def test_negative_weight_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast/ensemble",
            headers=HEADERS,
            json={
                "context": [[1.0, 2.0]],
                "config": {"horizon": 1},
                "weights": {"chronos-2": -1.0},
            },
        )
        assert resp.status_code == 400

    def test_no_model_field(self, client: TestClient) -> None:
        """The ensemble endpoint ignores any `model` field — should still work."""
        resp = client.post(
            "/v1/forecast/ensemble",
            headers=HEADERS,
            json={
                "model": "chronos-2",  # extra field — Pydantic ignores by default
                "context": [[1.0, 2.0]],
                "config": {"horizon": 1},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "ensemble"

    def test_average_of_stubs(self, client: TestClient) -> None:
        """conftest stubs make every model return mean(context) + step_offset.
        For three identical stub functions, the ensemble mean == any single's output."""
        resp = client.post(
            "/v1/forecast/ensemble",
            headers=HEADERS,
            json={
                "context": [[2.0, 4.0]],  # mean = 3.0
                "config": {"horizon": 2},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        # Stub: median[s][t] = mean(s) + t  →  [3.0, 4.0] for series 0.
        # Use approx since the weighted-mean across N stubs introduces
        # float-rounding noise on the last bits.
        assert body["median"][0] == pytest.approx([3.0, 4.0])

    def test_auth_required(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast/ensemble",
            json={"context": [[1.0]], "config": {"horizon": 1}},
        )
        assert resp.status_code == 401

    def test_bad_quantile_levels(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast/ensemble",
            headers=HEADERS,
            json={
                "context": [[1.0, 2.0]],
                "config": {"horizon": 1, "quantileLevels": [0.05]},
            },
        )
        assert resp.status_code == 400
