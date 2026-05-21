from __future__ import annotations

from fastapi.testclient import TestClient


HEADERS = {"Authorization": "Bearer testtoken"}


class TestForecastChronos2:
    def test_happy_path(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast",
            headers=HEADERS,
            json={
                "model": "chronos-2",
                "context": [[1.0, 2.0, 3.0, 4.0, 5.0]],
                "config": {"horizon": 3},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "chronos-2"
        assert body["horizon"] == 3
        assert body["quantileLevels"] == [0.1, 0.5, 0.9]
        assert len(body["median"]) == 1
        assert len(body["median"][0]) == 3
        assert set(body["quantiles"].keys()) == {"0.1", "0.5", "0.9"}

    def test_explicit_quantile_levels(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast",
            headers=HEADERS,
            json={
                "model": "chronos-2",
                "context": [[1.0, 2.0, 3.0]],
                "config": {"horizon": 2, "quantileLevels": [0.2, 0.5, 0.8]},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["quantileLevels"] == [0.2, 0.5, 0.8]
        assert set(body["quantiles"].keys()) == {"0.2", "0.5", "0.8"}

    def test_batched_series(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast",
            headers=HEADERS,
            json={
                "model": "chronos-2",
                "context": [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]],
                "config": {"horizon": 2},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["median"]) == 3
        for q in body["quantiles"].values():
            assert len(q) == 3

    def test_empty_context_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast",
            headers=HEADERS,
            json={"model": "chronos-2", "context": [], "config": {"horizon": 3}},
        )
        assert resp.status_code == 400

    def test_empty_series_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast",
            headers=HEADERS,
            json={"model": "chronos-2", "context": [[]], "config": {"horizon": 3}},
        )
        assert resp.status_code == 400

    def test_horizon_must_be_positive(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast",
            headers=HEADERS,
            json={
                "model": "chronos-2",
                "context": [[1.0, 2.0]],
                "config": {"horizon": 0},
            },
        )
        assert resp.status_code == 422  # pydantic Field(gt=0)

    def test_invalid_quantile_level_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast",
            headers=HEADERS,
            json={
                "model": "chronos-2",
                "context": [[1.0, 2.0]],
                "config": {"horizon": 1, "quantileLevels": [0.05]},
            },
        )
        assert resp.status_code == 400

    def test_unknown_model_404(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast",
            headers=HEADERS,
            json={
                "model": "nonexistent-model",
                "context": [[1.0]],
                "config": {"horizon": 1},
            },
        )
        assert resp.status_code == 404
