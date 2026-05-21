from __future__ import annotations

from fastapi.testclient import TestClient


HEADERS = {"Authorization": "Bearer testtoken"}


class TestForecastMoirai2:
    def test_happy_path(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast",
            headers=HEADERS,
            json={
                "model": "moirai-2",
                "context": [[1.0, 2.0, 3.0, 4.0, 5.0]],
                "config": {"horizon": 5, "contextLength": 4000},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "moirai-2"
        assert body["horizon"] == 5

    def test_duplicate_quantile_levels_dedup(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast",
            headers=HEADERS,
            json={
                "model": "moirai-2",
                "context": [[1.0, 2.0]],
                "config": {"horizon": 1, "quantileLevels": [0.5, 0.5, 0.5]},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["quantileLevels"] == [0.5]
