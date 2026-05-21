from __future__ import annotations

from fastapi.testclient import TestClient


HEADERS = {"Authorization": "Bearer testtoken"}


class TestForecastTimesFM:
    def test_happy_path(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast",
            headers=HEADERS,
            json={
                "model": "timesfm-2.5",
                "context": [[1.0, 2.0, 3.0, 4.0]],
                "config": {"horizon": 4},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "timesfm-2.5"
        assert body["horizon"] == 4

    def test_unload_flag(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast",
            headers=HEADERS,
            json={
                "model": "timesfm-2.5",
                "context": [[1.0, 2.0, 3.0]],
                "config": {"horizon": 1},
                "unload": True,
            },
        )
        assert resp.status_code == 200

    def test_default_quantile_levels(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast",
            headers=HEADERS,
            json={
                "model": "timesfm-2.5",
                "context": [[1.0, 2.0, 3.0]],
                "config": {"horizon": 1},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["quantileLevels"] == [0.1, 0.5, 0.9]
