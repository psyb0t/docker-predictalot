from __future__ import annotations

from fastapi.testclient import TestClient


HEADERS = {"Authorization": "Bearer testtoken"}


class TestForecastToto1:
    def test_happy_path(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast",
            headers=HEADERS,
            json={
                "model": "toto-1",
                "context": [[1.0, 2.0, 3.0, 4.0, 5.0]],
                "config": {"horizon": 3},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "toto-1"
        assert body["horizon"] == 3
        assert len(body["median"]) == 1
        assert len(body["median"][0]) == 3

    def test_batched_series(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast",
            headers=HEADERS,
            json={
                "model": "toto-1",
                "context": [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
                "config": {"horizon": 2},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["median"]) == 2

    def test_appears_in_ensemble_members(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast/ensemble",
            headers=HEADERS,
            json={
                "context": [[1.0, 2.0]],
                "config": {"horizon": 1},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "toto-1" in body["ensembleMembers"]
