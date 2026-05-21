from __future__ import annotations

from fastapi.testclient import TestClient


HEADERS = {"Authorization": "Bearer testtoken"}


class TestForecastSundial:
    def test_happy_path(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/forecast",
            headers=HEADERS,
            json={
                "model": "sundial-base-128m",
                "context": [[1.0, 2.0, 3.0, 4.0, 5.0]],
                "config": {"horizon": 3},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "sundial-base-128m"
        assert body["horizon"] == 3

    def test_appears_in_models_list(self, client: TestClient) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        slugs = [m["slug"] for m in resp.json()["models"]]
        assert "sundial-base-128m" in slugs

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
        assert "sundial-base-128m" in body["ensembleMembers"]
