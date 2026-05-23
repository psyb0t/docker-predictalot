"""POST /v1/univariate/forecast/ensemble — weighted-mean ensemble across all
5 univariate-supporting backends."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


HEADERS = {"Authorization": "Bearer testtoken"}
URL = "/v1/univariate/forecast/ensemble"
MEMBERS = ["chronos-2", "timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m"]


class TestUnivariateEnsemble:
    def test_happy_path(self, client: TestClient) -> None:
        resp = client.post(
            URL,
            headers=HEADERS,
            json={
                "context": [[1.0, 2.0, 3.0, 4.0, 5.0]],
                "config": {"horizon": 3},
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model"] == "ensemble"
        assert body["horizon"] == 3
        assert body["quantileLevels"] == [0.1, 0.5, 0.9]
        assert sorted(body["ensembleMembers"]) == sorted(MEMBERS)
        assert len(body["median"]) == 1
        assert len(body["median"][0]) == 3
        assert set(body["quantiles"].keys()) == {"0.1", "0.5", "0.9"}

        assert set(body["individual"].keys()) == set(MEMBERS)
        for slug, ind in body["individual"].items():
            assert ind["model"] == slug
            assert ind["horizon"] == 3
            assert len(ind["median"]) == 1
            assert len(ind["median"][0]) == 3
            assert set(ind["quantiles"].keys()) == {"0.1", "0.5", "0.9"}
            assert ind["weight"] == pytest.approx(1 / 5)
        assert body["weights"] == {slug: pytest.approx(1 / 5) for slug in MEMBERS}

    def test_custom_weights_normalize(self, client: TestClient) -> None:
        resp = client.post(
            URL,
            headers=HEADERS,
            json={
                "context": [[2.0, 4.0]],
                "config": {"horizon": 1},
                "weights": {
                    "chronos-2": 2.0,
                    "timesfm-2.5": 1.0,
                    "moirai-2": 1.0,
                    "toto-1": 0.0,
                    "sundial-base-128m": 0.0,
                },
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["weights"]["chronos-2"] == pytest.approx(0.5)
        assert body["weights"]["timesfm-2.5"] == pytest.approx(0.25)
        assert body["weights"]["moirai-2"] == pytest.approx(0.25)
        assert "toto-1" not in body["weights"]
        assert "sundial-base-128m" not in body["weights"]
        for slug, expected in body["weights"].items():
            assert body["individual"][slug]["weight"] == pytest.approx(expected)

    def test_weight_zero_skips_model(self, client: TestClient) -> None:
        resp = client.post(
            URL,
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
            URL,
            headers=HEADERS,
            json={
                "context": [[1.0]],
                "config": {"horizon": 1},
                "weights": {slug: 0 for slug in MEMBERS},
            },
        )
        assert resp.status_code == 400

    def test_unknown_model_in_weights_rejected(self, client: TestClient) -> None:
        resp = client.post(
            URL,
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
            URL,
            headers=HEADERS,
            json={
                "context": [[1.0, 2.0]],
                "config": {"horizon": 1},
                "weights": {"chronos-2": -1.0},
            },
        )
        assert resp.status_code == 400

    def test_extra_model_field_ignored(self, client: TestClient) -> None:
        """Ensemble request schema has no `model` field — Pydantic ignores extras."""
        resp = client.post(
            URL,
            headers=HEADERS,
            json={
                "model": "chronos-2",
                "context": [[1.0, 2.0]],
                "config": {"horizon": 1},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == "ensemble"

    def test_aggregate_of_stubs(self, client: TestClient) -> None:
        """Stub returns mean(series) + step_offset for every member;
        any weighted mean of identical values yields the same value."""
        resp = client.post(
            URL,
            headers=HEADERS,
            json={
                "context": [[2.0, 4.0]],
                "config": {"horizon": 2},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["median"][0] == pytest.approx([3.0, 4.0])

    def test_auth_required(self, client: TestClient) -> None:
        resp = client.post(URL, json={"context": [[1.0]], "config": {"horizon": 1}})
        assert resp.status_code == 401

    def test_bad_quantile_levels(self, client: TestClient) -> None:
        resp = client.post(
            URL,
            headers=HEADERS,
            json={
                "context": [[1.0, 2.0]],
                "config": {"horizon": 1, "quantileLevels": [0.05]},
            },
        )
        assert resp.status_code == 400

    def test_infinite_weight_rejected(self, client: TestClient) -> None:
        # JSON spec disallows Infinity, but Python's stdlib json accepts it
        # (and many clients serialize it as the bare token). An infinite
        # weight would silently NaN-poison the entire ensemble output, so
        # dispatch must reject it before normalization.
        resp = client.post(
            URL,
            headers={**HEADERS, "Content-Type": "application/json"},
            content=b'{"context":[[1.0,2.0]],"config":{"horizon":1},'
            b'"weights":{"chronos-2": Infinity}}',
        )
        assert resp.status_code == 400
        assert "finite" in resp.text.lower() or "infinit" in resp.text.lower()

    def test_nan_weight_rejected(self, client: TestClient) -> None:
        resp = client.post(
            URL,
            headers={**HEADERS, "Content-Type": "application/json"},
            content=b'{"context":[[1.0,2.0]],"config":{"horizon":1},'
            b'"weights":{"chronos-2": NaN}}',
        )
        assert resp.status_code == 400
        assert "finite" in resp.text.lower() or "nan" in resp.text.lower()
