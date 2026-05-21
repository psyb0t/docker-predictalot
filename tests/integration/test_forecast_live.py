"""End-to-end forecast tests against a real container with real ML libs.

First run per model downloads HuggingFace weights into the bind-mounted cache
(~120 MB chronos-2 + ~200 MB timesfm-2.5 + ~50 MB moirai-2). Cached on host
so subsequent runs are fast.
"""

from __future__ import annotations

import math

import httpx
import pytest


pytestmark = pytest.mark.integration


def _sane_floats(values: list[float]) -> None:
    """Assert every value is finite and not absurd."""
    for v in values:
        assert isinstance(v, (int, float))
        assert not math.isnan(v), f"NaN in prediction: {values}"
        assert math.isfinite(v), f"non-finite in prediction: {values}"
        assert -1e9 < v < 1e9, f"absurd magnitude in prediction: {v}"


class TestForecastLive:
    @pytest.mark.parametrize(
        "model",
        ["chronos-2", "timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m"],
    )
    def test_single_series(self, http_client: httpx.Client, model: str) -> None:
        # Simple monotonic input — forecast should produce reasonable values.
        context = [[float(i) for i in range(1, 50)]]
        resp = http_client.post(
            "/v1/forecast",
            json={"model": model, "context": context, "config": {"horizon": 5}},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model"] == model
        assert body["horizon"] == 5
        assert body["quantileLevels"] == [0.1, 0.5, 0.9]

        # Shape: one series in → one series out, length = horizon
        assert len(body["median"]) == 1
        assert len(body["median"][0]) == 5
        _sane_floats(body["median"][0])

        for q_key in ("0.1", "0.5", "0.9"):
            assert q_key in body["quantiles"]
            assert len(body["quantiles"][q_key]) == 1
            assert len(body["quantiles"][q_key][0]) == 5
            _sane_floats(body["quantiles"][q_key][0])

        # Quantile ordering: q0.1 <= q0.5 <= q0.9 per step (small tolerance
        # for numerical noise; foundation models don't always strictly satisfy this)
        for step in range(5):
            q1 = body["quantiles"]["0.1"][0][step]
            q5 = body["quantiles"]["0.5"][0][step]
            q9 = body["quantiles"]["0.9"][0][step]
            assert q1 <= q5 + 1e-3, f"q0.1 > q0.5 at step {step}: {q1} > {q5}"
            assert q5 <= q9 + 1e-3, f"q0.5 > q0.9 at step {step}: {q5} > {q9}"

    @pytest.mark.parametrize(
        "model",
        ["chronos-2", "timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m"],
    )
    def test_batched_series(self, http_client: httpx.Client, model: str) -> None:
        context = [
            [float(i) for i in range(1, 30)],
            [float(i * 10) for i in range(1, 30)],
            [float(i * 100) for i in range(1, 30)],
        ]
        resp = http_client.post(
            "/v1/forecast",
            json={"model": model, "context": context, "config": {"horizon": 3}},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["median"]) == 3
        for series in body["median"]:
            assert len(series) == 3
            _sane_floats(series)

    def test_custom_quantile_levels(self, http_client: httpx.Client) -> None:
        resp = http_client.post(
            "/v1/forecast",
            json={
                "model": "chronos-2",
                "context": [[1.0, 2.0, 3.0, 4.0, 5.0]],
                "config": {"horizon": 2, "quantileLevels": [0.2, 0.5, 0.8]},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["quantileLevels"] == [0.2, 0.5, 0.8]
        assert set(body["quantiles"].keys()) == {"0.2", "0.5", "0.8"}

    def test_unload_flag(self, http_client: httpx.Client) -> None:
        # Load a model
        r1 = http_client.post(
            "/v1/forecast",
            json={"model": "moirai-2", "context": [[1.0, 2.0, 3.0]], "config": {"horizon": 1}},
        )
        assert r1.status_code == 200

        # /v1/models should show it loaded
        info_loaded = http_client.get("/v1/models").json()
        moirai_info = next(m for m in info_loaded["models"] if m["slug"] == "moirai-2")
        assert moirai_info["loaded"] is True

        # Call again with unload=True
        r2 = http_client.post(
            "/v1/forecast",
            json={
                "model": "moirai-2",
                "context": [[1.0, 2.0, 3.0]],
                "config": {"horizon": 1},
                "unload": True,
            },
        )
        assert r2.status_code == 200

        # Now it should be unloaded
        info_unloaded = http_client.get("/v1/models").json()
        moirai_info = next(m for m in info_unloaded["models"] if m["slug"] == "moirai-2")
        assert moirai_info["loaded"] is False

    def test_ensemble(self, http_client: httpx.Client) -> None:
        context = [[float(i) for i in range(1, 50)]]
        resp = http_client.post(
            "/v1/forecast/ensemble",
            json={"context": context, "config": {"horizon": 5}},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model"] == "ensemble"
        assert body["horizon"] == 5
        assert sorted(body["ensembleMembers"]) == sorted(
            ["chronos-2", "timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m"]
        )
        assert len(body["median"]) == 1
        assert len(body["median"][0]) == 5
        _sane_floats(body["median"][0])
        for q in ("0.1", "0.5", "0.9"):
            _sane_floats(body["quantiles"][q][0])

    def test_cuda_when_available(
        self, http_client: httpx.Client, cuda_available: bool
    ) -> None:
        """If we're running on a CUDA host, the container should be on GPU."""
        if not cuda_available:
            pytest.skip("no CUDA on host — running CPU image")

        # No direct API to ask the container "are you on GPU?", but we can
        # check that the image tag was the cuda variant (indirectly via the
        # build-image fixture's tag).  Simpler: just verify a forecast works
        # and trust that --gpus all was passed.  A more thorough check would
        # exec into the container and run `nvidia-smi`.
        resp = http_client.post(
            "/v1/forecast",
            json={"model": "moirai-2", "context": [[1.0, 2.0, 3.0]], "config": {"horizon": 1}},
        )
        assert resp.status_code == 200
