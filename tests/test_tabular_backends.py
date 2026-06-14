"""Tests for each tabular backend's train + predict roundtrip across
the three modes (direction / value / quantile).

These exercise the REAL libraries (lightgbm / xgboost / sklearn) on
tiny synthetic data. Stubbing them out would test nothing useful;
they're fast enough to run for real.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

# Skip the whole module if the heavy ML stack isn't installed (dev
# image). Integration runs use the production image which ships all
# backends.
pytest.importorskip("lightgbm")
pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

import numpy as np  # noqa: E402

from predictalot import models  # noqa: E402


from typing import Any  # noqa: E402


# All 9 registered tabular backends.
ALL_SLUGS = [
    "lightgbm",
    "xgboost",
    "hist-gbt",
    "random-forest",
    "logistic",
    "mlp",
    "svm-rbf",
    "knn",
    "naive-bayes",
]

# Subset that surfaces a native, signal-aware importance score. The
# remaining backends (mlp/svm-rbf/knn/naive-bayes) either have no
# meaningful built-in importance or rely on permutation importance
# which is too noisy on n=200 to assert dominance.
NATIVE_IMPORTANCE_SLUGS = [
    "lightgbm",
    "xgboost",
    "hist-gbt",
    "random-forest",
    "logistic",
]


@dataclass
class _Config:
    """Minimal stand-in for the TrainConfig pydantic model."""

    mode: str
    horizon: int = 1
    quantile_levels: list[float] | None = None
    n_estimators: int | None = 50  # keep tests fast
    max_depth: int | None = 3
    learning_rate: float | None = 0.1
    num_leaves: int | None = 7
    min_samples: int | None = None
    random_state: int | None = 0
    # Tier-2 cross-backend knobs (default None so backends fall back).
    categorical_features: list[str] | None = None
    monotonic_constraints: dict[str, int] | None = None
    class_weight: Any = None
    sample_weight: list[float] | None = None
    early_stopping_rounds: int | None = None
    validation_fraction: float | None = None
    # Tier-3 per-backend escape hatch.
    extra: dict[str, Any] | None = None


def _synth(n: int = 200, seed: int = 0):
    """Linearly-separable synthetic dataset with a clear leaky
    feature ``b``."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 3))
    # column 1 ('b') carries the signal: large b → up, small b → down,
    # with a regression target = 2 * b + noise so all 3 modes have a
    # consistent ground truth.
    noise = rng.normal(0, 0.05, size=n)
    y_value = 2.0 * X[:, 1] + noise
    y_direction = (X[:, 1] > 0.5).astype(np.int64)
    return X, y_direction, y_value


@pytest.mark.parametrize("slug", ALL_SLUGS)
class TestTabularBackendDirection:
    def test_train_returns_blob_and_importance(self, slug: str) -> None:
        be = models.get_tabular_backend(slug)
        X, y, _ = _synth()
        out = be.train(X, y, ["a", "b", "c"], _Config(mode="direction"))
        assert isinstance(out["blob"], bytes)
        assert set(out["importance"]) == {"a", "b", "c"}
        if slug in NATIVE_IMPORTANCE_SLUGS:
            # 'b' is the leaky feature; native importance should rank it
            # at the top.
            assert out["importance"]["b"] > max(
                out["importance"]["a"], out["importance"]["c"]
            )

    def test_predict_returns_prob_up_in_unit_range(self, slug: str) -> None:
        be = models.get_tabular_backend(slug)
        X, y, _ = _synth()
        out = be.train(X, y, ["a", "b", "c"], _Config(mode="direction"))
        # Predict on rows with very high vs very low 'b'.
        X_test = np.array([
            [0.5, 0.95, 0.5],   # 'b' high → expect up
            [0.5, 0.05, 0.5],   # 'b' low → expect down
        ])
        pred = be.predict(out["blob"], X_test, "direction")
        assert "prob_up" in pred
        prob = pred["prob_up"]
        assert prob.shape == (2,)
        assert 0.0 <= prob[0] <= 1.0
        assert 0.0 <= prob[1] <= 1.0
        # The model should mostly get the leaky-signal right.
        assert prob[0] > prob[1]


@pytest.mark.parametrize("slug", ALL_SLUGS)
class TestTabularBackendValue:
    def test_predict_returns_floats(self, slug: str) -> None:
        be = models.get_tabular_backend(slug)
        X, _, y = _synth()
        out = be.train(X, y, ["a", "b", "c"], _Config(mode="value"))
        pred = be.predict(out["blob"], X[:5], "value")
        assert "predicted" in pred
        assert pred["predicted"].shape == (5,)

    def test_value_predictions_correlate_with_b(self, slug: str) -> None:
        be = models.get_tabular_backend(slug)
        X, _, y = _synth(n=400)
        out = be.train(X, y, ["a", "b", "c"], _Config(mode="value"))
        X_test = np.array([
            [0.5, 0.1, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.9, 0.5],
        ])
        pred = be.predict(out["blob"], X_test, "value")
        # Monotonic in 'b' (which is the regression-target driver).
        assert pred["predicted"][0] < pred["predicted"][1] < pred["predicted"][2]


@pytest.mark.parametrize("slug", ALL_SLUGS)
class TestTabularBackendQuantile:
    def test_quantile_returns_median_and_quantiles(self, slug: str) -> None:
        be = models.get_tabular_backend(slug)
        X, _, y = _synth(n=400)
        cfg = _Config(mode="quantile", quantile_levels=[0.1, 0.5, 0.9])
        out = be.train(X, y, ["a", "b", "c"], cfg)
        pred = be.predict(out["blob"], X[:5], "quantile", [0.1, 0.5, 0.9])
        assert "median" in pred
        assert "quantiles" in pred
        assert set(pred["quantiles"]) == {"0.1", "0.5", "0.9"}
        assert pred["median"].shape == (5,)
        for arr in pred["quantiles"].values():
            assert arr.shape == (5,)


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_quantile_mode_requires_levels(slug: str) -> None:
    be = models.get_tabular_backend(slug)
    X, _, y = _synth()
    with pytest.raises(ValueError, match="quantile_levels"):
        be.train(X, y, ["a", "b", "c"], _Config(mode="quantile"))


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_unknown_mode_raises(slug: str) -> None:
    be = models.get_tabular_backend(slug)
    X, _, y = _synth()
    with pytest.raises(ValueError, match="unsupported mode"):
        be.train(X, y, ["a", "b", "c"], _Config(mode="bogus"))


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_predict_with_wrong_mode_raises(slug: str) -> None:
    be = models.get_tabular_backend(slug)
    X, y, _ = _synth()
    out = be.train(X, y, ["a", "b", "c"], _Config(mode="direction"))
    with pytest.raises(ValueError, match="trained for mode='direction'"):
        be.predict(out["blob"], X[:3], "value")


def test_registry_lists_all_nine_backends() -> None:
    slugs = models.tabular_backend_slugs()
    assert set(slugs) == set(ALL_SLUGS)


def test_get_tabular_backend_raises_on_unknown_slug() -> None:
    with pytest.raises(KeyError, match="unknown tabular backend"):
        models.get_tabular_backend("does-not-exist")
