"""Tests for ``tabular_features.build_training_matrix`` and
``build_forecast_matrix``.

These are pure functions — no HTTP needed.
"""

from __future__ import annotations

import pytest

from predictalot import tabular_features as f


# ─── build_training_matrix ──────────────────────────────────────────────────


def test_direction_target_encoding_uses_sign_of_future_move() -> None:
    target = [10.0, 11.0, 10.5, 12.0, 11.5]
    feats = {"x": [1.0, 2.0, 3.0, 4.0, 5.0]}
    X, y, names, _ = f.build_training_matrix(
        target, feats, horizon=1, mode="direction", min_samples=None,
    )
    # rows 0..3 are usable (need horizon-ahead label)
    assert X.shape == (4, 1)
    assert names == ["x"]
    # target[1]=11>10 → up (1); 10.5<11 → 0; 12>10.5 → 1; 11.5<12 → 0
    assert y.tolist() == [1, 0, 1, 0]


def test_value_target_returns_future_value() -> None:
    target = [10.0, 11.0, 10.5, 12.0, 11.5]
    feats = {"x": [1.0, 2.0, 3.0, 4.0, 5.0]}
    X, y, _, _ = f.build_training_matrix(
        target, feats, horizon=2, mode="value", min_samples=None,
    )
    # rows 0..2 are usable (need horizon=2 ahead)
    assert X.shape == (3, 1)
    assert y.tolist() == [10.5, 12.0, 11.5]


def test_quantile_target_same_as_value() -> None:
    target = [1.0, 2.0, 3.0, 4.0]
    feats = {"x": [0.0, 0.0, 0.0, 0.0]}
    _, y_v, _, _ = f.build_training_matrix(
        target, feats, horizon=1, mode="value", min_samples=None,
    )
    _, y_q, _, _ = f.build_training_matrix(
        target, feats, horizon=1, mode="quantile", min_samples=None,
    )
    assert y_v.tolist() == y_q.tolist()


def test_feature_names_sorted_alphabetically() -> None:
    target = [1.0, 2.0, 3.0, 4.0]
    feats = {"z": [1.0, 2.0, 3.0, 4.0], "a": [5.0, 6.0, 7.0, 8.0], "m": [9.0]*4}
    _, _, names, _ = f.build_training_matrix(
        target, feats, horizon=1, mode="direction", min_samples=None,
    )
    assert names == ["a", "m", "z"]


def test_all_zero_feature_rows_get_pruned() -> None:
    # First two rows are pure zeros (warmup fill) — should drop.
    target = [10.0, 11.0, 12.0, 13.0, 14.0]
    feats = {"x": [0.0, 0.0, 1.0, 2.0, 3.0]}
    X, y, _, _ = f.build_training_matrix(
        target, feats, horizon=1, mode="direction", min_samples=None,
    )
    # Usable rows: 0,1,2,3; rows 0+1 pruned. Rows 2,3 remain.
    assert X.shape == (2, 1)
    assert y.tolist() == [1, 1]


def test_horizon_too_large_raises() -> None:
    with pytest.raises(ValueError, match="insufficient for horizon"):
        f.build_training_matrix(
            [1.0, 2.0, 3.0], {"x": [1.0, 2.0, 3.0]}, horizon=10,
            mode="direction", min_samples=None,
        )


def test_zero_horizon_raises() -> None:
    with pytest.raises(ValueError, match="horizon must be positive"):
        f.build_training_matrix(
            [1.0, 2.0, 3.0], {"x": [1.0, 2.0, 3.0]}, horizon=0,
            mode="direction", min_samples=None,
        )


def test_no_features_raises() -> None:
    with pytest.raises(ValueError, match="at least one feature"):
        f.build_training_matrix(
            [1.0, 2.0, 3.0], {}, horizon=1, mode="direction", min_samples=None,
        )


def test_feature_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length 2.*target has length 4"):
        f.build_training_matrix(
            [1.0, 2.0, 3.0, 4.0], {"x": [1.0, 2.0]}, horizon=1,
            mode="direction", min_samples=None,
        )


def test_min_samples_enforced_after_pruning() -> None:
    target = [10.0, 11.0, 12.0, 13.0]
    feats = {"x": [0.0, 0.0, 1.0, 2.0]}  # 2 rows survive pruning
    with pytest.raises(ValueError, match="min_samples=5 requires more"):
        f.build_training_matrix(
            target, feats, horizon=1, mode="direction", min_samples=5,
        )


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unsupported mode"):
        f.build_training_matrix(
            [1.0, 2.0, 3.0], {"x": [1.0, 2.0, 3.0]}, horizon=1,
            mode="bogus", min_samples=None,
        )


# ─── build_forecast_matrix ──────────────────────────────────────────────────


def test_forecast_matrix_uses_last_row_per_feature() -> None:
    feats = {"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]}
    X = f.build_forecast_matrix(feats, ["a", "b"])
    assert X.shape == (1, 2)
    assert X.tolist() == [[3.0, 30.0]]


def test_forecast_matrix_preserves_trained_feature_order() -> None:
    feats = {"a": [1.0], "b": [2.0], "c": [3.0]}
    X = f.build_forecast_matrix(feats, ["c", "a", "b"])
    assert X.tolist() == [[3.0, 1.0, 2.0]]


def test_forecast_matrix_missing_feature_raises() -> None:
    with pytest.raises(ValueError, match="missing names"):
        f.build_forecast_matrix({"a": [1.0]}, ["a", "b"])


def test_forecast_matrix_empty_features_raises() -> None:
    with pytest.raises(ValueError, match="features must not be empty"):
        f.build_forecast_matrix({}, ["a"])


def test_forecast_matrix_nan_inf_become_zero() -> None:
    feats = {"a": [1.0, float("nan")], "b": [2.0, float("inf")]}
    X = f.build_forecast_matrix(feats, ["a", "b"])
    assert X.tolist() == [[0.0, 0.0]]


def test_forecast_matrix_empty_series_raises() -> None:
    with pytest.raises(ValueError, match="feature 'a' is empty"):
        f.build_forecast_matrix({"a": []}, ["a"])


# ─── sample_weight propagation ─────────────────────────────────────────────


def test_sample_weight_returned_aligned_with_X() -> None:
    target = [10.0, 11.0, 10.5, 12.0, 11.5]
    feats = {"x": [1.0, 2.0, 3.0, 4.0, 5.0]}
    weights = [0.1, 0.2, 0.3, 0.4, 0.5]
    X, _, _, sw = f.build_training_matrix(
        target, feats, horizon=1, mode="direction", min_samples=None,
        sample_weight=weights,
    )
    assert sw is not None
    assert len(sw) == X.shape[0]
    # Rows 0..3 are kept; weights 0..3 land in sw.
    assert sw.tolist() == [0.1, 0.2, 0.3, 0.4]


def test_sample_weight_pruned_alongside_zero_rows() -> None:
    target = [10.0, 11.0, 12.0, 13.0, 14.0]
    feats = {"x": [0.0, 0.0, 1.0, 2.0, 3.0]}  # rows 0,1 dropped
    weights = [9.0, 9.0, 1.0, 2.0, 3.0]
    X, _, _, sw = f.build_training_matrix(
        target, feats, horizon=1, mode="direction", min_samples=None,
        sample_weight=weights,
    )
    assert sw is not None
    assert len(sw) == X.shape[0] == 2
    # The two preserved rows (2,3) → weights 1.0 and 2.0.
    assert sw.tolist() == [1.0, 2.0]


def test_sample_weight_none_returns_none() -> None:
    target = [10.0, 11.0, 12.0, 13.0]
    feats = {"x": [1.0, 2.0, 3.0, 4.0]}
    _, _, _, sw = f.build_training_matrix(
        target, feats, horizon=1, mode="direction", min_samples=None,
    )
    assert sw is None


def test_sample_weight_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="sample_weight has length"):
        f.build_training_matrix(
            [1.0, 2.0, 3.0, 4.0], {"x": [1.0, 2.0, 3.0, 4.0]},
            horizon=1, mode="direction", min_samples=None,
            sample_weight=[1.0, 2.0],  # wrong length
        )
