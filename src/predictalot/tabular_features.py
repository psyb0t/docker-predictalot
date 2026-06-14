"""Feature-matrix construction for tabular train + forecast.

Train input is two parallel lists per series:
  * target[t]            length T
  * features[name][t]    length T (must match target length per name)

We build a (rows × cols) matrix where each row is one anchor t and the
columns are the features at that anchor. Labels are derived from the
target series + horizon h:

  direction[t] = 1 if target[t+h] > target[t] else 0
  value[t]     = target[t+h]
  quantile[t]  = target[t+h]   (same as value; quantile changes the loss)

Rows where label_t isn't available (t > T - h - 1) get pruned at the
end. So do rows where every feature value is the warmup-zero default
(after the alignment in finpred).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def build_training_matrix(
    target: Sequence[float],
    feature_channels: dict[str, Sequence[float]],
    horizon: int,
    mode: str,
    min_samples: int | None,
    sample_weight: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    """Return ``(X, y, feature_names, sample_weight)``.

    sample_weight is pruned alongside X/y when warmup-zero rows are
    dropped, so callers receive a weight array that aligns 1:1 with
    the returned X. None on input → None on output.

    Raises ValueError on shape mismatches or insufficient rows.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if not feature_channels:
        raise ValueError("at least one feature channel required")

    n = len(target)
    feature_names = sorted(feature_channels.keys())
    for name in feature_names:
        if len(feature_channels[name]) != n:
            raise ValueError(
                f"feature {name!r} has length {len(feature_channels[name])}, "
                f"target has length {n}; must match"
            )
    if sample_weight is not None and len(sample_weight) != n:
        raise ValueError(
            f"sample_weight has length {len(sample_weight)}, target has "
            f"length {n}; must match"
        )

    # Build labels first so we know how many rows are usable.
    target_arr = np.asarray(target, dtype=np.float64)

    # Compute labels for t in [0, n - horizon - 1]
    last_valid = n - horizon
    if last_valid <= 0:
        raise ValueError(
            f"target length {n} insufficient for horizon {horizon}; need > horizon"
        )

    if mode == "direction":
        y = (target_arr[horizon:] > target_arr[: last_valid]).astype(np.int64)
    elif mode in ("value", "quantile"):
        y = target_arr[horizon:].astype(np.float64)
    else:
        raise ValueError(f"unsupported mode {mode!r}")

    # Build X from feature channels at rows [0, last_valid)
    cols: list[np.ndarray] = []
    for name in feature_names:
        col = np.asarray(feature_channels[name], dtype=np.float64)[:last_valid]
        cols.append(col)
    X = np.column_stack(cols)

    sw_arr: np.ndarray | None = None
    if sample_weight is not None:
        sw_arr = np.asarray(sample_weight, dtype=np.float64)[:last_valid]

    # Sanity: prune rows with all-zero features (warmup fill) when we have
    # enough data. Tabular models can't learn from a row of zeros.
    nonzero_mask = (X != 0.0).any(axis=1)
    X = X[nonzero_mask]
    y = y[nonzero_mask]
    if sw_arr is not None:
        sw_arr = sw_arr[nonzero_mask]

    if min_samples is not None and X.shape[0] < min_samples:
        raise ValueError(
            f"after pruning {X.shape[0]} rows remain; min_samples={min_samples} requires more"
        )

    return X, y, feature_names, sw_arr


def build_forecast_matrix(
    feature_channels: dict[str, Sequence[float]],
    feature_names: list[str],
) -> np.ndarray:
    """Return a single-row (1 × n_features) matrix from the LAST bar of
    each named feature series. Feature names must match the names from
    the matching train call exactly.
    """
    if not feature_channels:
        raise ValueError("features must not be empty")
    missing = [fn for fn in feature_names if fn not in feature_channels]
    if missing:
        raise ValueError(
            f"forecast features missing names: {missing}; trained on {feature_names}"
        )
    row: list[float] = []
    for fn in feature_names:
        series = feature_channels[fn]
        if not series:
            raise ValueError(f"feature {fn!r} is empty")
        v = float(series[-1])
        if math.isnan(v) or math.isinf(v):
            v = 0.0
        row.append(v)
    return np.asarray([row], dtype=np.float64)
