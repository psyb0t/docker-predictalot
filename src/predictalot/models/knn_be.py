"""k-Nearest Neighbors tabular backend.

Distance-based baseline. If k-NN matches GBT, the signal is just
nearest-neighbor — bad sign for generalization. If k-NN tanks, GBT
is learning genuine structure.

Quantile mode uses per-neighbor outputs to form the empirical
distribution at each row.
"""

from __future__ import annotations

import io
import pickle
from typing import Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from .tabular_base import (
    PredictionDict,
    TrainConfigLike,
    TrainOutput,
    extra_get,
)

SLUG = "knn"
DISPLAY_NAME = "k-Nearest Neighbors"
CATEGORY = "distance"
SUPPORTED_MODES = frozenset({"direction", "value", "quantile"})

# extras understood (via config.extra):
#   n_neighbors  (default 5)
#   weights      "uniform" | "distance" (default "distance")
#   metric       "euclidean" | "manhattan" | "minkowski" (default "minkowski")
#   p            Minkowski exponent (default 2 = Euclidean)


def _common_params(config: TrainConfigLike) -> dict[str, Any]:
    return {
        "n_neighbors": int(extra_get(config, "n_neighbors", 5)),
        "weights": extra_get(config, "weights", "distance"),
        "metric": extra_get(config, "metric", "minkowski"),
        "p": int(extra_get(config, "p", 2)),
        "n_jobs": -1,
    }


def train(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: TrainConfigLike,
    sample_weight: np.ndarray | None = None,  # noqa: ARG001 — knn fit ignores
) -> TrainOutput:
    mode = config.mode
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n = max(len(feature_names), 1)
    uniform = {fn: 1.0 / n for fn in feature_names}  # k-NN has no native importance

    if mode == "direction":
        clf = KNeighborsClassifier(**_common_params(config))
        clf.fit(Xs, y)
        payload = {"variant": "direction", "model": clf, "scaler": scaler}
    elif mode == "value":
        reg = KNeighborsRegressor(**_common_params(config))
        reg.fit(Xs, y)
        payload = {"variant": "value", "model": reg, "scaler": scaler}
    elif mode == "quantile":
        if not config.quantile_levels:
            raise ValueError("quantile mode requires quantile_levels")
        reg = KNeighborsRegressor(**_common_params(config))
        reg.fit(Xs, y)
        payload = {
            "variant": "quantile",
            "model": reg,
            "scaler": scaler,
            "y_train": np.asarray(y, dtype=np.float64),
            "quantile_levels": list(config.quantile_levels),
        }
    else:
        raise ValueError(f"unsupported mode {mode!r}")

    buf = io.BytesIO()
    pickle.dump(payload, buf)
    return {"blob": buf.getvalue(), "importance": uniform}


def predict(
    blob: bytes,
    X: np.ndarray,
    mode: str,
    quantile_levels: list[float] | None = None,
) -> PredictionDict:
    payload = pickle.loads(blob)
    variant = payload["variant"]
    if variant != mode:
        raise ValueError(
            f"model was trained for mode={variant!r}, forecast requested mode={mode!r}"
        )
    scaler: StandardScaler = payload["scaler"]
    Xs = scaler.transform(X)

    if mode == "direction":
        proba = payload["model"].predict_proba(Xs)
        prob_up = np.asarray(proba[:, 1], dtype=np.float64)
        return {"prob_up": prob_up}
    if mode == "value":
        pred = np.asarray(payload["model"].predict(Xs), dtype=np.float64)
        return {"predicted": pred}
    if mode == "quantile":
        # k neighbor labels give us the empirical distribution at each
        # row; take per-row quantiles of those labels.
        reg: KNeighborsRegressor = payload["model"]
        y_train: np.ndarray = payload["y_train"]
        # KNeighbors returns (distances, indices); we only need indices.
        _, idx = reg.kneighbors(Xs)
        neighbor_labels = y_train[idx]  # [n_rows, k]
        levels = (
            quantile_levels
            if quantile_levels is not None
            else payload.get("quantile_levels") or [0.1, 0.5, 0.9]
        )
        out: dict[str, np.ndarray] = {}
        median: np.ndarray | None = None
        for q in levels:
            arr = np.asarray(
                np.quantile(neighbor_labels, float(q), axis=1),
                dtype=np.float64,
            )
            out[f"{q:.1f}"] = arr
            if abs(q - 0.5) < 1e-9:
                median = arr
        if median is None:
            median = np.asarray(
                neighbor_labels.mean(axis=1), dtype=np.float64,
            )
        return {"median": median, "quantiles": out}
    raise ValueError(f"unsupported mode {mode!r}")
