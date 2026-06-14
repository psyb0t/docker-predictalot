"""Random Forest tabular backend.

Bagged tree ensemble — different inductive bias from boosting. If RF
matches a GBT result, the win is "ensemble of trees" rather than
"boosting specifically". Quantile mode uses sklearn's
RandomForestRegressor with per-leaf quantile aggregation.

Blob format: pickle of {"variant": ..., "model": ..., (optional "models")}.
"""

from __future__ import annotations

import io
import pickle
from typing import Any

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

from .tabular_base import (
    PredictionDict,
    TrainConfigLike,
    TrainOutput,
    extra_get,
)

SLUG = "random-forest"
DISPLAY_NAME = "Random Forest"
CATEGORY = "bagging"
SUPPORTED_MODES = frozenset({"direction", "value", "quantile"})

# extras understood (via config.extra):
#   min_samples_split   min rows to split (default 2)
#   min_samples_leaf    min rows at a leaf (default 1)
#   max_features        "sqrt" | "log2" | int | float | None (default "sqrt")
#   bootstrap           bootstrap sample rows (default True)
#   oob_score           track out-of-bag score (default False)


def _common_params(config: TrainConfigLike) -> dict[str, Any]:
    return {
        "n_estimators": config.n_estimators or 300,
        "max_depth": config.max_depth,  # None → unlimited
        "min_samples_split": int(extra_get(config, "min_samples_split", 2)),
        "min_samples_leaf": int(extra_get(config, "min_samples_leaf", 1)),
        "max_features": extra_get(config, "max_features", "sqrt"),
        "bootstrap": bool(extra_get(config, "bootstrap", True)),
        "oob_score": bool(extra_get(config, "oob_score", False)),
        "random_state": (
            config.random_state if config.random_state is not None else 0
        ),
        "n_jobs": -1,
    }


def train(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: TrainConfigLike,
    sample_weight: np.ndarray | None = None,
) -> TrainOutput:
    mode = config.mode
    fit_kw: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kw["sample_weight"] = sample_weight

    importance: dict[str, float] = {}

    if mode == "direction":
        params = _common_params(config)
        if config.class_weight is not None:
            params["class_weight"] = config.class_weight
        clf = RandomForestClassifier(**params)
        clf.fit(X, y, **fit_kw)
        importance = _importance(clf, feature_names)
        payload = {"variant": "direction", "model": clf}
    elif mode == "value":
        reg = RandomForestRegressor(**_common_params(config))
        reg.fit(X, y, **fit_kw)
        importance = _importance(reg, feature_names)
        payload = {"variant": "value", "model": reg}
    elif mode == "quantile":
        if not config.quantile_levels:
            raise ValueError("quantile mode requires quantile_levels")
        # Train ONE regressor; predict per-tree at inference to derive
        # quantile bands. This avoids fitting one forest per quantile.
        reg = RandomForestRegressor(**_common_params(config))
        reg.fit(X, y, **fit_kw)
        importance = _importance(reg, feature_names)
        payload = {
            "variant": "quantile",
            "model": reg,
            "quantile_levels": list(config.quantile_levels),
        }
    else:
        raise ValueError(f"unsupported mode {mode!r}")

    buf = io.BytesIO()
    pickle.dump(payload, buf)
    return {"blob": buf.getvalue(), "importance": importance}


def _importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    arr = np.asarray(model.feature_importances_, dtype=np.float64)
    total = float(arr.sum()) or 1.0
    return {fn: float(v) / total for fn, v in zip(feature_names, arr)}


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

    if mode == "direction":
        proba = payload["model"].predict_proba(X)
        prob_up = np.asarray(proba[:, 1], dtype=np.float64)
        return {"prob_up": prob_up}

    if mode == "value":
        pred = np.asarray(payload["model"].predict(X), dtype=np.float64)
        return {"predicted": pred}

    if mode == "quantile":
        # Per-tree predictions form the empirical distribution per row;
        # take np.quantile across the trees axis to get the band.
        reg: RandomForestRegressor = payload["model"]
        per_tree = np.stack(
            [tree.predict(X) for tree in reg.estimators_], axis=0,
        )  # [n_trees, n_rows]
        levels = (
            quantile_levels
            if quantile_levels is not None
            else payload.get("quantile_levels") or [0.1, 0.5, 0.9]
        )
        out: dict[str, np.ndarray] = {}
        median: np.ndarray | None = None
        for q in levels:
            arr = np.asarray(
                np.quantile(per_tree, float(q), axis=0), dtype=np.float64,
            )
            out[f"{q:.1f}"] = arr
            if abs(q - 0.5) < 1e-9:
                median = arr
        if median is None:
            median = np.asarray(per_tree.mean(axis=0), dtype=np.float64)
        return {"median": median, "quantiles": out}

    raise ValueError(f"unsupported mode {mode!r}")
