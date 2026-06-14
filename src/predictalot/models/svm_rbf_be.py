"""SVM with RBF kernel — kernel-based tabular backend.

Different decision boundary shape than trees or linear. Slow on
big datasets (O(n²)+ scaling) but our train_window is ~1500 so it
fits fine.

Quantile mode falls back to the residual-quantile trick used by
mlp_be — sklearn's SVR doesn't natively emit quantiles.
"""

from __future__ import annotations

import io
import pickle
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from .tabular_base import (
    PredictionDict,
    TrainConfigLike,
    TrainOutput,
    extra_get,
)

SLUG = "svm-rbf"
DISPLAY_NAME = "SVM with RBF kernel"
CATEGORY = "kernel"
SUPPORTED_MODES = frozenset({"direction", "value", "quantile"})

# extras understood (via config.extra):
#   C            regularization strength (default 1.0)
#   gamma        "scale" | "auto" | float (default "scale")
#   kernel       (default "rbf") — but slug name pins us to rbf
#   tol          (default 1e-3)
#   max_iter     (default -1 = unlimited)


def _common_params(config: TrainConfigLike) -> dict[str, Any]:
    return {
        "C": float(extra_get(config, "C", 1.0)),
        "gamma": extra_get(config, "gamma", "scale"),
        "kernel": "rbf",
        "tol": float(extra_get(config, "tol", 1e-3)),
        "max_iter": int(extra_get(config, "max_iter", -1)),
    }


def train(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: TrainConfigLike,
    sample_weight: np.ndarray | None = None,
) -> TrainOutput:
    mode = config.mode
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    fit_kw: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kw["sample_weight"] = sample_weight

    n = max(len(feature_names), 1)
    uniform = {fn: 1.0 / n for fn in feature_names}  # SVM has no native importance

    if mode == "direction":
        params = _common_params(config)
        params["probability"] = True
        if config.class_weight is not None:
            params["class_weight"] = config.class_weight
        if config.random_state is not None:
            params["random_state"] = config.random_state
        clf = SVC(**params)
        clf.fit(Xs, y, **fit_kw)
        payload = {"variant": "direction", "model": clf, "scaler": scaler}
        importance = uniform
    elif mode == "value":
        reg = SVR(**_common_params(config))
        reg.fit(Xs, y, **fit_kw)
        payload = {"variant": "value", "model": reg, "scaler": scaler}
        importance = uniform
    elif mode == "quantile":
        if not config.quantile_levels:
            raise ValueError("quantile mode requires quantile_levels")
        base = SVR(**_common_params(config))
        base.fit(Xs, y, **fit_kw)
        pred_train = base.predict(Xs)
        residuals = np.asarray(y, dtype=np.float64) - pred_train
        payload = {
            "variant": "quantile",
            "model": base,
            "scaler": scaler,
            "residuals": residuals,
            "quantile_levels": list(config.quantile_levels),
        }
        importance = uniform
    else:
        raise ValueError(f"unsupported mode {mode!r}")

    buf = io.BytesIO()
    pickle.dump(payload, buf)
    return {"blob": buf.getvalue(), "importance": importance}


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
        base_pred = np.asarray(payload["model"].predict(Xs), dtype=np.float64)
        residuals: np.ndarray = payload["residuals"]
        levels = (
            quantile_levels
            if quantile_levels is not None
            else payload.get("quantile_levels") or [0.1, 0.5, 0.9]
        )
        out: dict[str, np.ndarray] = {}
        median: np.ndarray | None = None
        for q in levels:
            shift = float(np.quantile(residuals, float(q)))
            arr = base_pred + shift
            out[f"{q:.1f}"] = arr
            if abs(q - 0.5) < 1e-9:
                median = arr
        if median is None:
            median = base_pred
        return {"median": median, "quantiles": out}
    raise ValueError(f"unsupported mode {mode!r}")
