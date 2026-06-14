"""Gaussian Naive Bayes tabular backend.

Assumes feature independence given the class. If NB is competitive
with GBT/MLP, our features are independent enough that nonlinear
models aren't gaining much — diagnostic value is high.

Quantile mode is not natively supported (NB outputs probabilities, not
distributions over the target value). We return a residual-quantile
band around the value prediction the same way mlp/svm do.
"""

from __future__ import annotations

import io
import pickle
from typing import Any

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from .tabular_base import (
    PredictionDict,
    TrainConfigLike,
    TrainOutput,
    extra_get,
)

SLUG = "naive-bayes"
DISPLAY_NAME = "Gaussian Naive Bayes / BayesianRidge"
CATEGORY = "independence"
SUPPORTED_MODES = frozenset({"direction", "value", "quantile"})

# extras understood (via config.extra):
#   var_smoothing  GaussianNB smoothing (default 1e-9)
#   priors         class priors [p_neg, p_pos] (default None = empirical)


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
    uniform = {fn: 1.0 / n for fn in feature_names}

    if mode == "direction":
        clf = GaussianNB(
            var_smoothing=float(extra_get(config, "var_smoothing", 1e-9)),
            priors=extra_get(config, "priors", None),
        )
        clf.fit(Xs, y, **fit_kw)
        payload = {"variant": "direction", "model": clf, "scaler": scaler}
    elif mode == "value":
        # GaussianNB doesn't do regression — use BayesianRidge as the
        # closest probabilistic-linear analog.
        reg = BayesianRidge()
        reg.fit(Xs, y, **fit_kw)
        payload = {"variant": "value", "model": reg, "scaler": scaler}
    elif mode == "quantile":
        if not config.quantile_levels:
            raise ValueError("quantile mode requires quantile_levels")
        base = BayesianRidge()
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
