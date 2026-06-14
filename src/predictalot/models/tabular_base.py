"""Common base + protocol for tabular backends.

Each backend module exposes module-level functions:

    SLUG: str                    — registry key
    DISPLAY_NAME: str            — human-readable name
    SUPPORTED_MODES: frozenset[str]  — subset of {direction, value, quantile}

    def train(X, y, config) -> bytes:
        Fit + serialize the trained estimator (any backend-defined format).

    def predict(blob, X, mode, quantile_levels=None) -> PredictionDict:
        Restore from `blob` and produce predictions on X.

PredictionDict is a flat dict keyed by mode:
    direction → {"prob_up": np.ndarray(n,)}
    value     → {"predicted": np.ndarray(n,)}
    quantile  → {"median": np.ndarray(n,), "quantiles": {level_str: np.ndarray(n,)}}

The router maps that to the wire schema. Backends never see the wire types.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict

import numpy as np

ModeStr = str  # "direction" | "value" | "quantile"


class TrainConfigLike(Protocol):
    """Subset of TrainConfig the backends consume.

    Tier 1 + tier 2 cross-backend knobs are typed here. Tier 3
    backend-specific knobs flow through ``extra`` (each backend
    documents the keys it reads).
    """

    mode: ModeStr
    horizon: int
    quantile_levels: list[float] | None
    # tier 2 cross-backend knobs
    n_estimators: int | None
    max_depth: int | None
    learning_rate: float | None
    num_leaves: int | None
    min_samples: int | None
    random_state: int | None
    categorical_features: list[str] | None
    monotonic_constraints: dict[str, int] | None
    class_weight: Any  # Literal["balanced"] | dict[str, float] | None
    sample_weight: list[float] | None
    early_stopping_rounds: int | None
    validation_fraction: float | None
    # tier 3 escape hatch
    extra: dict[str, Any] | None


def feature_indices_from_names(
    feature_names: list[str], names: list[str] | None,
) -> list[int] | None:
    """Map ``names`` (subset of ``feature_names``) → positional indices.

    Returns None when names is None/empty. Raises ValueError on
    unknown names.
    """
    if not names:
        return None
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    out: list[int] = []
    for n in names:
        if n not in name_to_idx:
            raise ValueError(
                f"feature name {n!r} not in trained feature set "
                f"{feature_names}"
            )
        out.append(name_to_idx[n])
    return out


def monotone_vector(
    feature_names: list[str], constraints: dict[str, int] | None,
) -> list[int] | None:
    """Build a (n_features,) list of -1/0/+1 from a name→direction dict.

    Returns None when constraints is None/empty.
    Raises ValueError on unknown feature names or invalid directions.
    """
    if not constraints:
        return None
    valid = {-1, 0, 1}
    for n, d in constraints.items():
        if n not in feature_names:
            raise ValueError(
                f"monotonic_constraints feature {n!r} not in trained set"
            )
        if d not in valid:
            raise ValueError(
                f"monotonic_constraints[{n!r}] = {d} not in {{-1, 0, 1}}"
            )
    return [constraints.get(n, 0) for n in feature_names]


def extra_get(
    config: "TrainConfigLike", key: str, default: Any = None,
) -> Any:
    """Convenience accessor: ``config.extra[key]`` with default."""
    if config.extra is None:
        return default
    return config.extra.get(key, default)


class PredictionDict(TypedDict, total=False):
    prob_up: np.ndarray
    predicted: np.ndarray
    median: np.ndarray
    quantiles: dict[str, np.ndarray]


class FeatureImportanceDict(TypedDict, total=False):
    importance: dict[str, float]


class TrainOutput(TypedDict):
    blob: bytes
    importance: dict[str, float]


class TabularBackend(Protocol):
    SLUG: str
    DISPLAY_NAME: str
    SUPPORTED_MODES: frozenset[str]

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        config: TrainConfigLike,
    ) -> TrainOutput: ...

    def predict(
        self,
        blob: bytes,
        X: np.ndarray,
        mode: ModeStr,
        quantile_levels: list[float] | None = None,
    ) -> PredictionDict: ...


def supports(backend: Any, mode: ModeStr) -> bool:
    return mode in backend.SUPPORTED_MODES
