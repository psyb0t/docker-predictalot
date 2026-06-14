"""Pydantic request/response models for /v1/tabular/*.

camelCase on the wire (alias_generator=to_camel) — matches the FM routers.
Target + features are generic float lists; no OHLC / "indicator" assumptions
are baked in. Caller engineers features however they want.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

# Shared wire config — matches the FM router pattern in routers/schemas.py.
_WIRE_CFG = ConfigDict(
    alias_generator=to_camel,
    populate_by_name=True,
    protected_namespaces=(),
)

# Modes the tabular endpoint supports.
ModeStr = Literal["direction", "value", "quantile"]


class TrainConfig(BaseModel):
    """Mode + horizon + hyperparams for one train call."""

    mode: ModeStr = Field(
        ...,
        description=(
            "What to learn: 'direction' = sign(target[t+h] - target[t]); "
            "'value' = target[t+h] regression; "
            "'quantile' = quantile regression at the requested levels."
        ),
    )
    horizon: int = Field(
        ...,
        gt=0,
        description="Bars ahead the target measures (sign or value at t+h).",
    )
    quantile_levels: list[float] | None = Field(
        default=None,
        description=(
            "Quantile cuts to train for. Required when mode='quantile'. "
            "Same constraints as the FM routes: subset of {0.1..0.9}."
        ),
    )
    # ── Tier 2: cross-backend knobs ──────────────────────────────────
    n_estimators: int | None = Field(
        default=None,
        gt=0,
        description="Number of trees / boosting rounds (GBT/RF).",
    )
    max_depth: int | None = Field(
        default=None,
        gt=0,
        description="Max tree depth (GBT/RF). None = unlimited where allowed.",
    )
    learning_rate: float | None = Field(
        default=None, gt=0, description="Learning rate (GBT)."
    )
    num_leaves: int | None = Field(
        default=None, gt=0, description="Leaves per tree (LightGBM)."
    )
    min_samples: int | None = Field(
        default=None,
        gt=0,
        description="Min training samples to keep an anchor. Skips warmup-zero rows.",
    )
    random_state: int | None = Field(
        default=None, description="Reproducibility seed; None = stochastic."
    )
    categorical_features: list[str] | None = Field(
        default=None,
        description=(
            "Feature names to mark categorical. GBTs (lightgbm/xgboost/"
            "hist-gbt) use specialized split logic for these; other "
            "backends ignore the hint. Treating a categorical as numeric "
            "is a silent footgun — name the columns here."
        ),
    )
    monotonic_constraints: dict[str, int] | None = Field(
        default=None,
        description=(
            "Per-feature monotonicity direction: -1 (decreasing), 0 "
            "(none), +1 (increasing) on the prediction. GBTs honor; "
            "other backends ignore. Use when you have domain prior "
            "knowledge (e.g. {'rsi': -1})."
        ),
    )
    class_weight: Literal["balanced"] | dict[str, float] | None = Field(
        default=None,
        description=(
            "Weighting for imbalanced classifiers. 'balanced' = inverse "
            "frequency; dict = explicit per-class weights. Classifiers "
            "honor this; regressors ignore."
        ),
    )
    sample_weight: list[float] | None = Field(
        default=None,
        description=(
            "Per-row training weight (same length as target series). "
            "Pruned alongside the target when warmup rows are dropped. "
            "Useful for time-decay or volume-weighted training."
        ),
    )
    early_stopping_rounds: int | None = Field(
        default=None,
        gt=0,
        description=(
            "GBT early-stopping patience. Requires validation_fraction>0. "
            "Ignored by non-iterative models."
        ),
    )
    validation_fraction: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        description="Fraction of training pool held out as validation set.",
    )

    # ── Tier 3: per-backend escape hatch ─────────────────────────────
    extra: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Backend-specific hyperparams that don't fit a cross-cutting "
            "shape. Each backend documents the keys it reads from here. "
            "Examples: svm-rbf reads {'C', 'gamma'}; mlp reads "
            "{'hidden_layer_sizes', 'activation'}; knn reads "
            "{'n_neighbors', 'weights', 'metric'}."
        ),
    )

    model_config = _WIRE_CFG


class TrainRequest(BaseModel):
    """Train + persist one tabular model.

    Shape note: a single 'series' is one independent training set. Most
    callers will send series=[1] (one target). The container fits one model
    per series internally — useful when you want a fleet of identical
    configs trained on different histories in one request.
    """

    model_id: str = Field(
        ...,
        description="Caller-chosen identifier. Used for storage + later forecast lookup.",
    )
    backend: str = Field(
        ...,
        description=(
            "Backend slug: 'lightgbm' | 'xgboost' | 'tabpfn' | 'logistic'. "
            "GET /v1/tabular/backends for the live list."
        ),
    )
    target: list[list[float]] = Field(
        ...,
        description="[series][time] — the scalar series whose direction/value we predict.",
    )
    features: list[dict[str, list[float]]] = Field(
        ...,
        description=(
            "Per series: featureName → time-aligned float list. "
            "Every series must have the same feature names. Lengths must "
            "match the corresponding target series."
        ),
    )
    config: TrainConfig
    overwrite: bool = Field(
        default=False,
        description=(
            "Allow overwriting an existing model_id. Default false — train calls "
            "on a known id are rejected unless overwrite=true."
        ),
    )
    model_config = _WIRE_CFG


class TrainResponse(BaseModel):
    model_id: str
    backend: str
    mode: ModeStr
    horizon: int
    n_training_rows: int = Field(
        ..., description="Rows actually used after warmup + label availability pruning."
    )
    n_features: int
    feature_names: list[str]
    feature_importance: dict[str, float] = Field(
        default_factory=dict,
        description="Backend-reported feature importance (gain-based for GBTs).",
    )
    train_secs: float
    model_config = _WIRE_CFG


class ForecastRequest(BaseModel):
    """Run a stored model on the LATEST feature snapshot.

    For multiple anchors at once, pass features series-by-series; one
    prediction per series.
    """

    model_id: str
    features: list[dict[str, list[float]]] = Field(
        ...,
        description=(
            "Per series: featureName → time-aligned float list. The LAST "
            "row is the prediction anchor; earlier rows are ignored. Feature "
            "names must match what the model was trained on."
        ),
    )
    model_config = _WIRE_CFG


class ForecastResponse(BaseModel):
    model_id: str
    backend: str
    mode: ModeStr
    horizon: int
    # direction mode
    prob_up: list[float] | None = Field(
        default=None,
        description="P(target rises at +h) per series. Populated when mode='direction'.",
    )
    confidence: list[float] | None = Field(
        default=None,
        description=(
            "|prob_up - 0.5| * 2 ∈ [0, 1] per series. Populated when mode='direction'."
        ),
    )
    # value mode
    predicted: list[float] | None = Field(
        default=None,
        description="Point prediction of target[t+h] per series. Populated when mode='value'.",
    )
    # quantile mode (shape matches FM routes for downstream compat)
    median: list[list[float]] | None = Field(
        default=None,
        description="[series][1] median per series. Populated when mode='quantile'.",
    )
    quantiles: dict[str, list[list[float]]] | None = Field(
        default=None,
        description="quantileLevel → [series][1]. Populated when mode='quantile'.",
    )
    model_config = _WIRE_CFG


class EnsembleForecastRequest(BaseModel):
    """Ensemble forecast across a list of previously-trained models.

    All members must share mode, horizon, and feature_names; mismatched
    members get rejected with a 400 (no silent coercion). The model_ids
    list defines membership; the optional weights map sets per-member
    weight. None / missing entries default to 1.0; zeros remove a
    member from the ensemble.
    """

    model_ids: list[str] = Field(
        ...,
        description="Stored model_ids to combine. Must contain at least 1.",
    )
    weights: dict[str, float] | None = Field(
        default=None,
        description=(
            "Per-member weight map. None = uniform. Unknown ids → 400. "
            "Weight 0 removes a member from the ensemble; negative → 400."
        ),
    )
    features: list[dict[str, list[float]]] = Field(
        ...,
        description=(
            "Per series: featureName → time-aligned float list. LAST row "
            "is the prediction anchor. Feature names must match what every "
            "member was trained on."
        ),
    )
    model_config = _WIRE_CFG


class EnsembleForecastResponse(BaseModel):
    mode: ModeStr
    horizon: int
    ensemble_members: list[str]
    weights: dict[str, float] = Field(
        ..., description="Normalized weights (sum to 1.0)."
    )
    individual: dict[str, dict[str, Any]] = Field(
        ..., description="member_id → that member's full ForecastResponse."
    )
    # Combined fields populated based on mode:
    prob_up: list[float] | None = None
    confidence: list[float] | None = None
    predicted: list[float] | None = None
    median: list[list[float]] | None = None
    quantiles: dict[str, list[list[float]]] | None = None
    model_config = _WIRE_CFG


class BaseMemberSpec(BaseModel):
    """One base learner spec inside a meta-train request."""

    backend: str = Field(
        ..., description="Tabular backend slug for this member.",
    )
    config: TrainConfig = Field(
        ..., description="Per-member train config. Mode/horizon must match the meta request's.",
    )
    model_config = _WIRE_CFG


# ── /v1/tabular/train/calibrated ───────────────────────────────────────
class CalibratedTrainRequest(BaseModel):
    """Train a base learner + post-hoc probability calibrator.

    Direction-only. The base learner emits prob_up; a calibrator
    (Platt / isotonic) is fit on a held-out tail of the training pool
    so the output probabilities are well-calibrated (i.e. when the
    model says 0.7, ~70% of those calls do go up).
    """

    model_id: str
    base_backend: str = Field(
        ..., description="Backend slug for the base learner.",
    )
    target: list[list[float]]
    features: list[dict[str, list[float]]]
    config: TrainConfig = Field(
        ...,
        description="Train config for the base learner. mode MUST be 'direction'.",
    )
    calibration_method: Literal["sigmoid", "isotonic"] = Field(
        default="sigmoid",
        description="Platt sigmoid (parametric) or isotonic (non-parametric).",
    )
    calibration_fraction: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of training rows held out (from the TAIL — preserving "
            "time order) to fit the calibrator on. Must be > 0 and < 1."
        ),
    )
    overwrite: bool = False
    model_config = _WIRE_CFG


# ── /v1/tabular/train/stacking ─────────────────────────────────────────
class StackingTrainRequest(BaseModel):
    """Train K base learners + a meta-learner on K-fold OOF predictions.

    Direction-mode for v1. The meta-learner sees the OOF
    prob_up of each member as its inputs.
    """

    model_id: str
    members: list[BaseMemberSpec] = Field(
        ...,
        min_length=2,
        description="Two or more base learners. All must use mode='direction'.",
    )
    meta_backend: str = Field(
        default="logistic",
        description="Backend slug for the meta-learner. Direction-mode only.",
    )
    target: list[list[float]]
    features: list[dict[str, list[float]]]
    horizon: int = Field(
        ..., gt=0, description="Horizon shared by every member + the meta-learner.",
    )
    n_folds: int = Field(
        default=5,
        ge=2,
        le=10,
        description="K-fold CV folds for generating out-of-fold member predictions.",
    )
    overwrite: bool = False
    model_config = _WIRE_CFG


# ── /v1/tabular/train/diversified ──────────────────────────────────────
class DiversifiedTrainRequest(BaseModel):
    """Train K candidate learners; SELECT a subset with low pairwise
    correlation in their out-of-fold predictions; combine equal-weight.

    Three modes supported (direction/value/quantile — must match across
    candidates). Selection optimizes for ensemble diversity: greedy
    add-from-best, skipping candidates whose OOF predictions correlate
    above ``max_pairwise_corr`` with anyone already selected.
    """

    model_id: str
    candidates: list[BaseMemberSpec] = Field(
        ...,
        min_length=2,
        description="Two or more candidate learners. Mode must match across all.",
    )
    target: list[list[float]]
    features: list[dict[str, list[float]]]
    horizon: int = Field(..., gt=0)
    mode: ModeStr = Field(..., description="Mode every candidate must run in.")
    quantile_levels: list[float] | None = Field(
        default=None,
        description="Required if mode='quantile'.",
    )
    n_folds: int = Field(
        default=3,
        ge=2,
        le=10,
        description="K-fold splits for OOF correlation estimates.",
    )
    max_pairwise_corr: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Reject a candidate whose max pairwise OOF correlation with any "
            "already-selected member is above this. Lower = more diverse "
            "ensemble. 0.85 is a reasonable default for finance signals."
        ),
    )
    min_members: int = Field(default=2, ge=1, le=20)
    max_members: int = Field(default=5, ge=1, le=20)
    overwrite: bool = False
    model_config = _WIRE_CFG


# ── Meta forecast requests ─────────────────────────────────────────────
class MetaForecastRequest(BaseModel):
    """Used by all 3 meta-forecast endpoints. They load the meta blob
    by model_id and dispatch internally based on the stored ``kind``.
    """

    model_id: str
    features: list[dict[str, list[float]]]
    model_config = _WIRE_CFG


class MetaForecastResponse(BaseModel):
    model_id: str
    kind: Literal["calibrated", "stacking", "diversified"]
    mode: ModeStr
    horizon: int
    # member breakdown (slug → individual forecast response shape)
    members: dict[str, dict[str, Any]] | None = None
    selected_members: list[str] | None = Field(
        default=None,
        description="Diversified only: which candidates survived selection.",
    )
    # direction
    prob_up: list[float] | None = None
    confidence: list[float] | None = None
    # value
    predicted: list[float] | None = None
    # quantile
    median: list[list[float]] | None = None
    quantiles: dict[str, list[list[float]]] | None = None
    model_config = _WIRE_CFG


class MetaTrainResponse(BaseModel):
    model_id: str
    kind: Literal["calibrated", "stacking", "diversified"]
    mode: ModeStr
    horizon: int
    members_used: list[str] = Field(
        ..., description="Backend slugs that ended up in the final stored model.",
    )
    n_training_rows: int
    n_features: int
    feature_names: list[str]
    train_secs: float
    # diversified only
    candidate_corr: dict[str, dict[str, float]] | None = None
    # stacking only
    oof_score: float | None = Field(
        default=None,
        description=(
            "Out-of-fold meta-learner score: AUC for direction mode."
        ),
    )
    model_config = _WIRE_CFG


class TabularModelInfo(BaseModel):
    model_id: str
    backend: str
    mode: ModeStr
    horizon: int
    n_features: int
    feature_names: list[str]
    n_training_rows: int
    trained_at_unix: float
    model_config = _WIRE_CFG


class TabularModelsResponse(BaseModel):
    models: list[TabularModelInfo]
    model_config = _WIRE_CFG
