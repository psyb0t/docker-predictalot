"""Pydantic request/response models for every forecast type.

camelCase on the wire (alias_generator=to_camel), snake_case in Python.

Naming convention:
  <Type>Request          — POST /v1/<type>/forecast body
  <Type>EnsembleRequest  — POST /v1/<type>/forecast/ensemble body
  <Type>Response         — single-model response
  <Type>EnsembleResponse — ensemble response (wraps per-member individual results)

Quantile-based types use the same ForecastConfig + median/quantiles output shape;
the only thing that changes is the dimensionality of `context` and `median` /
`quantiles`. Covariate types add named-covariate dicts. Samples type returns
raw sample paths instead of quantiles.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

# Shared model_config for every wire schema (camelCase aliases, accept both forms,
# allow the field name "model").
_WIRE_CFG = ConfigDict(
    alias_generator=to_camel,
    populate_by_name=True,
    protected_namespaces=(),
)


# ─── shared config blocks ─────────────────────────────────────────────────────


class ForecastConfig(BaseModel):
    """Quantile-forecast config: horizon + which quantile cuts to return.

    ``extra`` is a per-backend escape hatch: each FM backend
    documents the keys it consumes from this dict next to its
    ``SLUG``. Unknown keys are silently ignored. Examples:
      * chronos-2: batchSize, crossLearning, limitPredictionLength
      * timesfm-2.5: fixQuantileCrossing, normalizeInputs,
        perCoreBatchSize, useContinuousQuantileHead,
        inferIsPositive, forceFlipInvariance, windowSize,
        returnBackcast
      * moirai-2: moduleKwargs
      * toto-1: samplesPerBatch, useKvCache, numSamples
    """

    horizon: int = Field(..., gt=0, description="Steps into the future to forecast.")
    quantile_levels: list[float] | None = Field(
        default=None,
        description="Quantile cuts to return. Subset of {0.1..0.9} step 0.1.",
    )
    context_length: int | None = Field(
        default=None, gt=0, description="Max history points fed to the model."
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Backend-specific kwargs forwarded to the underlying "
            "predict call. Each backend documents the keys it reads; "
            "unknown keys are silently ignored. Camel-cased on the "
            "wire — backends remap to their library's expected snake "
            "naming internally."
        ),
    )
    model_config = _WIRE_CFG


class SamplesForecastConfig(BaseModel):
    """Samples-forecast config: horizon + sample count (no quantiles).

    ``extra`` follows the same per-backend convention as
    ForecastConfig.extra. Samples backends (toto-1, sundial-base-128m)
    typically read samplesPerBatch, batchSize, useKvCache from here.
    """

    horizon: int = Field(..., gt=0, description="Steps into the future to forecast.")
    num_samples: int | None = Field(
        default=None,
        gt=0,
        description="Independent sample paths to draw. None = backend default.",
    )
    context_length: int | None = Field(
        default=None, gt=0, description="Max history points fed to the model."
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Backend-specific kwargs forwarded to the predict call.",
    )
    model_config = _WIRE_CFG


# ─── univariate ───────────────────────────────────────────────────────────────


class UnivariateRequest(BaseModel):
    model: str
    context: list[list[float]] = Field(
        ..., description="One inner list of floats per series. Shape: [series][time]."
    )
    config: ForecastConfig
    unload: bool = False
    model_config = _WIRE_CFG


class UnivariateEnsembleRequest(BaseModel):
    context: list[list[float]] = Field(..., description="Shape: [series][time].")
    config: ForecastConfig
    weights: dict[str, float] | None = Field(
        default=None,
        description=(
            "Per-model weight map for the ensemble. None = uniform. Unknown slugs "
            "→ 400. Weight 0 skips a model entirely."
        ),
    )
    member_overrides: dict[str, dict] | None = Field(
        default=None,
        description=(
            "Per-member kwargs overrides. Map slug → partial "
            "ForecastConfig (or SamplesForecastConfig) dict. Each "
            "key listed there overrides the global ``config`` value "
            "for that specific member only. Use to give different "
            "members different contextLength, extra knobs, etc. in "
            "a single ensemble call."
        ),
    )
    unload: bool = False
    model_config = _WIRE_CFG


class UnivariateResponse(BaseModel):
    model: str
    horizon: int
    quantile_levels: list[float]
    median: list[list[float]] = Field(..., description="Shape: [series][time].")
    quantiles: dict[str, list[list[float]]] = Field(
        ..., description="quantileLevel → [series][time]."
    )
    model_config = _WIRE_CFG


class UnivariateIndividual(UnivariateResponse):
    """Single member's full result inside an ensemble response."""

    weight: float


class UnivariateEnsembleResponse(BaseModel):
    model: str  # always "ensemble"
    horizon: int
    quantile_levels: list[float]
    median: list[list[float]]
    quantiles: dict[str, list[list[float]]]
    ensemble_members: list[str]
    weights: dict[str, float]
    individual: dict[str, UnivariateIndividual]
    model_config = _WIRE_CFG


# ─── multivariate ─────────────────────────────────────────────────────────────


class MultivariateRequest(BaseModel):
    model: str
    context: list[list[list[float]]] = Field(
        ...,
        description=(
            "Shape: [series][channel][time]. Every series must have the same "
            "channel count; channels may vary in count between series only if "
            "the chosen backend supports it (currently none — keep uniform)."
        ),
    )
    config: ForecastConfig
    unload: bool = False
    model_config = _WIRE_CFG


class MultivariateEnsembleRequest(BaseModel):
    context: list[list[list[float]]] = Field(..., description="Shape: [series][channel][time].")
    config: ForecastConfig
    weights: dict[str, float] | None = None
    member_overrides: dict[str, dict] | None = Field(
        default=None,
        description=(
            "Per-member kwargs overrides. Map slug → partial "
            "ForecastConfig (or SamplesForecastConfig) dict. Each "
            "key listed there overrides the global ``config`` value "
            "for that specific member only. Use to give different "
            "members different contextLength, extra knobs, etc. in "
            "a single ensemble call."
        ),
    )
    unload: bool = False
    model_config = _WIRE_CFG


class MultivariateResponse(BaseModel):
    model: str
    horizon: int
    quantile_levels: list[float]
    median: list[list[list[float]]] = Field(..., description="Shape: [series][channel][time].")
    quantiles: dict[str, list[list[list[float]]]] = Field(
        ..., description="quantileLevel → [series][channel][time]."
    )
    model_config = _WIRE_CFG


class MultivariateIndividual(MultivariateResponse):
    weight: float


class MultivariateEnsembleResponse(BaseModel):
    model: str
    horizon: int
    quantile_levels: list[float]
    median: list[list[list[float]]]
    quantiles: dict[str, list[list[list[float]]]]
    ensemble_members: list[str]
    weights: dict[str, float]
    individual: dict[str, MultivariateIndividual]
    model_config = _WIRE_CFG


# ─── covariates: past only ────────────────────────────────────────────────────


class CovariatesPastRequest(BaseModel):
    model: str
    context: list[list[float]] = Field(
        ..., description="Univariate target. Shape: [series][time]."
    )
    past_covariates: list[dict[str, list[float]]] = Field(
        ...,
        description=(
            "Per series: covariateName → 1D float list, same length as the matching "
            "context series. Every series must have the same covariate names."
        ),
    )
    config: ForecastConfig
    unload: bool = False
    model_config = _WIRE_CFG


class CovariatesPastEnsembleRequest(BaseModel):
    context: list[list[float]]
    past_covariates: list[dict[str, list[float]]]
    config: ForecastConfig
    weights: dict[str, float] | None = None
    member_overrides: dict[str, dict] | None = Field(
        default=None,
        description=(
            "Per-member kwargs overrides. Map slug → partial "
            "ForecastConfig (or SamplesForecastConfig) dict. Each "
            "key listed there overrides the global ``config`` value "
            "for that specific member only. Use to give different "
            "members different contextLength, extra knobs, etc. in "
            "a single ensemble call."
        ),
    )
    unload: bool = False
    model_config = _WIRE_CFG


CovariatesPastResponse = UnivariateResponse
CovariatesPastIndividual = UnivariateIndividual
CovariatesPastEnsembleResponse = UnivariateEnsembleResponse


# ─── covariates: future only ──────────────────────────────────────────────────


class CovariatesFutureRequest(BaseModel):
    model: str
    context: list[list[float]] = Field(..., description="Shape: [series][time].")
    future_covariates: list[dict[str, list[float]]] = Field(
        ...,
        description=(
            "Per series: covariateName → 1D float list of length=horizon. "
            "Every series must have the same covariate names."
        ),
    )
    config: ForecastConfig
    unload: bool = False
    model_config = _WIRE_CFG


class CovariatesFutureEnsembleRequest(BaseModel):
    context: list[list[float]]
    future_covariates: list[dict[str, list[float]]]
    config: ForecastConfig
    weights: dict[str, float] | None = None
    member_overrides: dict[str, dict] | None = Field(
        default=None,
        description=(
            "Per-member kwargs overrides. Map slug → partial "
            "ForecastConfig (or SamplesForecastConfig) dict. Each "
            "key listed there overrides the global ``config`` value "
            "for that specific member only. Use to give different "
            "members different contextLength, extra knobs, etc. in "
            "a single ensemble call."
        ),
    )
    unload: bool = False
    model_config = _WIRE_CFG


CovariatesFutureResponse = UnivariateResponse
CovariatesFutureIndividual = UnivariateIndividual
CovariatesFutureEnsembleResponse = UnivariateEnsembleResponse


# ─── covariates: past + future ────────────────────────────────────────────────


class CovariatesRequest(BaseModel):
    model: str
    context: list[list[float]]
    past_covariates: list[dict[str, list[float]]] = Field(
        ..., description="See past-only type. Same-length-as-context per name."
    )
    future_covariates: list[dict[str, list[float]]] = Field(
        ...,
        description=(
            "Per series: future values for the covariates known into the future. "
            "Every future-covariate name MUST also appear in past_covariates. "
            "Length per series = horizon."
        ),
    )
    config: ForecastConfig
    unload: bool = False
    model_config = _WIRE_CFG


class CovariatesEnsembleRequest(BaseModel):
    context: list[list[float]]
    past_covariates: list[dict[str, list[float]]]
    future_covariates: list[dict[str, list[float]]]
    config: ForecastConfig
    weights: dict[str, float] | None = None
    member_overrides: dict[str, dict] | None = Field(
        default=None,
        description=(
            "Per-member kwargs overrides. Map slug → partial "
            "ForecastConfig (or SamplesForecastConfig) dict. Each "
            "key listed there overrides the global ``config`` value "
            "for that specific member only. Use to give different "
            "members different contextLength, extra knobs, etc. in "
            "a single ensemble call."
        ),
    )
    unload: bool = False
    model_config = _WIRE_CFG


CovariatesResponse = UnivariateResponse
CovariatesIndividual = UnivariateIndividual
CovariatesEnsembleResponse = UnivariateEnsembleResponse


# ─── samples ──────────────────────────────────────────────────────────────────


class SamplesRequest(BaseModel):
    model: str
    context: list[list[float]] = Field(..., description="Univariate target. Shape: [series][time].")
    config: SamplesForecastConfig
    unload: bool = False
    model_config = _WIRE_CFG


class SamplesEnsembleRequest(BaseModel):
    context: list[list[float]]
    config: SamplesForecastConfig
    weights: dict[str, float] | None = Field(
        default=None,
        description=(
            "Per-model weight map. For samples, weight controls how many sample "
            "paths each model contributes (relative). Weight 0 skips a model."
        ),
    )
    member_overrides: dict[str, dict] | None = Field(
        default=None,
        description=(
            "Per-member kwargs overrides. Map slug → partial "
            "ForecastConfig (or SamplesForecastConfig) dict. Each "
            "key listed there overrides the global ``config`` value "
            "for that specific member only. Use to give different "
            "members different contextLength, extra knobs, etc. in "
            "a single ensemble call."
        ),
    )
    unload: bool = False
    model_config = _WIRE_CFG


class SamplesResponse(BaseModel):
    model: str
    horizon: int
    num_samples: int = Field(..., description="Actual number of sample paths in `samples`.")
    samples: list[list[list[float]]] = Field(
        ..., description="Shape: [series][sample][time]."
    )
    median: list[list[float]] = Field(
        ..., description="Convenience: median across samples. Shape: [series][time]."
    )
    model_config = _WIRE_CFG


class SamplesIndividual(SamplesResponse):
    weight: float


class SamplesEnsembleResponse(BaseModel):
    model: str
    horizon: int
    num_samples: int = Field(
        ..., description="Total sample paths across all included members."
    )
    samples: list[list[list[float]]] = Field(
        ...,
        description=(
            "Union of every member's samples, concatenated along the sample axis. "
            "Shape: [series][sample][time]. Order matches `ensembleMembers`."
        ),
    )
    median: list[list[float]] = Field(
        ..., description="Median across the full sample pool."
    )
    ensemble_members: list[str]
    weights: dict[str, float] = Field(
        ...,
        description=(
            "Normalized weights; for samples, each model's contributed sample "
            "count = round(weight * total_request_samples), with at least 1."
        ),
    )
    individual: dict[str, SamplesIndividual]
    model_config = _WIRE_CFG


# ─── shared list-models response ──────────────────────────────────────────────


class TypeModelInfo(BaseModel):
    slug: str
    loaded: bool
    last_used_secs_ago: float | None
    idle_timeout_secs: float
    model_config = _WIRE_CFG


class TypeModelsResponse(BaseModel):
    type: str
    models: list[TypeModelInfo]
    model_config = _WIRE_CFG
