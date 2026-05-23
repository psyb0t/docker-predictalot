"""Forecast type registry.

Each "type" is a forecasting modality (univariate, multivariate, covariate-aware,
samples-only). Types map to URL prefixes under /v1/<type>/ and gate which models
are available for each call.

Type slug naming is the single source of truth — used by:
  * URL routers (each type owns a router under /v1/<type>/)
  * Backend modules (each declares `SUPPORTED_TYPES: frozenset[str]`)
  * Ensemble dispatchers (each type has its own ensemble over its members)
  * MCP tools (one tool per (type, model) pair)
"""

from __future__ import annotations

from . import config

# Internal type slugs. These are the canonical names; URL prefixes differ.
TYPE_UNIVARIATE = "univariate"
TYPE_MULTIVARIATE = "multivariate"
TYPE_COVARIATES_PAST = "covariates-past"
TYPE_COVARIATES_FUTURE = "covariates-future"
TYPE_COVARIATES_BOTH = "covariates-both"
TYPE_SAMPLES = "samples"

TYPE_SLUGS: tuple[str, ...] = (
    TYPE_UNIVARIATE,
    TYPE_MULTIVARIATE,
    TYPE_COVARIATES_PAST,
    TYPE_COVARIATES_FUTURE,
    TYPE_COVARIATES_BOTH,
    TYPE_SAMPLES,
)

# Model membership per type — which model slugs support each modality.
# Source of truth for /v1/<type>/models and ensemble-member resolution.
TYPE_MEMBERS: dict[str, tuple[str, ...]] = {
    TYPE_UNIVARIATE: (
        "chronos-2",
        "timesfm-2.5",
        "moirai-2",
        "toto-1",
        "sundial-base-128m",
    ),
    TYPE_MULTIVARIATE: ("chronos-2", "moirai-2", "toto-1"),
    TYPE_COVARIATES_PAST: ("chronos-2", "moirai-2"),
    TYPE_COVARIATES_FUTURE: ("chronos-2",),
    TYPE_COVARIATES_BOTH: ("chronos-2",),
    TYPE_SAMPLES: ("toto-1", "sundial-base-128m"),
}

# Reverse map: model → tuple of type slugs it supports.
MODEL_TYPES: dict[str, tuple[str, ...]] = {}
for _slug in config.MODEL_SLUGS:
    MODEL_TYPES[_slug] = tuple(
        t for t in TYPE_SLUGS if _slug in TYPE_MEMBERS[t]
    )


class UnknownTypeError(ValueError):
    pass


class ModelDoesNotSupportTypeError(ValueError):
    pass


def members(type_slug: str) -> tuple[str, ...]:
    """Return the tuple of model slugs that support a given type."""
    if type_slug not in TYPE_MEMBERS:
        raise UnknownTypeError(
            f"unknown type {type_slug!r}; valid: {list(TYPE_SLUGS)}"
        )
    return TYPE_MEMBERS[type_slug]


def assert_supported(type_slug: str, model_slug: str) -> None:
    """Raise if `model_slug` does not support `type_slug`. Use at router boundary."""
    valid = members(type_slug)
    if model_slug not in valid:
        raise ModelDoesNotSupportTypeError(
            f"model {model_slug!r} does not support type {type_slug!r}; "
            f"valid for this type: {list(valid)}"
        )
