"""POST /v1/forecast — the only forecast endpoint.

Pydantic models speak camelCase on the wire (alias_generator=to_camel) and
snake_case internally.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from pydantic.alias_generators import to_camel

from .. import config, dispatch
from ..auth import check_bearer

log = logging.getLogger("predictalot.routers.forecast")

router = APIRouter(prefix="/v1", tags=["forecast"])


class ForecastConfig(BaseModel):
    horizon: int = Field(..., gt=0, description="Steps into the future to forecast.")
    quantile_levels: list[float] | None = Field(
        default=None,
        description="Quantile cuts to return. Subset of {0.1..0.9} step 0.1.",
    )
    context_length: int | None = Field(
        default=None, gt=0, description="Max history points to feed the model."
    )

    model_config = {"alias_generator": to_camel, "populate_by_name": True}


class ForecastRequest(BaseModel):
    model: str
    context: list[list[float]] = Field(..., description="One inner list per series.")
    config: ForecastConfig
    unload: bool = False

    model_config = {
        "alias_generator": to_camel,
        "populate_by_name": True,
        "protected_namespaces": (),  # allow field name "model"
    }


class ForecastResponse(BaseModel):
    model: str
    horizon: int
    quantile_levels: list[float]
    median: list[list[float]]
    quantiles: dict[str, list[list[float]]]

    model_config = {
        "alias_generator": to_camel,
        "populate_by_name": True,
        "protected_namespaces": (),
    }


class EnsembleRequest(BaseModel):
    """Same as ForecastRequest but no `model` field — runs many models at once.

    `weights` is optional; omitting it = uniform weight on every supported model.
    Weights are normalized internally (any positive numbers work). A weight of
    0 skips the model entirely (not called). Unknown model slugs → 400.
    """

    context: list[list[float]] = Field(..., description="One inner list per series.")
    config: ForecastConfig
    weights: dict[str, float] | None = None
    unload: bool = False

    model_config = {
        "alias_generator": to_camel,
        "populate_by_name": True,
        "protected_namespaces": (),
    }


class IndividualForecast(BaseModel):
    """One contributing model's forecast embedded in an EnsembleResponse."""

    model: str
    horizon: int
    quantile_levels: list[float]
    median: list[list[float]]
    quantiles: dict[str, list[list[float]]]
    weight: float

    model_config = {
        "alias_generator": to_camel,
        "populate_by_name": True,
        "protected_namespaces": (),
    }


class EnsembleResponse(BaseModel):
    model: str  # always "ensemble"
    horizon: int
    quantile_levels: list[float]
    median: list[list[float]]
    quantiles: dict[str, list[list[float]]]
    ensemble_members: list[str]
    weights: dict[str, float]  # normalized; mirrors the `weight` in each individual
    individual: dict[str, IndividualForecast]

    model_config = {
        "alias_generator": to_camel,
        "populate_by_name": True,
        "protected_namespaces": (),
    }


@router.post(
    "/forecast/ensemble",
    response_model=EnsembleResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_forecast_ensemble(body: EnsembleRequest) -> dict[str, Any]:
    """Run all three forecasters in parallel; return the element-wise mean
    median + averaged quantiles. Same wire shape as /v1/forecast plus an
    `ensembleMembers` list. Failure of any one model fails the whole call."""
    try:
        return await dispatch.forecast_ensemble(
            context=body.context,
            horizon=body.config.horizon,
            quantile_levels=body.config.quantile_levels,
            context_length=body.config.context_length,
            weights=body.weights,
            unload_after=body.unload,
        )
    except (dispatch.BadQuantileLevelsError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        log.exception("ensemble forecast failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/forecast",
    response_model=ForecastResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_forecast(body: ForecastRequest) -> dict[str, Any]:
    if body.model not in config.MODEL_SLUGS:
        raise HTTPException(
            status_code=404,
            detail=f"unknown model {body.model!r}; valid: {list(config.MODEL_SLUGS)}",
        )
    try:
        return await dispatch.forecast(
            model=body.model,
            context=body.context,
            horizon=body.config.horizon,
            quantile_levels=body.config.quantile_levels,
            context_length=body.config.context_length,
            unload_after=body.unload,
        )
    except dispatch.UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (dispatch.BadQuantileLevelsError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Snapshot download / inference failure.
        log.exception("forecast failed for model=%s", body.model)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
