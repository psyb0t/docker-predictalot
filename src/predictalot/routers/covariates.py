"""Routes for forecasts with past and future covariates."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .. import dispatch, models, types
from ..auth import check_bearer
from ._common import build_type_models_response
from .schemas import (
    CovariatesEnsembleRequest,
    CovariatesEnsembleResponse,
    CovariatesRequest,
    CovariatesResponse,
)

log = logging.getLogger("predictalot.routers.covariates")

router = APIRouter(prefix="/v1/timeseries/covariates", tags=["covariates"])


@router.post(
    "/forecast/ensemble",
    response_model=CovariatesEnsembleResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_ensemble(body: CovariatesEnsembleRequest) -> dict[str, Any]:
    try:
        return await dispatch.ensemble_covariates(
            context=body.context,
            past_covariates=body.past_covariates,
            future_covariates=body.future_covariates,
            horizon=body.config.horizon,
            quantile_levels=body.config.quantile_levels,
            context_length=body.config.context_length,
            weights=body.weights,
            unload_after=body.unload,
            extra=body.config.extra,
            member_overrides=body.member_overrides,
        )
    except (dispatch.BadQuantileLevelsError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        log.exception("covariates ensemble failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/forecast",
    response_model=CovariatesResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_forecast(body: CovariatesRequest) -> dict[str, Any]:
    try:
        return await dispatch.dispatch_covariates(
            model=body.model,
            context=body.context,
            past_covariates=body.past_covariates,
            future_covariates=body.future_covariates,
            horizon=body.config.horizon,
            quantile_levels=body.config.quantile_levels,
            context_length=body.config.context_length,
            unload_after=body.unload,
            extra=body.config.extra,
        )
    except (dispatch.UnknownModelError, types.UnknownTypeError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except types.ModelDoesNotSupportTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (dispatch.BadQuantileLevelsError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        log.exception("covariates forecast failed for model=%s", body.model)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/models", dependencies=[Depends(check_bearer)])
def list_models() -> dict[str, Any]:
    return build_type_models_response(types.TYPE_COVARIATES_BOTH, models)
