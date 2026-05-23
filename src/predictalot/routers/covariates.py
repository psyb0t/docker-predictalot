"""POST /v1/covariates/{forecast,forecast/ensemble} + GET /v1/covariates/models.

Past+future combined mode. Only chronos-2 supports this in v0.2.

Route ordering note: this router is registered AFTER the more-specific
/v1/covariates/past/ and /v1/covariates/future/ routers. Even so the paths
don't collide because the per-endpoint segments (forecast, models, etc.)
differ — but registering in the right order avoids any future ambiguity.
"""

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

router = APIRouter(prefix="/v1/covariates", tags=["covariates"])


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


@router.get("/models")
def list_models() -> dict[str, Any]:
    return build_type_models_response(types.TYPE_COVARIATES_BOTH, models)
