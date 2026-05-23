"""POST /v1/samples/{forecast,forecast/ensemble} + GET /v1/samples/models.

Returns raw sample paths (one path per Monte-Carlo draw) instead of quantiles.
Useful for callers that want to compute custom risk metrics, joint-distribution
queries, or scenario analysis over the raw draws.

Members: toto-1, sundial-base-128m.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .. import dispatch, models, types
from ..auth import check_bearer
from ._common import build_type_models_response
from .schemas import (
    SamplesEnsembleRequest,
    SamplesEnsembleResponse,
    SamplesRequest,
    SamplesResponse,
)

log = logging.getLogger("predictalot.routers.samples")

router = APIRouter(prefix="/v1/samples", tags=["samples"])


@router.post(
    "/forecast/ensemble",
    response_model=SamplesEnsembleResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_ensemble(body: SamplesEnsembleRequest) -> dict[str, Any]:
    try:
        return await dispatch.ensemble_samples(
            context=body.context,
            horizon=body.config.horizon,
            num_samples=body.config.num_samples,
            context_length=body.config.context_length,
            weights=body.weights,
            unload_after=body.unload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        log.exception("samples ensemble failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/forecast",
    response_model=SamplesResponse,
    dependencies=[Depends(check_bearer)],
)
async def post_forecast(body: SamplesRequest) -> dict[str, Any]:
    try:
        return await dispatch.dispatch_samples(
            model=body.model,
            context=body.context,
            horizon=body.config.horizon,
            num_samples=body.config.num_samples,
            context_length=body.config.context_length,
            unload_after=body.unload,
        )
    except (dispatch.UnknownModelError, types.UnknownTypeError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except types.ModelDoesNotSupportTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        log.exception("samples forecast failed for model=%s", body.model)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/models")
def list_models() -> dict[str, Any]:
    return build_type_models_response(types.TYPE_SAMPLES, models)
