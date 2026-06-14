# Error contract

Two shapes — application errors return a human-readable string; Pydantic validation errors (422) return a structured array (FastAPI default; we don't flatten).

**App errors** — `400`, `401`, `404`, `409`, `410`, `413`, `503`:

```json
{ "detail": "human-readable error string" }
```

**Validation errors** — `422` (wrong field type, missing required field, `horizon` not `> 0`, missing `pastCovariates` on a covariates endpoint, etc.):

```json
{
  "detail": [
    {
      "type": "greater_than",
      "loc": ["body", "config", "horizon"],
      "msg": "Input should be greater than 0",
      "input": 0
    }
  ]
}
```

## Status code reference

| Status | Shape | When |
|---|---|---|
| 400 | string | bad input — empty context, unsupported quantile level, horizon over compile-time cap, model is not a member of the requested type, unknown weights slug for the type, tabular train/forecast with conflicting mode, etc. |
| 401 | string | missing / wrong bearer token. Open-auth deployments (`PREDICTALOT_ALLOW_NO_AUTH=1` + empty token list) skip auth and never produce 401. |
| 404 | string | unknown model slug in `model` field; tabular `modelId` not found; meta-forecast called on a `modelId` that doesn't exist. |
| 409 | string | tabular train with `overwrite: false` against an existing `modelId`. |
| 410 | string | tabular forecast against a `modelId` whose backend is no longer registered (rare — happens if a backend slug is removed across versions). |
| 413 | string | request body > `PREDICTALOT_MAX_BODY_SIZE`. |
| 422 | array | Pydantic validation (field type / required / range constraints). |
| 503 | string | model snapshot download failed, inference error, or sidecar worker unreachable. |

## Common 400 causes by surface

**Univariate / multivariate:**
- Empty `context` or empty inner series.
- `model` not a member of the requested type (e.g. `timesfm-2.5` on multivariate).
- `horizon` exceeds `PREDICTALOT_TIMESFM_MAX_HORIZON` for timesfm-2.5.
- `quantileLevels` includes a value outside `{0.1, 0.2, ..., 0.9}`.
- `weights` contains a slug that isn't a member of the type.

**Covariates:**
- `pastCovariates` length doesn't match `context` length.
- `futureCovariates` length doesn't match `config.horizon`.
- A `futureCovariates` name that's absent from `pastCovariates` for the same series (chronos-2 constraint on the past+future type).

**Samples:**
- `numSamples ≤ 0`.

**Tabular train:**
- `target` and `features` series count mismatch.
- Feature key set differs across series.
- `mode` not supported by the chosen backend (rare — all 9 backends support all 3 modes today).
- `mode="quantile"` without `quantileLevels`.

**Tabular forecast:**
- Forecast features missing names the model was trained on.
- Forecast features include NaN/Inf in critical channels (those rows get zeroed silently — check `medianRet` if your forecasts look uniform).

**Tabular meta-train:**
- Stacking with members in non-direction modes (v1 limitation).
- Diversified with `mode="quantile"` but no `quantileLevels`.
- Calibrated with `mode!="direction"`.

## When you see 503

The 503 path covers three distinct causes — read the `detail` string to disambiguate:

1. **HuggingFace download failed** — the snapshot directory wasn't already cached and the HF API was unreachable / rate-limited. Retry. Pre-download with `PREDICTALOT_PREFETCH`.
2. **Inference threw** — the model loaded but the forward pass crashed (typically OOM on a too-long context, or a backend bug). Tighten `contextLength`; report repro.
3. **Sundial sidecar unreachable** — the sundial worker is down or restarting. Container's entrypoint auto-restarts within ~2s; retry. Check container logs (`docker logs predictalot`) — sundial worker stderr is tagged `[sundial]`.
