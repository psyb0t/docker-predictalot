# predictalot

> One HTTP service, six forecast types, five foundation time-series models, zero ceremony.

`POST /v1/univariate/forecast` with `{"model": "chronos-2", "context": [[...]], "config": {"horizon": 24}}` → quantile forecast back. Swap the slug — `chronos-2`, `timesfm-2.5`, `moirai-2`, `toto-1`, `sundial-base-128m` — and the wire shape stays identical.

Need richer modalities? Same wire family, different URL prefix: `/v1/multivariate`, `/v1/covariates/past`, `/v1/covariates/future`, `/v1/covariates`, `/v1/samples`. Each type advertises its own member-model list at `/v1/<type>/models`, accepts forecasts at `/v1/<type>/forecast`, and exposes a per-type ensemble at `/v1/<type>/forecast/ensemble`. A model either belongs to a type or it doesn't — no silent "this slug can't actually do that".

MCP streamable-http server at `/mcp` exposes one named tool per (type, model) cell — `forecast_univariate_chronos_2`, `forecast_multivariate_moirai_2`, `forecast_samples_toto_1`, etc. — plus a per-type `forecast_<type>_ensemble` and `list_<type>_models`. 26 tools total.

## Quick start

```bash
docker run -d --name predictalot \
  -v $HOME/predictalot-models:/models \
  -e PREDICTALOT_AUTH_TOKENS=changeme \
  -p 8080:8080 \
  psyb0t/predictalot:latest

# First call to a model downloads its weights into /models (~50-800MB each).
# Subsequent calls are fast. Bind-mount /models to persist across restarts.
curl -s http://localhost:8080/v1/univariate/forecast \
  -H "Authorization: Bearer changeme" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chronos-2",
    "context": [[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]],
    "config": {"horizon": 5}
  }' | jq

# Ensemble — run several univariate models in parallel, get a weighted mean
# PLUS each contributing model's individual forecast. Weight `0` disables a
# model.
curl -s http://localhost:8080/v1/univariate/forecast/ensemble \
  -H "Authorization: Bearer changeme" \
  -H "Content-Type: application/json" \
  -d '{
    "context": [[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]],
    "config": {"horizon": 5},
    "weights": {"chronos-2": 2.0, "moirai-2": 1.0, "timesfm-2.5": 0}
  }' | jq

# Which models implement each type (and their loaded/unloaded status).
# Same bearer requirement as the forecast endpoints.
for t in univariate multivariate covariates/past covariates/future covariates samples; do
  curl -s -H "Authorization: Bearer changeme" \
      "http://localhost:8080/v1/$t/models" | jq
done
```

GPU variant: pull `psyb0t/predictalot:latest-cuda` and add `--gpus all` to `docker run`. Both CPU and CUDA images are amd64-only (PyTorch's CPU wheel index has no aarch64 build at the pinned version).

## The five models

All five are PyTorch foundation forecasters trained for zero-shot use. Each one's *capabilities* (univariate-only / multivariate / past-covariate / future-covariate / samples) differ — that capability matrix is what drives the per-type URL family below.

| Slug | HF repo | What it is | Size | License |
|---|---|---|---|---|
| `chronos-2` | `amazon/chronos-2` | Amazon's tokenizer-based forecaster, configurable quantile output. Fastest of the five on CPU. The only model that handles both past AND future covariates. | ~120 MB | Apache 2.0 |
| `timesfm-2.5` | `google/timesfm-2.5-200m-pytorch` | Google Research's decoder-only model. Compile-time horizon cap. Univariate-only. | ~200 MB | Apache 2.0 |
| `moirai-2` | `Salesforce/moirai-2.0-R-small` | Salesforce's masked encoder. Native multivariate + past-covariate support. Fixed 9-quantile native output. Strong on clean seasonal data. | ~50 MB | CC-BY-NC-4.0 |
| `toto-1` | `Datadog/Toto-Open-Base-1.0` | Datadog's decoder transformer trained on ~2T observability metric points. Probabilistic via Student-T mixture sampling. Native multivariate; exposes raw sample paths. Strong on noisy / financial / trendy series. | ~580 MB | Apache 2.0 |
| `sundial-base-128m` | `thuml/sundial-base-128m` | Tsinghua's generative decoder-only with TimeFlow loss (flow-matching). ICML 2025 Oral, GIFT-Eval #1 MASE (May 2025). Exposes raw sample paths. Runs in a **sidecar venv** because it pins `transformers==4.40` — see [Architecture: sidecar pattern](#architecture-sidecar-pattern). | ~490 MB | Apache 2.0 |

## Forecast types — the capability matrix

Each row is one URL prefix. The members column lists which model slugs implement that type — those are the only slugs accepted by `<type>/forecast` and the only ones included in `<type>/forecast/ensemble`.

| Type | Base URL | Members | Request shape | Response shape |
|---|---|---|---|---|
| univariate | `/v1/univariate` | chronos-2, timesfm-2.5, moirai-2, toto-1, sundial-base-128m | `context: float[series][time]` | quantiles per series |
| multivariate | `/v1/multivariate` | chronos-2, moirai-2, toto-1 | `context: float[series][channel][time]` | quantiles per (series, channel) |
| covariates (past only) | `/v1/covariates/past` | chronos-2, moirai-2 | univariate target + `pastCovariates: list[dict[name, float[time]]]` | quantiles per series |
| covariates (future only) | `/v1/covariates/future` | chronos-2 | univariate target + `futureCovariates: list[dict[name, float[horizon]]]` | quantiles per series |
| covariates (past + future) | `/v1/covariates` | chronos-2 | univariate target + `pastCovariates` + `futureCovariates` | quantiles per series |
| samples | `/v1/samples` | toto-1, sundial-base-128m | univariate target + `numSamples` | `samples: float[series][sample][time]` (raw paths) |

Every base URL exposes the same three sub-paths: `<base>/forecast`, `<base>/forecast/ensemble`, `<base>/models`.

**Cross-combos deferred** (multivariate-covariates, multivariate-samples) — chronos-2 + toto-1 can do them natively; ship as separate types in v0.3 if asked.

## API — univariate (`/v1/univariate/forecast`)

The smallest and most common shape: one or more 1D series, return quantile forecasts.

### Request

```json
{
  "model": "chronos-2",
  "context": [[1.0, 2.0, 3.0, ...], [10.0, 11.0, ...]],
  "config": {
    "horizon": 24,
    "quantileLevels": [0.1, 0.5, 0.9],
    "contextLength": 2048
  },
  "unload": false
}
```

| Field | Required | Default | Notes |
|---|---|---|---|
| `model` | yes | — | One of `chronos-2`, `timesfm-2.5`, `moirai-2`, `toto-1`, `sundial-base-128m`. Unknown slug → 404; not-a-univariate-member (impossible currently — all five support it) → 400. |
| `context` | yes | — | `List[List[float]]`. One inner list per series. Single-series = `[[...]]`. Variable-length series are zero/edge-padded per model. |
| `config.horizon` | yes | — | Steps into the future to forecast. Per-model upper bounds (see Per-model quirks). |
| `config.quantileLevels` | no | `[0.1, 0.5, 0.9]` | Subset of `{0.1, 0.2, ..., 0.9}` (the only levels every model supports). |
| `config.contextLength` | no | per-model (2048 / 2048 / 4000 / 4096 / 2880) | Max history points to feed the model. Longer inputs sliced to the last N. |
| `unload` | no | `false` | Tear down the model after this response (frees RAM/VRAM immediately). |

### Response

```json
{
  "model": "chronos-2",
  "horizon": 24,
  "quantileLevels": [0.1, 0.5, 0.9],
  "median": [[...], [...]],
  "quantiles": {
    "0.1": [[...], [...]],
    "0.5": [[...], [...]],
    "0.9": [[...], [...]]
  }
}
```

`median` is always present — best-guess point forecast. For most backends (Moirai-2, Toto-1, Sundial), `quantiles["0.5"]` and `median` are the same value (both derived from sample paths). **Chronos-2 is the exception**: its `median` is the mean of the predictive distribution (E[X]) returned by `predict_quantiles`, not the strict 50th percentile, so `median` and `quantiles["0.5"]` can differ for asymmetric distributions.

### Ensemble — `POST /v1/univariate/forecast/ensemble`

Run every member of the type in parallel and combine into a weighted-mean forecast. Returns the ensemble + every contributing model's individual forecast, so you can inspect dissent or post-process.

```bash
curl -s http://localhost:8080/v1/univariate/forecast/ensemble \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "context": [[1.0, 2.0, 3.0, 4.0, 5.0]],
    "config": {"horizon": 5},
    "weights": {
      "chronos-2": 2.0,
      "moirai-2": 1.0,
      "timesfm-2.5": 0
    }
  }' | jq
```

Request shape — same as the per-model `/v1/univariate/forecast` minus `model`, plus an optional `weights` map:

| Field | Type | Default | Notes |
|---|---|---|---|
| `weights` | `{slug: float}` | uniform 1.0 | Non-negative per-model weights restricted to type members. Normalized internally (any positive numbers work). Weight `0` skips that model entirely (not called — that's how you disable a model). Unknown slug for this type → 400. Omitted entry → weight 1.0. |

Response adds three fields on top of the standard per-type forecast shape:

| Field | What it is |
|---|---|
| `ensembleMembers` | List of slugs that actually ran (i.e. weight > 0). |
| `weights` | The **normalized** weight per model that was applied to the average. |
| `individual` | `{slug: full forecast result}` map — each entry has `model`, `horizon`, `quantileLevels`, `median`, `quantiles`, **and `weight`** (mirrors the top-level `weights[slug]`). |

Failure of any one **included** model fails the whole call. The same ensemble pattern applies to every other type — `/v1/multivariate/forecast/ensemble`, `/v1/covariates/past/forecast/ensemble`, etc. The wire shapes match each type's per-model response.

### Per-model quirks

- **chronos-2** — native arbitrary-quantile output. No restrictions beyond `{0.1, ..., 0.9}`. Returns `list[Tensor]` (one per series, multivariate-shaped); we squeeze the channel axis for univariate output.
- **timesfm-2.5** — compile-time `max_horizon` (default 512, must be multiple of 128). Request horizon > max → 400. Bump via `PREDICTALOT_TIMESFM_MAX_HORIZON` and restart. We bypass the library's built-in padding (mask=True path produces NaN at this commit) and edge-pad short inputs ourselves.
- **moirai-2** — fixed 9-quantile output `{0.1..0.9}`; we filter to your requested subset. Wrapper context/horizon are baked at model-load time (`PREDICTALOT_MOIRAI_MAX_CONTEXT` / `_MAX_HORIZON`). Per-request inputs are zero-padded to the wrapper's context length with a `past_is_pad` mask.
- **toto-1** — multivariate-native. Univariate calls run as a single-channel series. Quantiles via Monte-Carlo sampling (256 draws → empirical percentiles); the same draws drive `/v1/samples/forecast` when called via that type. Returns `[batch, channels, horizon]` shape; we squeeze leading dims for univariate output.
- **sundial-base-128m** — runs in its own sidecar process. From the API surface it's identical to the others. Generative sampling (`num_samples=64` by default, tune via `PREDICTALOT_SUNDIAL_NUM_SAMPLES`). First request waits for the sidecar to be reachable on its unix socket — usually <2s after container start.

## API — multivariate (`/v1/multivariate/forecast`)

Each series carries multiple correlated channels (variates). Channels are forecast jointly per series.

### Request

```json
{
  "model": "moirai-2",
  "context": [
    [
      [1.0, 2.0, 3.0, 4.0],
      [10.0, 20.0, 30.0, 40.0]
    ]
  ],
  "config": {"horizon": 3}
}
```

| Field | Notes |
|---|---|
| `model` | `chronos-2`, `moirai-2`, or `toto-1`. Other slugs → 400 (not a multivariate member). |
| `context` | `List[List[List[float]]]` — `[series][channel][time]`. All series must have the same channel count (mismatch → 400). |
| `config` | Same `horizon`/`quantileLevels`/`contextLength` shape as univariate. |

> **Moirai-2 multivariate caveat** — the upstream multivariate path is not exercised by Salesforce's own benchmark suite. We've validated output shapes but not numerical accuracy against a known-good baseline. For high-stakes multivariate workloads prefer `chronos-2` or `toto-1` until upstream ships a multivariate-specific eval.

### Response

```json
{
  "model": "moirai-2",
  "horizon": 3,
  "quantileLevels": [0.1, 0.5, 0.9],
  "median": [[[...], [...]]],
  "quantiles": {
    "0.1": [[[...], [...]]],
    "0.5": [[[...], [...]]],
    "0.9": [[[...], [...]]]
  }
}
```

`median` and each `quantiles[level]` are shaped `[series][channel][time]`.

Ensemble is `POST /v1/multivariate/forecast/ensemble` — same `weights` semantics as univariate, restricted to multivariate members.

## API — covariates: past only (`/v1/covariates/past/forecast`)

Forecast a univariate target conditioned on covariates whose values are known up to *now* but not into the future (e.g. observed temperature, observed promo flag).

### Request

```json
{
  "model": "chronos-2",
  "context": [[1.0, 2.0, 3.0, 4.0]],
  "pastCovariates": [
    {
      "temp": [20.0, 21.0, 22.0, 23.0],
      "promo": [0.0, 0.0, 1.0, 1.0]
    }
  ],
  "config": {"horizon": 3}
}
```

| Field | Notes |
|---|---|
| `model` | `chronos-2` or `moirai-2`. |
| `context` | `[series][time]` — univariate target. |
| `pastCovariates` | One dict per series. Keys are covariate names; values are 1D float lists matching that series' context length. Every series must carry the same covariate names. |

### Response

Same shape as univariate (`median` and `quantiles[level]` are `[series][time]`).

## API — covariates: future only (`/v1/covariates/future/forecast`)

Forecast a univariate target conditioned on covariates known *only* over the future window (e.g. a planned price, a scheduled promotion, a weather forecast). The covariate value array per series must have length `horizon`.

### Request

```json
{
  "model": "chronos-2",
  "context": [[1.0, 2.0, 3.0, 4.0]],
  "futureCovariates": [
    {"price": [9.5, 9.6, 9.7]}
  ],
  "config": {"horizon": 3}
}
```

Only `chronos-2` implements this type in v0.2. Response shape matches univariate. The per-type ensemble endpoint exists for API symmetry — with a single member it's a degenerate single-result wrapper.

## API — covariates: past + future (`/v1/covariates/forecast`)

Forecast a univariate target with covariates that are observed up to *now* AND known into the future (e.g. `price`: known historical, known planned).

### Request

```json
{
  "model": "chronos-2",
  "context": [[1.0, 2.0, 3.0, 4.0]],
  "pastCovariates": [
    {"price": [9.0, 9.0, 9.5, 9.5]}
  ],
  "futureCovariates": [
    {"price": [9.5, 9.6, 9.7]}
  ],
  "config": {"horizon": 3}
}
```

**Constraint inherited from chronos-2:** every covariate name appearing in `futureCovariates` MUST also appear in `pastCovariates` for the same series. The backend rejects future-only names.

Only `chronos-2` implements this type in v0.2. Response shape matches univariate.

## API — samples (`/v1/samples/forecast`)

Returns raw Monte-Carlo sample paths instead of quantile summaries. Use this when you need joint distributions across timesteps, custom risk metrics, or scenario analysis on the actual draws.

### Request

```json
{
  "model": "toto-1",
  "context": [[1.0, 2.0, 3.0, 4.0]],
  "config": {"horizon": 3, "numSamples": 64}
}
```

| Field | Notes |
|---|---|
| `model` | `toto-1` or `sundial-base-128m`. |
| `config.numSamples` | Sample paths to draw per series. Default 64. `> 0`. |
| `config.contextLength` | As elsewhere. |
| `config.quantileLevels` | **Not used** here — the samples type returns paths, not summarized quantiles. |

### Response

```json
{
  "model": "toto-1",
  "horizon": 3,
  "numSamples": 64,
  "samples": [[[...], [...], ...]],
  "median": [[...]]
}
```

| Field | Shape | Notes |
|---|---|---|
| `samples` | `[series][sample][time]` | Raw draws — order is not stable across calls. |
| `median` | `[series][time]` | Convenience: per-step median across the sample axis. |

### Samples ensemble (`/v1/samples/forecast/ensemble`)

Weights control how many sample paths each model contributes. Per-member share is `max(1, round(weight * numSamples))` — a minority member with a small weight still contributes at least one path, so the ensemble never silently drops a model. Final `samples` is the concatenation of every member's draws along the sample axis; `median` is recomputed across the pooled paths.

```json
{
  "context": [[1.0, 2.0, 3.0, 4.0]],
  "config": {"horizon": 3, "numSamples": 32},
  "weights": {"toto-1": 1.0, "sundial-base-128m": 1.0}
}
```

→ each model draws 16 paths; response carries 32 total in `samples` and per-member detail in `individual`.

## Per-type `/models` listings

Every type advertises its members at `GET /v1/<type>/models`. Same bearer-token requirement as the forecast endpoints (the listing reveals which models are loaded and when they were last used). Returns the type slug + per-member runtime state.

```bash
curl -s -H "Authorization: Bearer changeme" \
    http://localhost:8080/v1/univariate/models | jq
```

```json
{
  "type": "univariate",
  "models": [
    {
      "slug": "chronos-2",
      "loaded": false,
      "lastUsedSecsAgo": null,
      "idleTimeoutSecs": 1800.0
    },
    {"slug": "timesfm-2.5",       "loaded": false, "lastUsedSecsAgo": null, "idleTimeoutSecs": 1800.0},
    {"slug": "moirai-2",          "loaded": true,  "lastUsedSecsAgo": 4.1,  "idleTimeoutSecs": 1800.0},
    {"slug": "toto-1",            "loaded": false, "lastUsedSecsAgo": null, "idleTimeoutSecs": 1800.0},
    {"slug": "sundial-base-128m", "loaded": false, "lastUsedSecsAgo": null, "idleTimeoutSecs": 1800.0}
  ]
}
```

A model that supports multiple types appears in every relevant listing (e.g. `chronos-2` shows up in all six). The `loaded` / `lastUsedSecsAgo` fields are shared across listings — there's one backend per slug, not one per type.

## Error contract

Two shapes — application errors return a human-readable string; Pydantic validation errors (422) return a structured array (FastAPI default; we don't flatten).

**App errors** — `400`, `401`, `404`, `413`, `503`:

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

| Status | Shape | When |
|---|---|---|
| 400 | string | bad input (empty context, unsupported quantile level, horizon over compile-time cap, model is not a member of the requested type, unknown weights slug for the type) |
| 401 | string | missing / wrong bearer token |
| 404 | string | unknown model slug in `model` field |
| 413 | string | request body > `PREDICTALOT_MAX_BODY_SIZE` |
| 422 | array | Pydantic validation (field type / required / range constraints) |
| 503 | string | model snapshot download failed, inference error, or sidecar worker unreachable |

## MCP — `/mcp`

Streamable-HTTP MCP server. Same auth as HTTP (bearer header or `?apiToken=...` query). 26 tools, organized by forecast type. The tool surface is identical to the HTTP API — one named tool per (type, model) cell in the matrix.

Naming convention:
- `forecast_<type>_<model>` — single-model forecast. Examples: `forecast_univariate_chronos_2`, `forecast_multivariate_moirai_2`, `forecast_samples_toto_1`.
- `forecast_<type>_ensemble` — per-type weighted ensemble. One per type (6 total).
- `list_<type>_models` — per-type runtime listing. One per type (6 total).

Model slug dashes/dots become underscores in the tool name (`sundial-base-128m` → `sundial_base_128m`, `timesfm-2.5` → `timesfm_2_5`).

| Type | Per-model tools | Ensemble | List |
|---|---|---|---|
| univariate | `forecast_univariate_chronos_2`, `..._timesfm_2_5`, `..._moirai_2`, `..._toto_1`, `..._sundial_base_128m` | `forecast_univariate_ensemble` | `list_univariate_models` |
| multivariate | `forecast_multivariate_chronos_2`, `..._moirai_2`, `..._toto_1` | `forecast_multivariate_ensemble` | `list_multivariate_models` |
| covariates-past | `forecast_covariates_past_chronos_2`, `..._moirai_2` | `forecast_covariates_past_ensemble` | `list_covariates_past_models` |
| covariates-future | `forecast_covariates_future_chronos_2` | `forecast_covariates_future_ensemble` | `list_covariates_future_models` |
| covariates (past + future) | `forecast_covariates_both_chronos_2` | `forecast_covariates_both_ensemble` | `list_covariates_both_models` |
| samples | `forecast_samples_toto_1`, `forecast_samples_sundial_base_128m` | `forecast_samples_ensemble` | `list_samples_models` |

Args per tool mirror the HTTP body, flattened to keyword arguments:
- quantile types: `context, horizon, quantile_levels=None, context_length=None, unload=False`
  (covariate variants add `past_covariates` and/or `future_covariates`)
- samples: `context, horizon, num_samples=None, context_length=None, unload=False`
- every `forecast_<type>_ensemble`: same args as the per-model tool minus `model`, plus `weights: dict[str, float] | None = None`

Named per-(type, model) tools (not one polymorphic `forecast(type=..., model=...)`) — LLM agents discover and pick named capabilities more reliably than they pick a pair of enum arguments.

## Configuration (env vars)

| Var | Default | What it does |
|---|---|---|
| `PREDICTALOT_HOST` | `0.0.0.0` | uvicorn bind host |
| `PREDICTALOT_PORT` | `8080` | uvicorn bind port |
| `PREDICTALOT_AUTH_TOKENS` | (empty) | Comma-separated bearer tokens. Empty = open (refused at startup unless allow-no-auth). |
| `PREDICTALOT_ALLOW_NO_AUTH` | `0` | Required to start with empty token list. |
| `PREDICTALOT_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `cuda:N`. |
| `PREDICTALOT_MODEL_DIR` | `/models` | Where snapshot directories land. Bind-mount this to persist across restarts. |
| `PREDICTALOT_PREFETCH` | (empty) | Comma-separated slugs or `all` — prefetched at container start before uvicorn boots. |
| `PREDICTALOT_PRELOAD` | (empty) | Comma-separated slugs to load into memory at boot. |
| `PREDICTALOT_MODEL_IDLE_TIMEOUT` | `30m` | Idle time before a loaded model is unloaded. Go-style: `30m`, `1h`, `1d2h3m`. `0` disables. |
| `PREDICTALOT_MODEL_IDLE_TIMEOUT_<SLUG>` | inherits global | Per-model override. Slug normalized: uppercase + `-`/`.` → `_` (e.g. `_MOIRAI_2`). |
| `PREDICTALOT_MAX_BODY_SIZE` | `32mb` | Cap on request body. Human-readable: `32mb`, `512k`, `1g`, plain int = bytes. |
| `PREDICTALOT_TIMESFM_MAX_CONTEXT` | `2048` | Compile-time max for TimesFM. Multiple of 32. |
| `PREDICTALOT_TIMESFM_MAX_HORIZON` | `512` | Compile-time max for TimesFM. Multiple of 128. |
| `PREDICTALOT_MOIRAI_MAX_CONTEXT` | `4000` | Wrapper context-length for Moirai-2. Per-request inputs zero-padded to this length. |
| `PREDICTALOT_MOIRAI_MAX_HORIZON` | `512` | Wrapper prediction-length for Moirai-2. Per-request horizons must be ≤ this. |
| `PREDICTALOT_SUNDIAL_SOCK` | `/tmp/predictalot/sundial.sock` | Unix-socket path the main service uses to talk to the sundial sidecar. |
| `PREDICTALOT_SUNDIAL_NUM_SAMPLES` | `64` | Monte-Carlo samples per sundial forecast (more = smoother quantiles, linearly slower). |
| `PREDICTALOT_SUNDIAL_READY_TIMEOUT` | `60s` | How long the main service waits for the sundial sidecar to come up on first request. |
| `PREDICTALOT_LOG_LEVEL` | `INFO` | Standard Python log levels. |

## Architecture: sidecar pattern

Four of the five models (`chronos-2`, `timesfm-2.5`, `moirai-2`, `toto-1`) live in the **main Python venv** at `/opt/venv`. They share `torch==2.4.1`, `transformers==4.57.6`, etc. — a single resolved dependency tree.

The fifth — `sundial-base-128m` — runs in its **own venv at `/opt/sundial-venv`** because Sundial's model code uses `transformers==4.40.1` internals (`DynamicCache.seen_tokens`, `get_max_length`, `get_usable_length`, plus a 4D-mask shape change) that were removed in transformers 4.42+. Shimming them from the outside breaks deeper inside transformers itself.

```
                  /tmp/predictalot/sundial.sock
                              │
┌──────────────────┐          │          ┌──────────────────┐
│ main predictalot │ ─────────┴────────► │ sundial worker   │
│ /opt/venv        │  httpx.AsyncClient  │ /opt/sundial-venv│
│ transformers 4.57│  (unix-domain HTTP) │ transformers 4.40│
│ chronos / timesfm│                     │ sundial deps     │
│ moirai / toto    │                     │ FastAPI worker   │
└──────────────────┘                     └──────────────────┘
        ▲                                          ▲
        │                                          │
   uvicorn :8080                              uvicorn --uds
   (public API)                            (internal only)
```

- The main service makes HTTP-over-UDS calls (`httpx.AsyncHTTPTransport(uds=...)`) to the sundial worker. From the main service's `models/sundial.py` it looks like any other backend (`get_model` waits for `/healthz`, `predict` POSTs to `/forecast`).
- The sundial worker is its own tiny FastAPI app (`sundial_worker/server.py`) — loads the model lazily, serves `/forecast`, exposes `/healthz`.
- The container's entrypoint starts the sundial worker as a background process with an auto-restart loop. If the worker crashes (OOM, ImportError, whatever) the loop relaunches it within ~2 seconds; mid-restart requests get 503 until it's back.

**This pattern is reusable.** Any future model with version conflicts that can't be shimmed drops in the same way: create `/opt/<name>-venv` with its own deps, write `<name>_worker/server.py`, add a thin `models/<name>.py` in the main service that talks to it via httpx-over-UDS, register in the entrypoint's restart loop.

Trade-offs:
- ✅ Clean dep isolation — Python import conflicts are impossible across venvs.
- ✅ Reusable pattern for the next conflicted model.
- ❌ Image size: sundial venv is ~2GB on its own (its own torch + transformers + numpy). Couldn't share via symlinks because uv's resolver clobbers them on dep updates.
- ❌ Two processes per container — small operational complexity, visible in `ps`.

## Accuracy & latency

Numbers from `make bench` against `psyb0t/predictalot-test:cuda` on an RTX 3060. Lower sMAPE = better forecast. Latency is round-trip wall-clock per single-series univariate forecast (after warmup, single-series in the request body).

### Accuracy (sMAPE %)

Three classic academic benchmarks:

| Dataset (horizon) | seasonal-naive | chronos-2 | timesfm-2.5 | moirai-2 | toto-1 | sundial | ens uniform |
|---|---:|---:|---:|---:|---:|---:|---:|
| AirPassengers (h=24, n=144 monthly) | 17.01 | 8.26 | 12.42 | **7.71** | 19.88 | 16.57 | 12.75 |
| Shampoo Sales (h=6, n=36 monthly) | **25.54** | 27.24 | 40.37 | 33.74 | 24.27 | 42.07 | 32.07 |
| Daily-Min-Temperatures (h=30, n=1460 daily) | 17.72 | 13.23 | 13.65 | 13.77 | 13.76 | **12.72** | 13.30 |

Three real-world benchmarks fetched live by `make bench`:

| Dataset (horizon) | seasonal-naive | chronos-2 | timesfm-2.5 | moirai-2 | toto-1 | sundial | ens uniform |
|---|---:|---:|---:|---:|---:|---:|---:|
| CO2 Mauna Loa (h=24, n=818 monthly) | 1.05 | 0.24 | 0.16 | 0.13 | **0.09** | 0.58 | 0.17 |
| Gold PAXG/USDT (h=30, n=1000 daily) | 2.18 | 3.26 | 2.67 | 3.12 | 3.58 | **1.67** | 2.81 |
| BTC/USDT (h=30, n=1000 daily) | 3.18 | 2.74 | 2.75 | 5.00 | **2.02** | 2.91 | 2.98 |

### Honest takeaways

1. **No single model dominates real-world data.** Each model wins on at least one dataset, and which one depends on the signal shape: moirai on clean seasonal, toto on observability-style noisy series, sundial on financial drift, chronos as a steady all-rounder. **Toto-1 wins on CO2 + BTC + Shampoo** (its training distribution favors trendy/noisy series). **Sundial wins on Gold + Daily Temps**. **Moirai wins on AirPassengers**.

2. **Uniform ensemble loses to the right single model on every real-world dataset.** Averaging with a model that has the wrong inductive bias drags the mean down. Pick `weights` per-domain when you know what fits.

3. **TimesFM 2.5 is the weakest of the five on every dataset we tested** — slowest AND worst or near-worst sMAPE. Consider setting `"weights": {"timesfm-2.5": 0}` in ensemble calls until a newer release ships.

4. **`ens no-timesfm`** (`{"chronos-2": 1, "timesfm-2.5": 0, "moirai-2": 1, "toto-1": 1, "sundial-base-128m": 1}`) is a strong "I don't know which model fits" default.

5. **Foundation models don't beat the market.** Best result on BTC (toto-1 at 2.02% sMAPE vs naive 3.18%) is real but tiny — definitely not "trade on this" predictive.

6. **For comparison:** hand-tuned ARIMA on AirPassengers gets ~3-5% sMAPE. predictalot's best on that dataset is moirai-2 at 7.71% sMAPE — competitive zero-shot, no per-series fitting.

### Latency per single-series univariate forecast

| Model | CPU (ms) | CUDA RTX 3060 (ms) | speedup |
|---|---:|---:|---:|
| chronos-2 | 60–180 | 24–35 | ~3-6× |
| timesfm-2.5 | 1240–1300 | 320–340 | ~3.7× |
| moirai-2 | 4200–4950 | 215–230 | ~20× |
| toto-1 | (large) | 45–435 | varies w/ context length |
| sundial-base-128m | (sidecar) | 65–85 | very fast on GPU |
| ensemble (N parallel) | ≈ slowest member | ≈ slowest member | ≈ slowest |

Cold-load (first request after container start):

| Model | CPU | CUDA |
|---|---:|---:|
| chronos-2 | ~4.1s | ~32 ms |
| timesfm-2.5 | ~3.0s | ~2.3s |
| moirai-2 | ~4.9s | ~720 ms |
| toto-1 | (large) | ~1.5s |
| sundial-base-128m | (sidecar boot ~1s + load ~7s) | (sidecar boot ~1s + load ~3s) |

CPU↔GPU produces near-identical forecasts (float rounding noise only) — running on CPU isn't an accuracy regression, just a latency one.

Run the bench yourself against a live container:

```bash
make bench                                                 # localhost:18080, devtok
PREDICTALOT_BENCH_URL=http://remote:8080 \
PREDICTALOT_BENCH_TOKEN=mytoken make bench                 # remote, custom token
```

## CPU vs CUDA images

| Image | Tag | Platforms | Notes |
|---|---|---|---|
| CPU | `psyb0t/predictalot:latest` | amd64 only | PyTorch CPU wheels (pytorch.org's CPU index has no manylinux aarch64 wheel at the pinned `torch==2.4.1+cpu`). |
| CUDA | `psyb0t/predictalot:latest-cuda` | amd64 only | PyTorch CUDA 12.4 wheels on CUDA 12.6 runtime base. Needs `--gpus all` + NVIDIA driver on host. CUDA on arm64 is a different stack (Jetson L4T / SBSA) and not on the menu. |

Both images are self-sufficient — same source, same API, same env vars. Pick the one that matches your host. The CUDA image also runs on CPU if `--gpus` isn't passed (useful for debugging).

Image sizes (approximate, compressed):
- CPU: ~3 GB
- CUDA: ~9 GB (PyTorch CUDA wheels are 2GB+ on their own; sundial sidecar adds ~2GB)

## Development

Everything runs in a sandboxed dev container — your host needs only `docker`, `make`, `git`, and a shell. Optionally `uv` if you want to run `make test-integration` / `make bench` directly from the host.

```bash
make help            # list all targets
make dev-image       # build the dev container (run once, rebuilt on lock changes)
make test            # unit tests with stubbed backends — fast + offline (no ML libs needed)
make lint            # flake8 + mypy inside the dev container
make format          # isort + black inside the dev container

make pkg-add PKG=foo==1.2.3   # add a dep (bumps exclude-newer first)
make pkg-upgrade              # bump exclude-newer + refresh all pins
make deps-lock          # regenerate requirements-{cpu,cuda}.txt (hash-locked)

make build           # build CPU production image
make build-cuda      # build CUDA production image
make build-all       # both
make run             # build + run CPU image with a dev token
make run-cuda        # build + run CUDA image with --gpus all

make test-integration  # build the prod image, run it, hit it with real ML calls
make bench             # accuracy + latency benchmark on real public datasets
```

`make test-integration` auto-detects CUDA (via `docker info | grep nvidia`) and uses the matching image with `--gpus all` if available. Models cache to `tests/integration/.fixtures/models/` (gitignored). Integration tests cover every type endpoint × member model plus per-type ensemble + unload flag + CUDA detection.

`make bench` requires a running container reachable at `PREDICTALOT_BENCH_URL` (default `http://127.0.0.1:18080`) with bearer token in `PREDICTALOT_BENCH_TOKEN` (default `devtok`). Compares each individual model against the seasonal-naive baseline and several weighted-ensemble variants on six datasets (three academic + three real-world public APIs: NOAA CO2, Binance gold/BTC). All bench calls go through `/v1/univariate/forecast`.

`make pkg-*` targets follow the supply-chain age-gate pattern: every dep mutation bumps `[tool.uv] exclude-newer` to today's UTC midnight first, so brand-new (potentially compromised) package versions are refused at install time.

## Security notes

Every dependency is exactly pinned with hash verification where possible:
- Lightweight runtime deps live in `uv.lock` (hash-verified).
- ML stack lives in `requirements-{cpu,cuda}.txt` (hash-verified, installed with `--require-hashes`).
- Sundial sidecar deps live in `requirements-sundial.txt` (manually pinned).
- Base images pinned by `@sha256:...` digest.
- `timesfm` is a git install pinned by full 40-char commit SHA.
- `[tool.uv] exclude-newer` refuses to install package versions newer than the gate date, blocking same-day supply-chain attacks at lockfile generation time.

Pinned versions of `torch` and `transformers` have open CVEs in OSV. Each was reviewed against the predictalot threat model and confirmed non-applicable:

| Library | Class of advisory | Why it doesn't apply here |
|---|---|---|
| `torch` | `torch.load()` RCE / deserialization | We never call `torch.load()` on untrusted files. Weights come from HuggingFace via `snapshot_download` from hardcoded official org repos. |
| `torch` | `RemoteModule` RCE | We don't use distributed `RemoteModule`. |
| `torch` | Local DoS in specific ops (`PairwiseDistance`, `cummin`, `lu`, etc.) | The five backends use Transformers/standard math paths, not the affected ops. |
| `torch` | Advisories listing "Affects 2.6.0/2.7.0/2.8.0" | OSV range-matches these back to 2.4.1; descriptions confirm the regressions were introduced in 2.6+. False-positive against our pin. |
| `transformers` | `Trainer` arbitrary code exec | We only do inference. No `Trainer`. |
| `transformers` | Per-model deserialization/conversion RCE (Perceiver, Transformer-XL, X-CLIP, GLM4, SEW, HuBERT, megatron_gpt2) | We load Chronos (T5-based) only via the chronos-forecasting package. None of those model classes is loaded. |

**Why not bump past the CVE-fixing versions?** `uni2ts==2.0.0` caps `torch<2.5`; the torch fix line for the most cited issue (GHSA-53q9-r3pm-6pq6, `weights_only=True` RCE) is `2.6.0`. Until uni2ts releases a newer version with torch>=2.5 support, we can't move. Same constraint forces transformers `<5` (chronos cap) — and the open transformers CVEs are all about loading untrusted model files or running training, neither of which predictalot does.

**Sundial sidecar venv** pins `transformers==4.40.1`, which has its own CVE set. Same threat model applies — sundial loads only the hardcoded `thuml/sundial-base-128m` checkpoint via `snapshot_download` from HF and does inference, no Trainer, no arbitrary model conversion.

If you point `PREDICTALOT_MODEL_DIR` at a directory containing **arbitrary user-provided model weights**, you're on your own — predictalot's auto-fetch only writes to `MODEL_DIR/<slug>/` from the hardcoded HuggingFace repo ids; there's no API surface that takes a model path.

Run `osv-scanner` against the image before deployment if you want a fresh advisory check.

## Roadmap

### Maybe in v0.3

- **Cross-combo types** — `/v1/multivariate/covariates/...` and `/v1/multivariate/samples/forecast` if anyone asks. chronos-2 + toto-1 do these natively; the per-type registry pattern absorbs them cleanly.
- **Toto-2.0** (released April 2026) — newer architecture, top-3 on GIFT-Eval CRPS. Blocked by torch 2.5+ requirement (uni2ts cap). Watch for PyPI packaging; currently git-only.
- **More sidecar models** — the pattern is in place; adding e.g. ibm-granite's TTM-r2 or FlowState requires a new venv + worker module, no architectural change.

### Other / no schedule

- **Big-context uploads** — multipart `form-data` or binary body when JSON-32MB isn't enough. Defaults are sized for normal-shape workloads.
- **Pre-baked weights** — build-arg option to bake selected models into the image for airgapped deploys. Currently weights are runtime-downloaded.

## License

WTFPL — Do What The Fuck You Want To Public License. See `LICENSE`.
