# Foundation time-series API — `/v1/timeseries/<type>/…`

Five PyTorch foundation forecasters trained for zero-shot use. Six forecast types selected by URL. Same wire shape across all five backends per type.

## The five models

| Slug | What it is | Size | License | Recommended for |
|---|---|---|---|---|
| `chronos-2` | Amazon's tokenizer-based forecaster, configurable quantile output. Fastest of the five on CPU. The only model that handles both past AND future covariates. | ~120 MB | Apache 2.0 | **Default all-rounder when you don't know the signal shape.** General-purpose forecasting, mixed signal types, when you need past+future covariates in one call, when CPU latency matters. |
| `timesfm-2.5` | Google Research's decoder-only model. Compile-time horizon cap. Univariate-only. | ~200 MB | Apache 2.0 | **Currently the weakest of the five** on every benchmarked dataset. Recommended to skip or set `weights={"timesfm-2.5": 0}` in ensemble calls until a newer release ships. Kept in the registry for forward compatibility. |
| `moirai-2` | Salesforce's masked encoder. Native multivariate + past-covariate support. Fixed 9-quantile native output. | ~50 MB | CC-BY-NC-4.0 | **Clean cyclic / seasonal series** (textbook AirPassengers-shape). When you have multiple correlated channels. Note the non-commercial license. |
| `toto-1` | Datadog's decoder transformer trained on ~2T observability metric points. Probabilistic via Student-T mixture sampling. Native multivariate; exposes raw sample paths. | ~580 MB | Apache 2.0 | **Noisy / observability-style / financial / trendy series.** Wins on BTC and CO2 in our benchmarks. When you need raw sample paths for joint risk metrics. |
| `sundial-base-128m` | Tsinghua's generative decoder-only with TimeFlow loss (flow-matching). ICML 2025 Oral, GIFT-Eval #1 MASE (May 2025). Exposes raw sample paths. Runs in a **sidecar venv** because it pins `transformers==4.40` (see [architecture.md](architecture.md)). | ~490 MB | Apache 2.0 | **Financial drift series, gold-style trends.** Strong on daily-temperature / drifting price data. When you need generative sample paths. |

## Capability matrix

| Type | Base URL | Members | Request shape | Response shape |
|---|---|---|---|---|
| univariate | `/v1/timeseries/univariate` | chronos-2, timesfm-2.5, moirai-2, toto-1, sundial-base-128m | `context: float[series][time]` | quantiles per series |
| multivariate | `/v1/timeseries/multivariate` | chronos-2, moirai-2, toto-1 | `context: float[series][channel][time]` | quantiles per (series, channel) |
| covariates (past only) | `/v1/timeseries/covariates/past` | chronos-2, moirai-2 | univariate target + `pastCovariates: list[dict[name, float[time]]]` | quantiles per series |
| covariates (future only) | `/v1/timeseries/covariates/future` | chronos-2 | univariate target + `futureCovariates: list[dict[name, float[horizon]]]` | quantiles per series |
| covariates (past + future) | `/v1/timeseries/covariates` | chronos-2 | univariate target + both | quantiles per series |
| samples | `/v1/timeseries/samples` | toto-1, sundial-base-128m | univariate target + `numSamples` | `samples: float[series][sample][time]` (raw paths) |

Every base URL exposes the same three sub-paths: `<base>/forecast`, `<base>/forecast/ensemble`, `<base>/models`.

**Type selection guide**:
- Plain forecasting → **univariate**
- Correlated channels (OHLC, sensor arrays) → **multivariate**
- Forecast conditioned on observed exog (weather observed up to now, observed promo flags) → **covariates/past**
- Forecast conditioned on planned exog (price you'll set tomorrow, scheduled event) → **covariates/future**
- Both observed AND planned exog → **covariates** (chronos-2 only)
- Need raw sample paths for joint risk / scenario simulation → **samples**

## API — univariate (`/v1/timeseries/univariate/forecast`)

The smallest and most common shape: one or more 1D series, return quantile forecasts.

### Request

```json
{
  "model": "chronos-2",
  "context": [[1.0, 2.0, 3.0, ...], [10.0, 11.0, ...]],
  "config": {
    "horizon": 24,
    "quantileLevels": [0.1, 0.5, 0.9],
    "contextLength": 2048,
    "extra": {}
  },
  "unload": false
}
```

| Field | Required | Default | Notes |
|---|---|---|---|
| `model` | yes | — | One of the slugs above. Unknown → 404; not-a-univariate-member → 400. |
| `context` | yes | — | `List[List[float]]`. One inner list per series. Single-series = `[[...]]`. Variable-length series are zero/edge-padded per model. |
| `config.horizon` | yes | — | Steps into the future to forecast. Per-model upper bounds (see Per-model quirks). |
| `config.quantileLevels` | no | `[0.1, 0.5, 0.9]` | Subset of `{0.1, 0.2, ..., 0.9}`. |
| `config.contextLength` | no | per-model (2048 / 2048 / 4000 / 4096 / 2880) | Max history points to feed the model. Longer inputs sliced to the last N. |
| `config.extra` | no | `null` | Per-backend escape hatch — `dict[str, Any]`. Forwarded to the backend adapter; unknown keys silently dropped (forward-compat). Today's adapters mostly no-op; concrete keys land per backend over time. |
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

`median` is always present — best-guess point forecast. For most backends (Moirai-2, Toto-1, Sundial), `quantiles["0.5"]` and `median` are the same value (both derived from sample paths). **Chronos-2 is the exception**: its `median` is the mean of the predictive distribution returned by `predict_quantiles`, not the strict 50th percentile, so `median` and `quantiles["0.5"]` can differ for asymmetric distributions.

### Ensemble — `POST /v1/timeseries/univariate/forecast/ensemble`

Run every member of the type in parallel and combine into a weighted-mean forecast. Returns the ensemble + every contributing model's individual forecast, so you can inspect dissent or post-process.

```bash
curl -s http://localhost:8080/v1/timeseries/univariate/forecast/ensemble \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "context": [[1.0, 2.0, 3.0, 4.0, 5.0]],
    "config": {"horizon": 5},
    "weights": {"chronos-2": 2.0, "moirai-2": 1.0, "timesfm-2.5": 0},
    "memberOverrides": {
      "chronos-2":   {"contextLength": 512, "extra": {"batch_size": 8}},
      "moirai-2":    {"extra": {"normalize_inputs": true}}
    }
  }' | jq
```

Request shape — same as the per-model `/v1/timeseries/univariate/forecast` minus `model`, plus:

| Field | Type | Default | Notes |
|---|---|---|---|
| `weights` | `{slug: float}` | uniform 1.0 | Non-negative per-model weights restricted to type members. Normalized internally. Weight `0` skips that model entirely (not called). Unknown slug for this type → 400. Omitted entry → weight 1.0. |
| `memberOverrides` | `{slug: {field: value}}` | `null` | Per-member shadow of the global `config`. Any key present overrides the global value for THAT member only. Use to give different members different `contextLength`, `extra`, etc. in a single call. Unknown slugs in the override map are silently ignored. |

Response adds three fields on top of the standard per-type forecast shape:

| Field | What it is |
|---|---|
| `ensembleMembers` | List of slugs that actually ran (weight > 0). |
| `weights` | The **normalized** weight per model that was applied to the average. |
| `individual` | `{slug: full forecast result}` map — each entry has the full per-model response, **plus `weight`** (mirrors the top-level `weights[slug]`). |

Failure of any one **included** model fails the whole call. Same ensemble pattern applies to every other type — `/v1/timeseries/multivariate/forecast/ensemble`, `/v1/timeseries/covariates/past/forecast/ensemble`, etc.

### Per-model quirks

- **chronos-2** — native arbitrary-quantile output. No restrictions beyond `{0.1, ..., 0.9}`. Returns `list[Tensor]` (one per series, multivariate-shaped); we squeeze the channel axis for univariate output.
- **timesfm-2.5** — compile-time `max_horizon` (default 512, must be multiple of 128). Request horizon > max → 400. Bump via `PREDICTALOT_TIMESFM_MAX_HORIZON` and restart. We bypass the library's built-in padding (mask=True path produces NaN at this commit) and edge-pad short inputs ourselves.
- **moirai-2** — fixed 9-quantile output `{0.1..0.9}`; we filter to your requested subset. Wrapper context/horizon are baked at model-load time (`PREDICTALOT_MOIRAI_MAX_CONTEXT` / `_MAX_HORIZON`). Per-request inputs are zero-padded to the wrapper's context length with a `past_is_pad` mask.
- **toto-1** — multivariate-native. Univariate calls run as a single-channel series. Quantiles via Monte-Carlo sampling (256 draws → empirical percentiles); the same draws drive `/v1/timeseries/samples/forecast` when called via that type. Returns `[batch, channels, horizon]` shape; we squeeze leading dims for univariate output.
- **sundial-base-128m** — runs in its own sidecar process. From the API surface it's identical to the others. Generative sampling (`num_samples=64` by default, tune via `PREDICTALOT_SUNDIAL_NUM_SAMPLES`). First request waits for the sidecar to be reachable on its unix socket — usually <2s after container start.

## API — multivariate (`/v1/timeseries/multivariate/forecast`)

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
| `config` | Same `horizon`/`quantileLevels`/`contextLength`/`extra` shape as univariate. |

> **Moirai-2 multivariate caveat** — the upstream multivariate path is not exercised by Salesforce's own benchmark suite. We've validated output shapes but not numerical accuracy against a known-good baseline. For high-stakes multivariate workloads prefer `chronos-2` or `toto-1` until upstream ships a multivariate-specific eval.

### Response

`median` and each `quantiles[level]` are shaped `[series][channel][time]`.

Ensemble: `POST /v1/timeseries/multivariate/forecast/ensemble` — same `weights` + `memberOverrides` semantics as univariate, restricted to multivariate members.

## API — covariates: past only (`/v1/timeseries/covariates/past/forecast`)

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

Response shape matches univariate. Ensemble + `memberOverrides` available at `…/forecast/ensemble`.

## API — covariates: future only (`/v1/timeseries/covariates/future/forecast`)

Forecast a univariate target conditioned on covariates known *only* over the future window (e.g. a planned price, a scheduled promotion, a weather forecast). The covariate value array per series must have length `horizon`.

```json
{
  "model": "chronos-2",
  "context": [[1.0, 2.0, 3.0, 4.0]],
  "futureCovariates": [{"price": [9.5, 9.6, 9.7]}],
  "config": {"horizon": 3}
}
```

Only `chronos-2` implements this type. Response shape matches univariate. The per-type ensemble endpoint exists for API symmetry — with a single member it's a degenerate single-result wrapper.

## API — covariates: past + future (`/v1/timeseries/covariates/forecast`)

Forecast a univariate target with covariates that are observed up to *now* AND known into the future (e.g. `price`: known historical + planned future).

```json
{
  "model": "chronos-2",
  "context": [[1.0, 2.0, 3.0, 4.0]],
  "pastCovariates":   [{"price": [9.0, 9.0, 9.5, 9.5]}],
  "futureCovariates": [{"price": [9.5, 9.6, 9.7]}],
  "config": {"horizon": 3}
}
```

**Constraint inherited from chronos-2:** every covariate name appearing in `futureCovariates` MUST also appear in `pastCovariates` for the same series. The backend rejects future-only names.

Only `chronos-2` implements this type. Response shape matches univariate.

## API — samples (`/v1/timeseries/samples/forecast`)

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
| `config.extra` | Per-backend hatch (same shape as the other types). |
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

### Samples ensemble (`/v1/timeseries/samples/forecast/ensemble`)

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

Every type advertises its members at `GET /v1/timeseries/<type>/models`. Same bearer-token requirement as the forecast endpoints (the listing reveals which models are loaded and when they were last used). Returns the type slug + per-member runtime state.

```bash
curl -s -H "Authorization: Bearer changeme" \
    http://localhost:8080/v1/timeseries/univariate/models | jq
```

```json
{
  "type": "univariate",
  "models": [
    {"slug": "chronos-2", "loaded": false, "lastUsedSecsAgo": null, "idleTimeoutSecs": 1800.0},
    {"slug": "timesfm-2.5",       "loaded": false, "lastUsedSecsAgo": null, "idleTimeoutSecs": 1800.0},
    {"slug": "moirai-2",          "loaded": true,  "lastUsedSecsAgo": 4.1,  "idleTimeoutSecs": 1800.0},
    {"slug": "toto-1",            "loaded": false, "lastUsedSecsAgo": null, "idleTimeoutSecs": 1800.0},
    {"slug": "sundial-base-128m", "loaded": false, "lastUsedSecsAgo": null, "idleTimeoutSecs": 1800.0}
  ]
}
```

A model that supports multiple types appears in every relevant listing (e.g. `chronos-2` shows up in all six). The `loaded` / `lastUsedSecsAgo` fields are shared across listings — there's one backend per slug, not one per type.

## Ensemble recipes

Drawn from the [accuracy benchmarks](accuracy.md):

- **"I don't know which model fits"** default — uniform ensemble minus timesfm-2.5 (consistently the weakest in our tests):
  ```json
  {"weights": {"chronos-2": 1, "timesfm-2.5": 0, "moirai-2": 1, "toto-1": 1, "sundial-base-128m": 1}}
  ```
- **Cyclic / seasonal data** — bias moirai-2 + chronos-2:
  ```json
  {"weights": {"chronos-2": 1, "moirai-2": 2}}
  ```
- **Noisy / financial / trendy** — bias toto-1 + sundial:
  ```json
  {"weights": {"toto-1": 2, "sundial-base-128m": 2, "chronos-2": 1}}
  ```
- **Raw path scenario analysis** — sample ensemble:
  ```json
  {"weights": {"toto-1": 1, "sundial-base-128m": 1}, "config": {"numSamples": 128}}
  ```
