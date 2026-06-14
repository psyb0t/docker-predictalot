# MCP — `/mcp`

Streamable-HTTP MCP server. Same auth as HTTP (bearer header or `?apiToken=...` query). Tools are organized by FM forecast type — one named tool per (type, model) cell, plus a per-type ensemble + listing tool. MCP currently surfaces the **FM timeseries side only**; the tabular endpoints are HTTP-only.

## Tool naming

- `forecast_<type>_<model>` — single-model forecast. Examples: `forecast_univariate_chronos_2`, `forecast_multivariate_moirai_2`, `forecast_samples_toto_1`.
- `forecast_<type>_ensemble` — per-type weighted ensemble. One per type (6 total).
- `list_<type>_models` — per-type runtime listing. One per type (6 total).

Model slug dashes/dots become underscores in the tool name (`sundial-base-128m` → `sundial_base_128m`, `timesfm-2.5` → `timesfm_2_5`).

## Per-type tool matrix

| Type | Per-model tools | Ensemble | List |
|---|---|---|---|
| univariate | `forecast_univariate_chronos_2`, `..._timesfm_2_5`, `..._moirai_2`, `..._toto_1`, `..._sundial_base_128m` | `forecast_univariate_ensemble` | `list_univariate_models` |
| multivariate | `forecast_multivariate_chronos_2`, `..._moirai_2`, `..._toto_1` | `forecast_multivariate_ensemble` | `list_multivariate_models` |
| covariates-past | `forecast_covariates_past_chronos_2`, `..._moirai_2` | `forecast_covariates_past_ensemble` | `list_covariates_past_models` |
| covariates-future | `forecast_covariates_future_chronos_2` | `forecast_covariates_future_ensemble` | `list_covariates_future_models` |
| covariates (past + future) | `forecast_covariates_both_chronos_2` | `forecast_covariates_both_ensemble` | `list_covariates_both_models` |
| samples | `forecast_samples_toto_1`, `forecast_samples_sundial_base_128m` | `forecast_samples_ensemble` | `list_samples_models` |

## Args per tool

Args mirror the HTTP body, flattened to keyword arguments:

- **quantile types** (univariate / multivariate / all covariates variants):
  ```
  context, horizon, quantile_levels=None, context_length=None, unload=False
  ```
  (covariate variants add `past_covariates` and/or `future_covariates`)
- **samples**:
  ```
  context, horizon, num_samples=None, context_length=None, unload=False
  ```
- **every `forecast_<type>_ensemble`**: same args as the per-model tool minus `model`, plus `weights: dict[str, float] | None = None`.

> Named per-(type, model) tools (not one polymorphic `forecast(type=..., model=...)`) — LLM agents discover and pick named capabilities more reliably than they pick a pair of enum arguments.

## Why no MCP for tabular yet?

The tabular surface is stateful: you train under a `modelId` then forecast against it later. MCP's tool-discovery model assumes stateless tools, so tabular fits awkwardly until we ship either:

1. Session-scoped MCP state, or
2. A "submit features and a backend slug, get an inline one-shot direction/value/quantile prediction" composite endpoint.

Until then, point your MCP-enabled clients at the FM tools and use plain HTTP for tabular.
