# predictalot

> One HTTP service, two model families, zero ceremony.

- **Foundation time-series** — 5 zero-shot forecasters (chronos-2, timesfm-2.5, moirai-2, toto-1, sundial-base-128m). Hand them a context window, get quantile or sample-path forecasts. No training step. Six modality-specific endpoints under `/v1/timeseries/<type>/`.
- **Tabular ML** — 9 supervised learners (lightgbm, xgboost, hist-gbt, random-forest, logistic, mlp, svm-rbf, knn, naive-bayes) + 3 meta-learners (calibrated, stacking, diversified). Train on YOUR engineered features, persist server-side by `modelId`, forecast on the latest snapshot. Under `/v1/tabular/`.
- **MCP** — streamable-HTTP tools at `/mcp`. One named tool per (FM type, model) cell plus per-type ensemble + listing. Tabular endpoints are HTTP-only for now.

## Quick start

```bash
docker run -d --name predictalot \
  -v $HOME/predictalot-models:/models \
  -e PREDICTALOT_AUTH_TOKENS=changeme \
  -p 8080:8080 \
  psyb0t/predictalot:latest

# Zero-shot FM forecast
curl -s http://localhost:8080/v1/timeseries/univariate/forecast \
  -H "Authorization: Bearer changeme" -H "Content-Type: application/json" \
  -d '{"model":"chronos-2","context":[[10,11,12,13,14,15,16,17,18,19,20]],"config":{"horizon":5}}' | jq

# Train + persist a tabular model on your own features
curl -s http://localhost:8080/v1/tabular/train \
  -H "Authorization: Bearer changeme" -H "Content-Type: application/json" \
  -d '{"modelId":"my-model","backend":"lightgbm","target":[[100,101,99,...]],
       "features":[{"rsi":[55,58,...],"macd":[0.3,0.4,...]}],
       "config":{"mode":"direction","horizon":3,"nEstimators":400}}' | jq

# Then forecast on the latest snapshot
curl -s http://localhost:8080/v1/tabular/forecast \
  -H "Authorization: Bearer changeme" -H "Content-Type: application/json" \
  -d '{"modelId":"my-model","features":[{"rsi":[58],"macd":[0.4]}]}' | jq
```

## Documentation

| Doc | What it covers |
|---|---|
| [docs/timeseries.md](docs/timeseries.md) | Foundation time-series API. All 5 models (capabilities + per-model quirks + **what each is recommended for**), all 6 forecast types, per-type ensemble with `weights` + `memberOverrides`, `extra` per-call hatch, `/models` listings. |
| [docs/tabular.md](docs/tabular.md) | Tabular ML API. All 9 backends (**what each is recommended for**), 3 modes (direction / value / quantile), tier-1/2/3 config knobs, the 3 meta-learners (calibrated / stacking / diversified), storage layout. |
| [docs/mcp.md](docs/mcp.md) | MCP streamable-HTTP server: tool naming, args, current scope (FM only). |
| [docs/configuration.md](docs/configuration.md) | Every `PREDICTALOT_*` env var. |
| [docs/architecture.md](docs/architecture.md) | Multi-venv sidecar pattern for sundial, CPU vs CUDA images, multi-stage build. |
| [docs/accuracy.md](docs/accuracy.md) | Benchmark sMAPE + latency on academic + real-world datasets. Honest takeaways including which models lose. |
| [docs/errors.md](docs/errors.md) | Error contract: 400 / 401 / 404 / 413 / 422 / 503 shapes. |

[CHANGELOG.md](CHANGELOG.md) tracks per-version changes.

## License

Code: MIT (see `LICENSE`).
Foundation models retain their upstream licenses — chronos-2 / timesfm-2.5 / toto-1 / sundial-base-128m: Apache 2.0; moirai-2: CC-BY-NC-4.0 (non-commercial). Tabular backends use their upstream licenses — lightgbm / xgboost / scikit-learn: permissive. Review each before commercial use.
