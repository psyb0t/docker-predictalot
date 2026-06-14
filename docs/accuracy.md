# Accuracy & latency

Numbers from `make bench` against `psyb0t/predictalot-test:cuda` on an RTX 3060. Lower sMAPE = better forecast. Latency is round-trip wall-clock per single-series univariate forecast (after warmup, single-series in the request body).

## Accuracy (sMAPE %)

### Academic benchmarks

| Dataset (horizon) | seasonal-naive | chronos-2 | timesfm-2.5 | moirai-2 | toto-1 | sundial | ens uniform |
|---|---:|---:|---:|---:|---:|---:|---:|
| AirPassengers (h=24, n=144 monthly) | 17.01 | 8.26 | 12.42 | **7.71** | 19.88 | 16.57 | 12.75 |
| Shampoo Sales (h=6, n=36 monthly) | **25.54** | 27.24 | 40.37 | 33.74 | 24.27 | 42.07 | 32.07 |
| Daily-Min-Temperatures (h=30, n=1460 daily) | 17.72 | 13.23 | 13.65 | 13.77 | 13.76 | **12.72** | 13.30 |

### Real-world benchmarks (fetched live by `make bench`)

| Dataset (horizon) | seasonal-naive | chronos-2 | timesfm-2.5 | moirai-2 | toto-1 | sundial | ens uniform |
|---|---:|---:|---:|---:|---:|---:|---:|
| CO2 Mauna Loa (h=24, n=818 monthly) | 1.05 | 0.24 | 0.16 | 0.13 | **0.09** | 0.58 | 0.17 |
| Gold PAXG/USDT (h=30, n=1000 daily) | 2.18 | 3.26 | 2.67 | 3.12 | 3.58 | **1.67** | 2.81 |
| BTC/USDT (h=30, n=1000 daily) | 3.18 | 2.74 | 2.75 | 5.00 | **2.02** | 2.91 | 2.98 |

## Honest takeaways

1. **No single model dominates real-world data.** Each model wins on at least one dataset, and which one depends on the signal shape: moirai on clean seasonal, toto on observability-style noisy series, sundial on financial drift, chronos as a steady all-rounder. **Toto-1 wins on CO2 + BTC + Shampoo** (its training distribution favors trendy/noisy series). **Sundial wins on Gold + Daily Temps**. **Moirai wins on AirPassengers**.

2. **Uniform ensemble loses to the right single model on every real-world dataset.** Averaging with a model that has the wrong inductive bias drags the mean down. Pick `weights` per-domain when you know what fits.

3. **TimesFM 2.5 is the weakest of the five on every dataset we tested** — slowest AND worst or near-worst sMAPE. Consider setting `"weights": {"timesfm-2.5": 0}` in ensemble calls until a newer release ships.

4. **`ens no-timesfm`** (`{"chronos-2": 1, "timesfm-2.5": 0, "moirai-2": 1, "toto-1": 1, "sundial-base-128m": 1}`) is a strong "I don't know which model fits" default.

5. **Foundation models don't beat the market.** Best result on BTC (toto-1 at 2.02% sMAPE vs naive 3.18%) is real but tiny — definitely not "trade on this" predictive.

6. **For comparison:** hand-tuned ARIMA on AirPassengers gets ~3-5% sMAPE. predictalot's best on that dataset is moirai-2 at 7.71% sMAPE — competitive zero-shot, no per-series fitting.

## Latency per single-series univariate forecast

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

## Running the bench yourself

```bash
make bench                                                 # localhost:18080, devtok
PREDICTALOT_BENCH_URL=http://remote:8080 \
PREDICTALOT_BENCH_TOKEN=mytoken make bench                 # remote, custom token
```

`make bench` requires a running container reachable at `PREDICTALOT_BENCH_URL` (default `http://127.0.0.1:18080`) with bearer token in `PREDICTALOT_BENCH_TOKEN` (default `devtok`). Compares each individual model against the seasonal-naive baseline and several weighted-ensemble variants on six datasets (three academic + three real-world public APIs: NOAA CO2, Binance gold/BTC). All bench calls go through `/v1/timeseries/univariate/forecast`.

## Tabular accuracy

No analogous benchmark suite ships in this repo — tabular accuracy is dataset-specific (your engineered features dominate, the backend choice is secondary). Standard practice:

1. Hold out the most recent N% of your labeled series as a never-touched test window.
2. Use walk-forward retraining on the remaining series with `retrainEvery` and a fixed `trainWindow`.
3. Score the surviving setups by edge-vs-base-rate (NOT raw hit rate) with bootstrap confidence intervals and per-fold consistency.

The `lightgbm` / `xgboost` / `random-forest` backends are mature, well-benchmarked algorithms outside this project — their general accuracy characteristics are documented in their upstream repos.
