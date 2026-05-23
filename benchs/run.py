"""Accuracy + latency benchmark for predictalot.

Pulls real public time-series, holds out the tail, hits /v1/univariate/forecast
for each model and times the round-trip. Compares MAE / RMSE / sMAPE against
the held-out actuals + a seasonal-naive baseline.

Usage:
    python benchs/run.py                                  # default localhost:18080
    PREDICTALOT_BENCH_URL=http://gpu-host:18080 python benchs/run.py

Set the bearer token via PREDICTALOT_BENCH_TOKEN (default: devtok).
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
import time
import urllib.request

import httpx


BASE_URL = os.environ.get("PREDICTALOT_BENCH_URL", "http://127.0.0.1:18080")
TOKEN = os.environ.get("PREDICTALOT_BENCH_TOKEN", "devtok")

MODELS = ("chronos-2", "timesfm-2.5", "moirai-2", "toto-1", "sundial-base-128m")


def fetch_csv(url: str, value_col: str) -> list[float]:
    raw = urllib.request.urlopen(url, timeout=30).read().decode()
    reader = csv.DictReader(io.StringIO(raw))
    return [float(row[value_col]) for row in reader if row.get(value_col)]


def fetch_binance_close(symbol: str, interval: str = "1d", limit: int = 1000) -> list[float]:
    """Daily closes from Binance public API (no auth)."""
    url = (
        f"https://api.binance.com/api/v3/klines?symbol={symbol}"
        f"&interval={interval}&limit={limit}"
    )
    raw = urllib.request.urlopen(url, timeout=30).read().decode()
    klines = json.loads(raw)
    # kline format: [openTime, open, high, low, close, volume, closeTime, ...]
    return [float(k[4]) for k in klines]


def fetch_noaa_co2() -> list[float]:
    """Mauna Loa monthly mean CO2, NOAA direct CSV."""
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
    raw = urllib.request.urlopen(url, timeout=30).read().decode()
    # NOAA's CSV has comment lines starting with `#`; strip them, then header row
    lines = [ln for ln in raw.splitlines() if ln and not ln.startswith("#")]
    reader = csv.DictReader(lines)
    # 'average' is the monthly mean ppm; sometimes missing → fall back to 'deseasonalized'
    out: list[float] = []
    for row in reader:
        v = row.get("average") or row.get("deseasonalized")
        if v and float(v) > 0:
            out.append(float(v))
    return out


def metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    errs = [a - p for a, p in zip(actual, predicted)]
    mae = sum(abs(e) for e in errs) / len(errs)
    rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    smape = (
        100.0
        * sum(abs(a - p) / ((abs(a) + abs(p)) / 2 + 1e-9) for a, p in zip(actual, predicted))
        / len(actual)
    )
    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape}


def forecast(client: httpx.Client, model: str, series: list[float], horizon: int) -> tuple[list[float], float]:
    """Returns (median forecast, wall-clock seconds)."""
    t0 = time.perf_counter()
    r = client.post(
        "/v1/univariate/forecast",
        json={"model": model, "context": [series], "config": {"horizon": horizon}},
    )
    r.raise_for_status()
    elapsed = time.perf_counter() - t0
    return r.json()["median"][0], elapsed


def forecast_ensemble(
    client: httpx.Client,
    series: list[float],
    horizon: int,
    weights: dict[str, float] | None = None,
) -> tuple[list[float], float]:
    body: dict = {"context": [series], "config": {"horizon": horizon}}
    if weights is not None:
        body["weights"] = weights
    t0 = time.perf_counter()
    r = client.post("/v1/univariate/forecast/ensemble", json=body)
    r.raise_for_status()
    elapsed = time.perf_counter() - t0
    return r.json()["median"][0], elapsed


def warmup(client: httpx.Client, model: str, series: list[float]) -> float:
    """One-shot small forecast to load weights. Returns load time in seconds."""
    return forecast(client, model, series[:48], 4)[1]


def seasonal_naive(train: list[float], horizon: int, seasonality: int | None) -> list[float]:
    if seasonality and len(train) >= seasonality:
        return [train[-seasonality + (i % seasonality)] for i in range(horizon)]
    return [train[-1]] * horizon


def run_dataset(client: httpx.Client, name: str, series: list[float], horizon: int, seasonality: int | None) -> None:
    print()
    print("=" * 78)
    print(f"{name}  (n={len(series)}, horizon={horizon}, seasonality={seasonality})")
    print("=" * 78)
    train, actual = series[:-horizon], series[-horizon:]
    print(f"  train: {len(train)} pts  |  test: {len(actual)} pts")
    print(f"  train tail: {[round(x, 2) for x in train[-5:]]}")
    print(f"  actual:     {[round(x, 2) for x in actual]}")

    baseline = seasonal_naive(train, horizon, seasonality)
    m = metrics(actual, baseline)
    print(
        f"  {'seasonal-naive':16s} MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}  sMAPE={m['sMAPE']:5.2f}%  (no inference)"
    )

    for model in MODELS:
        try:
            pred, dt = forecast(client, model, train, horizon)
            m = metrics(actual, pred)
            print(
                f"  {model:16s} MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}  sMAPE={m['sMAPE']:5.2f}%  ({dt*1000:6.0f} ms)"
            )
        except Exception as e:  # noqa: BLE001
            print(f"  {model:16s} ERROR: {e}")

    # Labels must match the actual members: an omitted slug defaults to
    # weight 1.0 (per /v1/univariate/forecast/ensemble semantics), so every
    # config that intends to exclude a model must set its weight to 0
    # explicitly. sundial-base-128m is included in this list because it
    # supports univariate; configs that don't mention it would silently pull
    # it in.
    ensemble_configs: list[tuple[str, dict | None]] = [
        ("ens uniform", None),
        (
            "ens no-timesfm",
            {
                "chronos-2": 1,
                "timesfm-2.5": 0,
                "moirai-2": 1,
                "toto-1": 1,
                "sundial-base-128m": 1,
            },
        ),
        (
            "ens no-toto",
            {
                "chronos-2": 1,
                "timesfm-2.5": 1,
                "moirai-2": 1,
                "toto-1": 0,
                "sundial-base-128m": 1,
            },
        ),
        (
            "ens chronos+moirai",
            {
                "chronos-2": 1,
                "timesfm-2.5": 0,
                "moirai-2": 1,
                "toto-1": 0,
                "sundial-base-128m": 0,
            },
        ),
        (
            "ens chronos-heavy",
            {
                "chronos-2": 2,
                "timesfm-2.5": 0.5,
                "moirai-2": 1,
                "toto-1": 0.5,
                "sundial-base-128m": 0.5,
            },
        ),
    ]
    for label, weights in ensemble_configs:
        try:
            pred, dt = forecast_ensemble(client, train, horizon, weights)
            m = metrics(actual, pred)
            print(
                f"  {label:16s} MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}  sMAPE={m['sMAPE']:5.2f}%  ({dt*1000:6.0f} ms)"
            )
        except Exception as e:  # noqa: BLE001
            print(f"  {label:16s} ERROR: {e}")


def main() -> int:
    print(f"benchmarking against {BASE_URL}")
    client = httpx.Client(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {TOKEN}"},
        timeout=600.0,
    )

    # Health check + warmup all models (first call per model downloads/loads weights;
    # we want subsequent benchmarks to measure inference, not load time)
    r = client.get("/healthz")
    r.raise_for_status()
    print(f"  healthz: {r.json()}")

    print("\nwarming up models (one tiny forecast each; first run loads weights):")
    warmup_series = [float(i) for i in range(1, 200)]
    for model in MODELS:
        try:
            dt = warmup(client, model, warmup_series)
            print(f"  {model:16s} warmup: {dt*1000:6.0f} ms")
        except Exception as e:  # noqa: BLE001
            print(f"  {model:16s} ERROR: {e}")

    # AirPassengers — monthly, 1949-1960, clear trend + 12-month seasonality
    air = fetch_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
        value_col="Passengers",
    )
    run_dataset(client, "AirPassengers (monthly, n=144)", air, horizon=24, seasonality=12)

    # Shampoo sales — monthly, 3 years, trend only
    shampoo = fetch_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv",
        value_col="Sales",
    )
    run_dataset(client, "Shampoo Sales (monthly, n=36)", shampoo, horizon=6, seasonality=None)

    # Daily-min-temperatures — Melbourne 1981-1990, noisy + yearly seasonality
    temps = fetch_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        value_col="Temp",
    )
    # Use last 4 years to keep context manageable
    temps = temps[-1460:]
    run_dataset(client, "Daily-Min-Temperatures (Melbourne, last 4y)", temps, horizon=30, seasonality=365)

    # ─── Real-world data — what foundation models do on actual messy series ───

    # CO2 Mauna Loa monthly — strong trend + 12-month seasonality, 800+ months.
    # Expected: all foundation models crush naive (clean signal).
    try:
        co2 = fetch_noaa_co2()
        run_dataset(client, "CO2 Mauna Loa (monthly, n={})".format(len(co2)),
                    co2, horizon=24, seasonality=12)
    except Exception as e:  # noqa: BLE001
        print(f"\n[CO2 dataset skipped: {e}]")

    # Gold daily close — via PAXG/USDT on Binance. PAXG is PAX Gold, an
    # ERC-20 token backed 1:1 by physical gold held in LBMA vaults; it tracks
    # gold spot to within fractions of a percent and is conveniently on
    # Binance's no-auth public API. Expected: foundation models beat naive
    # modestly — gold has macro drift + noise but no strong seasonality.
    try:
        gold = fetch_binance_close("PAXGUSDT", interval="1d", limit=1000)
        run_dataset(
            client,
            f"Gold (PAXG/USDT daily close, last n={len(gold)})",
            gold,
            horizon=30,
            seasonality=None,
        )
    except Exception as e:  # noqa: BLE001
        print(f"\n[Gold dataset skipped: {e}]")

    # BTC/USD daily close — financial random-walk, hardest case.
    # Expected: most models barely beat naive — honest test of low-signal data.
    try:
        btc = fetch_binance_close("BTCUSDT", interval="1d", limit=1000)
        run_dataset(client, f"BTC/USD (daily close, last n={len(btc)})",
                    btc, horizon=30, seasonality=None)
    except Exception as e:  # noqa: BLE001
        print(f"\n[BTC dataset skipped: {e}]")

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
