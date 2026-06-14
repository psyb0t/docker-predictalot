# Tabular ML API — `/v1/tabular/…`

Supervised tabular learning over **your own engineered features**. Train on a labeled history, persist by caller-chosen `modelId`, forecast on the latest snapshot. Three modes:

- **`direction`** — classification on `sign(target[t+h] - target[t])`. Response carries `probUp` + `confidence` per series.
- **`value`** — regression on `target[t+h]`. Response carries `predicted` per series.
- **`quantile`** — quantile regression at the requested levels. Response carries `median` + `quantiles[level]` per series.

Models persist server-side at `/models/tabular/<modelId>/`. Lifetime survives container restarts — bind-mount `/models` to keep them across replacement.

## The nine backends

| Slug | Category | Algorithm | Quantile mode | Recommended for |
|---|---|---|---|---|
| `lightgbm` | boosting | LightGBM | native (objective="quantile") | **Default starting point.** Fast training, native categorical support via `categoricalFeatures`, honors `monotonicConstraints`, handles missing values. Best out-of-the-box accuracy on most finance-style tabular problems. |
| `xgboost` | boosting | XGBoost | native | Alternative GBT with different default tuning + slightly different tree-construction algorithm. Use as a second opinion to lightgbm, ensemble both if results are close. |
| `hist-gbt` | boosting | sklearn HistGradientBoosting | native | Pure-sklearn alternative if lightgbm / xgboost wheels aren't available or you want one fewer C-extension dep. Native categorical + monotonic support. |
| `random-forest` | bagging | sklearn RandomForest | per-tree empirical quantile | **When you want a different inductive bias than boosting.** RF averages independent trees; GBT chains corrections. Strong when feature interactions matter and the signal-to-noise is moderate. Native per-tree quantile gives you a real distribution. |
| `logistic` | linear | LogisticRegression / Ridge / QuantileRegressor | native | **Simplest baseline. Sanity check.** If logistic is competitive with GBT, your signal is mostly linear and you don't need the heavier models. Cheap to train + interpret. |
| `mlp` | neural | sklearn MLPClassifier / MLPRegressor | residual-quantile band | **Detects nonlinear feature interactions GBT might miss.** If MLP beats GBT, there's nonlinear structure in your features. Needs more data than the linear / GBT alternatives to be useful. |
| `svm-rbf` | kernel | sklearn SVC / SVR with RBF kernel | residual-quantile band | **When you want a smooth decision boundary instead of trees' axis-aligned splits.** Slow on big datasets (O(n²)+); fine for windows up to ~5000 rows. |
| `knn` | distance | k-Nearest Neighbors | per-neighbor empirical quantile | **Diagnostic: "find similar past regimes."** If k-NN matches GBT, your signal might just be "look at recent neighbors." If k-NN tanks, GBT is learning real structure. |
| `naive-bayes` | independence | GaussianNB (direction) + BayesianRidge (value/quantile) | residual band | **Independence baseline.** If naive-bayes matches GBT, your features are independent enough that nonlinear interactions aren't earning their keep. High diagnostic value, never the "best" model on real-world feature sets. |

### Choosing a backend

- Just starting → `lightgbm`
- Have categorical features → `lightgbm` or `xgboost` or `hist-gbt`
- Want monotonic constraints (e.g. "higher RSI → higher prob_up") → `lightgbm` / `xgboost` / `hist-gbt`
- Small dataset (< 500 rows) → `logistic` or `naive-bayes`
- Suspect nonlinear interactions → add `mlp` as a comparison
- Want a non-tree second voice for an ensemble → `random-forest` or `svm-rbf`
- Need calibrated probabilities for position-sizing → wrap any backend in `/v1/tabular/train/calibrated`

## Train + forecast roundtrip

```bash
# 1. Train + persist
curl -s http://localhost:8080/v1/tabular/train \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "modelId": "btc-direction-h3",
    "backend": "lightgbm",
    "target":   [[ 50000, 50500, 50200, ...]],
    "features": [{"rsi": [55, 58, ...], "macdHist": [0.3, 0.4, ...], "ema21Dist": [0.01, 0.02, ...]}],
    "config":   {
      "mode": "direction", "horizon": 3,
      "nEstimators": 400, "learningRate": 0.05, "maxDepth": 6,
      "monotonicConstraints": {"rsi": -1}
    }
  }' | jq

# 2. Forecast on the LATEST bar's features
curl -s http://localhost:8080/v1/tabular/forecast \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "modelId": "btc-direction-h3",
    "features": [{"rsi": [58], "macdHist": [0.4], "ema21Dist": [0.02]}]
  }' | jq
# → {"probUp": [0.67], "confidence": [0.34], ...}

# 3. List + manage stored models
curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/v1/tabular/models | jq
curl -X DELETE -H "Authorization: Bearer $TOKEN" \
     http://localhost:8080/v1/tabular/models/btc-direction-h3
```

## Train config — the three tiers

### Tier 1: required + GBT-shape knobs

| Field | Required | Notes |
|---|---|---|
| `mode` | yes | `"direction"` \| `"value"` \| `"quantile"` |
| `horizon` | yes (`> 0`) | Bars ahead. Label is `target[t+h]` (or its sign for direction). |
| `quantileLevels` | mode=quantile | Subset of `{0.1, 0.2, ..., 0.9}`. |
| `nEstimators` | no | Trees / boosting rounds (GBTs + RF). |
| `maxDepth` | no | Per-tree depth cap (GBTs + RF). |
| `learningRate` | no | Shrinkage per round (GBTs). |
| `numLeaves` | no | LightGBM-style leaf count per tree. |
| `minSamples` | no | Discard anchors with fewer training samples (after warmup pruning). |
| `randomState` | no | Seed for reproducibility. Stochastic where `None`. |

### Tier 2: cross-backend hints (used where supported, ignored elsewhere)

| Field | Notes |
|---|---|
| `categoricalFeatures` | List of feature names. `lightgbm` and `hist-gbt` use specialized split logic for these. `xgboost` sets `enable_categorical=True` but only honors it for pandas-DataFrame-typed `category` inputs — with raw ndarrays today (the only path predictalot exposes) the flag is no-op. Other backends silently ignore. Treating a categorical as numeric is a silent footgun — name the columns here when working with lightgbm/hist-gbt. |
| `monotonicConstraints` | `{featureName: -1\|0\|+1}` direction per feature (GBTs honor; others ignore). Use when you have domain prior knowledge (e.g. `{"rsi": -1}` — higher RSI shouldn't increase bullish prob beyond a point). |
| `classWeight` | `"balanced"` or `{class: weight}` — for imbalanced classifiers. |
| `sampleWeight` | Per-row training weight list (same length as `target[i]`). Pruned alongside warmup rows. Useful for time-decay or volume-weighted training. |
| `earlyStoppingRounds` | GBT patience. Requires `validationFraction > 0`. Ignored by non-iterative backends. |
| `validationFraction` | Holdout fraction for early stopping. |

### Tier 3: per-backend escape hatch

`extra: dict[str, Any] | null` — backend-specific hyperparams that don't fit a cross-cutting schema. Each backend documents the keys it reads from `extra` in its source module.

| Backend | Reads from `extra` |
|---|---|
| `lightgbm` | `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `boosting_type` |
| `xgboost` | `subsample`, `colsample_bytree`, `colsample_bylevel`, `reg_alpha`, `reg_lambda`, `scale_pos_weight`, `grow_policy`, `gamma` |
| `hist-gbt` | `max_iter`, `l2_regularization`, `max_bins`, `min_samples_leaf`, `max_leaf_nodes` |
| `random-forest` | `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`, `oob_score` |
| `logistic` | `C`, `penalty`, `l1_ratio`, `solver`, `alpha`, `quantile_alpha`, `max_iter` |
| `mlp` | `hidden_layer_sizes`, `activation`, `alpha`, `learning_rate_init`, `max_iter`, `early_stopping`, `batch_size` |
| `svm-rbf` | `C`, `gamma`, `kernel`, `tol`, `max_iter` |
| `knn` | `n_neighbors`, `weights`, `metric`, `p` |
| `naive-bayes` | `var_smoothing`, `priors` |

Backends silently ignore unknown `extra` keys, so passing project-wide extras to mixed-backend ensembles is safe.

## Ensemble across stored models

```bash
curl -s http://localhost:8080/v1/tabular/forecast/ensemble \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "modelIds": ["btc-dir-h3-lgbm", "btc-dir-h3-xgb", "btc-dir-h3-rf"],
    "weights":  {"btc-dir-h3-lgbm": 2.0, "btc-dir-h3-xgb": 1.0, "btc-dir-h3-rf": 1.0},
    "features": [{"rsi": [58], "macdHist": [0.4], "ema21Dist": [0.02]}]
  }' | jq
```

Members must agree on `mode`, `horizon`, and `featureNames` — mismatched members get rejected with 400. The combined response carries each individual member's prediction + the weighted-mean combo, same shape as the FM ensembles.

## Meta-learners — train+persist as one operation

Three composite endpoints that orchestrate multi-model training atomically (impossible to do correctly with separate `/train` calls because they need coordinated holdouts).

### `POST /v1/tabular/train/calibrated` (+ `/forecast/calibrated`)

Base learner + post-hoc calibrator on a held-out TIME-ORDERED tail. Direction-mode only.

```json
{
  "modelId": "btc-cal-h3",
  "baseBackend": "lightgbm",
  "target":   [[...]],
  "features": [{...}],
  "config":   {"mode": "direction", "horizon": 3, "nEstimators": 400},
  "calibrationMethod": "sigmoid",
  "calibrationFraction": 0.2
}
```

| Field | Notes |
|---|---|
| `baseBackend` | Any tabular backend slug. The calibrator wraps its `prob_up` output. |
| `calibrationMethod` | `"sigmoid"` (Platt — parametric, smooth) or `"isotonic"` (non-parametric, more flexible). |
| `calibrationFraction` | TAIL fraction of training rows held out for the calibrator. `> 0`, `< 1`. |

**Recommended for:** position sizing by predicted probability. Without calibration, `prob_up = 0.7` from a tree-based classifier may correspond to ~55% real win rate. Calibration fixes this.

### `POST /v1/tabular/train/stacking` (+ `/forecast/stacking`)

K base learners + a meta-learner fit on K-fold OOF predictions. Direction-mode v1.

```json
{
  "modelId": "btc-stk-h3",
  "members": [
    {"backend": "lightgbm", "config": {"mode": "direction", "horizon": 3, "nEstimators": 200}},
    {"backend": "xgboost",  "config": {"mode": "direction", "horizon": 3, "nEstimators": 200}},
    {"backend": "logistic", "config": {"mode": "direction", "horizon": 3}}
  ],
  "metaBackend": "logistic",
  "target": [[...]], "features": [{...}],
  "horizon": 3, "nFolds": 5
}
```

K-fold (default 5, min 2, max 10) generates each base learner's OOF predictions; the meta-learner trains on those OOF columns → target. Bases are then RETRAINED on the full pool for the shippable model.

**Recommended for:** when you have several base learners with DIFFERENT inductive biases (boosting + linear + RF, say) and want a learned weighted combination instead of equal-weight averaging. Less useful when all bases are GBTs — their errors are too correlated.

### `POST /v1/tabular/train/diversified` (+ `/forecast/diversified`)

Candidate pool → score each on OOF → greedily pick a low-pairwise-correlation subset → equal-weight survivors. All three modes.

```json
{
  "modelId": "btc-div-h3",
  "candidates": [
    {"backend": "lightgbm",      "config": {"mode": "direction", "horizon": 3, "nEstimators": 200}},
    {"backend": "xgboost",       "config": {"mode": "direction", "horizon": 3, "nEstimators": 200}},
    {"backend": "logistic",      "config": {"mode": "direction", "horizon": 3}},
    {"backend": "random-forest", "config": {"mode": "direction", "horizon": 3, "nEstimators": 200}},
    {"backend": "naive-bayes",   "config": {"mode": "direction", "horizon": 3}}
  ],
  "target": [[...]], "features": [{...}],
  "horizon": 3, "mode": "direction",
  "nFolds": 3, "maxPairwiseCorr": 0.85,
  "minMembers": 2, "maxMembers": 4
}
```

Selection algorithm:
1. K-fold OOF predictions per candidate.
2. Score each candidate (AUC for direction, negative MAE for value / quantile).
3. Sort by score descending.
4. Add to portfolio in order, **skipping any candidate whose max pairwise OOF correlation with already-selected members exceeds `maxPairwiseCorr`**.
5. Stop at `maxMembers`; if below `minMembers`, pad with best-scoring leftovers regardless of correlation.

Response carries the selection result + the full candidate correlation matrix (so you can sanity-check what got dropped and why).

**Recommended for:** when you want maximum diversity in an ensemble — avoid the "all my models are basically the same" failure mode. The correlation map in the response is useful for understanding which feature representations your candidates actually agree on.

## Backends listing

```bash
curl -s -H "Authorization: Bearer changeme" \
    http://localhost:8080/v1/tabular/backends | jq
```

```json
{
  "backends": [
    {"slug": "hist-gbt",      "displayName": "HistGradientBoosting (sklearn)", "category": "boosting", "supportedModes": ["direction", "quantile", "value"]},
    {"slug": "knn",           "displayName": "k-Nearest Neighbors",            "category": "distance", "supportedModes": ["direction", "quantile", "value"]},
    {"slug": "lightgbm",      "displayName": "LightGBM",                       "category": "boosting", "supportedModes": ["direction", "quantile", "value"]},
    ...
  ]
}
```

`category` lets you group backends in UI / for diversified ensembles ("one boosting + one linear + one neural").

## Storage layout

```
/models/tabular/<modelId>/
  meta.json    # backend, mode, horizon, feature_names, n_training_rows, trained_at_unix
  model.blob   # opaque pickled estimator (or composite blob for meta-learners)
```

`GET /v1/tabular/models` lists every stored model with its metadata. `DELETE /v1/tabular/models/{id}` removes one (404 if absent).

Meta-learner models store their backend tag as `meta:calibrated` / `meta:stacking` / `meta:diversified` so the regular `/forecast` route won't accidentally try to deserialize them — they're served only by their dedicated `/forecast/{kind}` endpoint.
