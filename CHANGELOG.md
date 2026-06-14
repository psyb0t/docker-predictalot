# Changelog

All notable changes per release. Versions follow [semver](https://semver.org).
Pre-1.0 minor bumps could include breaking REST changes (called out
explicitly). From v1.0.0 onward the public API surface is stable and any
breaking change requires a major bump.

## v1.0.0 — 2026-06-14

API stabilization release. Adds a second model family (tabular ML) alongside
the existing foundation time-series stack, layers per-call escape hatches on
the FM side, and reorganizes the FM URL prefix under `/v1/timeseries/`.

### Breaking

- **REST prefix rename.** All FM forecast / ensemble / models endpoints move
  from `/v1/<type>/…` to `/v1/timeseries/<type>/…`. No redirect compatibility
  layer ships — callers must update URLs.
  - `/v1/univariate/forecast` → `/v1/timeseries/univariate/forecast`
  - `/v1/multivariate/forecast` → `/v1/timeseries/multivariate/forecast`
  - `/v1/covariates/past/forecast` → `/v1/timeseries/covariates/past/forecast`
  - `/v1/covariates/future/forecast` → `/v1/timeseries/covariates/future/forecast`
  - `/v1/covariates/forecast` → `/v1/timeseries/covariates/forecast`
  - `/v1/samples/forecast` → `/v1/timeseries/samples/forecast`
  - `…/forecast/ensemble` and `…/models` move identically.
  - Old paths return 404. This frees `/v1/tabular/` as a sibling family
    and makes future model families equally easy to slot in (`/v1/<family>/`).

### Added — tabular ML surface (`/v1/tabular/`)

- 9 backend slugs across 7 algorithm families:
  - boosting: `lightgbm`, `xgboost`, `hist-gbt`
  - bagging: `random-forest`
  - linear: `logistic` (classifier + Ridge + QuantileRegressor)
  - neural: `mlp`
  - kernel: `svm-rbf`
  - distance: `knn`
  - independence: `naive-bayes` (Gaussian NB + BayesianRidge)
- Three forecast modes per backend: `direction`, `value`, `quantile`.
- `POST /v1/tabular/train` — fit a backend on labeled series, persist by
  caller-chosen `modelId`. Stored under `/models/tabular/<id>/` (one
  metadata JSON + one binary blob). Supports per-row `sampleWeight`,
  `categoricalFeatures`, `monotonicConstraints`, `classWeight`,
  `earlyStoppingRounds` / `validationFraction`, and a per-backend `extra`
  escape-hatch dict.
- `POST /v1/tabular/forecast` — predict on the LATEST row of the supplied
  feature snapshot using a previously-trained model.
- `POST /v1/tabular/forecast/ensemble` — combine multiple stored models on
  the same features with per-member weights (same wire semantics as the FM
  ensembles).
- `GET /v1/tabular/backends` — lists registered backends with their
  `category`, `displayName`, and `supportedModes`.
- `GET /v1/tabular/models` — lists stored model metadata.
- `DELETE /v1/tabular/models/{id}` — removes a stored model.
- Tabular backend modules are **lazy-loaded** — `predictalot.models` imports
  with only the FM stack in scope. The first lookup of a tabular slug
  triggers `importlib.import_module()`, so dev images that don't ship the
  heavy ML wheels can still import the package for unrelated work.

### Added — tabular meta-learners

Three composite endpoints that train + persist as one atomic operation, each
with a matching forecast endpoint:

- `POST /v1/tabular/train/calibrated` (+ `/forecast/calibrated`) — base
  learner + post-hoc Platt-sigmoid or isotonic calibrator fit on a held-out
  TIME-ORDERED tail. Direction-mode only; produces well-calibrated
  probabilities (so "model says 0.7" actually means ~70% historical hit).
- `POST /v1/tabular/train/stacking` (+ `/forecast/stacking`) — K base
  learners + a meta-learner fit on K-fold out-of-fold predictions of the
  bases. Direction-mode v1.
- `POST /v1/tabular/train/diversified` (+ `/forecast/diversified`) — train
  K candidates, score each on OOF performance, greedily select a subset
  whose pairwise OOF correlation stays below `maxPairwiseCorr`, equal-weight
  the survivors. Supports all three modes.

### Added — FM per-call escape hatches

- `ForecastConfig.extra` / `SamplesForecastConfig.extra` (`dict[str, Any] | null`):
  forwarded to the underlying FM backend's `predict_*` adapter for
  per-backend kwargs that don't fit a cross-cutting schema. Backends drop
  keys they don't understand (forward-compat). Today's adapters mostly
  no-op the field; concrete keys land per backend over time.
- Every FM ensemble request (univariate / multivariate / covariates / past /
  future / both / samples) accepts `memberOverrides: {slug → partial-config}`.
  Each key in a member's override map shadows the corresponding key in the
  global `config` for that member ONLY. Use to give different ensemble
  members different `contextLength`, `extra` knobs, etc. in a single call.
  Unknown slugs in the override map are silently ignored.

### Added — tier-2 cross-backend tabular config

Five new optional config fields on the train request (each backend uses what
applies, ignores the rest):

- `categoricalFeatures: list[str] | null` — feature names to mark
  categorical. GBTs use specialized split logic; other backends ignore.
- `monotonicConstraints: dict[str, int] | null` — `{featureName: -1|0|+1}`
  monotonicity direction per feature (GBTs honor; others ignore).
- `classWeight: "balanced" | dict | null` — for imbalanced classifiers.
- `sampleWeight: list[float] | null` — per-row training weight, pruned
  alongside warmup rows.
- `earlyStoppingRounds`, `validationFraction` — GBT early-stopping patience
  and validation holdout fraction.

### Added — tests

- 41 unit tests covering all 9 tabular backends across all 3 modes (in
  `tests/test_tabular_backends.py`), gated on the heavy ML libs being
  importable (skipped in the dev image).
- 13 unit tests for the meta-endpoint router in `tests/test_tabular_meta.py`.
- 6 unit tests verifying `extra` and `memberOverrides` propagate through
  every FM dispatch path in `tests/test_fm_extra_passing.py`.
- 11 unit tests for the path-rename in `tests/test_timeseries_paths.py`.
- 41 real-container integration tests in `tests/integration/`:
  - `test_tabular_live.py`: 30 tests (9 backends × 3 modes + ensemble).
  - `test_tabular_meta_live.py`: 6 tests for the three meta endpoints.
  - `test_fm_extra_live.py`: 5 tests verifying `extra` /
    `memberOverrides` reach the live container.

### Docs

- README rewritten: tabular surface section added before Error Contract,
  every FM URL example updated to the new `/v1/timeseries/` prefix,
  `extra` and `memberOverrides` documented on the univariate config /
  ensemble tables.

## v0.2.1 — earlier

Security patch. Closes a usage-pattern info leak: with auth configured, the
per-type `GET /v1/<type>/models` endpoints shipped in v0.2.0 were missing
the `Depends(check_bearer)` guard and would return the installed model list,
loaded state, and last-used timestamps to unauthenticated callers. Now
bearer-protected. `/healthz` stays open. Open-auth deployments
(`PREDICTALOT_ALLOW_NO_AUTH=1` + empty token list) are unaffected.

## v0.2.0 — earlier

Type-routed API.

**Breaking.** `/v1/forecast` and `/v1/models` are removed and replaced by
six per-type endpoint triples under `/v1/<type>/`. MCP rewritten to 26
per-(type, model) tools. Adds hardening around streaming body size,
NaN/Infinity weights, jagged multivariate channels, covariate shape, and
non-ASCII bearer-token comparison. Lockfile fallback removed from prod
Dockerfiles. Lint baseline (`.flake8`) aligned with black.

## v0.1.1 — earlier

CPU image is amd64-only (no aarch64 torch wheel at the pinned version).

## v0.1.0 — earlier

Initial release: 5 foundation forecasters (chronos-2, timesfm-2.5, moirai-2,
toto-1, sundial-base-128m) + ensemble + sidecar pattern for sundial's
incompatible transformers pin.
