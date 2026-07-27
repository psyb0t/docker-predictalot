# Changelog

All notable changes per release. Versions follow [semver](https://semver.org).
Pre-1.0 minor bumps could include breaking REST changes (called out
explicitly). From v1.0.0 onward the public API surface is stable and any
breaking change requires a major bump.

## v1.1.8 — 2026-07-27

README fix. Docs only, no behavior change.

- The Codex subsection of `## Agent integrations` was missing its install command. Added `codex plugin add predictalot@psyb0t` right after the marketplace-add step.
- Clarified that the skill's invocation form depends on how it was picked up: installed via the marketplace it invokes as `$predictalot:predictalot`; picked up automatically from a repo's own `.agents/skills/` (no install needed) it invokes as plain `$predictalot`.

## v1.1.7 — 2026-07-27

Agent-integration manifests. Docs only, no behavior change.

- Added `.agents/.codex-plugin/plugin.json` and `.agents/.claude-plugin/plugin.json` — metadata-only manifests that make the existing `.agents/skills/predictalot` skill installable natively from Claude Code and Codex via the shared `psyb0t/agents` marketplace.
- Added a `## Agent integrations` README section with copy-pasteable install commands for Claude Code, Codex, and OpenClaw (including the `@psyb0t/predictalot` MCP-bridge plugin).

## v1.1.6 — 2026-07-27

README badges. Docs/CI only, no behavior change.

- Added a GitHub Actions CI status badge to the README.

## v1.1.5 — 2026-07-27

README badges. Docs/CI only, no behavior change.

- Added self-hosted version and license badges plus a Docker Hub pulls badge; wired a badges job into pipeline.yml.

## v1.1.4 — 2026-07-26

Listed on the official MCP Registry — no behavior change.

- Added `server.json` — published to the official Model Context Protocol Registry (`registry.modelcontextprotocol.io`) as `io.github.psyb0t/predictalot`, pointing at the `psyb0t/predictalot` Docker image. Ownership is proven by an `io.modelcontextprotocol.server.name` LABEL on the image; publishing runs on tag pushes via GitHub OIDC (secretless). Also added a `glama.json` maintainer claim.

## v1.1.3 — 2026-07-26

License-note correction. Documentation only, no behavior change.

- Corrected the README license line: the project's code is WTFPL (see `LICENSE`); the MCP plugin under `.agents/plugins/predictalot/` is MIT.

## v1.1.2 — 2026-07-26

Skill security-documentation hardening. Docs only, no behavior change.

- Added an explicit "External transmission" bullet to the skill's Security
  & safety section: forecast/train calls send your time series, feature
  values, and/or `modelId`s to whatever `PREDICTALOT_URL` points at, so
  that data leaves your host — point it only at a service you run or trust,
  prefer HTTPS. The data-transfer notes already inline throughout the
  skill are now also summarized in one place up top.

## v1.1.1 — 2026-07-25

Skill security hardening for the ClawHub SkillSpector scan. Docs only, no API change.

- Declared a `permissions:` block (network to the configured `PREDICTALOT_URL`, shell) in the skill frontmatter.
- Data-transfer warnings on the forecast/tabular calls — they transmit your series / features / model IDs to `PREDICTALOT_URL`, so point it at a trusted self-hosted instance, prefer HTTPS, and never echo the bearer.
- Loopback bind (`-p 127.0.0.1:8080:8080`) in the CUDA quick-install example to match the CPU one.

## v1.1.0 — 2026-07-25

ClawHub plugin + skill security pass. No REST API or service change.

- **New `@psyb0t/predictalot` code plugin** (`.agents/plugins/predictalot/`) — a stdio↔HTTP MCP bridge (`mcp-remote`) to the box's `/mcp` endpoint. MIT-licensed. CI publishes it alongside the skill via `clawhub-publish.yml`.
- **Skill hardening**: added a Security & safety section, marked `DELETE /v1/tabular/models/{modelId}` as destructive/irreversible with an explicit-confirmation rule, and replaced the weak example credential (`changeme`) with a generated-secret + loopback-by-default install pattern in `references/setup.md`.

## v1.0.1 — 2026-06-14

Docs-only patch release. No code, no API behavior, no dependency, no image
size change. Repackages the v1.0.0 README into a focused overview + a
`docs/*.md` tree.

### Changed

- `README.md` slimmed to a project pitch + quick-start + links to
  `docs/*.md`. Previously a single 650-line wall.
- `docs/timeseries.md` (new) — full FM API: 5 models, 6 types, per-model
  quirks, ensemble (`weights` + `memberOverrides`), `/models` listings.
  Adds **"Recommended for"** guidance per model + per type.
- `docs/tabular.md` (new) — full tabular API: 9 backends, 3 modes,
  tier-1/2/3 config knobs, 3 meta-learners (calibrated / stacking /
  diversified), storage layout. Adds **"Recommended for"** guidance per
  backend.
- `docs/mcp.md` (new) — MCP tool naming + per-type matrix.
- `docs/configuration.md` (new) — every `PREDICTALOT_*` env var + sample
  config recipes.
- `docs/architecture.md` (new) — sidecar pattern, multi-venv rationale,
  CPU vs CUDA images, lazy-load tabular backends, healthcheck.
- `docs/accuracy.md` (new) — benchmark sMAPE + latency tables + honest
  takeaways. Same data as v1.0.0; clearer surface.
- `docs/errors.md` (new) — error contract + common 400 causes per surface.

No source files changed.

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
