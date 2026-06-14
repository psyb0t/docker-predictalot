# Architecture

## Multi-venv sidecar pattern

Four of the five FM models (`chronos-2`, `timesfm-2.5`, `moirai-2`, `toto-1`) live in the **main Python venv** at `/opt/venv`. They share `torch==2.4.1`, `transformers==4.57.6`, etc. — a single resolved dependency tree.

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

## Where each family lives

| Surface | Process | Venv | Deps |
|---|---|---|---|
| `/v1/timeseries/{univariate,multivariate,covariates*}` | main uvicorn | `/opt/venv` | torch, chronos-forecasting, timesfm, uni2ts (moirai), toto-ts |
| `/v1/timeseries/samples` (chronos / toto / moirai paths) | main uvicorn | `/opt/venv` | same |
| `/v1/timeseries/samples` (sundial path) | main uvicorn → sundial worker | bridge venv → `/opt/sundial-venv` | sundial pinned transformers 4.40 |
| `/v1/tabular/*` | main uvicorn | `/opt/venv` | lightgbm, xgboost, scikit-learn (lazy-imported) |
| `/mcp` | main uvicorn | `/opt/venv` | fastmcp |

## Tabular backend lazy loading

Tabular backends are **lazy-imported** at first `/v1/tabular/forecast` (or `/train`) call:

- `predictalot.models.__init__` only eagerly imports the 5 FM modules. The 9 tabular backend modules are referenced by string in `_TABULAR_MODULE_NAMES`.
- On `get_tabular_backend(slug)`, `importlib.import_module()` imports the backend on demand and caches it for the process lifetime.
- This means a dev image that doesn't ship lightgbm / xgboost / sklearn can still `import predictalot.models` for unrelated tests. The production image installs the heavy ML stack and pays no extra cost.

## CPU vs CUDA images

| Image | Tag | Platforms | Notes |
|---|---|---|---|
| CPU | `psyb0t/predictalot:latest` | amd64 only | PyTorch CPU wheels (pytorch.org's CPU index has no manylinux aarch64 wheel at the pinned `torch==2.4.1+cpu`). |
| CUDA | `psyb0t/predictalot:latest-cuda` | amd64 only | PyTorch CUDA 12.4 wheels on CUDA 12.6 runtime base. Needs `--gpus all` + NVIDIA driver on host. CUDA on arm64 is a different stack (Jetson L4T / SBSA) and not on the menu. |

Both images are self-sufficient — same source, same API, same env vars. Pick the one that matches your host. The CUDA image also runs on CPU if `--gpus` isn't passed (useful for debugging).

## Multi-stage build

- **Builder stage** — installs uv, syncs lightweight deps from `uv.lock` (frozen), installs the heavy ML stack from hash-locked `requirements-{cpu,cuda}.txt`, builds the sundial sidecar venv. Build-time only.
- **Runtime stage** — copies `/opt/venv` + `/opt/sundial-venv` + `src/` + entrypoint. No build tools, no source-only deps, no scratch directories. Strips ~1GB from the final image.

The hash-locked `requirements-{cpu,cuda}.txt` files are generated by `scripts/compile-deps.sh` and committed to the repo. The Docker build refuses to re-resolve at install time (`--require-hashes`) — every byte that lands in the image is verified against a committed hash. Pair this with the `[tool.uv] exclude-newer` age-gate in `pyproject.toml` and you get: pin-by-hash + age-gate at lockfile-generation time + hash-verify at install time. A registry that swaps the bytes under an existing version number fails the install, not the runtime.

## Healthcheck

`/healthz` is the only un-authenticated endpoint. Used by:
- Docker `HEALTHCHECK` directive
- Kubernetes liveness / readiness probes
- The integration test harness (waits for /healthz before proceeding)

Returns `{"status": "ok"}` once uvicorn has booted; doesn't say anything about which backends are loaded (use the per-type `/models` listings for that).
