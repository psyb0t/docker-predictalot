# Configuration

All runtime configuration is via `PREDICTALOT_*` env vars. Set them via `docker run -e`, docker-compose `environment:`, k8s ConfigMap, or your orchestrator of choice.

| Var | Default | What it does |
|---|---|---|
| `PREDICTALOT_HOST` | `0.0.0.0` | uvicorn bind host. |
| `PREDICTALOT_PORT` | `8080` | uvicorn bind port. |
| `PREDICTALOT_AUTH_TOKENS` | (empty) | Comma-separated bearer tokens. Empty = refused at startup unless `PREDICTALOT_ALLOW_NO_AUTH=1`. |
| `PREDICTALOT_ALLOW_NO_AUTH` | `0` | Required to start with empty token list. |
| `PREDICTALOT_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `cuda:N`. |
| `PREDICTALOT_MODEL_DIR` | `/models` | Where snapshot directories land. **Bind-mount this** to persist across restarts (also where tabular models live: `/models/tabular/<id>/`). |
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
| `PREDICTALOT_LOG_LEVEL` | `INFO` | Standard Python log levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

## Sensible defaults

For a single-tenant dev box:
```bash
docker run -d --name predictalot \
  -v $HOME/predictalot-models:/models \
  -e PREDICTALOT_AUTH_TOKENS=changeme \
  -e PREDICTALOT_PRELOAD=chronos-2,toto-1 \
  -p 8080:8080 \
  psyb0t/predictalot:latest
```

For prod (CUDA host):
```bash
docker run -d --name predictalot --gpus all \
  -v /srv/predictalot-models:/models \
  -e PREDICTALOT_AUTH_TOKENS=$(cat /etc/predictalot/tokens) \
  -e PREDICTALOT_DEVICE=cuda \
  -e PREDICTALOT_PRELOAD=chronos-2,toto-1,sundial-base-128m \
  -e PREDICTALOT_MODEL_IDLE_TIMEOUT=2h \
  -e PREDICTALOT_MAX_BODY_SIZE=64mb \
  -p 8080:8080 \
  psyb0t/predictalot:latest-cuda
```

For zero-auth local experiments:
```bash
docker run --rm -p 8080:8080 \
  -e PREDICTALOT_ALLOW_NO_AUTH=1 \
  psyb0t/predictalot:latest
```
