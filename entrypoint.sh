#!/bin/bash
# predictalot entrypoint — runs as the `predictalot` user (set in Dockerfile).
#
# Responsibilities:
#  1. (optional) prefetch HF snapshots via PREDICTALOT_PREFETCH
#  2. start the sundial sidecar worker as a background process on a unix
#     socket (only if sundial is included in the build — guarded by
#     existence of /opt/sundial-venv). Auto-restart on crash.
#  3. exec the main predictalot service.

set -e

if [ -n "${PREDICTALOT_PREFETCH:-}" ]; then
    echo "[entrypoint] prefetching: ${PREDICTALOT_PREFETCH}"
    IFS=',' read -r -a slugs <<< "${PREDICTALOT_PREFETCH}"
    python -m predictalot.fetch "${slugs[@]}"
fi

# ─── sundial sidecar worker ──────────────────────────────────────────────
# Lives in /opt/sundial-venv with transformers==4.40.1. Listens on a unix
# socket; the main predictalot service talks to it like any other backend.
# An auto-restart loop keeps it alive if it crashes.
SUNDIAL_SOCK_DIR="${PREDICTALOT_SUNDIAL_SOCK_DIR:-/tmp/predictalot}"
SUNDIAL_SOCK="${PREDICTALOT_SUNDIAL_SOCK:-${SUNDIAL_SOCK_DIR}/sundial.sock}"
mkdir -p "${SUNDIAL_SOCK_DIR}"

if [ -x /opt/sundial-venv/bin/uvicorn ] && [ -d /opt/sundial_worker ]; then
    echo "[entrypoint] starting sundial worker on ${SUNDIAL_SOCK}"
    (
        cd /opt
        # PYTHONPATH so 'sundial_worker' (the package at /opt/sundial_worker)
        # is importable as `sundial_worker.server:app`.
        export PYTHONPATH=/opt
        export PREDICTALOT_SUNDIAL_SOCK="${SUNDIAL_SOCK}"
        while true; do
            /opt/sundial-venv/bin/uvicorn \
                sundial_worker.server:app \
                --uds "${SUNDIAL_SOCK}" \
                --log-level info 2>&1 \
                | sed -u 's/^/[sundial] /' || true
            echo "[entrypoint] sundial worker exited; restarting in 2s" >&2
            sleep 2
        done
    ) &
fi

exec python -m predictalot "$@"
