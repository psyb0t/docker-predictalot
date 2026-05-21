"""Centralized config — env var reading + validation, executed at import time.

Fail-fast: any malformed env var raises at import; the container won't start
with silently-bogus config.
"""

from __future__ import annotations

import os
from pathlib import Path

from .duration import parse_duration
from .size import parse_size

# ─── model registry ──────────────────────────────────────────────────────────

MODEL_SLUGS: tuple[str, ...] = (
    "chronos-2",
    "timesfm-2.5",
    "moirai-2",
    "toto-1",
    "sundial-base-128m",
)

# HF repo ids per slug — used by storage.ensure_snapshot() (except sundial,
# which is downloaded by its sidecar worker, not by the main process).
MODEL_REPO_IDS: dict[str, str] = {
    "chronos-2": "amazon/chronos-2",
    "timesfm-2.5": "google/timesfm-2.5-200m-pytorch",
    "moirai-2": "Salesforce/moirai-2.0-R-small",
    "toto-1": "Datadog/Toto-Open-Base-1.0",
    "sundial-base-128m": "thuml/sundial-base-128m",
}

# Default contextLength per model — used when caller omits it
DEFAULT_CONTEXT_LENGTH: dict[str, int] = {
    "chronos-2": 2048,
    "timesfm-2.5": 2048,
    "moirai-2": 4000,
    "toto-1": 4096,  # Toto-1 was pretrained with up to 4096-step context
    "sundial-base-128m": 2880,  # sundial's training context length
}

# The 9 quantile levels every supported model can produce
ALLOWED_QUANTILE_LEVELS: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(1, 10))

DEFAULT_QUANTILE_LEVELS: list[float] = [0.1, 0.5, 0.9]


# ─── env vars ────────────────────────────────────────────────────────────────


def _slug_to_envkey(slug: str) -> str:
    """Normalize a slug for env-var lookups: uppercase + - / . → _."""
    return slug.upper().replace("-", "_").replace(".", "_")


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return default
    return raw in ("1", "true", "yes", "on")


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not an integer") from exc


def _list_env(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [s.strip() for s in raw.split(",") if s.strip()]


HOST: str = os.environ.get("PREDICTALOT_HOST", "0.0.0.0")
PORT: int = _int_env("PREDICTALOT_PORT", 8080)

AUTH_TOKENS: list[str] = _list_env("PREDICTALOT_AUTH_TOKENS")
ALLOW_NO_AUTH: bool = _bool_env("PREDICTALOT_ALLOW_NO_AUTH", False)

DEVICE: str = os.environ.get("PREDICTALOT_DEVICE", "auto").strip() or "auto"
if DEVICE not in ("auto", "cpu", "cuda") and not DEVICE.startswith("cuda:"):
    raise ValueError(
        f"PREDICTALOT_DEVICE={DEVICE!r} must be 'auto', 'cpu', 'cuda', or 'cuda:N'"
    )


def _validate_slugs(name: str, slugs: list[str]) -> list[str]:
    unknown = [s for s in slugs if s not in MODEL_SLUGS]
    if unknown:
        raise ValueError(
            f"{name} contains unknown slugs: {unknown!r}; valid: {list(MODEL_SLUGS)}"
        )
    return slugs


_preload_raw = _list_env("PREDICTALOT_PRELOAD")
PRELOAD: list[str] = _validate_slugs("PREDICTALOT_PRELOAD", _preload_raw)

_prefetch_raw = _list_env("PREDICTALOT_PREFETCH")
if _prefetch_raw == ["all"]:
    PREFETCH: list[str] = list(MODEL_SLUGS)
else:
    PREFETCH = _validate_slugs("PREDICTALOT_PREFETCH", _prefetch_raw)


def _duration_env(name: str, default: str) -> float:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        raw = default
    try:
        return parse_duration(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r}: {exc}") from exc


def _size_env(name: str, default: str) -> int:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        raw = default
    try:
        return parse_size(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r}: {exc}") from exc


MODEL_IDLE_TIMEOUT: float = _duration_env("PREDICTALOT_MODEL_IDLE_TIMEOUT", "30m")

MODEL_IDLE_TIMEOUT_PER_SLUG: dict[str, float] = {}
for _slug in MODEL_SLUGS:
    _envkey = f"PREDICTALOT_MODEL_IDLE_TIMEOUT_{_slug_to_envkey(_slug)}"
    _raw = os.environ.get(_envkey, "").strip()
    if _raw:
        try:
            MODEL_IDLE_TIMEOUT_PER_SLUG[_slug] = parse_duration(_raw)
        except ValueError as exc:
            raise ValueError(f"{_envkey}={_raw!r}: {exc}") from exc

MAX_BODY_SIZE: int = _size_env("PREDICTALOT_MAX_BODY_SIZE", "32mb")

TIMESFM_MAX_CONTEXT: int = _int_env("PREDICTALOT_TIMESFM_MAX_CONTEXT", 2048)
TIMESFM_MAX_HORIZON: int = _int_env("PREDICTALOT_TIMESFM_MAX_HORIZON", 512)
if TIMESFM_MAX_CONTEXT % 32 != 0:
    raise ValueError(
        f"PREDICTALOT_TIMESFM_MAX_CONTEXT={TIMESFM_MAX_CONTEXT} must be a multiple of 32"
    )
if TIMESFM_MAX_HORIZON % 128 != 0:
    raise ValueError(
        f"PREDICTALOT_TIMESFM_MAX_HORIZON={TIMESFM_MAX_HORIZON} must be a multiple of 128"
    )

# Moirai-2 wrapper dimensions — baked into Moirai2Forecast at model-load time
# (it compiles internal patching at these sizes). Per-request inputs shorter
# than MAX_CONTEXT are zero-padded with past_is_pad=True; horizons must be
# <= MAX_HORIZON. Bump the env vars + restart to expand the envelope.
MOIRAI_MAX_CONTEXT: int = _int_env("PREDICTALOT_MOIRAI_MAX_CONTEXT", 4000)
MOIRAI_MAX_HORIZON: int = _int_env("PREDICTALOT_MOIRAI_MAX_HORIZON", 512)
if MOIRAI_MAX_CONTEXT <= 0:
    raise ValueError(f"PREDICTALOT_MOIRAI_MAX_CONTEXT={MOIRAI_MAX_CONTEXT} must be > 0")
if MOIRAI_MAX_HORIZON <= 0:
    raise ValueError(f"PREDICTALOT_MOIRAI_MAX_HORIZON={MOIRAI_MAX_HORIZON} must be > 0")

MODEL_DIR: Path = Path(os.environ.get("PREDICTALOT_MODEL_DIR", "/models")).resolve()


def idle_timeout_for(slug: str) -> float:
    """Return the idle-timeout (seconds) for a given model slug."""
    return MODEL_IDLE_TIMEOUT_PER_SLUG.get(slug, MODEL_IDLE_TIMEOUT)
