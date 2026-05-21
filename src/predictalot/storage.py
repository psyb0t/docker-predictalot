"""Model snapshot storage — download HF repos as clean local directories.

`ensure_snapshot(slug, repo_id)` is the only entry point. It:
  1. Checks `$PREDICTALOT_MODEL_DIR/<slug>/` for completeness (config.json +
     at least one weight file).
  2. If incomplete, calls `huggingface_hub.snapshot_download(...)` which is
     idempotent and resumable.
  3. Returns the local directory path; callers pass it to `from_pretrained()`.

No HF blob-cache fallback — one storage path, one mental model.
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import config

log = logging.getLogger("predictalot.storage")

# Files that signal a snapshot is "complete enough to load". Weight files use
# multiple possible extensions across these models.
_REQUIRED_FILE = "config.json"
_WEIGHT_GLOBS = ("*.safetensors", "*.bin", "*.pth", "*.ckpt", "*.msgpack")


def snapshot_dir(slug: str) -> Path:
    """Local snapshot directory for a given model slug."""
    return config.MODEL_DIR / slug


def snapshot_complete(slug: str) -> bool:
    """True if the local snapshot dir has at least a config.json + one weight file."""
    d = snapshot_dir(slug)
    if not d.is_dir():
        return False
    if not (d / _REQUIRED_FILE).is_file():
        return False
    for pattern in _WEIGHT_GLOBS:
        if any(d.glob(pattern)):
            return True
        # also accept weights one level deep (some HF repos use subdirs)
        if any(d.glob(f"*/{pattern}")):
            return True
    return False


def ensure_snapshot(slug: str, repo_id: str | None = None) -> Path:
    """Download the HF snapshot if missing; return the local directory path.

    Raises RuntimeError on download failure (caller surfaces as HTTP 503).
    """
    if repo_id is None:
        repo_id = config.MODEL_REPO_IDS.get(slug)
        if repo_id is None:
            raise ValueError(f"unknown model slug: {slug!r}")

    d = snapshot_dir(slug)
    if snapshot_complete(slug):
        return d

    d.mkdir(parents=True, exist_ok=True)
    log.info(
        "downloading %s from %s → %s (this may take a while)", slug, repo_id, d
    )

    try:
        # Lazy import: huggingface_hub is a runtime dep but importing at module
        # load would crash unit tests that don't need it.
        from huggingface_hub import snapshot_download

        # `local_dir=...` produces a clean directory tree (no symlinks) since
        # huggingface_hub 0.21+. The deprecated `local_dir_use_symlinks=False`
        # is omitted — it was a no-op in 0.21+ and removed in 1.x.
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(d),
        )
    except Exception as exc:
        raise RuntimeError(
            f"snapshot_download failed for {slug} ({repo_id}): {exc}"
        ) from exc

    if not snapshot_complete(slug):
        raise RuntimeError(
            f"snapshot for {slug} downloaded to {d} but looks incomplete "
            f"(missing {_REQUIRED_FILE} or weight files)"
        )

    log.info("downloaded %s", slug)
    return d
