"""CLI: ``python -m predictalot.fetch [all|slug ...]``.

Pre-download model snapshots to PREDICTALOT_MODEL_DIR. Useful as an
entrypoint warmup step (PREDICTALOT_PREFETCH) or for warming a CI / shared
volume before serving.
"""

from __future__ import annotations

import logging
import sys

from . import config, storage
from .logging import configure as configure_logging

log = logging.getLogger("predictalot.fetch")


def fetch(slugs: list[str]) -> int:
    """Download the listed slugs. Returns 0 on success, non-zero on failure."""
    failed = []
    for slug in slugs:
        if slug not in config.MODEL_SLUGS:
            log.error("unknown slug: %s (valid: %s)", slug, list(config.MODEL_SLUGS))
            failed.append(slug)
            continue
        try:
            path = storage.ensure_snapshot(slug)
            log.info("fetched %s → %s", slug, path)
        except Exception as exc:  # noqa: BLE001
            log.error("fetch %s failed: %s", slug, exc)
            failed.append(slug)
    return 0 if not failed else 1


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        log.error(
            "usage: python -m predictalot.fetch [all | <slug> [<slug> ...]]\n"
            "       valid slugs: %s",
            list(config.MODEL_SLUGS),
        )
        return 2
    if args == ["all"]:
        slugs = list(config.MODEL_SLUGS)
    else:
        slugs = args
    return fetch(slugs)


if __name__ == "__main__":
    raise SystemExit(main())
