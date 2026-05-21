"""Device resolution. Lazy-imports torch so module import doesn't require it."""

from __future__ import annotations

from . import config


def resolve_device() -> str:
    """Resolve PREDICTALOT_DEVICE into an actual device string.

    'auto' → 'cuda' if torch CUDA available, else 'cpu'.
    'cpu' / 'cuda' / 'cuda:N' → returned as-is.
    """
    d = config.DEVICE
    if d != "auto":
        return d
    try:
        import torch  # lazy

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
