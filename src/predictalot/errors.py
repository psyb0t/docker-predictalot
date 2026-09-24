"""Typed errors for predictalot lifecycle operations."""

from __future__ import annotations


class ModelLifecycleError(RuntimeError):
    """Base error for foundation-model lifecycle operations."""


class ModelBusyError(ModelLifecycleError):
    """An unload would interrupt a live foundation-model request."""


class ModelUnloadError(ModelLifecycleError):
    """One or more foundation models could not be fully released."""
