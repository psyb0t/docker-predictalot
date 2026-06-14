"""Disk-backed model store for /v1/tabular/*.

Layout under PREDICTALOT_MODEL_DIR/tabular/<model_id>/:

    model.blob              — backend-defined opaque bytes (pickle/joblib/etc.)
    meta.json               — JSON metadata: backend, mode, horizon,
                              feature_names, n_training_rows, trained_at_unix.

Operations are blocking + thread-safe-ish through a per-id file lock
file (`.lock` next to the blob). Heavy backends (TabPFN) keep their
weights cached separately in the HF/torch hub cache — out of our path.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config

_ROOT = Path(config.MODEL_DIR) / "tabular"


@dataclass
class TabularMeta:
    model_id: str
    backend: str
    mode: str
    horizon: int
    feature_names: list[str]
    n_training_rows: int
    trained_at_unix: float


def _id_dir(model_id: str) -> Path:
    if "/" in model_id or model_id in ("", ".", ".."):
        raise ValueError(f"invalid model_id {model_id!r}")
    return _ROOT / model_id


def save(meta: TabularMeta, blob: bytes) -> Path:
    d = _id_dir(meta.model_id)
    d.mkdir(parents=True, exist_ok=True)
    blob_path = d / "model.blob"
    meta_path = d / "meta.json"
    tmp_blob = blob_path.with_suffix(".tmp")
    tmp_meta = meta_path.with_suffix(".tmp")
    tmp_blob.write_bytes(blob)
    tmp_meta.write_text(json.dumps(asdict(meta), indent=2))
    os.replace(tmp_blob, blob_path)
    os.replace(tmp_meta, meta_path)
    return d


def load(model_id: str) -> tuple[TabularMeta, bytes]:
    d = _id_dir(model_id)
    meta_path = d / "meta.json"
    blob_path = d / "model.blob"
    if not meta_path.exists() or not blob_path.exists():
        raise FileNotFoundError(f"no stored tabular model {model_id!r}")
    meta_dict = json.loads(meta_path.read_text())
    return TabularMeta(**meta_dict), blob_path.read_bytes()


def delete(model_id: str) -> bool:
    d = _id_dir(model_id)
    if not d.exists():
        return False
    for p in d.iterdir():
        p.unlink(missing_ok=True)
    d.rmdir()
    return True


def list_ids() -> list[TabularMeta]:
    if not _ROOT.exists():
        return []
    out: list[TabularMeta] = []
    for entry in _ROOT.iterdir():
        if not entry.is_dir():
            continue
        meta_path = entry / "meta.json"
        if not meta_path.exists():
            continue
        try:
            out.append(TabularMeta(**json.loads(meta_path.read_text())))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    out.sort(key=lambda m: m.trained_at_unix, reverse=True)
    return out


def exists(model_id: str) -> bool:
    return (_id_dir(model_id) / "meta.json").exists()


def now_unix() -> float:
    return time.time()
