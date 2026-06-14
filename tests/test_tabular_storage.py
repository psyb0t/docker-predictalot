"""Tests for ``tabular_storage`` — disk-backed save/load/list/delete."""

from __future__ import annotations

import json

import pytest

from predictalot import tabular_storage as st


@pytest.fixture
def tmp_models(tmp_path, monkeypatch):
    """Point the storage root at a fresh tmp dir."""
    monkeypatch.setattr(st, "_ROOT", tmp_path / "tabular")
    return tmp_path / "tabular"


def _meta(model_id: str = "btcusd-D1-h1") -> st.TabularMeta:
    return st.TabularMeta(
        model_id=model_id,
        backend="lightgbm",
        mode="direction",
        horizon=1,
        feature_names=["rsi", "macd"],
        n_training_rows=1500,
        trained_at_unix=1700000000.0,
    )


def test_save_then_load_roundtrip(tmp_models) -> None:
    blob = b"fake-pickle-bytes"
    saved_dir = st.save(_meta(), blob)
    assert saved_dir.exists()
    meta2, blob2 = st.load("btcusd-D1-h1")
    assert blob2 == blob
    assert meta2.model_id == "btcusd-D1-h1"
    assert meta2.feature_names == ["rsi", "macd"]
    assert meta2.n_training_rows == 1500


def test_load_unknown_id_raises(tmp_models) -> None:
    with pytest.raises(FileNotFoundError):
        st.load("missing")


def test_list_returns_metas_newest_first(tmp_models) -> None:
    a = _meta("alpha")
    a.trained_at_unix = 1.0
    st.save(a, b"x")
    b = _meta("beta")
    b.trained_at_unix = 100.0
    st.save(b, b"y")
    rows = st.list_ids()
    assert [r.model_id for r in rows] == ["beta", "alpha"]


def test_list_skips_dirs_missing_meta(tmp_models) -> None:
    st.save(_meta("alpha"), b"x")
    # Stray empty dir without meta.json
    (tmp_models / "stray").mkdir()
    rows = st.list_ids()
    assert len(rows) == 1
    assert rows[0].model_id == "alpha"


def test_list_skips_dirs_with_bad_meta(tmp_models) -> None:
    st.save(_meta("alpha"), b"x")
    bad = tmp_models / "bad"
    bad.mkdir()
    (bad / "meta.json").write_text("{ this is not json")
    rows = st.list_ids()
    assert [r.model_id for r in rows] == ["alpha"]


def test_delete_returns_true_and_removes_files(tmp_models) -> None:
    st.save(_meta(), b"x")
    assert st.exists("btcusd-D1-h1")
    assert st.delete("btcusd-D1-h1") is True
    assert not st.exists("btcusd-D1-h1")


def test_delete_missing_returns_false(tmp_models) -> None:
    assert st.delete("nope") is False


def test_save_overwrites_in_place(tmp_models) -> None:
    st.save(_meta(), b"v1")
    st.save(_meta(), b"v2")
    _, blob = st.load("btcusd-D1-h1")
    assert blob == b"v2"


def test_save_writes_meta_with_correct_fields(tmp_models) -> None:
    st.save(_meta(), b"x")
    meta_dict = json.loads((tmp_models / "btcusd-D1-h1" / "meta.json").read_text())
    assert meta_dict["backend"] == "lightgbm"
    assert meta_dict["mode"] == "direction"
    assert meta_dict["horizon"] == 1
    assert meta_dict["feature_names"] == ["rsi", "macd"]


@pytest.mark.parametrize("bad_id", ["", ".", "..", "with/slash"])
def test_invalid_model_id_raises(tmp_models, bad_id: str) -> None:
    with pytest.raises(ValueError, match="invalid model_id"):
        st.save(_meta(bad_id), b"x")


def test_atomic_write_does_not_leave_tmp_files(tmp_models) -> None:
    st.save(_meta(), b"x")
    entries = list((tmp_models / "btcusd-D1-h1").iterdir())
    names = [e.name for e in entries]
    assert "model.blob" in names
    assert "meta.json" in names
    assert not any(n.endswith(".tmp") for n in names)
