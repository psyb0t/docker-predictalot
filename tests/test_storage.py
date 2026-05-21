from __future__ import annotations

from pathlib import Path

import pytest

from predictalot import config, storage


@pytest.fixture
def model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    return tmp_path


class TestSnapshotComplete:
    def test_missing_dir(self, model_dir: Path) -> None:
        assert storage.snapshot_complete("chronos-2") is False

    def test_empty_dir(self, model_dir: Path) -> None:
        (model_dir / "chronos-2").mkdir()
        assert storage.snapshot_complete("chronos-2") is False

    def test_config_but_no_weights(self, model_dir: Path) -> None:
        d = model_dir / "chronos-2"
        d.mkdir()
        (d / "config.json").write_text("{}")
        assert storage.snapshot_complete("chronos-2") is False

    def test_safetensors(self, model_dir: Path) -> None:
        d = model_dir / "chronos-2"
        d.mkdir()
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"\x00" * 16)
        assert storage.snapshot_complete("chronos-2") is True

    def test_bin_weights(self, model_dir: Path) -> None:
        d = model_dir / "moirai-2"
        d.mkdir()
        (d / "config.json").write_text("{}")
        (d / "pytorch_model.bin").write_bytes(b"\x00" * 16)
        assert storage.snapshot_complete("moirai-2") is True

    def test_nested_weights(self, model_dir: Path) -> None:
        d = model_dir / "timesfm-2.5"
        nested = d / "checkpoints"
        nested.mkdir(parents=True)
        (d / "config.json").write_text("{}")
        (nested / "model.safetensors").write_bytes(b"\x00" * 16)
        assert storage.snapshot_complete("timesfm-2.5") is True


class TestEnsureSnapshot:
    def test_unknown_slug(self, model_dir: Path) -> None:
        with pytest.raises(ValueError, match="unknown model slug"):
            storage.ensure_snapshot("not-a-model")

    def test_already_complete_no_download(
        self, model_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        d = model_dir / "chronos-2"
        d.mkdir()
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"\x00")

        def fail_if_called(*_a, **_k):
            raise AssertionError("snapshot_download should not be called when snapshot is complete")

        monkeypatch.setattr("huggingface_hub.snapshot_download", fail_if_called)
        out = storage.ensure_snapshot("chronos-2")
        assert out == d
