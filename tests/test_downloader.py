import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from onnx_manager.store.downloader import (
    model_id_to_dirname,
    detect_task_from_pipeline_tag,
    import_local_model,
)
from onnx_manager.store.registry import ModelRegistry


def test_model_id_to_dirname():
    assert model_id_to_dirname("BAAI/bge-small-en-v1.5") == "BAAI--bge-small-en-v1.5"
    assert model_id_to_dirname("mymodel") == "mymodel"


def test_detect_task_from_pipeline_tag():
    assert detect_task_from_pipeline_tag("feature-extraction") == "embedding"
    assert detect_task_from_pipeline_tag("sentence-similarity") == "embedding"
    assert detect_task_from_pipeline_tag("text-ranking") == "rerank"
    assert detect_task_from_pipeline_tag("text-generation") == "text-generation"
    assert detect_task_from_pipeline_tag("unknown-tag") is None


def test_import_local_model(tmp_onnx_home, tmp_path):
    # Create a fake onnx file and tokenizer
    fake_model = tmp_path / "model.onnx"
    fake_model.write_bytes(b"fake-onnx-bytes")
    fake_tokenizer = tmp_path / "tokenizer.json"
    fake_tokenizer.write_text('{"version": "1.0"}')
    fake_tokenizer_config = tmp_path / "tokenizer_config.json"
    fake_tokenizer_config.write_text('{"model_type": "bert"}')

    with patch("onnx_manager.store.downloader.onnx.checker.check_model"):
        reg = ModelRegistry()
        record = import_local_model(
            onnx_path=fake_model,
            name="mymodel",
            task="embedding",
            registry=reg,
        )

    assert record.id == "mymodel"
    assert record.task == "embedding"
    assert record.source == "local"
    dest_dir = tmp_onnx_home / "models" / "mymodel"
    assert (dest_dir / "model.onnx").exists()
    assert (dest_dir / "tokenizer.json").exists()
    assert (dest_dir / "tokenizer_config.json").exists()
    assert reg.get("mymodel") is not None


def test_import_local_model_no_tokenizer(tmp_onnx_home, tmp_path):
    fake_model = tmp_path / "model.onnx"
    fake_model.write_bytes(b"fake-onnx-bytes")
    # no tokenizer.json in the directory

    with patch("onnx_manager.store.downloader.onnx.checker.check_model"):
        reg = ModelRegistry()
        record = import_local_model(
            onnx_path=fake_model,
            name="no-tok-model",
            task="rerank",
            registry=reg,
        )

    assert record.id == "no-tok-model"
    dest_dir = tmp_onnx_home / "models" / "no-tok-model"
    assert (dest_dir / "model.onnx").exists()
    assert not (dest_dir / "tokenizer.json").exists()


def test_downloader_cleans_onnx_subdir_after_move(tmp_onnx_home, tmp_path):
    """When model is moved from onnx/ subdir, the subdir (with any extra files) is cleaned."""
    import shutil as _shutil
    from pathlib import Path as _Path

    dest_dir = tmp_onnx_home / "models" / "test--model"
    dest_dir.mkdir(parents=True)

    # Simulate: onnx/model.onnx exists with extra cache file
    onnx_subdir = dest_dir / "onnx"
    onnx_subdir.mkdir()
    onnx_src = onnx_subdir / "model.onnx"
    onnx_src.write_bytes(b"fake-onnx")
    (onnx_subdir / ".cache_info").write_text("extra")  # extra file that rmdir() can't handle

    onnx_dest = dest_dir / "model.onnx"
    _shutil.move(str(onnx_src), str(onnx_dest))

    # This is what the fixed code does:
    if onnx_src.parent != dest_dir:
        _shutil.rmtree(str(onnx_src.parent), ignore_errors=True)

    assert onnx_dest.exists()
    assert not onnx_subdir.exists()  # subdir cleaned up despite extra file
