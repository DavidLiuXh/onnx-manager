import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from onnx_manager.pool.manager import ModelPool
from onnx_manager.store.registry import ModelRecord


def _make_record(tmp_path, model_id="test/model", task="embedding"):
    model_dir = tmp_path / model_id.replace("/", "--")
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").write_bytes(b"fake")
    return ModelRecord(
        id=model_id, name="model", task=task, source="local",
        local_path=str(model_dir), size_bytes=4, pulled_at="2026-05-14T00:00:00",
    )


def test_load_and_get(tmp_path):
    record = _make_record(tmp_path)
    mock_session = MagicMock()
    mock_session.task = "embedding"

    with patch("onnx_manager.pool.manager.OnnxSession", return_value=mock_session):
        pool = ModelPool()
        pool.load(record)
        session = pool.get("test/model")

    assert session is mock_session


def test_load_idempotent(tmp_path):
    record = _make_record(tmp_path)
    mock_session = MagicMock()

    with patch("onnx_manager.pool.manager.OnnxSession", return_value=mock_session) as MockSession:
        pool = ModelPool()
        pool.load(record)
        pool.load(record)  # second call should not create another session

    assert MockSession.call_count == 1


def test_unload(tmp_path):
    record = _make_record(tmp_path)
    mock_session = MagicMock()

    with patch("onnx_manager.pool.manager.OnnxSession", return_value=mock_session):
        pool = ModelPool()
        pool.load(record)
        pool.unload("test/model")

    assert pool.get("test/model") is None


def test_list_loaded(tmp_path):
    rec_a = _make_record(tmp_path, "org/model-a", "embedding")
    rec_b = _make_record(tmp_path, "org/model-b", "rerank")

    with patch("onnx_manager.pool.manager.OnnxSession", return_value=MagicMock()):
        pool = ModelPool()
        pool.load(rec_a)
        pool.load(rec_b)
        loaded = pool.list_loaded()

    assert {m["id"] for m in loaded} == {"org/model-a", "org/model-b"}
