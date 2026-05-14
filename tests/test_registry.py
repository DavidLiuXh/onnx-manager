import pytest
from onnx_manager.store.registry import ModelRegistry, ModelRecord


def test_add_and_get(tmp_onnx_home):
    reg = ModelRegistry()
    record = ModelRecord(
        id="BAAI/bge-small-en-v1.5",
        name="bge-small-en-v1.5",
        task="embedding",
        source="huggingface",
        local_path=str(tmp_onnx_home / "models" / "BAAI--bge-small-en-v1.5"),
        size_bytes=23_000_000,
        pulled_at="2026-05-14T10:00:00",
    )
    reg.add(record)
    fetched = reg.get("BAAI/bge-small-en-v1.5")
    assert fetched is not None
    assert fetched.task == "embedding"
    assert fetched.source == "huggingface"


def test_list_all(tmp_onnx_home):
    reg = ModelRegistry()
    reg.add(ModelRecord(
        id="model-a", name="a", task="embedding", source="local",
        local_path="/tmp/a", size_bytes=1000, pulled_at="2026-05-14T10:00:00",
    ))
    reg.add(ModelRecord(
        id="model-b", name="b", task="rerank", source="huggingface",
        local_path="/tmp/b", size_bytes=2000, pulled_at="2026-05-14T10:01:00",
    ))
    records = reg.list_all()
    assert len(records) == 2
    # list_all() must return results sorted by pulled_at DESC
    assert records[0].id == "model-b"  # pulled_at 10:01, newer
    assert records[1].id == "model-a"  # pulled_at 10:00, older


def test_delete(tmp_onnx_home):
    reg = ModelRegistry()
    reg.add(ModelRecord(
        id="to-delete", name="d", task="rerank", source="local",
        local_path="/tmp/d", size_bytes=0, pulled_at="2026-05-14T10:00:00",
    ))
    reg.delete("to-delete")
    assert reg.get("to-delete") is None


def test_get_nonexistent(tmp_onnx_home):
    reg = ModelRegistry()
    assert reg.get("does-not-exist") is None
