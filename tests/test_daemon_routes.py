import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from onnx_manager.daemon.app import create_app
from onnx_manager.pool.manager import ModelPool
from onnx_manager.store.registry import ModelRegistry, ModelRecord


@pytest.fixture
def app(tmp_onnx_home):
    pool = ModelPool()
    registry = ModelRegistry()
    return create_app(pool=pool, registry=registry)


@pytest.fixture
def client(app):
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_list_models_empty(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["data"] == []


def test_embeddings(tmp_onnx_home, app):
    client = TestClient(app)
    mock_backend = MagicMock()
    mock_backend.run.return_value = [[0.1, 0.2, 0.3]]

    with patch("onnx_manager.daemon.routes.embeddings.EmbeddingBackend", return_value=mock_backend), \
         patch("onnx_manager.daemon.routes.embeddings._get_or_load_session") as mock_load:
        mock_load.return_value = MagicMock()
        resp = client.post("/v1/embeddings", json={
            "model": "BAAI/bge-small-en-v1.5",
            "input": "hello world",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert data["data"][0]["object"] == "embedding"


def test_rerank(tmp_onnx_home, app):
    client = TestClient(app)
    mock_backend = MagicMock()
    mock_backend.run.return_value = [0.9, 0.3, 0.7]

    with patch("onnx_manager.daemon.routes.rerank.RerankBackend", return_value=mock_backend), \
         patch("onnx_manager.daemon.routes.rerank._get_or_load_session") as mock_load:
        mock_load.return_value = MagicMock()
        resp = client.post("/v1/rerank", json={
            "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "query": "what is AI",
            "documents": ["doc1", "doc2", "doc3"],
        })

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 3
    scores = [r["score"] for r in data["results"]]
    assert scores == sorted(scores, reverse=True)


def test_completions(tmp_onnx_home, app):
    client = TestClient(app)
    mock_backend = MagicMock()
    mock_backend.run.return_value = " world"

    with patch("onnx_manager.daemon.routes.completions.TextGenerationBackend", return_value=mock_backend), \
         patch("onnx_manager.daemon.routes.completions._get_or_load_session") as mock_load:
        mock_load.return_value = MagicMock()
        resp = client.post("/v1/completions", json={
            "model": "microsoft/phi-3-mini",
            "prompt": "Hello",
            "max_tokens": 10,
        })

    assert resp.status_code == 200
    assert "choices" in resp.json()


def test_model_not_found(tmp_onnx_home, app):
    client = TestClient(app)
    resp = client.post("/v1/embeddings", json={
        "model": "nonexistent/model",
        "input": "test",
    })
    assert resp.status_code == 404
