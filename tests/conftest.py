import pytest
from pathlib import Path


@pytest.fixture
def tmp_onnx_home(tmp_path, monkeypatch):
    home = tmp_path / ".onnx"
    home.mkdir()
    (home / "models").mkdir()
    monkeypatch.setattr("onnx_manager.config.ONNX_HOME", home)
    monkeypatch.setattr("onnx_manager.config.MODELS_DIR", home / "models")
    monkeypatch.setattr("onnx_manager.config.REGISTRY_PATH", home / "registry.db")
    monkeypatch.setattr("onnx_manager.config.PID_FILE", home / "daemon.pid")
    return home
