import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from onnx_manager.pool.session import OnnxSession


def _make_fake_session(tmp_path):
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"fake")
    (model_dir / "tokenizer.json").write_text('{"version":"1.0"}')
    return model_dir


def test_session_loads(tmp_path):
    model_dir = _make_fake_session(tmp_path)
    mock_ort_session = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("onnxruntime.InferenceSession", return_value=mock_ort_session), \
         patch("tokenizers.Tokenizer.from_file", return_value=mock_tokenizer):
        session = OnnxSession(model_dir=model_dir, task="embedding")

    assert session.task == "embedding"
    assert session.ort_session is mock_ort_session
    assert session.tokenizer is mock_tokenizer


def test_session_no_tokenizer(tmp_path):
    model_dir = tmp_path / "no-tok"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"fake")
    mock_ort_session = MagicMock()

    with patch("onnxruntime.InferenceSession", return_value=mock_ort_session):
        session = OnnxSession(model_dir=model_dir, task="embedding")

    assert session.tokenizer is None
