import numpy as np
import pytest
from unittest.mock import MagicMock
from onnx_manager.inference.embedding import EmbeddingBackend


def _make_session(hidden=64):
    session = MagicMock()
    tokenizer = MagicMock()
    encoded = MagicMock()
    encoded.ids = [1, 2, 3]
    encoded.attention_mask = [1, 1, 1]
    encoded.type_ids = [0, 0, 0]
    tokenizer.encode.return_value = encoded

    raw_output = np.random.rand(1, 3, hidden).astype(np.float32)
    session.ort_session.run.return_value = [raw_output]

    mock_input_ids = MagicMock()
    mock_input_ids.name = "input_ids"
    mock_attention_mask = MagicMock()
    mock_attention_mask.name = "attention_mask"
    mock_token_type_ids = MagicMock()
    mock_token_type_ids.name = "token_type_ids"
    session.ort_session.get_inputs.return_value = [mock_input_ids, mock_attention_mask, mock_token_type_ids]

    session.tokenizer = tokenizer
    return session


def test_embed_single_text():
    session = _make_session()
    backend = EmbeddingBackend(session)
    result = backend.run(["hello world"])
    assert len(result) == 1
    assert len(result[0]) == 64
    norm = np.linalg.norm(result[0])
    assert abs(norm - 1.0) < 1e-5


def test_embed_batch():
    session = _make_session()
    backend = EmbeddingBackend(session)
    result = backend.run(["text one", "text two"])
    assert len(result) == 2


def test_embed_raises_without_tokenizer():
    session = MagicMock()
    session.tokenizer = None
    backend = EmbeddingBackend(session)
    with pytest.raises(ValueError, match="tokenizer"):
        backend.run(["hello"])
