import numpy as np
import pytest
import itertools
from unittest.mock import MagicMock
from onnx_manager.inference.rerank import RerankBackend


def _make_session():
    session = MagicMock()
    tokenizer = MagicMock()
    encoded = MagicMock()
    encoded.ids = [1, 2, 3]
    encoded.attention_mask = [1, 1, 1]
    encoded.type_ids = [0, 0, 1]  # segment IDs for pair encoding
    tokenizer.encode.return_value = encoded
    session.ort_session.run.return_value = [np.array([[0.1, 0.9]], dtype=np.float32)]

    mock_input_ids = MagicMock()
    mock_input_ids.name = "input_ids"
    mock_attention_mask = MagicMock()
    mock_attention_mask.name = "attention_mask"
    mock_token_type_ids = MagicMock()
    mock_token_type_ids.name = "token_type_ids"
    session.ort_session.get_inputs.return_value = [mock_input_ids, mock_attention_mask, mock_token_type_ids]

    session.tokenizer = tokenizer
    return session


def test_rerank_returns_scores():
    session = _make_session()
    backend = RerankBackend(session)
    scores = backend.run(query="what is AI", documents=["doc1", "doc2"])
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)


def test_rerank_score_order():
    session = _make_session()
    call_count = itertools.count()
    def mock_run(*args, **kwargs):
        n = next(call_count)
        if n == 0:
            return [np.array([[0.8, 0.2]], dtype=np.float32)]
        return [np.array([[0.1, 0.9]], dtype=np.float32)]
    session.ort_session.run.side_effect = mock_run

    backend = RerankBackend(session)
    scores = backend.run(query="q", documents=["doc1", "doc2"])
    assert scores[1] > scores[0]
