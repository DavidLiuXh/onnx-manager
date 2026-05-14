import numpy as np
import pytest
from unittest.mock import MagicMock
from onnx_manager.inference.text_generation import TextGenerationBackend


def _make_session(vocab_size=100):
    session = MagicMock()
    tokenizer = MagicMock()
    encoded = MagicMock()
    encoded.ids = [1, 2, 3]
    tokenizer.encode.return_value = encoded
    tokenizer.decode.return_value = "hello world"
    tokenizer.token_to_id.return_value = 2  # EOS = 2

    logits = np.zeros((1, 3, vocab_size), dtype=np.float32)
    logits[0, -1, 42] = 10.0  # force token 42
    session.ort_session.run.return_value = [logits]

    mock_input_ids = MagicMock()
    mock_input_ids.name = "input_ids"
    session.ort_session.get_inputs.return_value = [mock_input_ids]

    session.tokenizer = tokenizer
    return session


def test_text_generation_returns_string():
    session = _make_session()
    backend = TextGenerationBackend(session)
    result = backend.run(prompt="Hello", max_tokens=3)
    assert isinstance(result, str)


def test_text_generation_stops_at_eos():
    session = _make_session(vocab_size=100)
    # token_to_id("</s>") returns 2 (EOS), force next token = 2 immediately
    logits_eos = np.zeros((1, 3, 100), dtype=np.float32)
    logits_eos[0, -1, 2] = 10.0  # force EOS token
    session.ort_session.run.return_value = [logits_eos]

    backend = TextGenerationBackend(session)
    result = backend.run(prompt="Hello", max_tokens=10)
    # should stop immediately, returning empty decoded string
    assert isinstance(result, str)
    # verify ort_session.run was called only once (stopped after first token = EOS)
    assert session.ort_session.run.call_count == 1
