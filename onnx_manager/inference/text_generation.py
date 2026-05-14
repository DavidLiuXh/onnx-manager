import numpy as np
from onnx_manager.inference.base import InferenceBackend


class TextGenerationBackend(InferenceBackend):
    def run(self, prompt: str, max_tokens: int = 100) -> str:
        enc = self.session.tokenizer.encode(prompt)
        input_ids = enc.ids[:]
        eos_id = self.session.tokenizer.token_to_id("</s>") or 2
        generated = []

        for _ in range(max_tokens):
            feeds = {"input_ids": np.array([input_ids], dtype=np.int64)}
            logits = self.session.ort_session.run(None, feeds)[0]  # (1, seq, vocab)
            next_token_id = int(np.argmax(logits[0, -1, :]))
            if next_token_id == eos_id:
                break
            generated.append(next_token_id)
            input_ids.append(next_token_id)

        return self.session.tokenizer.decode(generated)
