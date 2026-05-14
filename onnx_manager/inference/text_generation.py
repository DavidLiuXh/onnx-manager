import numpy as np
from onnx_manager.inference.base import InferenceBackend


class TextGenerationBackend(InferenceBackend):
    def run(self, prompt: str, max_tokens: int = 100) -> str:
        enc = self.session.tokenizer.encode(prompt)
        input_ids = enc.ids[:]
        eos_id_raw = self.session.tokenizer.token_to_id("</s>")
        eos_id = eos_id_raw if eos_id_raw is not None else 2
        generated = []

        for _ in range(max_tokens):
            input_names = [i.name for i in self.session.ort_session.get_inputs()]
            feeds = {"input_ids": np.array([input_ids], dtype=np.int64)}
            if "attention_mask" in input_names:
                feeds["attention_mask"] = np.array([[1] * len(input_ids)], dtype=np.int64)
            logits = self.session.ort_session.run(None, feeds)[0]  # (1, seq, vocab)
            next_token_id = int(np.argmax(logits[0, -1, :]))
            if next_token_id == eos_id:
                break
            generated.append(next_token_id)
            input_ids.append(next_token_id)

        return self.session.tokenizer.decode(generated)
