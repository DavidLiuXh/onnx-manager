import numpy as np
from onnx_manager.inference.base import InferenceBackend


class EmbeddingBackend(InferenceBackend):
    def run(self, texts: list[str]) -> list[list[float]]:
        if self.session.tokenizer is None:
            raise ValueError(
                "This model has no tokenizer. Ensure tokenizer.json is present in the model directory."
            )
        input_names = [i.name for i in self.session.ort_session.get_inputs()]
        results = []
        for text in texts:
            enc = self.session.tokenizer.encode(text)
            feeds = {}
            if "input_ids" in input_names:
                feeds["input_ids"] = np.array([enc.ids], dtype=np.int64)
            if "attention_mask" in input_names:
                feeds["attention_mask"] = np.array([enc.attention_mask], dtype=np.int64)
            output = self.session.ort_session.run(None, feeds)[0]  # (1, seq, hidden)
            vec = output[0].mean(axis=0)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec.tolist())
        return results
