import numpy as np
from scipy.special import softmax
from onnx_manager.inference.base import InferenceBackend


class RerankBackend(InferenceBackend):
    def run(self, query: str, documents: list[str]) -> list[float]:
        if self.session.tokenizer is None:
            raise ValueError(
                "This model has no tokenizer. Ensure tokenizer.json is present in the model directory."
            )
        input_names = [i.name for i in self.session.ort_session.get_inputs()]
        scores = []
        for doc in documents:
            text = f"{query} [SEP] {doc}"
            enc = self.session.tokenizer.encode(text)
            feeds = {}
            if "input_ids" in input_names:
                feeds["input_ids"] = np.array([enc.ids], dtype=np.int64)
            if "attention_mask" in input_names:
                feeds["attention_mask"] = np.array([enc.attention_mask], dtype=np.int64)
            logits = self.session.ort_session.run(None, feeds)[0]  # (1, 2)
            probs = softmax(logits[0])
            scores.append(float(probs[-1]))
        return scores
