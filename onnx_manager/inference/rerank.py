import numpy as np
from scipy.special import softmax, expit
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
            # Use pair encoding so tokenizer handles [CLS]/[SEP] and segment IDs correctly
            enc = self.session.tokenizer.encode(query, doc)
            feeds = {}
            if "input_ids" in input_names:
                feeds["input_ids"] = np.array([enc.ids], dtype=np.int64)
            if "attention_mask" in input_names:
                feeds["attention_mask"] = np.array([enc.attention_mask], dtype=np.int64)
            if "token_type_ids" in input_names:
                feeds["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
            logits = self.session.ort_session.run(None, feeds)[0]  # (1, N)
            vec = logits[0]  # shape: (N,)
            if vec.shape[0] >= 2:
                # Binary classification (e.g. bge-reranker): softmax, take positive class
                score = float(softmax(vec)[-1])
            else:
                # Single logit (e.g. gte-multilingual-reranker): sigmoid to [0, 1]
                score = float(expit(vec[0]))
            scores.append(score)
        return scores
