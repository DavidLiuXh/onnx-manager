from pathlib import Path

import onnxruntime as ort


class OnnxSession:
    def __init__(self, model_dir: Path, task: str):
        self.model_dir = model_dir
        self.task = task
        self.ort_session: ort.InferenceSession = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self):
        tok_path = self.model_dir / "tokenizer.json"
        if not tok_path.exists():
            return None
        from tokenizers import Tokenizer
        return Tokenizer.from_file(str(tok_path))
