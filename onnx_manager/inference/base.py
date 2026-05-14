from abc import ABC
from onnx_manager.pool.session import OnnxSession


class InferenceBackend(ABC):
    def __init__(self, session: OnnxSession):
        self.session = session
