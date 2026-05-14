from abc import ABC, abstractmethod
from onnx_manager.pool.session import OnnxSession


class InferenceBackend(ABC):
    def __init__(self, session: OnnxSession):
        self.session = session

    @abstractmethod
    def run(self, *args, **kwargs):
        ...
