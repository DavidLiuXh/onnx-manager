import threading
from pathlib import Path
from typing import Optional

from onnx_manager.pool.session import OnnxSession
from onnx_manager.store.registry import ModelRecord


class ModelPool:
    def __init__(self):
        self._sessions: dict[str, OnnxSession] = {}
        self._lock = threading.Lock()

    def load(self, record: ModelRecord) -> OnnxSession:
        with self._lock:
            if record.id in self._sessions:
                return self._sessions[record.id]
            session = OnnxSession(
                model_dir=Path(record.local_path),
                task=record.task,
            )
            self._sessions[record.id] = session
            return session

    def unload(self, model_id: str) -> None:
        with self._lock:
            self._sessions.pop(model_id, None)

    def get(self, model_id: str) -> Optional[OnnxSession]:
        with self._lock:
            return self._sessions.get(model_id)

    def list_loaded(self) -> list[dict]:
        with self._lock:
            return [
                {"id": mid, "task": s.task}
                for mid, s in self._sessions.items()
            ]
