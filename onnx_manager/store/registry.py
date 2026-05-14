import sqlite3
from dataclasses import dataclass
from typing import Optional
import onnx_manager.config as config


@dataclass
class ModelRecord:
    id: str
    name: str
    task: str
    source: str
    local_path: str
    size_bytes: Optional[int]
    pulled_at: str


class ModelRegistry:
    def __init__(self):
        config.REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(config.REGISTRY_PATH))
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                task        TEXT NOT NULL,
                source      TEXT NOT NULL,
                local_path  TEXT NOT NULL,
                size_bytes  INTEGER,
                pulled_at   TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def add(self, record: ModelRecord) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO models VALUES (?,?,?,?,?,?,?)",
            (record.id, record.name, record.task, record.source,
             record.local_path, record.size_bytes, record.pulled_at),
        )
        self._conn.commit()

    def get(self, model_id: str) -> Optional[ModelRecord]:
        row = self._conn.execute(
            "SELECT * FROM models WHERE id = ?", (model_id,)
        ).fetchone()
        if row is None:
            return None
        return ModelRecord(**dict(row))

    def list_all(self) -> list[ModelRecord]:
        rows = self._conn.execute("SELECT * FROM models ORDER BY pulled_at DESC").fetchall()
        return [ModelRecord(**dict(r)) for r in rows]

    def delete(self, model_id: str) -> None:
        self._conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
        self._conn.commit()
