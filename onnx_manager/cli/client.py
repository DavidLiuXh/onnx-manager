import httpx
import onnx_manager.config as config
from onnx_manager.daemon.lifecycle import read_daemon_info


class DaemonClient:
    def __init__(self, host: str = None, port: int = None):
        if host is None or port is None:
            info = read_daemon_info()
            if info is not None:
                host = host or info["host"]
                port = port or info["port"]
        self.base_url = f"http://{host or config.DEFAULT_HOST}:{port or config.DEFAULT_PORT}"

    def is_alive(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=2)
            return r.status_code == 200
        except httpx.ConnectError:
            return False

    def list_loaded(self) -> list[dict]:
        r = httpx.get(f"{self.base_url}/v1/models", timeout=5)
        r.raise_for_status()
        return r.json()["data"]

    def load_model(self, model_id: str) -> dict:
        r = httpx.post(f"{self.base_url}/v1/models/{model_id}/load", timeout=60)
        r.raise_for_status()
        return r.json()

    def unload_model(self, model_id: str) -> dict:
        r = httpx.post(f"{self.base_url}/v1/models/{model_id}/unload", timeout=10)
        r.raise_for_status()
        return r.json()

    def embed(self, model_id: str, text: str) -> list[float]:
        r = httpx.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": model_id, "input": text},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]

    def rerank(self, model_id: str, query: str, documents: list[str]) -> list[dict]:
        r = httpx.post(
            f"{self.base_url}/v1/rerank",
            json={"model": model_id, "query": query, "documents": documents},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["results"]

    def complete(self, model_id: str, prompt: str, max_tokens: int = 100) -> str:
        r = httpx.post(
            f"{self.base_url}/v1/completions",
            json={"model": model_id, "prompt": prompt, "max_tokens": max_tokens},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["text"]
