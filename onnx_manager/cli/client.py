import httpx
import onnx_manager.config as config


class DaemonClient:
    def __init__(self, host: str = None, port: int = None):
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
