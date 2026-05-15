from pathlib import Path

ONNX_HOME = Path.home() / ".onnx"
MODELS_DIR = ONNX_HOME / "models"
REGISTRY_PATH = ONNX_HOME / "registry.db"
PID_FILE = ONNX_HOME / "daemon.pid"
DAEMON_INFO_FILE = ONNX_HOME / "daemon.json"
LOG_FILE = ONNX_HOME / "daemon.log"
DEFAULT_PORT = 11434
DEFAULT_HOST = "127.0.0.1"
TASK_TYPES = ("embedding", "rerank", "text-generation")
PIPELINE_TAG_MAP = {
    "feature-extraction": "embedding",
    "sentence-similarity": "embedding",
    "text-ranking": "rerank",
    "text-generation": "text-generation",
}

# Maps pipeline_tag → optimum-cli --task argument for ONNX export
PIPELINE_TAG_TO_OPTIMUM_TASK = {
    "feature-extraction": "feature-extraction",
    "sentence-similarity": "feature-extraction",
    "text-ranking": "text-classification",
    "text-generation": "text-generation",
}
