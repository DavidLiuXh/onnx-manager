# ONNX Manager Technical Design Document

**Date**: 2026-05-14

---

## 1. Overview

### Background

ONNX Runtime is a high-performance inference engine commonly used in Agent scenarios for running local embedding, rerank, and other inference models. The existing tooling ecosystem is fragmented — there is no tool like ollama that can uniformly manage and serve multiple ONNX models. This project fills that gap.

### Goals

Build a developer-facing local ONNX model management and serving tool that supports:

- Downloading from HuggingFace or importing ONNX models from local files
- A single daemon process serving multiple models simultaneously
- OpenAI-compatible REST API (embedding, rerank, text-generation)
- An ollama-like CLI experience

### Out of Scope

- Model training
- Model conversion (ONNX format conversion)
- Production-grade deployment (multi-node, load balancing)
- Streaming generation

---

## 2. Overall Architecture

```
+---------------------------------------------------+
|                   onnx CLI                        |
|  pull / push / list / load / unload / ps / rm    |
+--------------------+------------------------------+
                     | HTTP (localhost)
+--------------------v------------------------------+
|                 onnx daemon                       |
|                                                   |
|  +--------------+   +-----------------------+    |
|  | Model Store  |   |   REST API Server     |    |
|  |  ~/.onnx/    |   |   (FastAPI)           |    |
|  |  models/     |   |                       |    |
|  |  registry.db |   |  POST /v1/embeddings  |    |
|  +--------------+   |  POST /v1/rerank      |    |
|                     |  POST /v1/completions |    |
|  +--------------+   |  GET  /v1/models      |    |
|  | Model Pool   |   +-----------------------+    |
|  |  (in-memory) |                                 |
|  |  bge-small   |                                 |
|  |  ms-marco    |                                 |
|  |  phi-3-mini  |                                 |
|  +--------------+                                 |
+---------------------------------------------------+
```

### Key Design Decisions

- **Daemon lifecycle**: Started manually via `onnx serve`, stopped via `onnx stop`; does not auto-start with the system
- **Communication**: CLI communicates with the daemon via `localhost:11434` (default)
- **Storage separation**: Model files stored in `~/.onnx/models/`, metadata stored in SQLite
- **On-demand loading**: If a model is not loaded when an API call arrives, it is loaded automatically; the response header includes `X-Model-Load-Time-Ms`

---

## 3. CLI Command Design

```bash
# Model management
onnx pull BAAI/bge-small-en-v1.5                       # Download from HuggingFace
onnx pull ./path/to/model.onnx --name mymodel --task embedding  # Import local file
onnx list                                               # List downloaded models
onnx rm BAAI/bge-small-en-v1.5                         # Delete local model
onnx show BAAI/bge-small-en-v1.5                       # Show metadata

# Service management
onnx serve                                              # Start daemon in foreground
onnx serve --port 11434 --background                   # Start daemon in background
onnx stop                                               # Stop background daemon
onnx ps                                                 # List models loaded in memory

# Runtime management
onnx load BAAI/bge-small-en-v1.5                       # Manually load into memory
onnx unload BAAI/bge-small-en-v1.5                     # Unload from memory

# Quick test
onnx run BAAI/bge-small-en-v1.5 "hello world"          # Single inference test
```

### Behavioral Conventions

- Both API calls and `onnx run` support automatic on-demand loading; manual `load` is not required
- `onnx pull` automatically identifies task type from the `pipeline_tag` field in HuggingFace's `config.json`
- Local imports must explicitly specify task type via `--task`

---

## 4. REST API Design

### Embedding (OpenAI-compatible)

```
POST /v1/embeddings
{
  "model": "BAAI/bge-small-en-v1.5",
  "input": "hello world"        // or ["text1", "text2"]
}

-> {
  "data": [{"embedding": [...], "index": 0, "object": "embedding"}],
  "model": "BAAI/bge-small-en-v1.5",
  "usage": {"prompt_tokens": 5, "total_tokens": 5}
}
```

### Rerank (Cohere/Jina style)

```
POST /v1/rerank
{
  "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
  "query": "what is AI",
  "documents": ["doc1", "doc2", "doc3"],
  "top_n": 3                    // optional
}

-> {
  "results": [
    {"index": 1, "score": 0.92, "document": "doc2"},
    ...
  ]
}
```

### Text Generation (OpenAI-compatible)

```
POST /v1/completions
{
  "model": "microsoft/phi-3-mini",
  "prompt": "Hello, my name is",
  "max_tokens": 100,
  "stream": false               // streaming not supported yet
}
```

### Other Endpoints

```
GET  /v1/models                 # List all loaded models
GET  /v1/models/{model_id}      # Single model status
GET  /v1/models/{model_id}/load    # Manually load a model
POST /v1/models/{model_id}/unload  # Manually unload a model
GET  /health                    # Health check
```

---

## 5. Module Structure

### Directory Layout

```
onnx_manager/                       # Python package root
+-- __init__.py
+-- config.py                       # Constants: default port, home dir (~/.onnx)
+-- store/
|   +-- __init__.py
|   +-- registry.py                 # SQLite CRUD: model record management
|   +-- downloader.py               # HuggingFace download + local file import
+-- pool/
|   +-- __init__.py
|   +-- session.py                  # OnnxSession: wraps ort.InferenceSession + tokenizer
|   +-- manager.py                  # ModelPool singleton: load/unload/query sessions
+-- inference/
|   +-- __init__.py
|   +-- base.py                     # InferenceBackend abstract base class
|   +-- embedding.py                # EmbeddingBackend: tokenize -> run -> normalize
|   +-- rerank.py                   # RerankBackend: query+docs -> scores
|   +-- text_generation.py          # TextGenerationBackend: prompt -> token loop
+-- daemon/
|   +-- __init__.py
|   +-- app.py                      # FastAPI app factory function
|   +-- lifecycle.py                # PID file management, graceful shutdown
|   +-- routes/
|       +-- __init__.py             # Shared helpers (_get_or_load_session)
|       +-- health.py               # GET /health
|       +-- models.py               # GET /v1/models, GET /v1/models/{model_id}
|       +-- embeddings.py           # POST /v1/embeddings
|       +-- rerank.py               # POST /v1/rerank
|       +-- completions.py          # POST /v1/completions
+-- cli/
    +-- __init__.py
    +-- main.py                     # Click group root, registered as `onnx` entry point
    +-- client.py                   # DaemonClient: HTTP client wrapper for CLI commands
    +-- commands/
        +-- __init__.py
        +-- pull.py                 # onnx pull
        +-- serve.py                # onnx serve, onnx stop
        +-- model.py                # onnx list/rm/show/load/unload/ps
        +-- run.py                  # onnx run
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `config.py` | Global constants: `ONNX_HOME`, `MODELS_DIR`, `REGISTRY_PATH`, `PID_FILE`, `DEFAULT_PORT`, `PIPELINE_TAG_MAP` |
| `store/registry.py` | SQLite metadata CRUD; provides `ModelRegistry` class and `ModelRecord` dataclass |
| `store/downloader.py` | HuggingFace download (prefer `onnx/model.onnx`, fallback to root) + local file import + ONNX validation |
| `pool/session.py` | `OnnxSession`: wraps `ort.InferenceSession` + tokenizer, unified loading entry point |
| `pool/manager.py` | `ModelPool` singleton: thread-safe management of in-memory sessions, prevents duplicate loading |
| `inference/` | Pure inference logic per backend; no awareness of HTTP or file paths |
| `daemon/app.py` | FastAPI app factory; injects pool and registry into `app.state` |
| `daemon/lifecycle.py` | PID file read/write, daemon start/stop/crash recovery |
| `daemon/routes/` | HTTP route handlers; calls pool/inference to complete inference |
| `cli/client.py` | `DaemonClient`: thin HTTP wrapper used by CLI commands |
| `cli/commands/` | Click command implementations; operate directly on registry or communicate with daemon via client |

### Dependency Graph

```
CLI -> client.py -> daemon REST API
daemon routes -> pool.manager -> session -> inference backends
daemon routes -> store.registry (read metadata)
pull / run -> store.downloader + store.registry
```

### Constraints

- `inference/` backends handle pure inference only; they have no awareness of HTTP or file paths
- `pool/manager.py` is the sole location that operates ONNX Runtime sessions, preventing duplicate loading
- `store/registry.py` manages metadata in SQLite; model files are stored on the filesystem and linked via `model_id`

---

## 6. Data Storage

### Filesystem Layout

```
~/.onnx/
+-- registry.db
+-- daemon.pid
+-- daemon.log
+-- models/
    +-- BAAI--bge-small-en-v1.5/
    |   +-- model.onnx
    |   +-- tokenizer.json
    |   +-- tokenizer_config.json
    |   +-- meta.json
    +-- cross-encoder--ms-marco-MiniLM-L6-v2/
        +-- model.onnx
        +-- tokenizer.json
        +-- meta.json
```

### SQLite `models` Table Schema

```sql
CREATE TABLE models (
    id          TEXT PRIMARY KEY,   -- "BAAI/bge-small-en-v1.5"
    name        TEXT NOT NULL,
    task        TEXT NOT NULL,      -- "embedding" | "rerank" | "text-generation"
    source      TEXT NOT NULL,      -- "huggingface" | "local"
    local_path  TEXT NOT NULL,
    size_bytes  INTEGER,
    pulled_at   TEXT NOT NULL       -- ISO8601
);
```

### HuggingFace Download Strategy

- Prefer downloading `onnx/model.onnx`; fall back to `.onnx` files in the repository root
- Also download `tokenizer.json` + `tokenizer_config.json` (best effort)
- Replace `/` in model ID with `--` for directory names (e.g. `BAAI/bge-small-en-v1.5` -> `BAAI--bge-small-en-v1.5`)
- Task type is automatically identified from the `pipeline_tag` field in `config.json`, using the following mapping:

| `pipeline_tag` | task |
|---|---|
| `feature-extraction` | `embedding` |
| `sentence-similarity` | `embedding` |
| `text-ranking` | `rerank` |
| `text-generation` | `text-generation` |

---

## 7. Error Handling

| Scenario | Behavior |
|----------|----------|
| CLI called but daemon is not running | Prompt `run 'onnx serve' first`, exit |
| Daemon port is already in use | Error: `Port 11434 is already in use. Use --port to specify another.` |
| Daemon started twice | Detect `daemon.pid`; if process is alive, report error and exit |
| Download interrupted | Resume from checkpoint (natively supported by huggingface_hub) |
| Download temporary files | Write to temp directory; atomically move on completion to avoid partial files |
| Local import validation | Validate file integrity using `onnx.checker.check_model` |
| Model not loaded | Auto-load; add `X-Model-Load-Time-Ms` to response headers |
| Model file corrupted | Return 500; prompt user to re-run `onnx pull` |
| Batch input too long | Return 400; indicate maximum token limit |
| Out of memory | Return 503; list loaded models; suggest `onnx unload` |
| Ctrl+C / `onnx stop` | Wait for current request to complete (up to 30s); clean up PID file; graceful exit |
| Stale PID file after daemon crash | Next `onnx serve` detects process is dead; automatically cleans up PID file before starting |

---

## 8. Technology Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.10+ |
| CLI framework | Click |
| HTTP server | FastAPI + uvicorn |
| ONNX inference | onnxruntime |
| Tokenizer | tokenizers (HuggingFace) |
| HF download | huggingface_hub |
| Metadata storage | SQLite (stdlib sqlite3) |
| ONNX validation | onnx |
| Numerical computation | numpy |
| HTTP client | httpx |
| Rerank softmax | scipy |

---

## 9. Implementation Notes

### config Module Monkeypatch Design

All path constants in `config.py` (`ONNX_HOME`, `MODELS_DIR`, `REGISTRY_PATH`, `PID_FILE`) are module-level variables rather than values hardcoded inside individual modules. During testing, pytest's `monkeypatch.setattr` replaces these constants with temporary directories, so all modules (registry, downloader, lifecycle) automatically use isolated paths in tests without any changes to business logic.

```python
# tests/conftest.py shared fixture
@pytest.fixture
def tmp_onnx_home(tmp_path, monkeypatch):
    home = tmp_path / ".onnx"
    home.mkdir()
    (home / "models").mkdir()
    monkeypatch.setattr("onnx_manager.config.ONNX_HOME", home)
    monkeypatch.setattr("onnx_manager.config.MODELS_DIR", home / "models")
    monkeypatch.setattr("onnx_manager.config.REGISTRY_PATH", home / "registry.db")
    monkeypatch.setattr("onnx_manager.config.PID_FILE", home / "daemon.pid")
    return home
```

### ModelPool Thread Safety Design

`ModelPool` uses `threading.Lock` to protect all read and write operations on the internal `_sessions` dictionary. The `load` operation checks whether `model_id` already exists while holding the lock, ensuring idempotency (multiple calls create only one session). `get`, `unload`, and `list_loaded` also operate under the lock to prevent duplicate session creation or data races from concurrent requests.

### Tokenizer Pair Encoding in Inference Backends

The Rerank backend (`RerankBackend`) uses a cross-encoder architecture that requires query and document to be fed as a text pair. The implementation concatenates the query and document as `"{query} [SEP] {doc}"` and calls `tokenizer.encode`, leveraging the tokenizer's built-in separator handling to produce `input_ids` and `attention_mask`, which are then fed to the ONNX model. The logits are passed through softmax, and the positive-class probability (last dimension) is taken as the relevance score.

### Non-blocking asyncio Event Loop Design (run_in_executor)

Inference calls in FastAPI routes (`EmbeddingBackend.run`, `RerankBackend.run`, `TextGenerationBackend.run`) are CPU-intensive synchronous operations. Calling them directly in async routes would block uvicorn's event loop, preventing other requests from being handled. The correct approach is to submit inference tasks to a thread pool via `asyncio.get_event_loop().run_in_executor(None, ...)`, allowing the event loop to handle other requests while inference is in progress.

### Daemon Lifecycle PID Management

When the daemon starts, it writes `~/.onnx/daemon.pid`; when the process exits (normal exit or SIGTERM), the file is deleted. Before `onnx serve` starts, it checks for the PID file: if the file exists and the process is alive (`os.kill(pid, 0)` does not raise an exception), startup is refused and the user is notified; if the file exists but the process is dead (stale PID), the file is cleaned up automatically before starting. `onnx stop` reads the PID file to obtain the process ID and sends `SIGTERM` to trigger graceful shutdown.

---

## 10. Future Extensions

The following features are out of current scope but can be naturally extended from the architecture:

- **Text generation streaming**: Add `stream: true` support using FastAPI's `StreamingResponse` to return tokens incrementally
- **Model quantization**: Add INT8/FP16 quantization options in the downloader; `OnnxSession` selects the corresponding file on load
- **GPU acceleration**: Switch to `CUDAExecutionProvider` via the `providers` parameter when initializing `OnnxSession`
- **Model version management**: Add a `version` field to the `models` table; append version suffix to directory names to support multiple versions of the same model
- **Web UI management interface**: Mount static files on FastAPI or add a `/ui` route to provide a visual management interface
