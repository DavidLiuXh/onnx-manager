# onnx-manager

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

A local ONNX model management and serving tool for AI developers.

## Overview

onnx-manager lets you download, manage, and serve ONNX models through a unified daemon process and an OpenAI-compatible REST API. It solves the problem of running multiple ONNX inference models locally without writing boilerplate server code each time. Think of it as ollama for ONNX: one CLI to pull models, one daemon to serve them, one HTTP interface to query them.

## Features

- Pull ONNX models directly from HuggingFace Hub or import local `.onnx` files
- Unified model storage under `~/.onnx/models/` with a SQLite registry
- Single daemon process serves multiple models concurrently via a thread-safe memory pool
- OpenAI-compatible REST API for embeddings, reranking, and text generation
- Lazy loading and explicit load/unload control over in-memory models
- CLI experience modeled after ollama (`pull`, `list`, `rm`, `serve`, `ps`, `run`)

## Requirements

- Python 3.10 or higher
- Key runtime dependencies (installed automatically):
  - `onnxruntime>=1.17`
  - `fastapi>=0.110`, `uvicorn>=0.29`
  - `tokenizers>=0.19`
  - `huggingface-hub>=0.22`
  - `numpy>=1.26`, `scipy>=1.12`

## Installation

```bash
git clone https://github.com/your-org/onnx-manager.git
cd onnx-manager
pip install -e .
```

After installation the `onnx` command is available in your shell.

## Quick Start

**Step 1 — Pull a model from HuggingFace**

```bash
onnx pull BAAI/bge-small-en-v1.5
```

**Step 2 — Start the daemon**

```bash
onnx serve --background
```

**Step 3 — Send a request**

```bash
curl -s http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-small-en-v1.5", "input": "hello world"}' | jq .
```

## CLI Reference

### Model management

| Command | Description |
|---------|-------------|
| `onnx pull BAAI/bge-small-en-v1.5` | Download model from HuggingFace Hub |
| `onnx pull ./model.onnx --name mymodel --task embedding` | Import a local ONNX file |
| `onnx list` | List all downloaded models |
| `onnx rm BAAI/bge-small-en-v1.5` | Delete a model from local storage |
| `onnx show BAAI/bge-small-en-v1.5` | Display model metadata |

### Daemon control

| Command | Description |
|---------|-------------|
| `onnx serve` | Start daemon in the foreground (port 11434) |
| `onnx serve --port 11434 --background` | Start daemon in the background |
| `onnx stop` | Stop the running daemon |
| `onnx ps` | List models currently loaded in memory |

### Runtime operations

| Command | Description |
|---------|-------------|
| `onnx load BAAI/bge-small-en-v1.5` | Manually load a model into memory |
| `onnx unload BAAI/bge-small-en-v1.5` | Unload a model from memory |
| `onnx run BAAI/bge-small-en-v1.5 "hello world"` | Run an embedding inference test |
| `onnx run cross-encoder/ms-marco "doc text" --query "what is AI"` | Run a rerank inference test |
| `onnx run microsoft/phi-3-mini "Hello"` | Run a text-generation inference test |

## REST API

The daemon listens on `http://localhost:11434` by default.

### POST /v1/embeddings

OpenAI-compatible embedding endpoint. Accepts a single string or a list of strings.

```bash
# Single input
curl -s http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-small-en-v1.5", "input": "hello world"}'

# Batch input
curl -s http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-small-en-v1.5", "input": ["text one", "text two"]}'
```

### POST /v1/rerank

Cohere/Jina-style reranking endpoint. Returns documents ranked by relevance to the query.

```bash
curl -s http://localhost:11434/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "query": "what is AI",
    "documents": ["Artificial intelligence is ...", "The weather today is ..."],
    "top_n": 2
  }'
```

### POST /v1/completions

OpenAI-compatible text completion endpoint.

```bash
curl -s http://localhost:11434/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/phi-3-mini",
    "prompt": "Hello, my name is",
    "max_tokens": 100
  }'
```

### GET /v1/models

List all models currently loaded in the daemon's memory pool.

```bash
curl -s http://localhost:11434/v1/models
```

### GET /health

Health check endpoint.

```bash
curl -s http://localhost:11434/health
```

## Architecture

```
CLI (onnx) ──HTTP──► daemon (FastAPI :11434)
                          │
                ┌─────────┼─────────┐
           ModelPool   Registry   Inference
           (memory)   (SQLite)   backends
```

- **CLI** — Click-based command interface; communicates with the daemon over HTTP for runtime commands (`load`, `unload`, `ps`, `run`) and operates directly on local storage for management commands (`pull`, `list`, `rm`, `show`).
- **Daemon (FastAPI)** — Single uvicorn process that exposes the REST API and owns the model pool lifecycle.
- **ModelPool** — Thread-safe in-memory pool that holds loaded `onnxruntime.InferenceSession` instances. Models are loaded on first request or on explicit `onnx load`.
- **Registry (SQLite)** — Persistent metadata store at `~/.onnx/models/registry.db` tracking model names, tasks, file paths, and pull timestamps.
- **Inference backends** — Task-specific runners for embedding, reranking, and text generation, each using `onnxruntime` and `tokenizers`.

## Supported Model Tasks

| Task | Description | Example model |
|------|-------------|---------------|
| `embedding` | Dense vector representations of text | `BAAI/bge-small-en-v1.5` |
| `rerank` | Cross-encoder relevance scoring of query-document pairs | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| `text-generation` | Autoregressive token generation | `microsoft/phi-3-mini` |

When pulling from HuggingFace the task is inferred automatically from the model card. When importing a local file, specify it explicitly with `--task`.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Daemon port | `11434` | Override with `onnx serve --port <n>` |
| Model storage root | `~/.onnx/models/` | All downloaded and imported models |
| Registry database | `~/.onnx/models/registry.db` | SQLite file; created automatically on first use |

No configuration file is required. All defaults are designed to work out of the box.

## Development

Install development dependencies and run the test suite:

```bash
pip install -e ".[dev]"
pytest tests/
```

Tests use `pytest-asyncio` with `asyncio_mode = auto`. HTTP integration tests use `httpx` against a test client instance of the FastAPI app and do not require a running daemon.

## License

MIT License. See [LICENSE](LICENSE) for details.
