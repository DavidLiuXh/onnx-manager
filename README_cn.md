# onnx-manager

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

面向 AI 开发者的本地 ONNX 模型管理与推理服务工具。

## 概述

onnx-manager 通过统一的守护进程和兼容 OpenAI 的 REST API，让你可以下载、管理和部署 ONNX 模型。它解决了在本地运行多个 ONNX 推理模型时需要反复编写服务端样板代码的问题。可以将其理解为 ONNX 版的 ollama：一个 CLI 拉取模型，一个守护进程负责服务，一个 HTTP 接口统一调用。

## 功能特性

- 直接从 HuggingFace Hub 拉取 ONNX 模型，或导入本地 `.onnx` 文件
- 统一将模型存储在 `~/.onnx/models/` 目录下，并以 SQLite 作为模型注册表
- 单一守护进程通过线程安全的内存池并发服务多个模型
- 兼容 OpenAI 的 REST API，支持向量嵌入、重排序和文本生成任务
- 支持懒加载，以及对内存中模型的显式加载与卸载控制
- CLI 交互体验参照 ollama 设计（`pull`、`list`、`rm`、`serve`、`ps`、`run`）

## 环境要求

- Python 3.10 及以上版本
- 主要运行时依赖（安装时自动处理）：
  - `onnxruntime>=1.17`
  - `fastapi>=0.110`、`uvicorn>=0.29`
  - `tokenizers>=0.19`
  - `huggingface-hub>=0.22`
  - `numpy>=1.26`、`scipy>=1.12`

## 安装

```bash
git clone https://github.com/your-org/onnx-manager.git
cd onnx-manager
pip install -e .
```

安装完成后，`onnx` 命令即可在 shell 中直接使用。

## 快速开始

**第一步 — 从 HuggingFace 拉取模型**

```bash
onnx pull BAAI/bge-small-en-v1.5
```

**第二步 — 启动守护进程**

```bash
onnx serve --background
```

**第三步 — 发送请求**

```bash
curl -s http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-small-en-v1.5", "input": "hello world"}' | jq .
```

## CLI 命令参考

### 模型管理

| 命令 | 说明 |
|---------|-------------|
| `onnx pull BAAI/bge-small-en-v1.5` | 从 HuggingFace Hub 下载模型 |
| `onnx pull ./model.onnx --name mymodel --task embedding` | 导入本地 ONNX 文件 |
| `onnx list` | 列出所有已下载的模型 |
| `onnx rm BAAI/bge-small-en-v1.5` | 从本地存储中删除模型 |
| `onnx show BAAI/bge-small-en-v1.5` | 查看模型元数据 |

### 守护进程控制

| 命令 | 说明 |
|---------|-------------|
| `onnx serve` | 在前台启动守护进程（端口 11434） |
| `onnx serve --port 11434 --background` | 在后台启动守护进程 |
| `onnx stop` | 停止正在运行的守护进程 |
| `onnx ps` | 列出当前已加载到内存中的模型 |

### 运行时操作

| 命令 | 说明 |
|---------|-------------|
| `onnx load BAAI/bge-small-en-v1.5` | 手动将模型加载到内存 |
| `onnx unload BAAI/bge-small-en-v1.5` | 从内存中卸载模型 |
| `onnx run BAAI/bge-small-en-v1.5 "hello world"` | 执行向量嵌入推理测试 |
| `onnx run cross-encoder/ms-marco "doc text" --query "what is AI"` | 执行重排序推理测试 |
| `onnx run microsoft/phi-3-mini "Hello"` | 执行文本生成推理测试 |

## REST API

守护进程默认监听 `http://localhost:11434`。

### POST /v1/embeddings

兼容 OpenAI 的向量嵌入接口。接受单个字符串或字符串列表作为输入。

```bash
# 单条输入
curl -s http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-small-en-v1.5", "input": "hello world"}'

# 批量输入
curl -s http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-small-en-v1.5", "input": ["text one", "text two"]}'
```

### POST /v1/rerank

兼容 Cohere/Jina 风格的重排序接口。返回按与查询的相关性排序后的文档列表。

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

兼容 OpenAI 的文本补全接口。

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

列出守护进程内存池中当前已加载的所有模型。

```bash
curl -s http://localhost:11434/v1/models
```

### GET /health

健康检查接口。

```bash
curl -s http://localhost:11434/health
```

## 架构设计

```
CLI (onnx) ──HTTP──► daemon (FastAPI :11434)
                          │
                ┌─────────┼─────────┐
           ModelPool   Registry   Inference
           (memory)   (SQLite)   backends
```

- **CLI** — 基于 Click 的命令行界面。运行时命令（`load`、`unload`、`ps`、`run`）通过 HTTP 与守护进程通信；模型管理命令（`pull`、`list`、`rm`、`show`）直接操作本地存储。
- **守护进程（FastAPI）** — 单个 uvicorn 进程，对外暴露 REST API，并负责管理模型池的完整生命周期。
- **ModelPool** — 线程安全的内存池，持有已加载的 `onnxruntime.InferenceSession` 实例。模型在首次请求时自动加载，或通过 `onnx load` 显式加载。
- **注册表（SQLite）** — 持久化元数据存储，位于 `~/.onnx/models/registry.db`，记录模型名称、任务类型、文件路径和拉取时间戳。
- **推理后端** — 针对向量嵌入、重排序和文本生成各自实现的任务专用运行器，均基于 `onnxruntime` 和 `tokenizers`。

## 支持的模型任务

| 任务 | 说明 | 示例模型 |
|------|-------------|---------------|
| `embedding` | 文本的稠密向量表示 | `BAAI/bge-small-en-v1.5` |
| `rerank` | 基于交叉编码器的查询-文档相关性打分 | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| `text-generation` | 自回归式 token 生成 | `microsoft/phi-3-mini` |

从 HuggingFace 拉取模型时，任务类型会根据模型卡片自动推断。导入本地文件时，需通过 `--task` 参数显式指定。

## 配置说明

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| 守护进程端口 | `11434` | 可通过 `onnx serve --port <n>` 覆盖 |
| 模型存储根目录 | `~/.onnx/models/` | 所有已下载和导入的模型均存放于此 |
| 注册表数据库 | `~/.onnx/models/registry.db` | SQLite 文件，首次使用时自动创建 |

无需任何配置文件，所有默认值均可开箱即用。

## 开发

安装开发依赖并运行测试套件：

```bash
pip install -e ".[dev]"
pytest tests/
```

测试使用 `pytest-asyncio`，并设置 `asyncio_mode = auto`。HTTP 集成测试通过 `httpx` 对 FastAPI 应用的测试客户端实例发起请求，无需启动实际的守护进程。

## 许可证

MIT 许可证。详情请参阅 [LICENSE](LICENSE)。
