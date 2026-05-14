# ONNX Manager 技术设计文档

**日期**：2026-05-14

---

## 1. 标题和概述

### 背景

ONNX Runtime 是高性能推理引擎，在 Agent 场景中常用于本地运行 embedding、rerank 等推理模型。现有工具生态碎片化——没有一个像 ollama 一样的工具能统一管理和服务多个 ONNX 模型。本项目填补这一空白。

### 目标

构建一个面向开发者的本地 ONNX 模型管理与服务工具，支持：

- 从 HuggingFace 下载或从本地导入 ONNX 模型
- 一个 daemon 进程同时服务多个模型
- OpenAI 兼容的 REST API（embedding、rerank、text-generation）
- 类 ollama 的 CLI 体验

### 不在范围内

- 模型训练
- 模型转换（ONNX 格式转换）
- 生产级部署（多机、负载均衡）
- Streaming 生成

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────┐
│                   onnx CLI                      │
│  pull / push / list / load / unload / ps / rm  │
└────────────────────┬────────────────────────────┘
                     │ HTTP (localhost)
┌────────────────────▼────────────────────────────┐
│              onnx daemon (守护进程)               │
│                                                  │
│  ┌──────────────┐   ┌───────────────────────┐   │
│  │ Model Store  │   │   REST API Server     │   │
│  │  ~/.onnx/    │   │   (FastAPI)           │   │
│  │  models/     │   │                       │   │
│  │  registry.db │   │  POST /v1/embeddings  │   │
│  └──────────────┘   │  POST /v1/rerank      │   │
│                     │  POST /v1/completions │   │
│  ┌──────────────┐   │  GET  /v1/models      │   │
│  │ Model Pool   │   └───────────────────────┘   │
│  │  (内存)      │                                │
│  │  bge-small   │                                │
│  │  ms-marco    │                                │
│  │  phi-3-mini  │                                │
│  └──────────────┘                                │
└─────────────────────────────────────────────────┘
```

### 核心设计决策

- **daemon 生命周期**：由 `onnx serve` 手动启动，`onnx stop` 停止，不随系统自启
- **通信方式**：CLI 通过 `localhost:11434`（默认）与 daemon 通信
- **存储分离**：模型文件存 `~/.onnx/models/`，元数据存 SQLite
- **按需加载**：API 调用时若模型未加载，自动触发加载，响应头附加 `X-Model-Load-Time-Ms`

---

## 3. CLI 命令设计

```bash
# 模型管理
onnx pull BAAI/bge-small-en-v1.5                       # 从 HuggingFace 下载
onnx pull ./path/to/model.onnx --name mymodel --task embedding  # 导入本地文件
onnx list                                               # 列出已下载模型
onnx rm BAAI/bge-small-en-v1.5                         # 删除本地模型
onnx show BAAI/bge-small-en-v1.5                       # 查看元数据

# 服务管理
onnx serve                                              # 前台启动 daemon
onnx serve --port 11434 --background                   # 后台启动
onnx stop                                               # 停止后台 daemon
onnx ps                                                 # 查看内存中已加载的模型

# 运行时管理
onnx load BAAI/bge-small-en-v1.5                       # 手动加载到内存
onnx unload BAAI/bge-small-en-v1.5                     # 从内存卸载

# 快捷测试
onnx run BAAI/bge-small-en-v1.5 "hello world"          # 单次推理测试
```

### 行为约定

- API 调用和 `onnx run` 均支持按需自动加载，无需手动 `load`
- `onnx pull` 自动从 HuggingFace `config.json` 的 `pipeline_tag` 识别任务类型
- 本地导入必须通过 `--task` 显式指定任务类型

---

## 4. REST API 设计

### Embedding（OpenAI 兼容）

```
POST /v1/embeddings
{
  "model": "BAAI/bge-small-en-v1.5",
  "input": "hello world"        // 或 ["text1", "text2"]
}

→ {
  "data": [{"embedding": [...], "index": 0, "object": "embedding"}],
  "model": "BAAI/bge-small-en-v1.5",
  "usage": {"prompt_tokens": 5, "total_tokens": 5}
}
```

### Rerank（Cohere/Jina 风格）

```
POST /v1/rerank
{
  "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
  "query": "what is AI",
  "documents": ["doc1", "doc2", "doc3"],
  "top_n": 3                    // 可选
}

→ {
  "results": [
    {"index": 1, "score": 0.92, "document": "doc2"},
    ...
  ]
}
```

### Text Generation（OpenAI 兼容）

```
POST /v1/completions
{
  "model": "microsoft/phi-3-mini",
  "prompt": "Hello, my name is",
  "max_tokens": 100,
  "stream": false               // 暂不支持 streaming
}
```

### 其他接口

```
GET  /v1/models                 # 列出所有已加载模型
GET  /v1/models/{model_id}      # 单个模型状态
GET  /v1/models/{model_id}/load    # 手动加载模型
POST /v1/models/{model_id}/unload  # 手动卸载模型
GET  /health                    # 健康检查
```

---

## 5. 模块划分

### 目录结构

```
onnx_manager/                       # Python 包根目录
├── __init__.py
├── config.py                       # 常量：默认端口、主目录 (~/.onnx)
├── store/
│   ├── __init__.py
│   ├── registry.py                 # SQLite CRUD：增删改查模型记录
│   └── downloader.py               # HuggingFace 下载 + 本地文件导入
├── pool/
│   ├── __init__.py
│   ├── session.py                  # OnnxSession：封装 ort.InferenceSession + tokenizer
│   └── manager.py                  # ModelPool 单例：加载/卸载/查询 session
├── inference/
│   ├── __init__.py
│   ├── base.py                     # InferenceBackend 抽象基类
│   ├── embedding.py                # EmbeddingBackend：tokenize → run → normalize
│   ├── rerank.py                   # RerankBackend：query+docs → scores
│   └── text_generation.py          # TextGenerationBackend：prompt → token 循环
├── daemon/
│   ├── __init__.py
│   ├── app.py                      # FastAPI app 工厂函数
│   ├── lifecycle.py                # PID 文件管理、优雅退出
│   └── routes/
│       ├── __init__.py             # 共享辅助函数（_get_or_load_session）
│       ├── health.py               # GET /health
│       ├── models.py               # GET /v1/models, GET /v1/models/{model_id}
│       ├── embeddings.py           # POST /v1/embeddings
│       ├── rerank.py               # POST /v1/rerank
│       └── completions.py          # POST /v1/completions
└── cli/
    ├── __init__.py
    ├── main.py                     # Click group 根，注册为 `onnx` 入口
    ├── client.py                   # DaemonClient：封装与 daemon 通信的 HTTP 客户端
    └── commands/
        ├── __init__.py
        ├── pull.py                 # onnx pull
        ├── serve.py                # onnx serve, onnx stop
        ├── model.py                # onnx list/rm/show/load/unload/ps
        └── run.py                  # onnx run
```

### 各模块职责

| 模块 | 职责 |
|------|------|
| `config.py` | 全局常量：`ONNX_HOME`、`MODELS_DIR`、`REGISTRY_PATH`、`PID_FILE`、`DEFAULT_PORT`、`PIPELINE_TAG_MAP` |
| `store/registry.py` | SQLite 元数据 CRUD，提供 `ModelRegistry` 类和 `ModelRecord` 数据类 |
| `store/downloader.py` | HuggingFace 下载（优先 `onnx/model.onnx`，fallback 根目录）+ 本地文件导入 + ONNX 校验 |
| `pool/session.py` | `OnnxSession`：封装 `ort.InferenceSession` + tokenizer，统一加载入口 |
| `pool/manager.py` | `ModelPool` 单例：线程安全地管理内存中的 session，防止重复加载 |
| `inference/` | 各后端纯推理逻辑，不感知 HTTP 或文件路径 |
| `daemon/app.py` | FastAPI app 工厂，将 pool 和 registry 注入 `app.state` |
| `daemon/lifecycle.py` | PID 文件读写、daemon 启动/停止/崩溃恢复 |
| `daemon/routes/` | HTTP 路由处理，调用 pool/inference 完成推理 |
| `cli/client.py` | `DaemonClient`：thin HTTP wrapper，供 CLI 命令调用 |
| `cli/commands/` | Click 命令实现，直接操作 registry 或通过 client 与 daemon 通信 |

### 依赖关系

```
CLI → client.py → daemon REST API
daemon routes → pool.manager → session → inference backends
daemon routes → store.registry（查元数据）
pull / run → store.downloader + store.registry
```

### 约束

- `inference/` 各后端只负责纯推理，不感知 HTTP 或文件路径
- `pool/manager.py` 是唯一操作 ONNX Runtime session 的地方，防止重复加载
- `store/registry.py` 用 SQLite 管理元数据，文件系统存模型文件，通过 `model_id` 关联

---

## 6. 数据存储

### 文件系统布局

```
~/.onnx/
├── registry.db
├── daemon.pid
├── daemon.log
└── models/
    ├── BAAI--bge-small-en-v1.5/
    │   ├── model.onnx
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   └── meta.json
    └── cross-encoder--ms-marco-MiniLM-L6-v2/
        ├── model.onnx
        ├── tokenizer.json
        └── meta.json
```

### SQLite `models` 表结构

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

### HuggingFace 下载策略

- 优先下载 `onnx/model.onnx`，fallback 到仓库根目录 `.onnx` 文件
- 同时下载 `tokenizer.json` + `tokenizer_config.json`（best effort）
- 模型 ID 中的 `/` 替换为 `--` 作为目录名（如 `BAAI/bge-small-en-v1.5` → `BAAI--bge-small-en-v1.5`）
- 任务类型从 `config.json` 的 `pipeline_tag` 字段自动识别，映射规则：

| `pipeline_tag` | task |
|---|---|
| `feature-extraction` | `embedding` |
| `sentence-similarity` | `embedding` |
| `text-ranking` | `rerank` |
| `text-generation` | `text-generation` |

---

## 7. 错误处理

| 场景 | 行为 |
|------|------|
| CLI 调用但 daemon 未运行 | 提示 `run 'onnx serve' first`，退出 |
| daemon 端口被占用 | 报错：`Port 11434 is already in use. Use --port to specify another.` |
| daemon 重复启动 | 检测 `daemon.pid`，进程存活则报错退出 |
| 下载中断 | 断点续传（huggingface_hub 原生支持） |
| 下载临时文件 | 写入临时目录，完成后原子移动，避免半截文件 |
| 本地导入校验 | 用 `onnx.checker.check_model` 验证文件有效性 |
| 模型未加载 | 自动加载，响应头加 `X-Model-Load-Time-Ms` |
| 模型文件损坏 | 返回 500，提示重新 `onnx pull` |
| 批量输入超长 | 返回 400，提示最大 token 限制 |
| 内存不足 | 返回 503，列出已加载模型，建议 `onnx unload` |
| Ctrl+C / `onnx stop` | 等待当前请求完成（最多 30s），清理 PID 文件，优雅退出 |
| daemon 崩溃残留 PID | 下次 `onnx serve` 检测到进程不存在，自动清理 PID 文件 |

---

## 8. 技术选型

| 组件 | 选型 |
|------|------|
| 语言 | Python 3.10+ |
| CLI 框架 | Click |
| HTTP 服务 | FastAPI + uvicorn |
| ONNX 推理 | onnxruntime |
| Tokenizer | tokenizers（HuggingFace） |
| HF 下载 | huggingface_hub |
| 元数据存储 | SQLite（标准库 sqlite3） |
| ONNX 校验 | onnx |
| 数值计算 | numpy |
| HTTP 客户端 | httpx |
| Rerank softmax | scipy |

---

## 9. 实现要点

### config 模块的 monkeypatch 设计

`config.py` 中所有路径常量（`ONNX_HOME`、`MODELS_DIR`、`REGISTRY_PATH`、`PID_FILE`）均为模块级变量，而非硬编码到各模块内部。测试时通过 pytest 的 `monkeypatch.setattr` 将这些常量替换为临时目录，使得各模块（registry、downloader、lifecycle）在测试中自动使用隔离路径，无需改动业务代码。

```python
# tests/conftest.py 共享 fixture
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

### ModelPool 线程安全设计

`ModelPool` 使用 `threading.Lock` 保护内部 `_sessions` 字典的所有读写操作。`load` 操作在持锁状态下检查 model_id 是否已存在，确保幂等性（多次调用只创建一个 session）。`get`、`unload`、`list_loaded` 同样在锁内操作，避免并发请求导致 session 重复创建或数据竞争。

### 推理后端的 tokenizer pair encoding

Rerank 后端（`RerankBackend`）使用 cross-encoder 架构，需要将 query 和 document 作为文本对输入。实现方式是将 query 和 doc 拼接为 `"{query} [SEP] {doc}"` 后调用 `tokenizer.encode`，利用 tokenizer 的内置分隔符处理，输出 `input_ids` 和 `attention_mask`，再喂入 ONNX 模型。logits 经 softmax 后取正类概率（最后一维）作为相关性得分。

### asyncio 事件循环不阻塞设计（run_in_executor）

FastAPI 路由中的推理调用（`EmbeddingBackend.run`、`RerankBackend.run`、`TextGenerationBackend.run`）均为 CPU 密集型同步操作。若直接在 async 路由中调用，会阻塞 uvicorn 的事件循环，导致其他请求无法处理。正确做法是通过 `asyncio.get_event_loop().run_in_executor(None, ...)` 将推理任务提交到线程池，使事件循环在推理期间仍可处理其他请求。

### daemon lifecycle PID 管理

daemon 启动时写入 `~/.onnx/daemon.pid`，进程退出时（正常退出或 SIGTERM）删除该文件。`onnx serve` 启动前先检查 PID 文件：若文件存在且进程存活（`os.kill(pid, 0)` 不抛出异常），则拒绝启动并提示用户；若文件存在但进程已死（stale PID），则自动清理文件再启动。`onnx stop` 通过读取 PID 文件获取进程号，发送 `SIGTERM` 触发优雅退出。

---

## 10. 未来扩展

以下功能不在当前范围内，但架构上可自然扩展：

- **Text generation streaming**：添加 `stream: true` 支持，使用 FastAPI 的 `StreamingResponse` 按 token 返回
- **模型量化**：在 downloader 中增加 INT8/FP16 量化选项，`OnnxSession` 加载时选择对应文件
- **GPU 加速**：`OnnxSession` 初始化时通过 `providers` 参数切换 `CUDAExecutionProvider`
- **模型版本管理**：在 `models` 表增加 `version` 字段，目录名加版本后缀，支持同一模型多版本共存
- **Web UI 管理界面**：在 FastAPI 上挂载静态文件或增加 `/ui` 路由，提供可视化管理界面
