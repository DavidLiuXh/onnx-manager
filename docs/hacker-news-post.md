# Hacker News 推广文章

**日期**：2026-05-15

---

## 标题

```
Show HN: onnx-manager – ollama-style CLI for local ONNX model serving (embedding, rerank, text-gen)
```

---

## 正文

I built this after repeatedly hitting the same friction in AI agent pipelines: you need a local reranker to improve retrieval quality, or an embedding model for semantic search, but there's no standard way to manage and serve them. Every project ends up with its own ad-hoc loader.

ollama solves this beautifully for chat models. Nothing equivalent exists for the task-specific ONNX models that show up in retrieval pipelines.

**What it does**

```bash
onnx pull BAAI/bge-small-en-v1.5
onnx pull cross-encoder/ms-marco-MiniLM-L6-v2
onnx serve --background
```

Then from any code or tool:

```bash
# Embeddings — OpenAI-compatible, drop-in for LangChain/LlamaIndex
curl http://localhost:11434/v1/embeddings \
  -d '{"model": "BAAI/bge-small-en-v1.5", "input": ["doc1", "doc2"]}'

# Reranking — Cohere-style
curl http://localhost:11434/v1/rerank \
  -d '{"model": "cross-encoder/ms-marco-MiniLM-L6-v2",
       "query": "what is AI",
       "documents": ["doc1", "doc2", "doc3"],
       "top_n": 2}'
```

One daemon serves multiple models. Models load on first request, stay in a thread-safe memory pool, and get unloaded explicitly when you want the memory back.

**A few things I learned building this**

*ONNX models on HuggingFace have two layouts.* Small models pack weights into a single `model.onnx`. Large models split into a stub `model.onnx` (~600KB) and a separate `model.onnx_data` (~1–2GB). I initially only downloaded the stub and couldn't figure out why inference was failing. The fix was to call `list_repo_files()` and download everything matching `model.onnx*`.

*Rerankers don't agree on output format.* BGE rerankers output two logits `[neg, pos]` — you softmax and take the last. GTE models output a single logit — you sigmoid it. Treating them the same gives wrong scores. The fix is a shape check: `if vec.shape[0] >= 2: softmax else: sigmoid`.

*Not every model has an ONNX export.* When a repo has only `model.safetensors`, we now check `model_info().tags` before even listing files (saves a round-trip), then offer to convert via `optimum-cli`. If the model uses a custom architecture that optimum can't handle, we search HuggingFace for pre-converted community versions and suggest them directly:

```
Error: custom architecture, optimum-cli cannot export automatically.

Pre-converted ONNX versions found:
  onnx pull onnx-community/gte-multilingual-reranker-base
  onnx pull ConfidentialMind/gte-multilingual-reranker-base-onnx-op14-opt-gpu
```

*Port discovery is annoying.* The daemon writes `~/.onnx/daemon.json` with `{pid, host, port}` on startup. CLI commands read it automatically, so you never need `--port` flags even when running on a non-default port.

**What it doesn't do (yet)**

- GPU (CUDA EP) — CPU only for now
- Streaming text generation
- Quantization / format conversion beyond optimum-cli

**Source:** https://github.com/DavidLiuXh/onnx-manager

---

## 投稿建议

1. 工作日**周二或周三**早上 9–11am ET（北京时间晚上 9–11 点）发，流量最高
2. 发出后前 1 小时不要自己评论，等社区提问再回
3. 如果有人问"为什么不用 fastembed/triton/torchserve"，提前准备答案：fastembed 是 Python SDK 没有 REST server，triton/torchserve 是生产级工具需要 Docker，这个工具的定位是本地开发调试零配置
4. "学到的三件事"那段是 HN 最容易得票的内容，技术诚实 + 具体细节 > 功能列表
