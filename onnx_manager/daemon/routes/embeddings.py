import asyncio
import functools
import time
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from onnx_manager.daemon.routes import _get_or_load_session
from onnx_manager.inference.embedding import EmbeddingBackend

router = APIRouter()


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]


@router.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest, request: Request):
    pool = request.app.state.pool
    registry = request.app.state.registry
    already_loaded = pool.get(req.model) is not None
    t0 = time.monotonic()
    session = _get_or_load_session(req.model, pool, registry)
    load_ms = int((time.monotonic() - t0) * 1000)

    texts = [req.input] if isinstance(req.input, str) else req.input
    backend = EmbeddingBackend(session)
    loop = asyncio.get_event_loop()
    vectors = await loop.run_in_executor(None, backend.run, texts)

    # Approximate token count (word-based, not subword)
    total_tokens = sum(len(t.split()) for t in texts)
    body = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ],
        "model": req.model,
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    }
    headers = {} if already_loaded else {"X-Model-Load-Time-Ms": str(load_ms)}
    return JSONResponse(content=body, headers=headers)
