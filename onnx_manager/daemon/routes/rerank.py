import asyncio
import functools
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional
from onnx_manager.daemon.routes import _get_or_load_session
from onnx_manager.inference.rerank import RerankBackend

router = APIRouter()


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: Optional[int] = None


@router.post("/v1/rerank")
async def rerank(req: RerankRequest, request: Request):
    pool = request.app.state.pool
    registry = request.app.state.registry
    session = _get_or_load_session(req.model, pool, registry)

    backend = RerankBackend(session)
    loop = asyncio.get_event_loop()
    scores = await loop.run_in_executor(
        None,
        functools.partial(backend.run, query=req.query, documents=req.documents)
    )

    results = sorted(
        [{"index": i, "score": s, "document": req.documents[i]} for i, s in enumerate(scores)],
        key=lambda x: x["score"],
        reverse=True,
    )
    if req.top_n is not None:
        results = results[: req.top_n]

    return {"model": req.model, "results": results}
