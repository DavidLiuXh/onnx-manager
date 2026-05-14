from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from onnx_manager.daemon.routes import _get_or_load_session
from onnx_manager.inference.text_generation import TextGenerationBackend

router = APIRouter()


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 100
    stream: Optional[bool] = False


@router.post("/v1/completions")
async def create_completion(req: CompletionRequest, request: Request):
    if req.stream:
        raise HTTPException(status_code=501, detail="Streaming is not supported yet")

    pool = request.app.state.pool
    registry = request.app.state.registry
    session = _get_or_load_session(req.model, pool, registry)

    backend = TextGenerationBackend(session)
    text = backend.run(prompt=req.prompt, max_tokens=req.max_tokens)

    return {
        "id": "cmpl-local",
        "object": "text_completion",
        "model": req.model,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
    }
