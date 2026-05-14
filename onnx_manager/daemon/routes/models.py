from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/v1/models")
async def list_models(request: Request):
    pool = request.app.state.pool
    loaded = pool.list_loaded()
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "task": m["task"]}
            for m in loaded
        ],
    }


@router.get("/v1/models/{model_id:path}")
async def get_model(model_id: str, request: Request):
    pool = request.app.state.pool
    session = pool.get(model_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} is not loaded")
    return {"id": model_id, "object": "model", "task": session.task}


@router.post("/v1/models/{model_id:path}/load")
async def load_model(model_id: str, request: Request):
    pool = request.app.state.pool
    registry = request.app.state.registry
    record = registry.get(model_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found")
    pool.load(record)
    return {"id": model_id, "status": "loaded"}


@router.post("/v1/models/{model_id:path}/unload")
async def unload_model(model_id: str, request: Request):
    pool = request.app.state.pool
    pool.unload(model_id)
    return {"id": model_id, "status": "unloaded"}
