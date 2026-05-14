from fastapi import HTTPException
from onnx_manager.store.registry import ModelRegistry
from onnx_manager.pool.manager import ModelPool


def _get_or_load_session(model_id: str, pool: ModelPool, registry: ModelRegistry):
    session = pool.get(model_id)
    if session is not None:
        return session
    record = registry.get(model_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_id!r} not found. Run: onnx pull {model_id}",
        )
    try:
        return pool.load(record)
    except MemoryError:
        loaded = pool.list_loaded()
        ids = ", ".join(m["id"] for m in loaded)
        raise HTTPException(
            status_code=503,
            detail=f"Out of memory. Currently loaded: [{ids}]. Try: onnx unload <model>",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {e}. Try: onnx pull {model_id}",
        )
