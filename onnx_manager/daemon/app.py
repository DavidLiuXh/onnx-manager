from fastapi import FastAPI
from onnx_manager.pool.manager import ModelPool
from onnx_manager.store.registry import ModelRegistry
from onnx_manager.daemon.routes.health import router as health_router
from onnx_manager.daemon.routes.models import router as models_router
from onnx_manager.daemon.routes.embeddings import router as embeddings_router
from onnx_manager.daemon.routes.rerank import router as rerank_router
from onnx_manager.daemon.routes.completions import router as completions_router


def create_app(pool: ModelPool, registry: ModelRegistry) -> FastAPI:
    app = FastAPI(title="ONNX Manager", version="0.1.0")
    app.state.pool = pool
    app.state.registry = registry

    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(embeddings_router)
    app.include_router(rerank_router)
    app.include_router(completions_router)

    return app
