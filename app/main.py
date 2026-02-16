from __future__ import annotations
import asyncio
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from .settings import settings
from .db import make_engine, make_session_factory
from .models import Base, Endpoint
from .catalog import load_catalog
from .schemas import OpenAIModelsResponse
from .admin import router as admin_router
from .router_core import choose_ready_endpoint, health_check_endpoint
from .proxy import proxy_json_or_stream

app = FastAPI(title="vLLM Swapper Router", version="0.1.0")
app.include_router(admin_router)

engine = make_engine(settings.database_url)
SessionLocal = make_session_factory(engine)
Base.metadata.create_all(bind=engine)

CATALOG = load_catalog("config/models.yaml")

# Serve Web UI
app.mount("/ui", StaticFiles(directory="app/ui", html=True), name="ui")

@app.get("/")
def root_ui():
    return FileResponse("app/ui/index.html")


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()}

@app.get("/v1/models", response_model=OpenAIModelsResponse)
def v1_models():
    # List catalog models + mark those with READY endpoints
    with SessionLocal() as db:
        ready = set(
            db.execute(select(Endpoint.model).where(Endpoint.state == "READY")).scalars().all()
        )
    data = []
    for name, m in CATALOG.items():
        data.append({
            "id": name,
            "object": "model",
            "owned_by": "local-slurm",
            "ready": name in ready,
            "meta": {"gpus": m.gpus, "tensor_parallel_size": m.tensor_parallel_size, "notes": m.notes},
        })
    return OpenAIModelsResponse(data=data)

def _resolve_upstream(db: Session, model: str) -> str:
    ep = choose_ready_endpoint(db, model)
    if not ep:
        # CUSTOM ERROR FOR MANUAL SCHEDULING
        msg = (
            f"Model '{model}' is not currently running. "
            f"Please visit the Scheduler UI to launch it: http://{settings.public_hostname}:{settings.router_port}/"
        )
        raise HTTPException(status_code=503, detail=msg)
    return f"http://{ep.host}:{ep.port}"


async def _get_model_from_body(request: Request) -> str:
    try:
        j = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    model = j.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="Missing 'model' in request body")
    return model

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    model = await _get_model_from_body(request)
    with SessionLocal() as db:
        upstream = _resolve_upstream(db, model)
    return await proxy_json_or_stream(request, upstream_url=f"{upstream}/v1/chat/completions")

@app.post("/v1/messages")
async def messages(request: Request):
    model = await _get_model_from_body(request)
    with SessionLocal() as db:
        upstream = _resolve_upstream(db, model)
    return await proxy_json_or_stream(request, upstream_url=f"{upstream}/v1/messages")

async def default_model_worker():
    # If DEFAULT_MODEL is set, ensure it is running when no other READY model is running.
    # NOTE: This does NOT implement request fallback chains (LiteLLM already handles that).
    from .admin import ensure_default_model_running  # local import to avoid cycles
    while True:
        try:
            await ensure_default_model_running()
        except Exception:
            pass
        await asyncio.sleep(10)


async def health_worker():
    while True:
        try:
            with SessionLocal() as db:
                eps = db.execute(select(Endpoint)).scalars().all()
                for e in eps:
                    ok, err = await health_check_endpoint(e.host, e.port)
                    if ok:
                        e.state = "READY"
                        e.last_error = None
                    else:
                        # keep STARTING as STARTING for a bit; mark FAILED if persistent - simplified here.
                        if e.state == "READY":
                            e.state = "FAILED"
                        e.last_error = err
                    e.last_health_at = datetime.utcnow()
                db.commit()
        except Exception:
            # Don't crash the worker; next loop will retry.
            pass
        await asyncio.sleep(3)

@app.on_event("startup")
async def startup():
    asyncio.create_task(health_worker())
    asyncio.create_task(default_model_worker())
