from __future__ import annotations
import asyncio
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from .settings import settings
from .db import make_engine, make_session_factory
from .models import Base, Endpoint, Lease
from .catalog import load_catalog
from .schemas import OpenAIModelsResponse
from .admin import router as admin_router
from .router_core import choose_ready_endpoint, health_check_endpoint
from .proxy import proxy_json_or_stream
from .admin import _submit_to_slurm  # internal helper

app = FastAPI(title="vLLM Swapper Router", version="0.2.0")
app.include_router(admin_router)

engine = make_engine(settings.database_url)
SessionLocal = make_session_factory(engine)
Base.metadata.create_all(bind=engine)

CATALOG = load_catalog("config/models.yaml")

app.mount("/ui", StaticFiles(directory="app/ui", html=True), name="ui")

@app.get("/")
def root_ui():
    return FileResponse("app/ui/index.html")

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}

@app.get("/v1/models", response_model=OpenAIModelsResponse)
def v1_models():
    with SessionLocal() as db:
        ready = set(db.execute(select(Endpoint.model).where(Endpoint.state == "READY")).scalars().all())
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
        msg = (
            f"Model '{model}' is not currently running. "
            f"Please visit the Scheduler UI to start it: http://{settings.public_hostname}:{settings.router_port}/"
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
    from .admin import ensure_default_model_running
    while True:
        try:
            await ensure_default_model_running()
        except Exception:
            pass
        await asyncio.sleep(10)

def _ensure_aware_dt(dt: datetime) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


async def health_worker():
    while True:
        try:
            with SessionLocal() as db:
                eps = db.execute(select(Endpoint)).scalars().all()
                now = datetime.now(timezone.utc)
                for e in eps:
                    ok, err = await health_check_endpoint(e.host, e.port)
                    if ok:
                        e.state = "READY"
                        e.last_error = None
                    else:
                        if e.state == "READY":
                            e.state = "FAILED"
                        elif e.state == "STARTING":
                            # If stuck in STARTING for >10 min, mark FAILED
                            if e.created_at and (now - _ensure_aware_dt(e.created_at)).total_seconds() > 600:
                                e.state = "FAILED"
                        e.last_error = err
                    e.last_health_at = now
                db.commit()
        except Exception as e:
            print(f"health_worker error: {e}")
        await asyncio.sleep(3)


async def planned_submit_worker():
    """
    Submits PLANNED bookings shortly before begin_at.
    """
    while True:
        try:
            now = datetime.now(timezone.utc)
            lead = timedelta(seconds=settings.scheduler_submit_lead_seconds)

            with SessionLocal() as db:
                planned = db.execute(
                    select(Lease).where(
                        Lease.state == "PLANNED",
                        Lease.begin_at != None  # noqa
                    ).order_by(Lease.begin_at.asc())
                ).scalars().all()

                for l in planned:
                    if not l.begin_at:
                        continue
                    if l.begin_at <= now + lead:
                        try:
                            job_id = _submit_to_slurm(l)
                            l.slurm_job_id = job_id
                            l.state = "SUBMITTED"
                        except Exception as e:
                            l.state = "FAILED"
                            print(f"Failed to submit planned lease {l.id}: {e}")
                db.commit()
        except Exception as e:
            print(f"planned_submit_worker error: {e}")

        await asyncio.sleep(5)


async def endpoint_cleanup_worker():
    """Remove endpoints whose lease has ended or whose Slurm job is gone."""
    while True:
        try:
            now = datetime.now(timezone.utc)
            with SessionLocal() as db:
                eps = db.execute(
                    select(Endpoint).where(Endpoint.state.in_(["READY", "STARTING", "FAILED"]))
                ).scalars().all()
                for e in eps:
                    # Check if corresponding lease has ended
                    lease = db.execute(
                        select(Lease).where(Lease.slurm_job_id == e.slurm_job_id)
                    ).scalars().first()
                    if lease and lease.end_at and _ensure_aware_dt(lease.end_at) < now:
                        e.state = "STOPPED"
                        if lease.state == "RUNNING":
                            lease.state = "ENDED"
                    elif lease and lease.state in ("CANCELED", "FAILED"):
                        e.state = "STOPPED"
                db.commit()
        except Exception as e:
            print(f"endpoint_cleanup_worker error: {e}")
        await asyncio.sleep(15)


@app.on_event("startup")
async def startup():
    asyncio.create_task(health_worker())
    asyncio.create_task(default_model_worker())
    asyncio.create_task(planned_submit_worker())
    asyncio.create_task(endpoint_cleanup_worker())
