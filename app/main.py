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
from . import slurm

app = FastAPI(title="vLLM Swapper Router", version="0.3.0")
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
                    prev_state = e.state
                    ok, err = await health_check_endpoint(e.host, e.port)

                    if ok:
                        e.state = "READY"
                        e.last_error = None
                    else:
                        if e.state == "READY":
                            e.state = "FAILED"
                        elif e.state == "STARTING":
                            if e.created_at and (now - _ensure_aware_dt(e.created_at)).total_seconds() > 600:
                                e.state = "FAILED"
                        e.last_error = err

                    e.last_health_at = now

                    # If endpoint just became READY, mark corresponding lease RUNNING
                    if prev_state != "READY" and e.state == "READY":
                        lease = db.execute(
                            select(Lease).where(Lease.slurm_job_id == e.slurm_job_id)
                        ).scalars().first()
                        if lease and lease.state in ("SUBMITTED", "STARTING"):
                            lease.state = "RUNNING"

                db.commit()
        except Exception as e:
            print(f"health_worker error: {e}")
        await asyncio.sleep(3)


async def planned_submit_worker():
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
    while True:
        try:
            now = datetime.now(timezone.utc)
            with SessionLocal() as db:
                eps = db.execute(
                    select(Endpoint).where(Endpoint.state.in_(["READY", "STARTING", "FAILED"]))
                ).scalars().all()
                for e in eps:
                    lease = db.execute(
                        select(Lease).where(Lease.slurm_job_id == e.slurm_job_id)
                    ).scalars().first()
                    if lease and lease.end_at and _ensure_aware_dt(lease.end_at) < now:
                        e.state = "STOPPED"
                        if lease.state == "RUNNING":
                            lease.state = "ENDED"
                        elif lease.state == "FAILED":
                            # Leave as FAILED but stop the endpoint
                            pass
                    elif lease and lease.state in ("CANCELED", "ENDED"):
                        e.state = "STOPPED"
                    # Don't auto-stop FAILED leases — let them remain visible
                db.commit()
        except Exception as e:
            print(f"endpoint_cleanup_worker error: {e}")
        await asyncio.sleep(15)


async def slurm_reconcile_worker():
    """
    Reconcile leases/endpoints with Slurm reality.
    - If Slurm job is no longer in squeue, mark lease ENDED/FAILED (depending on endpoint readiness).
    - FAILED leases remain visible until their end_at passes, then get marked ENDED.
    """
    while True:
        try:
            now = datetime.now(timezone.utc)
            with SessionLocal() as db:
                active = db.execute(
                    select(Lease).where(Lease.state.in_(["SUBMITTED", "STARTING", "RUNNING", "FAILED"]))
                ).scalars().all()

                for l in active:
                    # For FAILED leases: check if end_at has passed, then mark ENDED
                    if l.state == "FAILED":
                        if l.end_at and _ensure_aware_dt(l.end_at) < now:
                            l.state = "ENDED"
                            if l.slurm_job_id:
                                ep = db.execute(
                                    select(Endpoint).where(Endpoint.slurm_job_id == l.slurm_job_id)
                                ).scalars().first()
                                if ep:
                                    ep.state = "STOPPED"
                        continue

                    if not l.slurm_job_id:
                        continue
                    state = slurm.squeue_job_state(l.slurm_job_id)

                    # Not in squeue anymore => finished/failed/canceled (or very fast fail)
                    if state is None:
                        ep = db.execute(
                            select(Endpoint).where(Endpoint.slurm_job_id == l.slurm_job_id)
                        ).scalars().first()

                        # If it never became READY (or is FAILED), treat as FAILED.
                        if ep is None or ep.state in ("FAILED", "STARTING"):
                            l.state = "FAILED"
                            if ep:
                                ep.state = "FAILED"
                        else:
                            # It was READY at some point, so treat disappearance as ENDED unless user canceled.
                            if l.state not in ("CANCELED",):
                                l.state = "ENDED"
                            if ep:
                                ep.state = "STOPPED"

                db.commit()
        except Exception as e:
            print(f"slurm_reconcile_worker error: {e}")
        await asyncio.sleep(5)


@app.on_event("startup")
async def startup():
    asyncio.create_task(health_worker())
    asyncio.create_task(planned_submit_worker())
    asyncio.create_task(endpoint_cleanup_worker())
    asyncio.create_task(slurm_reconcile_worker())
