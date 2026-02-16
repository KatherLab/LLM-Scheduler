from __future__ import annotations
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select
import os

from .settings import settings
from .db import make_engine, make_session_factory
from .models import Lease, Endpoint
from .schemas import LeaseCreate, LeaseOut, LeaseExtend, EndpointRegister, EndpointOut
from .catalog import load_catalog
from .ports import PortAllocator
from . import slurm

router = APIRouter(prefix="/admin", tags=["admin"])

engine = make_engine(settings.database_url)
SessionLocal = make_session_factory(engine)

CATALOG = load_catalog("config/models.yaml")
PORTS = PortAllocator(settings.port_min, settings.port_max)

def _time_limit_from_duration(seconds: int) -> str:
    # Slurm time limit as HH:MM:SS
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def _lease_to_out(l: Lease) -> LeaseOut:
    return LeaseOut(
        id=l.id,
        model=l.model,
        owner=l.owner,
        state=l.state,
        slurm_job_id=l.slurm_job_id,
        host=settings.public_hostname,
        port=l.requested_port,
        requested_gpus=l.requested_gpus,
        requested_tp=l.requested_tp,
        begin_at=l.begin_at,
        end_at=l.end_at,
        created_at=l.created_at,
    )

@router.get("/leases", response_model=list[LeaseOut])
def list_leases():
    with SessionLocal() as db:
        leases = db.execute(select(Lease).order_by(Lease.id.desc())).scalars().all()
        return [_lease_to_out(l) for l in leases]

@router.get("/endpoints", response_model=list[EndpointOut])
def list_endpoints():
    with SessionLocal() as db:
        eps = db.execute(select(Endpoint).order_by(Endpoint.id.desc())).scalars().all()
        return [EndpointOut(
            id=e.id, model=e.model, host=e.host, port=e.port, slurm_job_id=e.slurm_job_id, state=e.state,
            last_health_at=e.last_health_at, last_error=e.last_error, created_at=e.created_at
        ) for e in eps]

@router.post("/leases", response_model=LeaseOut)
def create_lease(req: LeaseCreate):
    if req.model not in CATALOG:
        raise HTTPException(status_code=404, detail=f"Unknown model '{req.model}'")

    cat = CATALOG[req.model]
    gpus = req.gpus or cat.gpus
    tp = req.tensor_parallel_size or cat.tensor_parallel_size
    port = PORTS.allocate(key=f"lease-{req.model}-{datetime.utcnow().timestamp()}")

    begin_at = req.begin_at
    end_at = (begin_at or datetime.utcnow()) + timedelta(seconds=req.duration_seconds)

    lease = Lease(
        model=req.model,
        requested_gpus=gpus,
        requested_tp=tp,
        requested_port=port,
        owner=req.owner,
        begin_at=begin_at,
        end_at=end_at,
        model_path=cat.model_path,
        tool_args=req.tool_args if req.tool_args is not None else cat.tool_args,
        extra_args=req.extra_args if req.extra_args is not None else cat.extra_args,
        reasoning_parser=req.reasoning_parser if req.reasoning_parser is not None else cat.reasoning_parser,
        gpu_memory_utilization=str(req.gpu_memory_utilization if req.gpu_memory_utilization is not None else cat.gpu_memory_utilization),
        state="REQUESTED",
    )

    time_limit = _time_limit_from_duration(req.duration_seconds)
    env = {
        "MODEL_PATH": lease.model_path,
        "SERVED_MODEL_NAME": lease.model,
        "TP_SIZE": str(lease.requested_tp),
        "PORT": str(lease.requested_port),
        "API_KEY": "test",  # keep private; LiteLLM should sit in front anyway
        "GPU_MEM_UTIL": lease.gpu_memory_utilization or "0.95",
        "EXTRA_ARGS": lease.extra_args or "",
        "TOOL_ARGS": lease.tool_args or "",
        "REASONING_PARSER": lease.reasoning_parser or "",
        "ROUTER_REGISTER_URL": f"http://{settings.public_hostname}:{settings.router_port}/admin/endpoints/register",
    }

    try:
        res = slurm.submit_vllm_job(
            template_path=settings.sbatch_template_path,
            job_name=f"vllm-{lease.model}",
            gpus=lease.requested_gpus,
            time_limit=time_limit,
            begin=lease.begin_at,
            env=env,
            partition=settings.slurm_partition,
            account=settings.slurm_account,
            qos=settings.slurm_qos,
            nodelist=settings.slurm_nodelist,
            cpus_per_task=settings.slurm_cpus_per_task,
        )
    except Exception as e:
        PORTS.release(key=str(port))
        raise HTTPException(status_code=500, detail=f"Failed to submit Slurm job: {e}")

    lease.slurm_job_id = res.job_id
    lease.state = "SUBMITTED"

    with SessionLocal() as db:
        db.add(lease)
        db.commit()
        db.refresh(lease)

    return _lease_to_out(lease)

@router.delete("/leases/{lease_id}")
def cancel_lease(lease_id: int):
    with SessionLocal() as db:
        lease = db.get(Lease, lease_id)
        if not lease:
            raise HTTPException(status_code=404, detail="Lease not found")
        if lease.slurm_job_id:
            try:
                slurm.cancel(lease.slurm_job_id)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to cancel Slurm job: {e}")
        lease.state = "CANCELED"
        db.commit()
    return {"ok": True}

@router.post("/leases/{lease_id}/extend")
def extend_lease(lease_id: int, req: LeaseExtend):
    with SessionLocal() as db:
        lease = db.get(Lease, lease_id)
        if not lease:
            raise HTTPException(status_code=404, detail="Lease not found")
        if not lease.slurm_job_id:
            raise HTTPException(status_code=409, detail="Lease has no Slurm job")
        # extend to new duration from now (simple); you can make this smarter.
        new_time = _time_limit_from_duration(req.duration_seconds)
        try:
            slurm.extend_time(lease.slurm_job_id, new_time)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to extend job time: {e}")
        lease.end_at = (lease.begin_at or lease.created_at) + timedelta(seconds=req.duration_seconds)
        db.commit()
    return {"ok": True, "new_time_limit": new_time}

@router.post("/endpoints/register", response_model=EndpointOut)
def register_endpoint(req: EndpointRegister):
    # Called by the Slurm job script once vLLM is up (or once it decides to register).
    with SessionLocal() as db:
        # Upsert by slurm_job_id
        existing = db.execute(select(Endpoint).where(Endpoint.slurm_job_id == req.slurm_job_id)).scalars().first()
        if existing:
            existing.model = req.model
            existing.host = req.host
            existing.port = req.port
            existing.state = "STARTING"
            db.commit()
            db.refresh(existing)
            e = existing
        else:
            e = Endpoint(model=req.model, host=req.host, port=req.port, slurm_job_id=req.slurm_job_id, state="STARTING")
            db.add(e)
            db.commit()
            db.refresh(e)
    return EndpointOut(
        id=e.id, model=e.model, host=e.host, port=e.port, slurm_job_id=e.slurm_job_id, state=e.state,
        last_health_at=e.last_health_at, last_error=e.last_error, created_at=e.created_at
    )

# --- Default model (idle fallback) -------------------------------------------------
# This is a simple controller that ensures DEFAULT_MODEL is running whenever there are
# no other READY endpoints. It does NOT affect request routing/fallback chains.

async def ensure_default_model_running():
    if not settings.default_model:
        return
    default_name = settings.default_model

    with SessionLocal() as db:
        # If any READY endpoint exists for a non-default model, do nothing
        non_default_ready = db.execute(
            select(Endpoint).where(Endpoint.state == "READY", Endpoint.model != default_name)
        ).scalars().first()
        if non_default_ready:
            return

        # Is default already present and not stopped?
        existing = db.execute(
            select(Endpoint).where(Endpoint.model == default_name, Endpoint.state.in_(["STARTING", "READY"]))
        ).scalars().first()
        if existing:
            return

    # Submit default lease "now" with a long duration if configured
    # Use catalog defaults if overrides not provided.
    if default_name not in CATALOG:
        return

    cat = CATALOG[default_name]
    gpus = settings.default_model_gpus or cat.gpus
    tp = settings.default_model_tp or cat.tensor_parallel_size

    # 12h default runtime; can be restarted by this worker.
    req = LeaseCreate(model=default_name, owner="default", begin_at=None, duration_seconds=12*3600, gpus=gpus, tensor_parallel_size=tp)
    try:
        create_lease(req)
    except Exception:
        # ignore; next loop will retry
        return
