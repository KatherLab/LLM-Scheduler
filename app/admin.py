from __future__ import annotations

import asyncio
import glob
import json
import os
from datetime import datetime, timedelta, timezone
from .auth import require_auth, require_internal_token
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select
from typing import Optional

from .settings import settings
from .models import Lease, Endpoint
from .schemas import (
    LeaseCreate, LeaseOut, LeaseExtend, LeaseUpdate, LeaseShortenRequest,
    EndpointRegister, EndpointOut, DashboardResponse, DashboardModel,
    EndpointStats, LogResponse,
)
from .catalog import get_catalog
from .planner import compute_placements, find_earliest_slot
from . import slurm
from .dependencies import SessionLocal, init_db
from .utils import ensure_utc

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_auth)])

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

# ---------- helpers -------------------------------------------------------------

def _time_limit_from_duration(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def _lease_begin(l: Lease) -> datetime:
    return ensure_utc(l.begin_at) if l.begin_at is not None else ensure_utc(l.created_at)

def _lease_end(l: Lease) -> datetime:
    if l.end_at is not None:
        return ensure_utc(l.end_at)
    return _lease_begin(l) + timedelta(hours=1)

def _lease_to_out(l: Lease, lane_start: Optional[int] = None, lane_count: Optional[int] = None, conflict: bool = False) -> LeaseOut:
    return LeaseOut(
        id=l.id,
        model=l.model,
        owner=l.owner,
        notes=l.notes,
        state=l.state,
        slurm_job_id=l.slurm_job_id,
        host=settings.public_hostname,
        port=l.requested_port or 0,
        requested_gpus=l.requested_gpus,
        requested_tp=l.requested_tp,
        begin_at=l.begin_at,
        end_at=l.end_at,
        created_at=l.created_at,
        lane_start=lane_start,
        lane_count=lane_count,
        conflict=conflict,
    )

def _build_job_env(lease: Lease) -> dict[str, str]:
    env = {
        "MODEL_PATH": lease.model_path,
        "SERVED_MODEL_NAME": lease.model,
        "TP_SIZE": str(lease.requested_tp),
        "API_KEY": settings.vllm_api_key,
        "GPU_MEM_UTIL": lease.gpu_memory_utilization or "0.95",
        "EXTRA_ARGS": lease.extra_args or "",
        "TOOL_ARGS": lease.tool_args or "",
        "REASONING_PARSER": lease.reasoning_parser or "",
        "ROUTER_REGISTER_URL": f"http://{settings.public_hostname}:{settings.router_port}/admin/endpoints/register",
    }

    env.update({
        "VLLM_HEALTH_TIMEOUT_SECONDS": str(settings.vllm_health_timeout_seconds),
        "VLLM_MAX_RETRIES": str(settings.vllm_max_retries),
        "VLLM_RETRY_DELAY_SECONDS": str(settings.vllm_retry_delay_seconds),
    })

    if lease.venv_activate:
        env["VENV_ACTIVATE"] = lease.venv_activate

    if lease.env_json:
        import json
        try:
            model_env = json.loads(lease.env_json)
            if isinstance(model_env, dict):
                env.update(model_env)
        except (json.JSONDecodeError, TypeError):
            pass

    return env

def _submit_to_slurm(lease: Lease) -> str:
    seconds = int((_lease_end(lease) - (_lease_begin(lease))).total_seconds())
    seconds = max(60, seconds)
    time_limit = _time_limit_from_duration(seconds)

    env = _build_job_env(lease)

    # Per-model CPU/mem from lease, fall back to global settings
    cpus = lease.requested_cpus if lease.requested_cpus else settings.slurm_cpus_per_task
    mem = lease.requested_mem  # None is fine — slurm.py won't add --mem

    res = slurm.submit_vllm_job(
        template_path=settings.sbatch_template_path,
        job_name=f"vllm-{lease.model}",
        gpus=lease.requested_gpus,
        time_limit=time_limit,
        begin=None,
        env=env,
        partition=settings.slurm_partition,
        account=settings.slurm_account,
        qos=settings.slurm_qos,
        nodelist=settings.slurm_nodelist,
        cpus_per_task=cpus,
        mem=mem,
        log_dir=settings.vllm_log_dir,
        mail_user=settings.slurm_mail_user,
        mail_type=settings.slurm_mail_type,
    )

    return res.job_id

def _submit_to_slurm_from_snapshot(snapshot: dict) -> str:
    """
    Submit a Slurm job using a plain-dict snapshot instead of an ORM object.
    """
    begin = ensure_utc(snapshot["begin_at"]) if snapshot["begin_at"] else ensure_utc(snapshot["created_at"])
    end = ensure_utc(snapshot["end_at"])
    seconds = int((end - begin).total_seconds())
    seconds = max(60, seconds)
    time_limit = _time_limit_from_duration(seconds)

    env = {
        "MODEL_PATH": snapshot["model_path"],
        "SERVED_MODEL_NAME": snapshot["model"],
        "TP_SIZE": str(snapshot["requested_tp"]),
        "API_KEY": settings.vllm_api_key,
        "GPU_MEM_UTIL": snapshot["gpu_memory_utilization"] or "0.95",
        "EXTRA_ARGS": snapshot["extra_args"] or "",
        "TOOL_ARGS": snapshot["tool_args"] or "",
        "REASONING_PARSER": snapshot["reasoning_parser"] or "",
        "ROUTER_REGISTER_URL": f"http://{settings.public_hostname}:{settings.router_port}/admin/endpoints/register",
        "VLLM_HEALTH_TIMEOUT_SECONDS": str(settings.vllm_health_timeout_seconds),
        "VLLM_MAX_RETRIES": str(settings.vllm_max_retries),
        "VLLM_RETRY_DELAY_SECONDS": str(settings.vllm_retry_delay_seconds),
    }
    if snapshot.get("venv_activate"):
        env["VENV_ACTIVATE"] = snapshot["venv_activate"]

    # Per-model CPU/mem from snapshot, fall back to global settings
    cpus = snapshot.get("requested_cpus") or settings.slurm_cpus_per_task
    mem = snapshot.get("requested_mem")  # None is fine

    if snapshot.get("env_json"):
        import json
        try:
            model_env = json.loads(snapshot["env_json"])
            if isinstance(model_env, dict):
                env.update(model_env)
        except (json.JSONDecodeError, TypeError):
            pass

    res = slurm.submit_vllm_job(
        template_path=settings.sbatch_template_path,
        job_name=f"vllm-{snapshot['model']}",
        gpus=snapshot["requested_gpus"],
        time_limit=time_limit,
        begin=None,
        env=env,
        partition=settings.slurm_partition,
        account=settings.slurm_account,
        qos=settings.slurm_qos,
        nodelist=settings.slurm_nodelist,
        cpus_per_task=cpus,
        mem=mem,
        log_dir=settings.vllm_log_dir,
        mail_user=settings.slurm_mail_user,
        mail_type=settings.slurm_mail_type,
    )
    return res.job_id

def _snapshot_lease(lease: "Lease") -> dict:
    """Create a plain-dict snapshot of a Lease for use outside a DB session."""
    return {
        "id": lease.id,
        "model": lease.model,
        "model_path": lease.model_path,
        "requested_tp": lease.requested_tp,
        "requested_gpus": lease.requested_gpus,
        "requested_cpus": lease.requested_cpus,
        "requested_mem": lease.requested_mem,
        "gpu_memory_utilization": lease.gpu_memory_utilization,
        "extra_args": lease.extra_args,
        "tool_args": lease.tool_args,
        "reasoning_parser": lease.reasoning_parser,
        "venv_activate": lease.venv_activate,
        "env_json": lease.env_json,
        "begin_at": lease.begin_at,
        "end_at": lease.end_at,
        "created_at": lease.created_at,
        "slurm_job_id": lease.slurm_job_id,
    }



def _validate_no_conflicts(db: Session, candidate: Lease) -> None:
    now = now_utc()
    horizon_start = now - timedelta(hours=1)
    horizon_end = now + timedelta(hours=48)

    leases = db.execute(
        select(Lease).where(Lease.state.in_(["PLANNED", "SUBMITTED", "STARTING", "RUNNING"]))
    ).scalars().all()

    leases = [l for l in leases if l.id != candidate.id] + [candidate]

    placements = compute_placements(
        leases=leases,
        total_gpus=settings.total_gpus,
        horizon_start=horizon_start,
        horizon_end=horizon_end,
    )
    p = placements.get(candidate.id)
    if p and p.conflict:
        raise HTTPException(
            status_code=409,
            detail=f"Not enough GPUs available for that time window (needs {candidate.requested_gpus}).",
        )

def _merge_same_model_if_applicable(db: Session, req: LeaseCreate, begin: datetime, end: datetime) -> Optional[Lease]:
    existing = db.execute(
        select(Lease).where(
            Lease.model == req.model,
            Lease.state.in_(["PLANNED", "SUBMITTED", "STARTING", "RUNNING"])
        ).order_by(Lease.id.desc())
    ).scalars().first()
    if not existing:
        return None
    ex_begin = _lease_begin(existing)
    ex_end = _lease_end(existing)
    begin = ensure_utc(begin)
    end = ensure_utc(end)
    touch = timedelta(minutes=5)
    overlaps_or_touches = not (end < ex_begin - touch or begin > ex_end + touch)
    if overlaps_or_touches:
        existing.end_at = max(ex_end, end)
        # Don't move start backward for active leases — only extend end
        if existing.state in ("RUNNING", "SUBMITTED", "STARTING"):
            pass  # keep existing.begin_at as-is
        elif existing.begin_at is not None:
            existing.begin_at = min(ex_begin, begin)
        return existing
    return None


def _read_log_file(path: str, max_bytes: int = 200_000) -> tuple[str, bool]:
    """Read the tail of a log file. Returns (content, truncated)."""
    if not os.path.isfile(path):
        return "", False
    size = os.path.getsize(path)
    truncated = size > max_bytes
    with open(path, "r", errors="replace") as f:
        if truncated:
            f.seek(size - max_bytes)
            _ = f.readline()  # skip partial line
        return f.read(), truncated


def _find_log_files(slurm_job_id: str) -> tuple[str, str]:
    """Find stdout/stderr log files for a Slurm job."""
    # Validate that slurm_job_id is purely numeric to prevent path traversal
    if not slurm_job_id.isdigit():
        return "", ""

    log_dir = os.path.abspath(settings.vllm_log_dir)

    stdout_pattern = os.path.join(log_dir, f"*-{slurm_job_id}.out")
    stderr_pattern = os.path.join(log_dir, f"*-{slurm_job_id}.err")

    stdout_files = glob.glob(stdout_pattern)
    stderr_files = glob.glob(stderr_pattern)

    # Extra safety: ensure resolved paths are within log_dir
    stdout_path = ""
    stderr_path = ""
    if stdout_files and os.path.abspath(stdout_files[0]).startswith(log_dir):
        stdout_path = stdout_files[0]
    if stderr_files and os.path.abspath(stderr_files[0]).startswith(log_dir):
        stderr_path = stderr_files[0]

    return stdout_path, stderr_path


# ---------- endpoints ------------------------------------------------------------

@router.get("/dashboard", response_model=DashboardResponse)
def dashboard():
    now = now_utc()
    horizon_start = now - timedelta(hours=1)
    horizon_end = now + timedelta(hours=48)

    with SessionLocal() as db:
        ready = set(
            db.execute(select(Endpoint.model).where(Endpoint.state == "READY")).scalars().all()
        )

        leases = db.execute(select(Lease).order_by(Lease.id.desc())).scalars().all()

        # Include FAILED leases that haven't passed their end_at yet
        active_like = [
            l for l in leases
            if l.state in ("PLANNED", "SUBMITTED", "STARTING", "RUNNING")
            or (l.state == "FAILED" and l.end_at and ensure_utc(l.end_at) > now)
        ]
        placements = compute_placements(
            leases=active_like,
            total_gpus=settings.total_gpus,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
        )

        out_leases: list[LeaseOut] = []
        for l in leases:
            p = placements.get(l.id)
            out_leases.append(_lease_to_out(
                l,
                lane_start=p.lane_start if p else None,
                lane_count=p.lane_count if p else None,
                conflict=p.conflict if p else False,
            ))

        catalog = get_catalog()
        models: list[DashboardModel] = []
        for name, m in catalog.items():
            models.append(DashboardModel(
                id=name,
                ready=(name in ready),
                meta={
                    "gpus": m.gpus,
                    "tensor_parallel_size": m.tensor_parallel_size,
                    "notes": m.notes,
                }
            ))

        # Endpoint stats
        eps = db.execute(
            select(Endpoint).where(Endpoint.state.in_(["READY", "STARTING"]))
        ).scalars().all()
        stats: list[EndpointStats] = []
        for e in eps:
            uptime = None
            if e.created_at:
                uptime = (now - ensure_utc(e.created_at)).total_seconds()
            stats.append(EndpointStats(
                model=e.model,
                host=e.host,
                port=e.port,
                state=e.state,
                slurm_job_id=e.slurm_job_id,
                last_health_at=e.last_health_at,
                uptime_seconds=uptime,
            ))

        return DashboardResponse(
            now=now,
            total_gpus=settings.total_gpus,
            models=models,
            leases=out_leases,
            endpoint_stats=stats,
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
async def create_lease(req: LeaseCreate):
    catalog = get_catalog()
    if req.model not in catalog:
        raise HTTPException(status_code=404, detail=f"Unknown model '{req.model}'")

    cat = catalog[req.model]
    gpus = cat.gpus
    tp = cat.tensor_parallel_size

    duration = timedelta(seconds=req.duration_seconds)

    # --- ASAP logic ---
    if req.asap:
        with SessionLocal() as db:
            active = db.execute(
                select(Lease).where(Lease.state.in_(["PLANNED", "SUBMITTED", "STARTING", "RUNNING"]))
            ).scalars().all()

            now = now_utc()
            horizon_end = now + timedelta(hours=48)

            earliest = find_earliest_slot(
                existing_leases=active,
                gpus_needed=gpus,
                duration=duration,
                total_gpus=settings.total_gpus,
                search_start=now,
                search_end=horizon_end,
            )
            if earliest is None:
                raise HTTPException(status_code=409, detail="No available slot in the next 48h for this model.")
            begin = earliest
            end = begin + duration

    else:
        begin = ensure_utc(req.begin_at) if req.begin_at else now_utc()
        end = begin + duration


    snapshot = None
    lease_id = None
    out = None

    with SessionLocal() as db:
        merged = _merge_same_model_if_applicable(db, req, begin, end)
        if merged:
            _validate_no_conflicts(db, merged)

            if merged.slurm_job_id and merged.state in ("SUBMITTED", "STARTING", "RUNNING"):
                total_seconds = int((_lease_end(merged) - _lease_begin(merged)).total_seconds())
                total_seconds = max(60, total_seconds)
                new_time_limit = _time_limit_from_duration(total_seconds)
                try:
                    await asyncio.to_thread(slurm.extend_time, merged.slurm_job_id, new_time_limit)
                    print(
                        f"create_lease: extended Slurm job {merged.slurm_job_id} "
                        f"time to {new_time_limit} after merge"
                    )
                except Exception as e:
                    print(f"Warning: failed to extend Slurm time for merged lease {merged.id}: {e}")

            db.add(merged)
            db.commit()
            db.refresh(merged)
            return _lease_to_out(merged)

        begin = ensure_utc(begin)
        end = ensure_utc(end)
        planned = (begin > now_utc() + timedelta(seconds=30))

        lease = Lease(
            model=req.model,
            requested_gpus=gpus,
            requested_tp=tp,
            requested_cpus=cat.cpus,
            requested_mem=cat.mem,
            requested_port=0,
            owner=req.owner,
            notes=req.notes,
            begin_at=begin if planned else None,
            end_at=end,
            created_at=now_utc(),
            model_path=cat.model_path,
            tool_args=req.tool_args if req.tool_args is not None else cat.tool_args,
            extra_args=req.extra_args if req.extra_args is not None else cat.extra_args,
            reasoning_parser=req.reasoning_parser if req.reasoning_parser is not None else cat.reasoning_parser,
            gpu_memory_utilization=str(
                req.gpu_memory_utilization if req.gpu_memory_utilization is not None else cat.gpu_memory_utilization
            ),
            venv_activate=cat.venv_activate,
            env_json=json.dumps(cat.env) if cat.env else None,
            state="PLANNED" if planned else "SUBMITTED",
        )

        _validate_no_conflicts(db, lease)

        db.add(lease)
        db.commit()
        db.refresh(lease)

        out = _lease_to_out(lease)
        lease_id = lease.id

        if lease.state == "SUBMITTED":
            snapshot = _snapshot_lease(lease)

    # ── Outside DB session: submit to Slurm if needed ──
    if snapshot is not None:
        try:
            job_id = await asyncio.to_thread(_submit_to_slurm_from_snapshot, snapshot)
        except Exception as e:
            with SessionLocal() as db:
                lease = db.get(Lease, lease_id)
                if lease:
                    lease.state = "FAILED"
                    lease.failed_at = now_utc()
                    db.commit()
            raise HTTPException(status_code=500, detail=f"Failed to submit Slurm job: {e}")

        with SessionLocal() as db:
            lease = db.get(Lease, lease_id)
            if lease:
                lease.slurm_job_id = job_id
                db.commit()
                db.refresh(lease)
                out = _lease_to_out(lease)

    return out

@router.patch("/leases/{lease_id}", response_model=LeaseOut)
def update_lease(lease_id: int, req: LeaseUpdate):
    with SessionLocal() as db:
        lease = db.get(Lease, lease_id)
        if not lease:
            raise HTTPException(status_code=404, detail="Booking not found")

        if lease.state != "PLANNED":
            raise HTTPException(status_code=409, detail="Only planned bookings can be edited (move/resize).")

        if req.begin_at is not None:
            lease.begin_at = req.begin_at
        if req.end_at is not None:
            lease.end_at = req.end_at
        if req.notes is not None:
            lease.notes = req.notes

        b = _lease_begin(lease)
        e = _lease_end(lease)
        if e <= b:
            raise HTTPException(status_code=400, detail="End time must be after start time.")
        if lease.requested_gpus > settings.total_gpus:
            raise HTTPException(status_code=400, detail=f"Cannot request more than {settings.total_gpus} GPUs.")

        _validate_no_conflicts(db, lease)
        db.commit()
        db.refresh(lease)
        return _lease_to_out(lease)


@router.patch("/leases/{lease_id}/notes")
def update_lease_notes(lease_id: int, req: dict):
    """Update notes on any active lease (not just PLANNED)."""
    with SessionLocal() as db:
        lease = db.get(Lease, lease_id)
        if not lease:
            raise HTTPException(status_code=404, detail="Booking not found")
        if lease.state in ("CANCELED", "ENDED"):
            raise HTTPException(status_code=409, detail="Cannot edit notes on a finished booking.")
        lease.notes = req.get("notes", "")
        db.commit()
        return {"ok": True, "notes": lease.notes}

@router.delete("/leases/{lease_id}")
async def cancel_lease(lease_id: int):
    with SessionLocal() as db:
        lease = db.get(Lease, lease_id)
        if not lease:
            raise HTTPException(status_code=404, detail="Booking not found")

        if lease.slurm_job_id:
            try:
                await slurm.async_cancel(lease.slurm_job_id)
            except Exception as e:
                print(f"Warning: Slurm cancel failed: {e}")

        lease.state = "CANCELED"

        if lease.slurm_job_id:
            ep = db.execute(select(Endpoint).where(Endpoint.slurm_job_id == lease.slurm_job_id)).scalars().first()
            if ep:
                ep.state = "STOPPED"

        db.commit()
    return {"ok": True}

@router.post("/leases/{lease_id}/extend")
async def extend_lease(lease_id: int, req: LeaseExtend):
    with SessionLocal() as db:
        lease = db.get(Lease, lease_id)
        if not lease:
            raise HTTPException(status_code=404, detail="Booking not found")

        b = _lease_begin(lease)
        e = _lease_end(lease)

        new_end = e + timedelta(seconds=req.duration_seconds)
        lease.end_at = new_end

        _validate_no_conflicts(db, lease)

        if lease.slurm_job_id and lease.state in ("SUBMITTED", "STARTING", "RUNNING"):
            total_seconds = int((new_end - b).total_seconds())
            total_seconds = max(60, total_seconds)
            new_time_limit = _time_limit_from_duration(total_seconds)
            try:
                await slurm.async_extend_time(lease.slurm_job_id, new_time_limit)
            except Exception as e:
                print(f"Warning: failed to extend Slurm time: {e}")

        db.commit()
        return {"ok": True, "new_end_at": lease.end_at}


@router.post("/leases/{lease_id}/shorten")
async def shorten_lease(lease_id: int, req: LeaseShortenRequest):
    """Shorten a running or submitted lease. The new end must be in the future and before the current end."""
    with SessionLocal() as db:
        lease = db.get(Lease, lease_id)
        if not lease:
            raise HTTPException(status_code=404, detail="Booking not found")

        if lease.state not in ("RUNNING", "SUBMITTED", "STARTING", "PLANNED"):
            raise HTTPException(status_code=409, detail="Can only shorten active bookings.")

        now = now_utc()
        new_end = ensure_utc(req.new_end_at)
        current_end = _lease_end(lease)
        b = _lease_begin(lease)

        if new_end <= now:
            raise HTTPException(status_code=400, detail="New end time must be in the future.")
        if new_end <= b:
            raise HTTPException(status_code=400, detail="New end time must be after start time.")
        if new_end >= current_end:
            raise HTTPException(status_code=400, detail="New end time must be before current end time. Use extend instead.")

        lease.end_at = new_end

        # Update Slurm time limit
        if lease.slurm_job_id and lease.state in ("SUBMITTED", "STARTING", "RUNNING"):
            total_seconds = int((new_end - b).total_seconds())
            total_seconds = max(60, total_seconds)
            new_time_limit = _time_limit_from_duration(total_seconds)
            try:
                await slurm.async_extend_time(lease.slurm_job_id, new_time_limit)
            except Exception as e:
                print(f"Warning: failed to shorten Slurm time: {e}")

        db.commit()
        return {"ok": True, "new_end_at": lease.end_at}


@router.post("/leases/{lease_id}/stop")
async def stop_lease_now(lease_id: int):
    """Immediately stop a running model — cancels the Slurm job."""
    with SessionLocal() as db:
        lease = db.get(Lease, lease_id)
        if not lease:
            raise HTTPException(status_code=404, detail="Booking not found")

        if lease.state not in ("RUNNING", "SUBMITTED", "STARTING", "PLANNED"):
            raise HTTPException(status_code=409, detail="Booking is not active.")

        if lease.slurm_job_id:
            try:
                await slurm.async_cancel(lease.slurm_job_id)
            except Exception as e:
                print(f"Warning: Slurm cancel failed: {e}")

        lease.state = "CANCELED"
        lease.end_at = now_utc()

        if lease.slurm_job_id:
            ep = db.execute(select(Endpoint).where(Endpoint.slurm_job_id == lease.slurm_job_id)).scalars().first()
            if ep:
                ep.state = "STOPPED"

        db.commit()
    return {"ok": True}


@router.get("/leases/{lease_id}/logs", response_model=LogResponse)
def get_lease_logs(lease_id: int):
    """Retrieve Slurm stdout/stderr logs for a lease."""
    with SessionLocal() as db:
        lease = db.get(Lease, lease_id)
        if not lease:
            raise HTTPException(status_code=404, detail="Booking not found")
        if not lease.slurm_job_id:
            raise HTTPException(status_code=404, detail="No Slurm job associated with this booking.")

        stdout_path, stderr_path = _find_log_files(lease.slurm_job_id)

        stdout_content, stdout_trunc = _read_log_file(stdout_path) if stdout_path else ("", False)
        stderr_content, stderr_trunc = _read_log_file(stderr_path) if stderr_path else ("", False)

        return LogResponse(
            slurm_job_id=lease.slurm_job_id,
            log_stdout=stdout_content,
            log_stderr=stderr_content,
            truncated=stdout_trunc or stderr_trunc,
        )

# ── Internal router (no session auth, uses internal token) ──────────────────
internal_router = APIRouter(prefix="/admin", tags=["internal"])


@internal_router.post("/endpoints/register", response_model=EndpointOut)
def register_endpoint(req: EndpointRegister, _: None = Depends(require_internal_token)):
    with SessionLocal() as db:
        existing = db.execute(select(Endpoint).where(Endpoint.slurm_job_id == req.slurm_job_id)).scalars().first()
        if existing:
            existing.model = req.model
            existing.host = req.host
            existing.port = req.port
            if existing.state not in ("READY",):
                existing.state = "STARTING"
            db.commit()
            db.refresh(existing)
            e = existing
        else:
            e = Endpoint(model=req.model, host=req.host, port=req.port, slurm_job_id=req.slurm_job_id, state="STARTING")
            db.add(e)
            db.commit()
            db.refresh(e)

        lease = db.execute(select(Lease).where(Lease.slurm_job_id == req.slurm_job_id)).scalars().first()
        if lease:
            lease.requested_port = req.port
            if lease.state in ("SUBMITTED", "PLANNED"):
                lease.state = "STARTING"
            db.commit()
            db.refresh(e)

        return EndpointOut(
            id=e.id, model=e.model, host=e.host, port=e.port,
            slurm_job_id=e.slurm_job_id, state=e.state,
            last_health_at=e.last_health_at, last_error=e.last_error,
            created_at=e.created_at
        )

