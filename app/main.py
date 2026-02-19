from __future__ import annotations
import asyncio
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from .settings import settings
from .dependencies import SessionLocal, init_db  # <-- shared engine
from .models import Endpoint, Lease
from .catalog import get_catalog
from .schemas import OpenAIModelsResponse
from .admin import router as admin_router
from .router_core import choose_ready_endpoint, health_check_endpoint
from .proxy import proxy_json_or_stream
from .admin import _submit_to_slurm  # internal helper
from . import slurm

app = FastAPI(title="vLLM Swapper Router", version="0.4.0")
app.include_router(admin_router)

# Initialize DB tables (uses shared engine from dependencies.py)
init_db()

app.mount("/ui", StaticFiles(directory="app/ui", html=True), name="ui")


@app.get("/")
def root_ui():
    return FileResponse("app/ui/index.html")


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/v1/models", response_model=OpenAIModelsResponse)
def v1_models():
    catalog = get_catalog()
    with SessionLocal() as db:
        ready = set(db.execute(select(Endpoint.model).where(Endpoint.state == "READY")).scalars().all())
    data = []
    for name, m in catalog.items():
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


# =============================================================================
# HEALTH WORKER — unified, adaptive polling
#
# - STARTING endpoints: polled every cycle (3s) for fast readiness detection
# - READY endpoints: polled every ~15s for liveness monitoring
# - Handles ALL state transitions:
#     STARTING → READY  (lease → RUNNING)
#     READY → FAILED    (lease → FAILED)
#     STARTING timeout → FAILED (lease → FAILED)
# - Fallback: any READY endpoint whose lease is still STARTING/SUBMITTED
#   gets its lease promoted to RUNNING (catches missed transitions)
# =============================================================================
async def health_worker():
    while True:
        try:
            with SessionLocal() as db:
                eps = db.execute(
                    select(Endpoint).where(
                        Endpoint.state.in_(["STARTING", "READY"])
                    )
                ).scalars().all()

                now = datetime.now(timezone.utc)

                for e in eps:
                    # --- Adaptive polling: skip READY endpoints checked recently ---
                    if e.state == "READY" and e.last_health_at:
                        since_last = (now - _ensure_aware_dt(e.last_health_at)).total_seconds()
                        if since_last < 12:  # READY endpoints: ~15s interval
                            continue

                    prev_state = e.state
                    ok, err = await health_check_endpoint(e.host, e.port)

                    if ok:
                        if e.state == "STARTING":
                            # === Transition: STARTING → READY ===
                            e.state = "READY"
                            e.last_error = None
                            print(
                                f"health_worker: endpoint {e.model} "
                                f"(job {e.slurm_job_id}) is now READY"
                            )

                            # Transition the lease too
                            lease = db.execute(
                                select(Lease).where(
                                    Lease.slurm_job_id == e.slurm_job_id
                                )
                            ).scalars().first()
                            if lease and lease.state in ("SUBMITTED", "STARTING"):
                                lease.state = "RUNNING"
                                print(
                                    f"health_worker: lease {lease.id} "
                                    f"({lease.model}) → RUNNING"
                                )

                        elif e.state == "READY":
                            # Still healthy — just clear any stale error
                            e.last_error = None

                    else:
                        # --- Health check failed ---
                        if e.state == "READY":
                            # === READY model went down ===
                            e.state = "FAILED"
                            e.last_error = err
                            print(
                                f"health_worker: endpoint {e.model} "
                                f"(job {e.slurm_job_id}) FAILED: {err}"
                            )

                            lease = db.execute(
                                select(Lease).where(
                                    Lease.slurm_job_id == e.slurm_job_id
                                )
                            ).scalars().first()
                            if lease and lease.state == "RUNNING":
                                lease.state = "FAILED"
                                lease.failed_at = now
                                print(
                                    f"health_worker: lease {lease.id} "
                                    f"({lease.model}) → FAILED"
                                )

                        elif e.state == "STARTING":
                            # Still starting — check for timeout
                            age = (
                                now - _ensure_aware_dt(e.created_at)
                            ).total_seconds()
                            if age > settings.vllm_health_timeout_seconds:
                                # === STARTING timeout → FAILED ===
                                e.state = "FAILED"
                                e.last_error = (
                                    f"Timed out after {age:.0f}s: {err}"
                                )
                                print(
                                    f"health_worker: endpoint {e.model} "
                                    f"(job {e.slurm_job_id}) timed out "
                                    f"after {age:.0f}s"
                                )

                                lease = db.execute(
                                    select(Lease).where(
                                        Lease.slurm_job_id == e.slurm_job_id
                                    )
                                ).scalars().first()
                                if lease and lease.state in (
                                    "SUBMITTED", "STARTING"
                                ):
                                    lease.state = "FAILED"
                                    lease.failed_at = now
                            else:
                                # Not timed out yet — just record the error
                                e.last_error = err

                    e.last_health_at = now

                # --- Fallback reconciliation ---
                # If any READY endpoint has a lease still stuck in
                # SUBMITTED/STARTING, promote it now. This catches any
                # edge case where the transition was missed above.
                ready_eps = db.execute(
                    select(Endpoint).where(Endpoint.state == "READY")
                ).scalars().all()
                for e in ready_eps:
                    lease = db.execute(
                        select(Lease).where(
                            Lease.slurm_job_id == e.slurm_job_id
                        )
                    ).scalars().first()
                    if lease and lease.state in ("SUBMITTED", "STARTING"):
                        lease.state = "RUNNING"
                        print(
                            f"health_worker (fallback): lease {lease.id} "
                            f"({lease.model}) → RUNNING"
                        )

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
                            l.failed_at = now
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
                    select(Endpoint).where(
                        Endpoint.state.in_(["READY", "STARTING", "FAILED"])
                    )
                ).scalars().all()
                for e in eps:
                    lease = db.execute(
                        select(Lease).where(
                            Lease.slurm_job_id == e.slurm_job_id
                        )
                    ).scalars().first()
                    if lease and lease.end_at and _ensure_aware_dt(lease.end_at) < now:
                        e.state = "STOPPED"
                        if lease.state == "RUNNING":
                            lease.state = "ENDED"
                        # Leave FAILED leases as FAILED but stop the endpoint
                    elif lease and lease.state in ("CANCELED", "ENDED"):
                        e.state = "STOPPED"
                db.commit()
        except Exception as e:
            print(f"endpoint_cleanup_worker error: {e}")
        await asyncio.sleep(15)


# =============================================================================
# SLURM RECONCILE WORKER
#
# Reconcile leases/endpoints with Slurm reality (squeue).
# - If Slurm job is gone from squeue → mark lease ENDED or FAILED
# - FAILED leases past their end_at → ENDED
# - Sets failed_at timestamp so retry_worker can pick it up
# =============================================================================
async def slurm_reconcile_worker():
    while True:
        try:
            now = datetime.now(timezone.utc)
            with SessionLocal() as db:
                active = db.execute(
                    select(Lease).where(
                        Lease.state.in_(
                            ["SUBMITTED", "STARTING", "RUNNING", "FAILED"]
                        )
                    )
                ).scalars().all()

                for l in active:
                    # For FAILED leases: check if end_at has passed → ENDED
                    if l.state == "FAILED":
                        if l.end_at and _ensure_aware_dt(l.end_at) < now:
                            l.state = "ENDED"
                            if l.slurm_job_id:
                                ep = db.execute(
                                    select(Endpoint).where(
                                        Endpoint.slurm_job_id == l.slurm_job_id
                                    )
                                ).scalars().first()
                                if ep:
                                    ep.state = "STOPPED"
                        continue

                    if not l.slurm_job_id:
                        continue

                    state = slurm.squeue_job_state(l.slurm_job_id)

                    # Not in squeue anymore → finished/failed/canceled
                    if state is None:
                        ep = db.execute(
                            select(Endpoint).where(
                                Endpoint.slurm_job_id == l.slurm_job_id
                            )
                        ).scalars().first()

                        if ep is None or ep.state in ("FAILED", "STARTING"):
                            # Never became READY → FAILED
                            l.state = "FAILED"
                            l.failed_at = now  # <-- enables retry_worker
                            if ep:
                                ep.state = "FAILED"
                            print(
                                f"slurm_reconcile: lease {l.id} "
                                f"({l.model}) → FAILED "
                                f"(job {l.slurm_job_id} gone from squeue)"
                            )
                        else:
                            # Was READY at some point → ENDED
                            if l.state not in ("CANCELED",):
                                l.state = "ENDED"
                            if ep:
                                ep.state = "STOPPED"
                            print(
                                f"slurm_reconcile: lease {l.id} "
                                f"({l.model}) → ENDED "
                                f"(job {l.slurm_job_id} gone from squeue)"
                            )

                db.commit()
        except Exception as e:
            print(f"slurm_reconcile_worker error: {e}")
        await asyncio.sleep(5)


# =============================================================================
# RETRY WORKER
#
# Scans for FAILED leases eligible for retry:
# - retry_count < settings.vllm_max_retries
# - Lease hasn't expired (end_at still in the future)
# - Enough time since failure (retry_delay_seconds cooldown)
# - Enough remaining time to be worth retrying (>120s)
#
# Cleans up old endpoint, resubmits to Slurm, resets lease to SUBMITTED.
# =============================================================================
async def retry_worker():
    while True:
        try:
            now = datetime.now(timezone.utc)
            with SessionLocal() as db:
                failed = db.execute(
                    select(Lease).where(Lease.state == "FAILED")
                ).scalars().all()

                for lease in failed:
                    # Skip if expired
                    if lease.end_at and _ensure_aware_dt(lease.end_at) < now:
                        continue

                    # Skip if max retries exhausted
                    if lease.retry_count >= settings.vllm_max_retries:
                        continue

                    # Skip if no failed_at (shouldn't happen, but be safe)
                    if not lease.failed_at:
                        continue

                    # Skip if not enough cooldown time since failure
                    since_fail = (
                        now - _ensure_aware_dt(lease.failed_at)
                    ).total_seconds()
                    if since_fail < settings.vllm_retry_delay_seconds:
                        continue

                    # Check remaining lease time
                    if lease.end_at:
                        remaining = (
                            _ensure_aware_dt(lease.end_at) - now
                        ).total_seconds()
                        if remaining < 120:
                            print(
                                f"retry_worker: lease {lease.id} "
                                f"({lease.model}) has only {remaining:.0f}s "
                                f"left, skipping retry"
                            )
                            continue

                    # --- Attempt retry ---
                    lease.retry_count += 1
                    print(
                        f"retry_worker: retrying lease {lease.id} "
                        f"({lease.model}), attempt "
                        f"{lease.retry_count}/{settings.vllm_max_retries}"
                    )

                    # Clean up old endpoint
                    if lease.slurm_job_id:
                        old_ep = db.execute(
                            select(Endpoint).where(
                                Endpoint.slurm_job_id == lease.slurm_job_id
                            )
                        ).scalars().first()
                        if old_ep:
                            old_ep.state = "STOPPED"

                    try:
                        new_job_id = _submit_to_slurm(lease)
                        lease.slurm_job_id = new_job_id
                        lease.state = "SUBMITTED"
                        lease.failed_at = None
                        print(
                            f"retry_worker: lease {lease.id} resubmitted "
                            f"as Slurm job {new_job_id}"
                        )
                    except Exception as ex:
                        print(
                            f"retry_worker: failed to resubmit "
                            f"lease {lease.id}: {ex}"
                        )
                        # Reset failed_at to restart cooldown
                        lease.failed_at = now

                db.commit()
        except Exception as e:
            print(f"retry_worker error: {e}")

        await asyncio.sleep(10)


@app.on_event("startup")
async def startup():
    asyncio.create_task(health_worker())
    asyncio.create_task(planned_submit_worker())
    asyncio.create_task(endpoint_cleanup_worker())
    asyncio.create_task(slurm_reconcile_worker())
    asyncio.create_task(retry_worker())  # <-- NEW
