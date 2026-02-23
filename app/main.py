from __future__ import annotations
import asyncio
from datetime import datetime, timedelta, timezone
import json
from fastapi import FastAPI, Request, HTTPException
from .auth import auth_router, require_auth, get_session
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

from .settings import settings
from .dependencies import SessionLocal, init_db  # <-- shared engine
from .models import Endpoint, Lease
from .catalog import get_catalog
from .schemas import OpenAIModelsResponse
from .admin import router as admin_router
from .router_core import choose_ready_endpoint, health_check_endpoint
from .proxy import proxy_json_or_stream
from .admin import internal_router
from . import slurm
from .utils import ensure_utc

# ── Supervised task wrapper ─────────────────────────────────────────────────
async def _supervised(name: str, coro_fn, restart_delay: float = 2.0):
    """
    Run an async worker forever. If it crashes, log and restart after a delay.
    Only exits on asyncio.CancelledError (i.e. shutdown).
    """
    while True:
        try:
            await coro_fn()
        except asyncio.CancelledError:
            print(f"{name}: cancelled, shutting down")
            return
        except Exception as e:
            print(f"{name}: crashed ({e}), restarting in {restart_delay}s")
            await asyncio.sleep(restart_delay)


# ── Lifespan (replaces deprecated @app.on_event) ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: reconcile DB with Slurm, then launch supervised background workers.
    Shutdown: cancel all workers, close the shared httpx proxy client.
    """
    # ── Startup ──
    reconcile_on_startup()

    tasks = [
        asyncio.create_task(_supervised("health_worker", health_worker)),
        asyncio.create_task(_supervised("planned_submit_worker", planned_submit_worker)),
        asyncio.create_task(_supervised("endpoint_cleanup_worker", endpoint_cleanup_worker)),
        asyncio.create_task(_supervised("slurm_reconcile_worker", slurm_reconcile_worker)),
        asyncio.create_task(_supervised("retry_worker", retry_worker)),
    ]

    yield  # ← app is running and serving requests

    # ── Shutdown ──
    print("lifespan: shutting down background workers...")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Close shared HTTP clients
    from .proxy import close_client
    from .router_core import close_health_client
    await close_client()
    await close_health_client()

    print("lifespan: shutdown complete")


app = FastAPI(title="KatherLab LLM Scheduler", version="0.4.0", lifespan=lifespan)
app.include_router(admin_router)
app.include_router(auth_router)
app.include_router(internal_router)

# Initialize DB tables (uses shared engine from dependencies.py)
init_db()

app.mount("/ui", StaticFiles(directory="app/ui", html=True), name="ui")


@app.get("/")
def root_ui(request: Request):
    session = get_session(request)
    if session is None:
        return RedirectResponse(url="/login", status_code=302)
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
            f"Please visit the Scheduler UI to start it: "
            f"http://{settings.public_hostname}:{settings.router_port}/"
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
    body = await request.body()
    try:
        j = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    model = j.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="Missing 'model' in request body")
    is_stream = bool(j.get("stream", False))

    with SessionLocal() as db:
        upstream = _resolve_upstream(db, model)
    return await proxy_json_or_stream(
        request,
        upstream_url=f"{upstream}/v1/chat/completions",
        body=body,
        is_stream=is_stream,
    )

@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.body()
    try:
        j = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    model = j.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="Missing 'model' in request body")
    is_stream = bool(j.get("stream", False))

    with SessionLocal() as db:
        upstream = _resolve_upstream(db, model)
    return await proxy_json_or_stream(
        request,
        upstream_url=f"{upstream}/v1/messages",
        body=body,
        is_stream=is_stream,
    )

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
    """
    Unified health poller.

    Phase 1: snapshot endpoint data from DB (fast, no HTTP).
    Phase 2: run all health checks concurrently OUTSIDE the DB session.
    Phase 3: re-open DB session and apply state transitions (fast, no HTTP).
    """
    while True:
        try:
            now = datetime.now(timezone.utc)

            # ── Phase 1: snapshot endpoints we need to check ────────────
            with SessionLocal() as db:
                eps = db.execute(
                    select(Endpoint).where(
                        Endpoint.state.in_(["STARTING", "READY"])
                    )
                ).scalars().all()

                checks: list[dict] = []
                for e in eps:
                    # Adaptive polling: skip READY endpoints checked recently
                    if e.state == "READY" and e.last_health_at:
                        since_last = (
                            now - ensure_utc(e.last_health_at)
                        ).total_seconds()
                        if since_last < 12:
                            continue

                    checks.append({
                        "id": e.id,
                        "host": e.host,
                        "port": e.port,
                        "state": e.state,
                        "created_at": e.created_at,
                        "slurm_job_id": e.slurm_job_id,
                        "model": e.model,
                    })

            # ── Phase 2: parallel health checks (no DB session held) ────
            if checks:
                results = await asyncio.gather(*[
                    health_check_endpoint(c["host"], c["port"])
                    for c in checks
                ])
            else:
                results = []

            # ── Phase 3: apply state transitions ────────────────────────
            with SessionLocal() as db:
                for check_info, (ok, err) in zip(checks, results):
                    ep = db.get(Endpoint, check_info["id"])
                    if not ep or ep.state not in ("STARTING", "READY"):
                        continue  # state changed between phases

                    if ok:
                        if ep.state == "STARTING":
                            ep.state = "READY"
                            ep.last_error = None
                            print(
                                f"health_worker: endpoint {ep.model} "
                                f"(job {ep.slurm_job_id}) is now READY"
                            )
                            lease = db.execute(
                                select(Lease).where(
                                    Lease.slurm_job_id == ep.slurm_job_id
                                )
                            ).scalars().first()
                            if lease and lease.state in (
                                "SUBMITTED", "STARTING"
                            ):
                                lease.state = "RUNNING"
                                print(
                                    f"health_worker: lease {lease.id} "
                                    f"({lease.model}) → RUNNING"
                                )
                        elif ep.state == "READY":
                            ep.last_error = None

                    else:
                        if ep.state == "READY":
                            ep.state = "FAILED"
                            ep.last_error = err
                            print(
                                f"health_worker: endpoint {ep.model} "
                                f"(job {ep.slurm_job_id}) FAILED: {err}"
                            )
                            lease = db.execute(
                                select(Lease).where(
                                    Lease.slurm_job_id == ep.slurm_job_id
                                )
                            ).scalars().first()
                            if lease and lease.state == "RUNNING":
                                lease.state = "FAILED"
                                lease.failed_at = now
                                print(
                                    f"health_worker: lease {lease.id} "
                                    f"({lease.model}) → FAILED"
                                )

                        elif ep.state == "STARTING":
                            age = (
                                now
                                - ensure_utc(check_info["created_at"])
                            ).total_seconds()
                            if age > settings.vllm_health_timeout_seconds:
                                ep.state = "FAILED"
                                ep.last_error = (
                                    f"Timed out after {age:.0f}s: {err}"
                                )
                                print(
                                    f"health_worker: endpoint {ep.model} "
                                    f"(job {ep.slurm_job_id}) timed out "
                                    f"after {age:.0f}s"
                                )
                                lease = db.execute(
                                    select(Lease).where(
                                        Lease.slurm_job_id
                                        == ep.slurm_job_id
                                    )
                                ).scalars().first()
                                if lease and lease.state in (
                                    "SUBMITTED", "STARTING"
                                ):
                                    lease.state = "FAILED"
                                    lease.failed_at = now
                            else:
                                ep.last_error = err

                    ep.last_health_at = now

                # ── Fallback reconciliation ─────────────────────────────
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

        await asyncio.sleep(60)


async def planned_submit_worker():
    from .admin import _submit_to_slurm_from_snapshot, _snapshot_lease

    while True:
        try:
            now = datetime.now(timezone.utc)
            lead = timedelta(seconds=settings.scheduler_submit_lead_seconds)

            # ── Phase 1: find leases that need submission ───────────────
            with SessionLocal() as db:
                planned = db.execute(
                    select(Lease).where(
                        Lease.state == "PLANNED",
                        Lease.begin_at != None  # noqa
                    ).order_by(Lease.begin_at.asc())
                ).scalars().all()

                to_submit: list[dict] = []
                to_expire: list[int] = []

                for l in planned:
                    if not l.begin_at:
                        continue

                    # Guard: skip leases whose end_at has already passed
                    if l.end_at and ensure_utc(l.end_at) < now:
                        to_expire.append(l.id)
                        continue

                    if ensure_utc(l.begin_at) <= now + lead:
                        to_submit.append(_snapshot_lease(l))

                # Mark expired leases immediately
                for lid in to_expire:
                    lease = db.get(Lease, lid)
                    if lease and lease.state == "PLANNED":
                        lease.state = "ENDED"
                        print(
                            f"planned_submit_worker: lease {lid} "
                            f"({lease.model}) → ENDED (end_at already passed)"
                        )
                db.commit()

            # ── Phase 2: submit to Slurm (NO DB session held) ──────────
            for snapshot in to_submit:
                try:
                    # Submit to Slurm without any DB session open
                    job_id = await asyncio.to_thread(
                        _submit_to_slurm_from_snapshot, snapshot
                    )

                    # Phase 3: write result back to DB
                    with SessionLocal() as db:
                        lease = db.get(Lease, snapshot["id"])
                        if not lease or lease.state != "PLANNED":
                            print(
                                f"planned_submit_worker: lease {snapshot['id']} "
                                f"state changed during submit, skipping"
                            )
                            continue

                        lease.slurm_job_id = job_id
                        lease.state = "SUBMITTED"
                        print(
                            f"planned_submit_worker: lease {lease.id} "
                            f"({lease.model}) → SUBMITTED "
                            f"(job {job_id})"
                        )
                        db.commit()

                except Exception as e:
                    with SessionLocal() as db:
                        lease = db.get(Lease, snapshot["id"])
                        if lease and lease.state == "PLANNED":
                            lease.state = "FAILED"
                            lease.failed_at = now
                            db.commit()
                    print(
                        f"planned_submit_worker: failed to submit "
                        f"lease {snapshot['id']}: {e}"
                    )

        except Exception as e:
            print(f"planned_submit_worker error: {e}")

        await asyncio.sleep(5)


async def endpoint_cleanup_worker():
    while True:
        try:
            now = datetime.now(timezone.utc)

            # ── Phase 1: find endpoints to clean up ─────────────────────
            with SessionLocal() as db:
                eps = db.execute(
                    select(Endpoint).where(
                        Endpoint.state.in_(["READY", "STARTING", "FAILED"])
                    )
                ).scalars().all()

                actions: list[dict] = []
                for e in eps:
                    lease = db.execute(
                        select(Lease).where(
                            Lease.slurm_job_id == e.slurm_job_id
                        )
                    ).scalars().first()

                    if lease and lease.end_at and ensure_utc(lease.end_at) < now:
                        actions.append({
                            "endpoint_id": e.id,
                            "lease_id": lease.id,
                            "lease_state": lease.state,
                            "slurm_job_id": lease.slurm_job_id,
                            "action": "expired",
                        })
                    elif lease and lease.state in ("CANCELED", "ENDED"):
                        actions.append({
                            "endpoint_id": e.id,
                            "lease_id": lease.id,
                            "lease_state": lease.state,
                            "slurm_job_id": lease.slurm_job_id,
                            "action": "lease_done",
                        })

            # ── Phase 2: scancel jobs (non-blocking) ───────────────────
            for act in actions:
                if act["slurm_job_id"]:
                    try:
                        await slurm.async_cancel(act["slurm_job_id"])
                        print(
                            f"endpoint_cleanup: scancel'd {act['action']} job "
                            f"{act['slurm_job_id']}"
                        )
                    except Exception as ex:
                        # Job might already be gone — that's fine
                        print(
                            f"endpoint_cleanup: scancel failed for "
                            f"{act['slurm_job_id']}: {ex}"
                        )


            # ── Phase 3: update DB state ────────────────────────────────
            if actions:
                with SessionLocal() as db:
                    for act in actions:
                        ep = db.get(Endpoint, act["endpoint_id"])
                        if not ep:
                            continue
                        ep.state = "STOPPED"

                        lease = db.get(Lease, act["lease_id"])
                        if not lease:
                            continue

                        if act["action"] == "expired":
                            if lease.state == "RUNNING":
                                lease.state = "ENDED"
                                print(
                                    f"endpoint_cleanup: lease {lease.id} "
                                    f"({lease.model}) → ENDED (expired)"
                                )
                            # Leave FAILED leases as FAILED
                        # For lease_done action, endpoint just gets STOPPED

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

            # ── Phase 1: snapshot leases that need reconciliation ───────
            with SessionLocal() as db:
                active = db.execute(
                    select(Lease).where(
                        Lease.state.in_(
                            ["SUBMITTED", "STARTING", "RUNNING", "FAILED", "RETRYING"]
                        )
                    )
                ).scalars().all()

                checks: list[dict] = []
                failed_expiry: list[dict] = []

                for l in active:
                    if l.state in ("FAILED", "RETRYING"):
                        if l.end_at and ensure_utc(l.end_at) < now:
                            failed_expiry.append({
                                "id": l.id,
                                "slurm_job_id": l.slurm_job_id,
                            })
                        continue

                    if not l.slurm_job_id:
                        continue

                    checks.append({
                        "id": l.id,
                        "model": l.model,
                        "slurm_job_id": l.slurm_job_id,
                        "state": l.state,
                    })

            # ── Phase 2: batch query Slurm (single subprocess call) ─────
            job_ids_to_check = [c["slurm_job_id"] for c in checks]
            if job_ids_to_check:
                try:
                    squeue_results = await slurm.async_squeue_job_states_batch(
                        job_ids_to_check
                    )
                except slurm.SlurmUnavailableError as e:
                    print(f"slurm_reconcile: Slurm controller unavailable ({e}), skipping this cycle")
                    await asyncio.sleep(5)
                    continue
            else:
                squeue_results = {}


            # ── Phase 3: apply state changes ────────────────────────────
            with SessionLocal() as db:
                # Handle FAILED leases past end_at
                for info in failed_expiry:
                    l = db.get(Lease, info["id"])
                    if not l or l.state != "FAILED":
                        continue
                    l.state = "ENDED"
                    if l.slurm_job_id:
                        ep = db.execute(
                            select(Endpoint).where(
                                Endpoint.slurm_job_id == l.slurm_job_id
                            )
                        ).scalars().first()
                        if ep:
                            ep.state = "STOPPED"

                # Handle active leases vs Slurm reality
                for info in checks:
                    slurm_state = squeue_results.get(info["slurm_job_id"])

                    if slurm_state is not None:
                        continue  # Job still in Slurm, nothing to do

                    l = db.get(Lease, info["id"])
                    if not l or l.state not in (
                        "SUBMITTED", "STARTING", "RUNNING"
                    ):
                        continue

                    ep = db.execute(
                        select(Endpoint).where(
                            Endpoint.slurm_job_id == info["slurm_job_id"]
                        )
                    ).scalars().first()

                    if ep is None or ep.state in ("FAILED", "STARTING"):
                        l.state = "FAILED"
                        l.failed_at = now
                        if ep:
                            ep.state = "FAILED"
                        print(
                            f"slurm_reconcile: lease {l.id} "
                            f"({l.model}) → FAILED "
                            f"(job {info['slurm_job_id']} gone from squeue)"
                        )
                    else:
                        if l.state not in ("CANCELED",):
                            l.state = "ENDED"
                        if ep:
                            ep.state = "STOPPED"
                        print(
                            f"slurm_reconcile: lease {l.id} "
                            f"({l.model}) → ENDED "
                            f"(job {info['slurm_job_id']} gone from squeue)"
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
    from .admin import _submit_to_slurm_from_snapshot, _snapshot_lease

    while True:
        try:
            now = datetime.now(timezone.utc)

            # ── Phase 1: find eligible retries & snapshot them ──────────
            candidates: list[dict] = []
            with SessionLocal() as db:
                failed = db.execute(
                    select(Lease).where(Lease.state == "FAILED")
                ).scalars().all()

                for lease in failed:
                    if lease.end_at and ensure_utc(lease.end_at) < now:
                        continue
                    if lease.retry_count >= settings.vllm_max_retries:
                        continue
                    if not lease.failed_at:
                        continue
                    since_fail = (
                        now - ensure_utc(lease.failed_at)
                    ).total_seconds()
                    if since_fail < settings.vllm_retry_delay_seconds:
                        continue
                    if lease.end_at:
                        remaining = (
                            ensure_utc(lease.end_at) - now
                        ).total_seconds()
                        if remaining < 120:
                            print(
                                f"retry_worker: lease {lease.id} "
                                f"({lease.model}) has only "
                                f"{remaining:.0f}s left, skipping"
                            )
                            continue

                    snapshot = _snapshot_lease(lease)
                    snapshot["retry_count"] = lease.retry_count
                    candidates.append(snapshot)

            # ── Phase 2: attempt retries (NO DB session held) ───────────
            for snapshot in candidates:
                try:
                    # Phase 2a: mark retry in progress with RETRYING state
                    with SessionLocal() as db:
                        lease = db.get(Lease, snapshot["id"])
                        if not lease or lease.state != "FAILED":
                            continue

                        lease.retry_count += 1
                        lease.state = "RETRYING"  # Intermediate state prevents other workers from touching it
                        old_slurm_job_id = lease.slurm_job_id

                        print(
                            f"retry_worker: retrying lease {lease.id} "
                            f"({lease.model}), attempt "
                            f"{lease.retry_count}/"
                            f"{settings.vllm_max_retries}"
                        )

                        # Clean up old endpoint
                        if lease.slurm_job_id:
                            old_ep = db.execute(
                                select(Endpoint).where(
                                    Endpoint.slurm_job_id
                                    == lease.slurm_job_id
                                )
                            ).scalars().first()
                            if old_ep:
                                old_ep.state = "STOPPED"

                        db.commit()

                    if old_slurm_job_id:
                        try:
                            await slurm.async_cancel(old_slurm_job_id)
                            print(f"retry_worker: scancel'd old job {old_slurm_job_id}")
                        except Exception as ex:
                            print(f"retry_worker: scancel failed for {old_slurm_job_id}: {ex}")

                    # Phase 2b: submit to Slurm (NO DB session open)
                    new_job_id = await asyncio.to_thread(
                        _submit_to_slurm_from_snapshot, snapshot
                    )

                    # Phase 2c: write result back to DB
                    with SessionLocal() as db:
                        lease = db.get(Lease, snapshot["id"])
                        if not lease:
                            continue
                        lease.slurm_job_id = new_job_id
                        lease.state = "SUBMITTED"
                        lease.failed_at = None
                        print(
                            f"retry_worker: lease {lease.id} resubmitted "
                            f"as Slurm job {new_job_id}"
                        )
                        db.commit()

                except Exception as ex:
                    print(
                        f"retry_worker: failed to resubmit "
                        f"lease {snapshot['id']}: {ex}"
                    )
                    # Revert to FAILED so it can be retried again later
                    with SessionLocal() as db:
                        lease = db.get(Lease, snapshot["id"])
                        if lease and lease.state == "RETRYING":
                            lease.state = "FAILED"
                            lease.failed_at = now
                            db.commit()

        except Exception as e:
            print(f"retry_worker error: {e}")

        await asyncio.sleep(10)


def reconcile_on_startup():
    """
    On startup, cross-reference DB state with Slurm reality.

    Uses batched squeue calls to avoid blocking startup with N subprocess calls.

    Handles:
    - Leases in SUBMITTED/STARTING/RUNNING whose Slurm jobs are gone → FAILED or ENDED
    - PLANNED leases whose end_at has already passed → ENDED
    - Endpoints in STARTING/READY whose Slurm jobs are gone → STOPPED
    """
    print("startup: reconciling DB state with Slurm...")
    now = datetime.now(timezone.utc)
    changes = 0

    with SessionLocal() as db:
        # 1. Check active leases against Slurm (batched)
        active_leases = db.execute(
            select(Lease).where(
                Lease.state.in_(["SUBMITTED", "STARTING", "RUNNING", "RETRYING"])
            )
        ).scalars().all()


        # Collect all job IDs we need to check
        lease_job_ids = [
            l.slurm_job_id for l in active_leases if l.slurm_job_id
        ]

        # Single batched squeue call
        try:
            job_states = slurm.squeue_job_states_batch(lease_job_ids) if lease_job_ids else {}
        except slurm.SlurmUnavailableError as e:
            print(f"  reconcile: Slurm controller unavailable ({e}), skipping lease reconciliation")
            job_states = None

        if job_states is None:
            # Skip lease reconciliation but still handle PLANNED expiry below
            pass
        else:
            for lease in active_leases:
                if not lease.slurm_job_id:
                    continue

                state = job_states.get(lease.slurm_job_id)

                if state is None:
                    # Job is gone from Slurm
                    ep = db.execute(
                        select(Endpoint).where(
                            Endpoint.slurm_job_id == lease.slurm_job_id
                        )
                    ).scalars().first()

                    if ep and ep.state == "READY":
                        # Was running fine, job ended (time limit, etc.)
                        lease.state = "ENDED"
                        ep.state = "STOPPED"
                        print(
                            f"  reconcile: lease {lease.id} ({lease.model}) "
                            f"→ ENDED (job {lease.slurm_job_id} gone, was READY)"
                        )
                    else:
                        # Never became READY or was in STARTING
                        lease.state = "FAILED"
                        lease.failed_at = now
                        if ep:
                            ep.state = "FAILED"
                        print(
                            f"  reconcile: lease {lease.id} ({lease.model}) "
                            f"→ FAILED (job {lease.slurm_job_id} gone from squeue)"
                        )
                    changes += 1
                else:
                    print(
                        f"  reconcile: lease {lease.id} ({lease.model}) "
                        f"— Slurm job {lease.slurm_job_id} still {state}"
                    )

        # 2. PLANNED leases whose end_at has passed
        planned = db.execute(
            select(Lease).where(Lease.state == "PLANNED")
        ).scalars().all()

        for lease in planned:
            if lease.end_at and ensure_utc(lease.end_at) < now:
                lease.state = "ENDED"
                print(
                    f"  reconcile: planned lease {lease.id} ({lease.model}) "
                    f"→ ENDED (end_at {lease.end_at} already passed)"
                )
                changes += 1

        # 2b. RETRYING leases left from a crash → revert to FAILED
        retrying = db.execute(
            select(Lease).where(Lease.state == "RETRYING")
        ).scalars().all()

        for lease in retrying:
            lease.state = "FAILED"
            lease.failed_at = now
            print(
                f"  reconcile: lease {lease.id} ({lease.model}) "
                f"RETRYING → FAILED (crash recovery)"
            )
            changes += 1

        # 3. Orphaned endpoints (no matching active lease)
        eps = db.execute(
            select(Endpoint).where(
                Endpoint.state.in_(["STARTING", "READY"])
            )
        ).scalars().all()

        orphan_job_ids = []
        orphan_eps = []
        for ep in eps:
            lease = db.execute(
                select(Lease).where(
                    Lease.slurm_job_id == ep.slurm_job_id,
                    Lease.state.in_(
                        ["SUBMITTED", "STARTING", "RUNNING"]
                    ),
                )
            ).scalars().first()
            if not lease:
                orphan_job_ids.append(ep.slurm_job_id)
                orphan_eps.append(ep)

        # Batch check orphan endpoints
        try:
            orphan_states = slurm.squeue_job_states_batch(orphan_job_ids) if orphan_job_ids else {}
        except slurm.SlurmUnavailableError:
            orphan_states = None

        if orphan_states is not None:
            for ep in orphan_eps:
                state = orphan_states.get(ep.slurm_job_id)
                if state is None:
                    ep.state = "STOPPED"
                    print(
                        f"  reconcile: orphan endpoint {ep.model} "
                        f"(job {ep.slurm_job_id}) → STOPPED"
                    )
                    changes += 1

        db.commit()

    print(f"startup: reconciliation complete ({changes} changes)")
