from __future__ import annotations
import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from .auth import auth_router, require_auth, get_session
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

from .settings import settings
from .dependencies import SessionLocal, init_db
from .models import Endpoint, Lease
from .catalog import get_catalog
from .schemas import OpenAIModelsResponse
from .admin import router as admin_router
from .router_core import choose_ready_endpoint, health_check_endpoint
from .proxy import proxy_get, proxy_json_or_stream, proxy_multipart
from .admin import internal_router
from .images_api import router as images_router
from .public_api import router as public_api_router, metrics_router
from . import slurm
from .backends import ClusterUnavailableError
from .backends import get_backend as slurm_backend
from .utils import ensure_utc
from .lifecycle_logger import log_health_check, log_state_transition, log_slurm_action
from .metrics import generate_latest, UPSTREAM_HEALTHY
from .leader import get_election, is_leader

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Make the app's own loggers visible.

    uvicorn installs handlers for `uvicorn.*` only; everything else inherits a
    bare root logger at WARNING, so our INFO messages vanish. Attach a handler
    to the `app` package rather than the root logger, so third-party libraries
    stay as quiet as they were.
    """
    level = getattr(logging, (settings.log_level or "INFO").upper(), logging.INFO)
    app_logger = logging.getLogger("app")
    app_logger.setLevel(level)

    # Only supply a handler when nothing else will. Under uvicorn the root
    # logger has none, so ours is the only route to stderr. When the host
    # application *has* configured logging (or pytest's caplog is attached),
    # propagation alone delivers the records and adding a handler here would
    # duplicate every line.
    #
    # Propagation deliberately stays on: turning it off silences the app for
    # anyone who configures logging at the root, and breaks caplog in tests.
    if not app_logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(levelname)s:    %(name)s: %(message)s"
        ))
        app_logger.addHandler(handler)


configure_logging()


# ── Supervised task wrapper ─────────────────────────────────────────────────
async def _supervised(name: str, coro_fn, restart_delay: float = 2.0):
    """
    Run an async worker forever. If it crashes, log and restart after a delay.
    Only exits on asyncio.CancelledError (i.e. shutdown).

    Under HA the worker only runs on the leader; followers idle here and take
    over within one lock TTL if the leader dies.
    """
    while True:
        try:
            if not is_leader():
                await asyncio.sleep(LEADER_POLL_SECONDS)
                continue
            await coro_fn()
        except asyncio.CancelledError:
            logger.info("%s: cancelled, shutting down", name)
            return
        except Exception as e:
            logger.error("%s: crashed (%s), restarting in %ss", name, e, restart_delay)
            await asyncio.sleep(restart_delay)


async def _forever(name: str, coro_fn, restart_delay: float = 2.0):
    """Like `_supervised`, but runs on every instance — not just the leader."""
    while True:
        try:
            await coro_fn()
        except asyncio.CancelledError:
            logger.info("%s: cancelled, shutting down", name)
            return
        except Exception as e:
            logger.error("%s: crashed (%s), restarting in %ss", name, e, restart_delay)
            await asyncio.sleep(restart_delay)


# ── Leader election ─────────────────────────────────────────────────────────
# Every instance serves proxy traffic; only the leader runs the workers.
LEADER_POLL_SECONDS = 5
HEARTBEAT_SECONDS = 15


async def leader_worker():
    """Acquire and heartbeat the worker lock.

    Runs on every instance regardless of leadership — it is what *decides*
    leadership.
    """
    election = get_election()
    while True:
        try:
            await asyncio.to_thread(election.try_acquire)
        except Exception as e:
            logger.error("leader_worker error: %s", e)
        await asyncio.sleep(HEARTBEAT_SECONDS)


def _ensure_job_log_dir() -> None:
    """Create the shared job-log directory if we can reach it.

    Slurm does not create it: a missing directory means the job cannot open
    its output file and dies at launch, with no log to say why. Creating it is
    only possible when the same filesystem is mounted here, so failure is a
    warning — off-cluster routers legitimately cannot see it, and the cluster
    admin may have to create it with the right ownership anyway.
    """
    local = os.path.abspath(settings.job_log_dir_local)
    try:
        os.makedirs(local, exist_ok=True)
    except OSError as e:
        logger.warning(
            "job log directory %s is not writable from here (%s); Slurm jobs "
            "will fail at launch unless %s exists on the compute nodes",
            local, e, settings.job_log_dir,
        )


# ── Lifespan (replaces deprecated @app.on_event) ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: reconcile DB with Slurm, then launch supervised background workers.
    Shutdown: cancel all workers, close the shared httpx proxy client.
    """
    # ── Startup ──
    _ensure_job_log_dir()
    # Only the leader may reconcile: it cancels and resubmits Slurm jobs, and
    # two instances doing that at once would fight.
    tasks = []
    if settings.ha_enabled:
        election = get_election()
        await asyncio.to_thread(election.try_acquire)
        tasks.append(asyncio.create_task(_forever("leader_worker", leader_worker)))
        if election.is_leader:
            reconcile_on_startup()
        else:
            logger.info("starting as follower — serving proxy traffic only")
    else:
        reconcile_on_startup()

    tasks += [
        asyncio.create_task(_supervised("inventory_worker", inventory_worker)),
        asyncio.create_task(_supervised("estimate_worker", estimate_worker)),
        asyncio.create_task(_supervised("health_worker", health_worker)),
        asyncio.create_task(_supervised("planned_submit_worker", planned_submit_worker)),
        asyncio.create_task(_supervised("endpoint_cleanup_worker", endpoint_cleanup_worker)),
        asyncio.create_task(_supervised("slurm_reconcile_worker", slurm_reconcile_worker)),
        asyncio.create_task(_supervised("retry_worker", retry_worker)),
        asyncio.create_task(_supervised("renew_worker", renew_worker)),
        asyncio.create_task(_supervised("image_build_worker", image_build_worker)),
    ]

    yield  # ← app is running and serving requests

    # ── Shutdown ──
    logger.info("lifespan: shutting down background workers...")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    if settings.ha_enabled:
        # Hand leadership over promptly instead of waiting out the TTL.
        await asyncio.to_thread(get_election().release)

    # Close shared HTTP clients
    from .proxy import close_client
    from .router_core import close_health_client
    await close_client()
    await close_health_client()

    logger.info("lifespan: shutdown complete")


app = FastAPI(title="KatherLab LLM Scheduler", version="0.4.0", lifespan=lifespan)
app.include_router(admin_router)
app.include_router(images_router)
app.include_router(auth_router)
app.include_router(internal_router)
app.include_router(public_api_router)
app.include_router(metrics_router)

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
            "tags": m.tags or [],
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


async def _get_model_from_multipart(request: Request) -> str:
    form = await request.form()
    model = form.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="Missing 'model' in multipart form data")
    return str(model)


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
        model=model,
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
        model=model,
    )


@app.post("/v1/responses")
async def responses(request: Request):
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
        upstream_url=f"{upstream}/v1/responses",
        body=body,
        is_stream=is_stream,
        model=model,
    )


@app.post("/v1/responses/{response_id}/cancel")
async def cancel_response(response_id: str, request: Request):
    body = await request.body()
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    model = j.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="Missing 'model' in request body")

    with SessionLocal() as db:
        upstream = _resolve_upstream(db, model)
    return await proxy_json_or_stream(
        request,
        upstream_url=f"{upstream}/v1/responses/{response_id}/cancel",
        body=body,
        is_stream=False,
        model=model,
    )

@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(request: Request):
    model = await _get_model_from_multipart(request)
    with SessionLocal() as db:
        upstream = _resolve_upstream(db, model)

    return await proxy_multipart(
        request,
        upstream_url=f"{upstream}/v1/audio/transcriptions",
        model=model,
    )


@app.post("/v1/audio/translations")
async def audio_translations(request: Request):
    model = await _get_model_from_multipart(request)
    with SessionLocal() as db:
        upstream = _resolve_upstream(db, model)

    return await proxy_multipart(
        request,
        upstream_url=f"{upstream}/v1/audio/translations",
        model=model,
    )


@app.get("/metrics")
async def metrics(request: Request, model: Optional[str] = None):
    """Proxy-level Prometheus metrics, or forward to a vLLM instance.

    - ``GET /metrics``               — proxy's own Prometheus metrics
    - ``GET /metrics?model=xxx``     — vLLM instance metrics for that model
    """
    if model is None:
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(generate_latest().decode("utf-8"), media_type="text/plain")
    with SessionLocal() as db:
        upstream = _resolve_upstream(db, model)
    return await proxy_get(request, upstream_url=f"{upstream}/metrics", model=model)


# =============================================================================
# HEALTH WORKER — unified, adaptive polling
#
# - STARTING endpoints: polled every cycle for fast readiness detection
# - READY endpoints: polled every ~15s for liveness monitoring
# - Handles ALL state transitions:
#     STARTING → READY  (lease → RUNNING)
#     READY → FAILED    (lease → FAILED) — after READY_FAIL_THRESHOLD consecutive failures
#     STARTING timeout → FAILED (lease → FAILED)
# - Fallback: any READY endpoint whose lease is still STARTING/SUBMITTED
#   gets its lease promoted to RUNNING (catches missed transitions)
# =============================================================================

# =============================================================================
# INVENTORY WORKER
#
# Refreshes the node/foreign-job snapshot that replaces the hand-maintained
# TOTAL_GPUS. Node topology changes rarely, but node *state* (drain/down) and
# foreign jobs change often enough that a stale view would mis-plan.
#
# Discovery failures keep the previous snapshot rather than reporting an empty
# cluster, which would make every booking look unplaceable.
# =============================================================================
INVENTORY_REFRESH_SECONDS = 60


async def inventory_worker():
    from . import inventory

    while True:
        try:
            inv = await inventory.refresh()
            if inv.error:
                logger.warning("inventory: %s", inv.error)
            else:
                logger.debug(
                    "inventory: %d nodes, %d GPUs, %d foreign jobs",
                    len(inv.nodes), inv.total_gpus, len(inv.foreign_jobs),
                )
        except Exception as e:
            logger.error("inventory_worker error: %s", e)
        await asyncio.sleep(INVENTORY_REFRESH_SECONDS)


# =============================================================================
# ESTIMATE WORKER
#
# Keeps Slurm's backfill start estimate fresh for queued jobs on `slurm` pools,
# where our calendar is not the allocator and the honest answer to "when does
# this start?" comes from Slurm.
#
# The estimate genuinely moves as the queue changes, so this must re-poll
# rather than record it once. Slurm recomputes it only every `bf_interval`
# (~30s default), so polling faster would just be noise.
# =============================================================================
ESTIMATE_REFRESH_SECONDS = 45


async def estimate_worker():
    from .cluster import get_cluster

    while True:
        try:
            cluster = get_cluster()

            # ── Phase 1: which queued leases need an estimate ────────────
            with SessionLocal() as db:
                pending = [
                    {"id": x.id, "job": x.slurm_job_id, "pool": x.pool}
                    for x in db.execute(
                        select(Lease).where(Lease.state.in_(["SUBMITTED", "STARTING"]))
                    ).scalars().all()
                    if x.slurm_job_id
                ]

            # Managed pools promise a start time; only `slurm` pools estimate.
            targets = [
                p for p in pending
                if (pool := cluster.pool(p["pool"])) is None or not pool.is_managed
            ]
            if not targets:
                await asyncio.sleep(ESTIMATE_REFRESH_SECONDS)
                continue

            # ── Phase 2: ask Slurm (outside the DB session) ──────────────
            try:
                states = await slurm_backend().job_states([t["job"] for t in targets])
            except ClusterUnavailableError as e:
                logger.warning("estimate_worker: cluster unavailable (%s), skipping", e)
                await asyncio.sleep(ESTIMATE_REFRESH_SECONDS)
                continue

            # ── Phase 3: write the estimates back ────────────────────────
            now = datetime.now(timezone.utc)
            with SessionLocal() as db:
                for target in targets:
                    state = states.get(target["job"])
                    if state is None:
                        continue
                    lease = db.get(Lease, target["id"])
                    if lease is None:
                        continue
                    if state.is_running:
                        # It started — the estimate is now history.
                        lease.estimated_start = None
                    elif state.start_time is not None:
                        lease.estimated_start = state.start_time
                    lease.estimate_updated_at = now
                db.commit()
        except Exception as e:
            logger.error("estimate_worker error: %s", e)
        await asyncio.sleep(ESTIMATE_REFRESH_SECONDS)


# =============================================================================
# RENEW WORKER  (mode: service)
#
# A long-running model cannot be booked for a month — clusters cap wall time.
# So a service deployment is a chain of bounded jobs handed over without
# downtime:
#
#   submit replacement -> wait for READY -> drain old -> cancel old
#
# The drain step depends on least-loaded routing: both replicas serve the same
# model briefly, and the drain flag steers traffic one way.
# =============================================================================
RENEW_INTERVAL_SECONDS = 30


async def renew_worker():
    from . import renew
    from .admin import _snapshot_lease, _submit_to_slurm_from_snapshot

    while True:
        try:
            now = datetime.now(timezone.utc)

            # ── Phase 1: decide what to do (no I/O) ─────────────────────
            with SessionLocal() as db:
                leases = db.execute(
                    select(Lease).where(Lease.state.in_(renew.ACTIVE))
                ).scalars().all()

                # (a) service leases nearing expiry get a replacement
                to_renew = renew.plan_renewals(leases, now)
                created: list[dict] = []
                for lease in to_renew:
                    replacement = renew.build_replacement(lease, now)
                    db.add(replacement)
                    db.flush()
                    log_state_transition(
                        entity="lease", entity_id=replacement.id, model=replacement.model,
                        old_state="-", new_state="PLANNED", slurm_job_id=None,
                        reason=f"service renewal, supersedes lease {lease.id}",
                    )
                    created.append(_snapshot_lease(replacement))

                # (b) replacements that are READY -> start draining the old one
                retire: list[int] = []
                for lease in leases:
                    if not lease.supersedes_id:
                        continue
                    if not renew.replacement_is_ready(db, lease):
                        continue
                    old = db.get(Lease, lease.supersedes_id)
                    if old is None or old.state not in renew.ACTIVE:
                        continue
                    renew.begin_drain(db, old, now)
                    if renew.drain_complete(db, old, now):
                        retire.append(old.id)

                db.commit()

            # ── Phase 2: Slurm calls, outside the DB session ────────────
            submitted: list[tuple[int, str]] = []
            for snapshot in created:
                try:
                    job_id = await asyncio.to_thread(
                        _submit_to_slurm_from_snapshot, snapshot
                    )
                    submitted.append((snapshot["id"], job_id))
                except Exception as e:
                    logger.error(
                        "renew_worker: submit failed for replacement lease %s: %s",
                        snapshot["id"], e,
                    )

            for lease_id in retire:
                with SessionLocal() as db:
                    old = db.get(Lease, lease_id)
                    job_id = old.slurm_job_id if old else None
                if job_id:
                    try:
                        await slurm.async_cancel(job_id)
                    except Exception as e:
                        logger.warning("renew_worker: cancel %s failed: %s", job_id, e)

            # ── Phase 3: write results back ─────────────────────────────
            with SessionLocal() as db:
                for lease_id, job_id in submitted:
                    lease = db.get(Lease, lease_id)
                    if lease:
                        lease.slurm_job_id = job_id
                        lease.state = "SUBMITTED"
                for lease_id in retire:
                    old = db.get(Lease, lease_id)
                    if old is None:
                        continue
                    renew.forget_endpoints(db, old)
                    old.state = "ENDED"
                    old.end_at = now
                    for endpoint in renew.endpoints_for(db, old):
                        endpoint.state = "STOPPED"
                    log_state_transition(
                        entity="lease", entity_id=old.id, model=old.model,
                        old_state="RUNNING", new_state="ENDED",
                        slurm_job_id=old.slurm_job_id,
                        reason="retired after service handover",
                    )
                db.commit()
        except Exception as e:
            logger.error("renew_worker error: %s", e)
        await asyncio.sleep(RENEW_INTERVAL_SECONDS)


READY_FAIL_THRESHOLD = 3  # consecutive health-check failures before marking READY → FAILED

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
                # Update UPSTREAM_HEALTHY gauge: 1 for READY models, 0 otherwise
                all_model_rows = db.execute(select(Endpoint.model)).scalars().all()
                ready_model_rows = db.execute(
                    select(Endpoint.model).where(Endpoint.state == "READY")
                ).scalars().all()
                ready_set = set(ready_model_rows)
                for m in set(all_model_rows):
                    UPSTREAM_HEALTHY.labels(model=m).set(1 if m in ready_set else 0)

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
                import time as _time
                t0_all = _time.perf_counter()
                results_raw = await asyncio.gather(*[
                    _timed_health_check(c["host"], c["port"])
                    for c in checks
                ])
                # results_raw is list of (ok, err, elapsed_ms)
            else:
                results_raw = []

            # ── Phase 3: apply state transitions ────────────────────────
            with SessionLocal() as db:
                for check_info, (ok, err, elapsed_ms) in zip(checks, results_raw):
                    ep = db.get(Endpoint, check_info["id"])
                    if not ep or ep.state not in ("STARTING", "READY"):
                        continue  # state changed between phases

                    if ok:
                        # ── Success ─────────────────────────────────────
                        if ep.state == "STARTING":
                            old_state = ep.state
                            ep.state = "READY"
                            ep.last_error = None
                            ep.health_fail_count = 0

                            log_state_transition(
                                entity="endpoint",
                                entity_id=ep.id,
                                model=ep.model,
                                old_state=old_state,
                                new_state="READY",
                                slurm_job_id=ep.slurm_job_id,
                                reason="health check passed",
                            )

                            lease = db.execute(
                                select(Lease).where(
                                    Lease.slurm_job_id == ep.slurm_job_id
                                )
                            ).scalars().first()
                            if lease and lease.state in (
                                "SUBMITTED", "STARTING"
                            ):
                                old_ls = lease.state
                                lease.state = "RUNNING"
                                # A model that reached READY has recovered;
                                # clear the counter so a future failure retries
                                # promptly instead of at the backoff ceiling.
                                lease.retry_count = 0
                                log_state_transition(
                                    entity="lease",
                                    entity_id=lease.id,
                                    model=lease.model,
                                    old_state=old_ls,
                                    new_state="RUNNING",
                                    slurm_job_id=ep.slurm_job_id,
                                )

                        elif ep.state == "READY":
                            ep.last_error = None
                            ep.health_fail_count = 0

                        log_health_check(
                            model=check_info["model"],
                            slurm_job_id=check_info["slurm_job_id"],
                            endpoint_state=check_info["state"],
                            success=True,
                            elapsed_ms=elapsed_ms,
                            fail_count=0,
                        )

                    else:
                        # ── Failure ─────────────────────────────────────
                        if ep.state == "READY":
                            ep.health_fail_count = (ep.health_fail_count or 0) + 1
                            ep.last_error = err

                            log_health_check(
                                model=check_info["model"],
                                slurm_job_id=check_info["slurm_job_id"],
                                endpoint_state="READY",
                                success=False,
                                error=err,
                                elapsed_ms=elapsed_ms,
                                fail_count=ep.health_fail_count,
                            )

                            if ep.health_fail_count >= READY_FAIL_THRESHOLD:
                                ep.state = "FAILED"

                                log_state_transition(
                                    entity="endpoint",
                                    entity_id=ep.id,
                                    model=ep.model,
                                    old_state="READY",
                                    new_state="FAILED",
                                    slurm_job_id=ep.slurm_job_id,
                                    reason=f"{ep.health_fail_count} consecutive failures: {err}",
                                )

                                lease = db.execute(
                                    select(Lease).where(
                                        Lease.slurm_job_id == ep.slurm_job_id
                                    )
                                ).scalars().first()
                                if lease and lease.state == "RUNNING":
                                    old_ls = lease.state
                                    lease.state = "FAILED"
                                    lease.failed_at = now
                                    log_state_transition(
                                        entity="lease",
                                        entity_id=lease.id,
                                        model=lease.model,
                                        old_state=old_ls,
                                        new_state="FAILED",
                                        slurm_job_id=ep.slurm_job_id,
                                        reason=f"endpoint failed after {ep.health_fail_count} consecutive health check failures",
                                    )

                        elif ep.state == "STARTING":
                            age = (
                                now
                                - ensure_utc(check_info["created_at"])
                            ).total_seconds()

                            log_health_check(
                                model=check_info["model"],
                                slurm_job_id=check_info["slurm_job_id"],
                                endpoint_state="STARTING",
                                success=False,
                                error=err,
                                elapsed_ms=elapsed_ms,
                            )

                            if age > settings.vllm_health_timeout_seconds:
                                ep.state = "FAILED"
                                ep.last_error = (
                                    f"Timed out after {age:.0f}s: {err}"
                                )

                                log_state_transition(
                                    entity="endpoint",
                                    entity_id=ep.id,
                                    model=ep.model,
                                    old_state="STARTING",
                                    new_state="FAILED",
                                    slurm_job_id=ep.slurm_job_id,
                                    reason=f"startup timeout after {age:.0f}s",
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
                                    old_ls = lease.state
                                    lease.state = "FAILED"
                                    lease.failed_at = now
                                    log_state_transition(
                                        entity="lease",
                                        entity_id=lease.id,
                                        model=lease.model,
                                        old_state=old_ls,
                                        new_state="FAILED",
                                        slurm_job_id=ep.slurm_job_id,
                                        reason=f"endpoint startup timeout after {age:.0f}s",
                                    )
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
                        old_ls = lease.state
                        lease.state = "RUNNING"
                        lease.retry_count = 0
                        log_state_transition(
                            entity="lease",
                            entity_id=lease.id,
                            model=lease.model,
                            old_state=old_ls,
                            new_state="RUNNING",
                            slurm_job_id=e.slurm_job_id,
                            reason="fallback reconciliation (endpoint already READY)",
                        )

                db.commit()

        except Exception as e:
            logger.error("health_worker error: %s", e)

        await asyncio.sleep(60)


async def _timed_health_check(host: str, port: int) -> tuple[bool, str | None, float]:
    """Wrapper that returns (ok, error, elapsed_ms)."""
    import time as _time
    t0 = _time.perf_counter()
    ok, err = await health_check_endpoint(host, port)
    elapsed_ms = (_time.perf_counter() - t0) * 1000.0
    return ok, err, elapsed_ms


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
                        logger.info(
                            "planned_submit_worker: lease %d (%s) → ENDED (end_at already passed)",
                            lid, lease.model,
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
                            logger.warning(
                                "planned_submit_worker: lease %d state changed during submit, skipping",
                                snapshot['id'],
                            )
                            continue

                        old_state = lease.state
                        lease.slurm_job_id = job_id
                        lease.state = "SUBMITTED"
                        log_slurm_action(
                            action="submit",
                            model=lease.model,
                            slurm_job_id=job_id,
                            lease_id=lease.id,
                            detail=f"planned submission",
                        )
                        log_state_transition(
                            entity="lease",
                            entity_id=lease.id,
                            model=lease.model,
                            old_state=old_state,
                            new_state="SUBMITTED",
                            slurm_job_id=job_id,
                        )
                        db.commit()


                except Exception as e:
                    with SessionLocal() as db:
                        lease = db.get(Lease, snapshot["id"])
                        if lease and lease.state == "PLANNED":
                            lease.state = "FAILED"
                            lease.failed_at = now
                            db.commit()
                    logger.error(
                        "planned_submit_worker: failed to submit lease %d: %s",
                        snapshot['id'], e,
                    )

        except Exception as e:
            logger.error("planned_submit_worker error: %s", e)

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

                    # Locked leases and service deployments outlive their
                    # booking window: the first by policy, the second because
                    # renew_worker retires them explicitly after a handover.
                    # Explicit cancellation still applies below.
                    if lease and lease.state not in ("CANCELED", "ENDED") and (
                        lease.locked or (lease.mode or "session") == "service"
                    ):
                        continue

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
                        log_slurm_action(
                            action="cancel",
                            model="(cleanup)",
                            slurm_job_id=act["slurm_job_id"],
                            lease_id=act["lease_id"],
                            detail=f"reason={act['action']}",
                        )
                    except Exception as ex:
                        # Job might already be gone — that's fine
                        logger.warning(
                            "endpoint_cleanup: scancel failed for %s: %s",
                            act["slurm_job_id"], ex,
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
                            if lease.state in ("RUNNING", "FAILED"):
                                old_ls = lease.state
                                lease.state = "ENDED"
                                log_state_transition(
                                    entity="lease",
                                    entity_id=lease.id,
                                    model=lease.model,
                                    old_state=old_ls,
                                    new_state="ENDED",
                                    slurm_job_id=act["slurm_job_id"],
                                    reason="lease expired",
                                )
                        # For lease_done action, endpoint just gets STOPPED

                    db.commit()

        except Exception as e:
            logger.error("endpoint_cleanup_worker error: %s", e)
        await asyncio.sleep(15)


# =============================================================================
# SLURM RECONCILE WORKER
#
# Reconcile leases/endpoints with Slurm reality (squeue + sacct).
# - If Slurm job is gone from squeue → check sacct for exit reason
# - OOM / NODE_FAIL / PREEMPTED / etc. → FAILED (eligible for retry)
# - Normal completion → ENDED
# - FAILED leases past their end_at → ENDED
# =============================================================================
ABNORMAL_SLURM_STATES = {
    "OUT_OF_MEMORY", "FAILED", "NODE_FAIL", "PREEMPTED", "TIMEOUT",
}

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
                    logger.warning("slurm_reconcile: Slurm controller unavailable (%s), skipping this cycle", e)
                    await asyncio.sleep(5)
                    continue
            else:
                squeue_results = {}

            # ── Phase 2b: sacct for jobs that disappeared from squeue ───
            gone_job_ids = [
                c["slurm_job_id"] for c in checks
                if squeue_results.get(c["slurm_job_id"]) is None
            ]
            if gone_job_ids:
                sacct_results = await slurm.async_sacct_job_exit_info_batch(gone_job_ids)
            else:
                sacct_results = {}

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

                    # Check sacct for exit info (always restart, even on clean exit)
                    exit_info = sacct_results.get(info["slurm_job_id"])
                    sacct_state = exit_info["state"] if exit_info else None

                    # ── Job disappeared from squeue → always restart (even on clean exit) ──
                    # Some vLLM crashes exit cleanly with code 0, so we can't rely on exit code alone.
                    # The scheduler should always restart the model when the Slurm job is gone.
                    old_ls = l.state
                    l.state = "FAILED"
                    l.failed_at = now
                    if ep:
                        ep.state = "FAILED"

                    reason = (
                        f"job gone from squeue, "
                        f"sacct_state={sacct_state}, "
                        f"exit_code={exit_info['exit_code'] if exit_info else '?'}"
                    )
                    log_state_transition(
                        entity="lease",
                        entity_id=l.id,
                        model=l.model,
                        old_state=old_ls,
                        new_state="FAILED",
                        slurm_job_id=info["slurm_job_id"],
                        reason=reason,
                    )

                db.commit()
        except Exception as e:
            logger.error("slurm_reconcile_worker error: %s", e)
        await asyncio.sleep(5)


# =============================================================================
# IMAGE BUILD WORKER
#
# Watches `apptainer build` jobs. They are ordinary Slurm jobs, so this is the
# same three-phase shape as the reconciler — with one addition: a build that
# left the queue is judged by whether the .sif is actually there. Slurm's exit
# code alone is not enough, because the job can succeed while producing
# nothing useful, and a finished-but-missing image must not look like success.
# =============================================================================
IMAGE_BUILD_POLL_SECONDS = 15


async def image_build_worker():
    from . import images
    from .models import ImageBuild

    while True:
        try:
            # ── Phase 1: snapshot ───────────────────────────────────────
            with SessionLocal() as db:
                active = db.execute(
                    select(ImageBuild).where(
                        ImageBuild.state.in_(["SUBMITTED", "RUNNING"])
                    )
                ).scalars().all()
                watching = [
                    {"id": b.id, "job_id": b.slurm_job_id, "state": b.state,
                     "name": b.image_name}
                    for b in active if b.slurm_job_id
                ]

            if not watching:
                await asyncio.sleep(IMAGE_BUILD_POLL_SECONDS)
                continue

            # ── Phase 2: ask Slurm, outside the session ─────────────────
            job_ids = [w["job_id"] for w in watching]
            try:
                states = await slurm.async_squeue_job_states_batch(job_ids)
            except slurm.SlurmUnavailableError as e:
                logger.warning(
                    "image_build_worker: Slurm unavailable (%s), skipping cycle", e
                )
                await asyncio.sleep(IMAGE_BUILD_POLL_SECONDS)
                continue

            gone = [w["job_id"] for w in watching if states.get(w["job_id"]) is None]
            exits = (
                await slurm.async_sacct_job_exit_info_batch(gone) if gone else {}
            )

            # Did the file actually appear? Only answerable where the images
            # directory is mounted here; without it we fall back to the exit
            # code and say so in the error.
            produced: dict[str, int | None] = {}
            for w in watching:
                if w["job_id"] not in gone:
                    continue
                try:
                    path = images.image_path(w["name"])
                    produced[w["name"]] = (
                        os.path.getsize(path) if os.path.isfile(path) else None
                    )
                except (images.ImageError, OSError):
                    produced[w["name"]] = -1  # cannot tell from here

            # ── Phase 3: apply ─────────────────────────────────────────
            now = datetime.now(timezone.utc)
            with SessionLocal() as db:
                for w in watching:
                    row = db.get(ImageBuild, w["id"])
                    if row is None or not row.is_active:
                        continue

                    state = states.get(w["job_id"])
                    if state is not None:
                        if state in ("RUNNING", "COMPLETING") and row.state != "RUNNING":
                            row.state = "RUNNING"
                            row.started_at = row.started_at or now
                        continue

                    size = produced.get(w["name"])
                    exit_info = exits.get(w["job_id"])
                    row.finished_at = now

                    if size is not None and size > 0:
                        row.state = "SUCCEEDED"
                        row.size_bytes = size
                        row.error = None
                    elif size == -1:
                        # No filesystem view: trust Slurm, and be honest that
                        # we could not verify the result.
                        completed = bool(exit_info and exit_info.state == "COMPLETED")
                        row.state = "SUCCEEDED" if completed else "FAILED"
                        row.error = (
                            "Job finished; the images directory is not visible "
                            "from the scheduler, so the image was not verified."
                            if completed else
                            f"Build job ended {exit_info.state if exit_info else 'unknown'}."
                        )
                    else:
                        row.state = "FAILED"
                        reason = exit_info.state if exit_info else "without a trace"
                        row.error = (
                            f"Build job ended {reason} and produced no image. "
                            "See the build log."
                        )
                    log_state_transition(
                        entity="image_build",
                        entity_id=row.id,
                        model=row.image_name,
                        old_state=w["state"],
                        new_state=row.state,
                        reason=row.error or "",
                        slurm_job_id=row.slurm_job_id,
                    )
                db.commit()

        except Exception as e:
            logger.error("image_build_worker error: %s", e, exc_info=True)

        await asyncio.sleep(IMAGE_BUILD_POLL_SECONDS)


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

                from . import renew

                for lease in failed:
                    if not renew.should_retry(
                        lease, now,
                        max_retries=settings.vllm_max_retries,
                        retry_delay=settings.vllm_retry_delay_seconds,
                    ):
                        continue

                    if renew.is_permanent(lease):
                        # Permanent deployments outlive their booking window;
                        # push it forward so the resubmitted job gets a full
                        # slot instead of one that expires immediately.
                        if lease.end_at is None or ensure_utc(lease.end_at) < now + timedelta(minutes=5):
                            lease.begin_at = now
                            lease.end_at = now + renew.RENEW_WINDOW
                        logger.info(
                            "retry_worker: resurrecting permanent model %s "
                            "(lease %d, attempt %d)",
                            lease.model, lease.id, (lease.retry_count or 0) + 1,
                        )

                    snapshot = _snapshot_lease(lease)
                    snapshot["retry_count"] = lease.retry_count
                    candidates.append(snapshot)
                db.commit()

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

                        log_slurm_action(
                            action="retry",
                            model=lease.model,
                            slurm_job_id=old_slurm_job_id,
                            lease_id=lease.id,
                            detail=f"attempt {lease.retry_count}/{settings.vllm_max_retries}",
                        )
                        log_state_transition(
                            entity="lease",
                            entity_id=lease.id,
                            model=lease.model,
                            old_state="FAILED",
                            new_state="RETRYING",
                            slurm_job_id=old_slurm_job_id,
                            reason=f"retry attempt {lease.retry_count}",
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
                            logger.info("retry_worker: scancel'd old job %s", old_slurm_job_id)
                        except Exception as ex:
                            logger.warning("retry_worker: scancel failed for %s: %s", old_slurm_job_id, ex)

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
                        log_slurm_action(
                            action="submit",
                            model=lease.model,
                            slurm_job_id=new_job_id,
                            lease_id=lease.id,
                            detail="resubmitted after retry",
                        )
                        log_state_transition(
                            entity="lease",
                            entity_id=lease.id,
                            model=lease.model,
                            old_state="RETRYING",
                            new_state="SUBMITTED",
                            slurm_job_id=new_job_id,
                        )
                        db.commit()


                except Exception as ex:
                    logger.error(
                        "retry_worker: failed to resubmit lease %d: %s",
                        snapshot["id"], ex,
                    )
                    # Revert to FAILED so it can be retried again later
                    with SessionLocal() as db:
                        lease = db.get(Lease, snapshot["id"])
                        if lease and lease.state == "RETRYING":
                            lease.state = "FAILED"
                            lease.failed_at = now
                            db.commit()

        except Exception as e:
            logger.error("retry_worker error: %s", e)

        await asyncio.sleep(10)


def reconcile_on_startup():
    """
    On startup, cross-reference DB state with Slurm reality (squeue + sacct).

    Uses batched squeue + sacct calls to avoid blocking startup with N subprocess calls.

    Handles:
    - Leases in SUBMITTED/STARTING/RUNNING whose Slurm jobs are gone → FAILED or ENDED
      (uses sacct to distinguish OOM/crash from normal completion)
    - PLANNED leases whose end_at has already passed → ENDED
    - Endpoints in STARTING/READY whose Slurm jobs are gone → STOPPED
    """
    logger.info("startup: reconciling DB state with Slurm...")
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
            logger.warning("  reconcile: Slurm controller unavailable (%s), skipping lease reconciliation", e)
            job_states = None

        if job_states is None:
            # Skip lease reconciliation but still handle PLANNED expiry below
            pass
        else:
            # sacct for jobs that disappeared from squeue
            gone_ids = [
                l.slurm_job_id for l in active_leases
                if l.slurm_job_id and job_states.get(l.slurm_job_id) is None
            ]
            sacct_info = slurm.sacct_job_exit_info_batch(gone_ids) if gone_ids else {}

            for lease in active_leases:
                if not lease.slurm_job_id:
                    continue

                state = job_states.get(lease.slurm_job_id)

                if state is None:
                    # Job is gone from Slurm — check sacct for exit reason
                    ep = db.execute(
                        select(Endpoint).where(
                            Endpoint.slurm_job_id == lease.slurm_job_id
                        )
                    ).scalars().first()

                    exit_info = sacct_info.get(lease.slurm_job_id)
                    sacct_state = exit_info["state"] if exit_info else None

                    # Job disappeared from squeue → always restart (even on clean exit).
                    # Some vLLM crashes exit cleanly with code 0, so we can't rely on exit code alone.
                    lease.state = "FAILED"
                    lease.failed_at = now
                    if ep:
                        ep.state = "FAILED"
                    logger.info(
                        "  reconcile: lease %d (%s) → FAILED (job %s gone, sacct_state=%s)",
                        lease.id, lease.model, lease.slurm_job_id, sacct_state,
                    )
                    changes += 1
                else:
                    logger.info(
                        "  reconcile: lease %d (%s) — Slurm job %s still %s",
                        lease.id, lease.model, lease.slurm_job_id, state,
                    )

        # 2. PLANNED leases whose end_at has passed
        planned = db.execute(
            select(Lease).where(Lease.state == "PLANNED")
        ).scalars().all()

        for lease in planned:
            if lease.end_at and ensure_utc(lease.end_at) < now:
                lease.state = "ENDED"
                logger.info(
                    "  reconcile: planned lease %d (%s) → ENDED (end_at %s already passed)",
                    lease.id, lease.model, lease.end_at,
                )
                changes += 1

        # 2b. RETRYING leases left from a crash → revert to FAILED
        retrying = db.execute(
            select(Lease).where(Lease.state == "RETRYING")
        ).scalars().all()

        for lease in retrying:
            lease.state = "FAILED"
            lease.failed_at = now
            logger.warning(
                "  reconcile: lease %d (%s) RETRYING → FAILED (crash recovery)",
                lease.id, lease.model,
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
                    logger.info(
                        "  reconcile: orphan endpoint %s (job %s) → STOPPED",
                        ep.model, ep.slurm_job_id,
                    )
                    changes += 1

        db.commit()

    logger.info("startup: reconciliation complete (%d changes)", changes)
