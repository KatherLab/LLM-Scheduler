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
from .metrics import get_metrics_summary
from .schemas import (
    LeaseCreate,
    LeaseOut,
    LeaseExtend,
    LeaseUpdate,
    LeaseShortenRequest,
    LeaseLockRequest,
    EndpointRegister,
    EndpointOut,
    DashboardResponse,
    DashboardModel,
    EndpointStats,
    LogResponse,
    NodeLaneOut,
    GpuClassOut,
    ForeignJobOut,
    LeasePreviewRequest,
    LeasePreviewResponse,
)
from .auth import current_user
from .authz import (
    CANCEL,
    EDIT,
    EXTEND,
    LOCK,
    UNLOCK,
    User,
    can,
    can_create,
    describe_denial,
)
from . import colocation
from .catalog import get_catalog, resolve_variant
from .cluster import SCHEDULING_MANAGED, SCHEDULING_SLURM, UNLIMITED, get_cluster
from . import inventory
from . import scheduling
from .backends import ClusterUnavailableError, JobSpec, get_estimate_backend
from .leader import booking_lock
from .lifecycle_logger import log_state_transition
from .router_core import fetch_vllm_metrics
from . import slurm
from .dependencies import SessionLocal, init_db
from .utils import ensure_utc

router = APIRouter(
    prefix="/admin", tags=["admin"], dependencies=[Depends(require_auth)]
)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ---------- helpers -------------------------------------------------------------


def _time_limit_from_duration(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _lease_begin(l: Lease) -> datetime:
    return (
        ensure_utc(l.begin_at) if l.begin_at is not None else ensure_utc(l.created_at)
    )


def _lease_end(l: Lease) -> datetime:
    if l.end_at is not None:
        return ensure_utc(l.end_at)
    return _lease_begin(l) + timedelta(hours=1)


def _lease_to_out(
    l: Lease,
    lane_start: Optional[int] = None,
    lane_count: Optional[int] = None,
    conflict: bool = False,
    node: Optional[str] = None,
    gpu_start: Optional[int] = None,
    user: Optional[User] = None,
) -> LeaseOut:
    # Evaluate permissions server-side so the UI never has to reimplement the
    # rule — it just hides what `can_*` says it cannot do.
    return LeaseOut(
        id=l.id,
        model=l.model,
        owner=l.owner,
        owner_sub=l.owner_sub,
        owner_group=l.owner_group,
        pool=l.pool,
        locked=bool(l.locked),
        locked_by=l.locked_by,
        locked_reason=l.locked_reason,
        can_edit=bool(user and can(user, EDIT, l)),
        can_cancel=bool(user and can(user, CANCEL, l)),
        can_lock=bool(user and can(user, LOCK, l)),
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
        node=node or l.node,
        pinned_node=l.pinned_node,
        gpu_start=gpu_start,
        gpu_class=l.gpu_class,
        scheduling=_scheduling_mode(l.pool),
        estimated_start=l.estimated_start,
        estimate_updated_at=l.estimate_updated_at,
        mode=l.mode or "session",
        replicas=l.replicas or 1,
        supersedes_id=l.supersedes_id,
        colocated=[t.model for t in colocation.decode(l.colocated_json)],
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

    env.update(
        {
            "VLLM_HEALTH_TIMEOUT_SECONDS": str(settings.vllm_health_timeout_seconds),
            "VLLM_MAX_RETRIES": str(settings.vllm_max_retries),
            "VLLM_RETRY_DELAY_SECONDS": str(settings.vllm_retry_delay_seconds),
        }
    )

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


def _scheduling_mode(pool_name: str | None) -> str:
    """Whether a booking's start time is a promise or an estimate.

    Pools we allocate ourselves are `managed`. Anything on a partition Slurm
    schedules — including the legacy no-cluster.yaml case, where we never told
    Slurm anything — is only ever an estimate.
    """
    pool = get_cluster().pool(pool_name)
    if pool is None:
        return SCHEDULING_MANAGED if not get_cluster().pools else SCHEDULING_SLURM
    return SCHEDULING_MANAGED if pool.is_managed else SCHEDULING_SLURM


def _mail_user_for(owner_email: str | None) -> str | None:
    """Prefer the booker's directory address over the cluster-wide fallback.

    Note that job_submit.lua hard-rejects placeholder addresses, so a bad
    SLURM_MAIL_USER fails the job at submit rather than merely losing mail.
    """
    return (owner_email or "").strip() or settings.slurm_mail_user


def _gres_for(gpu_class: str | None, gpus: int) -> str | None:
    """Typed GRES pinning the job to a GPU class.

    Returning None (untyped `--gres=gpu:N`) is **not** a neutral default on
    this cluster: job_submit.lua appends `gpu24` as a hard constraint to any
    untyped request, so the job silently never reaches a larger card. We only
    fall back to untyped when no cluster.yaml is configured at all.
    """
    if not gpu_class:
        return None
    return f"gpu:{gpu_class}:{max(1, gpus)}"


def _submit_to_slurm(lease: Lease) -> str:
    return _submit_to_slurm_from_snapshot(_snapshot_lease(lease))


def _submit_to_slurm_from_snapshot(snapshot: dict) -> str:
    """
    Submit a Slurm job using a plain-dict snapshot instead of an ORM object.
    """
    begin = (
        ensure_utc(snapshot["begin_at"])
        if snapshot["begin_at"]
        else ensure_utc(snapshot["created_at"])
    )
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

    # A co-located job is a GPU host: the template loops over these instead of
    # launching the single MODEL_PATH above.
    tenants = colocation.decode(snapshot.get("colocated_json"))
    if tenants:
        env.update(colocation.job_env(tenants))

    cluster = get_cluster()
    gpu_class = snapshot.get("gpu_class")
    pool = cluster.pool(snapshot.get("pool"))

    # Runtime is normally implied by the GPU class, so an aarch64 node gets an
    # aarch64 image without the model entry naming one.
    runtime = cluster.runtime_for(gpu_class, snapshot.get("runtime"))
    if runtime is not None:
        env.update(runtime.as_job_env())

    # On a managed pool our calendar is the allocator, so pin the job to the
    # node the planner chose — that is what makes the booking truthful rather
    # than decorative. On a `slurm` pool, Slurm decides and we must not pin.
    nodelist = settings.slurm_nodelist
    if pool is not None and pool.is_managed and snapshot.get("node"):
        nodelist = snapshot["node"]
    # An explicit user pin overrides both the global default and our planner:
    # they asked for that machine, so send them there even on a `slurm` pool
    # where we would otherwise let backfill choose.
    if snapshot.get("pinned_node"):
        nodelist = snapshot["pinned_node"]

    # All jobs run as one service account, so the requester survives only here.
    comment_bits = []
    if snapshot.get("owner_sub"):
        comment_bits.append(f"user:{snapshot['owner_sub']}")
    if snapshot.get("id"):
        comment_bits.append(f"lease:{snapshot['id']}")

    res = slurm.submit_vllm_job(
        template_path=settings.sbatch_template_path,
        job_name=f"vllm-{snapshot['model']}",
        gpus=snapshot["requested_gpus"],
        gres=_gres_for(gpu_class, snapshot["requested_gpus"]),
        time_limit=time_limit,
        begin=None,
        env=env,
        partition=(pool.partition if pool else settings.slurm_partition),
        account=(pool.account if pool and pool.account else settings.slurm_account),
        qos=(pool.qos if pool and pool.qos else settings.slurm_qos),
        nodelist=nodelist,
        reservation=(pool.reservation if pool else None),
        comment=",".join(comment_bits) or None,
        cpus_per_task=cpus,
        mem=mem,
        log_dir=settings.job_log_dir,
        mail_user=_mail_user_for(snapshot.get("owner_email")),
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
        "owner_email": lease.owner_email,
        "owner_sub": lease.owner_sub,
        "pool": lease.pool,
        "gpu_class": lease.gpu_class,
        "node": lease.node,
        "pinned_node": lease.pinned_node,
        "runtime": lease.runtime,
        "colocated_json": lease.colocated_json,
    }


def _enforce_quota(
    db: Session, user: User, *, gpus: int, begin: datetime, end: datetime,
    pool: str | None, exclude_lease_id: int | None = None, mode: str = "session",
) -> None:
    """Refuse a booking that would exceed the caller's quota.

    Evaluated against bookings that are still in flight, so a user who has
    finished for the day gets their allowance back. Admins are exempt.
    """
    quota = get_cluster().quota_for(user.groups, pool=pool, is_admin=user.is_admin)
    if quota == UNLIMITED:
        return

    now = now_utc()
    begin, end = ensure_utc(begin), ensure_utc(end)
    duration_hours = (end - begin).total_seconds() / 3600.0

    if quota.max_booking_duration_hours is not None and mode != "service":
        if duration_hours > quota.max_booking_duration_hours:
            raise HTTPException(status_code=409, detail=(
                f"Booking is {duration_hours:.1f}h but your limit is "
                f"{quota.max_booking_duration_hours:.0f}h. Shorten it, or ask an admin."
            ))

    if quota.max_booking_horizon_days is not None:
        horizon = now + timedelta(days=quota.max_booking_horizon_days)
        if begin > horizon:
            raise HTTPException(status_code=409, detail=(
                f"You can only book {quota.max_booking_horizon_days} days ahead."
            ))

    if quota.max_concurrent_gpus is None and quota.max_gpu_hours_inflight is None:
        return

    mine = [
        l for l in db.execute(
            select(Lease).where(
                Lease.owner_sub == user.sub,
                Lease.state.in_(["PLANNED", "SUBMITTED", "STARTING", "RUNNING"]),
            )
        ).scalars().all()
        if l.id != exclude_lease_id and _lease_end(l) > now
    ]

    if quota.max_concurrent_gpus is not None:
        # Peak simultaneous GPUs, not a naive sum: sequential bookings should
        # not count against each other.
        peak = _peak_concurrent_gpus(mine, extra=(begin, end, gpus))
        if peak > quota.max_concurrent_gpus:
            raise HTTPException(status_code=409, detail=(
                f"This would put {peak} GPUs in flight at once; your limit is "
                f"{quota.max_concurrent_gpus}. Cancel or shorten another booking."
            ))

    if quota.max_gpu_hours_inflight is not None:
        used = sum(
            l.requested_gpus * (_lease_end(l) - max(now, _lease_begin(l))).total_seconds() / 3600.0
            for l in mine
        )
        requested = gpus * duration_hours
        if used + requested > quota.max_gpu_hours_inflight:
            raise HTTPException(status_code=409, detail=(
                f"This would use {used + requested:.0f} GPU-hours; your limit is "
                f"{quota.max_gpu_hours_inflight:.0f}. You currently have {used:.0f} booked."
            ))


def _peak_concurrent_gpus(leases: list[Lease], extra: tuple) -> int:
    """Max GPUs held simultaneously across a set of bookings, via a sweep."""
    begin, end, gpus = extra
    events: list[tuple[datetime, int]] = [(begin, gpus), (end, -gpus)]
    for l in leases:
        events.append((_lease_begin(l), l.requested_gpus))
        events.append((_lease_end(l), -l.requested_gpus))
    # End events sort before start events at the same instant, so a booking
    # that ends exactly when another starts does not double-count.
    events.sort(key=lambda e: (e[0], e[1]))

    peak = running = 0
    for _, delta in events:
        running += delta
        peak = max(peak, running)
    return peak


def _authorize(user: User, action: str, lease: Lease) -> None:
    """Raise 403 with a message that says what to do about it."""
    if not can(user, action, lease):
        raise HTTPException(status_code=403, detail=describe_denial(user, action, lease))


def _validate_no_conflicts(db: Session, candidate: Lease) -> None:
    """Refuse a booking that cannot be placed on any node.

    Only enforced for pools we allocate ourselves. On a `slurm` pool Slurm's
    backfill scheduler decides, and our grid is advisory — refusing there would
    reject bookings the cluster could actually run.
    """
    now = now_utc()
    cluster = get_cluster()

    pool = cluster.pool(candidate.pool)
    if pool is not None and not pool.is_managed:
        return

    leases = scheduling.active_leases(
        db.execute(select(Lease)).scalars().all(), now
    )
    leases = [x for x in leases if x.id != candidate.id] + [candidate]

    placements, lanes = scheduling.plan(leases, inventory.current(), cluster)
    p = placements.get(candidate.id)
    if p and p.conflict:
        raise HTTPException(
            status_code=409,
            detail=scheduling.describe_conflict(candidate, leases, placements, lanes),
        )


def _merge_same_model_if_applicable(
    db: Session, req: LeaseCreate, begin: datetime, end: datetime
) -> Optional[Lease]:
    existing = (
        db.execute(
            select(Lease)
            .where(
                Lease.model == req.model,
                Lease.state.in_(["PLANNED", "SUBMITTED", "STARTING", "RUNNING"]),
            )
            .order_by(Lease.id.desc())
        )
        .scalars()
        .first()
    )
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
    """Find stdout/stderr log files for a Slurm job.

    The compute node wrote these to `JOB_LOG_DIR`; we read them through
    `JOB_LOG_DIR_LOCAL`, which differs whenever that shared directory is
    mounted here under another path. With no shared mount at all the glob
    simply finds nothing and log viewing is empty — everything else works.
    """
    # Validate that slurm_job_id is purely numeric to prevent path traversal
    if not slurm_job_id.isdigit():
        return "", ""

    log_dir = os.path.abspath(settings.job_log_dir_local)

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
async def dashboard(user: User = Depends(current_user)):
    now = now_utc()
    cluster = get_cluster()
    inv = inventory.current()

    with SessionLocal() as db:
        ready = set(
            db.execute(select(Endpoint.model).where(Endpoint.state == "READY"))
            .scalars()
            .all()
        )

        leases = db.execute(select(Lease).order_by(Lease.id.desc())).scalars().all()

        placements, lanes = scheduling.plan(
            scheduling.active_leases(leases, now), inv, cluster
        )

        out_leases: list[LeaseOut] = []
        for l in leases:
            p = placements.get(l.id)
            out_leases.append(
                _lease_to_out(
                    l,
                    # lane_* stay the flat global index the timeline already
                    # draws; `node`/`gpu_start` let it group rows per node.
                    lane_start=scheduling.lane_index(lanes, p) if p else None,
                    lane_count=p.gpu_count if p else None,
                    conflict=p.conflict if p else False,
                    node=p.node if p else None,
                    gpu_start=p.gpu_start if p else None,
                    user=user,
                )
            )

        catalog = get_catalog()
        models: list[DashboardModel] = []
        for name, m in catalog.items():
            models.append(
                DashboardModel(
                    id=name,
                    ready=(name in ready),
                    meta={
                        "gpus": m.gpus,
                        "tensor_parallel_size": m.tensor_parallel_size,
                        "notes": m.notes,
                        "tags": m.tags or [],
                        # Seeds the booking dialog's memory slider.
                        "memory_gb": m.memory_gb,
                        "gpu_memory_utilization": m.gpu_memory_utilization,
                        # Which GPU classes this model could actually land on.
                        # The dialog offers only these, and bounds the memory
                        # cap by whichever the user picks — bounding by the
                        # smallest class is only honest while it is unknown.
                        "gpu_classes": inventory.eligible_gpu_classes(
                            m, cluster, inv
                        ),
                        "min_vram_gb": m.min_vram_gb or None,
                    },
                )
            )

        # Endpoint stats (DB portion)
        eps = (
            db.execute(
                select(Endpoint).where(Endpoint.state.in_(["READY", "STARTING"]))
            )
            .scalars()
            .all()
        )
        stats_data: list[dict] = []
        for e in eps:
            uptime = None
            if e.created_at:
                uptime = (now - ensure_utc(e.created_at)).total_seconds()
            stats_data.append(
                dict(
                    model=e.model,
                    host=e.host,
                    port=e.port,
                    state=e.state,
                    slurm_job_id=e.slurm_job_id,
                    last_health_at=e.last_health_at,
                    uptime_seconds=uptime,
                    vllm_version=e.vllm_version,
                )
            )

    # Fetch vLLM /metrics for all endpoints in parallel (no DB session held)
    async def _enrich(s: dict) -> None:
        vm = await fetch_vllm_metrics(s["host"], s["port"])
        s["gpu_cache_usage"] = vm["gpu_cache_usage"]
        s["active_requests"] = vm["active_requests"]
        s["pending_requests"] = vm["pending_requests"]
        s["throughput_tps"] = vm["throughput_tps"]
        s["ttft_avg"] = vm["ttft_avg"]

    if stats_data:
        await asyncio.gather(*[_enrich(s) for s in stats_data])

    stats = [EndpointStats(**s) for s in stats_data]

    return DashboardResponse(
        now=now,
        # Sum over what was actually discovered; falls back to the synthetic
        # node built from TOTAL_GPUS when there is no inventory.
        total_gpus=sum(lane.gpu_count for lane in lanes),
        models=models,
        leases=out_leases,
        endpoint_stats=stats,
        gpu_classes=[
            GpuClassOut(
                name=c.name, vram_gb=c.vram_gb,
                usable_gb=round(c.usable_gb, 1),
                unified_memory=c.unified_memory,
            )
            for c in sorted(cluster.gpu_classes.values(), key=lambda x: x.vram_gb)
        ],
        nodes=[
            NodeLaneOut(
                name=lane.name,
                lane_offset=lane.lane_offset,
                gpu_count=lane.gpu_count,
                gpu_classes=[[cls, n] for cls, n in lane.gpu_classes],
                state=lane.state,
                pool=lane.pool,
                synthetic=lane.synthetic,
            )
            for lane in lanes
        ],
        foreign_jobs=[
            ForeignJobOut(
                job_id=j.job_id, user=j.user, state=j.state,
                nodes=list(j.nodes), gpus=j.gpus,
                gpu_indices=list(j.gpu_indices),
                begin_at=j.start_time, end_at=j.end_time,
            )
            for j in inv.foreign_jobs
        ],
        inventory_error=inv.error,
    )


@router.get("/metrics/summary")
def metrics_summary():
    """Return proxy metrics as structured JSON (no auth required for internal route)."""
    return get_metrics_summary()


@router.get("/leases", response_model=list[LeaseOut])
def list_leases(user: User = Depends(current_user)):
    with SessionLocal() as db:
        leases = db.execute(select(Lease).order_by(Lease.id.desc())).scalars().all()
        return [_lease_to_out(l, user=user) for l in leases]


@router.get("/endpoints", response_model=list[EndpointOut])
def list_endpoints():
    with SessionLocal() as db:
        eps = db.execute(select(Endpoint).order_by(Endpoint.id.desc())).scalars().all()
        return [
            EndpointOut(
                id=e.id,
                model=e.model,
                host=e.host,
                port=e.port,
                slurm_job_id=e.slurm_job_id,
                state=e.state,
                last_health_at=e.last_health_at,
                last_error=e.last_error,
                created_at=e.created_at,
                vllm_version=e.vllm_version,
            )
            for e in eps
        ]


@router.post("/leases/preview", response_model=LeasePreviewResponse)
async def preview_lease(req: LeasePreviewRequest, user: User = Depends(current_user)):
    """When would this booking start, without creating it?

    The answer differs by pool, and saying so honestly is the point:

    * `managed` — our calendar allocates, so we can **confirm** a slot.
    * `slurm`   — Slurm's backfill allocates. We ask `sbatch --test-only`,
                  which validates and reports a start time *without queueing*.
    * neither   — when no backend can run `--test-only` (e.g. the router runs
                  off-cluster with only slurmrestd), we report **unknown**
                  rather than inventing a time.
    """
    catalog = get_catalog()
    if req.model not in catalog:
        raise HTTPException(status_code=404, detail=f"Unknown model '{req.model}'")

    cluster = get_cluster()
    inv = inventory.current()
    pool = cluster.pool(req.pool)
    if req.pool and pool is None:
        raise HTTPException(status_code=404, detail=f"Unknown pool '{req.pool}'")

    cat = catalog[req.model]
    gpu_class = None
    if cluster.gpu_classes and not inv.is_empty:
        gpu_class = inventory.choose_gpu_class(
            cat, cluster, inv, pool=pool, preferred=req.gpu_class
        )
        if gpu_class is None:
            return LeasePreviewResponse(
                model=req.model, pool=req.pool,
                scheduling=_scheduling_mode(req.pool), confidence="impossible",
                detail=f"No GPU class available for '{req.model}'"
                       + (f" in pool '{pool.name}'." if pool else "."),
            )

    cat = resolve_variant(cat, gpu_class)
    duration = timedelta(seconds=req.duration_seconds)
    begin = ensure_utc(req.begin_at) if req.begin_at else now_utc()

    base = dict(model=req.model, pool=req.pool, gpu_class=gpu_class, gpus=cat.gpus,
                scheduling=_scheduling_mode(req.pool))

    # ── Managed pool: our own planner answers, and it is a promise ──────────
    if _scheduling_mode(req.pool) == SCHEDULING_MANAGED:
        with SessionLocal() as db:
            others = scheduling.active_leases(
                db.execute(select(Lease)).scalars().all(), now_utc()
            )
        probe = Lease(
            id=-1, model=req.model, requested_gpus=cat.gpus, requested_tp=cat.tensor_parallel_size,
            requested_port=0, model_path=cat.model_path, state="PLANNED",
            created_at=now_utc(), begin_at=begin, end_at=begin + duration,
            gpu_class=gpu_class, pool=pool.name if pool else None,
        )
        earliest = scheduling.earliest_start(
            probe, others, inv, cluster, pool=pool, search_end=begin + timedelta(days=14),
        )
        if earliest is None:
            return LeasePreviewResponse(
                **base, confidence="impossible",
                detail="No slot available in the next 14 days.",
            )
        placements, _ = scheduling.plan([*others, probe], inv, cluster)
        placed = placements.get(-1)
        return LeasePreviewResponse(
            **base, confidence="confirmed",
            start_at=earliest, end_at=earliest + duration,
            node=placed.node if placed and not placed.conflict else None,
            detail="Slot is reserved for you on this pool.",
        )

    # ── Slurm pool: only Slurm can answer, and only as an estimate ──────────
    backend = get_estimate_backend()
    if backend is None:
        return LeasePreviewResponse(
            **base, confidence="unknown", start_at=None, end_at=None,
            detail="Slurm decides when this runs. No start estimate is available "
                   "from here (needs a backend that can run `sbatch --test-only`).",
        )

    spec = _preview_job_spec(cat, gpu_class, pool, begin, duration, user)
    try:
        estimate = await backend.estimate_start(spec)
    except (NotImplementedError, ClusterUnavailableError) as exc:
        return LeasePreviewResponse(
            **base, confidence="unknown",
            detail=f"Could not reach Slurm for an estimate: {exc}",
        )

    if estimate.start_time is None:
        return LeasePreviewResponse(
            **base, confidence="unknown",
            detail="Slurm accepted the request but gave no start time yet — "
                   "the backfill scheduler has not considered it. "
                   + (estimate.raw.strip()[:200] if estimate.raw else ""),
        )

    return LeasePreviewResponse(
        **base, confidence="estimated",
        start_at=estimate.start_time, end_at=estimate.start_time + duration,
        node=estimate.nodes[0] if estimate.nodes else None,
        detail="Estimated by Slurm's backfill scheduler. This will move as the "
               "queue changes — it is not a reservation.",
    )


def _preview_job_spec(cat, gpu_class, pool, begin, duration, user) -> JobSpec:
    """A JobSpec matching what we would really submit.

    It must match, or the estimate describes a different job than the one the
    user would get.
    """
    seconds = max(60, int(duration.total_seconds()))
    return JobSpec(
        job_name=f"vllm-{cat.name}",
        script_path=settings.sbatch_template_path,
        gpus=cat.gpus,
        gres=_gres_for(gpu_class, cat.gpus),
        time_limit=_time_limit_from_duration(seconds),
        env={"MODEL_PATH": cat.model_path, "SERVED_MODEL_NAME": cat.name},
        cpus=cat.cpus or settings.slurm_cpus_per_task,
        mem=cat.mem,
        partition=(pool.partition if pool else settings.slurm_partition),
        account=(pool.account if pool and pool.account else settings.slurm_account),
        qos=(pool.qos if pool and pool.qos else settings.slurm_qos),
        reservation=(pool.reservation if pool else None),
        begin=begin,
        comment=f"user:{user.sub},preview",
        log_dir=settings.job_log_dir,
    )


@router.post("/leases", response_model=LeaseOut)
async def create_lease(req: LeaseCreate, user: User = Depends(current_user)):
    if not can_create(user):
        raise HTTPException(
            status_code=403,
            detail="You are not permitted to create bookings. Ask an admin for access.",
        )

    # Assigning group ownership to a group you are not in would let anyone
    # hand edit rights to an arbitrary team.
    if req.owner_group and not (user.is_admin or req.owner_group in user.groups):
        raise HTTPException(
            status_code=403,
            detail=f"You are not a member of group '{req.owner_group}'.",
        )

    catalog = get_catalog()
    if req.model not in catalog:
        raise HTTPException(status_code=404, detail=f"Unknown model '{req.model}'")

    cat = catalog[req.model]
    cluster = get_cluster()
    inv = inventory.current()

    # Which pool, and therefore which nodes, partition and scheduling mode.
    pool = cluster.pool(req.pool)
    if req.pool and pool is None:
        raise HTTPException(status_code=404, detail=f"Unknown pool '{req.pool}'")

    # Pick the GPU class before anything else: it decides the variant, and
    # submitting untyped would let job_submit.lua pin us to gpu24.
    gpu_class = None
    if cluster.gpu_classes and not inv.is_empty:
        gpu_class = inventory.choose_gpu_class(
            cat, cluster, inv, pool=pool, preferred=req.gpu_class
        )
        if gpu_class is None:
            detail = (
                f"No GPU class available for '{req.model}'"
                + (f" in pool '{pool.name}'" if pool else "")
                + "."
            )
            if req.gpu_class:
                detail = (
                    f"'{req.model}' cannot run on {req.gpu_class} "
                    "(insufficient VRAM, or no node has enough of that class)."
                )
            raise HTTPException(status_code=409, detail=detail)

    # ── Explicit node pin ──────────────────────────────────────────────────
    if req.node:
        node = next((n for n in inv.usable_nodes() if n.name == req.node), None)
        if node is None:
            known = ", ".join(sorted(n.name for n in inv.usable_nodes())) or "none"
            raise HTTPException(
                status_code=404,
                detail=f"Unknown or unavailable node '{req.node}'. Available: {known}",
            )
        if gpu_class and node.count_of(gpu_class) < (req.gpus or cat.gpus):
            raise HTTPException(status_code=409, detail=(
                f"Node '{req.node}' has {node.count_of(gpu_class)}x {gpu_class}, "
                f"but this booking needs {req.gpus or cat.gpus}."
            ))
        if pool is not None and not pool.accepts_node(req.node):
            raise HTTPException(status_code=409, detail=(
                f"Node '{req.node}' is not part of pool '{pool.name}'."
            ))

    # ── Co-location ────────────────────────────────────────────────────────
    # Several models inside one allocation. Validated before submission: a
    # half-started group is worse than a refused booking, because the models
    # that did start look healthy.
    tenants: list = []
    if req.colocate:
        unknown = [m for m in req.colocate if m not in catalog]
        if unknown:
            raise HTTPException(
                status_code=404, detail=f"Unknown model(s): {', '.join(unknown)}"
            )
        try:
            tenants = colocation.resolve_group(
                [catalog[req.model]] + [catalog[m] for m in req.colocate],
                cluster.gpu_class(gpu_class),
            )
        except colocation.ColocationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    # gpus/tp/utilization are a function of where the model lands, not fixed
    # properties of the model.
    cat = resolve_variant(cat, gpu_class)
    gpus = req.gpus or cat.gpus

    # A per-lease `gpus` override can undercut the model's aggregate memory
    # requirement, so re-check after the override rather than only at class
    # selection.
    shortfall = catalog[req.model].requires.shortfall(
        cluster.gpu_class(gpu_class), gpus
    )
    if gpu_class and shortfall:
        raise HTTPException(
            status_code=409,
            detail=f"'{req.model}' {shortfall}.",
        )

    tp = req.tensor_parallel_size or cat.tensor_parallel_size

    # Memory resolution, most specific first. Absolute GB is preferred at every
    # layer because a fraction means different things per GPU class.
    cls = cluster.gpu_class(gpu_class)
    util = None
    if req.memory_gb is not None and cls is not None:
        util = cls.utilization_for_gb(req.memory_gb)
    elif req.gpu_memory_utilization is not None:
        util = req.gpu_memory_utilization
    else:
        # Running alone: take the card. `cat.memory_gb` is deliberately NOT
        # consulted here — it is the budget for *sharing* a GPU, and applying
        # it solo would leave most of the card unused and starve the KV cache.
        util = cat.gpu_memory_utilization
    # Hard per-class ceiling — load-bearing on unified-memory parts, where the
    # fraction is of memory shared with the OS.
    util = cluster.cap_utilization(gpu_class, util)

    duration = timedelta(seconds=req.duration_seconds)

    # --- ASAP logic ---
    if req.asap:
        with SessionLocal() as db:
            active = (
                db.execute(
                    select(Lease).where(
                        Lease.state.in_(["PLANNED", "SUBMITTED", "STARTING", "RUNNING"])
                    )
                )
                .scalars()
                .all()
            )

            now = now_utc()
            horizon_end = now + timedelta(hours=48)

            # A throwaway row so the search sees the same shape that will be
            # booked — including the GPU class, which decides candidate nodes.
            probe = Lease(
                id=-1, model=req.model, requested_gpus=gpus, requested_tp=tp,
                requested_port=0, model_path=cat.model_path, state="PLANNED",
                created_at=now, begin_at=now, end_at=now + duration,
                gpu_class=gpu_class, pool=pool.name if pool else None,
            )
            earliest = scheduling.earliest_start(
                probe, active, inv, cluster, pool=pool, search_end=horizon_end,
            )
            if earliest is None:
                where = f" in pool '{pool.name}'" if pool else ""
                what = f"{gpus} × {gpu_class}" if gpu_class else f"{gpus} GPUs"
                raise HTTPException(
                    status_code=409,
                    detail=f"No slot for {what}{where} in the next 48h.",
                )
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

            if merged.slurm_job_id and merged.state in (
                "SUBMITTED",
                "STARTING",
                "RUNNING",
            ):
                total_seconds = int(
                    (_lease_end(merged) - _lease_begin(merged)).total_seconds()
                )
                total_seconds = max(60, total_seconds)
                new_time_limit = _time_limit_from_duration(total_seconds)
                try:
                    await asyncio.to_thread(
                        slurm.extend_time, merged.slurm_job_id, new_time_limit
                    )
                    print(
                        f"create_lease: extended Slurm job {merged.slurm_job_id} "
                        f"time to {new_time_limit} after merge"
                    )
                except Exception as e:
                    print(
                        f"Warning: failed to extend Slurm time for merged lease {merged.id}: {e}"
                    )

            db.add(merged)
            db.commit()
            db.refresh(merged)
            return _lease_to_out(merged, user=user)

        begin = ensure_utc(begin)
        end = ensure_utc(end)
        planned = begin > now_utc() + timedelta(seconds=30)

        lease = Lease(
            model=req.model,
            requested_gpus=gpus,
            requested_tp=tp,
            requested_cpus=cat.cpus,
            requested_mem=cat.mem,
            requested_port=0,
            # owner is display text; owner_sub is the stable handle authz uses.
            # The caller may not claim a different identity.
            owner=req.owner or user.display_name,
            owner_sub=user.sub,
            owner_group=req.owner_group,
            owner_email=user.email or None,
            notes=req.notes,
            begin_at=begin if planned else None,
            end_at=end,
            created_at=now_utc(),
            model_path=cat.model_path,
            tool_args=req.tool_args if req.tool_args is not None else cat.tool_args,
            extra_args=req.extra_args if req.extra_args is not None else cat.extra_args,
            reasoning_parser=req.reasoning_parser
            if req.reasoning_parser is not None
            else cat.reasoning_parser,
            gpu_memory_utilization=str(util),
            venv_activate=cat.venv_activate,
            env_json=json.dumps(cat.env) if cat.env else None,
            pool=pool.name if pool else None,
            gpu_class=gpu_class,
            pinned_node=req.node,
            runtime=cat.runtime,
            mode=req.mode,
            replicas=req.replicas,
            colocated_json=colocation.encode(tenants) if tenants else None,
            state="PLANNED" if planned else "SUBMITTED",
        )

        # Validation reads the whole schedule and then inserts; without this
        # two concurrent bookings can both pass and double-book the same GPUs.
        with booking_lock(db):
            _enforce_quota(
                db, user, gpus=gpus, begin=begin, end=end,
                pool=pool.name if pool else None, mode=req.mode,
            )
            _validate_no_conflicts(db, lease)
            db.add(lease)
        db.commit()
        db.refresh(lease)

        out = _lease_to_out(lease, user=user)
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
            raise HTTPException(
                status_code=500, detail=f"Failed to submit Slurm job: {e}"
            )

        with SessionLocal() as db:
            lease = db.get(Lease, lease_id)
            if lease:
                lease.slurm_job_id = job_id
                db.commit()
                db.refresh(lease)
                out = _lease_to_out(lease, user=user)

    return out


@router.patch("/leases/{lease_id}", response_model=LeaseOut)
def update_lease(lease_id: int, req: LeaseUpdate, user: User = Depends(current_user)):
    with SessionLocal() as db:
        lease = db.get(Lease, lease_id)
        if not lease:
            raise HTTPException(status_code=404, detail="Booking not found")
        _authorize(user, EDIT, lease)

        if lease.state != "PLANNED":
            raise HTTPException(
                status_code=409,
                detail="Only planned bookings can be edited (move/resize).",
            )

        if req.begin_at is not None:
            lease.begin_at = req.begin_at
        if req.end_at is not None:
            lease.end_at = req.end_at
        if req.notes is not None:
            lease.notes = req.notes

        b = _lease_begin(lease)
        e = _lease_end(lease)
        if e <= b:
            raise HTTPException(
                status_code=400, detail="End time must be after start time."
            )
        if lease.requested_gpus > settings.total_gpus:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot request more than {settings.total_gpus} GPUs.",
            )

        _validate_no_conflicts(db, lease)
        db.commit()
        db.refresh(lease)
        return _lease_to_out(lease, user=user)


@router.patch("/leases/{lease_id}/notes")
def update_lease_notes(lease_id: int, req: dict, user: User = Depends(current_user)):
    """Update notes on any active lease (not just PLANNED)."""
    with SessionLocal() as db:
        lease = db.get(Lease, lease_id)
        if not lease:
            raise HTTPException(status_code=404, detail="Booking not found")
        _authorize(user, EDIT, lease)
        if lease.state in ("CANCELED", "ENDED"):
            raise HTTPException(
                status_code=409, detail="Cannot edit notes on a finished booking."
            )
        lease.notes = req.get("notes", "")
        db.commit()
        return {"ok": True, "notes": lease.notes}


@router.post("/leases/{lease_id}/lock", response_model=LeaseOut)
def lock_lease(
    lease_id: int, req: LeaseLockRequest, user: User = Depends(current_user)
):
    """Mark a deployment as production.

    A locked lease refuses owner cancellation and is skipped by the cleanup
    worker — it is how a long-running shared model stays up.
    """
    with SessionLocal() as db:
        lease = db.get(Lease, lease_id)
        if not lease:
            raise HTTPException(status_code=404, detail="Booking not found")
        _authorize(user, LOCK, lease)

        lease.locked = True
        lease.locked_by = user.sub
        lease.locked_reason = req.reason
        if req.permanent:
            # locked + service = permanent: survives its booking window *and*
            # is brought back after a node reboot or cluster outage.
            lease.mode = "service"
            lease.retry_count = 0
        db.commit()
        db.refresh(lease)
        log_state_transition(
            entity="lease", entity_id=lease.id, model=lease.model,
            old_state=lease.state, new_state=lease.state,
            slurm_job_id=lease.slurm_job_id,
            reason=f"locked by {user.sub}: {req.reason or 'no reason given'}",
        )
        return _lease_to_out(lease, user=user)


@router.post("/leases/{lease_id}/unlock", response_model=LeaseOut)
def unlock_lease(lease_id: int, user: User = Depends(current_user)):
    with SessionLocal() as db:
        lease = db.get(Lease, lease_id)
        if not lease:
            raise HTTPException(status_code=404, detail="Booking not found")
        _authorize(user, UNLOCK, lease)

        lease.locked = False
        lease.locked_by = None
        lease.locked_reason = None
        # Stop renewing it; the current window is allowed to run out normally.
        lease.mode = "session"
        db.commit()
        db.refresh(lease)
        log_state_transition(
            entity="lease", entity_id=lease.id, model=lease.model,
            old_state=lease.state, new_state=lease.state,
            slurm_job_id=lease.slurm_job_id,
            reason=f"unlocked by {user.sub}",
        )
        return _lease_to_out(lease, user=user)


@router.delete("/leases/{lease_id}")
async def cancel_lease(lease_id: int, user: User = Depends(current_user)):
    with SessionLocal() as db:
        lease = db.get(Lease, lease_id)
        if not lease:
            raise HTTPException(status_code=404, detail="Booking not found")
        _authorize(user, CANCEL, lease)

        if lease.slurm_job_id:
            try:
                await slurm.async_cancel(lease.slurm_job_id)
            except Exception as e:
                print(f"Warning: Slurm cancel failed: {e}")

        lease.state = "CANCELED"

        if lease.slurm_job_id:
            ep = (
                db.execute(
                    select(Endpoint).where(Endpoint.slurm_job_id == lease.slurm_job_id)
                )
                .scalars()
                .first()
            )
            if ep:
                ep.state = "STOPPED"

        db.commit()
    return {"ok": True}


@router.post("/leases/{lease_id}/extend")
async def extend_lease(
    lease_id: int, req: LeaseExtend, user: User = Depends(current_user)
):
    with SessionLocal() as db:
        lease = db.get(Lease, lease_id)
        if not lease:
            raise HTTPException(status_code=404, detail="Booking not found")
        _authorize(user, EXTEND, lease)

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
async def shorten_lease(
    lease_id: int, req: LeaseShortenRequest, user: User = Depends(current_user)
):
    """Shorten a running or submitted lease. The new end must be in the future and before the current end."""
    with SessionLocal() as db:
        lease = db.get(Lease, lease_id)
        if not lease:
            raise HTTPException(status_code=404, detail="Booking not found")
        _authorize(user, EDIT, lease)

        if lease.state not in ("RUNNING", "SUBMITTED", "STARTING", "PLANNED"):
            raise HTTPException(
                status_code=409, detail="Can only shorten active bookings."
            )

        now = now_utc()
        new_end = ensure_utc(req.new_end_at)
        current_end = _lease_end(lease)
        b = _lease_begin(lease)

        if new_end <= now:
            raise HTTPException(
                status_code=400, detail="New end time must be in the future."
            )
        if new_end <= b:
            raise HTTPException(
                status_code=400, detail="New end time must be after start time."
            )
        if new_end >= current_end:
            raise HTTPException(
                status_code=400,
                detail="New end time must be before current end time. Use extend instead.",
            )

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
async def stop_lease_now(lease_id: int, user: User = Depends(current_user)):
    """Immediately stop a running model — cancels the Slurm job."""
    with SessionLocal() as db:
        lease = db.get(Lease, lease_id)
        if not lease:
            raise HTTPException(status_code=404, detail="Booking not found")
        _authorize(user, CANCEL, lease)

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
            ep = (
                db.execute(
                    select(Endpoint).where(Endpoint.slurm_job_id == lease.slurm_job_id)
                )
                .scalars()
                .first()
            )
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
            raise HTTPException(
                status_code=404, detail="No Slurm job associated with this booking."
            )

        stdout_path, stderr_path = _find_log_files(lease.slurm_job_id)

        stdout_content, stdout_trunc = (
            _read_log_file(stdout_path) if stdout_path else ("", False)
        )
        stderr_content, stderr_trunc = (
            _read_log_file(stderr_path) if stderr_path else ("", False)
        )

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
        existing = (
            db.execute(
                select(Endpoint).where(Endpoint.slurm_job_id == req.slurm_job_id)
            )
            .scalars()
            .first()
        )
        if existing:
            existing.model = req.model
            existing.host = req.host
            existing.port = req.port
            if req.vllm_version:
                existing.vllm_version = req.vllm_version
            if existing.state not in ("READY",):
                existing.state = "STARTING"
            db.commit()
            db.refresh(existing)
            e = existing
        else:
            e = Endpoint(
                model=req.model,
                host=req.host,
                port=req.port,
                slurm_job_id=req.slurm_job_id,
                vllm_version=req.vllm_version,
                state="STARTING",
            )
            db.add(e)
            db.commit()
            db.refresh(e)

        lease = (
            db.execute(select(Lease).where(Lease.slurm_job_id == req.slurm_job_id))
            .scalars()
            .first()
        )
        if lease:
            lease.requested_port = req.port
            if lease.state in ("SUBMITTED", "PLANNED"):
                lease.state = "STARTING"
            db.commit()
            db.refresh(e)

        return EndpointOut(
            id=e.id,
            model=e.model,
            host=e.host,
            port=e.port,
            slurm_job_id=e.slurm_job_id,
            state=e.state,
            last_health_at=e.last_health_at,
            last_error=e.last_error,
            created_at=e.created_at,
            vllm_version=e.vllm_version,
        )
