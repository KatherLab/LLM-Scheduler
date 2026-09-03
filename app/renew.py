"""Rolling renewal for `mode: service` deployments.

A long-running model cannot simply be booked for a month: clusters cap job
wall time (`MaxWall`), and this one is no exception. So a service deployment is
a *chain* of bounded jobs, handed over without downtime:

    1. the current lease nears its end
    2. submit a replacement covering the next window
    3. wait for the replacement's endpoint to become READY
    4. stop routing new work to the old one (drain)
    5. once its in-flight requests finish, cancel it

Steps 3–5 are what make it zero-downtime, and they depend on the least-loaded
router: for a short period both replicas serve the same model, and the drain
flag is what keeps traffic moving in one direction.

This also gives rolling vLLM upgrades for free — the replacement picks up
whatever the catalog and runtime say *now*.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from .loadbalancer import REGISTRY, LoadRegistry
from .models import Endpoint, Lease

logger = logging.getLogger(__name__)

MODE_SESSION = "session"
MODE_SERVICE = "service"

#: How long before a service lease expires to start the handover. Must exceed
#: the worst-case model load time, or the replacement is not READY in time and
#: the service blinks.
RENEW_LEAD = timedelta(minutes=20)

#: Length of each link in the chain. Kept well under a typical MaxWall.
RENEW_WINDOW = timedelta(hours=12)

#: If in-flight requests have not finished by now, cancel anyway — a stuck
#: stream must not hold a GPU indefinitely.
DRAIN_TIMEOUT = timedelta(minutes=10)

ACTIVE = ("SUBMITTED", "STARTING", "RUNNING")


def is_service(lease: Lease) -> bool:
    return (lease.mode or MODE_SESSION) == MODE_SERVICE


def due_for_renewal(lease: Lease, now: datetime) -> bool:
    """True when a service lease is close enough to its end to hand over."""
    if not is_service(lease) or lease.state not in ACTIVE:
        return False
    if lease.end_at is None:
        return False
    return lease.end_at - now <= RENEW_LEAD


def plan_renewals(leases, now: datetime) -> list[Lease]:
    """Service leases needing a replacement, excluding ones already handled.

    A lease that something already supersedes must not be renewed again — that
    would spawn a replacement per tick.
    """
    superseded = {x.supersedes_id for x in leases if x.supersedes_id}
    return [
        lease for lease in leases
        if due_for_renewal(lease, now) and lease.id not in superseded
    ]


def build_replacement(lease: Lease, now: datetime) -> Lease:
    """A copy of `lease` covering the next window.

    Deliberately re-reads nothing from the catalog here: the caller refreshes
    launch config if it wants a rolling upgrade, so a renewal cannot silently
    change what is running.
    """
    begin = max(now, (lease.end_at or now) - RENEW_LEAD)
    return Lease(
        model=lease.model,
        requested_gpus=lease.requested_gpus,
        requested_cpus=lease.requested_cpus,
        requested_mem=lease.requested_mem,
        requested_tp=lease.requested_tp,
        requested_port=0,
        owner=lease.owner,
        owner_sub=lease.owner_sub,
        owner_group=lease.owner_group,
        owner_email=lease.owner_email,
        pool=lease.pool,
        gpu_class=lease.gpu_class,
        node=lease.node,
        runtime=lease.runtime,
        mode=MODE_SERVICE,
        replicas=lease.replicas,
        supersedes_id=lease.id,
        notes=lease.notes,
        locked=lease.locked,
        locked_by=lease.locked_by,
        locked_reason=lease.locked_reason,
        begin_at=begin,
        end_at=begin + RENEW_WINDOW,
        created_at=now,
        state="PLANNED",
        model_path=lease.model_path,
        tool_args=lease.tool_args,
        extra_args=lease.extra_args,
        reasoning_parser=lease.reasoning_parser,
        gpu_memory_utilization=lease.gpu_memory_utilization,
        venv_activate=lease.venv_activate,
        env_json=lease.env_json,
    )


def endpoints_for(db, lease: Lease) -> list[Endpoint]:
    if not lease.slurm_job_id:
        return []
    return list(db.execute(
        select(Endpoint).where(Endpoint.slurm_job_id == lease.slurm_job_id)
    ).scalars().all())


def replacement_is_ready(db, replacement: Lease) -> bool:
    """Only a READY endpoint counts — a submitted job is not yet serving."""
    return any(e.state == "READY" for e in endpoints_for(db, replacement))


def begin_drain(db, lease: Lease, now: datetime, registry: LoadRegistry | None = None) -> None:
    """Stop routing new work to a lease's endpoints, without killing them."""
    registry = registry or REGISTRY
    for endpoint in endpoints_for(db, lease):
        registry.set_draining(LoadRegistry.key(endpoint.host, endpoint.port), True)
    if lease.draining_since is None:
        lease.draining_since = now
        logger.info(
            "renew: draining lease %d (%s), job %s",
            lease.id, lease.model, lease.slurm_job_id,
        )


def drain_complete(db, lease: Lease, now: datetime, registry: LoadRegistry | None = None) -> bool:
    """True once in-flight work has finished, or the timeout has elapsed.

    The timeout matters: a hung stream would otherwise keep a GPU forever.
    """
    registry = registry or REGISTRY
    if lease.draining_since and now - lease.draining_since >= DRAIN_TIMEOUT:
        logger.warning(
            "renew: drain timeout for lease %d (%s) — retiring with requests still in flight",
            lease.id, lease.model,
        )
        return True
    endpoints = endpoints_for(db, lease)
    if not endpoints:
        return True
    return all(
        registry.in_flight(LoadRegistry.key(e.host, e.port)) == 0 for e in endpoints
    )


def forget_endpoints(db, lease: Lease, registry: LoadRegistry | None = None) -> None:
    """Drop load state for a retired lease so a reused host:port starts clean."""
    registry = registry or REGISTRY
    for endpoint in endpoints_for(db, lease):
        registry.forget(LoadRegistry.key(endpoint.host, endpoint.port))


def missing_replicas(db, lease: Lease) -> int:
    """How many more instances this deployment should have running.

    Counts endpoints rather than leases, because a replica only counts once it
    is actually serving.
    """
    target = max(1, lease.replicas or 1)
    live = sum(
        1 for e in endpoints_for(db, lease) if e.state in ("READY", "STARTING")
    )
    return max(0, target - live)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ── Permanent deployments ────────────────────────────────────────────────────

#: Backoff caps here rather than growing forever: a model that is broken (bad
#: weights, missing venv) must not resubmit every 10s, but a model down because
#: a node rebooted should come back promptly once the node returns.
RETRY_BACKOFF_BASE = 10.0
RETRY_BACKOFF_MAX = 600.0


def is_permanent(lease) -> bool:
    """A deployment that must stay up until an admin says otherwise.

    `locked` alone means "do not reap"; `service` alone means "renew on
    schedule". Permanence is both: it survives its booking window *and* is
    resurrected after a node reboot, a cluster outage, or any crash.
    """
    return bool(getattr(lease, "locked", False)) and is_service(lease)


def retry_backoff_seconds(retry_count: int, base: float = RETRY_BACKOFF_BASE) -> float:
    """Exponential backoff, capped.

    Unlimited retries without backoff would turn one broken permanent model
    into a submission storm against the cluster.
    """
    return min(RETRY_BACKOFF_MAX, base * (2 ** max(0, retry_count)))


def should_retry(lease, now: datetime, *, max_retries: int, retry_delay: float) -> bool:
    """Whether a FAILED lease is eligible to be resubmitted.

    Permanent deployments ignore the retry ceiling and the booking window —
    that is what makes them permanent. Everything else keeps the historical
    fail-fast behaviour.
    """
    if not lease.failed_at:
        return False

    since_fail = (now - ensure_utc_dt(lease.failed_at)).total_seconds()

    if is_permanent(lease):
        return since_fail >= retry_backoff_seconds(lease.retry_count or 0)

    if lease.retry_count >= max_retries:
        return False
    if since_fail < retry_delay:
        return False
    if lease.end_at is None:
        return True
    remaining = (ensure_utc_dt(lease.end_at) - now).total_seconds()
    # Not worth a resubmit that would be reaped almost immediately.
    return remaining >= 120


def ensure_utc_dt(value: datetime) -> datetime:
    from .utils import ensure_utc

    return ensure_utc(value)
