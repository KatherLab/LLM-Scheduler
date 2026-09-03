"""Rolling renewal for `mode: service`.

The property under test is that a service model never goes dark: the
replacement must be READY before the old one stops receiving traffic, and the
old one must finish its in-flight work before it is cancelled.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app import renew
from app.dependencies import SessionLocal, init_db
from app.loadbalancer import LoadRegistry
from app.models import Endpoint, Lease

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db():
    init_db()
    with SessionLocal() as session:
        yield session
        for table in (Endpoint, Lease):
            for row in session.query(table).all():
                session.delete(row)
        session.commit()


@pytest.fixture
def reg():
    return LoadRegistry()


def _lease(db, *, mode="service", state="RUNNING", ends_in=timedelta(hours=6),
           job="1000", supersedes=None, replicas=1):
    lease = Lease(
        model="m", requested_gpus=2, requested_tp=2, requested_port=0,
        model_path="p", state=state, owner_sub="alice", created_at=NOW,
        begin_at=NOW - timedelta(hours=1), end_at=NOW + ends_in,
        mode=mode, replicas=replicas, slurm_job_id=job, supersedes_id=supersedes,
        gpu_class="gpu96", pool="general",
    )
    db.add(lease)
    db.commit()
    return lease


def _endpoint(db, lease, host="gpu01", port=8000, state="READY"):
    ep = Endpoint(model=lease.model, host=host, port=port,
                  slurm_job_id=lease.slurm_job_id, state=state, created_at=NOW)
    db.add(ep)
    db.commit()
    return ep


# ── When to renew ────────────────────────────────────────────────────────────

def test_service_lease_near_expiry_is_due(db):
    lease = _lease(db, ends_in=renew.RENEW_LEAD - timedelta(minutes=1))
    assert renew.due_for_renewal(lease, NOW)


def test_service_lease_far_from_expiry_is_not_due(db):
    assert not renew.due_for_renewal(_lease(db, ends_in=timedelta(hours=10)), NOW)


def test_session_lease_is_never_renewed(db):
    """Benchmarks must stop when booked to stop."""
    lease = _lease(db, mode="session", ends_in=timedelta(minutes=1))
    assert not renew.due_for_renewal(lease, NOW)


def test_finished_lease_is_not_renewed(db):
    lease = _lease(db, state="ENDED", ends_in=timedelta(minutes=1))
    assert not renew.due_for_renewal(lease, NOW)


def test_lease_with_no_end_is_not_renewed(db):
    lease = _lease(db)
    lease.end_at = None
    db.commit()
    assert not renew.due_for_renewal(lease, NOW)


def test_a_lease_already_being_replaced_is_not_renewed_again(db):
    """Otherwise every tick spawns another replacement."""
    old = _lease(db, ends_in=timedelta(minutes=5))
    replacement = _lease(db, job="1001", state="SUBMITTED",
                         ends_in=timedelta(hours=12), supersedes=old.id)
    due = renew.plan_renewals([old, replacement], NOW)
    assert due == []


def test_plan_renewals_picks_only_the_due_ones(db):
    due = _lease(db, ends_in=timedelta(minutes=5), job="1")
    later = _lease(db, ends_in=timedelta(hours=10), job="2")
    session = _lease(db, mode="session", ends_in=timedelta(minutes=5), job="3")
    assert [x.id for x in renew.plan_renewals([due, later, session], NOW)] == [due.id]


# ── The replacement ──────────────────────────────────────────────────────────

def test_replacement_copies_the_launch_configuration(db):
    old = _lease(db, ends_in=timedelta(minutes=5))
    new = renew.build_replacement(old, NOW)

    assert new.model == old.model
    assert new.gpu_class == old.gpu_class
    assert new.requested_gpus == old.requested_gpus
    assert new.model_path == old.model_path
    assert new.mode == "service"


def test_replacement_points_at_what_it_supersedes(db):
    old = _lease(db, ends_in=timedelta(minutes=5))
    assert renew.build_replacement(old, NOW).supersedes_id == old.id


def test_replacement_overlaps_the_outgoing_window(db):
    """It must be READY *before* the old one ends, or the service blinks."""
    old = _lease(db, ends_in=renew.RENEW_LEAD)
    new = renew.build_replacement(old, NOW)
    assert new.begin_at <= old.end_at
    assert new.end_at > old.end_at


def test_replacement_inherits_ownership_and_lock(db):
    old = _lease(db, ends_in=timedelta(minutes=5))
    old.locked, old.locked_by = True, "root"
    db.commit()
    new = renew.build_replacement(old, NOW)
    assert new.owner_sub == "alice"
    assert new.locked and new.locked_by == "root"


# ── Handover ─────────────────────────────────────────────────────────────────

def test_replacement_is_not_ready_before_its_endpoint_registers(db):
    replacement = _lease(db, job="1001", state="SUBMITTED")
    assert not renew.replacement_is_ready(db, replacement)


def test_a_starting_endpoint_does_not_count_as_ready(db):
    """A submitted job is not yet serving traffic."""
    replacement = _lease(db, job="1001", state="STARTING")
    _endpoint(db, replacement, host="gpu02", state="STARTING")
    assert not renew.replacement_is_ready(db, replacement)


def test_ready_endpoint_makes_the_replacement_ready(db):
    replacement = _lease(db, job="1001")
    _endpoint(db, replacement, host="gpu02", state="READY")
    assert renew.replacement_is_ready(db, replacement)


def test_draining_stops_new_traffic_without_killing_the_old_replica(db, reg):
    old = _lease(db)
    ep = _endpoint(db, old)
    renew.begin_drain(db, old, NOW, reg)

    assert reg.is_draining(LoadRegistry.key(ep.host, ep.port))
    assert old.draining_since == NOW


def test_drain_is_not_restarted_on_a_later_tick(db, reg):
    old = _lease(db)
    _endpoint(db, old)
    renew.begin_drain(db, old, NOW, reg)
    renew.begin_drain(db, old, NOW + timedelta(minutes=5), reg)
    assert old.draining_since == NOW      # the timeout must not keep resetting


def test_drain_incomplete_while_requests_are_in_flight(db, reg):
    old = _lease(db)
    ep = _endpoint(db, old)
    reg.acquire(LoadRegistry.key(ep.host, ep.port))
    renew.begin_drain(db, old, NOW, reg)

    assert not renew.drain_complete(db, old, NOW, reg)


def test_drain_completes_once_requests_finish(db, reg):
    old = _lease(db)
    ep = _endpoint(db, old)
    key = LoadRegistry.key(ep.host, ep.port)
    reg.acquire(key)
    renew.begin_drain(db, old, NOW, reg)
    reg.release(key)

    assert renew.drain_complete(db, old, NOW, reg)


def test_drain_times_out_so_a_hung_stream_cannot_hold_a_gpu(db, reg):
    old = _lease(db)
    ep = _endpoint(db, old)
    reg.acquire(LoadRegistry.key(ep.host, ep.port))
    renew.begin_drain(db, old, NOW, reg)

    later = NOW + renew.DRAIN_TIMEOUT + timedelta(seconds=1)
    assert renew.drain_complete(db, old, later, reg)


def test_lease_with_no_endpoints_drains_immediately(db, reg):
    old = _lease(db)
    assert renew.drain_complete(db, old, NOW, reg)


def test_retiring_forgets_load_state(db, reg):
    old = _lease(db)
    ep = _endpoint(db, old)
    key = LoadRegistry.key(ep.host, ep.port)
    reg.acquire(key)
    reg.set_draining(key)

    renew.forget_endpoints(db, old, reg)

    assert reg.in_flight(key) == 0
    assert not reg.is_draining(key)


# ── Replicas ─────────────────────────────────────────────────────────────────

def test_single_replica_deployment_is_satisfied_by_one_endpoint(db):
    lease = _lease(db, replicas=1)
    _endpoint(db, lease)
    assert renew.missing_replicas(db, lease) == 0


def test_missing_replicas_counts_the_shortfall(db):
    lease = _lease(db, replicas=3)
    _endpoint(db, lease, host="gpu01")
    assert renew.missing_replicas(db, lease) == 2


def test_starting_replicas_count_so_we_do_not_over_submit(db):
    lease = _lease(db, replicas=2)
    _endpoint(db, lease, host="gpu01", state="READY")
    _endpoint(db, lease, host="gpu02", state="STARTING")
    assert renew.missing_replicas(db, lease) == 0


def test_failed_replicas_do_not_count(db):
    lease = _lease(db, replicas=2)
    _endpoint(db, lease, host="gpu01", state="READY")
    _endpoint(db, lease, host="gpu02", state="FAILED")
    assert renew.missing_replicas(db, lease) == 1


# ── Permanent deployments ────────────────────────────────────────────────────

def test_permanent_is_locked_plus_service(db):
    lease = _lease(db, mode="service")
    lease.locked = True
    db.commit()
    assert renew.is_permanent(lease)


@pytest.mark.parametrize("mode,locked", [("service", False), ("session", True), ("session", False)])
def test_not_permanent_without_both(db, mode, locked):
    """`locked` alone means "do not reap"; `service` alone means "renew on
    schedule". Only both together survive a node reboot."""
    lease = _lease(db, mode=mode)
    lease.locked = locked
    db.commit()
    assert not renew.is_permanent(lease)


def test_permanent_ignores_the_retry_ceiling(db):
    """A node reboot must not exhaust VLLM_MAX_RETRIES and leave it down."""
    lease = _lease(db, mode="service", state="FAILED")
    lease.locked = True
    lease.retry_count = 99
    lease.failed_at = NOW - timedelta(hours=1)
    db.commit()
    assert renew.should_retry(lease, NOW, max_retries=1, retry_delay=10)


def test_ordinary_lease_still_gives_up(db):
    lease = _lease(db, mode="session", state="FAILED")
    lease.retry_count = 1
    lease.failed_at = NOW - timedelta(hours=1)
    db.commit()
    assert not renew.should_retry(lease, NOW, max_retries=1, retry_delay=10)


def test_permanent_ignores_an_expired_booking_window(db):
    """Its window is meaningless — renewal owns the schedule."""
    lease = _lease(db, mode="service", state="FAILED", ends_in=timedelta(hours=-5))
    lease.locked = True
    lease.failed_at = NOW - timedelta(hours=1)
    db.commit()
    assert renew.should_retry(lease, NOW, max_retries=1, retry_delay=10)


def test_permanent_respects_backoff(db):
    """Unlimited retries without backoff would be a submission storm."""
    lease = _lease(db, mode="service", state="FAILED")
    lease.locked = True
    lease.retry_count = 3
    lease.failed_at = NOW - timedelta(seconds=5)
    db.commit()
    assert not renew.should_retry(lease, NOW, max_retries=1, retry_delay=10)


def test_backoff_grows_then_caps():
    seconds = [renew.retry_backoff_seconds(i) for i in range(12)]
    assert seconds[0] < seconds[1] < seconds[2]
    assert max(seconds) == renew.RETRY_BACKOFF_MAX


def test_a_lease_with_no_failure_time_is_not_retried(db):
    lease = _lease(db, mode="service", state="FAILED")
    lease.locked = True
    db.commit()
    assert not renew.should_retry(lease, NOW, max_retries=1, retry_delay=10)
