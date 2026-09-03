"""Quota enforcement at booking time."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from app.admin import _enforce_quota, _peak_concurrent_gpus
from app.authz import User
from app.cluster import ClusterConfig, Quota, set_cluster
from app.dependencies import SessionLocal
from app.models import Lease

NOW = datetime.now(timezone.utc)

ALICE = User(sub="alice", display_name="Alice", groups=frozenset({"radiology"}),
             is_user=True, via="ldap")
POWER = User(sub="pete", display_name="Pete", groups=frozenset({"llm-power-users"}),
             is_user=True, via="ldap")
ADMIN = User(sub="root", display_name="Root", is_admin=True, is_user=True, via="ldap")

CLUSTER = ClusterConfig(
    default_quota=Quota(
        max_concurrent_gpus=4,
        max_gpu_hours_inflight=96,
        max_booking_horizon_days=14,
        max_booking_duration_hours=48,
    ),
    group_quotas={"llm-power-users": Quota(max_concurrent_gpus=8)},
)


@pytest.fixture(autouse=True)
def _cluster():
    set_cluster(CLUSTER)
    yield
    set_cluster(None)
    with SessionLocal() as db:
        for lease in db.query(Lease).all():
            db.delete(lease)
        db.commit()


@pytest.fixture
def db():
    from app.dependencies import init_db
    init_db()
    with SessionLocal() as session:
        yield session


def _book(db, owner_sub, gpus, start_h, hours, state="PLANNED"):
    lease = Lease(
        model="m", requested_gpus=gpus, requested_tp=1, requested_port=0,
        model_path="p", state=state, owner_sub=owner_sub, created_at=NOW,
        begin_at=NOW + timedelta(hours=start_h),
        end_at=NOW + timedelta(hours=start_h + hours),
    )
    db.add(lease)
    db.commit()
    return lease


def _check(db, user, gpus, start_h=1, hours=2, pool=None):
    _enforce_quota(
        db, user, gpus=gpus,
        begin=NOW + timedelta(hours=start_h),
        end=NOW + timedelta(hours=start_h + hours),
        pool=pool,
    )


# ── Peak concurrency, not naive sums ─────────────────────────────────────────

def test_sequential_bookings_do_not_stack(db):
    """Yesterday's booking should not consume today's allowance."""
    _book(db, "alice", gpus=4, start_h=1, hours=2)
    _check(db, ALICE, gpus=4, start_h=5, hours=2)   # after the first ends


def test_overlapping_bookings_do_stack(db):
    _book(db, "alice", gpus=3, start_h=1, hours=4)
    with pytest.raises(HTTPException) as exc:
        _check(db, ALICE, gpus=3, start_h=2, hours=2)
    assert exc.value.status_code == 409
    assert "GPUs in flight" in exc.value.detail


def test_booking_touching_the_limit_exactly_is_allowed(db):
    _book(db, "alice", gpus=2, start_h=1, hours=4)
    _check(db, ALICE, gpus=2, start_h=2, hours=1)   # 2 + 2 == limit of 4


def test_back_to_back_bookings_do_not_double_count(db):
    """One ending exactly when the next starts is sequential, not concurrent."""
    _book(db, "alice", gpus=4, start_h=1, hours=2)
    _check(db, ALICE, gpus=4, start_h=3, hours=2)


def test_another_users_bookings_are_irrelevant(db):
    _book(db, "bob", gpus=4, start_h=1, hours=4)
    _check(db, ALICE, gpus=4, start_h=1, hours=2)


def test_finished_bookings_release_the_allowance(db):
    _book(db, "alice", gpus=4, start_h=1, hours=2, state="ENDED")
    _check(db, ALICE, gpus=4, start_h=1, hours=2)


# ── Per-axis limits ──────────────────────────────────────────────────────────

def test_duration_limit(db):
    with pytest.raises(HTTPException, match="limit"):
        _check(db, ALICE, gpus=1, hours=72)      # limit is 48h


def test_horizon_limit(db):
    with pytest.raises(HTTPException, match="days ahead"):
        _check(db, ALICE, gpus=1, start_h=24 * 30)   # limit is 14 days


def test_gpu_hours_limit(db):
    """4 GPUs × 40h = 160 GPU-hours, over the 96 limit."""
    with pytest.raises(HTTPException, match="GPU-hours"):
        _check(db, ALICE, gpus=4, hours=40)


# ── Roles ────────────────────────────────────────────────────────────────────

def test_group_override_raises_the_concurrency_limit(db):
    _book(db, "pete", gpus=4, start_h=1, hours=4)
    _check(db, POWER, gpus=4, start_h=1, hours=2)    # 8 total, at their limit


def test_group_override_does_not_relax_other_axes(db):
    """Only max_concurrent_gpus is overridden; duration still applies."""
    with pytest.raises(HTTPException, match="limit"):
        _check(db, POWER, gpus=1, hours=72)


def test_admins_are_exempt(db):
    _book(db, "root", gpus=8, start_h=1, hours=4)
    _check(db, ADMIN, gpus=8, start_h=1, hours=200)


def test_no_quota_configured_means_no_limit(db):
    set_cluster(ClusterConfig())
    _check(db, ALICE, gpus=64, hours=1000)


# ── The sweep itself ─────────────────────────────────────────────────────────

def test_peak_of_a_single_booking():
    assert _peak_concurrent_gpus([], extra=(NOW, NOW + timedelta(hours=1), 3)) == 3


def test_peak_ignores_a_disjoint_booking():
    lease = Lease(model="m", requested_gpus=4, requested_tp=1, requested_port=0,
                  model_path="p", state="PLANNED", created_at=NOW,
                  begin_at=NOW, end_at=NOW + timedelta(hours=1))
    peak = _peak_concurrent_gpus(
        [lease], extra=(NOW + timedelta(hours=2), NOW + timedelta(hours=3), 4)
    )
    assert peak == 4
