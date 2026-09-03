"""Leader election for the background workers.

The failure this must prevent is two instances submitting for the same booking.
A brief gap with *no* leader is acceptable — every worker is periodic — but two
leaders at once is not.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.dependencies import SessionLocal, init_db
from app.leader import LEASE_TTL, LeaderElection, Lock, is_leader, set_election
from app.settings import settings


@pytest.fixture(autouse=True)
def _clean():
    init_db()
    yield
    set_election(None)
    with SessionLocal() as db:
        for row in db.query(Lock).all():
            db.delete(row)
        db.commit()


def _election(holder: str, ttl: timedelta = LEASE_TTL) -> LeaderElection:
    return LeaderElection(SessionLocal, holder=holder, ttl=ttl)


def _backdate(name: str, age: timedelta) -> None:
    """Age the heartbeat to simulate an instance that has stopped."""
    with SessionLocal() as db:
        row = db.query(Lock).filter(Lock.name == name).one()
        row.heartbeat_at = datetime.now(timezone.utc) - age
        db.commit()


# ── Acquiring ────────────────────────────────────────────────────────────────

def test_first_instance_becomes_leader():
    a = _election("a")
    assert a.try_acquire()
    assert a.is_leader


def test_second_instance_does_not():
    a, b = _election("a"), _election("b")
    a.try_acquire()

    assert not b.try_acquire()
    assert not b.is_leader
    assert a.is_leader


def test_only_one_leader_among_many():
    elections = [_election(f"n{i}") for i in range(5)]
    results = [e.try_acquire() for e in elections]
    assert sum(results) == 1


def test_leader_renews_its_own_lock():
    a = _election("a")
    a.try_acquire()
    assert a.try_acquire()
    assert a.is_leader


def test_renewal_updates_the_heartbeat():
    a = _election("a")
    a.try_acquire()
    _backdate("background-workers", timedelta(seconds=30))
    a.try_acquire()

    with SessionLocal() as db:
        row = db.query(Lock).filter(Lock.name == "background-workers").one()
        assert datetime.now(timezone.utc) - row.heartbeat_at < timedelta(seconds=5)


# ── Failover ─────────────────────────────────────────────────────────────────

def test_expired_lock_is_taken_over():
    a, b = _election("a"), _election("b")
    a.try_acquire()
    _backdate("background-workers", LEASE_TTL + timedelta(seconds=5))

    assert b.try_acquire()
    assert b.is_leader


def test_a_live_lock_is_not_stolen():
    a, b = _election("a"), _election("b")
    a.try_acquire()
    _backdate("background-workers", LEASE_TTL - timedelta(seconds=5))
    assert not b.try_acquire()


def test_displaced_leader_stands_down_on_its_next_check():
    """Critical: the old leader must stop submitting once it has been replaced."""
    a, b = _election("a"), _election("b")
    a.try_acquire()
    _backdate("background-workers", LEASE_TTL + timedelta(seconds=5))
    b.try_acquire()

    assert not a.try_acquire()
    assert not a.is_leader
    assert b.is_leader


def test_two_instances_racing_an_expired_lock_do_not_both_win():
    a, b, c = _election("a"), _election("b"), _election("c")
    a.try_acquire()
    _backdate("background-workers", LEASE_TTL + timedelta(seconds=5))

    results = [b.try_acquire(), c.try_acquire()]
    assert sum(results) == 1
    assert sum([b.is_leader, c.is_leader]) == 1


def test_release_lets_a_peer_take_over_immediately():
    a, b = _election("a"), _election("b")
    a.try_acquire()
    assert not b.try_acquire()

    a.release()

    assert not a.is_leader
    assert b.try_acquire()


def test_release_by_a_non_leader_is_a_no_op():
    a, b = _election("a"), _election("b")
    a.try_acquire()
    b.release()
    assert a.try_acquire()      # a still holds it


# ── Failure handling ─────────────────────────────────────────────────────────

def test_database_failure_stands_down_rather_than_assuming_leadership():
    """A stale `is_leader=True` during a DB blip would let two instances
    submit jobs."""
    a = _election("a")
    a.try_acquire()
    assert a.is_leader

    from sqlalchemy.exc import SQLAlchemyError

    class Boom:
        def __call__(self):
            raise SQLAlchemyError("connection lost")

    a._sessions = Boom()
    assert not a.try_acquire()
    assert not a.is_leader


# ── HA toggle ────────────────────────────────────────────────────────────────

def test_ha_disabled_means_always_leader(monkeypatch):
    """Single-instance deployments must behave exactly as before."""
    monkeypatch.setattr(settings, "ha_enabled", False)
    assert is_leader()


def test_ha_enabled_defers_to_the_election(monkeypatch):
    monkeypatch.setattr(settings, "ha_enabled", True)
    a = _election("a")
    set_election(a)

    assert not is_leader()
    a.try_acquire()
    assert is_leader()
