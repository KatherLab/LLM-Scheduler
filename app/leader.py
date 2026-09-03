"""Single-writer election for the background workers.

With one app instance serving every team, the proxy becomes a single point of
failure for all inference. The fix is to run several — but the workers must not
all run: two instances submitting for the same booking would double-submit, and
two reconcilers would fight over cancelling jobs.

So: **every replica serves proxy traffic; only the leader runs the workers.**

The lock is a heartbeat row, which works identically on SQLite and Postgres and
needs no extra infrastructure. It is deliberately not a distributed consensus
protocol — the failure mode it must avoid is "two leaders submitting jobs", and
a lease with a generous timeout plus idempotent workers covers that. A brief
gap with *no* leader is fine: the workers are all periodic.
"""

from __future__ import annotations

import logging
import os
import socket
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from sqlalchemy import String, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base
from .models import TZDateTime, utc_now

logger = logging.getLogger(__name__)

#: How long a lock is honoured after its last heartbeat. Must comfortably
#: exceed the renew interval, or a slow tick hands leadership to a peer while
#: the current leader is still mid-cycle.
LEASE_TTL = timedelta(seconds=45)

#: How often the leader re-stamps the row.
HEARTBEAT_INTERVAL = timedelta(seconds=15)

WORKER_LOCK = "background-workers"


class Lock(Base):
    """A single named lock row, held by whoever last heartbeat it."""

    __tablename__ = "locks"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    holder: Mapped[str] = mapped_column(String(128))
    acquired_at: Mapped[datetime] = mapped_column(TZDateTime(), default=utc_now)
    heartbeat_at: Mapped[datetime] = mapped_column(TZDateTime(), default=utc_now)


def instance_id() -> str:
    """Identifies this process in the lock row and in logs."""
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


class LeaderElection:
    """Tracks whether this process currently holds the worker lock."""

    def __init__(self, session_factory, *, name: str = WORKER_LOCK,
                 holder: str | None = None, ttl: timedelta = LEASE_TTL):
        self._sessions = session_factory
        self.name = name
        self.holder = holder or instance_id()
        self.ttl = ttl
        self._is_leader = False

    @property
    def is_leader(self) -> bool:
        return self._is_leader

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def try_acquire(self) -> bool:
        """Take or renew the lock. Returns whether we hold it afterwards.

        The steal is a conditional UPDATE, so two instances racing on an
        expired lock cannot both win: whichever `UPDATE ... WHERE holder = old`
        matches a row first takes it, and the other's update matches nothing.
        """
        now = self._now()
        cutoff = now - self.ttl

        try:
            with self._sessions() as db:
                row = db.execute(
                    select(Lock).where(Lock.name == self.name)
                ).scalars().first()

                if row is None:
                    db.add(Lock(name=self.name, holder=self.holder,
                                acquired_at=now, heartbeat_at=now))
                    db.commit()
                    self._became_leader()
                    return True

                if row.holder == self.holder:
                    # Renew our own lock.
                    row.heartbeat_at = now
                    db.commit()
                    self._is_leader = True
                    return True

                if row.heartbeat_at is not None and row.heartbeat_at > cutoff:
                    # Someone else holds it and is alive.
                    self._lost_leadership(row.holder)
                    return False

                # Expired — steal it, but only if nobody beat us to it.
                result = db.execute(
                    update(Lock)
                    .where(Lock.name == self.name, Lock.holder == row.holder)
                    .values(holder=self.holder, acquired_at=now, heartbeat_at=now)
                )
                db.commit()
                if result.rowcount == 1:
                    self._became_leader(stolen_from=row.holder)
                    return True
                self._lost_leadership(None)
                return False
        except SQLAlchemyError as exc:
            # A database blip must not leave a stale `is_leader=True` that lets
            # two instances submit jobs.
            logger.error("leader election failed, standing down: %s", exc)
            self._is_leader = False
            return False

    def release(self) -> None:
        """Give up the lock on clean shutdown so a peer takes over promptly."""
        if not self._is_leader:
            return
        try:
            with self._sessions() as db:
                db.execute(
                    update(Lock)
                    .where(Lock.name == self.name, Lock.holder == self.holder)
                    .values(heartbeat_at=datetime(1970, 1, 1, tzinfo=timezone.utc))
                )
                db.commit()
        except SQLAlchemyError as exc:
            logger.warning("could not release leader lock: %s", exc)
        finally:
            self._is_leader = False
            logger.info("released leader lock %r", self.name)

    def _became_leader(self, stolen_from: str | None = None) -> None:
        if not self._is_leader:
            if stolen_from:
                logger.info(
                    "acquired leader lock %r from expired holder %s",
                    self.name, stolen_from,
                )
            else:
                logger.info("acquired leader lock %r as %s", self.name, self.holder)
        self._is_leader = True

    def _lost_leadership(self, holder: str | None) -> None:
        if self._is_leader:
            logger.warning(
                "lost leader lock %r to %s — standing down", self.name, holder or "another instance"
            )
        self._is_leader = False


_election: LeaderElection | None = None


def get_election() -> LeaderElection:
    global _election
    if _election is None:
        from .dependencies import SessionLocal

        _election = LeaderElection(SessionLocal)
    return _election


def set_election(election: LeaderElection | None) -> None:
    """Override the election. For tests."""
    global _election
    _election = election


def is_leader() -> bool:
    """Whether this process should be running the background workers.

    Returns True when HA is disabled, so single-instance deployments behave
    exactly as before.
    """
    from .settings import settings

    if not settings.ha_enabled:
        return True
    return get_election().is_leader


# ── Booking serialization ────────────────────────────────────────────────────

BOOKING_LOCK = "booking-writes"


@contextmanager
def booking_lock(db):
    """Serialize the read-validate-insert path for bookings.

    Conflict validation and quota checks read the whole schedule, decide, then
    insert. Without serialization two concurrent bookings can both pass and
    double-book the same GPUs — and with HA every instance serves bookings, so
    an in-process lock is not enough.

    On Postgres this is a row lock (`SELECT ... FOR UPDATE`), which blocks the
    second transaction until the first commits. SQLite has no `FOR UPDATE`, but
    its writes are already serialized, so acquiring the row is enough to force
    the transaction into write mode early.
    """
    is_postgres = db.get_bind().dialect.name == "postgresql"
    try:
        stmt = select(Lock).where(Lock.name == BOOKING_LOCK)
        if is_postgres:
            stmt = stmt.with_for_update()
        row = db.execute(stmt).scalars().first()
        if row is None:
            # First booking ever: create the sentinel, then take it properly.
            db.add(Lock(name=BOOKING_LOCK, holder="-"))
            db.commit()
            stmt = select(Lock).where(Lock.name == BOOKING_LOCK)
            if is_postgres:
                stmt = stmt.with_for_update()
            db.execute(stmt).scalars().first()
        yield
    finally:
        # The lock is released when the caller commits or rolls back; there is
        # nothing to undo here.
        pass
