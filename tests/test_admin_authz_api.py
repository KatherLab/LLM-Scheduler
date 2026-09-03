"""Authorization enforced through the real HTTP routes.

`test_authz.py` proves the rule; this proves the rule is actually wired into
every mutating endpoint. Unenforced RBAC is the failure mode worth testing for.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.admin import router as admin_router
from app.auth import current_user
from app.authz import User
from app.catalog import CatalogModel
from app.dependencies import SessionLocal, init_db
from app.models import Lease

ALICE = User(sub="alice", display_name="Alice", groups=frozenset({"radiology"}),
             is_user=True, via="ldap")
BOB = User(sub="bob", display_name="Bob", groups=frozenset({"pathology"}),
           is_user=True, via="ldap")
CAROL = User(sub="carol", display_name="Carol", groups=frozenset({"radiology"}),
             is_user=True, via="ldap")
ADMIN = User(sub="root", display_name="Root", is_admin=True, is_user=True, via="ldap")

CATALOG = {
    "test-model": CatalogModel(
        name="test-model", model_path="Qwen/Qwen3-0.6B",
        gpus=1, tensor_parallel_size=1,
    )
}


@pytest.fixture
def client(monkeypatch):
    """An app with only the admin router, and a swappable current user."""
    from fastapi import FastAPI

    monkeypatch.setattr("app.admin.get_catalog", lambda *a, **k: CATALOG)
    # Never touch a real scheduler from a route test.
    monkeypatch.setattr("app.admin._submit_to_slurm", lambda lease: "job-1")

    init_db()
    app = FastAPI()
    app.include_router(admin_router)

    state = {"user": ALICE}
    app.dependency_overrides[current_user] = lambda: state["user"]
    # The router-level auth dependency is satisfied by the override above.
    from app.auth import require_auth
    app.dependency_overrides[require_auth] = lambda: {"sub": state["user"].sub}

    c = TestClient(app)
    c.act_as = lambda user: state.__setitem__("user", user)
    yield c

    with SessionLocal() as db:
        for lease in db.query(Lease).all():
            db.delete(lease)
        db.commit()


def _make_lease(**over) -> int:
    """Insert a lease directly — booking flow is not what we are testing."""
    now = datetime.now(timezone.utc)
    fields = dict(
        model="test-model", requested_gpus=1, requested_tp=1, requested_port=0,
        model_path="Qwen/Qwen3-0.6B", state="PLANNED",
        owner="Alice", owner_sub="alice", created_at=now,
        begin_at=now + timedelta(hours=2), end_at=now + timedelta(hours=4),
    )
    fields.update(over)
    with SessionLocal() as db:
        lease = Lease(**fields)
        db.add(lease)
        db.commit()
        return lease.id


def _get_lease(lease_id: int) -> Lease:
    with SessionLocal() as db:
        return db.get(Lease, lease_id)


# ── Ownership enforcement ────────────────────────────────────────────────────

def test_owner_can_cancel_own_booking(client):
    lease_id = _make_lease()
    client.act_as(ALICE)
    assert client.delete(f"/admin/leases/{lease_id}").status_code == 200
    assert _get_lease(lease_id).state == "CANCELED"


def test_stranger_cannot_cancel(client):
    lease_id = _make_lease()
    client.act_as(BOB)
    res = client.delete(f"/admin/leases/{lease_id}")
    assert res.status_code == 403
    assert _get_lease(lease_id).state == "PLANNED"


def test_stranger_cannot_edit(client):
    lease_id = _make_lease()
    client.act_as(BOB)
    res = client.patch(f"/admin/leases/{lease_id}", json={"notes": "hijacked"})
    assert res.status_code == 403


def test_stranger_cannot_edit_notes(client):
    """The notes route is a separate endpoint and was easy to forget."""
    lease_id = _make_lease()
    client.act_as(BOB)
    assert client.patch(f"/admin/leases/{lease_id}/notes",
                        json={"notes": "hijacked"}).status_code == 403


def test_stranger_cannot_stop_now(client):
    lease_id = _make_lease(state="RUNNING")
    client.act_as(BOB)
    assert client.post(f"/admin/leases/{lease_id}/stop").status_code == 403


def test_stranger_cannot_extend(client):
    lease_id = _make_lease(state="RUNNING")
    client.act_as(BOB)
    assert client.post(f"/admin/leases/{lease_id}/extend",
                       json={"duration_seconds": 3600}).status_code == 403


def test_stranger_cannot_shorten(client):
    lease_id = _make_lease(state="RUNNING")
    new_end = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
    client.act_as(BOB)
    assert client.post(f"/admin/leases/{lease_id}/shorten",
                       json={"new_end_at": new_end}).status_code == 403


def test_group_member_can_cancel_group_owned_booking(client):
    lease_id = _make_lease(owner_group="radiology")
    client.act_as(CAROL)          # same group, different person
    assert client.delete(f"/admin/leases/{lease_id}").status_code == 200


def test_admin_can_cancel_anything(client):
    lease_id = _make_lease()
    client.act_as(ADMIN)
    assert client.delete(f"/admin/leases/{lease_id}").status_code == 200


# ── Locking ──────────────────────────────────────────────────────────────────

def test_owner_cannot_lock(client):
    lease_id = _make_lease()
    client.act_as(ALICE)
    assert client.post(f"/admin/leases/{lease_id}/lock",
                       json={"reason": "mine"}).status_code == 403


def test_admin_can_lock_and_it_records_who_and_why(client):
    lease_id = _make_lease()
    client.act_as(ADMIN)
    res = client.post(f"/admin/leases/{lease_id}/lock",
                      json={"reason": "shared prod endpoint"})
    assert res.status_code == 200

    lease = _get_lease(lease_id)
    assert lease.locked is True
    assert lease.locked_by == "root"
    assert lease.locked_reason == "shared prod endpoint"


def test_locked_booking_resists_owner_cancellation(client):
    lease_id = _make_lease(locked=True, locked_by="root", locked_reason="prod")
    client.act_as(ALICE)
    res = client.delete(f"/admin/leases/{lease_id}")
    assert res.status_code == 403
    assert "locked" in res.json()["detail"].lower()
    assert "prod" in res.json()["detail"]
    assert _get_lease(lease_id).state == "PLANNED"


def test_locked_booking_can_still_be_cancelled_by_admin(client):
    lease_id = _make_lease(locked=True)
    client.act_as(ADMIN)
    assert client.delete(f"/admin/leases/{lease_id}").status_code == 200


def test_unlock_restores_owner_control(client):
    lease_id = _make_lease(locked=True, locked_by="root")

    client.act_as(ADMIN)
    assert client.post(f"/admin/leases/{lease_id}/unlock").status_code == 200

    client.act_as(ALICE)
    assert client.delete(f"/admin/leases/{lease_id}").status_code == 200


# ── Permission hints on read ─────────────────────────────────────────────────

def test_lease_list_reports_per_user_permissions(client):
    lease_id = _make_lease()

    client.act_as(ALICE)
    mine = next(x for x in client.get("/admin/leases").json() if x["id"] == lease_id)
    assert mine["can_edit"] and mine["can_cancel"] and not mine["can_lock"]

    client.act_as(BOB)
    theirs = next(x for x in client.get("/admin/leases").json() if x["id"] == lease_id)
    assert not theirs["can_edit"] and not theirs["can_cancel"]

    client.act_as(ADMIN)
    admin_view = next(x for x in client.get("/admin/leases").json() if x["id"] == lease_id)
    assert admin_view["can_lock"]


# ── Creation ─────────────────────────────────────────────────────────────────

def test_created_booking_is_stamped_with_the_caller(client):
    client.act_as(ALICE)
    res = client.post("/admin/leases", json={
        "model": "test-model", "duration_seconds": 3600,
        "begin_at": (datetime.now(timezone.utc) + timedelta(days=3)).isoformat(),
    })
    assert res.status_code == 200
    assert _get_lease(res.json()["id"]).owner_sub == "alice"


def test_cannot_assign_ownership_to_a_group_you_are_not_in(client):
    """Otherwise anyone could hand edit rights to an arbitrary team."""
    client.act_as(BOB)          # in 'pathology', not 'radiology'
    res = client.post("/admin/leases", json={
        "model": "test-model", "duration_seconds": 3600, "owner_group": "radiology",
        "begin_at": (datetime.now(timezone.utc) + timedelta(days=4)).isoformat(),
    })
    assert res.status_code == 403


def test_user_without_create_permission_is_refused(client):
    viewer = User(sub="viewer", display_name="V", is_user=False, via="ldap")
    client.act_as(viewer)
    res = client.post("/admin/leases", json={
        "model": "test-model", "duration_seconds": 3600,
    })
    assert res.status_code == 403
