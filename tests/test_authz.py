"""Authorization rules.

The interesting cases are the ones that pure "only the owner may edit" gets
wrong: group-owned deployments, pool operators, and locked production models.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.authz import (
    ANONYMOUS,
    CANCEL,
    CREATE,
    EDIT,
    EXTEND,
    LOCK,
    UNLOCK,
    VIEW,
    build_user,
    can,
    can_create,
    describe_denial,
    owns,
)
from app.identity import Principal
from app.settings import settings


@dataclass
class FakeLease:
    owner_sub: str | None = "alice"
    owner_group: str | None = None
    locked: bool = False
    locked_by: str | None = None
    locked_reason: str | None = None
    pool: str | None = "general"


@pytest.fixture(autouse=True)
def _rbac_config(monkeypatch):
    monkeypatch.setattr(settings, "admin_groups", "llm-admins")
    monkeypatch.setattr(settings, "user_groups", "")
    monkeypatch.setattr(settings, "pool_operators", "llm-dedicated:llm-gpu-team")


def _user(sub: str, groups: list[str] | None = None, via: str = "ldap"):
    return build_user(Principal(
        sub=sub, display_name=sub, groups=frozenset(groups or []), via=via
    ))


# ── Role derivation ──────────────────────────────────────────────────────────

def test_admin_group_grants_admin():
    assert _user("root", ["llm-admins"]).is_admin


def test_ordinary_user_is_not_admin():
    assert not _user("alice", ["llm-users"]).is_admin


def test_empty_user_groups_means_any_identity_may_book():
    assert _user("alice", []).is_user


def test_user_groups_restricts_booking(monkeypatch):
    monkeypatch.setattr(settings, "user_groups", "llm-users")
    assert _user("alice", ["llm-users"]).is_user
    assert not _user("bob", ["other"]).is_user


def test_operator_pools_resolved_from_groups():
    assert _user("ops", ["llm-gpu-team"]).operator_pools == frozenset({"llm-dedicated"})
    assert _user("alice", []).operator_pools == frozenset()


def test_break_glass_account_is_admin_without_any_groups():
    """It exists precisely for when group lookup is impossible."""
    admin = _user("local-admin", [], via="local")
    assert admin.is_admin
    assert admin.is_local_admin


# ── Ownership ────────────────────────────────────────────────────────────────

def test_owner_can_edit_own_lease():
    assert can(_user("alice"), EDIT, FakeLease(owner_sub="alice"))


def test_non_owner_cannot_edit():
    assert not can(_user("bob"), EDIT, FakeLease(owner_sub="alice"))


def test_group_member_can_edit_group_owned_lease():
    """The vacation / shared-team-model case, without escalating to admin."""
    lease = FakeLease(owner_sub="alice", owner_group="radiology")
    assert can(_user("bob", ["radiology"]), EDIT, lease)


def test_non_member_cannot_edit_group_owned_lease():
    lease = FakeLease(owner_sub="alice", owner_group="radiology")
    assert not can(_user("bob", ["pathology"]), EDIT, lease)


def test_ownerless_lease_is_not_owned_by_everyone():
    """Historical rows have no owner_sub; they must not become a free-for-all."""
    assert not owns(_user("alice"), FakeLease(owner_sub=None))
    assert not can(_user("alice"), EDIT, FakeLease(owner_sub=None))


def test_admin_can_manage_ownerless_lease():
    """...but an admin must still be able to clean them up."""
    assert can(_user("root", ["llm-admins"]), CANCEL, FakeLease(owner_sub=None))


# ── Everyone can look ────────────────────────────────────────────────────────

def test_any_authenticated_user_can_view():
    assert can(_user("bob"), VIEW, FakeLease(owner_sub="alice"))


def test_anonymous_can_do_nothing():
    assert not can(ANONYMOUS, VIEW, FakeLease())
    assert not can_create(ANONYMOUS)


# ── Pool operators ───────────────────────────────────────────────────────────

def test_operator_can_manage_anything_in_their_pool():
    lease = FakeLease(owner_sub="alice", pool="llm-dedicated")
    assert can(_user("ops", ["llm-gpu-team"]), CANCEL, lease)


def test_operator_has_no_power_in_other_pools():
    """Stewardship is scoped per pool — that is the point of scoping it."""
    lease = FakeLease(owner_sub="alice", pool="general")
    assert not can(_user("ops", ["llm-gpu-team"]), CANCEL, lease)


# ── Locking ──────────────────────────────────────────────────────────────────

def test_locked_lease_rejects_owner_mutations():
    lease = FakeLease(owner_sub="alice", locked=True)
    for action in (EDIT, CANCEL, EXTEND):
        assert not can(_user("alice"), action, lease)


def test_locked_lease_is_still_viewable_by_owner():
    assert can(_user("alice"), VIEW, FakeLease(owner_sub="alice", locked=True))


def test_admin_can_mutate_a_locked_lease():
    assert can(_user("root", ["llm-admins"]), CANCEL, FakeLease(locked=True))


def test_operator_can_mutate_a_locked_lease_in_their_pool():
    lease = FakeLease(owner_sub="alice", locked=True, pool="llm-dedicated")
    assert can(_user("ops", ["llm-gpu-team"]), CANCEL, lease)


def test_owner_cannot_lock_their_own_lease():
    """Otherwise anyone could make their booking immune to cleanup."""
    assert not can(_user("alice"), LOCK, FakeLease(owner_sub="alice"))


def test_admin_and_pool_operator_can_lock():
    assert can(_user("root", ["llm-admins"]), LOCK, FakeLease())
    lease = FakeLease(pool="llm-dedicated")
    assert can(_user("ops", ["llm-gpu-team"]), LOCK, lease)


def test_unlock_needs_the_same_privilege_as_lock():
    lease = FakeLease(owner_sub="alice", locked=True)
    assert not can(_user("alice"), UNLOCK, lease)
    assert can(_user("root", ["llm-admins"]), UNLOCK, lease)


# ── Creation ─────────────────────────────────────────────────────────────────

def test_authenticated_user_can_create_by_default():
    assert can_create(_user("alice"))


def test_creation_denied_when_not_in_user_groups(monkeypatch):
    monkeypatch.setattr(settings, "user_groups", "llm-users")
    assert not can_create(_user("outsider", ["other"]))
    assert can_create(_user("alice", ["llm-users"]))


def test_admin_can_always_create(monkeypatch):
    monkeypatch.setattr(settings, "user_groups", "llm-users")
    assert can_create(_user("root", ["llm-admins"]))


def test_create_is_not_routed_through_can():
    """CREATE has no resource to own; can() would fall through to ownership."""
    assert not can(_user("alice"), CREATE, FakeLease(owner_sub="bob"))


# ── Denial messages ──────────────────────────────────────────────────────────

def test_denial_message_for_locked_names_the_locker_and_reason():
    lease = FakeLease(locked=True, locked_by="root", locked_reason="prod endpoint")
    msg = describe_denial(_user("alice"), CANCEL, lease)
    assert "locked" in msg and "root" in msg and "prod endpoint" in msg


def test_denial_message_for_ownership_suggests_a_route_forward():
    msg = describe_denial(_user("bob"), EDIT, FakeLease(owner_sub="alice"))
    assert "own" in msg and "admin" in msg
