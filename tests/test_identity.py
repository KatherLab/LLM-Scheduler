"""Authentication and the break-glass path.

The fallback behaviour is the part worth pinning down: it has to work when the
directory is down, without becoming a way to bypass the directory when it is up.
"""

from __future__ import annotations

import pytest

from app.auth import session_to_user
from app.identity import (
    LOCAL_ADMIN_SUB,
    AuthenticationError,
    StaticIdentityProvider,
    authenticate,
    authenticate_local_admin,
    set_provider,
)
from app.identity import _group_name_from_dn
from app.settings import settings


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(settings, "auth_mode", "ldap")
    monkeypatch.setattr(settings, "auth_password", "break-glass-secret")
    monkeypatch.setattr(settings, "local_admin_enabled", True)
    monkeypatch.setattr(settings, "admin_groups", "llm-admins")
    monkeypatch.setattr(settings, "user_groups", "")
    monkeypatch.setattr(settings, "pool_operators", "")

    p = StaticIdentityProvider()
    p.add("alice", "pw-alice", groups=["llm-users", "radiology"], display_name="Alice A.")
    p.add("root", "pw-root", groups=["llm-admins"])
    set_provider(p)
    yield p
    set_provider(None)


# ── Directory authentication ─────────────────────────────────────────────────

def test_valid_credentials_return_principal_with_groups(provider):
    principal = authenticate("alice", "pw-alice")
    assert principal.sub == "alice"
    assert principal.display_name == "Alice A."
    assert "radiology" in principal.groups
    assert principal.via == "static"


def test_wrong_password_is_rejected(provider):
    with pytest.raises(AuthenticationError):
        authenticate("alice", "wrong")


def test_unknown_user_is_rejected(provider):
    with pytest.raises(AuthenticationError):
        authenticate("nobody", "whatever")


# ── Break-glass ──────────────────────────────────────────────────────────────

def test_break_glass_works_when_the_directory_is_down(provider):
    """The scheduler must stay reachable during an IPA outage — it is what we
    would need in order to respond to one."""
    provider.unavailable = True
    principal = authenticate("anyone", "break-glass-secret")
    assert principal.sub == LOCAL_ADMIN_SUB
    assert principal.is_local_admin


def test_break_glass_still_requires_the_right_password(provider):
    provider.unavailable = True
    with pytest.raises(AuthenticationError):
        authenticate("anyone", "not-the-secret")


def test_directory_outage_does_not_let_a_real_user_in_with_a_bad_password(provider):
    provider.unavailable = True
    with pytest.raises(AuthenticationError):
        authenticate("alice", "pw-alice")   # not the break-glass secret


def test_break_glass_can_be_disabled(provider, monkeypatch):
    monkeypatch.setattr(settings, "local_admin_enabled", False)
    provider.unavailable = True
    with pytest.raises(AuthenticationError):
        authenticate("anyone", "break-glass-secret")


def test_local_admin_is_marked_so_the_ui_can_warn():
    principal = authenticate_local_admin(settings.auth_password)
    assert principal.is_local_admin
    assert principal.groups == frozenset()


# ── Password mode (no directory configured) ──────────────────────────────────

def test_password_mode_uses_the_shared_secret(monkeypatch):
    """Existing single-password deployments keep working unchanged."""
    monkeypatch.setattr(settings, "auth_mode", "password")
    monkeypatch.setattr(settings, "auth_password", "changeme")
    set_provider(None)
    principal = authenticate("", "changeme")
    assert principal.sub == LOCAL_ADMIN_SUB


# ── Session round-trip ───────────────────────────────────────────────────────

def test_session_rebuilds_user_with_roles(provider):
    principal = authenticate("root", "pw-root")
    session = {
        "sub": principal.sub,
        "name": principal.display_name,
        "groups": sorted(principal.groups),
        "via": principal.via,
    }
    user = session_to_user(session)
    assert user.sub == "root"
    assert user.is_admin


def test_missing_session_is_anonymous():
    assert session_to_user(None).sub == ""
    assert session_to_user({}).sub == ""


def test_roles_are_rederived_not_stored(provider, monkeypatch):
    """Changing ADMIN_GROUPS takes effect without forcing everyone to re-login."""
    session = {"sub": "alice", "groups": ["radiology"], "via": "ldap"}
    assert not session_to_user(session).is_admin

    monkeypatch.setattr(settings, "admin_groups", "radiology")
    assert session_to_user(session).is_admin


# ── FreeIPA DN parsing ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "dn,expected",
    [
        ("cn=llm-admins,cn=groups,cn=accounts,dc=example,dc=de", "llm-admins"),
        ("CN=llm-users,CN=groups,DC=x", "llm-users"),
        ("uid=alice,cn=users", None),
        ("", None),
    ],
)
def test_group_name_from_dn(dn, expected):
    assert _group_name_from_dn(dn) == expected
