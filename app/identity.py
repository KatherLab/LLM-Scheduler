"""Who is making the request.

Authentication only — what a user is *allowed* to do lives in `app/authz.py`.

Two providers:

  LdapIdentityProvider    simple bind against FreeIPA, then read group membership
  StaticIdentityProvider  fixture accounts, for tests and laptop development

Both sit behind `IdentityProvider`, and both are wrapped by a break-glass local
admin account (`AUTH_PASSWORD`). That fallback is not optional: an IPA outage
must not lock us out of our own scheduler, which is the thing we would need in
order to react to the outage.
"""

from __future__ import annotations

import hmac
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from .settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Principal:
    """An authenticated identity, before roles are applied."""

    sub: str                              # LDAP uid — the stable handle
    display_name: str = ""
    email: str = ""                       # from the directory's `mail` attribute
    groups: frozenset[str] = field(default_factory=frozenset)
    via: str = "unknown"                  # "ldap" | "local" | "static"

    @property
    def is_local_admin(self) -> bool:
        return self.via == "local"


class AuthenticationError(Exception):
    """Bad credentials. Deliberately carries no detail for the caller to leak."""


class IdentityUnavailableError(Exception):
    """The identity source is unreachable — distinct from bad credentials.

    Callers surface this differently: wrong password is the user's problem,
    an unreachable IPA is ours, and only the latter justifies falling back to
    the break-glass account.
    """


class IdentityProvider(Protocol):
    name: str

    def authenticate(self, username: str, password: str) -> Principal:
        ...


# ── Static provider (tests, development) ─────────────────────────────────────

class StaticIdentityProvider:
    """In-memory accounts. `CLUSTER_BACKEND=local`'s counterpart for auth."""

    name = "static"

    def __init__(self, accounts: dict[str, dict] | None = None):
        # {username: {"password": ..., "groups": [...], "display_name": ...}}
        self._accounts = accounts or {}
        self.unavailable = False

    def add(self, username: str, password: str, groups: list[str] | None = None,
            display_name: str = "", email: str = "") -> None:
        self._accounts[username] = {
            "password": password,
            "groups": groups or [],
            "display_name": display_name or username,
            "email": email,
        }

    def authenticate(self, username: str, password: str) -> Principal:
        if self.unavailable:
            raise IdentityUnavailableError("static provider: simulated outage")
        account = self._accounts.get(username)
        if account is None or not hmac.compare_digest(password, account["password"]):
            raise AuthenticationError("invalid credentials")
        return Principal(
            sub=username,
            display_name=account.get("display_name") or username,
            email=account.get("email", ""),
            groups=frozenset(account.get("groups") or []),
            via="static",
        )


# ── LDAP / FreeIPA ───────────────────────────────────────────────────────────

_CN_RE = re.compile(r"^cn=([^,]+)", re.IGNORECASE)


def _group_name_from_dn(dn: str) -> str | None:
    """`cn=llm-admins,cn=groups,cn=accounts,dc=x` -> `llm-admins`."""
    m = _CN_RE.match(dn.strip())
    return m.group(1) if m else None


class LdapIdentityProvider:
    """Simple bind against FreeIPA, then resolve group membership.

    Membership is read from the user entry's `memberOf` where available (IPA
    populates it, and it costs no extra search); otherwise it falls back to a
    group search. Results are cached briefly so a login storm does not become an
    LDAP storm.
    """

    name = "ldap"

    def __init__(self):
        # username -> (cached_at, (groups, display_name, email))
        self._cache: dict[str, tuple[float, tuple[frozenset[str], str, str]]] = {}

    def _connect(self, user_dn: str, password: str):
        try:
            from ldap3 import ALL, Connection, Server
            from ldap3.core.exceptions import (
                LDAPBindError,
                LDAPException,
                LDAPInvalidCredentialsResult,
            )
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise IdentityUnavailableError(f"ldap3 not installed: {exc}") from exc

        server = Server(
            settings.ldap_url,
            get_info=ALL,
            connect_timeout=settings.ldap_timeout_seconds,
        )
        try:
            conn = Connection(
                server,
                user=user_dn,
                password=password,
                auto_bind=False,
                receive_timeout=settings.ldap_timeout_seconds,
            )
            if settings.ldap_start_tls:
                conn.start_tls()
            if not conn.bind():
                # A failed bind with a reachable server means bad credentials.
                raise AuthenticationError("invalid credentials")
            return conn
        except (LDAPInvalidCredentialsResult, LDAPBindError) as exc:
            raise AuthenticationError("invalid credentials") from exc
        except LDAPException as exc:
            raise IdentityUnavailableError(f"LDAP unreachable: {exc}") from exc

    def authenticate(self, username: str, password: str) -> Principal:
        if not settings.ldap_url:
            raise IdentityUnavailableError("AUTH_MODE=ldap but LDAP_URL is unset")
        # An empty password would be an unauthenticated bind, which LDAP accepts.
        if not username or not password:
            raise AuthenticationError("invalid credentials")

        user_dn = settings.ldap_user_dn_template.format(username=username)
        conn = self._connect(user_dn, password)
        try:
            groups, display_name, email = self._read_user(conn, user_dn, username)
        finally:
            try:
                conn.unbind()
            except Exception:
                pass

        return Principal(
            sub=username,
            display_name=display_name or username,
            email=email,
            groups=groups,
            via="ldap",
        )

    def _read_user(
        self, conn, user_dn: str, username: str
    ) -> tuple[frozenset[str], str, str]:
        cached = self._cache.get(username)
        now = time.monotonic()
        if cached and now - cached[0] < settings.ldap_group_cache_seconds:
            groups, display_name, email = cached[1]
            return groups, display_name, email

        groups: set[str] = set()
        display_name = ""
        email = ""

        try:
            conn.search(
                search_base=user_dn,
                search_filter="(objectClass=*)",
                search_scope="BASE",
                attributes=["memberOf", "displayName", "cn", "mail"],
            )
            if conn.entries:
                entry = conn.entries[0]
                for dn in _attr_list(entry, "memberOf"):
                    name = _group_name_from_dn(dn)
                    if name:
                        groups.add(name)
                display_name = _attr_first(entry, "displayName") or _attr_first(entry, "cn")
                # Used for Slurm job notifications in preference to
                # SLURM_MAIL_USER, so mail reaches whoever booked the model.
                email = _attr_first(entry, "mail")
        except Exception as exc:
            logger.warning("ldap: user lookup failed for %s: %s", username, exc)

        if not groups and settings.ldap_group_base_dn:
            groups = self._search_groups(conn, user_dn, username)

        result = (frozenset(groups), display_name, email)
        self._cache[username] = (now, result)
        return result

    def _search_groups(self, conn, user_dn: str, username: str) -> set[str]:
        """Fallback for directories that do not expose `memberOf`."""
        groups: set[str] = set()
        try:
            conn.search(
                search_base=settings.ldap_group_base_dn,
                search_filter=settings.ldap_group_filter.format(
                    user_dn=user_dn, username=username
                ),
                attributes=["cn"],
            )
            for entry in conn.entries:
                cn = _attr_first(entry, "cn")
                if cn:
                    groups.add(cn)
        except Exception as exc:
            logger.warning("ldap: group search failed for %s: %s", username, exc)
        return groups

    def invalidate(self, username: str | None = None) -> None:
        if username is None:
            self._cache.clear()
        else:
            self._cache.pop(username, None)


def _attr_list(entry, name: str) -> list[str]:
    try:
        value = entry[name].values
    except Exception:
        return []
    return [str(v) for v in value or []]


def _attr_first(entry, name: str) -> str:
    values = _attr_list(entry, name)
    return values[0] if values else ""


# ── Break-glass local admin ──────────────────────────────────────────────────

LOCAL_ADMIN_SUB = "local-admin"


def authenticate_local_admin(password: str) -> Principal:
    """The account that works when the directory does not.

    Its use is marked on the session (`via="local"`) so the UI can show that
    the current session is not a real identity.
    """
    if not settings.local_admin_enabled:
        raise AuthenticationError("local admin disabled")
    if not hmac.compare_digest(password, settings.auth_password):
        raise AuthenticationError("invalid credentials")
    return Principal(
        sub=LOCAL_ADMIN_SUB,
        display_name="Local admin (break-glass)",
        groups=frozenset(),
        via="local",
    )


# ── Provider registry ────────────────────────────────────────────────────────

_provider: IdentityProvider | None = None


def get_provider() -> IdentityProvider | None:
    """The configured provider, or None in password-only mode."""
    global _provider
    if _provider is None and settings.auth_mode.lower() == "ldap":
        _provider = LdapIdentityProvider()
    return _provider


def set_provider(provider: IdentityProvider | None) -> None:
    """Override the provider. For tests and development."""
    global _provider
    _provider = provider


def authenticate(username: str, password: str) -> Principal:
    """Authenticate against the configured provider, then the local admin.

    The local admin is tried when the directory is *unreachable*, and also when
    the directory rejected the credentials — the latter so that the break-glass
    account keeps working while LDAP is merely misconfigured (a wrong DN
    template looks exactly like a wrong password).
    """
    provider = get_provider()
    if provider is None:
        return authenticate_local_admin(password)

    try:
        return provider.authenticate(username, password)
    except IdentityUnavailableError as exc:
        logger.error("identity provider unavailable (%s); trying local admin", exc)
    except AuthenticationError:
        pass

    return authenticate_local_admin(password)
