# app/auth.py
"""
Simple session-based authentication.

Current: password-only (shared password from AUTH_PASSWORD env var).
Future:  swap `verify_credentials()` and add an SSO/OIDC callback route.

Sessions are signed cookies (HMAC). No DB table needed.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Response
from fastapi.responses import FileResponse

from .settings import settings

logger = logging.getLogger(__name__)

# ── Secret key (auto-generate if not configured) ────────────────────────────
_secret_key: bytes = b""


def _get_secret_key() -> bytes:
    global _secret_key
    if _secret_key:
        return _secret_key
    if settings.auth_secret_key:
        _secret_key = settings.auth_secret_key.encode()
    else:
        # Auto-generate a stable key derived from the password + a fixed salt.
        # For production, set AUTH_SECRET_KEY explicitly so sessions survive restarts.
        _secret_key = hashlib.sha256(
            f"vllm-router-session-{settings.auth_password}".encode()
        ).digest()
    return _secret_key


# ── Cookie signing ───────────────────────────────────────────────────────────
COOKIE_NAME = "vllm_session"


def _sign(payload: dict) -> str:
    """Create a signed cookie value: base64(json) + '.' + hex(hmac)."""
    import base64
    raw = json.dumps(payload, separators=(",", ":")).encode()
    b64 = base64.urlsafe_b64encode(raw).decode()
    sig = hmac.new(_get_secret_key(), raw, hashlib.sha256).hexdigest()
    return f"{b64}.{sig}"


def _verify(cookie_value: str) -> Optional[dict]:
    """Verify and decode a signed cookie. Returns None if invalid."""
    import base64
    try:
        b64, sig = cookie_value.rsplit(".", 1)
        raw = base64.urlsafe_b64decode(b64)
        expected_sig = hmac.new(_get_secret_key(), raw, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        payload = json.loads(raw)
        # Check expiry
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


# ── Session helpers ──────────────────────────────────────────────────────────

def create_session_cookie(
    response: Response,
    username: str = "user",
    *,
    groups: list[str] | None = None,
    display_name: str = "",
    email: str = "",
    via: str = "local",
) -> None:
    """Set a signed session cookie on the response.

    Group membership is carried in the cookie so authorization does not hit
    LDAP on every request. The trade-off is that a group change only takes
    effect at the user's next login, which is acceptable for a session TTL
    measured in hours.
    """
    payload = {
        "sub": username,
        "name": display_name or username,
        "email": email,
        "groups": sorted(groups or []),
        "via": via,
        "iat": int(time.time()),
        "exp": int(time.time()) + settings.auth_session_max_age_seconds,
    }
    response.set_cookie(
        key=COOKIE_NAME,
        value=_sign(payload),
        max_age=settings.auth_session_max_age_seconds,
        httponly=True,
        samesite="lax",
        path="/",
        # Set secure=True if you're behind HTTPS:
        # secure=True,
    )


def get_session(request: Request) -> Optional[dict]:
    """Extract and verify the session from the request cookie."""
    cookie = request.cookies.get(COOKIE_NAME)
    if not cookie:
        return None
    return _verify(cookie)


# ── FastAPI dependency for protecting routes ─────────────────────────────────

def require_auth(request: Request) -> dict:
    """
    FastAPI dependency: raises 401 if no valid session.
    Returns the session payload dict on success.

    For SSO migration: this is the single point to change.
    You'd check for an OIDC token or session from your IdP here.
    """
    session = get_session(request)
    if session is None:
        # For API calls (Accept: application/json), return 401 JSON.
        # For browser navigation, we could redirect, but 401 is cleaner
        # since the JS handles it.
        raise HTTPException(status_code=401, detail="Not authenticated")
    return session


def session_to_user(session: dict | None):
    """Rebuild the authorization `User` from a session cookie.

    Roles are re-derived from the cookie's groups on every request rather than
    stored, so changing ADMIN_GROUPS takes effect without forcing re-login.
    """
    from .authz import ANONYMOUS, build_user
    from .identity import Principal

    if not session or not session.get("sub"):
        return ANONYMOUS
    return build_user(Principal(
        sub=session["sub"],
        display_name=session.get("name", ""),
        email=session.get("email", ""),
        groups=frozenset(session.get("groups") or []),
        via=session.get("via", "unknown"),
    ))


def current_user(request: Request):
    """FastAPI dependency: the authenticated `User`, or 401."""
    return session_to_user(require_auth(request))


# ── Internal endpoint auth (for vLLM job registration) ──────────────────────

def require_internal_token(request: Request) -> None:
    """
    Verify that the request carries the internal API key.
    Used for endpoints called by Slurm jobs (e.g., /admin/endpoints/register).
    """
    auth_header = request.headers.get("authorization", "")
    token = request.query_params.get("token", "")

    expected = settings.vllm_api_key

    # Accept via Authorization: Bearer <token> or ?token=<token>
    if auth_header.startswith("Bearer "):
        provided = auth_header[7:].strip()
        if hmac.compare_digest(provided, expected):
            return

    if token and hmac.compare_digest(token, expected):
        return

    raise HTTPException(status_code=403, detail="Invalid internal token")


def require_schedule_key(request: Request) -> None:
    """
    Verify that the request carries the read-only schedule API key.
    Used for external dashboards / status pages that need read-only access
    to the schedule and model catalog.

    Accepts:
      - Authorization: Bearer <SCHEDULE_API_KEY>
      - ?token=<SCHEDULE_API_KEY>

    Returns 403 if the key is missing/wrong, or if SCHEDULE_API_KEY is not configured.
    """
    expected = settings.schedule_api_key
    if not expected:
        raise HTTPException(
            status_code=403,
            detail="Schedule API is not configured (SCHEDULE_API_KEY is empty)",
        )

    auth_header = request.headers.get("authorization", "")
    token = request.query_params.get("token", "")

    if auth_header.startswith("Bearer "):
        provided = auth_header[7:].strip()
        if hmac.compare_digest(provided, expected):
            return

    if token and hmac.compare_digest(token, expected):
        return

    raise HTTPException(status_code=403, detail="Invalid schedule API key")


# ── Auth router (login/logout pages + API) ───────────────────────────────────

auth_router = APIRouter(tags=["auth"])


@auth_router.get("/login")
def login_page():
    """Serve the login HTML page."""
    return FileResponse("app/ui/login.html")


@auth_router.get("/api/auth/config")
def auth_config():
    """What the login form should render.

    In password mode there is no username field, which keeps the historical
    single-secret login intact for deployments that have not moved to LDAP.
    """
    return {"mode": settings.auth_mode.lower()}


@auth_router.get("/api/me")
def whoami(request: Request):
    """The current identity and what it may do — the UI hides controls on this."""
    user = session_to_user(get_session(request))
    return {
        "authenticated": bool(user.sub),
        "sub": user.sub,
        "display_name": user.display_name,
        "groups": sorted(user.groups),
        "is_admin": user.is_admin,
        "operator_pools": sorted(user.operator_pools),
        "can_create": user.is_user or user.is_admin,
        # Surfaced so the UI can warn that this is not a real identity.
        "is_local_admin": user.is_local_admin,
        "via": user.via,
    }


@auth_router.post("/api/login")
def login(request: Request, response: Response, body: dict = None):
    """
    Authenticate against the configured identity provider.

    AUTH_MODE=password  -> shared secret, session is the break-glass admin
    AUTH_MODE=ldap      -> FreeIPA simple bind, with the break-glass admin as
                           a fallback when the directory is unreachable

    For SSO: add a /auth/callback route that validates the OIDC token and calls
    create_session_cookie() with the claims.
    """
    from .identity import AuthenticationError, authenticate

    if body is None:
        body = {}
    username = (body.get("username") or "").strip()
    password = body.get("password", "")

    try:
        principal = authenticate(username, password)
    except AuthenticationError:
        # One message for every failure mode — do not reveal whether the
        # account exists.
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if principal.is_local_admin and settings.auth_mode.lower() == "ldap":
        logger.warning(
            "break-glass local admin login used (LDAP mode active) from %s",
            request.client.host if request.client else "unknown",
        )

    resp = Response(content=json.dumps({
        "ok": True,
        "sub": principal.sub,
        "display_name": principal.display_name,
        "is_local_admin": principal.is_local_admin,
    }), media_type="application/json")

    create_session_cookie(
        resp,
        username=principal.sub,
        groups=sorted(principal.groups),
        display_name=principal.display_name,
        email=principal.email,
        via=principal.via,
    )
    return resp


@auth_router.post("/api/logout")
def logout(response: Response):
    """Clear the session cookie."""
    resp = Response(
        content=json.dumps({"ok": True}),
        media_type="application/json",
    )
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp
