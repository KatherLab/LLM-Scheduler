"""Slurm JWT acquisition and renewal.

A `scontrol token` is short-lived — 30 minutes by default — so a static
`SLURM_JWT` in `.env` turns into an outage at an unpredictable time. Worse, the
failure is quiet: slurmrestd answers 511 and the scheduler pauses, which looks
like a cluster problem rather than an expired credential.

Three sources, in increasing order of automation:

``static``   `SLURM_JWT` from the environment. Simple; expires.
``file``     Read from a path something else refreshes — a cron job, a systemd
             timer, a Kubernetes secret. **The safest option**, because the
             scheduler never holds a credential that can mint more credentials.
``command``  Run a command that prints a token. The documented recipe is SSH to
             a host that has `scontrol`, but any command works (Vault, a helper
             script, or plain `scontrol token` where the binaries exist).

Renewal is driven by the token's own `exp` claim rather than an assumed
lifespan, so it stays correct if the cluster hands out a different one than we
asked for. It is proactive (refresh at `refresh_margin` before expiry) *and*
reactive (a 511 forces a refresh and one retry), so a token that dies early
self-heals instead of pausing the scheduler.
"""

from __future__ import annotations

import base64
import json
import logging
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

#: `scontrol token` prints `SLURM_JWT=eyJ...`; accept a bare token too.
_PREFIX = "SLURM_JWT="


class TokenError(Exception):
    """Could not obtain a usable token."""


def parse_token_output(output: str) -> str:
    """Extract the token from `scontrol token` output.

    Tolerates surrounding noise (an SSH banner, an MOTD) by scanning lines
    rather than assuming the token is the whole of stdout.
    """
    for line in (output or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(_PREFIX):
            return line[len(_PREFIX):].strip()
    # Fall back to a bare JWT on its own line.
    for line in (output or "").splitlines():
        line = line.strip()
        if line.count(".") == 2 and len(line) > 40 and " " not in line:
            return line
    raise TokenError(
        "could not find a token in the command output; expected a line like "
        "'SLURM_JWT=eyJ...'"
    )


def token_expiry(token: str) -> float | None:
    """Read the `exp` claim without verifying the signature.

    We are not authenticating the token — slurmrestd does that. We only need to
    know when to ask for a new one, and the issuer's own answer beats guessing
    from a configured lifespan.
    """
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return None
    exp = claims.get("exp")
    return float(exp) if isinstance(exp, (int, float)) else None


def token_username(token: str) -> str | None:
    """The account the token authenticates as (`sun` claim), for logging.

    Useful because a token minted by root submits jobs as root unless
    `SLURM_REST_USER` says otherwise.
    """
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return None
    return claims.get("sun") or claims.get("username")


@dataclass
class _Cached:
    token: str
    expires_at: float | None


class TokenProvider:
    """Caches a token and renews it before it expires.

    Thread-safe: renewal runs under a lock so a burst of expired requests
    triggers one refresh, not one per caller.
    """

    def __init__(
        self,
        fetch: Callable[[], str],
        *,
        refresh_margin: float = 300.0,
        name: str = "token",
    ):
        self._fetch = fetch
        self._margin = refresh_margin
        self._name = name
        self._lock = threading.Lock()
        self._cached: _Cached | None = None

    def _needs_refresh(self, cached: _Cached | None) -> bool:
        if cached is None:
            return True
        if cached.expires_at is None:
            # No exp claim: nothing to renew against, so keep using it and let
            # a 511 trigger the reactive path.
            return False
        return time.time() >= cached.expires_at - self._margin

    def get(self, *, force: bool = False) -> str:
        cached = self._cached
        if not force and not self._needs_refresh(cached):
            return cached.token

        with self._lock:
            # Re-check: another thread may have refreshed while we waited.
            cached = self._cached
            if not force and not self._needs_refresh(cached):
                return cached.token

            token = self._fetch()
            if not token:
                raise TokenError(f"{self._name}: fetch returned an empty token")

            expires_at = token_expiry(token)
            self._cached = _Cached(token=token, expires_at=expires_at)

            user = token_username(token)
            if expires_at:
                remaining = int(expires_at - time.time())
                logger.info(
                    "%s: obtained token for %s, valid %d min",
                    self._name, user or "?", max(0, remaining // 60),
                )
                if remaining < self._margin:
                    logger.warning(
                        "%s: token lifespan (%ds) is shorter than the refresh "
                        "margin (%ds) — it will be renewed on almost every call. "
                        "Ask for a longer `lifespan=`.",
                        self._name, remaining, int(self._margin),
                    )
            else:
                logger.info(
                    "%s: obtained token for %s (no exp claim; renewal is "
                    "reactive only)", self._name, user or "?",
                )
            return self._cached.token

    def invalidate(self) -> None:
        """Drop the cache so the next `get()` fetches a fresh token."""
        with self._lock:
            self._cached = None


# ── Sources ──────────────────────────────────────────────────────────────────

def static_source(token: str) -> Callable[[], str]:
    """A fixed token from the environment. Expires; renewal is impossible."""
    def fetch() -> str:
        if not token:
            raise TokenError("SLURM_JWT is empty")
        return token
    return fetch


def file_source(path: str) -> Callable[[], str]:
    """Read the token from a file refreshed by something outside this app.

    Preferred where possible: the scheduler holds only a token, never a
    credential capable of minting one.
    """
    def fetch() -> str:
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise TokenError(f"cannot read token file {path}: {exc}") from exc
        try:
            return parse_token_output(raw)
        except TokenError:
            # A bare token with no SLURM_JWT= prefix is fine here.
            stripped = raw.strip()
            if stripped:
                return stripped
            raise
    return fetch


def command_source(command: str, timeout: float = 30.0) -> Callable[[], str]:
    """Run a command that prints a token.

    The command is split with `shlex`, **not** run through a shell, so a stray
    quote cannot turn into command injection. Use a wrapper script if you need
    pipes or expansion.
    """
    argv = shlex.split(command)
    if not argv:
        raise TokenError("SLURM_TOKEN_COMMAND is empty")

    def fetch() -> str:
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout, check=False,
            )
        except FileNotFoundError as exc:
            raise TokenError(f"token command not found: {argv[0]} ({exc})") from exc
        except subprocess.TimeoutExpired as exc:
            raise TokenError(
                f"token command timed out after {timeout}s: {argv[0]}"
            ) from exc

        if proc.returncode != 0:
            # stderr can carry an SSH failure reason; it does not contain the
            # token, so it is safe to surface.
            raise TokenError(
                f"token command failed ({proc.returncode}): "
                f"{(proc.stderr or proc.stdout or '').strip()[:300]}"
            )
        return parse_token_output(proc.stdout)
    return fetch


def ssh_command(
    host: str, *, user: str | None = None, key: str | None = None,
    lifespan: int = 3600, known_hosts: str | None = None,
    slurm_user: str | None = None, port: int | None = None,
) -> str:
    """Build the SSH command for the documented renewal recipe.

    Host key checking stays **on**: an attacker who can impersonate the host can
    hand us a token pointing at their own slurmrestd. Supply `known_hosts`.
    """
    target = f"{user}@{host}" if user else host
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    if port:
        argv += ["-p", str(port)]
    if known_hosts:
        argv += ["-o", f"UserKnownHostsFile={known_hosts}",
                 "-o", "StrictHostKeyChecking=yes"]
    if key:
        argv += ["-i", key, "-o", "IdentitiesOnly=yes"]
    argv += [target, "scontrol", "token", f"lifespan={lifespan}"]
    if slurm_user:
        # Minting for another account needs SlurmUser/root privileges remotely.
        argv.append(f"username={slurm_user}")
    return shlex.join(argv)


# ── Wiring ───────────────────────────────────────────────────────────────────

_provider: TokenProvider | None = None


def build_provider(settings) -> TokenProvider:
    """Construct the provider described by settings."""
    mode = (settings.slurm_token_mode or "static").strip().lower()
    margin = float(settings.slurm_token_refresh_margin_seconds)

    if mode == "file":
        if not settings.slurm_token_file:
            raise TokenError("SLURM_TOKEN_MODE=file requires SLURM_TOKEN_FILE")
        return TokenProvider(
            file_source(settings.slurm_token_file),
            refresh_margin=margin, name="slurm-token(file)",
        )

    if mode in ("command", "ssh"):
        command = settings.slurm_token_command
        if not command and settings.slurm_token_ssh_host:
            command = ssh_command(
                settings.slurm_token_ssh_host,
                user=settings.slurm_token_ssh_user,
                key=settings.slurm_token_ssh_key,
                lifespan=settings.slurm_token_lifespan_seconds,
                known_hosts=settings.slurm_token_ssh_known_hosts,
                slurm_user=settings.slurm_rest_user,
                port=settings.slurm_token_ssh_port or None,
            )
        if not command:
            raise TokenError(
                "SLURM_TOKEN_MODE=command requires SLURM_TOKEN_COMMAND or "
                "SLURM_TOKEN_SSH_HOST"
            )
        logger.info("slurm token: renewing via `%s`", _redact(command))
        return TokenProvider(
            command_source(command, timeout=settings.slurm_token_timeout_seconds),
            refresh_margin=margin, name="slurm-token(command)",
        )

    return TokenProvider(
        static_source(settings.slurm_jwt),
        refresh_margin=margin, name="slurm-token(static)",
    )


def _redact(command: str) -> str:
    """Keep private key paths out of the logs."""
    parts = shlex.split(command)
    out = []
    skip = False
    for part in parts:
        if skip:
            out.append("<key>")
            skip = False
            continue
        if part == "-i":
            skip = True
        out.append(part)
    return shlex.join(out)


def get_provider() -> TokenProvider:
    global _provider
    if _provider is None:
        from .settings import settings

        _provider = build_provider(settings)
    return _provider


def set_provider(provider: TokenProvider | None) -> None:
    """Override the provider. For tests."""
    global _provider
    _provider = provider
