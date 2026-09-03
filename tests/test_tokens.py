"""Slurm JWT acquisition and renewal.

A token that expires quietly is the worst failure mode here: slurmrestd answers
511, the scheduler pauses, and it looks like a cluster outage rather than an
expired credential.
"""

from __future__ import annotations

import base64
import json
import time

import pytest

from app import tokens
from app.tokens import (
    TokenError,
    TokenProvider,
    build_provider,
    command_source,
    file_source,
    parse_token_output,
    ssh_command,
    static_source,
    token_expiry,
    token_username,
)


def _jwt(exp: float | None = None, user: str = "svc-llm") -> str:
    """A structurally valid, unsigned JWT — we only ever read its claims."""
    def b64(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")
    claims = {"sun": user}
    if exp is not None:
        claims["exp"] = exp
    return f"{b64({'alg': 'HS256'})}.{b64(claims)}.signature"


# ── Parsing scontrol output ──────────────────────────────────────────────────

def test_parses_the_scontrol_prefix():
    assert parse_token_output("SLURM_JWT=abc.def.ghi") == "abc.def.ghi"


def test_ignores_an_ssh_banner_before_the_token():
    """SSH prints an MOTD; the token is not necessarily the whole of stdout."""
    out = "Welcome to titan\nLast login: yesterday\nSLURM_JWT=abc.def.ghi\n"
    assert parse_token_output(out) == "abc.def.ghi"


def test_accepts_a_bare_token():
    assert parse_token_output("  abc.def." + "x" * 50 + "  ") == "abc.def." + "x" * 50


def test_empty_output_is_an_error():
    with pytest.raises(TokenError, match="could not find a token"):
        parse_token_output("")


def test_noise_without_a_token_is_an_error():
    with pytest.raises(TokenError):
        parse_token_output("Permission denied (publickey).")


# ── Claims ───────────────────────────────────────────────────────────────────

def test_reads_the_expiry_claim():
    exp = time.time() + 3600
    assert token_expiry(_jwt(exp)) == pytest.approx(exp)


def test_reads_the_username_claim():
    """Worth logging: a root token submits jobs as root unless told otherwise."""
    assert token_username(_jwt(time.time() + 60, user="root")) == "root"


def test_a_malformed_token_has_no_claims():
    assert token_expiry("not-a-jwt") is None
    assert token_username("not-a-jwt") is None


def test_a_token_without_exp_reports_none():
    assert token_expiry(_jwt(None)) is None


# ── Caching and renewal ──────────────────────────────────────────────────────

def test_a_valid_token_is_reused_not_refetched():
    calls = []

    def fetch():
        calls.append(1)
        return _jwt(time.time() + 3600)

    provider = TokenProvider(fetch, refresh_margin=300)
    assert provider.get() == provider.get()
    assert len(calls) == 1


def test_a_token_near_expiry_is_renewed_before_it_dies():
    """Proactive renewal: the margin exists so we never serve a dead token."""
    calls = []

    def fetch():
        calls.append(1)
        # Always about to expire, inside the margin.
        return _jwt(time.time() + 60)

    provider = TokenProvider(fetch, refresh_margin=300)
    provider.get()
    provider.get()
    assert len(calls) == 2


def test_a_token_without_an_expiry_is_not_proactively_renewed():
    """Nothing to renew against; the 511 path handles it reactively."""
    calls = []

    def fetch():
        calls.append(1)
        return _jwt(None)

    provider = TokenProvider(fetch)
    provider.get()
    provider.get()
    assert len(calls) == 1


def test_invalidate_forces_the_next_fetch():
    """This is what a 511 triggers."""
    calls = []

    def fetch():
        calls.append(1)
        return _jwt(time.time() + 3600)

    provider = TokenProvider(fetch)
    provider.get()
    provider.invalidate()
    provider.get()
    assert len(calls) == 2


def test_force_refetches_even_when_valid():
    calls = []

    def fetch():
        calls.append(1)
        return _jwt(time.time() + 3600)

    provider = TokenProvider(fetch)
    provider.get()
    provider.get(force=True)
    assert len(calls) == 2


def test_an_empty_fetch_result_is_rejected():
    with pytest.raises(TokenError, match="empty token"):
        TokenProvider(lambda: "").get()


def test_a_short_lifespan_warns_rather_than_thrashing_silently(caplog):
    """A 60s token with a 300s margin renews on every call — say so."""
    provider = TokenProvider(lambda: _jwt(time.time() + 60), refresh_margin=300)
    with caplog.at_level("WARNING"):
        provider.get()
    assert any("lifespan" in r.message for r in caplog.records)


# ── Sources ──────────────────────────────────────────────────────────────────

def test_static_source_returns_the_configured_token():
    assert static_source("abc")() == "abc"


def test_static_source_rejects_an_empty_token():
    with pytest.raises(TokenError, match="SLURM_JWT is empty"):
        static_source("")()


def test_file_source_reads_a_prefixed_token(tmp_path):
    path = tmp_path / "token"
    path.write_text("SLURM_JWT=abc.def.ghi\n")
    assert file_source(str(path))() == "abc.def.ghi"


def test_file_source_reads_a_bare_token(tmp_path):
    path = tmp_path / "token"
    path.write_text("abc.def.ghi\n")
    assert file_source(str(path))() == "abc.def.ghi"


def test_file_source_reports_a_missing_file_clearly(tmp_path):
    with pytest.raises(TokenError, match="cannot read token file"):
        file_source(str(tmp_path / "nope"))()


def test_command_source_runs_and_parses():
    fetch = command_source("printf SLURM_JWT=abc.def.ghi")
    assert fetch() == "abc.def.ghi"


def test_command_source_surfaces_a_failure():
    with pytest.raises(TokenError, match="failed"):
        command_source("sh -c 'echo denied >&2; exit 255'")()


def test_command_source_reports_a_missing_binary():
    with pytest.raises(TokenError, match="not found"):
        command_source("definitely-not-a-real-binary-xyz")()


def test_command_source_does_not_use_a_shell():
    """Splitting with shlex means a stray quote cannot become injection."""
    fetch = command_source("printf 'SLURM_JWT=a.b.c; rm -rf /'")
    assert fetch() == "a.b.c; rm -rf /"


def test_command_source_times_out():
    with pytest.raises(TokenError, match="timed out"):
        command_source("sleep 5", timeout=0.2)()


# ── SSH recipe ───────────────────────────────────────────────────────────────

def test_ssh_command_shape():
    cmd = ssh_command("titan", user="svc-tokens", key="/keys/id", lifespan=7200)
    assert "svc-tokens@titan" in cmd
    assert "scontrol token lifespan=7200" in cmd
    assert "-i /keys/id" in cmd


def test_ssh_command_uses_batch_mode_so_it_cannot_hang_on_a_prompt():
    assert "BatchMode=yes" in ssh_command("titan")


def test_ssh_command_keeps_host_key_checking_on():
    """An attacker impersonating the host could hand us a token pointing at
    their own slurmrestd."""
    cmd = ssh_command("titan", known_hosts="/keys/known_hosts")
    assert "StrictHostKeyChecking=yes" in cmd
    assert "UserKnownHostsFile=/keys/known_hosts" in cmd


def test_ssh_command_can_mint_for_another_account():
    cmd = ssh_command("titan", slurm_user="svc-llm")
    assert "username=svc-llm" in cmd


# ── Settings wiring ──────────────────────────────────────────────────────────

class _Settings:
    slurm_token_mode = "static"
    slurm_jwt = "abc"
    slurm_token_file = ""
    slurm_token_command = ""
    slurm_token_refresh_margin_seconds = 300.0
    slurm_token_timeout_seconds = 30.0
    slurm_token_lifespan_seconds = 3600
    slurm_token_ssh_host = ""
    slurm_token_ssh_user = ""
    slurm_token_ssh_key = ""
    slurm_token_ssh_port = 0
    slurm_token_ssh_known_hosts = ""
    slurm_rest_user = None


def test_static_mode_is_the_default():
    assert build_provider(_Settings()).get() == "abc"


def test_file_mode_requires_a_path():
    s = _Settings()
    s.slurm_token_mode = "file"
    with pytest.raises(TokenError, match="SLURM_TOKEN_FILE"):
        build_provider(s)


def test_command_mode_requires_a_command_or_ssh_host():
    s = _Settings()
    s.slurm_token_mode = "command"
    with pytest.raises(TokenError, match="SLURM_TOKEN_COMMAND"):
        build_provider(s)


def test_ssh_host_alone_is_enough_to_build_the_command():
    s = _Settings()
    s.slurm_token_mode = "command"
    s.slurm_token_ssh_host = "titan"
    s.slurm_token_ssh_user = "svc-tokens"
    assert build_provider(s) is not None


def test_the_key_path_is_redacted_from_logs(caplog):
    s = _Settings()
    s.slurm_token_mode = "command"
    s.slurm_token_ssh_host = "titan"
    s.slurm_token_ssh_key = "/run/secrets/very-secret-key"
    with caplog.at_level("INFO"):
        build_provider(s)
    joined = " ".join(r.message for r in caplog.records)
    assert "very-secret-key" not in joined
    assert "<key>" in joined


@pytest.fixture(autouse=True)
def _reset():
    yield
    tokens.set_provider(None)


def test_ssh_command_supports_a_nonstandard_port():
    cmd = ssh_command("gateway", port=12222)
    assert "-p 12222" in cmd
