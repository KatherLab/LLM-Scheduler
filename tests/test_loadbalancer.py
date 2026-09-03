"""Least-loaded routing across replicas."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.loadbalancer import REGISTRY, LoadRegistry, choose_least_loaded


def _ep(id, host, port=8000):
    return SimpleNamespace(id=id, host=host, port=port, model="m", state="READY")


A, B, C = _ep(1, "gpu01"), _ep(2, "gpu02"), _ep(3, "gpu03")


@pytest.fixture
def reg():
    r = LoadRegistry()
    yield r
    REGISTRY.reset()


def _key(ep):
    return LoadRegistry.key(ep.host, ep.port)


# ── In-flight accounting ─────────────────────────────────────────────────────

def test_acquire_and_release(reg):
    reg.acquire(_key(A))
    reg.acquire(_key(A))
    assert reg.in_flight(_key(A)) == 2
    reg.release(_key(A))
    assert reg.in_flight(_key(A)) == 1


def test_release_without_acquire_never_goes_negative(reg):
    """A restart mid-flight must not leave a replica looking infinitely idle."""
    reg.release(_key(A))
    reg.release(_key(A))
    assert reg.in_flight(_key(A)) == 0


def test_unknown_endpoint_starts_idle(reg):
    assert reg.in_flight("never:seen") == 0


# ── Selection ────────────────────────────────────────────────────────────────

def test_no_endpoints_yields_none(reg):
    assert choose_least_loaded([], reg) is None


def test_single_endpoint_is_chosen(reg):
    assert choose_least_loaded([A], reg) is A


def test_busiest_replica_is_avoided(reg):
    for _ in range(5):
        reg.acquire(_key(A))
    reg.acquire(_key(B))
    assert choose_least_loaded([A, B], reg) is B


def test_traffic_spreads_rather_than_pinning_to_the_newest(reg):
    """The old behaviour returned the newest endpoint, so every request went to
    one replica while the rest idled."""
    picks = set()
    for _ in range(30):
        chosen = choose_least_loaded([A, B, C], reg)
        reg.acquire(_key(chosen))
        picks.add(chosen.id)
    assert picks == {1, 2, 3}


def test_load_is_balanced_across_replicas(reg):
    for _ in range(30):
        reg.acquire(_key(choose_least_loaded([A, B, C], reg)))
    counts = sorted(reg.in_flight(_key(e)) for e in (A, B, C))
    assert counts == [10, 10, 10]


def test_ties_are_broken_randomly_not_deterministically(reg):
    """A burst arriving before any request registers must not all pick the
    same replica."""
    seen = {choose_least_loaded([A, B, C], reg).id for _ in range(40)}
    assert len(seen) > 1


# ── Scraped queue depth ──────────────────────────────────────────────────────

def test_queue_depth_breaks_a_tie_in_inflight(reg):
    """Both idle from our perspective, but one engine is backed up."""
    reg.record_scrape(_key(A), pending=12, running=4)
    reg.record_scrape(_key(B), pending=0, running=1)
    assert choose_least_loaded([A, B], reg) is B


def test_inflight_outranks_queue_depth(reg):
    """Our own dispatch count is exact and current; the scrape is neither."""
    reg.acquire(_key(B))
    reg.record_scrape(_key(A), pending=50, running=50)
    assert choose_least_loaded([A, B], reg) is A


def test_stale_scrape_is_ignored(reg, monkeypatch):
    import app.loadbalancer as mod
    reg.record_scrape(_key(A), pending=99, running=99)
    # Jump past the TTL: a stale queue depth keeps steering traffic away from a
    # replica that has since drained.
    real = mod.time.monotonic
    monkeypatch.setattr(mod.time, "monotonic", lambda: real() + mod.SCRAPE_TTL_SECONDS + 1)
    assert reg.score(_key(A)) == (0, 0)


# ── Draining ─────────────────────────────────────────────────────────────────

def test_draining_endpoint_is_skipped(reg):
    reg.set_draining(_key(A))
    assert choose_least_loaded([A, B], reg) is B


def test_draining_endpoint_is_used_if_it_is_all_there_is(reg):
    """Refusing to serve is worse than using a replica about to be replaced."""
    reg.set_draining(_key(A))
    assert choose_least_loaded([A], reg) is A


def test_draining_is_reversible(reg):
    reg.set_draining(_key(A))
    reg.set_draining(_key(A), False)
    for _ in range(3):
        reg.acquire(_key(B))
    assert choose_least_loaded([A, B], reg) is A


def test_drained_only_when_inflight_reaches_zero(reg):
    reg.acquire(_key(A))
    reg.set_draining(_key(A))
    assert not reg.drained(_key(A))
    reg.release(_key(A))
    assert reg.drained(_key(A))


def test_idle_endpoint_is_not_drained_unless_marked(reg):
    assert not reg.drained(_key(A))


# ── Housekeeping ─────────────────────────────────────────────────────────────

def test_forget_clears_state_so_a_reused_port_starts_clean(reg):
    reg.acquire(_key(A))
    reg.set_draining(_key(A))
    reg.forget(_key(A))
    assert reg.in_flight(_key(A)) == 0
    assert not reg.is_draining(_key(A))


def test_snapshot_reports_all_tracked_endpoints(reg):
    reg.acquire(_key(A))
    reg.record_scrape(_key(B), pending=2, running=1)
    snap = reg.snapshot()
    assert snap[_key(A)]["in_flight"] == 1
    assert snap[_key(B)]["pending"] == 2


# ── Wiring into the proxy ────────────────────────────────────────────────────

async def test_track_proxy_counts_in_flight_per_host_not_per_route():
    """`endpoint_label` is the route ("chat.completions"), identical across
    replicas — keying on it would make every replica look equally loaded."""
    from app.metrics import track_proxy

    REGISTRY.reset()
    async with track_proxy("http://gpu01:8000/v1/chat/completions", model="m") as ctx:
        ctx["status"] = 200
        assert REGISTRY.in_flight("gpu01:8000") == 1
        assert REGISTRY.in_flight("chat.completions") == 0
    assert REGISTRY.in_flight("gpu01:8000") == 0


async def test_track_proxy_releases_on_error():
    from app.metrics import track_proxy

    REGISTRY.reset()
    with pytest.raises(RuntimeError):
        async with track_proxy("http://gpu02:8000/v1/chat/completions", model="m") as ctx:
            ctx["error"] = "RuntimeError"
            raise RuntimeError("upstream died")
    assert REGISTRY.in_flight("gpu02:8000") == 0
