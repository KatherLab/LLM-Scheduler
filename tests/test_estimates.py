"""Start-time estimates for Slurm-scheduled pools.

The rule under test: a booking on a `slurm` pool is never presented as
confirmed. It is either Slurm's live estimate, or explicitly unknown — the app
must not invent a start time it cannot deliver.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.admin import _scheduling_mode, router as admin_router
from app.auth import current_user, require_auth
from app.authz import User
from app.backends import set_backend, set_estimate_backend
from app.backends.types import (
    CAP_TEST_ONLY,
    ClusterUnavailableError,
    GpuGroup,
    NodeInfo,
    StartEstimate,
)
from app.catalog import CatalogModel
from app.cluster import ClusterConfig, GpuClass, Pool, set_cluster
from app.dependencies import SessionLocal, init_db
from app.inventory import Inventory, set_inventory
from app.models import Lease

NOW = datetime.now(timezone.utc)
ALICE = User(sub="alice", display_name="Alice", is_user=True, via="ldap")

CLUSTER = ClusterConfig(
    gpu_classes={"gpu96": GpuClass(name="gpu96", vram_gb=96),
                 "gpu48": GpuClass(name="gpu48", vram_gb=48)},
    pools={
        "llm-dedicated": Pool(name="llm-dedicated", partition="gpu",
                              scheduling="managed", nodes=("jupiter",)),
        "general": Pool(name="general", partition="gpu", scheduling="slurm"),
    },
)
# jupiter is carved out into the dedicated pool by its explicit node list;
# titan is left to the partition-wide `general` pool.
NODES = (
    NodeInfo(name="jupiter", gpus=(GpuGroup("gpu96", 4),), partitions=("gpu",), state="idle"),
    NodeInfo(name="titan", gpus=(GpuGroup("gpu48", 4),), partitions=("gpu",), state="idle"),
)
INVENTORY = Inventory(
    nodes=NODES, fetched_at=NOW,
    node_pools={n.name: CLUSTER.pool_for_node(n.name, n.partitions).name for n in NODES},
)
CATALOG = {"m": CatalogModel(name="m", model_path="p", gpus=2, tensor_parallel_size=2)}


class NoEstimatePrimary:
    """A primary backend that cannot estimate — like slurmrestd.

    Needed because `get_estimate_backend()` prefers the primary when it already
    has CAP_TEST_ONLY, and LocalBackend does.
    """

    name = "no-estimate"
    capabilities = frozenset()


class FakeEstimator:
    """Stands in for a backend that can run `sbatch --test-only`."""

    name = "fake-estimator"
    capabilities = frozenset({CAP_TEST_ONLY})

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def estimate_start(self, spec):
        self.calls.append(spec)
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.admin.get_catalog", lambda *a, **k: CATALOG)
    set_cluster(CLUSTER)
    set_inventory(INVENTORY)
    set_backend(NoEstimatePrimary())
    set_estimate_backend(None)
    init_db()

    app = FastAPI()
    app.include_router(admin_router)
    app.dependency_overrides[current_user] = lambda: ALICE
    app.dependency_overrides[require_auth] = lambda: {"sub": "alice"}

    yield TestClient(app)

    set_cluster(None)
    set_inventory(None)
    set_estimate_backend(None)
    set_backend(None)
    with SessionLocal() as db:
        for lease in db.query(Lease).all():
            db.delete(lease)
        db.commit()


def _preview(client, **kw):
    body = {"model": "m", "duration_seconds": 3600}
    body.update(kw)
    res = client.post("/admin/leases/preview", json=body)
    assert res.status_code == 200, res.text
    return res.json()


# ── Scheduling mode ──────────────────────────────────────────────────────────

def test_managed_pool_is_a_promise(client):
    assert _scheduling_mode("llm-dedicated") == "managed"


def test_slurm_pool_is_only_an_estimate(client):
    assert _scheduling_mode("general") == "slurm"


def test_unknown_pool_with_a_configured_cluster_is_not_a_promise(client):
    """We never told Slurm anything about it, so we cannot promise it."""
    assert _scheduling_mode(None) == "slurm"


def test_legacy_deployment_without_pools_stays_managed():
    """No cluster.yaml means the old single-node behaviour, which did promise."""
    set_cluster(ClusterConfig())
    try:
        assert _scheduling_mode(None) == "managed"
    finally:
        set_cluster(None)


# ── Preview on a managed pool ────────────────────────────────────────────────

def test_managed_pool_preview_is_confirmed(client):
    out = _preview(client, pool="llm-dedicated")
    assert out["confidence"] == "confirmed"
    assert out["start_at"] is not None
    assert out["node"] == "jupiter"
    assert out["gpu_class"] == "gpu96"


def test_managed_preview_does_not_consult_the_estimator(client):
    """Our own calendar is authoritative there; asking Slurm would be noise."""
    fake = FakeEstimator(StartEstimate(start_time=NOW))
    set_estimate_backend(fake)
    _preview(client, pool="llm-dedicated")
    assert fake.calls == []


def test_managed_preview_reports_impossible_when_nothing_fits(client):
    """4 GPUs exist on jupiter; a 2-GPU model fits, but not five bookings."""
    with SessionLocal() as db:
        for i in range(2):
            db.add(Lease(
                model="m", requested_gpus=2, requested_tp=2, requested_port=0,
                model_path="p", state="RUNNING", owner_sub="bob", created_at=NOW,
                begin_at=NOW, end_at=NOW + timedelta(days=30),
                gpu_class="gpu96", pool="llm-dedicated", node="jupiter",
            ))
        db.commit()
    out = _preview(client, pool="llm-dedicated")
    assert out["confidence"] == "impossible"


# ── Preview on a Slurm pool ──────────────────────────────────────────────────

def test_slurm_pool_preview_is_estimated_not_confirmed(client):
    when = NOW + timedelta(hours=5)
    set_estimate_backend(FakeEstimator(
        StartEstimate(start_time=when, nodes=("jupiter",), raw="Job 1 to start at ...")
    ))
    out = _preview(client, pool="general")

    assert out["confidence"] == "estimated"
    assert out["start_at"].startswith(when.strftime("%Y-%m-%dT%H:%M"))
    assert "not a reservation" in out["detail"]


def test_slurm_preview_uses_the_spec_we_would_really_submit(client):
    """Otherwise the estimate describes a different job than the user gets."""
    fake = FakeEstimator(StartEstimate(start_time=NOW))
    set_estimate_backend(fake)
    _preview(client, pool="general")

    spec = fake.calls[0]
    assert spec.gres == "gpu:gpu48:2"
    assert spec.partition == "gpu"


def test_no_estimator_reports_unknown_rather_than_guessing(client):
    """The off-cluster REST deployment: honest beats plausible."""
    set_estimate_backend(None)
    out = _preview(client, pool="general")

    assert out["confidence"] == "unknown"
    assert out["start_at"] is None
    assert "Slurm decides" in out["detail"]


def test_estimator_failure_degrades_to_unknown(client):
    set_estimate_backend(FakeEstimator(error=ClusterUnavailableError("controller down")))
    out = _preview(client, pool="general")
    assert out["confidence"] == "unknown"
    assert "controller down" in out["detail"]


def test_backfill_with_no_opinion_yet_is_unknown(client):
    """`--test-only` succeeded but Slurm gave no time — do not fabricate one."""
    set_estimate_backend(FakeEstimator(StartEstimate(start_time=None, raw="queued")))
    out = _preview(client, pool="general")
    assert out["confidence"] == "unknown"
    assert out["start_at"] is None


def test_preview_rejects_an_unknown_model(client):
    res = client.post("/admin/leases/preview", json={"model": "nope"})
    assert res.status_code == 404


def test_preview_rejects_an_unknown_pool(client):
    res = client.post("/admin/leases/preview", json={"model": "m", "pool": "nope"})
    assert res.status_code == 404


# ── Estimates on existing bookings ───────────────────────────────────────────

def _book(pool, state="SUBMITTED", estimated_start=None):
    with SessionLocal() as db:
        lease = Lease(
            model="m", requested_gpus=2, requested_tp=2, requested_port=0,
            model_path="p", state=state, owner_sub="alice", created_at=NOW,
            begin_at=NOW + timedelta(hours=1), end_at=NOW + timedelta(hours=3),
            gpu_class="gpu96", pool=pool, slurm_job_id="1234",
            estimated_start=estimated_start,
            estimate_updated_at=NOW if estimated_start else None,
        )
        db.add(lease)
        db.commit()
        return lease.id


def test_lease_on_a_slurm_pool_is_marked_as_estimated(client):
    lease_id = _book("general")
    lease = next(x for x in client.get("/admin/leases").json() if x["id"] == lease_id)
    assert lease["scheduling"] == "slurm"


def test_lease_on_a_managed_pool_is_not(client):
    lease_id = _book("llm-dedicated")
    lease = next(x for x in client.get("/admin/leases").json() if x["id"] == lease_id)
    assert lease["scheduling"] == "managed"


def test_backfill_estimate_is_surfaced_to_the_timeline(client):
    when = NOW + timedelta(hours=9)
    lease_id = _book("general", estimated_start=when)
    lease = next(x for x in client.get("/admin/leases").json() if x["id"] == lease_id)

    assert lease["estimated_start"] is not None
    assert lease["estimated_start"].startswith(when.strftime("%Y-%m-%dT%H:%M"))
    assert lease["estimate_updated_at"] is not None


def test_lease_without_an_estimate_reports_none(client):
    lease_id = _book("general")
    lease = next(x for x in client.get("/admin/leases").json() if x["id"] == lease_id)
    assert lease["estimated_start"] is None
