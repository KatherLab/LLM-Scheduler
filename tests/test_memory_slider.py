"""Per-booking GPU memory override.

Resolution order, most specific first — and absolute GB is preferred at every
layer, because a fraction means different things on a 48 GB card and a 128 GB
Spark where the memory is shared with the OS.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.admin import router as admin_router
from app.auth import current_user, require_auth
from app.authz import User
from app.backends.types import GpuGroup, NodeInfo
from app.catalog import CatalogModel
from app.cluster import ClusterConfig, GpuClass, Pool, set_cluster
from app.dependencies import SessionLocal, init_db
from app.inventory import Inventory, set_inventory
from app.models import Lease

NOW = datetime.now(timezone.utc)
ALICE = User(sub="alice", display_name="Alice", is_user=True, via="ldap")

CLUSTER = ClusterConfig(
    gpu_classes={
        "gpu48": GpuClass(name="gpu48", vram_gb=48),
        "gb10": GpuClass(name="gb10", vram_gb=128, arch="aarch64",
                         unified_memory=True, reserved_gb=40,
                         gpu_memory_utilization_max=0.70),
    },
    pools={"general": Pool(name="general", partition="gpu", scheduling="slurm")},
)
NODES = (
    NodeInfo(name="titan", gpus=(GpuGroup("gpu48", 4),), partitions=("gpu",), state="idle"),
)
INVENTORY = Inventory(nodes=NODES, fetched_at=NOW, node_pools={"titan": "general"})

CATALOG = {
    # Declares an absolute budget — the slider seeds from this.
    "bge-m3": CatalogModel(name="bge-m3", model_path="BAAI/bge-m3", gpus=1,
                           tensor_parallel_size=1, memory_gb=12),
    # Only a fraction — no meaningful slider range, so the UI hides it.
    "legacy": CatalogModel(name="legacy", model_path="org/legacy", gpus=1,
                           tensor_parallel_size=1, gpu_memory_utilization=0.85),
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.admin.get_catalog", lambda *a, **k: CATALOG)
    monkeypatch.setattr("app.admin._submit_to_slurm", lambda lease: "job-1")
    set_cluster(CLUSTER)
    set_inventory(INVENTORY)
    init_db()

    app = FastAPI()
    app.include_router(admin_router)
    app.dependency_overrides[current_user] = lambda: ALICE
    app.dependency_overrides[require_auth] = lambda: {"sub": "alice"}

    yield TestClient(app)

    set_cluster(None)
    set_inventory(None)
    with SessionLocal() as db:
        for lease in db.query(Lease).all():
            db.delete(lease)
        db.commit()


def _book(client, **kw):
    body = {"model": "bge-m3", "duration_seconds": 3600,
            "begin_at": (NOW + timedelta(days=1)).isoformat()}
    body.update(kw)
    res = client.post("/admin/leases", json=body)
    assert res.status_code == 200, res.text
    with SessionLocal() as db:
        return db.get(Lease, res.json()["id"])


# ── Resolution order ─────────────────────────────────────────────────────────

def test_running_alone_takes_the_whole_card(client):
    """`memory_gb` is a *sharing* budget. A model running alone should still
    get the greedy default — capping it at 12 GB on a 48 GB card would waste
    the rest and starve the KV cache."""
    lease = _book(client)
    assert float(lease.gpu_memory_utilization) == pytest.approx(0.95)


def test_memory_gb_still_bounds_which_gpus_are_possible(client):
    """It doubles as the implied minimum, so it need not be repeated under
    `requires`."""
    from app.cluster import GpuClass
    model = CATALOG["bge-m3"]
    assert model.min_vram_gb == 12
    assert model.supports_class(GpuClass(name="gpu48", vram_gb=48))
    assert not model.supports_class(GpuClass(name="tiny", vram_gb=8))


def test_slider_override_wins_over_the_catalog(client):
    lease = _book(client, memory_gb=24)
    assert float(lease.gpu_memory_utilization) == pytest.approx(24 / 48, abs=1e-3)


def test_absolute_memory_wins_over_an_explicit_fraction(client):
    """Both supplied: GB is the less ambiguous unit, so it takes precedence."""
    lease = _book(client, memory_gb=24, gpu_memory_utilization=0.9)
    assert float(lease.gpu_memory_utilization) == pytest.approx(24 / 48, abs=1e-3)


def test_explicit_fraction_still_works_without_a_budget(client):
    lease = _book(client, gpu_memory_utilization=0.5)
    assert float(lease.gpu_memory_utilization) == pytest.approx(0.5)


def test_model_without_a_budget_falls_back_to_its_fraction(client):
    lease = _book(client, model="legacy")
    assert float(lease.gpu_memory_utilization) == pytest.approx(0.85)


# ── Class caps still apply ───────────────────────────────────────────────────

def test_the_class_ceiling_caps_a_greedy_slider_value():
    """A user dragging to the maximum must not be able to destabilise a Spark."""
    cls = CLUSTER.gpu_classes["gb10"]
    assert cls.utilization_for_gb(120) == 0.70


def test_the_same_gb_yields_a_different_fraction_per_class():
    """Precisely why the slider is in GB and not a percentage."""
    on48 = CLUSTER.gpu_classes["gpu48"].utilization_for_gb(24)
    on_spark = CLUSTER.gpu_classes["gb10"].utilization_for_gb(24)
    assert on48 == pytest.approx(0.5)
    assert on_spark == pytest.approx(24 / 128)


def test_slider_cannot_exceed_the_card(client):
    """Asking for more than the GPU has is capped, not passed through as >1."""
    lease = _book(client, memory_gb=200)
    assert float(lease.gpu_memory_utilization) <= 1.0


# ── Dashboard metadata the slider needs ──────────────────────────────────────

def test_dashboard_exposes_the_model_memory_default(client):
    payload = client.get("/admin/dashboard").json()
    meta = next(m for m in payload["models"] if m["id"] == "bge-m3")["meta"]
    assert meta["memory_gb"] == 12


def test_model_without_a_budget_reports_none_so_the_slider_hides(client):
    payload = client.get("/admin/dashboard").json()
    meta = next(m for m in payload["models"] if m["id"] == "legacy")["meta"]
    assert meta["memory_gb"] is None


def test_dashboard_exposes_gpu_class_bounds(client):
    payload = client.get("/admin/dashboard").json()
    classes = {c["name"]: c for c in payload["gpu_classes"]}

    assert classes["gpu48"]["vram_gb"] == 48
    # Spark: 128 GB raw, but most of it belongs to the host.
    assert classes["gb10"]["usable_gb"] < 128
    assert classes["gb10"]["unified_memory"] is True


def test_gpu_classes_are_ordered_by_size(client):
    payload = client.get("/admin/dashboard").json()
    sizes = [c["vram_gb"] for c in payload["gpu_classes"]]
    assert sizes == sorted(sizes)
