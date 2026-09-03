"""Opt-in ("scale-out") pools.

Shared partitions are capacity we can reach but should not default to. The
claim under test is that "not shown" and "not used" are the same switch: a pool
hidden from the timeline must also be invisible to GPU-class selection and to
the placer, or bookings would keep landing on lanes nobody can see.

Note the smaller cards live in the scale-out pool here, which is the real
cluster's shape — `choose_gpu_class` prefers the smallest sufficient class, so
without the exclusion the opt-in pool would be the *default* landing place.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.admin import router as admin_router
from app.auth import current_user, require_auth
from app.authz import User
from app.backends.types import GpuGroup, NodeInfo
from app.catalog import CatalogModel
from app.cluster import ClusterConfig, GpuClass, Pool, load_cluster, set_cluster
from app.dependencies import SessionLocal, init_db
from app.inventory import (
    Inventory,
    choose_gpu_class,
    eligible_gpu_classes,
    set_inventory,
)
from app.models import Lease
from app.scheduling import node_lanes, plan

NOW = datetime.now(timezone.utc)
ALICE = User(sub="alice", display_name="Alice", is_user=True, via="ldap")

CLUSTER = ClusterConfig(
    gpu_classes={
        "gpu24": GpuClass(name="gpu24", vram_gb=24),
        "gpu96": GpuClass(name="gpu96", vram_gb=96),
    },
    pools={
        "llm": Pool(name="llm", partition="llm", scheduling="managed"),
        "general": Pool(name="general", partition="gpu", scheduling="slurm",
                        scale_out=True),
    },
)

# alien0 sorts first alphabetically but is scale-out, so it must land *last*.
NODES = (
    NodeInfo(name="alien0", gpus=(GpuGroup("gpu24", 2),),
             partitions=("gpu",), state="idle"),
    NodeInfo(name="h200", gpus=(GpuGroup("gpu96", 2),),
             partitions=("llm",), state="idle"),
)

INVENTORY = Inventory(
    nodes=NODES, fetched_at=NOW,
    node_pools={"alien0": "general", "h200": "llm"},
)

MODEL = CatalogModel(name="test-model", model_path="p", gpus=1,
                     tensor_parallel_size=1)
CATALOG = {"test-model": MODEL}


@pytest.fixture(autouse=True)
def _reset():
    yield
    set_cluster(None)
    set_inventory(None)


# ── Config ───────────────────────────────────────────────────────────────────

def test_scale_out_defaults_to_false(tmp_path):
    p = tmp_path / "cluster.yaml"
    p.write_text(yaml.safe_dump({
        "pools": [{"name": "llm", "partition": "llm", "scheduling": "managed"}],
    }))
    assert load_cluster(str(p)).pool("llm").scale_out is False


def test_scale_out_is_read_from_the_pool(tmp_path):
    p = tmp_path / "cluster.yaml"
    p.write_text(yaml.safe_dump({
        "pools": [{"name": "general", "partition": "gpu", "scale_out": True}],
    }))
    cluster = load_cluster(str(p))
    assert cluster.pool("general").scale_out is True
    assert cluster.is_scale_out("general") is True
    assert cluster.has_scale_out is True


# ── Candidate nodes ──────────────────────────────────────────────────────────

def test_default_candidates_exclude_scale_out_nodes():
    names = [n.name for n in INVENTORY.candidate_nodes(CLUSTER)]
    assert names == ["h200"]


def test_naming_the_pool_is_the_opt_in():
    names = [
        n.name for n in INVENTORY.candidate_nodes(CLUSTER, CLUSTER.pool("general"))
    ]
    assert names == ["alien0"]


# ── GPU class selection ──────────────────────────────────────────────────────

def test_smallest_class_does_not_win_when_it_is_scale_out_only():
    assert choose_gpu_class(MODEL, CLUSTER, INVENTORY) == "gpu96"


def test_scale_out_class_is_available_once_its_pool_is_named():
    assert choose_gpu_class(
        MODEL, CLUSTER, INVENTORY, pool=CLUSTER.pool("general")
    ) == "gpu24"


def test_asking_for_a_scale_out_class_without_its_pool_is_refused():
    assert choose_gpu_class(
        MODEL, CLUSTER, INVENTORY, preferred="gpu24"
    ) is None


def test_eligible_classes_hide_scale_out_until_asked_for():
    assert eligible_gpu_classes(MODEL, CLUSTER, INVENTORY) == ["gpu96"]
    assert eligible_gpu_classes(
        MODEL, CLUSTER, INVENTORY, include_scale_out=True
    ) == ["gpu24", "gpu96"]


# ── Lane layout ──────────────────────────────────────────────────────────────

def test_scale_out_nodes_sort_to_the_end_of_the_strip():
    lanes = node_lanes(list(NODES), INVENTORY, CLUSTER)
    assert [(x.name, x.lane_offset, x.scale_out) for x in lanes] == [
        ("h200", 0, False),
        ("alien0", 2, True),
    ]


# ── Placement ────────────────────────────────────────────────────────────────

def _lease(id, gpu_class, pool=None, pinned_node=None, gpus=1):
    return Lease(
        id=id, model="test-model", requested_gpus=gpus, requested_tp=1,
        requested_port=0, model_path="p", state="PLANNED", created_at=NOW,
        begin_at=NOW + timedelta(hours=1), end_at=NOW + timedelta(hours=3),
        gpu_class=gpu_class, pool=pool, pinned_node=pinned_node,
    )


def test_a_booking_without_a_pool_is_never_placed_on_scale_out_capacity():
    placements, _ = plan([_lease(1, "gpu24")], INVENTORY, CLUSTER)
    assert placements[1].conflict is True
    assert placements[1].node is None


def test_a_booking_naming_the_pool_lands_there():
    placements, _ = plan([_lease(1, "gpu24", pool="general")], INVENTORY, CLUSTER)
    assert placements[1].node == "alien0"
    assert placements[1].conflict is False


def test_a_pinned_node_is_drawn_on_that_node():
    """Dropping onto a row submits `--nodelist`, so the block has to agree."""
    placements, _ = plan(
        [_lease(1, "gpu96", pinned_node="h200")], INVENTORY, CLUSTER
    )
    assert placements[1].node == "h200"


def test_no_scale_out_pool_means_no_restriction_at_all():
    """The legacy path: without `scale_out` anywhere, placement is unchanged."""
    cluster = ClusterConfig(
        gpu_classes=CLUSTER.gpu_classes,
        pools={
            "llm": CLUSTER.pool("llm"),
            "general": Pool(name="general", partition="gpu", scheduling="slurm"),
        },
    )
    placements, _ = plan([_lease(1, "gpu24")], INVENTORY, cluster)
    assert placements[1].node == "alien0"


# ── API ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.admin.get_catalog", lambda *a, **k: CATALOG)
    set_cluster(CLUSTER)
    set_inventory(INVENTORY)
    init_db()

    app = FastAPI()
    app.include_router(admin_router)
    app.dependency_overrides[current_user] = lambda: ALICE
    app.dependency_overrides[require_auth] = lambda: {"sub": "alice"}

    yield TestClient(app)

    with SessionLocal() as db:
        for lease in db.query(Lease).all():
            db.delete(lease)
        db.commit()


def test_dashboard_reports_the_default_height_separately(client):
    payload = client.get("/admin/dashboard").json()
    assert payload["total_gpus"] == 4
    assert payload["default_gpus"] == 2
    flags = {n["name"]: n["scale_out"] for n in payload["nodes"]}
    assert flags == {"h200": False, "alien0": True}


def test_dashboard_offers_scale_out_classes_as_a_separate_list(client):
    meta = client.get("/admin/dashboard").json()["models"][0]["meta"]
    assert meta["gpu_classes"] == ["gpu96"]
    assert meta["gpu_classes_scale_out"] == ["gpu24"]


def _create(client, **extra):
    body = {
        "model": "test-model", "notes": "alice",
        "begin_at": (NOW + timedelta(hours=2)).isoformat(),
        "duration_seconds": 3600,
    }
    body.update(extra)
    res = client.post("/admin/leases", json=body)
    assert res.status_code == 200, res.text
    return res.json()


def test_a_default_booking_is_a_promise_not_an_estimate(client):
    """Naming no pool now means "the default pools", and those are managed —
    so the timeline must not draw it as a Slurm guess."""
    lease = _create(client)
    assert lease["pool"] is None
    assert lease["gpu_class"] == "gpu96"
    assert lease["scheduling"] == "managed"


def test_a_scale_out_booking_is_still_only_an_estimate(client):
    lease = _create(client, node="alien0")
    assert lease["scheduling"] == "slurm"


def test_booking_a_node_infers_its_pool(client):
    """Dropping onto a scale-out row must carry the pool, or the job would be
    submitted to whatever the global default partition happens to be."""
    res = client.post("/admin/leases", json={
        "model": "test-model",
        "node": "alien0",
        "notes": "alice",
        "begin_at": (NOW + timedelta(hours=2)).isoformat(),
        "duration_seconds": 3600,
    })
    assert res.status_code == 200, res.text
    assert res.json()["pool"] == "general"
    assert res.json()["gpu_class"] == "gpu24"
