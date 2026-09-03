"""The dashboard payload the timeline renders from.

Checks the contract the UI depends on: lane offsets, node metadata, foreign
jobs, and the fallback that keeps single-node deployments unchanged.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.admin import router as admin_router
from app.auth import current_user, require_auth
from app.authz import User
from app.backends.types import ForeignJob, GpuGroup, NodeInfo
from app.catalog import CatalogModel
from app.cluster import ClusterConfig, GpuClass, Pool, set_cluster
from app.dependencies import SessionLocal, init_db
from app.inventory import Inventory, set_inventory
from app.models import Lease
from app.settings import settings

NOW = datetime.now(timezone.utc)
ALICE = User(sub="alice", display_name="Alice", is_user=True, via="ldap")

CLUSTER = ClusterConfig(
    gpu_classes={
        "gpu24": GpuClass(name="gpu24", vram_gb=24),
        "gpu48": GpuClass(name="gpu48", vram_gb=48),
        "gpu96": GpuClass(name="gpu96", vram_gb=96),
    },
    pools={
        "llm-dedicated": Pool(name="llm-dedicated", partition="gpu",
                              scheduling="managed", nodes=("jupiter",)),
        "general": Pool(name="general", partition="gpu", scheduling="slurm"),
    },
)

NODES = (
    NodeInfo(name="europa", gpus=(GpuGroup("gpu24", 1), GpuGroup("gpu48", 1)),
             partitions=("gpu",), state="mixed"),
    NodeInfo(name="jupiter", gpus=(GpuGroup("gpu96", 4),), partitions=("gpu",), state="idle"),
    NodeInfo(name="titan", gpus=(GpuGroup("gpu48", 2),), partitions=("gpu",), state="idle"),
)

INVENTORY = Inventory(
    nodes=NODES,
    fetched_at=NOW,
    node_pools={n.name: CLUSTER.pool_for_node(n.name, n.partitions).name for n in NODES},
)

CATALOG = {"test-model": CatalogModel(name="test-model", model_path="p",
                                      gpus=2, tensor_parallel_size=2)}


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

    set_cluster(None)
    set_inventory(None)
    with SessionLocal() as db:
        for lease in db.query(Lease).all():
            db.delete(lease)
        db.commit()


def _book(gpu_class="gpu96", gpus=2, start_h=1, hours=2, state="PLANNED"):
    with SessionLocal() as db:
        lease = Lease(
            model="test-model", requested_gpus=gpus, requested_tp=gpus,
            requested_port=0, model_path="p", state=state, owner_sub="alice",
            created_at=NOW, begin_at=NOW + timedelta(hours=start_h),
            end_at=NOW + timedelta(hours=start_h + hours),
            gpu_class=gpu_class, pool="llm-dedicated",
        )
        db.add(lease)
        db.commit()
        return lease.id


def _dash(client):
    res = client.get("/admin/dashboard")
    assert res.status_code == 200
    return res.json()


# ── Node layout ──────────────────────────────────────────────────────────────

def test_dashboard_lists_nodes_with_stable_lane_offsets(client):
    nodes = _dash(client)["nodes"]
    assert [(n["name"], n["lane_offset"]) for n in nodes] == [
        ("europa", 0), ("jupiter", 2), ("titan", 6),
    ]


def test_total_gpus_is_derived_from_inventory(client):
    assert _dash(client)["total_gpus"] == 2 + 4 + 2


def test_node_reports_its_mixed_gpu_classes(client):
    europa = next(n for n in _dash(client)["nodes"] if n["name"] == "europa")
    assert europa["gpu_classes"] == [["gpu24", 1], ["gpu48", 1]]


def test_nodes_carry_their_pool(client):
    nodes = {n["name"]: n["pool"] for n in _dash(client)["nodes"]}
    assert nodes["jupiter"] == "llm-dedicated"
    assert nodes["titan"] == "general"


def test_discovered_nodes_are_not_synthetic(client):
    assert all(not n["synthetic"] for n in _dash(client)["nodes"])


# ── Lease placement in the payload ───────────────────────────────────────────

def test_booking_lands_on_a_node_and_gets_a_global_lane(client):
    lease_id = _book(gpu_class="gpu96", gpus=2)
    lease = next(x for x in _dash(client)["leases"] if x["id"] == lease_id)

    assert lease["node"] == "jupiter"
    assert lease["gpu_start"] == 0
    # jupiter's slice starts at lane 2.
    assert lease["lane_start"] == 2
    assert lease["lane_count"] == 2
    assert lease["conflict"] is False


def test_second_booking_stacks_within_the_same_node(client):
    _book(gpu_class="gpu96", gpus=2)
    second = _book(gpu_class="gpu96", gpus=2)
    lease = next(x for x in _dash(client)["leases"] if x["id"] == second)

    assert lease["node"] == "jupiter"
    assert lease["lane_start"] == 4      # offset 2 + gpu index 2


def test_unplaceable_booking_is_flagged_as_conflicting(client):
    lease_id = _book(gpu_class="gpu96", gpus=8)   # no node has 8 × gpu96
    lease = next(x for x in _dash(client)["leases"] if x["id"] == lease_id)

    assert lease["conflict"] is True
    assert lease["lane_start"] is None


def test_gpu_class_is_surfaced_on_the_lease(client):
    lease_id = _book(gpu_class="gpu48", gpus=2)
    lease = next(x for x in _dash(client)["leases"] if x["id"] == lease_id)
    assert lease["gpu_class"] == "gpu48"
    assert lease["node"] == "titan"


# ── Foreign jobs ─────────────────────────────────────────────────────────────

def test_foreign_jobs_are_exposed_to_the_timeline(client):
    set_inventory(Inventory(
        nodes=NODES, fetched_at=NOW, node_pools=INVENTORY.node_pools,
        foreign_jobs=(ForeignJob(
            job_id="9001", user="someone-else", state="RUNNING",
            nodes=("jupiter",), gpus=2,
            start_time=NOW, end_time=NOW + timedelta(hours=6),
        ),),
    ))
    payload = _dash(client)
    assert len(payload["foreign_jobs"]) == 1
    assert payload["foreign_jobs"][0]["user"] == "someone-else"
    assert payload["foreign_jobs"][0]["nodes"] == ["jupiter"]


def test_foreign_job_pushes_our_booking_onto_other_gpus(client):
    set_inventory(Inventory(
        nodes=NODES, fetched_at=NOW, node_pools=INVENTORY.node_pools,
        foreign_jobs=(ForeignJob(
            job_id="9001", user="someone-else", state="RUNNING",
            nodes=("jupiter",), gpus=2,
            start_time=NOW, end_time=NOW + timedelta(hours=6),
        ),),
    ))
    lease_id = _book(gpu_class="gpu96", gpus=2)
    lease = next(x for x in _dash(client)["leases"] if x["id"] == lease_id)

    # The foreign job holds jupiter's first two GPUs.
    assert lease["node"] == "jupiter"
    assert lease["gpu_start"] == 2


# ── Fallback for deployments with no inventory ───────────────────────────────

def test_no_inventory_falls_back_to_a_single_synthetic_node(client, monkeypatch):
    monkeypatch.setattr(settings, "total_gpus", 8)
    set_cluster(ClusterConfig())
    set_inventory(None)

    payload = _dash(client)

    assert payload["total_gpus"] == 8
    assert len(payload["nodes"]) == 1
    assert payload["nodes"][0]["synthetic"] is True


def test_fallback_places_bookings_on_flat_lanes(client, monkeypatch):
    """Identical to the pre-multi-node behaviour, which is the migration's
    whole safety argument."""
    monkeypatch.setattr(settings, "total_gpus", 8)
    set_cluster(ClusterConfig())
    set_inventory(None)

    first = _book(gpu_class=None, gpus=4)
    second = _book(gpu_class=None, gpus=4)
    leases = {x["id"]: x for x in _dash(client)["leases"]}

    assert leases[first]["lane_start"] == 0
    assert leases[second]["lane_start"] == 4


def test_inventory_error_is_surfaced(client):
    set_inventory(Inventory(
        nodes=NODES, fetched_at=NOW, node_pools=INVENTORY.node_pools,
        error="Slurm controller unavailable",
    ))
    assert "unavailable" in _dash(client)["inventory_error"]


def test_healthy_inventory_reports_no_error(client):
    assert _dash(client)["inventory_error"] is None
