"""The ORM <-> placer bridge, and legacy-behaviour equivalence.

The migration's safety argument is that the old flat-lane model *is* "one node
with N untyped GPUs". These tests hold that claim to account: with no
cluster.yaml and no discovered inventory, placement must reproduce what the
retired `planner.py` did — including the three regressions its own tests
guarded.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.backends.types import ForeignJob, GpuGroup, NodeInfo
from app.cluster import ClusterConfig, GpuClass, Pool, set_cluster
from app.inventory import Inventory
from app.scheduling import (
    SYNTHETIC_NODE,
    active_leases,
    effective_nodes,
    lane_index,
    lease_to_demand,
    node_lanes,
    plan,
)
from app.settings import settings

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
NO_CLUSTER = ClusterConfig()


def _lease(id, begin, end, gpus=1, state="PLANNED", gpu_class=None, node=None):
    return SimpleNamespace(
        id=id, state=state, begin_at=begin, end_at=end, created_at=begin,
        requested_gpus=gpus, gpu_class=gpu_class, node=node, model=f"m{id}",
    )


@pytest.fixture(autouse=True)
def _reset():
    yield
    set_cluster(None)


# ── Synthetic node: legacy equivalence ───────────────────────────────────────

def test_no_inventory_yields_one_synthetic_node_sized_by_total_gpus(monkeypatch):
    monkeypatch.setattr(settings, "total_gpus", 8)
    nodes = effective_nodes(Inventory())
    assert len(nodes) == 1
    assert nodes[0].name == SYNTHETIC_NODE
    assert nodes[0].gpu_count == 8


def test_synthetic_gpus_are_untyped_so_any_demand_matches(monkeypatch):
    monkeypatch.setattr(settings, "total_gpus", 4)
    assert effective_nodes(Inventory())[0].gpus == (GpuGroup(None, 4),)


def test_discovered_inventory_replaces_the_synthetic_node():
    inv = Inventory(nodes=(NodeInfo(name="titan", gpus=(GpuGroup("gpu48", 2),),
                                    state="idle"),))
    assert [n.name for n in effective_nodes(inv)] == ["titan"]


def _legacy_plan(leases, total_gpus, monkeypatch):
    monkeypatch.setattr(settings, "total_gpus", total_gpus)
    placements, lanes = plan(leases, Inventory(), NO_CLUSTER)
    return placements, lanes


# The three regressions the retired planner's tests guarded.

def test_future_planned_lease_still_gets_a_lane(monkeypatch):
    """Regression: leases far in the future used to fall outside the placement
    horizon, serialize as lane_start=None, and collapse onto lane 0."""
    begin = NOW + timedelta(days=5)
    lease = _lease(1, begin, begin + timedelta(hours=2), gpus=4)

    placements, lanes = _legacy_plan([lease], 8, monkeypatch)

    assert not placements[1].conflict
    assert lane_index(lanes, placements[1]) == 0
    assert placements[1].gpu_count == 4


def test_two_concurrent_leases_stack_on_distinct_lanes(monkeypatch):
    begin = NOW + timedelta(days=5)
    end = begin + timedelta(hours=2)
    placements, lanes = _legacy_plan(
        [_lease(1, begin, end, gpus=4), _lease(2, begin, end, gpus=4)], 8, monkeypatch
    )
    assert lane_index(lanes, placements[1]) == 0
    assert lane_index(lanes, placements[2]) == 4
    assert not placements[1].conflict and not placements[2].conflict


def test_overlapping_leases_beyond_capacity_conflict(monkeypatch):
    begin = NOW + timedelta(days=5)
    end = begin + timedelta(hours=2)
    leases = [_lease(i, begin, end, gpus=4) for i in (1, 2, 3)]

    placements, _ = _legacy_plan(leases, 8, monkeypatch)

    assert sum(1 for p in placements.values() if p.conflict) == 1
    assert sum(1 for p in placements.values() if not p.conflict) == 2


def test_back_to_back_bookings_remain_allowed_by_default(monkeypatch):
    """The old OVERLAP_TOLERANCE permitted "A ends 18:00, B starts 18:00".
    The default guard gap is 0 precisely to preserve that."""
    a = _lease(1, NOW, NOW + timedelta(hours=1), gpus=8)
    b = _lease(2, NOW + timedelta(hours=1), NOW + timedelta(hours=2), gpus=8)

    placements, _ = _legacy_plan([a, b], 8, monkeypatch)

    assert not placements[1].conflict
    assert not placements[2].conflict


# ── Lane index across several nodes ──────────────────────────────────────────

MULTI = Inventory(nodes=(
    NodeInfo(name="europa", gpus=(GpuGroup("gpu24", 1), GpuGroup("gpu48", 1)), state="idle"),
    NodeInfo(name="jupiter", gpus=(GpuGroup("gpu96", 4),), state="idle"),
    NodeInfo(name="titan", gpus=(GpuGroup("gpu48", 2),), state="idle"),
))


def test_lane_offsets_are_assigned_in_stable_name_order():
    lanes = node_lanes(effective_nodes(MULTI), MULTI)
    assert [(x.name, x.lane_offset) for x in lanes] == [
        ("europa", 0), ("jupiter", 2), ("titan", 6),
    ]


def test_lane_index_maps_node_plus_gpu_to_a_global_row():
    cluster = ClusterConfig(gpu_classes={"gpu96": GpuClass(name="gpu96", vram_gb=96)})
    lease = _lease(1, NOW, NOW + timedelta(hours=1), gpus=2, gpu_class="gpu96")

    placements, lanes = plan([lease], MULTI, cluster)

    # jupiter starts at lane 2; the booking takes its first two GPUs.
    assert placements[1].node == "jupiter"
    assert lane_index(lanes, placements[1]) == 2


def test_unplaced_lease_has_no_lane_index():
    cluster = ClusterConfig(gpu_classes={"gpu96": GpuClass(name="gpu96", vram_gb=96)})
    lease = _lease(1, NOW, NOW + timedelta(hours=1), gpus=8, gpu_class="gpu96")

    placements, lanes = plan([lease], MULTI, cluster)

    assert placements[1].conflict
    assert lane_index(lanes, placements[1]) is None


def test_drained_node_is_excluded_from_the_lane_layout():
    inv = Inventory(nodes=(
        NodeInfo(name="alien0", gpus=(GpuGroup("gpu24", 1),), state="drain"),
        NodeInfo(name="titan", gpus=(GpuGroup("gpu48", 2),), state="idle"),
    ))
    assert [x.name for x in node_lanes(effective_nodes(inv), inv)] == ["titan"]


# ── Demands from leases ──────────────────────────────────────────────────────

def test_running_lease_is_pinned_to_where_it_actually_landed():
    """Re-planning must not 'move' a model that is already serving traffic."""
    lease = _lease(1, NOW, NOW + timedelta(hours=1), state="RUNNING", node="titan")
    assert lease_to_demand(lease).pinned_node == "titan"


def test_planned_lease_is_not_pinned():
    lease = _lease(1, NOW, NOW + timedelta(hours=1), state="PLANNED", node="titan")
    assert lease_to_demand(lease).pinned_node is None


def test_lease_without_begin_falls_back_to_created_at():
    lease = SimpleNamespace(
        id=1, state="PLANNED", begin_at=None, end_at=NOW + timedelta(hours=1),
        created_at=NOW, requested_gpus=1, gpu_class=None, node=None, model="m",
    )
    assert lease_to_demand(lease).begin == NOW


def test_lease_without_end_gets_an_hour():
    lease = SimpleNamespace(
        id=1, state="PLANNED", begin_at=NOW, end_at=None, created_at=NOW,
        requested_gpus=1, gpu_class=None, node=None, model="m",
    )
    demand = lease_to_demand(lease)
    assert demand.end - demand.begin == timedelta(hours=1)


# ── Which leases occupy GPUs ─────────────────────────────────────────────────

@pytest.mark.parametrize("state", ["PLANNED", "SUBMITTED", "STARTING", "RUNNING"])
def test_active_states_occupy_gpus(state):
    lease = _lease(1, NOW, NOW + timedelta(hours=1), state=state)
    assert active_leases([lease], NOW) == [lease]


@pytest.mark.parametrize("state", ["CANCELED", "ENDED"])
def test_finished_states_release_gpus(state):
    lease = _lease(1, NOW, NOW + timedelta(hours=1), state=state)
    assert active_leases([lease], NOW) == []


def test_failed_lease_holds_its_slot_until_the_window_expires():
    """It may still retry, so the reservation must not be handed away."""
    lease = _lease(1, NOW, NOW + timedelta(hours=1), state="FAILED")
    assert active_leases([lease], NOW) == [lease]


def test_failed_lease_past_its_window_releases():
    lease = _lease(1, NOW - timedelta(hours=3), NOW - timedelta(hours=1), state="FAILED")
    assert active_leases([lease], NOW) == []


# ── Foreign jobs reach the plan ──────────────────────────────────────────────

def test_foreign_jobs_from_inventory_block_placement():
    cluster = ClusterConfig(gpu_classes={"gpu96": GpuClass(name="gpu96", vram_gb=96)})
    inv = Inventory(
        nodes=MULTI.nodes,
        foreign_jobs=(ForeignJob(
            job_id="1", user="someone", state="RUNNING", nodes=("jupiter",), gpus=4,
            start_time=NOW, end_time=NOW + timedelta(hours=6),
        ),),
    )
    lease = _lease(1, NOW + timedelta(hours=1), NOW + timedelta(hours=2),
                   gpus=4, gpu_class="gpu96")

    placements, _ = plan([lease], inv, cluster)

    assert placements[1].conflict


def test_foreign_jobs_can_be_excluded_for_a_managed_view():
    cluster = ClusterConfig(gpu_classes={"gpu96": GpuClass(name="gpu96", vram_gb=96)})
    inv = Inventory(
        nodes=MULTI.nodes,
        foreign_jobs=(ForeignJob(
            job_id="1", user="someone", state="RUNNING", nodes=("jupiter",), gpus=4,
            start_time=NOW, end_time=NOW + timedelta(hours=6),
        ),),
    )
    lease = _lease(1, NOW + timedelta(hours=1), NOW + timedelta(hours=2),
                   gpus=4, gpu_class="gpu96")

    placements, _ = plan([lease], inv, cluster, include_foreign=False)

    assert not placements[1].conflict


# ── Pool metadata on lanes ───────────────────────────────────────────────────

def test_lanes_carry_the_pool_each_node_belongs_to():
    cluster = ClusterConfig(pools={
        "llm-dedicated": Pool(name="llm-dedicated", partition="gpu",
                              scheduling="managed", nodes=("jupiter",)),
        "general": Pool(name="general", partition="gpu"),
    })
    nodes = tuple(
        NodeInfo(name=n, gpus=(GpuGroup("gpu96", 4),), partitions=("gpu",), state="idle")
        for n in ("jupiter", "titan")
    )
    inv = Inventory(
        nodes=nodes,
        node_pools={n.name: cluster.pool_for_node(n.name, n.partitions).name for n in nodes},
    )
    lanes = {x.name: x.pool for x in node_lanes(effective_nodes(inv), inv)}
    assert lanes == {"jupiter": "llm-dedicated", "titan": "general"}
