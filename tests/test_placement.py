"""Per-node placement.

Uses this cluster's real topology, so the cases that matter are concrete:
`europa` mixing two GPU classes, `gpu96` existing only on `jupiter`, and
`gpu80` only on `dgx`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.backends.types import ForeignJob, GpuGroup, NodeInfo
from app.cluster import ClusterConfig, GpuClass
from app.placement import Demand, build_slots, compute_placements, find_earliest_slot

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


def _node(name, groups, state="idle", features=()):
    return NodeInfo(
        name=name,
        gpus=tuple(GpuGroup(cls, n) for cls, n in groups),
        features=features,
        partitions=("gpu",),
        state=state,
    )


CLUSTER_NODES = [
    _node("titan", [("gpu48", 2)]),
    _node("europa", [("gpu24", 1), ("gpu48", 1)]),
    _node("jupiter", [("gpu96", 4)]),
    _node("ganymede", [("gpu24", 8)]),
    _node("dgx", [("gpu80", 4)]),
]

# Tiny guard gap keeps the arithmetic in these tests obvious.
CLUSTER = ClusterConfig(gpu_classes={
    name: GpuClass(name=name, vram_gb=vram, guard_gap_seconds=30)
    for name, vram in [("gpu24", 24), ("gpu48", 48), ("gpu80", 80), ("gpu96", 96)]
})


def _demand(lease_id, gpus, gpu_class=None, start_h=1, hours=2, **kw):
    return Demand(
        lease_id=lease_id, gpus=gpus, gpu_class=gpu_class,
        begin=NOW + timedelta(hours=start_h),
        end=NOW + timedelta(hours=start_h + hours),
        **kw,
    )


def _place(demands, nodes=None, **kw):
    return compute_placements(
        demands, nodes if nodes is not None else CLUSTER_NODES, cluster=CLUSTER, **kw
    )


# ── Slot construction ────────────────────────────────────────────────────────

def test_slots_are_per_node_and_per_class():
    slots = build_slots(CLUSTER_NODES)
    assert len(slots) == 2 + 2 + 4 + 8 + 4


def test_mixed_class_node_indexes_each_gpu_distinctly():
    """europa's gpu24 and gpu48 are different GPUs, not interchangeable ones."""
    slots = build_slots([_node("europa", [("gpu24", 1), ("gpu48", 1)])])
    assert [(s.index, s.gpu_class) for s in slots] == [(0, "gpu24"), (1, "gpu48")]


def test_class_filter_selects_only_matching_slots():
    slots = build_slots(CLUSTER_NODES, gpu_class="gpu96")
    assert {s.node for s in slots} == {"jupiter"}
    assert len(slots) == 4


def test_drained_nodes_produce_no_slots():
    """A booking on a drained node is a calendar entry that can never run."""
    nodes = [_node("alien0", [("gpu24", 1)], state="drain")]
    assert build_slots(nodes) == []


# ── Basic placement ──────────────────────────────────────────────────────────

def test_placement_records_node_and_class():
    p = _place([_demand(1, 2, "gpu48")])[1]
    assert p.node == "titan"
    assert p.gpu_class == "gpu48"
    assert p.gpu_count == 2
    assert not p.conflict


def test_gpu96_can_only_land_on_jupiter():
    p = _place([_demand(1, 4, "gpu96")])[1]
    assert p.node == "jupiter"


def test_gpu80_can_only_land_on_dgx():
    p = _place([_demand(1, 2, "gpu80")])[1]
    assert p.node == "dgx"


def test_request_exceeding_any_single_node_conflicts():
    """No node has 8 × gpu48; contiguity is per node, so this cannot be split."""
    p = _place([_demand(1, 8, "gpu48")])[1]
    assert p.conflict
    assert p.node is None


def test_mixed_class_node_cannot_satisfy_two_gpus_of_one_class():
    """europa has 1×gpu24 + 1×gpu48 — that is not 2×gpu48."""
    nodes = [_node("europa", [("gpu24", 1), ("gpu48", 1)])]
    assert _place([_demand(1, 2, "gpu48")], nodes)[1].conflict


def test_mixed_class_node_satisfies_one_gpu_of_each_class():
    nodes = [_node("europa", [("gpu24", 1), ("gpu48", 1)])]
    result = _place([_demand(1, 1, "gpu24"), _demand(2, 1, "gpu48")], nodes)
    assert result[1].gpu_start == 0 and not result[1].conflict
    assert result[2].gpu_start == 1 and not result[2].conflict


# ── Contiguity and packing ───────────────────────────────────────────────────

def test_two_bookings_stack_on_distinct_gpus():
    result = _place([_demand(1, 2, "gpu96"), _demand(2, 2, "gpu96")])
    assert result[1].gpu_indices == (0, 1)
    assert result[2].gpu_indices == (2, 3)


def test_capacity_is_respected_within_a_node():
    """jupiter has 4 × gpu96; three 2-GPU bookings cannot all fit."""
    result = _place([_demand(i, 2, "gpu96") for i in (1, 2, 3)])
    assert sum(1 for p in result.values() if p.conflict) == 1


def test_non_overlapping_bookings_reuse_the_same_gpus():
    result = _place([
        _demand(1, 4, "gpu96", start_h=1, hours=2),
        _demand(2, 4, "gpu96", start_h=5, hours=2),
    ])
    assert not result[1].conflict and not result[2].conflict
    assert result[1].node == result[2].node == "jupiter"


def test_guard_gap_separates_back_to_back_bookings():
    """A gap smaller than the class guard is treated as an overlap: weight
    loading and teardown make a true back-to-back handover unreliable."""
    cluster = ClusterConfig(gpu_classes={
        "gpu96": GpuClass(name="gpu96", vram_gb=96, guard_gap_seconds=600)
    })
    demands = [
        Demand(1, 4, NOW, NOW + timedelta(hours=1), gpu_class="gpu96"),
        # starts 5 minutes after the first ends — inside the 10 minute guard
        Demand(2, 4, NOW + timedelta(hours=1, minutes=5),
               NOW + timedelta(hours=2), gpu_class="gpu96"),
    ]
    result = compute_placements(demands, CLUSTER_NODES, cluster=cluster)
    assert result[2].conflict


def test_a_gap_larger_than_the_guard_is_fine():
    cluster = ClusterConfig(gpu_classes={
        "gpu96": GpuClass(name="gpu96", vram_gb=96, guard_gap_seconds=600)
    })
    demands = [
        Demand(1, 4, NOW, NOW + timedelta(hours=1), gpu_class="gpu96"),
        Demand(2, 4, NOW + timedelta(hours=1, minutes=30),
               NOW + timedelta(hours=3), gpu_class="gpu96"),
    ]
    result = compute_placements(demands, CLUSTER_NODES, cluster=cluster)
    assert not result[2].conflict


# ── Node restriction ─────────────────────────────────────────────────────────

def test_pool_node_list_restricts_placement():
    p = _place([_demand(1, 1, "gpu24", nodes=("ganymede",))])[1]
    assert p.node == "ganymede"


def test_restriction_to_a_node_without_the_class_conflicts():
    assert _place([_demand(1, 1, "gpu96", nodes=("ganymede",))])[1].conflict


def test_pinned_node_is_honoured():
    p = _place([_demand(1, 1, "gpu24", pinned_node="europa")])[1]
    assert p.node == "europa"


# ── Foreign jobs ─────────────────────────────────────────────────────────────

def _foreign(node, gpus, start_h=0, hours=6):
    return ForeignJob(
        job_id="9001", user="someone-else", state="RUNNING", nodes=(node,), gpus=gpus,
        start_time=NOW + timedelta(hours=start_h),
        end_time=NOW + timedelta(hours=start_h + hours),
    )


def test_foreign_job_blocks_the_gpus_it_occupies():
    """The whole point: on a shared partition we do not own what we can see."""
    result = _place([_demand(1, 4, "gpu96")], foreign_jobs=[_foreign("jupiter", 2)])
    assert result[1].conflict


def test_foreign_job_leaves_the_rest_of_the_node_usable():
    result = _place([_demand(1, 2, "gpu96")], foreign_jobs=[_foreign("jupiter", 2)])
    assert not result[1].conflict
    assert result[1].gpu_indices == (2, 3)


def test_foreign_job_on_another_node_is_irrelevant():
    result = _place([_demand(1, 4, "gpu96")], foreign_jobs=[_foreign("ganymede", 8)])
    assert not result[1].conflict


def test_foreign_job_that_ends_before_the_booking_does_not_block():
    result = _place(
        [_demand(1, 4, "gpu96", start_h=10)],
        foreign_jobs=[_foreign("jupiter", 4, start_h=0, hours=2)],
    )
    assert not result[1].conflict


def test_foreign_job_without_an_end_time_still_blocks():
    """A running job of unknown length must block bookings, not vanish."""
    job = ForeignJob(
        job_id="1", user="x", state="RUNNING", nodes=("jupiter",), gpus=4,
        start_time=NOW, end_time=None,
    )
    assert _place([_demand(1, 4, "gpu96")], foreign_jobs=[job])[1].conflict


def test_foreign_job_with_no_times_at_all_is_ignored():
    """Nothing to place it on the timeline with; better than inventing a window."""
    job = ForeignJob(job_id="1", user="x", state="PENDING", nodes=("jupiter",), gpus=4)
    assert not _place([_demand(1, 4, "gpu96")], foreign_jobs=[job])[1].conflict


# ── Earliest slot ────────────────────────────────────────────────────────────

def test_earliest_slot_is_now_when_the_cluster_is_free():
    demand = _demand(99, 4, "gpu96", start_h=0, hours=1)
    found = find_earliest_slot(demand, CLUSTER_NODES, [], cluster=CLUSTER)
    assert found == NOW


def test_earliest_slot_waits_for_an_occupying_booking_to_end():
    existing = [Demand(1, 4, NOW, NOW + timedelta(hours=3), gpu_class="gpu96")]
    demand = Demand(99, 4, NOW, NOW + timedelta(hours=1), gpu_class="gpu96")
    found = find_earliest_slot(demand, CLUSTER_NODES, existing, cluster=CLUSTER)
    assert found is not None and found >= NOW + timedelta(hours=3)


def test_earliest_slot_returns_none_when_it_can_never_fit():
    demand = _demand(99, 8, "gpu96")
    assert find_earliest_slot(demand, CLUSTER_NODES, [], cluster=CLUSTER) is None


def test_earliest_slot_accounts_for_foreign_jobs():
    demand = Demand(99, 4, NOW, NOW + timedelta(hours=1), gpu_class="gpu96")
    found = find_earliest_slot(
        demand, CLUSTER_NODES, [], cluster=CLUSTER,
        foreign_jobs=[_foreign("jupiter", 4, start_h=0, hours=3)],
    )
    assert found is not None and found >= NOW + timedelta(hours=3)


@pytest.mark.parametrize("gpus,expected_node", [
    # 1×gpu48 goes to europa (which has exactly one) rather than splitting
    # titan's only contiguous pair.
    (1, "europa"),
    (2, "titan"),
])
def test_placement_is_least_fragmenting(gpus, expected_node):
    p = _place([_demand(1, gpus, "gpu48")])[1]
    assert p.node == expected_node


def test_small_booking_does_not_block_a_later_large_one():
    """The reason best-fit matters: naive ordering would put the 1-GPU job on
    titan and leave no contiguous gpu48 pair anywhere."""
    result = _place([_demand(1, 1, "gpu48"), _demand(2, 2, "gpu48")])
    assert result[1].node == "europa"
    assert result[2].node == "titan"
    assert not result[2].conflict
