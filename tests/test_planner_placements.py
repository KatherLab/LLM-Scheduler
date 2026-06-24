"""Tests for lane placement in app.planner.

Regression test for the bug where PLANNED leases starting more than 48h in the
future were skipped by compute_placements (horizon cutoff), serialized with
lane_start=None, and collapsed onto lane 0 by the timeline frontend.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.planner import compute_placements


def _lease(id, begin, end, gpus=1, state="PLANNED"):
    return SimpleNamespace(
        id=id,
        state=state,
        begin_at=begin,
        end_at=end,
        created_at=begin,
        requested_gpus=gpus,
    )


def _run(leases, total_gpus=8, now=None):
    now = now or datetime.now(timezone.utc)
    horizon_start = now - timedelta(hours=1)
    horizon_end = now + timedelta(hours=24 * 15 + 1)  # match admin PLACEMENT_HORIZON
    return compute_placements(
        leases=leases,
        total_gpus=total_gpus,
        horizon_start=horizon_start,
        horizon_end=horizon_end,
    )


def test_future_planned_lease_beyond_old_48h_horizon_gets_a_lane():
    """A PLANNED lease starting 5 days out must still receive a real lane."""
    now = datetime.now(timezone.utc)
    begin = now + timedelta(days=5)
    end = begin + timedelta(hours=2)
    lease = _lease(1, begin, end, gpus=4)

    placements = _run([lease], now=now)

    p = placements[1]
    assert p is not None
    assert p.lane_start is not None, "future planned lease must be placed, not skipped"
    assert p.lane_count == 4
    assert p.conflict is False


def test_two_non_overlapping_future_leases_stack_on_different_lanes():
    """Two same-time future leases that fit must land on distinct lanes."""
    now = datetime.now(timezone.utc)
    begin = now + timedelta(days=5)
    end = begin + timedelta(hours=2)
    a = _lease(1, begin, end, gpus=4)
    b = _lease(2, begin, end, gpus=4)

    placements = _run([a, b], total_gpus=8, now=now)

    assert placements[1].lane_start == 0
    assert placements[2].lane_start == 4, "second 4-GPU lease must stack on lane 4"
    assert placements[1].conflict is False
    assert placements[2].conflict is False


def test_overlapping_leases_that_exceed_capacity_conflict():
    """Three 4-GPU leases at the same time on 8 GPUs cannot all fit."""
    now = datetime.now(timezone.utc)
    begin = now + timedelta(days=5)
    end = begin + timedelta(hours=2)
    leases = [_lease(i, begin, end, gpus=4) for i in range(1, 4)]

    placements = _run(leases, total_gpus=8, now=now)

    # Two fit (lanes 0 and 4), the third must conflict with no lane.
    placed = [p for p in placements.values() if p.lane_start is not None]
    conflicts = [p for p in placements.values() if p.conflict]
    assert len(placed) == 2
    assert len(conflicts) == 1
    assert conflicts[0].lane_start is None
