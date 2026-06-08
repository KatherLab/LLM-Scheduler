from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional
from .utils import ensure_utc

# Bookings ending within this tolerance of another's start are NOT overlapping.
# This allows back-to-back scheduling (e.g., model A ends 18:00, model B starts 18:00).
OVERLAP_TOLERANCE = timedelta(seconds=30)

@dataclass
class Placement:
    lease_id: int
    lane_start: Optional[int]
    lane_count: Optional[int]
    conflict: bool

def _t(dt: Optional[datetime], fallback: datetime) -> datetime:
    dt = ensure_utc(dt) if dt is not None else ensure_utc(fallback)
    return dt

def compute_placements(
    *,
    leases: Iterable,
    total_gpus: int,
    horizon_start: datetime,
    horizon_end: datetime,
) -> dict[int, Placement]:
    horizon_start = ensure_utc(horizon_start)
    horizon_end = ensure_utc(horizon_end)

    occ: list[list[tuple[datetime, datetime]]] = [[] for _ in range(total_gpus)]
    items = []
    for l in leases:
        begin = _t(l.begin_at, l.created_at)
        end = ensure_utc(l.end_at) if l.end_at else (begin + timedelta(hours=1))
        if end <= horizon_start or begin >= horizon_end:
            continue
        items.append((begin, end, l))
    items.sort(key=lambda x: (x[0], -(x[1] - x[0]).total_seconds()))
    placements: dict[int, Placement] = {}

    def overlaps(a0, a1, b0, b1):
        """
        Check if two time intervals overlap, with a tolerance for back-to-back bookings.
        Intervals that are merely adjacent (or within OVERLAP_TOLERANCE of each other)
        are NOT considered overlapping.
        """
        return not (a1 <= b0 + OVERLAP_TOLERANCE or b1 <= a0 + OVERLAP_TOLERANCE)

    def block_free(lane_idx, begin, end):
        for (s, e) in occ[lane_idx]:
            if overlaps(begin, end, s, e):
                return False
        return True

    for begin, end, l in items:
        g = max(1, int(getattr(l, "requested_gpus", 1) or 1))
        placed = False
        for start_lane in range(0, total_gpus - g + 1):
            ok = True
            for lane in range(start_lane, start_lane + g):
                if not block_free(lane, begin, end):
                    ok = False
                    break
            if ok:
                for lane in range(start_lane, start_lane + g):
                    occ[lane].append((begin, end))
                placements[l.id] = Placement(
                    lease_id=l.id, lane_start=start_lane, lane_count=g, conflict=False,
                )
                placed = True
                break
        if not placed:
            placements[l.id] = Placement(
                lease_id=l.id, lane_start=None, lane_count=g, conflict=True,
            )
    return placements


def find_earliest_slot(
    *,
    existing_leases: Iterable,
    gpus_needed: int,
    duration: timedelta,
    total_gpus: int,
    search_start: datetime,
    search_end: datetime,
    step: timedelta = timedelta(minutes=15),
) -> Optional[datetime]:
    """
    Find the earliest time within [search_start, search_end] where `gpus_needed`
    contiguous GPUs are free for the full `duration`.

    Uses lane-aware placement: builds an occupancy grid from existing leases,
    then sweeps candidate start times to find when a contiguous block of free lanes
    is available for the full duration.
    """
    search_start = ensure_utc(search_start)
    search_end = ensure_utc(search_end)

    if gpus_needed > total_gpus:
        return None

    # Use compute_placements to get proper lane assignments for existing leases
    placements = compute_placements(
        leases=existing_leases,
        total_gpus=total_gpus,
        horizon_start=search_start,
        horizon_end=search_end + duration,
    )

    # Build occupancy grid from placed leases
    occ: list[list[tuple[datetime, datetime]]] = [[] for _ in range(total_gpus)]
    for l in existing_leases:
        lb = _t(l.begin_at, l.created_at)
        le = ensure_utc(l.end_at) if l.end_at else (lb + timedelta(hours=1))
        if le <= search_start or lb >= search_end + duration:
            continue
        p = placements.get(l.id)
        if p and p.lane_start is not None and not p.conflict:
            for lane in range(p.lane_start, p.lane_start + p.lane_count):
                occ[lane].append((lb, le))

    def overlaps(a0, a1, b0, b1):
        return not (a1 <= b0 + OVERLAP_TOLERANCE or b1 <= a0 + OVERLAP_TOLERANCE)

    def lanes_free_at(begin: datetime, end: datetime, g: int) -> Optional[int]:
        """Check if there's a contiguous block of g free lanes for [begin, end).
        Returns the starting lane index, or None if no block is available."""
        for start_lane in range(0, total_gpus - g + 1):
            ok = True
            for lane in range(start_lane, start_lane + g):
                for (s, e) in occ[lane]:
                    if overlaps(begin, end, s, e):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return start_lane
        return None

    # Build interesting candidate start times
    interesting_times: set[datetime] = {search_start}
    for l in existing_leases:
        lb = _t(l.begin_at, l.created_at)
        le = ensure_utc(l.end_at) if l.end_at else (lb + timedelta(hours=1))
        if le <= search_start or lb >= search_end + duration:
            continue
        for t in (lb, le):
            if search_start <= t <= search_end:
                interesting_times.add(t)
            snapped = search_start + step * int((t - search_start) / step)
            if search_start <= snapped <= search_end:
                interesting_times.add(snapped)

    t = search_start
    while t + duration <= search_end:
        interesting_times.add(t)
        t += step

    sorted_times = sorted(interesting_times)

    for candidate in sorted_times:
        candidate_end = candidate + duration
        if candidate_end > search_end:
            break

        start_lane = lanes_free_at(candidate, candidate_end, gpus_needed)
        if start_lane is not None:
            return candidate

    return None
