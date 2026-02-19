from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional
from .utils import ensure_utc

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
        return not (a1 <= b0 or b1 <= a0)

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
    GPUs are free for the full `duration`.

    Uses a sweep-line approach: builds a sorted list of GPU usage change events,
    then sweeps to find windows where enough GPUs are free.
    """
    search_start = ensure_utc(search_start)
    search_end = ensure_utc(search_end)

    if gpus_needed > total_gpus:
        return None

    # Build event list: (time, gpu_delta)
    # +gpus at lease begin, -gpus at lease end
    events: list[tuple[datetime, int]] = []
    for l in existing_leases:
        begin = _t(l.begin_at, l.created_at)
        end = ensure_utc(l.end_at) if l.end_at else (begin + timedelta(hours=1))
        g = max(1, int(getattr(l, "requested_gpus", 1) or 1))
        # Only consider leases that overlap with our search window
        if end <= search_start or begin >= search_end + duration:
            continue
        events.append((begin, g))
        events.append((end, -g))

    # Sort events by time (ties broken by delta: releases before acquisitions)
    events.sort(key=lambda e: (e[0], e[1]))

    # Build a sorted list of all "interesting" time points to check
    # These are: search_start, every event time, and step-aligned times
    interesting_times: set[datetime] = {search_start}
    for evt_time, _ in events:
        if search_start <= evt_time <= search_end:
            interesting_times.add(evt_time)
        # Also add step-snapped version
        snapped = search_start + step * int((evt_time - search_start) / step)
        if search_start <= snapped <= search_end:
            interesting_times.add(snapped)

    # Also add step-aligned times for completeness (but limit to reasonable count)
    t = search_start
    while t + duration <= search_end:
        interesting_times.add(t)
        t += step

    sorted_times = sorted(interesting_times)

    # For each candidate start, check if GPUs are free for the entire duration
    # We use the events to compute GPU usage at any point
    def gpus_used_at(t: datetime) -> int:
        """Compute GPUs in use at time t by summing all events before t."""
        used = 0
        for evt_time, delta in events:
            if evt_time <= t:
                used += delta
            else:
                break
        return used

    # Precompute: for each candidate, we need to check that gpus_used < threshold
    # for the entire [candidate, candidate+duration) window.
    # Collect all event times within each candidate window.
    for candidate in sorted_times:
        candidate_end = candidate + duration
        if candidate_end > search_end:
            break

        # Check at candidate start and at every event within the window
        ok = True

        # Check at candidate start
        if total_gpus - gpus_used_at(candidate) < gpus_needed:
            continue

        # Check at every event time within [candidate, candidate_end)
        for evt_time, _ in events:
            if evt_time < candidate:
                continue
            if evt_time >= candidate_end:
                break
            if total_gpus - gpus_used_at(evt_time) < gpus_needed:
                ok = False
                break

        if ok:
            return candidate

    return None
