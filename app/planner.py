from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

@dataclass
class Placement:
    lease_id: int
    lane_start: Optional[int]
    lane_count: Optional[int]
    conflict: bool

def _ensure_aware(dt: datetime) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

def _t(dt: Optional[datetime], fallback: datetime) -> datetime:
    dt = _ensure_aware(dt) if dt is not None else _ensure_aware(fallback)
    return dt

def compute_placements(
    *,
    leases: Iterable,
    total_gpus: int,
    horizon_start: datetime,
    horizon_end: datetime,
) -> dict[int, Placement]:
    horizon_start = _ensure_aware(horizon_start)
    horizon_end = _ensure_aware(horizon_end)

    occ: list[list[tuple[datetime, datetime]]] = [[] for _ in range(total_gpus)]
    items = []
    for l in leases:
        begin = _t(l.begin_at, l.created_at)
        end = _ensure_aware(l.end_at) if l.end_at else (begin + timedelta(hours=1))
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

    Returns the start datetime, or None if no slot exists.
    """
    search_start = _ensure_aware(search_start)
    search_end = _ensure_aware(search_end)

    # Pre-process existing leases into (begin, end, gpus) tuples
    intervals = []
    for l in existing_leases:
        begin = _t(l.begin_at, l.created_at)
        end = _ensure_aware(l.end_at) if l.end_at else (begin + timedelta(hours=1))
        g = max(1, int(getattr(l, "requested_gpus", 1) or 1))
        intervals.append((begin, end, g))

    def gpus_free_at(t: datetime) -> int:
        used = 0
        for (b, e, g) in intervals:
            if b <= t < e:
                used += g
        return total_gpus - used

    def slot_available(candidate_start: datetime) -> bool:
        """Check if gpus_needed are free for the entire [candidate_start, candidate_start + duration)."""
        candidate_end = candidate_start + duration
        if candidate_end > search_end:
            return False
        # Check at every step within the candidate window
        t = candidate_start
        while t < candidate_end:
            if gpus_free_at(t) < gpus_needed:
                return False
            t += step
        return True

    # Scan from search_start in `step` increments
    candidate = search_start
    while candidate + duration <= search_end:
        if slot_available(candidate):
            return candidate
        candidate += step

    return None
