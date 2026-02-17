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
    """Ensure datetime is timezone-aware (UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

def _t(dt: Optional[datetime], fallback: datetime) -> datetime:
    return _ensure_aware(dt if dt is not None else fallback)

def compute_placements(
    *,
    leases: Iterable,
    total_gpus: int,
    horizon_start: datetime,
    horizon_end: datetime,
) -> dict[int, Placement]:
    """
    Place leases onto GPU lanes for visualization.
    - We treat each GPU as a lane.
    - A lease with g GPUs occupies a contiguous block of g lanes.
    - First-fit in time order.
    - If a lease cannot be placed (overbooked), mark conflict=True.
    """
    # Ensure horizon bounds are aware
    horizon_start = _ensure_aware(horizon_start)
    horizon_end = _ensure_aware(horizon_end)
    
    # Occupancy: for each lane, list of (start,end) intervals
    occ: list[list[tuple[datetime, datetime]]] = [[] for _ in range(total_gpus)]
    # Normalize/clip intervals to horizon (still used for overlap checks)
    items = []
    for l in leases:
        begin = _t(l.begin_at, l.created_at)
        end = _ensure_aware(l.end_at) if l.end_at else (begin + timedelta(hours=1))
        # ignore fully outside horizon
        if end <= horizon_start or begin >= horizon_end:
            continue
        items.append((begin, end, l))
    # sort by begin time, then longer first helps packing
    items.sort(key=lambda x: (x[0], -(x[1]-x[0]).total_seconds()))
    placements: dict[int, Placement] = {}
    def overlaps(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> bool:
        return not (a1 <= b0 or b1 <= a0)
    def block_free(lane_idx: int, begin: datetime, end: datetime) -> bool:
        for (s, e) in occ[lane_idx]:
            if overlaps(begin, end, s, e):
                return False
        return True
    for begin, end, l in items:
        g = max(1, int(getattr(l, "requested_gpus", 1) or 1))
        placed = False
        # find contiguous block
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
                    lease_id=l.id,
                    lane_start=start_lane,
                    lane_count=g,
                    conflict=False,
                )
                placed = True
                break
        if not placed:
            placements[l.id] = Placement(
                lease_id=l.id,
                lane_start=None,
                lane_count=g,
                conflict=True,
            )
    return placements