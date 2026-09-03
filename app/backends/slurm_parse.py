"""Pure parsers for Slurm's textual output.

Kept separate from the subprocess layer so they can be unit-tested without a
cluster, and so the slurmrestd backend can reuse the vocabulary without
inheriting the parsing.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

# Slurm prints these instead of a value when a field is unset.
_NULLISH = {"", "(null)", "none", "n/a", "unknown", "unlimited"}


def is_nullish(value: str | None) -> bool:
    return value is None or value.strip().lower() in _NULLISH


def expand_hostlist(spec: str | None) -> list[str]:
    """Expand a Slurm compact hostlist.

    ``gpu[01-03,07],node5`` -> ``[gpu01, gpu02, gpu03, gpu07, node5]``

    Zero-padding is preserved from the range's lower bound, which is what Slurm
    itself does. Implemented in Python rather than shelling out to
    ``scontrol show hostnames`` so that parsing a queue full of foreign jobs
    costs zero subprocesses.
    """
    if is_nullish(spec):
        return []

    # Split on commas that are not inside brackets.
    parts: list[str] = []
    depth = 0
    current = ""
    for ch in spec:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current:
        parts.append(current)

    out: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(.*?)\[([^\]]+)\](.*)$", part)
        if not m:
            out.append(part)
            continue
        prefix, ranges, suffix = m.groups()
        for chunk in ranges.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                lo, hi = chunk.split("-", 1)
                width = len(lo)
                try:
                    for i in range(int(lo), int(hi) + 1):
                        out.append(f"{prefix}{str(i).zfill(width)}{suffix}")
                except ValueError:
                    out.append(f"{prefix}{chunk}{suffix}")
            else:
                out.append(f"{prefix}{chunk}{suffix}")
    return out


def parse_gres_map(spec: str | None) -> dict[str | None, int]:
    """Extract ``{gpu_class: count}`` from a Gres/TRES string.

    Returns a *map* rather than a single type because a node can hold several
    GPU classes at once — ``europa`` on this cluster is
    ``gpu:gpu24:1,gpu:gpu48:1``, and collapsing that to one type would let the
    planner put a gpu48 model on a 24 GB card.

    Handles the forms Slurm actually emits::

        gpu:4
        gpu:gpu48:2
        gpu:gpu48:2(S:0-1)
        gres/gpu=4
        gres:gpu:2,gres:nic:1

    Non-GPU resources are ignored. An untyped entry is keyed under ``None``.
    """
    out: dict[str | None, int] = {}
    if is_nullish(spec):
        return out

    # Drop socket/topology annotations: gpu:gpu48:2(S:0-1) -> gpu:gpu48:2
    text = re.sub(r"\([^)]*\)", "", spec.strip())

    def add(key: str | None, count: int) -> None:
        out[key] = out.get(key, 0) + count

    for entry in re.split(r"[,+]", text):
        entry = entry.strip()
        if not entry:
            continue
        # gres/gpu=4  or  gres/gpu:gpu48=2
        m = re.match(r"^gres[/:]gpu(?::([^=]+))?=(\d+)$", entry, re.IGNORECASE)
        if m:
            add(m.group(1), int(m.group(2)))
            continue
        # [gres/ | gres:]gpu[:type]:count
        # The `gres/` form is what slurmrestd reports in `tres_per_node`.
        m = re.match(r"^(?:gres[/:])?gpu(?::([^:]+))?:(\d+)$", entry, re.IGNORECASE)
        if m:
            gpu_type = m.group(1)
            # "gpu:4" matches this branch with type="4" — that is a count.
            if gpu_type is not None and gpu_type.isdigit():
                gpu_type = None
            add(gpu_type, int(m.group(2)))
            continue
        # bare "gpu" with no count means one
        if entry.lower() in ("gpu", "gres:gpu", "gres/gpu"):
            add(None, 1)

    return out


def parse_gres(spec: str | None) -> tuple[str | None, int]:
    """``(first_gpu_class, total_gpu_count)`` — for callers that only need a total.

    Used for foreign jobs, where the count is what occupies the grid. Prefer
    :func:`parse_gres_map` anywhere the class breakdown matters.
    """
    mapping = parse_gres_map(spec)
    if not mapping:
        return None, 0
    first_typed = next((k for k in mapping if k is not None), None)
    return first_typed, sum(mapping.values())


def parse_slurm_time(value: str | None) -> datetime | None:
    """Parse a Slurm timestamp (``2026-08-20T15:40:00``) as UTC-aware.

    Slurm emits times in the controller's local zone with no offset. We treat
    them as UTC, consistent with how the rest of the app stores time; deployments
    whose controller is not on UTC should set ``TZ=UTC`` for slurmd.
    """
    if is_nullish(value):
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    # Epoch seconds (slurmrestd style)
    try:
        return datetime.fromtimestamp(int(text), tz=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return None


def parse_mem_mb(value: str | None) -> int:
    """``sinfo %m`` gives megabytes, sometimes with a K/M/G/T suffix."""
    if is_nullish(value):
        return 0
    text = value.strip().rstrip("+")
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([KMGT])?$", text, re.IGNORECASE)
    if not m:
        return 0
    amount = float(m.group(1))
    unit = (m.group(2) or "M").upper()
    factor = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[unit]
    return int(amount * factor)


_TEST_ONLY_RE = re.compile(
    r"Job\s+\d+\s+to start at\s+(\S+?)(?:\s+using|\s*$)", re.IGNORECASE
)
_TEST_ONLY_NODES_RE = re.compile(r"on nodes?\s+(\S+)", re.IGNORECASE)


def parse_test_only(output: str) -> tuple[datetime | None, list[str]]:
    """Parse ``sbatch --test-only`` output (which Slurm writes to stderr).

    Example::

        sbatch: Job 1234 to start at 2026-08-20T15:40:00 using 4 processors
                on nodes gpu07 in partition general
    """
    start = None
    nodes: list[str] = []
    m = _TEST_ONLY_RE.search(output)
    if m:
        start = parse_slurm_time(m.group(1))
    m = _TEST_ONLY_NODES_RE.search(output)
    if m:
        nodes = expand_hostlist(m.group(1))
    return start, nodes


def parse_scontrol_kv(line: str) -> dict[str, str]:
    """Parse one ``scontrol show ... --oneliner`` line into a dict.

    The format is space-separated ``Key=Value``. Values may themselves contain
    ``=`` (e.g. ``Command=/x --a=b``) but not spaces, except for trailing
    free-text fields we do not care about.
    """
    out: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        if key and key not in out:
            out[key] = value
    return out
