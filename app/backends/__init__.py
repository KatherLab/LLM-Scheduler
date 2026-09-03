"""Cluster backend abstraction.

Everything that talks to the batch scheduler goes through `ClusterBackend`.
Select the implementation with `CLUSTER_BACKEND` (`slurm_cli` | `local`).

`slurm_rest` (slurmrestd over JWT) lands in this package next; it is the one
that removes the requirement to run on a host with Slurm binaries and munge.
"""

from __future__ import annotations

import logging

from .base import ClusterBackend
from .local import LocalBackend
from .slurm_cli import SlurmCliBackend
from .slurm_rest import SlurmRestBackend
from .types import (
    CAP_ACCOUNTING,
    CAP_FOREIGN_JOBS,
    CAP_NODE_DISCOVERY,
    CAP_RESERVATIONS,
    CAP_TEST_ONLY,
    ClusterCommandError,
    ClusterUnavailableError,
    ExitInfo,
    ForeignJob,
    JobSpec,
    JobState,
    NodeInfo,
    StartEstimate,
    SubmitResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ClusterBackend",
    "LocalBackend",
    "SlurmCliBackend",
    "SlurmRestBackend",
    "JobSpec",
    "JobState",
    "SubmitResult",
    "ExitInfo",
    "NodeInfo",
    "ForeignJob",
    "StartEstimate",
    "ClusterUnavailableError",
    "ClusterCommandError",
    "CAP_ACCOUNTING",
    "CAP_TEST_ONLY",
    "CAP_RESERVATIONS",
    "CAP_NODE_DISCOVERY",
    "CAP_FOREIGN_JOBS",
    "get_backend",
    "set_backend",
    "get_estimate_backend",
    "set_estimate_backend",
]

_backend: ClusterBackend | None = None


def _build(kind: str) -> ClusterBackend:
    from ..settings import settings

    kind = (kind or "slurm_cli").strip().lower()
    if kind in ("local", "fake", "memory"):
        return LocalBackend()
    if kind in ("slurm_cli", "slurm", "cli"):
        return SlurmCliBackend()
    if kind in ("slurm_rest", "rest", "slurmrestd"):
        from ..tokens import get_provider

        return SlurmRestBackend(
            settings.slurm_rest_url,
            get_provider(),
            username=settings.slurm_rest_user,
            timeout=settings.slurm_rest_timeout_seconds,
            verify=settings.slurm_rest_verify_tls,
        )
    raise ValueError(
        f"Unknown CLUSTER_BACKEND {kind!r}; "
        "expected 'slurm_rest', 'slurm_cli' or 'local'"
    )


_estimate_backend: ClusterBackend | None = None
_estimate_probed = False


def get_estimate_backend() -> ClusterBackend | None:
    """A backend that can answer "when would this start?", or None.

    `sbatch --test-only` has no slurmrestd equivalent, so a REST deployment
    needs the CLI alongside it for pre-submit previews. When the router runs
    off-cluster there are no Slurm binaries either — then this returns None and
    callers must say "start time unknown" rather than invent one.

    Note this is only needed *before* submission: once a job is queued, Slurm's
    own backfill estimate arrives via `job_states().start_time`.
    """
    global _estimate_backend, _estimate_probed

    primary = get_backend()
    if CAP_TEST_ONLY in primary.capabilities:
        return primary

    if not _estimate_probed:
        _estimate_probed = True
        from ..settings import settings

        kind = (settings.estimate_backend or "").strip().lower()
        if kind and kind not in ("none", "off"):
            try:
                candidate = _build(kind)
            except Exception as exc:
                logger.warning("estimate backend %r unavailable: %s", kind, exc)
            else:
                if CAP_TEST_ONLY in candidate.capabilities:
                    _estimate_backend = candidate
                    logger.info("start estimates via %s backend", candidate.name)
                else:
                    logger.info(
                        "estimate backend %s lacks --test-only; pre-submit "
                        "previews will report an unknown start time", candidate.name
                    )
    return _estimate_backend


def set_estimate_backend(backend: ClusterBackend | None) -> None:
    """Override the estimate backend. For tests."""
    global _estimate_backend, _estimate_probed
    _estimate_backend = backend
    _estimate_probed = True


def get_backend() -> ClusterBackend:
    """Return the process-wide backend, constructing it on first use.

    Construction probes for binaries, so it is deferred rather than done at
    import time — that keeps `import app.backends` cheap in tests.
    """
    global _backend
    if _backend is None:
        from ..settings import settings

        _backend = _build(settings.cluster_backend)
        logger.info(
            "cluster backend: %s (capabilities: %s)",
            _backend.name,
            ", ".join(sorted(_backend.capabilities)) or "none",
        )
    return _backend


def set_backend(backend: ClusterBackend | None) -> None:
    """Override the backend. For tests and for wiring a fake in development."""
    global _backend, _estimate_backend, _estimate_probed
    _backend = backend
    _estimate_backend = None
    _estimate_probed = False
