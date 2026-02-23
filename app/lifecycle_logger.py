"""
Dedicated logger for model lifecycle events.

Writes to a separate file so events are never buried by request proxy logs.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from typing import Optional

from .settings import settings

_logger: logging.Logger | None = None


def get_lifecycle_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    _logger = logging.getLogger("vllm_lifecycle")
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False  # don't duplicate to root/uvicorn logger

    log_dir = os.path.abspath(settings.vllm_log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "lifecycle.log")

    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=50 * 1024 * 1024,  # 50 MB
        backupCount=5,
        encoding="utf-8",
    )
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

    # Also add a stderr handler so you can still see events in the console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    _logger.addHandler(console)

    return _logger


# ── Convenience functions with structured context ────────────────────────────

def log_health_check(
    model: str,
    slurm_job_id: Optional[str],
    endpoint_state: str,
    success: bool,
    error: Optional[str] = None,
    elapsed_ms: Optional[float] = None,
    fail_count: Optional[int] = None,
):
    lg = get_lifecycle_logger()
    status = "OK" if success else "FAIL"
    parts = [
        f"HEALTH_CHECK {status}",
        f"model={model}",
        f"job={slurm_job_id}",
        f"ep_state={endpoint_state}",
    ]
    if elapsed_ms is not None:
        parts.append(f"elapsed_ms={elapsed_ms:.1f}")
    if fail_count is not None:
        parts.append(f"consecutive_fails={fail_count}")
    if error:
        parts.append(f"error={error!r}")
    lg.info(" | ".join(parts))


def log_state_transition(
    entity: str,  # "lease" or "endpoint"
    entity_id,
    model: str,
    old_state: str,
    new_state: str,
    reason: str = "",
    slurm_job_id: Optional[str] = None,
):
    lg = get_lifecycle_logger()
    parts = [
        f"STATE_CHANGE {entity}={entity_id}",
        f"model={model}",
        f"{old_state} -> {new_state}",
    ]
    if slurm_job_id:
        parts.append(f"job={slurm_job_id}")
    if reason:
        parts.append(f"reason={reason}")
    lg.info(" | ".join(parts))


def log_slurm_action(
    action: str,  # "submit", "cancel", "extend", "retry"
    model: str,
    slurm_job_id: Optional[str] = None,
    lease_id: Optional[int] = None,
    detail: str = "",
):
    lg = get_lifecycle_logger()
    parts = [
        f"SLURM_{action.upper()}",
        f"model={model}",
    ]
    if lease_id is not None:
        parts.append(f"lease={lease_id}")
    if slurm_job_id:
        parts.append(f"job={slurm_job_id}")
    if detail:
        parts.append(detail)
    lg.info(" | ".join(parts))
