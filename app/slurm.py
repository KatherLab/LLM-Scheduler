"""Compatibility shim over `app.backends`.

The real implementation now lives in `app/backends/` behind `ClusterBackend`.
This module keeps the historical function names and return shapes so existing
call sites in `main.py` and `admin.py` are untouched while the backend swap
lands; new code should use `app.backends.get_backend()` directly.

Two behavioural notes:

* `sacct_job_exit_info_batch` now falls back to `scontrol` when sacct is
  unavailable, so the app no longer has to run on the accounting host. Where it
  previously returned `None`, it may now return a real exit reason.
* `SlurmUnavailableError` is an alias of `ClusterUnavailableError`, so
  `except slurm.SlurmUnavailableError` keeps working.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .backends import ClusterUnavailableError, JobSpec, get_backend

# Preserved name — the reconciler catches this to skip a cycle rather than
# concluding every job has died.
SlurmUnavailableError = ClusterUnavailableError


@dataclass
class SlurmSubmitResult:
    job_id: str
    raw: str


def _call_sync(sync_name: str, async_name: str, *args):
    """Invoke a backend method from sync code.

    Prefers the backend's native sync entry point (the CLI backend has one for
    every call). Falls back to driving the coroutine on a private loop in a
    worker thread, which is needed because `reconcile_on_startup()` runs inside
    the FastAPI lifespan and therefore already has a running loop.
    """
    backend = get_backend()
    fn = getattr(backend, sync_name, None)
    if fn is not None:
        return fn(*args)

    coro_fn = getattr(backend, async_name)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_fn(*args))

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro_fn(*args))).result()


# ── Submission ───────────────────────────────────────────────────────────────

def _build_job_spec(
    *,
    template_path: str,
    job_name: str,
    gpus: int,
    time_limit: str,
    begin: Optional[datetime],
    env: dict[str, str],
    partition: str | None = None,
    account: str | None = None,
    qos: str | None = None,
    nodelist: str | None = None,
    cpus_per_task: int = 32,
    mem: str | None = None,
    log_dir: str = "./logs",
    mail_user: str | None = None,
    mail_type: str | None = None,
    gres: str | None = None,
    reservation: str | None = None,
    comment: str | None = None,
) -> JobSpec:
    return JobSpec(
        job_name=job_name,
        script_path=template_path,
        gpus=gpus,
        time_limit=time_limit,
        env=env,
        cpus=cpus_per_task,
        mem=mem,
        partition=partition,
        account=account,
        qos=qos,
        nodelist=nodelist,
        gres=gres,
        reservation=reservation,
        begin=begin,
        comment=comment,
        log_dir=log_dir,
        mail_user=mail_user,
        mail_type=mail_type,
    )


def submit_vllm_job(**kwargs) -> SlurmSubmitResult:
    spec = _build_job_spec(**kwargs)
    res = _call_sync("submit_sync", "submit", spec)
    return SlurmSubmitResult(job_id=res.job_id, raw=res.raw)


async def async_submit_vllm_job(**kwargs) -> SlurmSubmitResult:
    """Submit on the *current* running loop.

    Unlike the sync `submit_vllm_job`, this must not be driven through
    `_call_sync`'s worker-thread + `asyncio.run()` fallback: backends like
    `SlurmRestBackend` hold a persistent `httpx.AsyncClient` bound to the loop
    they were constructed on, and handing its coroutine to a freshly spun-up
    loop in another thread raises "bound to a different event loop". The CLI
    backend's `submit()` already wraps its blocking subprocess call in
    `asyncio.to_thread` internally, so calling it directly here is safe too.
    """
    spec = _build_job_spec(**kwargs)
    res = await get_backend().submit(spec)
    return SlurmSubmitResult(job_id=res.job_id, raw=res.raw)


# ── Lifecycle ────────────────────────────────────────────────────────────────

def cancel(job_id: str) -> None:
    _call_sync("cancel_sync", "cancel", job_id)


async def async_cancel(job_id: str) -> None:
    await get_backend().cancel(job_id)


def extend_time(job_id: str, new_time_limit: str) -> None:
    _call_sync("extend_time_sync", "extend_time", job_id, new_time_limit)


async def async_extend_time(job_id: str, new_time_limit: str) -> None:
    await get_backend().extend_time(job_id, new_time_limit)


# ── Queue state (legacy shapes: plain state strings) ─────────────────────────

def squeue_job_states_batch(job_ids: list[str]) -> dict[str, str | None]:
    states = _call_sync("job_states_sync", "job_states", job_ids)
    return {jid: (st.state if st else None) for jid, st in states.items()}


async def async_squeue_job_states_batch(job_ids: list[str]) -> dict[str, str | None]:
    states = await get_backend().job_states(job_ids)
    return {jid: (st.state if st else None) for jid, st in states.items()}


def squeue_job_state(job_id: str) -> str | None:
    return squeue_job_states_batch([job_id]).get(job_id)


async def async_squeue_job_state(job_id: str) -> str | None:
    return (await async_squeue_job_states_batch([job_id])).get(job_id)


# ── Exit reasons (legacy shape: {"state": ..., "exit_code": ...}) ────────────

def _exit_to_legacy(info) -> dict | None:
    if info is None:
        return None
    return {"state": info.state, "exit_code": info.exit_code, "source": info.source}


def sacct_job_exit_info_batch(job_ids: list[str]) -> dict[str, dict | None]:
    infos = _call_sync("job_exit_info_sync", "job_exit_info", job_ids)
    return {jid: _exit_to_legacy(info) for jid, info in infos.items()}


async def async_sacct_job_exit_info_batch(job_ids: list[str]) -> dict[str, dict | None]:
    infos = await get_backend().job_exit_info(job_ids)
    return {jid: _exit_to_legacy(info) for jid, info in infos.items()}
