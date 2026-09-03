"""Slurm backend over slurmrestd (JWT-authenticated HTTP).

This is the implementation that removes the app's host dependencies: no Slurm
binaries, no munge socket, no matching slurm.conf, no bind-mounts, no container
UID gymnastics. The router only needs to reach slurmrestd over HTTP.

Verified against Slurm 25.02 / `data_parser/v0.0.42`. Field shapes were taken
from the OpenAPI document slurmrestd serves at `/openapi/v3` rather than
guessed, because they shift between API versions:

* numbers are wrapped: ``{"set": bool, "infinite": bool, "number": N}``
* ``job_state`` and node ``state`` are *lists* of flags, not strings
* ``--gres`` is expressed as ``tres_per_node="gres/gpu:gpu48:2"``
* ``time_limit`` is **minutes**, ``memory_per_node`` is **megabytes**
* ``mail_type`` is an enum that has no ``TIME_LIMIT`` — it is ``TIME=100%``

One capability is deliberately absent: `sbatch --test-only` has no REST
equivalent, so start-time estimation still needs `SlurmCliBackend`.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from .slurm_parse import expand_hostlist, is_nullish, parse_gres, parse_gres_map
from .types import (
    CAP_ACCOUNTING,
    CAP_FOREIGN_JOBS,
    CAP_NODE_DISCOVERY,
    CAP_RESERVATIONS,
    ClusterCommandError,
    ClusterUnavailableError,
    ExitInfo,
    ForeignJob,
    GpuGroup,
    JobSpec,
    JobState,
    NodeInfo,
    StartEstimate,
    SubmitResult,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..tokens import TokenProvider

logger = logging.getLogger(__name__)

#: `--mail-type` values Slurm's CLI accepts, mapped onto the REST enum.
_MAIL_TYPE_MAP = {
    "BEGIN": "BEGIN",
    "END": "END",
    "FAIL": "FAIL",
    "REQUEUE": "REQUEUE",
    "STAGE_OUT": "STAGE_OUT",
    "ARRAY_TASKS": "ARRAY_TASKS",
    "INVALID_DEPEND": "INVALID_DEPENDENCY",
    "INVALID_DEPENDENCY": "INVALID_DEPENDENCY",
    # The CLI spellings for time warnings have no direct enum member.
    "TIME_LIMIT": "TIME=100%",
    "TIME_LIMIT_90": "TIME=90%",
    "TIME_LIMIT_80": "TIME=80%",
    "TIME_LIMIT_50": "TIME=50%",
}

#: `gres_detail` entries look like `gpu:gpu48:2(IDX:0-1)` — the indices are the
#: precise GPUs a job holds, which beats guessing.
_IDX_RE = re.compile(r"IDX:([0-9,\-]+)")

_MEM_UNITS = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}


def _num(field: Any) -> int | None:
    """Unwrap ``{"set": .., "infinite": .., "number": ..}``.

    Returns None when unset or infinite — callers must not treat "no limit" as
    zero, which would look like a job that ended in 1970.
    """
    if field is None:
        return None
    if isinstance(field, (int, float)):
        return int(field)
    if isinstance(field, dict):
        if field.get("infinite") or not field.get("set", False):
            return None
        value = field.get("number")
        return int(value) if value is not None else None
    return None


def _epoch(field: Any) -> datetime | None:
    seconds = _num(field)
    if seconds is None or seconds <= 0:
        return None
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def _flags(field: Any) -> str:
    """Join a state flag list: ``['MIXED','DRAIN']`` -> ``MIXED+DRAIN``.

    Slurm's own textual convention, and it keeps the substring checks in
    `NodeInfo.is_usable` working.
    """
    if field is None:
        return ""
    if isinstance(field, str):
        return field
    if isinstance(field, (list, tuple)):
        return "+".join(str(x) for x in field)
    return str(field)


def _primary_state(field: Any) -> str:
    """The first flag is the base state; the rest are modifiers."""
    if isinstance(field, (list, tuple)) and field:
        return str(field[0])
    return _flags(field)


def parse_gres_detail(entries) -> tuple[str | None, int, tuple[int, ...]]:
    """``['gpu:gpu48:2(IDX:0-1)']`` -> ``('gpu48', 2, (0, 1))``.

    The indices matter: knowing exactly which GPUs a foreign job holds lets the
    planner pack around it instead of assuming it took the first N.
    """
    if not entries:
        return None, 0, ()
    gpu_class: str | None = None
    total = 0
    indices: list[int] = []
    for entry in entries:
        text = str(entry)
        m = _IDX_RE.search(text)
        if m:
            for chunk in m.group(1).split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if "-" in chunk:
                    lo, hi = chunk.split("-", 1)
                    try:
                        indices.extend(range(int(lo), int(hi) + 1))
                    except ValueError:
                        pass
                else:
                    try:
                        indices.append(int(chunk))
                    except ValueError:
                        pass
        cls, count = parse_gres(_IDX_RE.sub("", text).replace("()", ""))
        gpu_class = gpu_class or cls
        total += count
    if not total and indices:
        total = len(indices)
    return gpu_class, total, tuple(sorted(set(indices)))


def _mem_to_mb(mem: str | None) -> int | None:
    """``"500G"`` -> 512000. REST wants megabytes; the CLI took suffixes."""
    if is_nullish(mem):
        return None
    text = str(mem).strip()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([KMGT])?B?$", text, re.IGNORECASE)
    if not m:
        return None
    return int(float(m.group(1)) * _MEM_UNITS[(m.group(2) or "M").upper()])


def _time_limit_minutes(time_limit: str) -> int:
    """``"06:00:00"`` / ``"1-00:00:00"`` -> minutes (rounded up)."""
    text = (time_limit or "").strip()
    days = 0
    if "-" in text:
        day_part, _, text = text.partition("-")
        days = int(day_part or 0)
    parts = [int(p or 0) for p in text.split(":")] if text else [0]
    while len(parts) < 3:
        parts.append(0)
    hours, minutes, seconds = parts[0], parts[1], parts[2]
    total = days * 1440 + hours * 60 + minutes + (1 if seconds else 0)
    return max(1, total)


def _mail_types(raw: str | None) -> list[str]:
    out: list[str] = []
    for token in (raw or "").split(","):
        token = token.strip().upper()
        if not token or token == "NONE":
            continue
        mapped = _MAIL_TYPE_MAP.get(token)
        if mapped is None:
            logger.warning("slurmrest: dropping unsupported mail type %r", token)
            continue
        if mapped not in out:
            out.append(mapped)
    return out


class SlurmRestBackend:
    """Talks to slurmrestd. Construct with an explicit base URL and token."""

    name = "slurm-rest"

    def __init__(
        self,
        base_url: str,
        token: "str | TokenProvider",
        *,
        username: str | None = None,
        timeout: float = 30.0,
        verify: bool = True,
    ):
        if not base_url:
            raise ValueError("SLURM_REST_URL is required for the slurm_rest backend")
        if not token:
            raise ValueError("SLURM_JWT is required for the slurm_rest backend")

        self.base_url = base_url.rstrip("/")
        # /slurm/v0.0.42 -> /slurmdb/v0.0.42 for accounting lookups.
        self.db_url = re.sub(r"/slurm/(v[\d.]+)$", r"/slurmdb/\1", self.base_url)
        self.username = username
        self._timeout = timeout

        # A plain string stays supported, but a provider lets the token be
        # renewed without restarting the app.
        if isinstance(token, str):
            from ..tokens import TokenProvider, static_source
            self._tokens = TokenProvider(static_source(token), name="slurm-token(static)")
        else:
            self._tokens = token

        headers = {"Accept": "application/json"}
        if username:
            # The JWT may belong to a privileged user; naming the account keeps
            # jobs off root even when the token would allow it.
            headers["X-SLURM-USER-NAME"] = username
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)),
            verify=verify,
        )

        # `--test-only` has no REST equivalent, so CAP_TEST_ONLY is absent and
        # callers fall back to the CLI backend for start estimates.
        self.capabilities = frozenset({
            CAP_NODE_DISCOVERY, CAP_FOREIGN_JOBS, CAP_RESERVATIONS, CAP_ACCOUNTING,
        })

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── transport ───────────────────────────────────────────────────────────
    async def _request(
        self, method: str, path: str, *, db: bool = False, _retry: bool = True, **kw
    ) -> dict:
        url = f"{self.db_url if db else self.base_url}{path}"

        import asyncio as _asyncio

        from ..tokens import TokenError

        try:
            # Renewal may shell out (SSH), so keep it off the event loop.
            token = await _asyncio.to_thread(self._tokens.get)
        except TokenError as exc:
            raise ClusterUnavailableError(f"could not obtain a Slurm token: {exc}") from exc

        headers = {**kw.pop("headers", {}), "X-SLURM-USER-TOKEN": token}
        try:
            resp = await self._client.request(method, url, headers=headers, **kw)
        except httpx.RequestError as exc:
            # Network-level failure: the controller may be fine, but we cannot
            # tell, so callers must skip rather than assume jobs are gone.
            raise ClusterUnavailableError(f"slurmrestd unreachable at {url}: {exc}") from exc

        # 511 is what slurmrestd actually returns for an expired or invalid
        # token — not 401 — and the body says only "Protocol authentication
        # error", which is not a useful thing to show an operator at 3am.
        if resp.status_code in (401, 403, 511) or _is_auth_failure(resp):
            # Reactive renewal: a token can die earlier than its `exp` claim
            # suggests (revoked, key rotated, clock skew). Refresh once and
            # retry before declaring the cluster unreachable.
            if _retry:
                logger.info("slurmrest: auth rejected, renewing token and retrying")
                self._tokens.invalidate()
                return await self._request(method, path, db=db, _retry=False, **kw)
            raise ClusterUnavailableError(
                f"slurmrestd rejected our credentials ({resp.status_code}). "
                "SLURM_JWT is expired or invalid — regenerate it with "
                "`scontrol token lifespan=<seconds>` and restart. "
                "Scheduling is paused until then; no jobs were touched."
            )
        if resp.status_code >= 500:
            raise ClusterUnavailableError(
                f"slurmrestd {resp.status_code} for {path}: {resp.text[:300]}"
            )
        try:
            payload = resp.json()
        except ValueError as exc:
            raise ClusterUnavailableError(
                f"slurmrestd returned non-JSON for {path}: {resp.text[:200]}"
            ) from exc

        errors = payload.get("errors") or []
        if errors and resp.status_code >= 400:
            raise ClusterCommandError(_format_errors(errors))
        for warning in payload.get("warnings") or []:
            logger.debug("slurmrest warning on %s: %s", path, warning)
        return payload

    async def ping(self) -> dict:
        """Liveness plus the version/plugin info, for diagnostics."""
        return await self._request("GET", "/ping")

    # ── submission ──────────────────────────────────────────────────────────
    def _job_desc(self, spec: JobSpec) -> dict:
        env = dict(spec.env)
        # slurmrestd requires a non-empty environment, and the job needs a PATH
        # since it does not inherit the caller's like `--export=ALL` does.
        env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
        env.setdefault("SLURM_EXPORT_ENV", "ALL")

        log_dir = str(Path(spec.log_dir).expanduser())
        desc: dict[str, Any] = {
            "name": spec.job_name,
            "environment": [f"{k}={v}" for k, v in sorted(env.items())],
            "current_working_directory": log_dir,
            "standard_output": f"{log_dir}/%x-%j.out",
            "standard_error": f"{log_dir}/%x-%j.err",
            "time_limit": {"set": True, "number": _time_limit_minutes(spec.time_limit)},
            "cpus_per_task": spec.cpus,
            "tasks": 1,
            "nodes": "1",
        }

        # --gres has no REST field; it is expressed as a TRES-per-node string.
        gres = spec.gres or (f"gpu:{spec.gpus}" if spec.gpus else None)
        if gres:
            desc["tres_per_node"] = f"gres/{gres}"

        mem_mb = _mem_to_mb(spec.mem)
        if mem_mb:
            desc["memory_per_node"] = {"set": True, "number": mem_mb}
        if spec.partition:
            desc["partition"] = spec.partition
        if spec.account:
            desc["account"] = spec.account
        if spec.qos:
            desc["qos"] = spec.qos
        if spec.nodelist:
            desc["required_nodes"] = expand_hostlist(spec.nodelist) or [spec.nodelist]
        if spec.reservation:
            desc["reservation"] = spec.reservation
        if spec.comment:
            desc["comment"] = spec.comment
        if spec.begin is not None:
            desc["begin_time"] = {"set": True, "number": int(spec.begin.timestamp())}
        if spec.mail_user:
            types = _mail_types(spec.mail_type)
            if types:
                desc["mail_user"] = spec.mail_user
                desc["mail_type"] = types
        return desc

    async def submit(self, spec: JobSpec) -> SubmitResult:
        """Submit a job.

        Unlike sbatch, REST takes the script *contents*, not a path — so the
        template does not need to exist on a shared filesystem.
        """
        try:
            script = Path(spec.script_path).read_text(encoding="utf-8")
        except OSError as exc:
            raise ClusterCommandError(
                f"cannot read job template {spec.script_path}: {exc}"
            ) from exc

        payload = {"script": script, "job": self._job_desc(spec)}
        result = await self._request("POST", "/job/submit", json=payload)

        job_id = _num(result.get("job_id"))
        if job_id is None:
            raise ClusterCommandError(
                f"slurmrestd accepted the request but returned no job id: "
                f"{_format_errors(result.get('errors') or []) or result}"
            )
        for warning in result.get("warnings") or []:
            logger.info("slurmrest submit warning: %s", warning)
        return SubmitResult(job_id=str(job_id), raw=str(result.get("job_submit_user_msg") or job_id))

    async def cancel(self, job_id: str) -> None:
        try:
            await self._request("DELETE", f"/job/{job_id}")
        except ClusterCommandError as exc:
            # Cancelling a job that already finished is a no-op, not a failure.
            logger.info("slurmrest cancel %s: %s", job_id, exc)

    async def retarget_gpu_class(self, job_id: str, gres: str, gpu_class: str) -> None:
        """Point a *pending* job at a different GPU class.

        Verified against Slurm 25.05: `TresPerNode` and `Features` are both
        updatable while a job is PENDING, so a booking can follow whichever
        class frees up first without submitting a speculative second job.

        Note this bypasses `job_submit.lua` — that plugin only runs on submit,
        and its `slurm_job_modify` hook validates nothing but `mail_user`. So we
        must emit exactly what it would have produced (typed GRES *and* the
        matching feature), because nothing downstream will correct us.
        """
        await self._request("POST", f"/job/{job_id}", json={"job": {
            "tres_per_node": f"gres/{gres}",
            "constraints": gpu_class,
        }})

    async def extend_time(self, job_id: str, new_time_limit: str) -> None:
        await self._request("POST", f"/job/{job_id}", json={
            "job": {"time_limit": {"set": True, "number": _time_limit_minutes(new_time_limit)}}
        })

    # ── queue state ─────────────────────────────────────────────────────────
    async def job_states(self, job_ids: list[str]) -> dict[str, JobState | None]:
        result: dict[str, JobState | None] = {jid: None for jid in job_ids}
        if not job_ids:
            return result

        # One call for the whole queue beats N per-job round trips; the queue is
        # small relative to the HTTP overhead.
        payload = await self._request("GET", "/jobs/")
        for job in payload.get("jobs") or []:
            jid = str(_num(job.get("job_id")) or "")
            if jid not in result:
                continue
            result[jid] = JobState(
                state=_primary_state(job.get("job_state")),
                nodes=None if is_nullish(job.get("nodes")) else job.get("nodes"),
                start_time=_epoch(job.get("start_time")),
            )
        return result

    async def job_exit_info(self, job_ids: list[str]) -> dict[str, ExitInfo | None]:
        """Exit reasons from slurmdbd, falling back to the controller's memory."""
        result: dict[str, ExitInfo | None] = {jid: None for jid in job_ids}
        if not job_ids:
            return result

        for jid in job_ids:
            info = await self._exit_from_db(jid)
            if info is None:
                info = await self._exit_from_ctld(jid)
            result[jid] = info
        return result

    async def _exit_from_db(self, job_id: str) -> ExitInfo | None:
        try:
            payload = await self._request("GET", f"/job/{job_id}", db=True)
        except (ClusterCommandError, ClusterUnavailableError) as exc:
            logger.debug("slurmrest: slurmdb lookup failed for %s: %s", job_id, exc)
            return None
        for job in payload.get("jobs") or []:
            state = job.get("state") or {}
            current = _flags(state.get("current")) if isinstance(state, dict) else _flags(state)
            if current:
                return ExitInfo(
                    state=current,
                    exit_code=str(_num((job.get("exit_code") or {}).get("return_code")) or 0),
                    source="slurmdb",
                )
        return None

    async def _exit_from_ctld(self, job_id: str) -> ExitInfo | None:
        try:
            payload = await self._request("GET", f"/job/{job_id}")
        except (ClusterCommandError, ClusterUnavailableError):
            return None
        for job in payload.get("jobs") or []:
            exit_code = job.get("exit_code") or {}
            return ExitInfo(
                state=_flags(job.get("job_state")),
                exit_code=str(_num(exit_code.get("return_code")) or 0),
                source="slurmctld",
            )
        return None

    # ── inventory ───────────────────────────────────────────────────────────
    async def nodes(self) -> list[NodeInfo]:
        payload = await self._request("GET", "/nodes/")
        out: list[NodeInfo] = []
        for node in payload.get("nodes") or []:
            name = node.get("name")
            if not name:
                continue
            gres_map = parse_gres_map(node.get("gres"))
            out.append(NodeInfo(
                name=name,
                gpus=tuple(
                    GpuGroup(gpu_class=cls, count=count)
                    for cls, count in sorted(
                        gres_map.items(), key=lambda kv: (kv[0] is None, kv[0] or "")
                    )
                ),
                features=tuple(node.get("features") or ()),
                partitions=tuple(node.get("partitions") or ()),
                # Flags are a list here; join so DOWN/DRAIN stay detectable.
                state=_flags(node.get("state")),
                cpus=_num(node.get("cpus")) or 0,
                mem_mb=_num(node.get("real_memory")) or 0,
            ))
        return sorted(out, key=lambda n: n.name)

    async def foreign_jobs(self, partition: str | None = None) -> list[ForeignJob]:
        payload = await self._request("GET", "/jobs/")
        jobs: list[ForeignJob] = []
        for job in payload.get("jobs") or []:
            state = _primary_state(job.get("job_state"))
            if state not in ("RUNNING", "PENDING", "COMPLETING", "CONFIGURING", "SUSPENDED"):
                continue
            part = job.get("partition")
            if partition and part != partition:
                continue

            _, gpus, indices = parse_gres_detail(job.get("gres_detail"))
            if not gpus:
                # PENDING jobs have no gres_detail yet; fall back to the request.
                _, gpus = parse_gres(job.get("tres_per_node"))

            jobs.append(ForeignJob(
                job_id=str(_num(job.get("job_id")) or ""),
                user=job.get("user_name") or str(_num(job.get("user_id")) or "?"),
                state=state,
                nodes=tuple(expand_hostlist(job.get("nodes"))),
                gpus=gpus,
                partition=part,
                start_time=_epoch(job.get("start_time")),
                end_time=_epoch(job.get("end_time")),
                gpu_indices=indices,
            ))
        return jobs

    async def estimate_start(self, spec: JobSpec) -> StartEstimate:
        """Not available over REST — `sbatch --test-only` has no equivalent.

        Callers should check `CAP_TEST_ONLY` and fall back to the CLI backend.
        """
        raise NotImplementedError(
            "slurmrestd has no --test-only equivalent; use SlurmCliBackend for "
            "start-time estimates"
        )


def _format_errors(errors) -> str:
    parts = []
    for err in errors or []:
        if isinstance(err, dict):
            msg = err.get("description") or err.get("error") or str(err)
            source = err.get("source")
            parts.append(f"{msg}" + (f" ({source})" if source else ""))
        else:
            parts.append(str(err))
    return "; ".join(parts)


#: Slurm error numbers that mean "we could not authenticate", regardless of the
#: HTTP status the request happened to carry.
_AUTH_ERROR_NUMBERS = {1007, 2001, 5005}


def _is_auth_failure(resp) -> bool:
    """Detect an auth failure reported in the body rather than the status.

    slurmrestd can return a 200 with an `errors` array when slurmctld rejects
    the credentials, so status alone is not enough to tell "expired token" from
    "empty cluster" — and confusing those two is exactly the failure we must
    avoid.
    """
    try:
        payload = resp.json()
    except Exception:
        return False
    for err in (payload.get("errors") or []):
        if not isinstance(err, dict):
            continue
        if err.get("error_number") in _AUTH_ERROR_NUMBERS:
            return True
        if "authentication" in str(err.get("error", "")).lower():
            return True
    return False
