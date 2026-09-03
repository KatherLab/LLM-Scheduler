"""Slurm backend that shells out to the CLI binaries.

This is the implementation the app has always had, lifted behind
`ClusterBackend` and extended with node discovery, foreign-job visibility and
`--test-only` estimates.

Its limitation is structural and is the reason `SlurmRestBackend` exists: it
requires the Slurm binaries, munge and matching config on the app host. The one
thing it keeps for the REST backend is `--test-only`, which has no clean REST
equivalent on most Slurm versions.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess

from .base import ClusterBackend  # noqa: F401  (documents the contract)
from .slurm_parse import (
    expand_hostlist,
    is_nullish,
    parse_gres,
    parse_gres_map,
    parse_mem_mb,
    parse_scontrol_kv,
    parse_slurm_time,
    parse_test_only,
)
from .types import (
    CAP_ACCOUNTING,
    CAP_FOREIGN_JOBS,
    CAP_NODE_DISCOVERY,
    CAP_TEST_ONLY,
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

logger = logging.getLogger(__name__)

# Phrases that mean "the controller is unreachable" rather than "the job is
# gone". Conflating the two would make the reconciler kill every live model.
_CONTROLLER_DOWN = (
    "slurm_load_jobs error",
    "unable to contact slurm controller",
    "socket timed out",
    "connection refused",
    "slurmdbd:",
    "zero bytes were transmitted",
)

# scontrol is queried one job at a time; bound it so a pathological reconcile
# cycle cannot spawn hundreds of subprocesses.
_MAX_SCONTROL_LOOKUPS = 20


class _Cmd:
    """Result of a command that we do not want to raise on."""

    __slots__ = ("returncode", "stdout", "stderr")

    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    @property
    def combined(self) -> str:
        return f"{self.stdout}\n{self.stderr}"


def _run(cmd: list[str], extra_env: dict[str, str] | None = None) -> str:
    """Run a command, raising CalledProcessError on failure."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    p = subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env,
    )
    return p.stdout.strip()


def _run_soft(cmd: list[str], extra_env: dict[str, str] | None = None) -> _Cmd:
    """Run a command without raising; callers inspect returncode/stderr."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    p = subprocess.run(
        cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env,
    )
    return _Cmd(p.returncode, p.stdout.strip(), p.stderr.strip())


def _raise_if_controller_down(output: str) -> None:
    low = output.lower()
    if any(ind in low for ind in _CONTROLLER_DOWN):
        raise ClusterUnavailableError(f"Slurm controller unavailable: {output.strip()}")


def build_sbatch_argv(spec: JobSpec) -> list[str]:
    """Translate a JobSpec into sbatch flags.

    Split out from submission so it can be reused verbatim by `--test-only`,
    guaranteeing the estimate reflects the job we would actually submit.
    """
    log_dir_abs = os.path.abspath(spec.log_dir)

    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name={spec.job_name}",
        f"--cpus-per-task={spec.cpus}",
        f"--time={spec.time_limit}",
        f"--output={os.path.join(log_dir_abs, '%x-%j.out')}",
        f"--error={os.path.join(log_dir_abs, '%x-%j.err')}",
    ]

    # A typed GRES pins the job to a GPU class; the bare count is the fallback.
    # A job that wants no GPU at all (an image build) must send neither: on
    # this cluster `--gres=gpu:0` would be rewritten by job_submit.lua into a
    # hard gpu24 constraint, queueing a CPU-only job behind a specific card.
    if spec.gres:
        cmd.append(f"--gres={spec.gres}")
    elif spec.gpus:
        cmd.append(f"--gres=gpu:{spec.gpus}")

    if spec.mem:
        cmd.append(f"--mem={spec.mem}")
    if spec.begin is not None:
        cmd.append(f"--begin={spec.begin.strftime('%Y-%m-%dT%H:%M:%S')}")
    if spec.partition:
        cmd.append(f"--partition={spec.partition}")
    if spec.account:
        cmd.append(f"--account={spec.account}")
    if spec.qos:
        cmd.append(f"--qos={spec.qos}")
    if spec.nodelist:
        cmd.append(f"--nodelist={spec.nodelist}")
    if spec.reservation:
        cmd.append(f"--reservation={spec.reservation}")
    # All jobs run as one service account, so the requester only survives here.
    if spec.comment:
        cmd.append(f"--comment={spec.comment}")
    if spec.mail_user:
        cmd.append(f"--mail-user={spec.mail_user}")
        cmd.append(f"--mail-type={spec.mail_type or 'FAIL,END,TIME_LIMIT'}")

    cmd.append("--export=ALL")
    cmd.append(spec.script_path)
    return cmd


class SlurmCliBackend:
    name = "slurm-cli"

    def __init__(self, *, probe: bool = True):
        self.capabilities = self._probe() if probe else frozenset()

    # ── capability probing ──────────────────────────────────────────────────
    @staticmethod
    def _probe() -> frozenset[str]:
        """Detect which binaries exist so callers degrade instead of crashing.

        `sacct` in particular is what tied the app to the accounting host; it is
        now optional and its absence only costs us richer failure reasons.
        """
        caps: set[str] = set()
        if shutil.which("sacct"):
            caps.add(CAP_ACCOUNTING)
        if shutil.which("sbatch"):
            caps.add(CAP_TEST_ONLY)
        if shutil.which("sinfo"):
            caps.add(CAP_NODE_DISCOVERY)
        if shutil.which("squeue"):
            caps.add(CAP_FOREIGN_JOBS)
        missing = [b for b in ("sbatch", "squeue", "scancel", "scontrol")
                   if not shutil.which(b)]
        if missing:
            logger.warning(
                "slurm-cli backend: missing binaries %s — those operations will fail",
                ", ".join(missing),
            )
        if CAP_ACCOUNTING not in caps:
            logger.info(
                "slurm-cli backend: sacct unavailable; falling back to scontrol "
                "for job exit reasons"
            )
        return frozenset(caps)

    # ── submission ──────────────────────────────────────────────────────────
    def submit_sync(self, spec: JobSpec) -> SubmitResult:
        os.makedirs(os.path.abspath(spec.log_dir), exist_ok=True)
        argv = build_sbatch_argv(spec)
        res = _run_soft(argv, extra_env=spec.env)
        if res.returncode != 0:
            _raise_if_controller_down(res.combined)
            raise subprocess.CalledProcessError(
                res.returncode, argv, output=res.stdout, stderr=res.stderr
            )
        job_id = res.stdout.split(";")[0].strip()
        return SubmitResult(job_id=job_id, raw=res.stdout)

    async def submit(self, spec: JobSpec) -> SubmitResult:
        return await asyncio.to_thread(self.submit_sync, spec)

    def estimate_start_sync(self, spec: JobSpec) -> StartEstimate:
        """`sbatch --test-only` — what the drag-and-drop preview is built on.

        Validates and reports a start time without queueing anything. Slurm
        writes the answer to stderr and exits non-zero when the job could never
        run, which we report as "no estimate" rather than an error.
        """
        argv = build_sbatch_argv(spec)
        argv.insert(1, "--test-only")
        res = _run_soft(argv, extra_env=spec.env)
        _raise_if_controller_down(res.combined)
        start, nodes = parse_test_only(res.combined)
        return StartEstimate(start_time=start, nodes=tuple(nodes), raw=res.combined.strip())

    async def estimate_start(self, spec: JobSpec) -> StartEstimate:
        return await asyncio.to_thread(self.estimate_start_sync, spec)

    # ── lifecycle ───────────────────────────────────────────────────────────
    def cancel_sync(self, job_id: str) -> None:
        res = _run_soft(["scancel", job_id])
        if res.returncode != 0:
            _raise_if_controller_down(res.combined)
            # Cancelling an already-finished job is a no-op, not a failure.
            logger.info("scancel %s: %s", job_id, res.stderr.strip())

    async def cancel(self, job_id: str) -> None:
        await asyncio.to_thread(self.cancel_sync, job_id)

    def retarget_gpu_class_sync(self, job_id: str, gres: str, gpu_class: str) -> None:
        """Point a pending job at a different GPU class. See the REST backend
        for why the feature must be set alongside the GRES."""
        _run(["scontrol", "update", f"JobId={job_id}",
              f"TresPerNode=gres/{gres}", f"Features={gpu_class}"])

    async def retarget_gpu_class(self, job_id: str, gres: str, gpu_class: str) -> None:
        await asyncio.to_thread(self.retarget_gpu_class_sync, job_id, gres, gpu_class)

    def extend_time_sync(self, job_id: str, new_time_limit: str) -> None:
        _run(["scontrol", "update", f"JobId={job_id}", f"TimeLimit={new_time_limit}"])

    async def extend_time(self, job_id: str, new_time_limit: str) -> None:
        await asyncio.to_thread(self.extend_time_sync, job_id, new_time_limit)

    # ── queue state ─────────────────────────────────────────────────────────
    def job_states_sync(self, job_ids: list[str]) -> dict[str, JobState | None]:
        """One squeue call for all ids. `None` means the job left the queue."""
        result: dict[str, JobState | None] = {jid: None for jid in job_ids}
        if not job_ids:
            return result

        res = _run_soft(
            ["squeue", "-j", ",".join(job_ids), "-h", "-o", "%i|%T|%N|%S"]
        )
        if res.returncode != 0:
            _raise_if_controller_down(res.combined)
            # "Invalid job id specified" — the jobs really are gone.
            return result

        for line in res.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            jid, state = parts[0].strip(), parts[1].strip()
            if jid not in result:
                continue
            nodes = parts[2].strip() if len(parts) > 2 else ""
            start = parts[3].strip() if len(parts) > 3 else ""
            result[jid] = JobState(
                state=state,
                nodes=None if is_nullish(nodes) else nodes,
                start_time=parse_slurm_time(start),
            )
        return result

    async def job_states(self, job_ids: list[str]) -> dict[str, JobState | None]:
        return await asyncio.to_thread(self.job_states_sync, job_ids)

    # ── exit reasons (sacct optional) ───────────────────────────────────────
    def job_exit_info_sync(self, job_ids: list[str]) -> dict[str, ExitInfo | None]:
        """Best-effort exit reason, degrading through three sources.

        sacct is authoritative but needs the accounting host; scontrol works
        from the controller's memory for `MinJobAge` (default 300s) and needs no
        slurmdbd; beyond that we report nothing and the caller falls back to the
        job's stderr log.
        """
        result: dict[str, ExitInfo | None] = {jid: None for jid in job_ids}
        if not job_ids:
            return result

        if CAP_ACCOUNTING in self.capabilities:
            result.update(self._sacct_exit_info(job_ids))

        missing = [jid for jid, info in result.items() if info is None]
        if missing:
            result.update(self._scontrol_exit_info(missing))
        return result

    def _sacct_exit_info(self, job_ids: list[str]) -> dict[str, ExitInfo]:
        out: dict[str, ExitInfo] = {}
        res = _run_soft([
            "sacct", "-j", ",".join(job_ids),
            "--format=JobID,State,ExitCode",
            "--noheader", "--parsable2", "--allocations",
        ])
        if res.returncode != 0:
            logger.info("sacct lookup failed, will try scontrol: %s", res.stderr.strip())
            return out
        wanted = set(job_ids)
        for line in res.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 3:
                continue
            jid = parts[0].split(".")[0]  # strip .batch / .extern suffixes
            if jid in wanted:
                out[jid] = ExitInfo(state=parts[1], exit_code=parts[2], source="sacct")
        return out

    def _scontrol_exit_info(self, job_ids: list[str]) -> dict[str, ExitInfo]:
        out: dict[str, ExitInfo] = {}
        lookups = job_ids[:_MAX_SCONTROL_LOOKUPS]
        if len(job_ids) > _MAX_SCONTROL_LOOKUPS:
            logger.warning(
                "scontrol exit-info lookup capped at %d of %d jobs; the remainder "
                "will report an unknown exit reason this cycle",
                _MAX_SCONTROL_LOOKUPS, len(job_ids),
            )
        for jid in lookups:
            res = _run_soft(["scontrol", "show", "job", jid, "--oneliner"])
            if res.returncode != 0 or not res.stdout:
                continue  # older than MinJobAge — nothing to learn
            kv = parse_scontrol_kv(res.stdout.splitlines()[0])
            state = kv.get("JobState")
            if state:
                out[jid] = ExitInfo(
                    state=state,
                    exit_code=kv.get("ExitCode", "?"),
                    source="scontrol",
                )
        return out

    async def job_exit_info(self, job_ids: list[str]) -> dict[str, ExitInfo | None]:
        return await asyncio.to_thread(self.job_exit_info_sync, job_ids)

    # ── inventory ───────────────────────────────────────────────────────────
    def nodes_sync(self) -> list[NodeInfo]:
        """Node inventory via sinfo. Replaces the configured TOTAL_GPUS.

        `%G` (Gres) carries the GPU *classes* the job_submit plugin expects
        (gpu24/gpu48/…); `%f` (Features) carries the GPU *models*
        (rtx_6000_ada, l40, a100). Both are needed: the planner allocates by
        class, but the catalog may target a specific model.
        """
        res = _run_soft(["sinfo", "-N", "-h", "-o", "%N|%P|%c|%m|%T|%G|%f"])
        if res.returncode != 0:
            _raise_if_controller_down(res.combined)
            return []

        # -N emits one line per (node, partition); merge them.
        merged: dict[str, dict] = {}
        for line in res.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 6:
                continue
            name, partition, cpus, mem, state, gres = (p.strip() for p in parts[:6])
            features = parts[6].strip() if len(parts) > 6 else ""
            if not name:
                continue

            entry = merged.setdefault(name, {
                "partitions": [],
                "cpus": int(cpus) if cpus.isdigit() else 0,
                "mem_mb": parse_mem_mb(mem),
                "state": state,
                "gpus": parse_gres_map(gres),
                "features": [],
            })
            # Slurm marks a node's default partition with a trailing '*'.
            partition = partition.rstrip("*")
            if partition and partition not in entry["partitions"]:
                entry["partitions"].append(partition)
            if not is_nullish(features):
                for feat in features.replace("&", ",").split(","):
                    feat = feat.strip()
                    if feat and feat not in entry["features"]:
                        entry["features"].append(feat)

        return [
            NodeInfo(
                name=name,
                gpus=tuple(
                    GpuGroup(gpu_class=cls, count=count)
                    # Sort for stable output; untyped (None) last.
                    for cls, count in sorted(
                        e["gpus"].items(), key=lambda kv: (kv[0] is None, kv[0] or "")
                    )
                ),
                features=tuple(e["features"]),
                partitions=tuple(e["partitions"]),
                state=e["state"],
                cpus=e["cpus"],
                mem_mb=e["mem_mb"],
            )
            for name, e in sorted(merged.items())
        ]

    async def nodes(self) -> list[NodeInfo]:
        return await asyncio.to_thread(self.nodes_sync)

    # ── foreign jobs ────────────────────────────────────────────────────────
    def foreign_jobs_sync(self, partition: str | None = None) -> list[ForeignJob]:
        """Jobs from all users, so the planner can see GPUs it does not control.

        Includes PENDING jobs because `%S` carries Slurm's backfill estimate,
        which is what the timeline renders as a ghost block.
        """
        argv = [
            "squeue", "-a", "-h",
            "-t", "RUNNING,PENDING,COMPLETING,CONFIGURING",
            "-o", "%i|%u|%T|%N|%b|%S|%e|%P",
        ]
        if partition:
            argv += ["-p", partition]

        res = _run_soft(argv)
        if res.returncode != 0:
            _raise_if_controller_down(res.combined)
            return []

        jobs: list[ForeignJob] = []
        for line in res.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 8:
                continue
            jid, user, state, nodes, gres, start, end, part = (p.strip() for p in parts[:8])
            if not jid:
                continue
            _, gpus = parse_gres(gres)
            jobs.append(ForeignJob(
                job_id=jid,
                user=user,
                state=state,
                nodes=tuple(expand_hostlist(nodes)),
                gpus=gpus,
                partition=part or None,
                start_time=parse_slurm_time(start),
                end_time=parse_slurm_time(end),
            ))
        return jobs

    async def foreign_jobs(self, partition: str | None = None) -> list[ForeignJob]:
        return await asyncio.to_thread(self.foreign_jobs_sync, partition)
