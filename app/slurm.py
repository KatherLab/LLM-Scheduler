import subprocess
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import asyncio

@dataclass
class SlurmSubmitResult:
    job_id: str
    raw: str


def _run(cmd: list[str], extra_env: dict[str, str] | None = None) -> str:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    p = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    return p.stdout.strip()


def submit_vllm_job(
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
    mem: str | None = None,          # NEW
    log_dir: str = "./logs",
) -> SlurmSubmitResult:
    # Make logs deterministic and independent of router working directory:
    # Slurm expands %x=%jobname, %j=%jobid
    log_dir_abs = os.path.abspath(log_dir)
    os.makedirs(log_dir_abs, exist_ok=True)

    stdout_path = os.path.join(log_dir_abs, "%x-%j.out")
    stderr_path = os.path.join(log_dir_abs, "%x-%j.err")

    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--gres=gpu:{gpus}",
        f"--cpus-per-task={cpus_per_task}",
        f"--time={time_limit}",
        f"--output={stdout_path}",
        f"--error={stderr_path}",
    ]

    if mem:
        cmd.append(f"--mem={mem}")

    if begin is not None:
        cmd.append(f"--begin={begin.strftime('%Y-%m-%dT%H:%M:%S')}")
    if partition:
        cmd.append(f"--partition={partition}")
    if account:
        cmd.append(f"--account={account}")
    if qos:
        cmd.append(f"--qos={qos}")
    if nodelist:
        cmd.append(f"--nodelist={nodelist}")

    # Inherit environment then set vars in the process env
    cmd.append("--export=ALL")
    cmd.append(template_path)

    out = _run(cmd, extra_env=env)
    job_id = out.split(";")[0]
    return SlurmSubmitResult(job_id=job_id, raw=out)


def cancel(job_id: str) -> None:
    _run(["scancel", job_id])


def extend_time(job_id: str, new_time_limit: str) -> None:
    _run(["scontrol", "update", f"JobId={job_id}", f"TimeLimit={new_time_limit}"])


def squeue_job_state(job_id: str) -> str | None:
    try:
        out = _run(["squeue", "-j", job_id, "-h", "-o", "%T"])
        return out.strip() or None
    except subprocess.CalledProcessError:
        return None

async def async_squeue_job_state(job_id: str) -> str | None:
    """Non-blocking version of squeue_job_state for async contexts."""
    return await asyncio.to_thread(squeue_job_state, job_id)


async def async_cancel(job_id: str) -> None:
    """Non-blocking version of cancel for async contexts."""
    await asyncio.to_thread(cancel, job_id)


async def async_extend_time(job_id: str, new_time_limit: str) -> None:
    """Non-blocking version of extend_time for async contexts."""
    await asyncio.to_thread(extend_time, job_id, new_time_limit)


async def async_submit_vllm_job(**kwargs) -> SlurmSubmitResult:
    """Non-blocking version of submit_vllm_job for async contexts."""
    return await asyncio.to_thread(lambda: submit_vllm_job(**kwargs))

class SlurmUnavailableError(Exception):
    """Raised when the Slurm controller itself is unreachable."""
    pass

def squeue_job_states_batch(job_ids: list[str]) -> dict[str, str | None]:
    """
    Query Slurm for multiple job states in a single squeue call.
    Returns a dict of {job_id: state_or_None}.

    Raises SlurmUnavailableError if the Slurm controller is down,
    so callers don't mistakenly treat all jobs as gone.
    """
    if not job_ids:
        return {}

    result: dict[str, str | None] = {jid: None for jid in job_ids}
    try:
        job_list = ",".join(job_ids)
        out = _run(["squeue", "-j", job_list, "-h", "-o", "%i %T"])
        for line in out.strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                jid, state = parts[0], parts[1]
                if jid in result:
                    result[jid] = state
    except subprocess.CalledProcessError as e:
        output = (e.stdout or "") + " " + (e.stderr or "")
        # Slurm controller errors → don't assume jobs are gone
        slurm_down_indicators = [
            "slurm_load_jobs error",
            "Unable to contact slurm controller",
            "Socket timed out",
            "Connection refused",
            "slurmdbd:",
        ]
        if any(indicator in output for indicator in slurm_down_indicators):
            raise SlurmUnavailableError(f"Slurm controller unavailable: {output.strip()}")
        # Otherwise: the jobs are genuinely gone (e.g., "Invalid job id specified")
        # result stays all-None, which is correct
    return result

async def async_squeue_job_states_batch(job_ids: list[str]) -> dict[str, str | None]:
    """Non-blocking version of squeue_job_states_batch."""
    return await asyncio.to_thread(squeue_job_states_batch, job_ids)
