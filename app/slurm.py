from __future__ import annotations
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

@dataclass
class SlurmSubmitResult:
    job_id: str
    raw: str

def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
) -> SlurmSubmitResult:
    cmd = ["sbatch", "--parsable", f"--job-name={job_name}", f"--gres=gpu:{gpus}", f"--cpus-per-task={cpus_per_task}", f"--time={time_limit}"]
    if begin is not None:
        # Slurm accepts ISO-ish; convert to local naive string without timezone
        cmd.append(f"--begin={begin.strftime('%Y-%m-%dT%H:%M:%S')}")
    if partition:
        cmd.append(f"--partition={partition}")
    if account:
        cmd.append(f"--account={account}")
    if qos:
        cmd.append(f"--qos={qos}")
    if nodelist:
        cmd.append(f"--nodelist={nodelist}")

    export_pairs = ",".join([f"{k}={v}" for k, v in env.items()])
    cmd.append(f"--export=ALL,{export_pairs}")
    cmd.append(template_path)

    out = _run(cmd)
    # sbatch --parsable returns jobid (sometimes with ;cluster)
    job_id = out.split(";")[0]
    return SlurmSubmitResult(job_id=job_id, raw=out)

def cancel(job_id: str) -> None:
    _run(["scancel", job_id])

def extend_time(job_id: str, new_time_limit: str) -> None:
    _run(["scontrol", "update", f"JobId={job_id}", f"TimeLimit={new_time_limit}"])

def squeue_job_state(job_id: str) -> str | None:
    # Returns state code or None if not in queue
    try:
        out = _run(["squeue", "-j", job_id, "-h", "-o", "%T"])
        return out.strip() or None
    except subprocess.CalledProcessError:
        return None
