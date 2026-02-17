import subprocess
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class SlurmSubmitResult:
    job_id: str
    raw: str

def _run(cmd: list[str], extra_env: dict[str, str] | None = None) -> str:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
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
    cmd = [
        "sbatch", "--parsable",
        f"--job-name={job_name}",
        f"--gres=gpu:{gpus}",
        f"--cpus-per-task={cpus_per_task}",
        f"--time={time_limit}",
    ]
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

    # Use --export=ALL to inherit environment, then set vars in the process env
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
