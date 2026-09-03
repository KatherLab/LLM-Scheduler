"""Several models on one GPU, inside a single Slurm allocation.

The cluster offers no way to subdivide a GPU for us:

* **MPS is disabled** cluster-wide (`job_submit.lua` rejects any `mps:` GRES).
* **MIG** is static, coarse, and unavailable on GB10 (DGX Spark). Where it *is*
  configured — `ganymede` is partitioned into 8 slices exposed as `gpu24` — the
  scheduler already sees plain GPUs and needs none of this.
* **`gres/shard`** would work but needs a `slurm.conf` change from the cluster
  admins, and still gives no memory enforcement.

So co-location happens *inside* one job: Slurm grants one GPU, and the job runs
several vLLM servers on it, each on its own port, each registering separately.
To the router they are ordinary endpoints, so routing, health checks and
metrics all work unchanged.

What this trades away, stated plainly:

* **Shared fate.** One Slurm job, so all co-tenants stop together. Individual
  vLLM crashes are restarted inside the job.
* **Compute contention.** The GPU time-slices between processes; throughput per
  model drops and latency gets noisy. Fine for embedding and reranking models,
  which are small and bursty. **Not** fine for benchmarking, where it silently
  invalidates the numbers.
* **No memory isolation.** Mitigated by construction: vLLM pre-allocates its KV
  cache at startup, so a group that fits at launch stays fitting. That is
  exactly why every co-tenant must declare an absolute budget.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from .catalog import CatalogModel, resolve_variant
from .cluster import GpuClass

#: Headroom left unallocated on a shared card for CUDA contexts, fragmentation
#: and the loader processes. Co-location is tighter than a single model, so a
#: little slack avoids an OOM at the last replica's startup.
COLOCATION_HEADROOM_GB = 2.0


class ColocationError(Exception):
    """A co-location group that cannot be launched as requested."""


@dataclass(frozen=True)
class CoTenant:
    """One model within a co-located group."""

    model: str
    model_path: str
    memory_gb: float
    gpu_memory_utilization: float
    tensor_parallel_size: int = 1
    extra_args: str = ""
    tool_args: str = ""
    reasoning_parser: str | None = None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "model_path": self.model_path,
            "memory_gb": self.memory_gb,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "extra_args": self.extra_args,
            "tool_args": self.tool_args,
            "reasoning_parser": self.reasoning_parser or "",
        }

    @staticmethod
    def from_dict(raw: dict) -> "CoTenant":
        return CoTenant(
            model=raw["model"],
            model_path=raw["model_path"],
            memory_gb=float(raw.get("memory_gb") or 0),
            gpu_memory_utilization=float(raw.get("gpu_memory_utilization") or 0),
            tensor_parallel_size=int(raw.get("tensor_parallel_size") or 1),
            extra_args=raw.get("extra_args") or "",
            tool_args=raw.get("tool_args") or "",
            reasoning_parser=raw.get("reasoning_parser") or None,
        )


def resolve_group(
    models: list[CatalogModel],
    gpu_class: GpuClass | None,
    *,
    headroom_gb: float = COLOCATION_HEADROOM_GB,
) -> list[CoTenant]:
    """Turn catalog entries into co-tenants that provably fit on one GPU.

    Raises `ColocationError` with an actionable message rather than letting a
    group launch and OOM halfway through — a half-started group is worse than a
    refused booking, because the models that did start look healthy.
    """
    if len(models) < 2:
        raise ColocationError("Co-location needs at least two models.")

    resolved = [resolve_variant(m, gpu_class.name if gpu_class else None) for m in models]

    names = [m.name for m in resolved]
    if len(set(names)) != len(names):
        raise ColocationError(
            "The same model cannot be co-located with itself; use replicas instead."
        )

    multi_gpu = [m.name for m in resolved if m.gpus > 1]
    if multi_gpu:
        # A tensor-parallel model owns whole devices; sharing one of them with
        # another server would deadlock the collective.
        raise ColocationError(
            f"Co-location is single-GPU only, but {', '.join(multi_gpu)} "
            f"need{'s' if len(multi_gpu) == 1 else ''} more than one GPU."
        )

    undeclared = [m.name for m in resolved if not m.memory_gb]
    if undeclared:
        raise ColocationError(
            f"Co-location requires an explicit memory_gb for every model; "
            f"missing on {', '.join(undeclared)}. Fractions of a shared card "
            f"cannot be made to add up."
        )

    if gpu_class is None or not gpu_class.vram_gb:
        raise ColocationError(
            "Co-location needs a GPU class with a known memory size."
        )

    budget = max(0.0, gpu_class.usable_gb - headroom_gb)
    total = sum(m.memory_gb for m in resolved)
    if total > budget:
        raise ColocationError(
            f"These models need {total:.0f} GB together but only {budget:.0f} GB "
            f"is usable on a {gpu_class.name} card "
            f"({gpu_class.vram_gb} GB, minus {gpu_class.reserved_gb:.0f} GB reserved "
            f"and {headroom_gb:.0f} GB headroom). Drop one, or lower their memory_gb."
        )

    return [
        CoTenant(
            model=m.name,
            model_path=m.model_path,
            memory_gb=m.memory_gb,
            # vLLM takes a fraction; the absolute budget is what we reason in.
            # Floored, not rounded: rounding up would let the fractions sum to
            # slightly more than the budget we just proved fits.
            gpu_memory_utilization=math.floor(
                (m.memory_gb / gpu_class.vram_gb) * 10_000
            ) / 10_000,
            tensor_parallel_size=m.tensor_parallel_size,
            extra_args=m.extra_args,
            tool_args=m.tool_args,
            reasoning_parser=m.reasoning_parser,
        )
        for m in resolved
    ]


def encode(tenants: list[CoTenant]) -> str:
    """Serialize for storage on the lease."""
    return json.dumps([t.to_dict() for t in tenants])


def decode(raw: str | None) -> list[CoTenant]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [CoTenant.from_dict(item) for item in data if isinstance(item, dict)]


def job_env(tenants: list[CoTenant]) -> dict[str, str]:
    """Environment the job template reads to launch the group.

    Passed as one JSON blob rather than numbered variables: the template loops
    over it, and adding a field later does not need a new variable name.
    """
    if not tenants:
        return {}
    return {
        "COLOCATED_MODELS": encode(tenants),
        "COLOCATED_COUNT": str(len(tenants)),
    }


def total_memory_gb(tenants: list[CoTenant]) -> float:
    return sum(t.memory_gb for t in tenants)


def describe(tenants: list[CoTenant]) -> str:
    """Short human summary, e.g. 'bge-m3 (12 GB) + bge-reranker (8 GB)'."""
    return " + ".join(f"{t.model} ({t.memory_gb:.0f} GB)" for t in tenants)
