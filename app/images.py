"""Apptainer images: what exists on the shared filesystem, and how to build more.

Two halves, split because they run in different places:

* **Inspection and deletion** are ordinary filesystem operations on the shared
  images directory, which this process must have mounted. Fast and synchronous.
* **Building** is a Slurm job. It has to be: Apptainer cannot cross-build, so
  an aarch64 `.sif` can only be produced on an aarch64 node. Submitting a job
  also gets us queue placement, logs and cancellation for free.

Nothing here rewrites `cluster.yaml`. Which image a GPU class uses stays a
deliberate edit to the config file — a build appearing in the directory must
never silently change what the next job runs.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from .backends.types import JobSpec, NodeInfo
from .cluster import ClusterConfig, RUNTIME_APPTAINER
from .settings import settings

SUFFIX = ".sif"

#: A build target name. No path separators, no leading dot, and it must end in
#: `.sif` — the file lands in a directory every compute node can read, so a
#: name that escapes it is a much bigger problem than a rejected request.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}\.sif$")

#: A container reference we are willing to hand to `apptainer build`.
#: Deliberately strict: this string ends up in a job script, and the set of
#: characters below cannot express a shell metacharacter, a flag, or a second
#: argument. `docker://` is the default scheme when none is given.
_REF_RE = re.compile(
    r"^(?:(docker|oras|library|shub|docker-daemon)://)?"   # optional scheme
    r"[A-Za-z0-9][A-Za-z0-9._:-]*"                          # host or first path part
    r"(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*"                     # further path parts
    r"(?::[A-Za-z0-9][A-Za-z0-9._-]*)?"                     # :tag
    r"(?:@sha256:[a-f0-9]{64})?$"                           # @digest
)

ARCHES = ("x86_64", "aarch64")


class ImageError(Exception):
    """A request we refuse, with a message meant for the user."""


# ── Validation ───────────────────────────────────────────────────────────────

def validate_name(name: str) -> str:
    name = (name or "").strip()
    if not name.endswith(SUFFIX):
        name = f"{name}{SUFFIX}"
    if not _NAME_RE.match(name):
        raise ImageError(
            f"Invalid image name {name!r}. Use letters, digits, dot, dash, "
            "plus or underscore, ending in .sif."
        )
    return name


def validate_ref(ref: str) -> str:
    """Normalise and check a source reference.

    A bare `vllm/vllm-openai:v0.11.0` is taken to mean `docker://…`, because
    that is what everyone types and the alternative is a confusing error.
    """
    ref = (ref or "").strip()
    if not ref:
        raise ImageError("A source image is required, e.g. vllm/vllm-openai:v0.11.0")
    if not _REF_RE.match(ref):
        raise ImageError(
            f"Invalid source reference {ref!r}. Expected something like "
            "vllm/vllm-openai:v0.11.0 or docker://nvcr.io/nvidia/vllm:25.09."
        )
    if "://" not in ref:
        ref = f"docker://{ref}"
    return ref


def validate_arch(arch: str) -> str:
    arch = (arch or "").strip()
    if arch not in ARCHES:
        raise ImageError(f"Unknown architecture {arch!r}. Expected one of {', '.join(ARCHES)}.")
    return arch


def suggest_name(ref: str, arch: str) -> str:
    """A default file name derived from the reference, e.g.
    `docker://vllm/vllm-openai:v0.11.0` + x86_64 -> `vllm-openai-v0.11.0-x86_64.sif`.
    """
    body = ref.split("://", 1)[-1]
    body = body.split("@", 1)[0]
    repo, _, tag = body.rpartition(":")
    if not repo:                       # no tag present
        repo, tag = body, "latest"
    short = repo.rstrip("/").rsplit("/", 1)[-1]
    safe = re.sub(r"[^A-Za-z0-9._+-]", "-", f"{short}-{tag}-{arch}")
    return f"{safe}{SUFFIX}"


# ── The images directory ─────────────────────────────────────────────────────

def image_dir() -> str:
    if not settings.apptainer_image_dir:
        raise ImageError(
            "No images directory configured. Set APPTAINER_IMAGE_DIR to the "
            "shared directory the compute nodes read images from."
        )
    return os.path.abspath(os.path.expanduser(settings.apptainer_image_dir))


def image_path(name: str) -> str:
    """Absolute path of a validated name, confined to the images directory.

    `validate_name` already forbids separators; the containment check is the
    belt to that pair of braces, and costs nothing.
    """
    directory = image_dir()
    path = os.path.abspath(os.path.join(directory, validate_name(name)))
    if os.path.dirname(path) != directory:
        raise ImageError(f"Refusing a path outside the images directory: {name!r}")
    return path


def build_scratch_dir() -> str:
    """Where Apptainer unpacks layers and caches blobs.

    Not a detail: an unpacked vLLM image is several times the size of the
    `.sif`, and the default is `$HOME` — which the service account does not
    have on every node. It also must not be a RAM-backed `/tmp`: vega's is a
    47 GB tmpfs, so an 8 GB image would be built in RAM.

    The build script scopes `APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR` under here
    per-job and `rm -rf`s them on exit, so this can point at fast node-local
    disk instead of the NFS-backed default without accumulating unpacked
    layers there build after build.
    """
    if settings.image_build_scratch:
        return os.path.abspath(os.path.expanduser(settings.image_build_scratch))
    return os.path.join(os.path.dirname(image_dir()), "build-tmp")


@dataclass(frozen=True)
class ImageInfo:
    name: str
    path: str
    size_bytes: int
    modified_at: datetime
    #: Runtimes in cluster.yaml pointing at this file. Non-empty means jobs
    #: depend on it and deleting it breaks them.
    used_by_runtimes: tuple[str, ...] = ()
    #: GPU classes that reach those runtimes — the user-visible consequence.
    used_by_gpu_classes: tuple[str, ...] = ()


def _runtime_users(cluster: ClusterConfig, name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Which runtimes and GPU classes can select one image file by name.

    `runtime.image` is a filename glob, not a fixed path, so several files can
    match one runtime — each is a selectable version, and every one of them
    counts as "in use" for delete protection.
    """
    runtimes = tuple(sorted(
        r.name for r in cluster.runtimes.values()
        if r.kind == RUNTIME_APPTAINER and r.image
        and fnmatch.fnmatch(name, os.path.basename(r.image))
    ))
    classes = tuple(sorted(
        c.name for c in cluster.gpu_classes.values() if c.runtime in runtimes
    ))
    return runtimes, classes


def list_versions(cluster: ClusterConfig, runtime_name: str) -> list[ImageInfo]:
    """Images one runtime can select, newest first."""
    return [i for i in list_images(cluster) if runtime_name in i.used_by_runtimes]


def resolve_runtime_image(
    cluster: ClusterConfig, runtime_name: str, requested: str | None = None
) -> str:
    """Absolute path of the image a job should launch: a pin, or the newest match."""
    candidates = list_versions(cluster, runtime_name)
    if not candidates:
        raise ImageError(
            f"No image matches runtime {runtime_name!r} in {image_dir()}."
        )
    if requested:
        for c in candidates:
            if c.name == requested:
                return c.path
        raise ImageError(f"Image {requested!r} is not available for runtime {runtime_name!r}.")
    return candidates[0].path


def list_images(cluster: ClusterConfig) -> list[ImageInfo]:
    """Every `.sif` in the images directory, newest first.

    Raises `ImageError` when the directory is not visible from here — that is
    a deployment fact worth stating plainly rather than an empty list, which
    would read as "you have no images".
    """
    directory = image_dir()
    try:
        entries = os.scandir(directory)
    except OSError as exc:
        raise ImageError(
            f"Images directory {directory} is not readable from the scheduler "
            f"({exc.strerror}). It must be mounted here to manage images."
        ) from exc

    out: list[ImageInfo] = []
    with entries as it:
        for entry in it:
            if not entry.is_file() or not entry.name.endswith(SUFFIX):
                continue
            stat = entry.stat()
            path = os.path.join(directory, entry.name)
            runtimes, classes = _runtime_users(cluster, entry.name)
            out.append(ImageInfo(
                name=entry.name,
                path=path,
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc),
                used_by_runtimes=runtimes,
                used_by_gpu_classes=classes,
            ))
    out.sort(key=lambda i: i.modified_at, reverse=True)
    return out


def delete_image(cluster: ClusterConfig, name: str, force: bool = False) -> str:
    """Remove one image. Refuses while `cluster.yaml` still points at it.

    Deleting a referenced image does not fail loudly — the file simply stops
    existing and every *future* job for that GPU class dies at launch. So the
    reference is checked here, and overriding it has to be explicit.
    """
    path = image_path(name)
    runtimes, classes = _runtime_users(cluster, name)
    if runtimes and not force:
        used = ", ".join(runtimes)
        via = f" (GPU classes: {', '.join(classes)})" if classes else ""
        raise ImageError(
            f"{name} is the image for runtime {used}{via}. Point cluster.yaml "
            "somewhere else first, or delete it with force if you know the "
            "runtime is unused."
        )
    try:
        os.remove(path)
    except FileNotFoundError as exc:
        raise ImageError(f"{name} does not exist.") from exc
    except OSError as exc:
        raise ImageError(f"Could not delete {name}: {exc.strerror}") from exc
    return path


# ── Where a build can run ────────────────────────────────────────────────────

@dataclass(frozen=True)
class BuildTarget:
    """A partition that can build for one architecture.

    Architecture is a property of the GPU class in `cluster.yaml`, and nodes
    are discovered with their classes — so "where can I build aarch64" is
    answerable without any extra configuration.
    """

    arch: str
    partition: str
    pool: str
    #: Nodes of this architecture in the partition.
    nodes: tuple[str, ...]
    #: Set only when the partition also holds nodes of *another* architecture,
    #: in which case the job has to name a machine. Leaving it None where the
    #: partition is uniform lets Slurm schedule the build wherever it fits.
    pin_node: str | None = None


def _node_arches(cluster: ClusterConfig, node: NodeInfo) -> set[str]:
    return {
        cluster.gpu_class(g.gpu_class).arch
        for g in node.gpus
        if cluster.gpu_class(g.gpu_class) is not None
    }


def build_targets(cluster: ClusterConfig, nodes: list[NodeInfo]) -> list[BuildTarget]:
    by_key: dict[tuple[str, str, str], list[NodeInfo]] = {}
    partition_arches: dict[str, set[str]] = {}

    for node in nodes:
        if not node.is_usable:
            continue
        pool = cluster.pool_for_node(node.name, node.partitions)
        if pool is None:
            continue
        for arch in _node_arches(cluster, node):
            if arch not in ARCHES:
                continue
            by_key.setdefault((arch, pool.partition, pool.name), []).append(node)
            partition_arches.setdefault(pool.partition, set()).add(arch)

    targets = []
    for (arch, part, pool), members in by_key.items():
        # An idle node first: a build is short and should not wait behind a
        # week-long model booking when a free machine is sitting there.
        members.sort(key=lambda n: ("IDLE" not in n.state.upper(), n.name))
        mixed = len(partition_arches.get(part, set())) > 1
        targets.append(BuildTarget(
            arch=arch,
            partition=part,
            pool=pool,
            nodes=tuple(n.name for n in members),
            pin_node=members[0].name if mixed and members else None,
        ))
    targets.sort(key=lambda t: (t.arch, t.partition))
    return targets


def resolve_target(
    cluster: ClusterConfig, nodes: list[NodeInfo], arch: str, partition: str | None
) -> BuildTarget:
    targets = [t for t in build_targets(cluster, nodes) if t.arch == arch]
    if not targets:
        raise ImageError(
            f"No node in this cluster is {arch}. Apptainer cannot cross-build, "
            "so there is nowhere to build that image."
        )
    if partition:
        for t in targets:
            if t.partition == partition:
                return t
        available = ", ".join(sorted({t.partition for t in targets}))
        raise ImageError(
            f"Partition {partition!r} has no {arch} nodes. Try: {available}."
        )
    return targets[0]


# ── Submitting the build ─────────────────────────────────────────────────────

def build_job_spec(
    *,
    name: str,
    source_ref: str,
    target: BuildTarget,
    requested_by: str | None = None,
) -> JobSpec:
    """The Slurm job that produces one image.

    No GPU is requested: `apptainer build` needs CPU and disk, and asking for a
    card would queue the build behind real work — and hold a GPU idle while it
    unpacks layers.
    """
    final_path = image_path(name)
    scratch = build_scratch_dir()

    env = {
        "BUILD_SOURCE_REF": source_ref,
        "BUILD_TARGET": final_path,
        "BUILD_SCRATCH": scratch,
        "BUILD_EXPECTED_ARCH": target.arch,
        # APPTAINER_CACHEDIR/APPTAINER_TMPDIR are left unset here: the template
        # derives them from BUILD_SCRATCH plus the Slurm job ID (unknown until
        # the job runs) and removes them on exit, so each build gets its own
        # disposable scratch instead of piling up across builds.
    }
    if settings.image_registry_username and settings.image_registry_password:
        env["APPTAINER_DOCKER_USERNAME"] = settings.image_registry_username
        env["APPTAINER_DOCKER_PASSWORD"] = settings.image_registry_password

    comment_bits = ["image-build"]
    if requested_by:
        comment_bits.append(f"user:{requested_by}")

    return JobSpec(
        job_name=f"sif-{name[: -len(SUFFIX)]}",
        script_path=settings.image_build_template_path,
        gpus=0,
        gres=None,
        time_limit=settings.image_build_time_limit,
        env=env,
        cpus=settings.image_build_cpus,
        partition=target.partition,
        nodelist=target.pin_node,
        comment=",".join(comment_bits),
        log_dir=settings.job_log_dir,
        begin=None,
    )
