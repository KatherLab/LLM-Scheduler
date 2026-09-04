"""Admin API for Apptainer images.

Admin-only throughout. A build writes multiple gigabytes to shared storage and
takes cluster time; a delete can break every future job for a GPU class. Those
are not booking-level operations.

The split mirrors `app/images.py`: listing and deleting touch the filesystem
directly, building goes out as a Slurm job and is tracked in `image_builds`.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from . import images, inventory
from .auth import current_user, require_auth
from .authz import User
from .backends import ClusterCommandError, ClusterUnavailableError, get_backend
from .cluster import get_cluster
from .dependencies import SessionLocal
from .lifecycle_logger import log_slurm_action
from .models import ImageBuild
from .schemas import (
    BuildTargetOut,
    ImageBuildOut,
    ImageBuildProgressOut,
    ImageBuildRequest,
    ImageOut,
    ImagesResponse,
    LogResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin/images", tags=["images"], dependencies=[Depends(require_auth)]
)

ACTIVE_STATES = ("SUBMITTED", "RUNNING")

#: Enough log tail to hold several heartbeats plus apptainer's last words.
PROGRESS_TAIL_BYTES = 16_000


def _require_admin(user: User) -> None:
    if not user.is_admin:
        raise HTTPException(
            status_code=403,
            detail="Only admins can manage container images.",
        )


def _progress_for(job_id: str | None) -> ImageBuildProgressOut | None:
    """What the build is doing right now, from the tail of its job log.

    Both streams are read: the heartbeat lines go to stdout, but apptainer's
    own commentary — the most recent human-readable thing that happened — goes
    to stderr. Only the tail is needed, so this stays a few kilobytes of I/O
    even for a build that has been logging for an hour.

    Returns None when there is nothing to say, including when the job log
    directory is not mounted here. That is the same "no view" the log viewer
    already degrades to, and the UI says so rather than implying a stall.
    """
    if not job_id:
        return None
    from .admin import _find_log_files, _read_log_file

    text = ""
    newest: datetime | None = None
    for path in _find_log_files(job_id):
        if not path:
            continue
        chunk, _ = _read_log_file(path, max_bytes=PROGRESS_TAIL_BYTES)
        text += chunk
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
        except OSError:
            continue
        if newest is None or mtime > newest:
            newest = mtime

    progress = images.parse_build_progress(text)
    if progress is None:
        return None
    return ImageBuildProgressOut(
        phase=progress.phase,
        label=progress.label,
        elapsed_seconds=progress.elapsed_seconds,
        downloaded_bytes=progress.downloaded_bytes,
        unpacked_bytes=progress.unpacked_bytes,
        image_bytes=progress.image_bytes,
        bytes_per_second=progress.bytes_per_second,
        last_line=progress.last_line,
        updated_at=newest,
    )


def _build_out(
    row: ImageBuild, progress: ImageBuildProgressOut | None = None
) -> ImageBuildOut:
    return ImageBuildOut(
        id=row.id,
        image_name=row.image_name,
        source_ref=row.source_ref,
        arch=row.arch,
        state=row.state,
        partition=row.partition,
        nodelist=row.nodelist,
        slurm_job_id=row.slurm_job_id,
        requested_by=row.requested_by,
        error=row.error,
        size_bytes=row.size_bytes,
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        progress=progress,
    )


@router.get("", response_model=ImagesResponse)
async def list_images(user: User = Depends(current_user)) -> ImagesResponse:
    _require_admin(user)
    cluster = get_cluster()

    with SessionLocal() as db:
        # Progress is read from the log for active builds only: a finished one
        # cannot move, and its own row already says how it ended.
        builds = [
            _build_out(b, _progress_for(b.slurm_job_id) if b.is_active else None)
            for b in db.execute(
                select(ImageBuild).order_by(ImageBuild.id.desc()).limit(50)
            ).scalars().all()
        ]

    # Where a build could run, from the discovered inventory. An empty list
    # here means node discovery has not succeeded yet, not that the cluster is
    # homogeneous — so the UI keeps the architecture choice open either way.
    inv = inventory.current()
    targets = [
        BuildTargetOut(arch=t.arch, partition=t.partition, pool=t.pool, nodes=list(t.nodes))
        for t in images.build_targets(cluster, list(inv.nodes))
    ]

    building = {b.image_name for b in builds if b.state in ACTIVE_STATES}
    try:
        found = [
            ImageOut(
                name=i.name,
                path=i.path,
                size_bytes=i.size_bytes,
                modified_at=i.modified_at,
                used_by_runtimes=list(i.used_by_runtimes),
                used_by_gpu_classes=list(i.used_by_gpu_classes),
                can_delete=not i.used_by_runtimes and i.name not in building,
            )
            for i in images.list_images(cluster)
        ]
        directory = images.image_dir()
        error = None
    except images.ImageError as exc:
        found, directory, error = [], "", str(exc)

    return ImagesResponse(
        images=found, builds=builds, targets=targets, image_dir=directory, error=error
    )


@router.post("/build", response_model=ImageBuildOut)
async def start_build(
    req: ImageBuildRequest, user: User = Depends(current_user)
) -> ImageBuildOut:
    _require_admin(user)
    cluster = get_cluster()

    try:
        source_ref = images.validate_ref(req.source_ref)
        arch = images.validate_arch(req.arch)
        name = images.validate_name(req.name or images.suggest_name(source_ref, arch))
        target = images.resolve_target(
            cluster, list(inventory.current().nodes), arch, req.partition
        )
        path = images.image_path(name)
    except images.ImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if os.path.exists(path) and not req.overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"{name} already exists. Choose another name, or rebuild it "
                   "explicitly to replace it.",
        )

    with SessionLocal() as db:
        clash = db.execute(
            select(ImageBuild).where(
                ImageBuild.image_name == name,
                ImageBuild.state.in_(ACTIVE_STATES),
            )
        ).scalars().first()
        if clash is not None:
            raise HTTPException(
                status_code=409,
                detail=f"A build of {name} is already {clash.state.lower()} "
                       f"(job {clash.slurm_job_id or '?'}).",
            )

        row = ImageBuild(
            image_name=name,
            source_ref=source_ref,
            arch=arch,
            partition=target.partition,
            nodelist=target.pin_node,
            state="SUBMITTED",
            requested_by=user.sub,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        build_id = row.id

    spec = images.build_job_spec(
        name=name, source_ref=source_ref, target=target, requested_by=user.sub
    )
    try:
        result = await get_backend().submit(spec)
    except (ClusterCommandError, ClusterUnavailableError, OSError) as exc:
        with SessionLocal() as db:
            row = db.get(ImageBuild, build_id)
            if row is not None:
                row.state = "FAILED"
                row.error = f"Submission failed: {exc}"
                db.commit()
        raise HTTPException(status_code=502, detail=f"Could not submit build: {exc}") from exc

    with SessionLocal() as db:
        row = db.get(ImageBuild, build_id)
        row.slurm_job_id = result.job_id
        db.commit()
        db.refresh(row)
        out = _build_out(row)

    log_slurm_action(
        action="image_build",
        model=name,
        slurm_job_id=result.job_id,
        detail=f"{source_ref} -> {arch} on {target.partition} by {user.sub}",
    )
    return out


@router.post("/builds/{build_id}/cancel", response_model=ImageBuildOut)
async def cancel_build(build_id: int, user: User = Depends(current_user)) -> ImageBuildOut:
    _require_admin(user)

    with SessionLocal() as db:
        row = db.get(ImageBuild, build_id)
        if row is None:
            raise HTTPException(status_code=404, detail="No such build.")
        if not row.is_active:
            raise HTTPException(
                status_code=409, detail=f"That build is already {row.state.lower()}."
            )
        job_id = row.slurm_job_id

    if job_id:
        try:
            await get_backend().cancel(job_id)
        except ClusterUnavailableError as exc:
            raise HTTPException(
                status_code=502, detail=f"Could not reach the scheduler: {exc}"
            ) from exc

    with SessionLocal() as db:
        row = db.get(ImageBuild, build_id)
        row.state = "CANCELED"
        row.error = f"Canceled by {user.sub}"
        db.commit()
        db.refresh(row)
        return _build_out(row)


@router.get("/builds/{build_id}/logs", response_model=LogResponse)
async def build_logs(build_id: int, user: User = Depends(current_user)) -> LogResponse:
    """The build job's own stdout/stderr, read the same way lease logs are."""
    _require_admin(user)
    from .admin import _find_log_files, _read_log_file

    with SessionLocal() as db:
        row = db.get(ImageBuild, build_id)
        if row is None:
            raise HTTPException(status_code=404, detail="No such build.")
        job_id = row.slurm_job_id

    if not job_id:
        return LogResponse(slurm_job_id="", log_stdout="", log_stderr="", truncated=False)

    stdout_path, stderr_path = _find_log_files(job_id)
    out, out_trunc = _read_log_file(stdout_path) if stdout_path else ("", False)
    err, err_trunc = _read_log_file(stderr_path) if stderr_path else ("", False)
    return LogResponse(
        slurm_job_id=job_id,
        log_stdout=out,
        log_stderr=err,
        truncated=out_trunc or err_trunc,
    )


@router.delete("/{name}")
async def delete_image(
    name: str, force: bool = False, user: User = Depends(current_user)
) -> dict:
    _require_admin(user)

    with SessionLocal() as db:
        active = db.execute(
            select(ImageBuild).where(
                ImageBuild.image_name == name,
                ImageBuild.state.in_(ACTIVE_STATES),
            )
        ).scalars().first()
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=f"A build is currently writing {name}. Cancel it first.",
        )

    try:
        path = images.delete_image(get_cluster(), name, force=force)
    except images.ImageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    logger.info("image %s deleted by %s", path, user.sub)
    return {"deleted": name, "path": path}
