#!/usr/bin/env bash
# Build one Apptainer image from a container registry.
#
# This runs as a Slurm job because Apptainer cannot cross-build: an aarch64
# .sif can only be produced on an aarch64 node. The scheduler picks a node of
# the right architecture and this script does the rest.
#
# Everything arrives through the environment, never interpolated into a
# command string — the source reference is user input and must not be able to
# become a second argument or a shell metacharacter.
#
#   BUILD_SOURCE_REF     docker://vllm/vllm-openai:v0.11.0
#   BUILD_TARGET         /shared/images/vllm-0.11.0-x86_64.sif
#   BUILD_SCRATCH        working area: blob cache, unpacked rootfs, and — when
#                        it is on a different filesystem from the images
#                        directory — the .sif itself while it is being built
#   BUILD_EXPECTED_ARCH  x86_64 | aarch64 — verified before doing any work
#   BUILD_PROGRESS_INTERVAL  seconds between progress heartbeats (default 15,
#                        0 reports phase changes only)
set -euo pipefail

echo "Image build job ${SLURM_JOB_ID:-?} on ${SLURMD_NODENAME:-$(hostname)}"
echo "Source: ${BUILD_SOURCE_REF}"
echo "Target: ${BUILD_TARGET}"

for var in BUILD_SOURCE_REF BUILD_TARGET BUILD_SCRATCH BUILD_EXPECTED_ARCH; do
  if [[ -z "${!var:-}" ]]; then
    echo "ERROR: ${var} is empty" >&2
    exit 2
  fi
done

# Landing on the wrong architecture would produce a .sif that fails much later,
# on a node far away from this log. Check it while the check is still cheap.
ACTUAL_ARCH="$(uname -m)"
if [[ "${ACTUAL_ARCH}" != "${BUILD_EXPECTED_ARCH}" ]]; then
  echo "ERROR: asked for ${BUILD_EXPECTED_ARCH} but this node is ${ACTUAL_ARCH}." >&2
  echo "Apptainer cannot cross-build, so this job would produce the wrong image." >&2
  exit 2
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer is not installed on ${SLURMD_NODENAME:-this node}" >&2
  exit 2
fi
apptainer --version

JOB_ID="${SLURM_JOB_ID:-$$}"

# Per-job, not per-arch: two builds of the same architecture can run
# concurrently, and the cleanup trap below rm -rf's these directories, so
# sharing them across jobs would let one build's cleanup delete another
# build's cache out from under it.
: "${APPTAINER_CACHEDIR:=${BUILD_SCRATCH}/cache-${BUILD_EXPECTED_ARCH}-${JOB_ID}}"
: "${APPTAINER_TMPDIR:=${BUILD_SCRATCH}/tmp-${BUILD_EXPECTED_ARCH}-${JOB_ID}}"
export APPTAINER_CACHEDIR APPTAINER_TMPDIR

mkdir -p "${BUILD_SCRATCH}" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" \
         "$(dirname "${BUILD_TARGET}")"

# A .sif must appear under its final name only once it is complete: a crashed
# or cancelled build that left a truncated image would be worse than no image,
# because a model job would happily try to run it. So the last step is always a
# rename within the images directory, which is atomic.
#
# Where the *build* happens is a separate question, and it matters for speed.
# mksquashfs writes the image with many small writes; doing that against NFS
# pays the round trip on every one of them. When BUILD_SCRATCH is on a
# different filesystem — the whole point of pointing it at node-local disk —
# the image is built there and copied over as one finished file at the end, so
# the shared filesystem sees a single sequential write instead of the whole
# squash. When scratch and images share a filesystem there is nothing to gain
# and a full extra copy to lose, so the build stays in place.
#
# If `df` cannot answer, we assume in-place: that is the historical path, and
# an unnecessary copy is a worse failure than a missed optimisation.
IMAGE_DIR="$(dirname "${BUILD_TARGET}")"
fs_of() { df -P "$1" 2>/dev/null | awk 'NR==2 {print $1}'; }
SCRATCH_FS="$(fs_of "${BUILD_SCRATCH}")"
IMAGES_FS="$(fs_of "${IMAGE_DIR}")"

if [[ -n "${SCRATCH_FS}" && -n "${IMAGES_FS}" && "${SCRATCH_FS}" != "${IMAGES_FS}" ]]; then
  BUILD_LOCALLY=1
  STAGING="${BUILD_SCRATCH}/$(basename "${BUILD_TARGET}").building.${JOB_ID}"
else
  BUILD_LOCALLY=0
  STAGING="${BUILD_TARGET}.building.${JOB_ID}"
fi
#: Only used when the image is built elsewhere: the landing name in the images
#: directory, so the copy is renamed into place rather than written into place.
INCOMING="${BUILD_TARGET}.incoming.${JOB_ID}"

PHASE_FILE="${BUILD_SCRATCH}/phase-${JOB_ID}"
cleanup() {
  # The reporter first: it reads directories the next two lines delete.
  if [[ -n "${REPORTER_PID:-}" ]]; then
    kill "${REPORTER_PID}" 2>/dev/null || true
  fi
  rm -f "${STAGING}" "${INCOMING}" "${PHASE_FILE}"
  # Unpacked layers run several times the size of the finished .sif — leaving
  # them behind would silently fill BUILD_SCRATCH across every build. Safe to
  # remove unconditionally: this cache/tmp pair is scoped to this job alone.
  rm -rf "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
}
trap cleanup EXIT

# ── Progress heartbeat ───────────────────────────────────────────────────────
# Without this a build is several minutes of silence: apptainer's own progress
# bars are drawn for a terminal and collapse to almost nothing once stdout is a
# file. So the script publishes its own, one machine-readable line at a fixed
# interval, which the scheduler parses out of the job log:
#
#   PROGRESS phase=<phase> elapsed=<s> cache_bytes=<n> tmp_bytes=<n> sif_bytes=<n>
#
# Only the coarse phase is stated here. Downloading versus extracting follows
# from *which* counter is growing, and that is a comparison between two
# heartbeats — left to the reader rather than guessed at here. A counter we
# cannot measure is reported as -1, never as 0: "unknown" and "nothing yet"
# look identical otherwise, and only one of them is a reason to worry.
#
# The interval is 15s rather than a couple of seconds because the tmpdir holds
# an unpacked container rootfs — a `du` over a few hundred thousand files is
# not free, and nothing here is worth spending build CPU on.
PROGRESS_INTERVAL="${BUILD_PROGRESS_INTERVAL:-15}"
START_TS="$(date +%s)"

dir_bytes() {
  local n
  n="$(du -sxb "$1" 2>/dev/null | cut -f1)" || n=""
  echo "${n:--1}"
}

first_bytes() {   # size of the first of these paths that exists, else 0
  local p n
  for p in "$@"; do
    n="$(stat -c %s "${p}" 2>/dev/null)" || n=""
    if [[ -n "${n}" ]]; then echo "${n}"; return 0; fi
  done
  echo 0
}

# The image has three names over its life and the phase says which one is live.
# Asking in a fixed order instead would report the *old* image for the whole
# build whenever one is being replaced, and would make the copy look frozen
# because the finished local file outranks the growing one.
image_bytes() {
  case "$1" in
    publishing) first_bytes "${INCOMING}" "${BUILD_TARGET}" ;;
    complete)   first_bytes "${BUILD_TARGET}" ;;
    *)          first_bytes "${STAGING}" ;;
  esac
}

progress_line() {
  local phase
  phase="$(cat "${PHASE_FILE}" 2>/dev/null || echo unknown)"
  printf 'PROGRESS phase=%s elapsed=%s cache_bytes=%s tmp_bytes=%s sif_bytes=%s\n' \
    "${phase}" \
    "$(( $(date +%s) - START_TS ))" \
    "$(dir_bytes "${APPTAINER_CACHEDIR}")" \
    "$(dir_bytes "${APPTAINER_TMPDIR}")" \
    "$(image_bytes "${phase}")"
}

set_phase() {
  echo "$1" > "${PHASE_FILE}"
  progress_line || true
}

# One-second naps rather than one long one: killing the reporter at exit
# cannot interrupt a `sleep`, only the loop around it, so a long sleep would
# be orphaned still holding this job's stdout open. A second of that is fine;
# a quarter of a minute of it delays whoever is waiting for the log to close.
reporter() {
  local waited=0
  while true; do
    sleep 1
    waited=$(( waited + 1 ))
    if (( waited >= PROGRESS_INTERVAL )); then
      progress_line || true
      waited=0
    fi
  done
}

set_phase preparing
if (( PROGRESS_INTERVAL > 0 )); then
  reporter &
  REPORTER_PID=$!
  # Off the job table, so killing it at exit does not print "Terminated" into
  # the job's stderr — which reads as a failure in a log that ends in success.
  disown "${REPORTER_PID}" 2>/dev/null || true
else
  echo "Periodic progress heartbeat disabled (BUILD_PROGRESS_INTERVAL=0);"
  echo "phase changes are still reported."
fi

echo "--- disk before ---"
df -h "${BUILD_SCRATCH}" "${IMAGE_DIR}" 2>&1 || true

if [[ "${BUILD_LOCALLY}" == 1 ]]; then
  echo "Scratch (${SCRATCH_FS}) and images (${IMAGES_FS}) are separate filesystems:"
  echo "building into scratch, then copying the finished .sif across."
  echo "Scratch must hold the blob cache, the unpacked rootfs AND the image."
else
  echo "Scratch and images share a filesystem: building in place, nothing to copy."
fi

echo "--- building ---"
set_phase building
# --force applies to the staging path only, which we own for this job.
time apptainer build --force "${STAGING}" "${BUILD_SOURCE_REF}"

if [[ ! -s "${STAGING}" ]]; then
  echo "ERROR: build reported success but produced no image" >&2
  exit 3
fi

# Verify it actually runs on this architecture before publishing it. A .sif
# that cannot execute is worse than no .sif: it looks like a working upgrade.
echo "--- verifying ---"
set_phase verifying
apptainer exec "${STAGING}" /bin/true

set_phase publishing
if [[ "${BUILD_LOCALLY}" == 1 ]]; then
  echo "--- copying to ${IMAGE_DIR} ---"
  cp -f "${STAGING}" "${INCOMING}"

  # A short copy is the one way this path can publish a broken image, and it
  # is silent: cp can exit 0 having written less than it read only in odd
  # cases, but ENOSPC on the shared filesystem is not odd at all.
  SRC_BYTES="$(first_bytes "${STAGING}")"
  DST_BYTES="$(first_bytes "${INCOMING}")"
  if [[ "${SRC_BYTES}" != "${DST_BYTES}" ]]; then
    echo "ERROR: copied ${DST_BYTES} of ${SRC_BYTES} bytes to ${IMAGE_DIR}." >&2
    echo "The image was NOT published. Check free space there." >&2
    exit 4
  fi

  # Same directory as the target, so this is an atomic rename: the final name
  # never exists holding a partial file.
  mv -f "${INCOMING}" "${BUILD_TARGET}"
  rm -f "${STAGING}"
else
  mv -f "${STAGING}" "${BUILD_TARGET}"
fi

# Cleanup trap stays armed: it still needs to remove APPTAINER_CACHEDIR and
# APPTAINER_TMPDIR on this, the success path. Removing the staging and incoming
# files is a no-op by now — both have been renamed or deleted above.
chmod 0644 "${BUILD_TARGET}"
ls -l "${BUILD_TARGET}"
set_phase complete
echo "Build complete: ${BUILD_TARGET}"
