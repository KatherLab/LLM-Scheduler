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
#   BUILD_SCRATCH        working area for layer unpacking and the blob cache
#   BUILD_EXPECTED_ARCH  x86_64 | aarch64 — verified before doing any work
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

mkdir -p "${BUILD_SCRATCH}" "${APPTAINER_CACHEDIR:-${BUILD_SCRATCH}/cache}" \
         "${APPTAINER_TMPDIR:-${BUILD_SCRATCH}/tmp}" "$(dirname "${BUILD_TARGET}")"

# Build to a temporary name in the SAME directory, then rename. A .sif appears
# under its final name only once it is complete, so a crashed or cancelled
# build cannot leave a truncated image that a model job would happily try to
# run. Same directory keeps the rename atomic — across filesystems it is a copy.
STAGING="${BUILD_TARGET}.building.${SLURM_JOB_ID:-$$}"
cleanup() {
  rm -f "${STAGING}"
}
trap cleanup EXIT

echo "--- disk before ---"
df -h "${BUILD_SCRATCH}" "$(dirname "${BUILD_TARGET}")" 2>&1 || true

echo "--- building ---"
# --force applies to the staging path only, which we own for this job.
time apptainer build --force "${STAGING}" "${BUILD_SOURCE_REF}"

if [[ ! -s "${STAGING}" ]]; then
  echo "ERROR: build reported success but produced no image" >&2
  exit 3
fi

# Verify it actually runs on this architecture before publishing it. A .sif
# that cannot execute is worse than no .sif: it looks like a working upgrade.
echo "--- verifying ---"
apptainer exec "${STAGING}" /bin/true

mv -f "${STAGING}" "${BUILD_TARGET}"
trap - EXIT

chmod 0644 "${BUILD_TARGET}"
ls -l "${BUILD_TARGET}"
echo "Build complete: ${BUILD_TARGET}"
