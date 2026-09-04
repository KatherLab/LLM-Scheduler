"""`templates/apptainer_build.sh`, run for real against a stubbed apptainer.

Worth testing as a script rather than by reading it: this is the code that
decides what lands in the shared images directory, and its failure mode is a
truncated `.sif` under a name jobs will happily launch. The branch that makes
that possible — build on node-local disk, copy across at the end — is chosen
from `df` output at runtime, so both sides of it are exercised here.

Everything external is stubbed onto PATH: `apptainer` (no container runtime in
CI), `df` (two filesystems cannot be conjured in a tmpdir), and GNU `stat`/`du`
(the script targets Linux compute nodes; the shims keep the test honest on
macOS instead of quietly measuring nothing).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess

import pytest

from app import images

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "templates", "apptainer_build.sh")

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="needs bash"
)

#: What the fake `apptainer build` writes, so a short copy is detectable.
IMAGE_CONTENT = b"SIF!" * 4096

APPTAINER_STUB = """#!/usr/bin/env bash
case "$1" in
  --version) echo "apptainer version 1.3.0-stub" ;;
  build)     [[ -n "${FAKE_BUILD_FAILS:-}" ]] && { echo "FATAL: pull failed" >&2; exit 255; }
             printf '%s' "${FAKE_IMAGE_CONTENT}" > "$3" ;;
  exec)      exit 0 ;;
esac
"""

# Absolute paths inside the shims: our stub directory is first on PATH, so
# calling `stat` by name would recurse into this file.
STAT_SHIM = """#!/usr/bin/env bash
if [[ "$(uname -s)" == "Darwin" ]]; then exec /usr/bin/stat -f %z "${@: -1}"; fi
exec /usr/bin/stat "$@"
"""

DU_SHIM = """#!/usr/bin/env bash
d="${@: -1}"
if [[ "$(uname -s)" == "Darwin" ]]; then
  n="$(/usr/bin/find "$d" -type f -exec /usr/bin/stat -f %z {} + 2>/dev/null \
       | /usr/bin/awk '{s+=$1} END {print s+0}')"
  printf '%s\\t%s\\n' "${n:-0}" "$d"; exit 0
fi
exec /usr/bin/du "$@"
"""

# One filesystem or two, decided by FAKE_DF_SPLIT. Only the second column of
# the second line is ever read by the script.
DF_STUB = """#!/usr/bin/env bash
for d in "$@"; do
  [[ "$d" == -* ]] && continue
  echo "Filesystem 1024-blocks Used Available Capacity Mounted-on"
  if [[ -n "${FAKE_DF_SPLIT:-}" && "$d" == *scratch* ]]; then
    echo "/dev/nvme0n1 100 50 50 50% /local"
  else
    echo "nfs:/shared 100 50 50 50% /shared"
  fi
done
"""

TRUNCATING_CP = """#!/usr/bin/env bash
# Half a file, exit 0 — what a full shared filesystem can look like.
/bin/dd if="$2" of="$3" bs=1 count=64 2>/dev/null
"""


class Build:
    def __init__(self, tmp_path):
        self.bin = tmp_path / "bin"
        self.images = tmp_path / "shared" / "images"
        self.scratch = tmp_path / "scratch"
        for d in (self.bin, self.images, self.scratch):
            d.mkdir(parents=True)
        for name, body in (("apptainer", APPTAINER_STUB), ("stat", STAT_SHIM),
                           ("du", DU_SHIM), ("df", DF_STUB)):
            path = self.bin / name
            path.write_text(body)
            path.chmod(0o755)

    def stub(self, name: str, body: str) -> None:
        path = self.bin / name
        path.write_text(body)
        path.chmod(0o755)

    def run(self, **extra) -> subprocess.CompletedProcess:
        env = {
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "BUILD_SOURCE_REF": "docker://vllm/vllm-openai:v0.11.0",
            "BUILD_TARGET": str(self.images / "vllm.sif"),
            "BUILD_SCRATCH": str(self.scratch),
            "BUILD_EXPECTED_ARCH": platform.machine(),
            # Phase changes only. The periodic reporter naps in a `sleep` that
            # outlives it by up to a second, still holding the pipe this test
            # reads — irrelevant under Slurm, where stdout is a file, but it
            # would put a second on every test here. One test below turns it
            # on deliberately.
            "BUILD_PROGRESS_INTERVAL": "0",
            "SLURM_JOB_ID": "4242",
            "FAKE_IMAGE_CONTENT": IMAGE_CONTENT.decode(),
        }
        env.update(extra)
        return subprocess.run(
            ["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=60
        )

    @property
    def published(self):
        return self.images / "vllm.sif"

    def leftovers(self) -> list[str]:
        """Anything either directory should not be holding once a job is over."""
        return sorted(
            p.name
            for d in (self.images, self.scratch)
            for p in d.iterdir()
            if p.name != "vllm.sif"
        )


@pytest.fixture
def build(tmp_path):
    return Build(tmp_path)


def test_one_filesystem_builds_in_place(build):
    r = build.run()
    assert r.returncode == 0, r.stderr
    assert "building in place" in r.stdout
    assert build.published.read_bytes() == IMAGE_CONTENT
    assert build.leftovers() == []


def test_separate_filesystems_build_in_scratch_and_copy_the_finished_file(build):
    """The point of the exercise: the shared filesystem sees one sequential
    write of a finished image instead of every write mksquashfs makes."""
    r = build.run(FAKE_DF_SPLIT="1")
    assert r.returncode == 0, r.stderr
    assert "building into scratch" in r.stdout
    assert "--- copying to" in r.stdout
    assert build.published.read_bytes() == IMAGE_CONTENT
    # Neither the staging copy in scratch nor the landing name in the images
    # directory may survive the job.
    assert build.leftovers() == []


def test_the_image_is_built_in_scratch_not_in_the_images_directory(build):
    """Guards the actual optimisation: if the .sif were still written straight
    to shared storage the copy would be pure overhead, and this test is the
    only thing that would notice."""
    spy = build.scratch / "seen"
    build.stub("apptainer", APPTAINER_STUB.replace(
        'printf \'%s\' "${FAKE_IMAGE_CONTENT}" > "$3"',
        f'echo "$3" >> {spy}; printf \'%s\' "${{FAKE_IMAGE_CONTENT}}" > "$3"',
    ))
    build.run(FAKE_DF_SPLIT="1")
    assert str(build.scratch) in spy.read_text()
    assert str(build.images) not in spy.read_text()


def test_a_short_copy_is_never_published(build):
    build.stub("cp", TRUNCATING_CP)
    r = build.run(FAKE_DF_SPLIT="1")
    assert r.returncode != 0
    assert "NOT published" in r.stderr
    assert not build.published.exists()
    assert build.leftovers() == []


def test_a_failed_build_publishes_nothing(build):
    r = build.run(FAKE_DF_SPLIT="1", FAKE_BUILD_FAILS="1")
    assert r.returncode != 0
    assert not build.published.exists()
    assert build.leftovers() == []


def test_the_heartbeat_it_prints_is_the_one_the_scheduler_parses(build):
    """The script and `parse_build_progress` are one contract; this is the
    only place both halves are exercised together."""
    r = build.run(FAKE_DF_SPLIT="1")
    progress = images.parse_build_progress(r.stdout)
    assert progress.phase == "complete"
    assert progress.image_bytes == len(IMAGE_CONTENT)
    assert "PROGRESS phase=publishing" in r.stdout


def test_the_periodic_reporter_keeps_talking_during_a_long_step(build):
    """The phase lines above are printed synchronously; this is the part that
    turns a silent ten-minute pull into something a user can watch."""
    build.stub("apptainer", APPTAINER_STUB.replace(
        "build)     ", "build)     sleep 2; ",
    ))
    r = build.run(BUILD_PROGRESS_INTERVAL="1")
    beats = [ln for ln in r.stdout.splitlines() if ln.startswith("PROGRESS phase=building")]
    # One of these is `set_phase building` itself, so two means the reporter
    # spoke on its own at least once.
    assert len(beats) >= 2, r.stdout
