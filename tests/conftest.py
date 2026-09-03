"""Test-wide environment.

Set before any `app.*` import so `Settings()` picks it up: the default
DATABASE_URL points at /var/lib and the default backend shells out to Slurm,
neither of which exists in a test run.
"""

import os
import tempfile

os.environ.setdefault("DATABASE_URL", f"sqlite:///{tempfile.mkdtemp()}/test.db")
os.environ.setdefault("CLUSTER_BACKEND", "local")
os.environ.setdefault("AUTH_PASSWORD", "test-break-glass")
