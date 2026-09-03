"""Endpoint selection across replicas of the same model.

`choose_ready_endpoint` used to return the *newest* READY endpoint, which is
fine with one instance per model and useless with several: all traffic lands on
one replica while the others idle.

Two signals decide instead:

* **in-flight requests we dispatched** — exact, real-time, and free, because
  this process *is* the proxy. This is the primary signal.
* **vLLM's own queue depth** — scraped periodically. It catches state we cannot
  see from request counts alone (a replica still loading weights, or one whose
  engine is backed up), so it breaks ties.

Deliberately not weighted-by-latency: with streaming responses, request
duration mostly measures how much the *client* asked for, not how loaded the
replica is.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass

#: Scraped load older than this is ignored — a stale queue depth is worse than
#: no queue depth, because it keeps steering traffic at a replica that has
#: since drained.
SCRAPE_TTL_SECONDS = 90.0


@dataclass
class _Load:
    in_flight: int = 0
    pending: int | None = None      # vLLM num_requests_waiting
    running: int | None = None      # vLLM num_requests_running
    scraped_at: float = 0.0
    #: Set while a replica is being drained for a rolling restart: it keeps
    #: serving what it has but must not be given new work.
    draining: bool = False


class LoadRegistry:
    """In-process load per endpoint, keyed by ``host:port``.

    Per-process by design. With several router replicas each sees only its own
    in-flight count, which still balances correctly as long as they receive
    comparable traffic; the shared vLLM queue depth corrects the rest.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._loads: dict[str, _Load] = {}

    @staticmethod
    def key(host: str, port: int) -> str:
        return f"{host}:{port}"

    def _entry(self, key: str) -> _Load:
        load = self._loads.get(key)
        if load is None:
            load = _Load()
            self._loads[key] = load
        return load

    # ── in-flight accounting ────────────────────────────────────────────────
    def acquire(self, key: str) -> None:
        with self._lock:
            self._entry(key).in_flight += 1

    def release(self, key: str) -> None:
        with self._lock:
            load = self._entry(key)
            # Clamp: a release without a matching acquire (restart mid-flight)
            # must not drive the count negative and pin traffic to one replica.
            load.in_flight = max(0, load.in_flight - 1)

    def in_flight(self, key: str) -> int:
        with self._lock:
            return self._entry(key).in_flight

    # ── scraped signal ──────────────────────────────────────────────────────
    def record_scrape(
        self, key: str, *, pending: int | None, running: int | None
    ) -> None:
        with self._lock:
            load = self._entry(key)
            load.pending = pending
            load.running = running
            load.scraped_at = time.monotonic()

    # ── draining ────────────────────────────────────────────────────────────
    def set_draining(self, key: str, draining: bool = True) -> None:
        """Stop sending new work here without killing what is in flight."""
        with self._lock:
            self._entry(key).draining = draining

    def is_draining(self, key: str) -> bool:
        with self._lock:
            return self._entry(key).draining

    def drained(self, key: str) -> bool:
        """True once a draining endpoint has finished its in-flight work."""
        with self._lock:
            load = self._entry(key)
            return load.draining and load.in_flight == 0

    # ── selection ───────────────────────────────────────────────────────────
    def score(self, key: str) -> tuple[int, int]:
        """Lower is better: ``(in_flight, queue_depth)``.

        Stale scrapes contribute 0 rather than their last value.
        """
        with self._lock:
            load = self._entry(key)
            queue = 0
            if load.scraped_at and (time.monotonic() - load.scraped_at) <= SCRAPE_TTL_SECONDS:
                queue = (load.pending or 0) + (load.running or 0)
            return load.in_flight, queue

    def forget(self, key: str) -> None:
        with self._lock:
            self._loads.pop(key, None)

    def snapshot(self) -> dict[str, dict]:
        """For the dashboard and tests."""
        with self._lock:
            return {
                key: {
                    "in_flight": load.in_flight,
                    "pending": load.pending,
                    "running": load.running,
                    "draining": load.draining,
                }
                for key, load in self._loads.items()
            }

    def reset(self) -> None:
        with self._lock:
            self._loads.clear()


REGISTRY = LoadRegistry()


def choose_least_loaded(endpoints, registry: LoadRegistry | None = None):
    """Pick the least-loaded endpoint from `endpoints`.

    Draining endpoints are skipped entirely unless they are all that is left —
    refusing to serve is worse than sending work to a replica that is about to
    be replaced.

    Ties are broken randomly rather than by id, so a burst of concurrent
    requests arriving before any of them register as in-flight does not all
    land on the same replica.
    """
    endpoints = list(endpoints)
    if not endpoints:
        return None

    registry = registry or REGISTRY
    live = [e for e in endpoints if not registry.is_draining(LoadRegistry.key(e.host, e.port))]
    candidates = live or endpoints

    scored = [(registry.score(LoadRegistry.key(e.host, e.port)), e) for e in candidates]
    best = min(s for s, _ in scored)
    tied = [e for s, e in scored if s == best]
    return random.choice(tied)
