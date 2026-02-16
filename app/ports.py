from __future__ import annotations
import fcntl
import json
import os
import socket
from dataclasses import dataclass
from typing import Optional

@dataclass
class PortAllocator:
    port_min: int
    port_max: int
    state_path: str = "/var/run/vllm-router/ports.json"
    lock_path: str = "/var/run/vllm-router/ports.lock"

    def _port_in_use(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.1)
            return s.connect_ex(("127.0.0.1", port)) == 0

    def allocate(self, key: str) -> int:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)

        with open(self.lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)

            used: dict[str, int] = {}
            if os.path.exists(self.state_path):
                try:
                    with open(self.state_path, "r") as f:
                        used = {k: int(v) for k, v in json.load(f).items()}
                except Exception:
                    used = {}

            # clean stale
            used = {k: v for k, v in used.items() if self._port_in_use(v)}

            used_ports = set(used.values())
            for p in range(self.port_min, self.port_max + 1):
                if p in used_ports:
                    continue
                if self._port_in_use(p):
                    continue
                used[key] = p
                with open(self.state_path, "w") as f:
                    json.dump({k: v for k, v in used.items()}, f)
                return p

        raise RuntimeError("No free ports in pool")

    def release(self, key: str) -> None:
        if not os.path.exists(self.state_path):
            return
        with open(self.lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                with open(self.state_path, "r") as f:
                    used = json.load(f)
            except Exception:
                return
            if key in used:
                del used[key]
                with open(self.state_path, "w") as f:
                    json.dump(used, f)
