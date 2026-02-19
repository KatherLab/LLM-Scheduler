from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import yaml

@dataclass(frozen=True)
class CatalogModel:
    name: str
    model_path: str
    gpus: int
    tensor_parallel_size: int
    gpu_memory_utilization: float = 0.95
    extra_args: str = ""
    tool_args: str = ""
    reasoning_parser: str | None = None
    venv_activate: str | None = None
    notes: str = ""
    cpus: int | None = None        # NEW: per-model CPU cores
    mem: str | None = None          # NEW: per-model memory (e.g. "64G", "128000M")


def load_catalog(path: str) -> dict[str, CatalogModel]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    out: dict[str, CatalogModel] = {}
    for item in data.get("models", []):
        m = CatalogModel(
            name=item["name"],
            model_path=item["model_path"],
            gpus=int(item["gpus"]),
            tensor_parallel_size=int(item["tensor_parallel_size"]),
            gpu_memory_utilization=float(item.get("gpu_memory_utilization", 0.95)),
            extra_args=str(item.get("extra_args", "") or ""),
            tool_args=str(item.get("tool_args", "") or ""),
            reasoning_parser=item.get("reasoning_parser"),
            venv_activate=item.get("venv_activate"),
            notes=str(item.get("notes", "") or ""),
            cpus=int(item["cpus"]) if item.get("cpus") else None,
            mem=str(item["mem"]) if item.get("mem") else None,
        )
        out[m.name] = m
    return out


# ---------------------------------------------------------------------------
# Auto-reloading catalog: re-reads models.yaml only when the file changes
# ---------------------------------------------------------------------------
_catalog_cache: dict[str, CatalogModel] | None = None
_catalog_mtime: float = 0.0
_catalog_lock = Lock()
_CATALOG_PATH = "config/models.yaml"


def get_catalog(path: str = _CATALOG_PATH) -> dict[str, CatalogModel]:
    """Return the catalog, reloading from disk if the file's mtime has changed."""
    global _catalog_cache, _catalog_mtime

    p = Path(path)
    try:
        current_mtime = p.stat().st_mtime
    except OSError:
        if _catalog_cache is not None:
            return _catalog_cache
        raise

    # Fast path: no change
    if _catalog_cache is not None and current_mtime <= _catalog_mtime:
        return _catalog_cache

    with _catalog_lock:
        # Double-check after acquiring lock
        if _catalog_cache is not None and current_mtime <= _catalog_mtime:
            return _catalog_cache

        _catalog_cache = load_catalog(path)
        _catalog_mtime = current_mtime
        print(f"catalog: reloaded {len(_catalog_cache)} models from {path}")
        return _catalog_cache
