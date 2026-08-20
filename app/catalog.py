from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import shlex
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
    cpus: int | None = None
    mem: str | None = None
    env: dict[str, str] | None = None
    tags: list[str] | None = None


def _flag_name(token: str) -> str:
    """`--max-model-len=4096` -> `--max-model-len`, otherwise the token itself."""
    return token.split("=", 1)[0]


def merge_args(defaults: str, model: str) -> str:
    """
    Prepend shared CLI args to a model's own args.

    A default is dropped when the model sets the same flag, so a model can
    override a common option instead of passing it twice
    (e.g. default `--max-model-len 8192` + model `--max-model-len 2048`).
    """
    defaults = (defaults or "").strip()
    model = (model or "").strip()
    if not defaults:
        return model
    if not model:
        return defaults

    try:
        default_tokens = shlex.split(defaults)
        model_tokens = shlex.split(model)
    except ValueError:
        # Unbalanced quotes — leave both strings untouched and let vLLM complain.
        return f"{defaults} {model}"

    model_flags = {_flag_name(t) for t in model_tokens if t.startswith("-")}

    kept: list[str] = []
    keep_group = True
    for token in default_tokens:
        if token.startswith("-"):
            keep_group = _flag_name(token) not in model_flags
        if keep_group:
            kept.append(token)

    return " ".join(shlex.join(kept + model_tokens).split())


def load_catalog(path: str) -> dict[str, CatalogModel]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    defaults = data.get("defaults") or {}
    out: dict[str, CatalogModel] = {}
    for item in data.get("models", []):
        merged = {**defaults, **item}
        env = {**(defaults.get("env") or {}), **(item.get("env") or {})}
        m = CatalogModel(
            name=item["name"],
            model_path=item["model_path"],
            gpus=int(merged["gpus"]),
            tensor_parallel_size=int(merged["tensor_parallel_size"]),
            gpu_memory_utilization=float(merged.get("gpu_memory_utilization", 0.95)),
            extra_args=merge_args(
                str(defaults.get("extra_args", "") or ""),
                str(item.get("extra_args", "") or ""),
            ),
            tool_args=merge_args(
                str(defaults.get("tool_args", "") or ""),
                str(item.get("tool_args", "") or ""),
            ),
            reasoning_parser=merged.get("reasoning_parser"),
            venv_activate=merged.get("venv_activate"),
            notes=str(item.get("notes", "") or ""),
            cpus=int(merged["cpus"]) if merged.get("cpus") else None,
            mem=str(merged["mem"]) if merged.get("mem") else None,
            env=env or None,
            tags=list(item["tags"]) if item.get("tags") else None,
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
