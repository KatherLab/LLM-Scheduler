from __future__ import annotations
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
import shlex
import yaml


@dataclass(frozen=True)
class ModelRequirements:
    """Which GPU classes a model can run on at all."""

    #: Floor on a *single* GPU's memory. Weights are sharded across the TP
    #: group, so this is what one shard plus its KV cache must fit into.
    min_vram_gb: int = 0
    #: Floor on the memory of the whole TP group. `min_vram_gb` alone cannot
    #: express "needs 8x H200": it would happily accept 2 of them, and the
    #: booking would queue, allocate, and only then OOM at load.
    min_total_vram_gb: float = 0.0
    arch: str | None = None
    #: Explicit allow-list; empty means "any class that satisfies min_vram_gb".
    gpu_classes: tuple[str, ...] = ()

    def allows(self, gpu_class) -> bool:
        """`gpu_class` is a `cluster.GpuClass` (duck-typed to keep the import out)."""
        if gpu_class is None:
            return not self.gpu_classes and self.min_vram_gb == 0
        if self.gpu_classes and gpu_class.name not in self.gpu_classes:
            return False
        if self.min_vram_gb and gpu_class.vram_gb < self.min_vram_gb:
            return False
        if self.arch and gpu_class.arch != self.arch:
            return False
        return True

    def fits_on(self, gpu_class, gpus: int) -> bool:
        """Whether `gpus` cards of this class give the model enough memory.

        Checked before submission: a job that cannot possibly load is worth
        refusing immediately rather than after it has queued for hours.
        """
        if not self.allows(gpu_class):
            return False
        if not self.min_total_vram_gb:
            return True
        if gpu_class is None:
            return False
        return gpu_class.usable_gb * max(1, gpus) >= self.min_total_vram_gb

    def shortfall(self, gpu_class, gpus: int) -> str | None:
        """A message naming the gap, or None when it fits."""
        if self.fits_on(gpu_class, gpus):
            return None
        if gpu_class is None:
            return "no GPU class selected"
        have = gpu_class.usable_gb * max(1, gpus)
        if self.min_total_vram_gb and have < self.min_total_vram_gb:
            return (
                f"needs {self.min_total_vram_gb:.0f} GB across the TP group but "
                f"{gpus}x {gpu_class.name} gives only {have:.0f} GB usable"
            )
        if self.min_vram_gb and gpu_class.vram_gb < self.min_vram_gb:
            return (
                f"needs {self.min_vram_gb} GB per GPU but {gpu_class.name} "
                f"has {gpu_class.vram_gb} GB"
            )
        return f"cannot run on {gpu_class.name}"


@dataclass(frozen=True)
class CatalogModel:
    name: str
    model_path: str
    gpus: int
    tensor_parallel_size: int
    #: Fraction of the card to take when running **alone**. The default is
    #: greedy on purpose: whatever is not weights becomes KV cache, and more KV
    #: cache is almost always better.
    gpu_memory_utilization: float = 0.95
    #: Absolute budget in GB, used when this model **shares** a GPU. Required
    #: to co-locate, because fractions of a shared card cannot be made to add
    #: up but absolute budgets can. It does *not* cap the solo case — running
    #: alone should still take the whole card.
    memory_gb: float | None = None
    extra_args: str = ""
    tool_args: str = ""
    reasoning_parser: str | None = None
    venv_activate: str | None = None
    notes: str = ""
    cpus: int | None = None
    mem: str | None = None
    env: dict[str, str] | None = None
    tags: list[str] | None = None
    #: Runtime override; normally the GPU class decides.
    runtime: str | None = None
    requires: ModelRequirements = field(default_factory=ModelRequirements)
    #: Per-GPU-class overrides, keyed by class name. Raw dicts — resolved by
    #: `resolve_variant()` so the base entry stays usable without a cluster.
    variants: dict[str, dict] = field(default_factory=dict)

    @property
    def min_vram_gb(self) -> float:
        """Smallest card this model can run on at all.

        `requires.min_vram_gb` is the explicit answer. Falling back to
        `memory_gb` saves declaring the same number twice: a model that needs
        10 GB to co-locate needs at least 10 GB to run.
        """
        if self.requires.min_vram_gb:
            return float(self.requires.min_vram_gb)
        return float(self.memory_gb or 0)

    def supports_class(self, gpu_class) -> bool:
        """A variant is an explicit statement of support; otherwise `requires`."""
        if gpu_class is not None and gpu_class.name in self.variants:
            return True
        if not self.requires.allows(gpu_class):
            return False
        # Implied floor from memory_gb, so it need not be repeated in requires.
        if gpu_class is not None and self.min_vram_gb:
            return gpu_class.usable_gb >= self.min_vram_gb
        return True


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
            memory_gb=(
                float(merged["memory_gb"]) if merged.get("memory_gb") is not None else None
            ),
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
            runtime=merged.get("runtime"),
            requires=_parse_requirements(merged.get("requires")),
            variants={
                str(k): dict(v or {})
                for k, v in (item.get("variants") or {}).items()
            },
        )
        out[m.name] = m
    return out


def _parse_requirements(raw) -> ModelRequirements:
    if not raw:
        return ModelRequirements()
    classes = raw.get("gpu_classes") or []
    if isinstance(classes, str):
        classes = [c.strip() for c in classes.split(",") if c.strip()]
    return ModelRequirements(
        min_vram_gb=int(raw.get("min_vram_gb", 0) or 0),
        min_total_vram_gb=float(raw.get("min_total_vram_gb", 0) or 0),
        arch=raw.get("arch"),
        gpu_classes=tuple(str(c) for c in classes),
    )


def resolve_variant(model: CatalogModel, gpu_class_name: str | None) -> CatalogModel:
    """Apply the per-GPU-class overrides for where this model will actually land.

    Replaces "gpus/tp are fixed per model" with "gpus/tp are a function of the
    hardware", which is what a heterogeneous cluster requires: the same weights
    need tp=2 on a 96 GB card and tp=8 on a 24 GB one.

    Layering is `defaults -> model -> variants[class]`, and `extra_args` /
    `tool_args` go through the same `merge_args()` flag-override semantics as
    the defaults block, so a variant can override a flag rather than duplicate
    it. Returns the model unchanged when there is no variant.
    """
    if not gpu_class_name:
        return model
    variant = model.variants.get(gpu_class_name)
    if not variant:
        return model

    changes: dict = {}
    for key in ("gpus", "cpus"):
        if variant.get(key) is not None:
            changes[key] = int(variant[key])
    # `tp` is accepted as an alias — it is what people actually write.
    tp = variant.get("tensor_parallel_size", variant.get("tp"))
    if tp is not None:
        changes["tensor_parallel_size"] = int(tp)
    if variant.get("gpu_memory_utilization") is not None:
        changes["gpu_memory_utilization"] = float(variant["gpu_memory_utilization"])
    if variant.get("memory_gb") is not None:
        changes["memory_gb"] = float(variant["memory_gb"])
    for key in ("mem", "reasoning_parser", "venv_activate", "model_path", "runtime"):
        if variant.get(key) is not None:
            changes[key] = variant[key]
    for key in ("extra_args", "tool_args"):
        if variant.get(key) is not None:
            changes[key] = merge_args(getattr(model, key), str(variant[key]))
    if variant.get("env"):
        changes["env"] = {**(model.env or {}), **{
            str(k): str(v) for k, v in variant["env"].items()
        }}

    return replace(model, **changes)


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
