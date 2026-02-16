from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
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
    notes: str = ""

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
            notes=str(item.get("notes", "") or ""),
        )
        out[m.name] = m
    return out
