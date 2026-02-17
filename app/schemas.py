from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Any

class LeaseCreate(BaseModel):
    model: str
    owner: Optional[str] = None
    begin_at: Optional[datetime] = None
    duration_seconds: int = Field(default=6*3600, ge=60)
    asap: bool = False  # NEW: find earliest available slot

    gpus: Optional[int] = Field(default=None, ge=1)
    tensor_parallel_size: Optional[int] = Field(default=None, ge=1)
    gpu_memory_utilization: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    extra_args: Optional[str] = None
    tool_args: Optional[str] = None
    reasoning_parser: Optional[str] = None

class LeaseUpdate(BaseModel):
    begin_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    requested_gpus: Optional[int] = Field(default=None, ge=1)
    requested_tp: Optional[int] = Field(default=None, ge=1)

class LeaseOut(BaseModel):
    id: int
    model: str
    owner: Optional[str]
    state: str
    slurm_job_id: Optional[str]
    host: str
    port: int
    requested_gpus: int
    requested_tp: int
    begin_at: Optional[datetime]
    end_at: Optional[datetime]
    created_at: datetime

    lane_start: Optional[int] = None
    lane_count: Optional[int] = None
    conflict: bool = False

class LeaseExtend(BaseModel):
    duration_seconds: int = Field(..., ge=60)

class LeaseShortenRequest(BaseModel):
    """Shorten a running/submitted lease to a new end time."""
    new_end_at: datetime

class EndpointRegister(BaseModel):
    slurm_job_id: str
    model: str
    host: str
    port: int

class EndpointOut(BaseModel):
    id: int
    model: str
    host: str
    port: int
    slurm_job_id: str
    state: str
    last_health_at: Optional[datetime]
    last_error: Optional[str]
    created_at: datetime

class OpenAIModelsResponse(BaseModel):
    object: str = "list"
    data: list[dict[str, Any]]

class DashboardModel(BaseModel):
    id: str
    ready: bool
    meta: dict[str, Any]

class EndpointStats(BaseModel):
    model: str
    host: str
    port: int
    state: str
    slurm_job_id: str
    last_health_at: Optional[datetime]
    # vLLM /metrics or /v1/models based stats
    active_requests: Optional[int] = None
    pending_requests: Optional[int] = None
    gpu_cache_usage: Optional[float] = None
    uptime_seconds: Optional[float] = None

class DashboardResponse(BaseModel):
    now: datetime
    total_gpus: int
    models: list[DashboardModel]
    leases: list[LeaseOut]
    endpoint_stats: list[EndpointStats] = []

class LogResponse(BaseModel):
    slurm_job_id: str
    log_stdout: str
    log_stderr: str
    truncated: bool = False
