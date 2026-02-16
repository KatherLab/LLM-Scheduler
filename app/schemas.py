from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Any

class LeaseCreate(BaseModel):
    model: str
    owner: Optional[str] = None
    # schedule:
    begin_at: Optional[datetime] = None
    duration_seconds: int = Field(default=6*3600, ge=60)

    # optional overrides; if omitted use catalog defaults
    gpus: Optional[int] = Field(default=None, ge=1)
    tensor_parallel_size: Optional[int] = Field(default=None, ge=1)
    gpu_memory_utilization: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    extra_args: Optional[str] = None
    tool_args: Optional[str] = None
    reasoning_parser: Optional[str] = None

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

class LeaseExtend(BaseModel):
    duration_seconds: int = Field(..., ge=60)

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
