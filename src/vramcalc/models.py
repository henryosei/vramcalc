"""Pydantic schemas — the API contract."""

from typing import Literal

from pydantic import BaseModel, Field

Dtype = Literal["fp32", "fp16", "bf16", "fp8", "int8", "int4"]


class ModelSpec(BaseModel):
    """Architecture facts needed to size a model in memory."""

    name: str
    params_billion: float = Field(gt=0, description="Total parameter count in billions")
    num_layers: int = Field(gt=0)
    num_kv_heads: int = Field(gt=0, description="KV heads (grouped-query attention)")
    head_dim: int = Field(gt=0)


class EstimateRequest(BaseModel):
    """Either name a preset or supply a full custom spec."""

    preset: str | None = None
    spec: ModelSpec | None = None
    weight_dtype: Dtype = "fp16"
    kv_dtype: Dtype = "fp16"
    context_length: int = Field(default=8192, gt=0)
    batch_size: int = Field(default=1, gt=0)


class GpuFit(BaseModel):
    gpu: str
    vram_gib: float
    fits: bool


class EstimateResponse(BaseModel):
    model: str
    weights_gib: float
    kv_cache_gib: float
    overhead_gib: float
    total_gib: float
    gpu_fits: list[GpuFit]
