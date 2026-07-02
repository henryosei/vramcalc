"""Pure VRAM math. No FastAPI imports — testable in isolation.

The serving memory budget is three parts:
  1. weights      = params x bytes-per-param
  2. KV cache     = 2 (K and V) x layers x kv_heads x head_dim x context x batch x bytes
  3. overhead     = CUDA context + activations + fragmentation (rule of thumb, not exact)
"""

from vramcalc.models import Dtype, EstimateResponse, GpuFit, ModelSpec

GIB = 1024**3

DTYPE_BYTES: dict[Dtype, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}

# Marketing "GB" on GPUs is close enough to GiB for fit checks.
COMMON_GPUS: dict[str, float] = {
    "RTX 4090 (24 GB)": 24,
    "L4 (24 GB)": 24,
    "A10G (24 GB)": 24,
    "L40S (48 GB)": 48,
    "A100 (40 GB)": 40,
    "A100 (80 GB)": 80,
    "H100 (80 GB)": 80,
}

# Fixed CUDA context cost plus a fractional buffer for activations and
# allocator fragmentation. Deliberately conservative; real usage varies by engine.
CUDA_CONTEXT_GIB = 0.75
OVERHEAD_FRACTION = 0.10


def weights_gib(params_billion: float, dtype: Dtype) -> float:
    return params_billion * 1e9 * DTYPE_BYTES[dtype] / GIB


def kv_cache_gib(
    spec: ModelSpec, context_length: int, batch_size: int, dtype: Dtype
) -> float:
    bytes_total = (
        2  # K and V
        * spec.num_layers
        * spec.num_kv_heads
        * spec.head_dim
        * context_length
        * batch_size
        * DTYPE_BYTES[dtype]
    )
    return bytes_total / GIB


def estimate(
    spec: ModelSpec,
    weight_dtype: Dtype,
    kv_dtype: Dtype,
    context_length: int,
    batch_size: int,
) -> EstimateResponse:
    weights = weights_gib(spec.params_billion, weight_dtype)
    kv = kv_cache_gib(spec, context_length, batch_size, kv_dtype)
    overhead = CUDA_CONTEXT_GIB + (weights + kv) * OVERHEAD_FRACTION
    total = weights + kv + overhead
    return EstimateResponse(
        model=spec.name,
        weights_gib=round(weights, 2),
        kv_cache_gib=round(kv, 2),
        overhead_gib=round(overhead, 2),
        total_gib=round(total, 2),
        gpu_fits=[
            GpuFit(gpu=name, vram_gib=vram, fits=total <= vram)
            for name, vram in COMMON_GPUS.items()
        ],
    )
