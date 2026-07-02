"""HTTP layer. Thin on purpose — all logic lives in calc.py."""

from fastapi import FastAPI, HTTPException

from vramcalc import calc, presets
from vramcalc.models import (
    EstimateRequest,
    EstimateResponse,
    MaxBatchRequest,
    MaxBatchResponse,
    ModelSpec,
)

app = FastAPI(
    title="vramcalc",
    description="Estimate GPU VRAM requirements for serving LLMs",
    version="0.1.0",
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe: process is up and can serve requests."""
    return {"status": "ok"}


@app.get("/models")
async def list_models() -> list[ModelSpec]:
    return list(presets.PRESETS.values())


def _resolve_spec(preset: str | None, spec: ModelSpec | None) -> ModelSpec:
    if preset is not None:
        resolved = presets.PRESETS.get(preset)
        if resolved is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown preset {preset!r}. See GET /models for options.",
            )
        return resolved
    if spec is not None:
        return spec
    raise HTTPException(status_code=422, detail="Provide either 'preset' or 'spec'.")


@app.post("/estimate")
async def estimate(req: EstimateRequest) -> EstimateResponse:
    spec = _resolve_spec(req.preset, req.spec)
    return calc.estimate(
        spec=spec,
        weight_dtype=req.weight_dtype,
        kv_dtype=req.kv_dtype,
        context_length=req.context_length,
        batch_size=req.batch_size,
    )


@app.post("/max-batch")
async def max_batch(req: MaxBatchRequest) -> MaxBatchResponse:
    spec = _resolve_spec(req.preset, req.spec)

    if req.gpu is not None:
        vram = calc.COMMON_GPUS.get(req.gpu)
        if vram is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown GPU {req.gpu!r}. Options: {sorted(calc.COMMON_GPUS)}",
            )
    elif req.vram_gib is not None:
        vram = req.vram_gib
    else:
        raise HTTPException(status_code=422, detail="Provide either 'gpu' or 'vram_gib'.")

    return MaxBatchResponse(
        model=spec.name,
        vram_gib=vram,
        max_batch_size=calc.max_batch_size(
            spec=spec,
            weight_dtype=req.weight_dtype,
            kv_dtype=req.kv_dtype,
            context_length=req.context_length,
            vram_gib=vram,
        ),
        weights_gib=round(calc.weights_gib(spec.params_billion, req.weight_dtype), 2),
        kv_per_request_gib=round(
            calc.kv_cache_gib(spec, req.context_length, batch_size=1, dtype=req.kv_dtype), 2
        ),
    )
