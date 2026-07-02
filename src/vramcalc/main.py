"""HTTP layer. Thin on purpose — all logic lives in calc.py."""

from fastapi import FastAPI, HTTPException

from vramcalc import calc, presets
from vramcalc.models import EstimateRequest, EstimateResponse, ModelSpec

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


@app.post("/estimate")
async def estimate(req: EstimateRequest) -> EstimateResponse:
    if req.preset is not None:
        spec = presets.PRESETS.get(req.preset)
        if spec is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown preset {req.preset!r}. See GET /models for options.",
            )
    elif req.spec is not None:
        spec = req.spec
    else:
        raise HTTPException(status_code=422, detail="Provide either 'preset' or 'spec'.")

    return calc.estimate(
        spec=spec,
        weight_dtype=req.weight_dtype,
        kv_dtype=req.kv_dtype,
        context_length=req.context_length,
        batch_size=req.batch_size,
    )
