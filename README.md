# vramcalc

API that estimates GPU VRAM required to serve an LLM. Phase 0 project of my
AI platform engineering roadmap: the goal is production-grade Python packaging,
containerization, and CI — the VRAM math is a bonus that pays off later.

## The math

Serving memory has three parts:

| Component | Formula |
|-----------|---------|
| Weights | `params x bytes_per_param` (fp16 = 2 bytes, int8 = 1, int4 = 0.5) |
| KV cache | `2 x layers x kv_heads x head_dim x context_len x batch x bytes` |
| Overhead | CUDA context (~0.75 GiB) + ~10% for activations and fragmentation |

Example: Llama-3.1-8B in fp16 with one 8k-context request ≈ 15 GiB weights +
1 GiB KV cache + overhead ≈ 17.4 GiB → fits a 24 GB GPU, but batch 32 needs
32 GiB of KV cache alone and does not.

## Run it

```bash
uv sync
uv run uvicorn vramcalc.main:app --reload
```

```bash
curl -s localhost:8000/models | jq
curl -s -X POST localhost:8000/estimate \
  -H 'content-type: application/json' \
  -d '{"preset": "llama-3.1-70b", "weight_dtype": "int4", "batch_size": 8}' | jq
```

Interactive docs at http://localhost:8000/docs.

## Test and lint

```bash
uv run pytest -v
uv run ruff check .
```

## Docker

```bash
docker build -t vramcalc .
docker run -p 8000:8000 vramcalc
```

Multi-stage build: uv resolves dependencies in a builder stage, runtime image is
plain `python:3.12-slim` with the virtualenv copied in, running as a non-root
user with a container healthcheck against `/healthz`.

## CI

GitHub Actions (`.github/workflows/ci.yml`): lint → tests → Docker build →
container smoke test against the health endpoint.
