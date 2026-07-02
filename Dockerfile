# --- Stage 1: build the virtualenv with uv ---------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first, source second: dependency layer stays cached
# across code-only changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ src/
# --no-editable: install the project as a real package into site-packages.
# Default editable install only writes a .pth pointer to /app/src, which
# does not exist in the runtime stage below.
RUN uv sync --frozen --no-dev --no-editable

# --- Stage 2: slim runtime, no uv, no build tools ---------------------------
FROM python:3.12-slim-bookworm

RUN groupadd -r app && useradd -r -g app app

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Verify the shipped venv is self-contained — must run in THIS stage,
# where src/ no longer exists, or it proves nothing.
RUN python -c "import vramcalc"

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

CMD ["uvicorn", "vramcalc.main:app", "--host", "0.0.0.0", "--port", "8000"]
