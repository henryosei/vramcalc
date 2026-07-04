"""Runtime configuration via environment variables.

Defaults match the previous hardcoded constants, so no env = no behavior change.
In Kubernetes these arrive through a ConfigMap via envFrom.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VRAMCALC_")

    # Fixed CUDA context cost plus a fractional buffer for activations and
    # allocator fragmentation. Conservative; real usage varies by engine.
    cuda_context_gib: float = 0.75
    overhead_fraction: float = 0.10


@lru_cache
def get_settings() -> Settings:
    """Cached for process lifetime — env is read once at startup, mirroring
    how Kubernetes injects env vars (pod restart required to pick up changes)."""
    return Settings()
