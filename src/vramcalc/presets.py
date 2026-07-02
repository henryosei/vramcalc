"""Known model architectures. Numbers from each model's config.json on Hugging Face."""

from vramcalc.models import ModelSpec

PRESETS: dict[str, ModelSpec] = {
    "llama-3.1-8b": ModelSpec(
        name="llama-3.1-8b", params_billion=8.03, num_layers=32, num_kv_heads=8, head_dim=128
    ),
    "llama-3.1-70b": ModelSpec(
        name="llama-3.1-70b", params_billion=70.6, num_layers=80, num_kv_heads=8, head_dim=128
    ),
    "mistral-7b": ModelSpec(
        name="mistral-7b", params_billion=7.25, num_layers=32, num_kv_heads=8, head_dim=128
    ),
    "qwen2.5-32b": ModelSpec(
        name="qwen2.5-32b", params_billion=32.8, num_layers=64, num_kv_heads=8, head_dim=128
    ),
}
