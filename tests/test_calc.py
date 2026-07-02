"""Unit tests for the pure math — no HTTP involved."""

import pytest

from vramcalc import calc
from vramcalc.models import ModelSpec

LLAMA_8B = ModelSpec(
    name="llama-3.1-8b", params_billion=8.03, num_layers=32, num_kv_heads=8, head_dim=128
)


def test_weights_fp16():
    # 8.03e9 params x 2 bytes = 16.06e9 bytes = ~14.96 GiB
    assert calc.weights_gib(8.03, "fp16") == pytest.approx(14.96, abs=0.01)


def test_weights_int4_is_quarter_of_fp16():
    assert calc.weights_gib(8.03, "int4") == pytest.approx(
        calc.weights_gib(8.03, "fp16") / 4
    )


def test_kv_cache_llama8b_8k_context_is_exactly_1gib():
    # 2 x 32 layers x 8 kv_heads x 128 head_dim x 8192 ctx x 1 batch x 2 bytes
    # = 1_073_741_824 bytes = exactly 1 GiB
    assert calc.kv_cache_gib(LLAMA_8B, context_length=8192, batch_size=1, dtype="fp16") == 1.0


def test_kv_cache_scales_linearly_with_batch():
    one = calc.kv_cache_gib(LLAMA_8B, context_length=8192, batch_size=1, dtype="fp16")
    eight = calc.kv_cache_gib(LLAMA_8B, context_length=8192, batch_size=8, dtype="fp16")
    assert eight == pytest.approx(one * 8)


def test_estimate_total_is_sum_of_parts():
    result = calc.estimate(
        spec=LLAMA_8B,
        weight_dtype="fp16",
        kv_dtype="fp16",
        context_length=8192,
        batch_size=1,
    )
    assert result.total_gib == pytest.approx(
        result.weights_gib + result.kv_cache_gib + result.overhead_gib, abs=0.05
    )


def test_llama8b_fp16_fits_on_24gb_gpu():
    result = calc.estimate(
        spec=LLAMA_8B,
        weight_dtype="fp16",
        kv_dtype="fp16",
        context_length=8192,
        batch_size=1,
    )
    a10g = next(f for f in result.gpu_fits if f.gpu.startswith("A10G"))
    assert a10g.fits


def test_llama8b_fp16_large_batch_does_not_fit_24gb():
    # 32 concurrent 8k-context requests = 32 GiB of KV cache alone
    result = calc.estimate(
        spec=LLAMA_8B,
        weight_dtype="fp16",
        kv_dtype="fp16",
        context_length=8192,
        batch_size=32,
    )
    a10g = next(f for f in result.gpu_fits if f.gpu.startswith("A10G"))
    assert not a10g.fits
