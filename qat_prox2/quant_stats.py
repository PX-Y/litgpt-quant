from __future__ import annotations

from typing import Optional, Tuple

import torch

from .param_filter import QuantParamSelector, iter_named_quant_params
from .quant_ops import quantize_to_grid


@torch.no_grad()
def compute_quantization_rate_fast(
    model,
    n_bits_w: int,
    step: float,
    atol: float = 1e-3,
    selector: Optional[QuantParamSelector] = None,
    include_substrings=None,
    exclude_substrings=("bias", "norm", "ln_", "wte", "wpe", "lm_head"),
) -> Tuple[float, float, int]:
    """Return (hit_rate, sat_rate, total_numel).

    hit_rate: fraction with |w-q(w)| <= atol
    sat_rate: fraction with |q(w)| at clipping boundary
    """
    L = (2 ** (n_bits_w - 1)) - 1
    clip = L * step

    total = 0
    within = 0
    sat = 0

    for name, p in iter_named_quant_params(
        model,
        selector=selector,
        include_substrings=include_substrings,
        exclude_substrings=exclude_substrings,
    ):
        w = p.data
        q = quantize_to_grid(w, n_bits_w, step)
        d = (w - q).abs()

        total += d.numel()
        within += int((d <= atol).sum().item())
        sat += int((q.abs() >= (clip - 1e-12)).sum().item())

    hit_rate = 0.0 if total == 0 else within / total
    sat_rate = 0.0 if total == 0 else sat / total
    return hit_rate, sat_rate, total
