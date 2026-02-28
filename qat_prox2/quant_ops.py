from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch

from .param_filter import iter_named_quant_params


@torch.no_grad()
def quantize_to_grid(w: torch.Tensor, n_bits: int, step: float) -> torch.Tensor:
    """
    Grid: {k*step | k in [-L,...,L]}, L = 2^(b-1)-1
    Clip: [-L*step, L*step]
    """
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")

    L = (2 ** (n_bits - 1)) - 1
    clip = L * step
    q = torch.round(w / step) * step
    q = torch.clamp(q, -clip, clip)
    return q


@torch.no_grad()
def selective_hard_quantize_model_inplace(
    model,
    n_bits_w: int,
    step: float,
    atol: float,
    selector=None,
    include_substrings=None,
    exclude_substrings=("bias", "norm", "ln_", "wte", "wpe", "lm_head"),
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Only snap entries with |w-q|<=atol to q, keep the rest unchanged.
    """
    backup: Dict[str, torch.Tensor] = {}
    total = 0
    quantized = 0

    for name, p in iter_named_quant_params(
        model,
        selector=selector,
        include_substrings=include_substrings,
        exclude_substrings=exclude_substrings,
    ):
        w = p.data
        q = quantize_to_grid(w, n_bits_w, step)
        d = (w - q).abs()
        mask = d <= atol

        backup[name] = w.clone()
        w.copy_(torch.where(mask, q, w))

        total += w.numel()
        quantized += int(mask.sum().item())

    if verbose:
        rate = 0.0 if total == 0 else 100.0 * quantized / total
        print(f"[selective-hard] quantized {rate:.3f}% ({quantized:,}/{total:,}) atol={atol}")

    return backup


@torch.no_grad()
def hard_quantize_model_inplace(
    model,
    n_bits_w: int,
    step: float,
    selector=None,
    include_substrings=None,
    exclude_substrings=("bias", "norm", "ln_", "wte", "wpe", "lm_head"),
) -> Dict[str, torch.Tensor]:
    """Snap ALL selected params to grid."""
    backup: Dict[str, torch.Tensor] = {}

    for name, p in iter_named_quant_params(
        model,
        selector=selector,
        include_substrings=include_substrings,
        exclude_substrings=exclude_substrings,
    ):
        backup[name] = p.data.clone()
        q = quantize_to_grid(p.data, n_bits_w, step)
        p.data.copy_(q)

    return backup


@torch.no_grad()
def restore_model_from_backup(model, backup: Dict[str, torch.Tensor]) -> None:
    for name, p in model.named_parameters():
        if name in backup:
            p.data.copy_(backup[name])
