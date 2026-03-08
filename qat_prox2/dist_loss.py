from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, List

import torch

from .param_filter import QuantParamSelector, iter_named_quant_params
from .quant_ops import quantize_to_grid
from .sensitivity import SensitivityEMA


def compute_dist_loss(
    model,
    n_bits_w: int,
    step: float,
    selector: Optional[QuantParamSelector] = None,
    include_substrings=None,
    exclude_substrings=("bias", "norm", "ln_", "wte", "wpe", "lm_head"),
    sens: Optional[SensitivityEMA] = None,
    return_debug: bool = False,
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    """Compute (possibly sensitivity-weighted) sum_i ||w_i - q(w_i)||^2 (normalized).

    Returns:
      dist_loss (tensor scalar with grads),
      debug dict (optional)
    """
    items: List[Tuple[Optional[torch.Tensor], torch.Tensor]] = []

    for name, p in iter_named_quant_params(
        model,
        selector=selector,
        include_substrings=include_substrings,
        exclude_substrings=exclude_substrings,
    ):
        q = quantize_to_grid(p, n_bits_w, step).detach()
        mse = (p - q).pow(2).mean() * p.numel()
        s = sens.get(name, p.device) if sens is not None else None
        items.append((s, mse))

    if len(items) == 0:
        z = torch.zeros((), device=next(model.parameters()).device)
        return (z, None) if return_debug else (z, None)

    K = len(items)

    # norm of sensitivity, making it not too small or too big
    s_list = [s for (s, _) in items if s is not None]
    if len(s_list) == 0:
        ws = torch.ones((K,), device=items[0][1].device, dtype=torch.float32)
    else:
        S = torch.stack([s for s in s_list]) + 1e-12
        z = torch.log(S)
        mu = z.mean()
        sigma = z.std() + 1e-6

        ws_list = []
        for (s, _) in items:
            if s is None:
                w = torch.tensor(3.0, device=mu.device)  # neutral default
            else:
                zz = torch.log(s + 1e-12)
                score = (mu - zz) / sigma
                w = 1.0 + 4.0 * torch.sigmoid(score)
            ws_list.append(w)
        ws = torch.stack(ws_list).detach().float()

    w_mean = ws.mean() + 1e-12

    dist = 0.0
    for i, (_, mse) in enumerate(items):
        rel = ws[i] / w_mean
        dist = dist + rel * mse

    out = dist / K

    if not return_debug:
        return out, None

    rels = (ws / w_mean).detach().cpu()
    dbg = {
        "K": K,
        "w_min": float(ws.min().item()),
        "w_mean": float(ws.mean().item()),
        "w_max": float(ws.max().item()),
        "rel_min": float(rels.min().item()),
        "rel_mean": float(rels.mean().item()),
        "rel_max": float(rels.max().item()),
    }
    return out, dbg
