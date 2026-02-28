from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class QuantParamSelector:
    """ select which params to quantize."""
    mode: str = "mlp"  # "mlp" | "attn" | "mlp_attn" | "all_linear"
    weight_only: bool = True
    require_grad: bool = True

    def allow(self, name: str, p: torch.nn.Parameter) -> bool:
        if self.require_grad and (not p.requires_grad):
            return False
        if self.weight_only and (not name.endswith(".weight")):
            return False

        n = name.lower()
        if self.mode == "mlp":
            return ".mlp." in n
        if self.mode == "attn":
            return ".attn." in n
        if self.mode == "mlp_attn":
            return (".mlp." in n) or (".attn." in n)
        if self.mode == "all_linear":
            # any weight param under transformer blocks
            return (".transformer." in n or ".model." in n or ".h." in n) and name.endswith(".weight")
        raise ValueError(f"Unknown mode={self.mode}")


def iter_named_quant_params(
    model,
    selector: Optional[QuantParamSelector] = None,
    include_substrings=None,
    exclude_substrings=("bias", "norm", "ln_", "wte", "wpe", "lm_head"),
) -> Iterator[Tuple[str, torch.nn.Parameter]]:
    """Yield (name, param) for selected parameters."""
    selector = selector or QuantParamSelector()

    for name, p in model.named_parameters():
        if exclude_substrings and any(s in name for s in exclude_substrings):
            continue
        if include_substrings is not None and (not any(s in name for s in include_substrings)):
            continue
        if not selector.allow(name, p):
            continue
        yield name, p
