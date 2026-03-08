from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from .param_filter import QuantParamSelector, iter_named_quant_params


@dataclass
class SensitivityEMA:
    momentum: float = 0.95
    eps: float = 1e-8
    state: Dict[str, torch.Tensor] = field(default_factory=dict)  # store g^2 mean (CPU float32)

    @torch.no_grad()
    def update_from_grads(
        self,
        model,
        selector: Optional[QuantParamSelector] = None,
        include_substrings=None,
        exclude_substrings=("bias", "norm", "ln_", "wte", "wpe", "lm_head"),
    ) -> None:
        for name, p in iter_named_quant_params(
            model,
            selector=selector,
            include_substrings=include_substrings,
            exclude_substrings=exclude_substrings,
        ):
            if p.grad is None:
                continue
            g2 = p.grad.detach().float().pow(2).mean().cpu()
            if name not in self.state:
                self.state[name] = g2
            else:
                self.state[name] = self.momentum * self.state[name] + (1.0 - self.momentum) * g2

    def get(self, name: str, device: torch.device) -> Optional[torch.Tensor]:
        v = self.state.get(name, None)
        if v is None:
            return None
        return v.to(device=device, dtype=torch.float32)

    def mean_max(self) -> Tuple[float, float]:
        if not self.state:
            return 0.0, 0.0
        vals = torch.stack([v.float() for v in self.state.values()])
        return float(vals.mean().item()), float(vals.max().item())
