from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class QATConfig:
    # master switch(False = no quant)
    enabled: bool = True

    # (atol 容忍度)
    n_bits_w: int = 4
    step_w: float = 0.02
    atol: float = 1e-3

    # which params to quantize
    selector_mode: str = "mlp"  # "mlp" | "attn" | "mlp_attn" | "all_linear"
    include_substrings: Optional[Tuple[str, ...]] = None
    exclude_substrings: Tuple[str, ...] = ("bias", "norm", "ln_", "wte", "wpe", "lm_head")

    #(宏观测试distloss数量级)
    dist_scale: float = 1.0

    # ---- Lagrange constraint: lambda*(f - beta) ----
    use_lagrange: bool = True
    beta: float = 1
    dual_lr: float = 5e-3
    lambda_init: float = 3.0
    lambda_max: Optional[float] = None

    # PI controller for lambda
    lambda_use_pi: bool = True
    lambda_kp: float = 3.0
    lambda_ki: float = 0.5
    lambda_i_clamp: float = 10.0

    # ---- gamma controller ----
    q_target: float = 0.9
    gamma_init: float = 1.0
    gamma_lr: float = 0.1
    gamma_max: float = 5.0
    gamma_start_step: int = 300
    qrate_every: int = 100
    qrate_ema_momentum: float = 0.9

    # ---- sensitivity EMA ----
    sens_enable: bool = True
    sens_ema: float = 0.95

    #--- hess ---
    hess_enable: bool = True
    hess_inverse: bool = True
    hess_rel_min: float = 0.2
    hess_rel_max: float = 5.0
