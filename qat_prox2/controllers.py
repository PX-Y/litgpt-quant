from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GammaController:
    q_target: float = 0.95
    gamma_init: float = 0.0
    gamma_lr: float = 0.5
    gamma_max: float = 5.0
    start_step: int = 0
    ema_momentum: float = 0.9

    gamma: float = 0.0
    q_ema: Optional[float] = None

    def __post_init__(self):
        self.gamma = float(self.gamma_init)

    def step(self, qrate: float, step_count: int) -> float:
        if step_count < self.start_step:
            self.gamma = 0.0
            return self.gamma

        if self.q_ema is None:
            self.q_ema = float(qrate)
        else:
            self.q_ema = self.ema_momentum * self.q_ema + (1.0 - self.ema_momentum) * float(qrate)

        self.gamma = float(self.gamma + self.gamma_lr * (self.q_target - self.q_ema))
        self.gamma = max(0.0, min(self.gamma, self.gamma_max))
        return self.gamma


@dataclass
class DualController:
    beta: float = 2.9
    dual_lr: float = 1e-3
    lambda_init: float = 1.0
    lambda_max: Optional[float] = None

    use_pi: bool = True
    kp: float = 1.0
    ki: float = 0.2
    i_clamp: float = 10.0

    lam: float = 1.0
    g_int: float = 0.0

    def __post_init__(self):
        self.lam = float(self.lambda_init)

    def step(self, f_mean: float) -> float:
        g = float(f_mean - self.beta)

        if self.use_pi:
            self.g_int = float(self.g_int + g)
            if self.g_int > self.i_clamp:
                self.g_int = self.i_clamp
            elif self.g_int < -self.i_clamp:
                self.g_int = -self.i_clamp
            self.lam = float(self.lam + self.dual_lr * (self.kp * g + self.ki * self.g_int))
        else:
            self.lam = float(self.lam + self.dual_lr * g)

        if self.lam < 0.0:
            self.lam = 0.0
        if self.lambda_max is not None:
            self.lam = min(self.lam, float(self.lambda_max))
        return self.lam
