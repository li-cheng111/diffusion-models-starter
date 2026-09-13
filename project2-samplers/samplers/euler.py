"""Euler and Euler ancestral reference samplers."""

from __future__ import annotations

import torch

from .base import BaseSampler, linear_timesteps, randn


class EulerSampler(BaseSampler):
    def __init__(self, model, schedule, device="cuda", ancestral: bool = False):
        super().__init__(model, schedule, device)
        self.ancestral = bool(ancestral)

    def get_timesteps(self, num_steps: int) -> list[int]:
        return linear_timesteps(num_steps, self.T)

    @torch.no_grad()
    def step(self, x_t, eps_pred, t, t_prev, generator=None):
        alpha_bar_t = self.schedule.alphas_cumprod[t].to(x_t)
        tiny = torch.finfo(x_t.dtype).eps
        x_hat_0 = (
            x_t - torch.sqrt((1.0 - alpha_bar_t).clamp(min=0.0)) * eps_pred
        ) / torch.sqrt(alpha_bar_t.clamp(min=tiny))
        x_hat_0 = x_hat_0.clamp(-1.0, 1.0)
        if t_prev < 0:
            return x_hat_0

        alpha_bar_prev = self.schedule.alphas_cumprod[t_prev].to(x_t)
        if not self.ancestral:
            return (
                torch.sqrt(alpha_bar_prev) * x_hat_0
                + torch.sqrt((1.0 - alpha_bar_prev).clamp(min=0.0)) * eps_pred
            )

        variance = (
            (1.0 - alpha_bar_prev)
            / (1.0 - alpha_bar_t).clamp(min=tiny)
            * (1.0 - alpha_bar_t / alpha_bar_prev.clamp(min=tiny))
        ).clamp(min=0.0)
        sigma_up = torch.sqrt(variance)
        sigma_down = torch.sqrt(
            (1.0 - alpha_bar_prev - sigma_up.square()).clamp(min=0.0)
        )
        noise = randn(
            x_t.shape,
            device=x_t.device,
            dtype=x_t.dtype,
            generator=generator,
        )
        return (
            torch.sqrt(alpha_bar_prev) * x_hat_0
            + sigma_down * eps_pred
            + sigma_up * noise
        )

    @property
    def name(self) -> str:
        return "EulerA" if self.ancestral else "Euler"
