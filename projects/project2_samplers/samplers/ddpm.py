"""DDPM ancestral sampler used as the full-trajectory reference."""

from __future__ import annotations

import torch

from .base import BaseSampler, randn, sqrt_one_minus_alpha_bar


class DDPMSampler(BaseSampler):
    def get_timesteps(self, num_steps: int) -> list[int]:
        if num_steps != self.T:
            print(f"[warning] DDPM requires T={self.T}; ignoring num_steps={num_steps}")
        return list(range(self.T - 1, -1, -1))

    @torch.no_grad()
    def step(self, x_t, eps_pred, t, t_prev, generator=None):
        schedule = self.schedule
        beta_t = schedule.betas[t]
        alpha_t = schedule.alphas[t]
        mean = (
            x_t - beta_t / sqrt_one_minus_alpha_bar(schedule)[t] * eps_pred
        ) / torch.sqrt(alpha_t)
        if t == 0:
            return mean
        variance = schedule.posterior_variance[t].clamp(min=0)
        noise = randn(
            x_t.shape,
            device=x_t.device,
            dtype=x_t.dtype,
            generator=generator,
        )
        return mean + torch.sqrt(variance) * noise

    @property
    def name(self) -> str:
        return "DDPM"
