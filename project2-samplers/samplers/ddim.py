"""Denoising Diffusion Implicit Model (DDIM) sampler."""

from __future__ import annotations

import torch

from .base import BaseSampler, linear_timesteps, quadratic_timesteps, randn


class DDIMSampler(BaseSampler):
    def __init__(
        self,
        model,
        schedule,
        device="cuda",
        eta: float = 0.0,
        timestep_strategy: str = "linear",
        clip_denoised: bool = True,
    ):
        super().__init__(model, schedule, device)
        if not 0.0 <= eta <= 1.0:
            raise ValueError(f"eta must be in [0, 1], got {eta}")
        self.eta = float(eta)
        self.timestep_strategy = timestep_strategy
        self.clip_denoised = bool(clip_denoised)

    def get_timesteps(self, num_steps: int) -> list[int]:
        if self.timestep_strategy == "linear":
            return linear_timesteps(num_steps, self.T)
        if self.timestep_strategy == "quadratic":
            return quadratic_timesteps(num_steps, self.T)
        raise ValueError(f"Unknown DDIM timestep strategy: {self.timestep_strategy}")

    @torch.no_grad()
    def step(self, x_t, eps_pred, t, t_prev, generator=None):
        alpha_bar_t = self.schedule.alphas_cumprod[t].to(
            device=x_t.device, dtype=x_t.dtype
        )
        tiny = torch.finfo(x_t.dtype).eps
        x_hat_0 = (
            x_t - torch.sqrt((1.0 - alpha_bar_t).clamp(min=0)) * eps_pred
        ) / torch.sqrt(alpha_bar_t.clamp(min=tiny))
        if self.clip_denoised:
            x_hat_0 = x_hat_0.clamp(-1.0, 1.0)

        if t_prev < 0:
            return x_hat_0

        alpha_bar_prev = self.schedule.alphas_cumprod[t_prev].to(
            device=x_t.device, dtype=x_t.dtype
        )
        variance = (
            (1.0 - alpha_bar_prev)
            / (1.0 - alpha_bar_t).clamp(min=tiny)
            * (1.0 - alpha_bar_t / alpha_bar_prev.clamp(min=tiny))
        ).clamp(min=0.0)
        sigma = self.eta * torch.sqrt(variance)
        direction = torch.sqrt(
            (1.0 - alpha_bar_prev - sigma.square()).clamp(min=0.0)
        ) * eps_pred
        x_prev = torch.sqrt(alpha_bar_prev) * x_hat_0 + direction
        if self.eta > 0.0 and float(sigma) > 0.0:
            noise = randn(
                x_t.shape,
                device=x_t.device,
                dtype=x_t.dtype,
                generator=generator,
            )
            x_prev = x_prev + sigma * noise
        return x_prev

    @property
    def name(self) -> str:
        return f"DDIM(eta={self.eta:g})"


if __name__ == "__main__":
    from project1_path import add_project1_to_path

    add_project1_to_path()
    from schedule import DDPMSchedule

    class DummyModel(torch.nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    schedule = DDPMSchedule(T=100, beta_schedule="linear").to("cpu")
    sampler = DDIMSampler(DummyModel(), schedule, device="cpu")
    output = sampler.sample((2, 3, 8, 8), num_steps=10)
    print(f"DDIM self-test passed: shape={tuple(output.shape)}")
