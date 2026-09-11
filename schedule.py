"""Noise schedules and precomputed coefficients for a DDPM."""

from __future__ import annotations

import math
from typing import Union

import torch
from torch import nn


def _validate_timesteps(T: int) -> None:
    if not isinstance(T, int) or T <= 0:
        raise ValueError(f"T must be a positive integer, got {T!r}")


def linear_beta_schedule(
    T: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Return the original DDPM linear beta schedule."""

    _validate_timesteps(T)
    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError("Require 0 < beta_start < beta_end < 1")
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Return the cosine schedule from Improved DDPM.

    This helper is included for later experiments; Project 1's baseline uses
    the linear schedule.
    """

    _validate_timesteps(T)
    if s < 0:
        raise ValueError(f"s must be non-negative, got {s}")

    steps = torch.linspace(0, T, T + 1, dtype=torch.float32)
    t = steps / T
    alpha_bar = torch.cos((t + s) / (1.0 + s) * math.pi / 2.0).pow(2)
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(min=1e-5, max=0.999)


class DDPMSchedule(nn.Module):
    """Precompute all scalar coefficients used by DDPM training and sampling."""

    def __init__(
        self,
        T: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        s: float = 0.008,
    ) -> None:
        super().__init__()
        _validate_timesteps(T)

        schedule_name = beta_schedule.lower()
        if schedule_name == "linear":
            betas = linear_beta_schedule(T, beta_start, beta_end)
        elif schedule_name == "cosine":
            betas = cosine_beta_schedule(T, s)
        else:
            raise ValueError(
                f"Unknown beta_schedule={beta_schedule!r}; use 'linear' or 'cosine'"
            )

        if not torch.isfinite(betas).all() or not ((betas > 0).all() and (betas < 1).all()):
            raise ValueError("beta schedule must contain finite values in (0, 1)")

        self.T = T
        self.beta_schedule = schedule_name

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=betas.dtype), alphas_cumprod[:-1]], dim=0
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # At t=0 the true posterior is a point mass. The explicit zero also
        # makes the last reverse step numerically and semantically clear.
        posterior_variance[0] = 0.0

        buffers = {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "alphas_cumprod_prev": alphas_cumprod_prev,
            "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
            "sqrt_recip_alphas": torch.rsqrt(alphas),
            "posterior_variance": posterior_variance,
            "posterior_log_variance_clipped": torch.log(
                posterior_variance.clamp(min=1e-20)
            ),
        }
        for name, value in buffers.items():
            self.register_buffer(name, value)


ScheduleLike = Union[DDPMSchedule, nn.Module]

