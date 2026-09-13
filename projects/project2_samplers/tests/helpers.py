from __future__ import annotations

import torch


class ToySchedule(torch.nn.Module):
    def __init__(self, T: int = 100, plural_aliases: bool = True):
        super().__init__()
        self.T = T
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        previous = torch.cat([torch.ones(1), cumulative[:-1]])
        posterior = betas * (1.0 - previous) / (1.0 - cumulative)
        posterior[0] = 0.0
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", cumulative)
        self.register_buffer("alphas_cumprod_prev", previous)
        self.register_buffer("posterior_variance", posterior)
        if plural_aliases:
            self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(cumulative))
            self.register_buffer(
                "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - cumulative)
            )
        else:
            self.register_buffer("sqrt_alpha_bar", torch.sqrt(cumulative))
            self.register_buffer(
                "sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - cumulative)
            )


class ZeroModel(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


class CountingZeroModel(ZeroModel):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, t):
        self.calls += 1
        return super().forward(x, t)
