"""Time-step embeddings used by the DDPM U-Net."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalPosEmb(nn.Module):
    """Encode integer diffusion steps with sinusoidal Fourier features."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            t = t.reshape(-1)
        t = t.float()
        half_dim = self.dim // 2
        if half_dim == 0:
            return t.new_zeros((t.shape[0], self.dim))

        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )
        angles = t[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = F.pad(embedding, (0, 1))
        return embedding

