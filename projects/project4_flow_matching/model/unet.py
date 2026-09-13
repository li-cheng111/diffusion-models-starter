"""简化版 conditional UNet for Flow Matching / DDPM on CIFAR-10.

最小可用，约 30M 参数，主要为教学清晰。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal time embedding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) float
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)
        )
        a = t[:, None].float() * freqs[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_ch, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.emb_proj(F.silu(emb)).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    """
    简化 UNet，3 个分辨率：32 → 16 → 8 → 16 → 32。
    支持 (image, time, class) → image-shaped output。
    """

    def __init__(self, in_channels=3, out_channels=3, base_channels=64, num_classes=10):
        super().__init__()
        self.num_classes = num_classes  # 类别数；num_classes index 作为 null token
        emb_ch = base_channels * 4

        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(base_channels),
            nn.Linear(base_channels, emb_ch),
            nn.SiLU(),
            nn.Linear(emb_ch, emb_ch),
        )
        self.class_emb = nn.Embedding(num_classes + 1, emb_ch)  # +1 for null

        # Encoder
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.enc1 = ResBlock(base_channels, base_channels, emb_ch)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.enc2 = ResBlock(base_channels * 2, base_channels * 2, emb_ch)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)

        # Middle
        self.mid1 = ResBlock(base_channels * 4, base_channels * 4, emb_ch)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, emb_ch)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.dec2 = ResBlock(base_channels * 4, base_channels * 2, emb_ch)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.dec1 = ResBlock(base_channels * 2, base_channels, emb_ch)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x, t, y):
        # x: (B, C, 32, 32), t: (B,) float, y: (B,) long
        emb = self.time_emb(t) + self.class_emb(y)

        h0 = self.in_conv(x)
        h1 = self.enc1(h0, emb)
        h2 = self.enc2(self.down1(h1), emb)
        h3 = self.mid2(self.mid1(self.down2(h2), emb), emb)

        h = self.dec2(torch.cat([self.up2(h3), h2], dim=1), emb)
        h = self.dec1(torch.cat([self.up1(h), h1], dim=1), emb)
        return self.out_conv(h)
