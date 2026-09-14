import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if importlib.util.find_spec("diffusers") is None or importlib.util.find_spec("transformers") is None:
    pytest.skip("Diffusers/Transformers are installed on AutoDL for Project 3 tests", allow_module_level=True)

spec = importlib.util.spec_from_file_location("p3_lora", ROOT / "03_lora_finetune.py")
p3_lora = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(p3_lora)


class TinyVAE:
    def encode(self, images):
        latent = F.interpolate(images[:, :1], size=(8, 8), mode="bilinear", align_corners=False).repeat(1, 4, 1, 1)
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: latent))


class TinyText:
    def __call__(self, token_ids):
        return (torch.zeros(token_ids.shape[0], 77, 4, device=token_ids.device),)


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, kernel_size=1)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states):
        return SimpleNamespace(sample=self.conv(noisy_latents))


class TinyScheduler:
    config = SimpleNamespace(num_train_timesteps=10)

    def add_noise(self, latents, noise, timesteps):
        return 0.8 * latents + 0.6 * noise


def test_train_one_step_updates_only_trainable_model():
    torch.manual_seed(0)
    unet = TinyUNet()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    before = unet.conv.weight.detach().clone()
    batch = (torch.randn(2, 3, 32, 32), torch.zeros(2, 77, dtype=torch.long))
    loss = p3_lora.train_one_step(
        batch, unet, TinyVAE(), TinyText(), TinyScheduler(), optimizer,
        torch.device("cpu"), torch.float32, None, scaler,
    )
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before, unet.conv.weight.detach())
