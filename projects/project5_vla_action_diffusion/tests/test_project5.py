import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import evaluate
from env import Reach2DEnv
from model import DiffusionPolicy, ResNetVisionEncoder, VisionEncoder
from train import DDPMScheduler, behavior_cloning_loss, diffusion_loss, flow_matching_loss


def batch(horizon=8):
    return {"image": torch.randn(2, 3, 64, 64), "state": torch.randn(2, 2),
            "goal": torch.randn(2, 2), "action": torch.randn(2, horizon, 2)}


def test_vision_encoder_shape_and_budget():
    model = VisionEncoder(out_dim=128)
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 128)
    assert sum(p.numel() for p in model.parameters()) < 1_000_000


def test_losses_are_scalar_and_differentiable():
    model = DiffusionPolicy(horizon=8, state_dim=2, use_vision=True)
    scheduler = DDPMScheduler(T=10)
    b = batch()
    for loss in (diffusion_loss(model, b, scheduler, "cpu", True),
                 flow_matching_loss(model, b, "cpu", True),
                 behavior_cloning_loss(model, b, "cpu", True)):
        assert loss.ndim == 0 and torch.isfinite(loss)
        model.zero_grad()
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())


def test_scheduler_shapes():
    scheduler = DDPMScheduler(T=10)
    x = torch.randn(3, 8, 2)
    t = torch.tensor([0, 4, 9])
    out = scheduler.add_noise(x, t, torch.zeros_like(x))
    assert out.shape == x.shape
    assert torch.allclose(out[0], x[0] * scheduler.sqrt_ac[0])


def test_closed_loop_returns_complete_metrics():
    cfg = {"n_distractors": 1, "use_vision": False, "chunk_size": 8,
           "diffusion_steps": 10}
    model = DiffusionPolicy(horizon=8, state_dim=4, use_vision=False)
    result = evaluate(model, DDPMScheduler(T=10), cfg, "cpu", n_episodes=2,
                      exec_steps=4, chunk_size=8, n_sample_steps=2,
                      seed_start=10000, method="bc")
    assert result["episodes"] == 2
    assert 0 <= result["success_rate"] <= 1
    assert result["successes"] + result["collisions"] + result["timeouts"] == 2


def test_bonus_target_modes_and_moving_distractors():
    choices = [[-0.65, 0.55], [0.65, 0.55]]
    env = Reach2DEnv(n_distractors=1, target_choices=choices,
                     moving_distractors=True, seed=7)
    assert any(np.allclose(env.target_pos, choice) for choice in choices)
    before = env.distractors.copy()
    env.step([0.0, 0.0])
    assert not np.allclose(before, env.distractors)


def test_resnet_encoder_offline_shape():
    encoder = ResNetVisionEncoder(out_dim=32, pretrained=False)
    assert encoder(torch.randn(2, 3, 64, 64)).shape == (2, 32)
