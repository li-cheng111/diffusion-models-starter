"""Verify that Project 2 can consume this repository's Project 1 implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from checkpoint_utils import load_model_weights
from project1_path import add_project1_to_path
from samplers import DDIMInverter, get_sampler
from samplers.base import get_schedule_buffer


class CountingZeroModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, t):
        self.calls += 1
        return torch.zeros_like(x)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project1_path", default=None)
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()
    ok = True

    try:
        project1 = add_project1_to_path(args.project1_path)
        print(f"[1/5] Project 1: {project1}")
    except FileNotFoundError as error:
        print(f"[1/5] FAILED: {error}")
        return 1

    from schedule import DDPMSchedule

    schedule = DDPMSchedule(T=100, beta_schedule="linear").to("cpu")
    print("[2/5] schedule construction: OK")
    alias_groups = (
        ("betas",),
        ("alphas",),
        ("alphas_cumprod",),
        ("sqrt_alpha_bar", "sqrt_alphas_cumprod"),
        ("sqrt_one_minus_alpha_bar", "sqrt_one_minus_alphas_cumprod"),
        ("posterior_variance",),
    )
    try:
        for aliases in alias_groups:
            get_schedule_buffer(schedule, *aliases)
        print("[3/5] schedule buffers and aliases: OK")
    except AttributeError as error:
        ok = False
        print(f"[3/5] FAILED: {error}")

    shape = (2, 3, 8, 8)
    for name, steps in (("ddpm", 100), ("ddim", 10), ("euler", 10)):
        model = CountingZeroModel()
        output = get_sampler(name, model, schedule, device="cpu").sample(shape, steps)
        if output.shape != shape or not torch.isfinite(output).all():
            ok = False
            print(f"[4/5] {name}: FAILED")
        else:
            print(f"[4/5] {name}: OK")

    model = CountingZeroModel()
    dpm = get_sampler("dpm-solver", model, schedule, device="cpu")
    output = dpm.sample(shape, 5)
    expected_nfe = dpm.nfe_for_steps(5)
    if model.calls != expected_nfe or not torch.isfinite(output).all():
        ok = False
        print(f"[4/5] dpm-solver: FAILED, calls={model.calls}, expected={expected_nfe}")
    else:
        print(f"[4/5] dpm-solver: OK, NFE={model.calls}")

    inverter = DDIMInverter(CountingZeroModel(), schedule, device="cpu")
    source = torch.randn(shape)
    _, reconstructed = inverter.invert_and_reconstruct(source, num_steps=10)
    if not torch.allclose(source, reconstructed, atol=1e-5, rtol=1e-5):
        ok = False
        print("[4/5] DDIM inversion zero-model round trip: FAILED")
    else:
        print("[4/5] DDIM inversion zero-model round trip: OK")

    if args.ckpt:
        from model import UNet

        checkpoint_path = Path(args.ckpt)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = UNet(**checkpoint["config"]["model"])
        weight_type = load_model_weights(model, checkpoint, use_ema=True)
        print(f"[5/5] checkpoint {checkpoint_path}: OK ({weight_type})")
    else:
        print("[5/5] checkpoint load: skipped (pass --ckpt to enable)")

    print("Compatibility check PASSED" if ok else "Compatibility check FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
