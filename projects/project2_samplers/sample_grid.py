"""Generate a small qualitative grid without requiring the FID dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

try:
    from .checkpoint_utils import load_model_weights
    from .project1_path import add_project1_to_path
    from .samplers import get_sampler
except ImportError:  # pragma: no cover - direct script compatibility
    from checkpoint_utils import load_model_weights
    from project1_path import add_project1_to_path
    from samplers import get_sampler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--sampler", choices=["ddpm", "ddim", "euler", "dpm-solver"], default="ddim"
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="samples/smoke/grid.png")
    parser.add_argument("--project1_path", default=None)
    args = parser.parse_args()

    add_project1_to_path(args.project1_path)
    from dataset import denormalize
    from model import UNet
    from schedule import DDPMSchedule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = UNet(**config["model"]).to(device)
    schedule = DDPMSchedule(**config["diffusion"]).to(device)
    weight_type = load_model_weights(model, checkpoint, use_ema=True)
    model.eval()
    strategy = "lambda" if args.sampler == "dpm-solver" else "linear"
    sampler = get_sampler(
        args.sampler,
        model,
        schedule,
        device=device,
        timestep_strategy=strategy,
    )
    shape = (
        args.num_samples,
        int(config["model"]["in_channels"]),
        int(config["model"]["image_size"]),
        int(config["model"]["image_size"]),
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad():
        images = denormalize(
            sampler.sample(shape, num_steps=args.steps, generator=generator)
        ).cpu()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, output, nrow=max(1, round(args.num_samples**0.5)))
    print(
        f"Saved {args.num_samples} {config['dataset']['name']} samples to {output} "
        f"using {sampler.name}, NFE={sampler.nfe_for_steps(args.steps)}, {weight_type}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
