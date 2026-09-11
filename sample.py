"""Generate images from a trained DDPM checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils

from dataset import denormalize
from diffusion import p_sample_loop
from model import UNet
from schedule import DDPMSchedule


def load_model_from_ckpt(
    ckpt_path: str, use_ema: bool = True
) -> tuple[UNet, DDPMSchedule, dict]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg = checkpoint["config"]
    model = UNet(**cfg["model"])
    schedule = DDPMSchedule(**cfg["diffusion"])
    if use_ema and "ema" in checkpoint:
        ema_state = checkpoint["ema"]
        model.load_state_dict(ema_state.get("model", ema_state))
    else:
        model.load_state_dict(checkpoint["model"])
    return model, schedule, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_grid", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, schedule, cfg = load_model_from_ckpt(args.ckpt, use_ema=not args.no_ema)
    model = model.to(device).eval()
    schedule = schedule.to(device)

    suffix = "_noema" if args.no_ema else "_ema"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.ckpt).parent.parent / f"samples_inference{suffix}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = cfg["model"]["image_size"]
    channels = cfg["model"]["in_channels"]
    all_samples = []
    generated = 0
    while generated < args.num_samples:
        batch_size = min(args.batch_size, args.num_samples - generated)
        samples = p_sample_loop(
            model,
            (batch_size, channels, image_size, image_size),
            schedule,
            device=device,
        )
        samples = denormalize(samples).cpu()
        all_samples.append(samples)
        for index, sample in enumerate(samples):
            vutils.save_image(sample, output_dir / f"sample_{generated + index:04d}.png")
        generated += batch_size

    if args.save_grid:
        samples = torch.cat(all_samples, dim=0)
        nrow = max(1, int(args.num_samples**0.5))
        vutils.save_image(
            samples,
            output_dir / "grid.png",
            nrow=nrow,
            padding=2,
        )
    print(f"Generated {args.num_samples} samples in {output_dir}")


if __name__ == "__main__":
    main()

