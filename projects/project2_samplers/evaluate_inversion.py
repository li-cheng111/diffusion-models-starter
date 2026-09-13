"""Evaluate DDIM inversion/reconstruction error over fixed CIFAR-10 images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

try:
    from .checkpoint_utils import load_model_weights, repository_commit
    from .project1_path import add_project1_to_path
    from .samplers import DDIMInverter
except ImportError:  # pragma: no cover - direct script compatibility
    from checkpoint_utils import load_model_weights, repository_commit
    from project1_path import add_project1_to_path
    from samplers import DDIMInverter


def batch_metrics(source: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, torch.Tensor]:
    difference = source - reconstructed
    flattened = difference.flatten(1)
    mse = flattened.square().mean(dim=1)
    return {
        "l2": torch.linalg.vector_norm(flattened, dim=1),
        "mae": flattened.abs().mean(dim=1),
        "mse": mse,
        "psnr": 10.0 * torch.log10(4.0 / mse.clamp(min=1e-12)),
    }


def save_comparison(
    originals: torch.Tensor,
    reconstructions: dict[int, torch.Tensor],
    output: Path,
) -> None:
    columns = ["original"] + [f"{steps} steps" for steps in reconstructions]
    rows = originals.shape[0]
    figure, axes = plt.subplots(
        rows,
        len(columns),
        figsize=(2.0 * len(columns), 2.0 * rows),
        squeeze=False,
    )
    for row in range(rows):
        images = [originals[row]] + [values[row] for values in reconstructions.values()]
        for column, image in enumerate(images):
            display = ((image.clamp(-1, 1) + 1.0) / 2.0).cpu()
            if display.shape[0] == 1:
                axes[row, column].imshow(display[0].numpy(), cmap="gray", vmin=0, vmax=1)
            else:
                axes[row, column].imshow(display.permute(1, 2, 0).numpy().clip(0, 1))
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(columns[column])
    figure.suptitle("DDIM inversion and deterministic reconstruction")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[10, 20, 50, 100, 250])
    parser.add_argument("--num_images", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_show", type=int, default=6)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--output", default="runs/inversion_metrics.json")
    parser.add_argument("--plot", default="runs/inversion_compare.png")
    parser.add_argument("--project1_path", default=None)
    args = parser.parse_args()

    project1 = add_project1_to_path(args.project1_path)
    from dataset import get_dataset
    from model import UNet
    from schedule import DDPMSchedule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = UNet(**config["model"]).to(device)
    schedule = DDPMSchedule(**config["diffusion"]).to(device)
    weight_type = load_model_weights(model, checkpoint, use_ema=True)
    model.eval()

    dataset = get_dataset(
        config["dataset"]["name"],
        root=args.data_root,
        image_size=config["model"]["image_size"],
        train=False,
        augment=False,
    )
    if args.num_images > len(dataset):
        parser.error(f"Requested {args.num_images} images, dataset has {len(dataset)}")
    subset = torch.utils.data.Subset(dataset, list(range(args.num_images)))
    loader = torch.utils.data.DataLoader(
        subset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    inverter = DDIMInverter(model, schedule, device=device)
    originals_for_plot: torch.Tensor | None = None
    reconstructions_for_plot: dict[int, torch.Tensor] = {}
    results: list[dict] = []

    for num_steps in args.steps:
        collected = {name: [] for name in ("l2", "mae", "mse", "psnr")}
        preview: list[torch.Tensor] = []
        first_originals: list[torch.Tensor] = []
        for batch in loader:
            source = batch.to(device)
            _, reconstructed = inverter.invert_and_reconstruct(source, num_steps)
            metrics = batch_metrics(source, reconstructed)
            for name, values in metrics.items():
                collected[name].append(values.detach().cpu())
            remaining = args.num_show - sum(item.shape[0] for item in preview)
            if remaining > 0:
                preview.append(reconstructed[:remaining].detach().cpu())
                first_originals.append(source[:remaining].detach().cpu())
        aggregated = {
            name: torch.cat(values).float() for name, values in collected.items()
        }
        results.append(
            {
                "num_steps": num_steps,
                **{
                    f"mean_{name}": float(values.mean().item())
                    for name, values in aggregated.items()
                },
                **{
                    f"std_{name}": float(values.std(unbiased=True).item())
                    for name, values in aggregated.items()
                },
            }
        )
        reconstructions_for_plot[num_steps] = torch.cat(preview, dim=0)
        if originals_for_plot is None:
            originals_for_plot = torch.cat(first_originals, dim=0)
        print(
            f"steps={num_steps}: MAE={aggregated['mae'].mean():.6f}, "
            f"PSNR={aggregated['psnr'].mean():.3f} dB"
        )

    payload = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "project1_path": str(project1),
        "git_commit": repository_commit(),
        "weights": weight_type,
        "dataset": config["dataset"]["name"],
        "split": "test",
        "indices": f"0:{args.num_images}",
        "clip_denoised": False,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert originals_for_plot is not None
    save_comparison(originals_for_plot, reconstructions_for_plot, Path(args.plot))
    print(f"Saved {output} and {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
