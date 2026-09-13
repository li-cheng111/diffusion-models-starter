"""Reload LoRA checkpoints in fresh pipelines and create before/after grids."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionPipeline

try:
    from .experiment_utils import add_file_hashes, git_commit, runtime_metadata, set_seed, write_json
except ImportError:  # direct script execution from the project directory
    from experiment_utils import add_file_hashes, git_commit, runtime_metadata, set_seed, write_json


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
DEFAULT_PROMPTS = [
    "a lighthouse by the sea in sks style",
    "a quiet street at night in sks style",
    "a vase of sunflowers in sks style",
    "a mountain landscape in sks style",
]


def _load_pipeline(model_id: str, revision: str | None, device: torch.device, dtype: torch.dtype):
    kwargs = {"torch_dtype": dtype}
    if revision:
        kwargs["revision"] = revision
    pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs).to(device)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def _load_adapter(pipe, adapter_dir: Path) -> None:
    """Load either the native Diffusers or PEFT fallback format."""
    weight = adapter_dir / "pytorch_lora_weights.safetensors"
    # ``UNet.save_lora_adapter`` writes an UNet-only state dict (without the
    # pipeline's ``unet.`` prefix). Load it through the matching UNet API;
    # calling pipeline.load_lora_weights here silently ignores those keys.
    if weight.exists() and hasattr(pipe.unet, "load_lora_adapter"):
        pipe.unet.load_lora_adapter(
            adapter_dir, adapter_name="default", prefix=None, weight_name=weight.name,
        )
        if hasattr(pipe.unet, "set_adapter"):
            pipe.unet.set_adapter("default")
        return
    if weight.exists() and hasattr(pipe, "load_lora_weights"):
        pipe.load_lora_weights(adapter_dir, weight_name=weight.name)
        return
    if hasattr(pipe.unet, "load_lora_adapter"):
        pipe.unet.load_lora_adapter(adapter_dir, adapter_name="default")
        if hasattr(pipe.unet, "set_adapter"):
            pipe.unet.set_adapter("default")
        return
    pipe.load_lora_weights(adapter_dir)


def _read_prompts(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_PROMPTS
    values = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return [value for value in values if value and not value.startswith("#")]


def _checkpoint_dirs(lora_dir: Path, explicit: list[str] | None) -> list[Path]:
    if explicit:
        return [Path(value) for value in explicit]
    return sorted(lora_dir.glob("checkpoint-*"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument("--prompts_file", default=None)
    parser.add_argument("--checkpoint_dirs", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    lora_dir = Path(args.lora_dir)
    output_dir = Path(args.output_dir) if args.output_dir else lora_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = _read_prompts(args.prompts_file)
    checkpoints = _checkpoint_dirs(lora_dir, args.checkpoint_dirs)
    labels = [("base", None)] + [(path.name, path) for path in checkpoints]
    if not checkpoints:
        labels.append(("final", lora_dir))
    images_for_grid: dict[str, list] = {}
    started = time.time()
    for label, adapter_dir in labels:
        pipe = _load_pipeline(args.model_id, args.model_revision, device, dtype)
        if adapter_dir is not None:
            _load_adapter(pipe, adapter_dir)
        images_for_grid[label] = []
        for index, prompt in enumerate(prompts):
            generator = torch.Generator(device=device).manual_seed(args.seed + index)
            image = pipe(
                prompt, negative_prompt="low quality, blurry",
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]
            images_for_grid[label].append(image)
            image.save(output_dir / f"{label}_prompt{index:02d}.png")
        del pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fig, axes = plt.subplots(len(prompts), len(labels), figsize=(3.2 * len(labels), 3.2 * len(prompts)), squeeze=False)
    for row, prompt in enumerate(prompts):
        for col, (label, _) in enumerate(labels):
            axes[row][col].imshow(images_for_grid[label][row])
            axes[row][col].set_title(label)
            axes[row][col].axis("off")
        axes[row][0].set_ylabel(prompt[:28], fontsize=8)
    fig.suptitle(f"LoRA evaluation | seed={args.seed} | steps={args.steps}")
    fig.tight_layout()
    grid_path = output_dir / "lora_before_after.png"
    fig.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    metadata = runtime_metadata(seed=args.seed, model_id=args.model_id, model_revision=args.model_revision)
    metadata.update({
        "git_commit": git_commit(),
        "prompts": prompts,
        "labels": [label for label, _ in labels],
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "elapsed_seconds": round(time.time() - started, 3),
        "files": sorted(str(path.relative_to(output_dir)) for path in output_dir.glob("*.png")),
    })
    add_file_hashes(metadata, [output_dir / name for name in metadata["files"]])
    write_json(output_dir / "metadata.json", metadata)
    print(f"[saved] {grid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
