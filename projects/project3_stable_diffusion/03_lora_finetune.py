"""Train a small LoRA adapter for Stable Diffusion 1.5.

This is intentionally an educational, transparent training loop.  The VAE and
CLIP text encoder stay frozen; only attention LoRA parameters in the UNet are
updated.  Large model/data files belong outside Git.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

try:
    from .experiment_utils import git_commit, runtime_metadata, set_seed, write_json
except ImportError:  # direct script execution from the project directory
    from experiment_utils import git_commit, runtime_metadata, set_seed, write_json


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
LATENT_SCALE = 0.18215


class InstanceDataset(Dataset):
    """Each image is paired with the same instance prompt."""

    def __init__(self, data_dir: str | os.PathLike[str], prompt: str,
                 tokenizer, size: int = 512, random_flip: bool = False):
        self.data_dir = Path(data_dir)
        self.image_paths = sorted(
            p for p in self.data_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.data_dir}")
        self.prompt = prompt
        self.tokenizer = tokenizer
        ops = [transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
               transforms.CenterCrop(size)]
        if random_flip:
            ops.append(transforms.RandomHorizontalFlip())
        ops.extend([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        with Image.open(self.image_paths[idx]) as source:
            image = source.convert("RGB")
        image = self.transform(image)
        token_ids = self.tokenizer(
            self.prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        ).input_ids[0]
        return image, token_ids


def add_lora_to_unet(unet, rank: int = 8):
    """Attach LoRA to all SD attention projections and freeze the base UNet."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    # Diffusers' UNet implements PeftAdapterMixin in current releases.  Keep a
    # PEFT fallback for older course environments.
    if hasattr(unet, "add_adapter"):
        unet.add_adapter(config)
    else:  # pragma: no cover - compatibility path for old diffusers
        unet = get_peft_model(unet, config)
    unet.requires_grad_(False)
    for name, parameter in unet.named_parameters():
        if "lora" in name.lower():
            parameter.requires_grad_(True)
            parameter.data = parameter.data.float()
    trainable = [p for p in unet.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA adapter was not attached; no trainable parameters found")
    print(f"Trainable LoRA parameters: {sum(p.numel() for p in trainable) / 1e6:.3f}M")
    return unet


def _autocast(device: torch.device, dtype: torch.dtype | None):
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)


def train_one_step(batch, unet, vae, text_encoder, scheduler, optimizer,
                   device, weight_dtype, autocast_dtype, scaler,
                   max_grad_norm: float = 1.0):
    """Run one noise-prediction update and return a finite Python float."""
    images, token_ids = batch
    images = images.to(device, dtype=weight_dtype, non_blocking=True)
    token_ids = token_ids.to(device, non_blocking=True)
    batch_size = images.shape[0]
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample() * LATENT_SCALE
        latents = latents.float()
        text_emb = text_encoder(token_ids)[0]

    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps,
        (batch_size,), device=device, dtype=torch.long,
    )
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    with _autocast(device, autocast_dtype):
        noise_pred = unet(
            noisy_latents.to(autocast_dtype or torch.float32),
            timesteps,
            encoder_hidden_states=text_emb.to(autocast_dtype or torch.float32),
        ).sample
        loss = F.mse_loss(noise_pred.float(), noise.float())
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite LoRA loss: {loss.detach().item()}")

    scaler.scale(loss).backward()
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    trainable = [p for p in unet.parameters() if p.requires_grad]
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach().item())


def save_lora_weights(unet, output_dir: str | os.PathLike[str]) -> Path:
    """Save adapter-only weights in a format reloadable by Diffusers."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    if hasattr(unet, "save_lora_adapter"):
        unet.save_lora_adapter(
            target, adapter_name="default",
            weight_name="pytorch_lora_weights.safetensors",
            safe_serialization=True,
        )
        path = target / "pytorch_lora_weights.safetensors"
    else:  # pragma: no cover - old diffusers fallback
        unet.save_pretrained(target, safe_serialization=True)
        path = target / "adapter_model.safetensors"
    if not path.exists():
        candidates = sorted(target.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"LoRA save produced no safetensors in {target}")
        path = candidates[0]
    print(f"[saved] LoRA weights -> {path}")
    return path


def _make_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except TypeError:  # pragma: no cover - PyTorch < 2.0
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def save_validation_images(
    prompts: list[str], output_dir: Path, step: int, seed: int,
    model_id: str, model_revision: str | None, tokenizer, text_encoder,
    vae, unet, train_scheduler, device: torch.device, dtype: torch.dtype,
) -> list[Path]:
    """Generate fixed-seed validation images without reloading base weights."""
    if not prompts:
        return []
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDIMScheduler.from_config(train_scheduler.config),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe.unet, "set_adapter"):
        pipe.unet.set_adapter("default")
    paths: list[Path] = []
    for index, prompt in enumerate(prompts):
        generator = torch.Generator(device=device).manual_seed(seed + index)
        with _autocast(device, dtype if device.type == "cuda" and dtype != torch.float32 else None):
            image = pipe(
                prompt, negative_prompt="low quality, blurry",
                num_inference_steps=20, guidance_scale=7.5,
                generator=generator,
            ).images[0]
        path = validation_dir / f"step-{step:04d}_prompt-{index:02d}.png"
        image.save(path)
        paths.append(path)
    del pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", required=True)
    parser.add_argument("--instance_prompt", required=True)
    parser.add_argument("--output_dir", default="./outputs/lora")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument("--num_train_steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpointing_steps", type=int, default=200)
    parser.add_argument("--validation_prompts", nargs="*", default=[])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    args = parser.parse_args()
    if args.num_train_steps <= 0 or args.checkpointing_steps <= 0:
        raise ValueError("num_train_steps and checkpointing_steps must be positive")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.mixed_precision != "no":
        print("[warn] CPU has no supported fp16 training path; using fp32")
        args.mixed_precision = "no"
    weight_dtype = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.mixed_precision]
    autocast_dtype = None if args.mixed_precision == "no" else weight_dtype
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_path = output_dir / "training_metrics.jsonl"

    load_kwargs = {"revision": args.model_revision} if args.model_revision else {}
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer", **load_kwargs)
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder", **load_kwargs).to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae", **load_kwargs).to(device, dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", **load_kwargs).to(device, dtype=torch.float32)
    scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler", **load_kwargs)
    text_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    unet.requires_grad_(False)
    unet = add_lora_to_unet(unet, rank=args.rank)
    if args.gradient_checkpointing and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    dataset = InstanceDataset(args.train_data_dir, args.instance_prompt, tokenizer, random_flip=args.random_flip)
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        generator=loader_generator,
    )
    trainable = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler = _make_scaler(device, enabled=args.mixed_precision == "fp16")
    metadata = runtime_metadata(seed=args.seed, model_id=args.model_id,
                                model_revision=args.model_revision)
    metadata.update({
        "git_commit": git_commit(),
        "dataset_dir": str(Path(args.train_data_dir).resolve()),
        "dataset_size": len(dataset),
        "instance_prompt": args.instance_prompt,
        "num_train_steps": args.num_train_steps,
        "checkpointing_steps": args.checkpointing_steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "rank": args.rank,
        "alpha": args.rank,
        "max_grad_norm": args.max_grad_norm,
        "random_flip": args.random_flip,
        "mixed_precision": args.mixed_precision,
        "gradient_checkpointing": args.gradient_checkpointing,
        "validation_prompts": args.validation_prompts,
    })
    write_json(output_dir / "training_config.json", metadata)
    _write_jsonl(logs_path, {"event": "start", **metadata})

    start = time.time()
    step = 0
    losses: list[float] = []
    progress = tqdm(total=args.num_train_steps, desc="LoRA")
    while step < args.num_train_steps:
        for batch in loader:
            loss = train_one_step(
                batch, unet, vae, text_encoder, scheduler, optimizer,
                device, weight_dtype, autocast_dtype, scaler,
                max_grad_norm=args.max_grad_norm,
            )
            step += 1
            losses.append(loss)
            progress.update(1)
            progress.set_postfix(loss=f"{loss:.4f}")
            _write_jsonl(logs_path, {"event": "step", "step": step, "loss": loss, "elapsed_seconds": time.time() - start})
            if step % args.checkpointing_steps == 0 or step == args.num_train_steps:
                checkpoint_dir = output_dir / f"checkpoint-{step:04d}"
                save_lora_weights(unet, checkpoint_dir)
                write_json(checkpoint_dir / "step.json", {"step": step, "loss": loss, "seed": args.seed})
                validation_paths = save_validation_images(
                    args.validation_prompts, output_dir, step, args.seed,
                    args.model_id, args.model_revision, tokenizer, text_encoder,
                    vae, unet, scheduler, device, weight_dtype,
                )
                if validation_paths:
                    _write_jsonl(logs_path, {
                        "event": "validation", "step": step,
                        "files": [str(path) for path in validation_paths],
                    })
            if step >= args.num_train_steps:
                break
    progress.close()
    final_path = save_lora_weights(unet, output_dir)
    summary = {
        **metadata,
        "final_step": step,
        "final_loss": losses[-1],
        "initial_loss": losses[0],
        "mean_last_50_loss": sum(losses[-50:]) / min(50, len(losses)),
        "elapsed_seconds": round(time.time() - start, 3),
        "final_weights": str(final_path),
    }
    write_json(output_dir / "training_summary.json", summary)
    _write_jsonl(logs_path, {"event": "finish", **summary})
    print(f"[done] {step} steps; mean(last 50)={summary['mean_last_50_loss']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
