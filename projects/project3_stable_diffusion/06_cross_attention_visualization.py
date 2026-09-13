"""Visualize Stable Diffusion cross-attention for selected prompt tokens.

The script uses the model components directly and a small, explicit attention
processor.  This keeps the challenge reproducible across pipeline callback
APIs and makes it clear that the maps are the conditional half of CFG.
"""

from __future__ import annotations

import argparse
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

try:
    from .experiment_utils import add_file_hashes, git_commit, runtime_metadata, set_seed, write_json
except ImportError:  # direct script execution from the project directory
    from experiment_utils import add_file_hashes, git_commit, runtime_metadata, set_seed, write_json


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
LATENT_SCALE = 0.18215


class AttentionCollector:
    def __init__(self, selected_steps: set[int]):
        self.selected_steps = selected_steps
        self.current_step: int | None = None
        self.maps: dict[str, list[torch.Tensor]] = defaultdict(list)

    def begin_step(self, step: int) -> None:
        self.current_step = step if step in self.selected_steps else None

    def add(self, name: str, probs: torch.Tensor, batch_size: int, heads: int) -> None:
        if self.current_step is None:
            return
        # probs is (batch * heads, query, key); keep the conditional CFG half.
        query_len, key_len = probs.shape[-2:]
        if batch_size < 2 or int(math.sqrt(query_len)) ** 2 != query_len:
            return
        reshaped = probs.reshape(batch_size, heads, query_len, key_len)[1]
        self.maps[name].append(reshaped.mean(dim=0).detach().float().cpu())


class CaptureAttnProcessor:
    """Diffusers AttnProcessor-compatible implementation with a capture hook."""

    def __init__(self, name: str, collector: AttentionCollector):
        self.name = name
        self.collector = collector

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", False) and getattr(attn, "norm_encoder_hidden_states", None) is not None:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        heads = getattr(attn, "heads", 1)
        head_dim = inner_dim // heads
        query = query.view(batch_size, -1, heads, head_dim).transpose(1, 2).reshape(batch_size * heads, -1, head_dim)
        key = key.view(batch_size, -1, heads, head_dim).transpose(1, 2).reshape(batch_size * heads, -1, head_dim)
        value = value.view(batch_size, -1, heads, head_dim).transpose(1, 2).reshape(batch_size * heads, -1, head_dim)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, key.shape[1], batch_size)
            if attention_mask.ndim == 3:
                attention_mask = attention_mask.repeat_interleave(heads, dim=0)
        probs = attn.get_attention_scores(query, key, attention_mask)
        self.collector.add(self.name, probs, batch_size, heads)
        hidden_states = torch.bmm(probs, value)
        hidden_states = hidden_states.reshape(batch_size, heads, -1, head_dim).transpose(1, 2).reshape(batch_size, -1, heads * head_dim)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if getattr(attn, "residual_connection", True):
            hidden_states = hidden_states + residual
        return hidden_states / getattr(attn, "rescale_output_factor", 1.0)


def _install_processors(unet, collector: AttentionCollector) -> list[str]:
    names = []
    for name, module in unet.named_modules():
        if hasattr(module, "processor") and hasattr(module, "to_q") and hasattr(module, "to_k"):
            # Cross-attention modules have a non-null cross_attention_dim.  The
            # self-attention blocks are left untouched.
            if getattr(module, "cross_attention_dim", None) is not None:
                module.processor = CaptureAttnProcessor(name, collector)
                names.append(name)
    if not names:
        raise RuntimeError("no cross-attention modules found; check Diffusers compatibility")
    return names


def _token_indices(tokenizer, prompt: str, requested: list[str]) -> tuple[list[int], list[str], list[int]]:
    encoded = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    ids = encoded.input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    special = set(tokenizer.all_special_ids)
    wanted: list[int] = []
    for query in requested:
        query_lower = query.lower()
        for index, token in enumerate(tokens):
            clean = token.replace("</w>", "").replace("Ġ", "").lower()
            if ids[index] not in special and (clean == query_lower or query_lower in clean):
                wanted.append(index)
    wanted = sorted(set(wanted))
    if not wanted:
        raise ValueError(f"none of {requested!r} matched tokens {tokens!r}")
    return wanted, tokens, ids


def _overlay(image: Image.Image, heat: np.ndarray, title: str, output: Path) -> None:
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)
    heat = heat - heat.min()
    if heat.max() > 0:
        heat /= heat.max()
    heat_image = Image.fromarray(np.uint8(heat * 255)).resize(image.size, Image.Resampling.BICUBIC)
    cmap = plt.get_cmap("magma")(np.asarray(heat_image) / 255.0)[..., :3]
    base = np.asarray(image).astype(np.float32) / 255.0
    blended = np.clip(0.55 * base + 0.45 * cmap, 0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image); axes[0].set_title("generated"); axes[0].axis("off")
    axes[1].imshow(blended); axes[1].set_title(title); axes[1].axis("off")
    fig.tight_layout(); fig.savefig(output, dpi=140, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="a red cat wearing a blue wizard hat in a green forest")
    parser.add_argument("--tokens", nargs="+", default=["cat", "wizard", "hat", "forest"])
    parser.add_argument("--output_dir", default="./outputs/attention")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=7.5)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"torch_dtype": dtype}
    if args.model_revision:
        kwargs["revision"] = args.model_revision
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, **kwargs).to(device)
    pipe.safety_checker = None
    tokenizer = pipe.tokenizer
    text_encoder: CLIPTextModel = pipe.text_encoder
    unet: UNet2DConditionModel = pipe.unet
    vae: AutoencoderKL = pipe.vae
    scheduler: DDIMScheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    text_ids = tokenizer(args.prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
    uncond_ids = tokenizer("", padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        text_emb = torch.cat([text_encoder(uncond_ids)[0], text_encoder(text_ids)[0]], dim=0)
    collector = AttentionCollector(selected_steps=set(range(max(0, args.steps // 3), args.steps)))
    attention_names = _install_processors(unet, collector)
    scheduler.set_timesteps(args.steps, device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent = torch.randn((1, 4, 64, 64), generator=generator, device=device, dtype=dtype) * scheduler.init_noise_sigma
    started = time.time()
    for step_index, timestep in enumerate(scheduler.timesteps):
        collector.begin_step(step_index)
        model_input = scheduler.scale_model_input(torch.cat([latent, latent]), timestep)
        with torch.no_grad():
            noise = unet(model_input, timestep, encoder_hidden_states=text_emb).sample
        noise_uncond, noise_cond = noise.chunk(2)
        guided = noise_uncond + args.cfg * (noise_cond - noise_uncond)
        latent = scheduler.step(guided, timestep, latent).prev_sample
    with torch.no_grad():
        image_tensor = (vae.decode(latent / LATENT_SCALE).sample / 2 + 0.5).clamp(0, 1)
    image = Image.fromarray(np.uint8(image_tensor[0].permute(1, 2, 0).float().cpu().numpy() * 255))
    image_path = output_dir / "generated.png"; image.save(image_path)
    selected_indices, token_strings, token_ids = _token_indices(tokenizer, args.prompt, args.tokens)
    aggregates: dict[int, list[torch.Tensor]] = defaultdict(list)
    for layer_maps in collector.maps.values():
        if not layer_maps:
            continue
        stacked = torch.stack(layer_maps).mean(dim=0)
        query_len = stacked.shape[1]
        resolution = int(math.sqrt(query_len))
        if resolution in (16, 32):
            aggregates[resolution].append(stacked)
    if not aggregates:
        raise RuntimeError("no 16x16 or 32x32 cross-attention maps were captured")
    token_heatmaps: dict[int, np.ndarray] = {}
    for index in selected_indices:
        maps = []
        for resolution, values in aggregates.items():
            maps.append(torch.stack([value[:, index].reshape(resolution, resolution) for value in values]).mean(dim=0))
        token_heatmaps[index] = torch.stack([torch.nn.functional.interpolate(m[None, None], size=(32, 32), mode="bilinear", align_corners=False)[0, 0] for m in maps]).mean(dim=0).numpy()
    overlay_files = []
    for index, heat in token_heatmaps.items():
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", token_strings[index].replace("Ġ", "")).strip("._")
        label = label or f"token{index}"
        output = output_dir / f"attention_{index:02d}_{label}.png"
        _overlay(image, heat, f"token {index}: {token_strings[index]}", output)
        overlay_files.append(output)
    metadata = runtime_metadata(seed=args.seed, model_id=args.model_id, model_revision=args.model_revision)
    metadata.update({
        "git_commit": git_commit(), "prompt": args.prompt, "tokens_requested": args.tokens,
        "token_ids": token_ids, "token_strings": token_strings, "token_indices": selected_indices,
        "steps": args.steps, "cfg": args.cfg, "captured_layers": attention_names,
        "captured_resolutions": sorted(aggregates), "elapsed_seconds": round(time.time() - started, 3),
        "files": [str(image_path.relative_to(output_dir))] + [str(path.relative_to(output_dir)) for path in overlay_files],
    })
    add_file_hashes(metadata, [output_dir / name for name in metadata["files"]])
    write_json(output_dir / "metadata.json", metadata)
    print(f"[saved] attention maps -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
