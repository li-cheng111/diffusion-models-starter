"""
SD 参数扫描

固定 prompt 和 seed，扫描 CFG / steps / sampler 对生成的影响。

输出网格图 + 单独保存每张图。

用法：
    python 02_parameter_sweep.py --prompt "a cat playing chess" --seed 42
"""

import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)


SAMPLER_REGISTRY = {
    "DDIM": DDIMScheduler,
    "Euler-A": EulerAncestralDiscreteScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str,
                        default="a photograph of a cat wearing a wizard hat, fantasy art")
    parser.add_argument('--negative_prompt', type=str,
                        default="low quality, blurry")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./outputs/sweep')
    parser.add_argument('--model_id', type=str,
                        default="runwayml/stable-diffusion-v1-5")

    # 扫描范围
    parser.add_argument('--cfg_scales', nargs='+', type=float,
                        default=[1.0, 3.0, 7.5, 15.0, 25.0])
    parser.add_argument('--steps', nargs='+', type=int,
                        default=[10, 20, 50, 100])
    parser.add_argument('--samplers', nargs='+', type=str,
                        default=['DDIM', 'Euler-A', 'DPM++ 2M'])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading {args.model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    pipe.safety_checker = None  # 简化，建议自己使用时也禁用
    pipe.set_progress_bar_config(disable=True)

    # ─────────── 实验 1: CFG scale 扫描 ───────────
    print("\n=== Experiment 1: CFG scale sweep ===")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    fig, axes = plt.subplots(1, len(args.cfg_scales),
                              figsize=(3 * len(args.cfg_scales), 3), squeeze=False)
    axes = axes[0]
    for i, cfg in enumerate(args.cfg_scales):
        gen = torch.Generator(device=device).manual_seed(args.seed)
        img = pipe(args.prompt, negative_prompt=args.negative_prompt,
                   num_inference_steps=50, guidance_scale=cfg,
                   generator=gen).images[0]
        axes[i].imshow(img)
        axes[i].set_title(f"CFG={cfg}")
        axes[i].axis('off')
        img.save(output_dir / f"cfg_{cfg}.png")
    plt.suptitle(f"CFG sweep: '{args.prompt[:50]}'")
    plt.tight_layout()
    plt.savefig(output_dir / 'sweep_cfg.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  saved {output_dir / 'sweep_cfg.png'}")

    # ─────────── 实验 2: steps 扫描 ───────────
    print("\n=== Experiment 2: Steps sweep ===")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    fig, axes = plt.subplots(1, len(args.steps), figsize=(3 * len(args.steps), 3),
                              squeeze=False)
    axes = axes[0]
    for i, steps in enumerate(args.steps):
        gen = torch.Generator(device=device).manual_seed(args.seed)
        img = pipe(args.prompt, negative_prompt=args.negative_prompt,
                   num_inference_steps=steps, guidance_scale=7.5,
                   generator=gen).images[0]
        axes[i].imshow(img)
        axes[i].set_title(f"steps={steps}")
        axes[i].axis('off')
        img.save(output_dir / f"steps_{steps}.png")
    plt.suptitle(f"Steps sweep (CFG=7.5)")
    plt.tight_layout()
    plt.savefig(output_dir / 'sweep_steps.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  saved {output_dir / 'sweep_steps.png'}")

    # ─────────── 实验 3: sampler 扫描 ───────────
    print("\n=== Experiment 3: Sampler sweep ===")
    fig, axes = plt.subplots(1, len(args.samplers), figsize=(3 * len(args.samplers), 3),
                              squeeze=False)
    axes = axes[0]
    for i, sampler_name in enumerate(args.samplers):
        scheduler_cls = SAMPLER_REGISTRY[sampler_name]
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
        gen = torch.Generator(device=device).manual_seed(args.seed)
        img = pipe(args.prompt, negative_prompt=args.negative_prompt,
                   num_inference_steps=30, guidance_scale=7.5,
                   generator=gen).images[0]
        axes[i].imshow(img)
        axes[i].set_title(f"{sampler_name}")
        axes[i].axis('off')
        img.save(output_dir / f"sampler_{sampler_name.replace(' ', '_').replace('+','p')}.png")
    plt.suptitle(f"Sampler sweep (30 steps, CFG=7.5)")
    plt.tight_layout()
    plt.savefig(output_dir / 'sweep_sampler.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  saved {output_dir / 'sweep_sampler.png'}")

    # ─────────── 实验 4: 2D 网格 CFG × steps ───────────
    print("\n=== Experiment 4: 2D grid CFG × Steps ===")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    cfgs_2d = [3.0, 7.5, 15.0]
    steps_2d = [10, 20, 50]
    fig, axes = plt.subplots(len(cfgs_2d), len(steps_2d),
                              figsize=(3 * len(steps_2d), 3 * len(cfgs_2d)),
                              squeeze=False)
    for i, cfg in enumerate(cfgs_2d):
        for j, steps in enumerate(steps_2d):
            gen = torch.Generator(device=device).manual_seed(args.seed)
            img = pipe(args.prompt, negative_prompt=args.negative_prompt,
                       num_inference_steps=steps, guidance_scale=cfg,
                       generator=gen).images[0]
            axes[i][j].imshow(img)
            axes[i][j].set_title(f"CFG={cfg}, steps={steps}")
            axes[i][j].axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'grid_2d.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  saved {output_dir / 'grid_2d.png'}")

    print(f"\n[done] All outputs in {output_dir}")


if __name__ == '__main__':
    main()
