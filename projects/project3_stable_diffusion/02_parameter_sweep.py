"""
SD 参数扫描

固定 prompt 和 seed，扫描 CFG / steps / sampler 对生成的影响。

输出网格图 + 单独保存每张图。

用法：
    python 02_parameter_sweep.py --prompt "a cat playing chess" --seed 42
"""

import argparse
import time
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

try:
    from .experiment_utils import add_file_hashes, git_commit, runtime_metadata, write_json
except ImportError:  # direct ``python 02_parameter_sweep.py`` execution
    from experiment_utils import add_file_hashes, git_commit, runtime_metadata, write_json


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
                        default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument('--model_revision', type=str, default='451f4fe16113bff5a5d2269ed5ad43b0592e9a14')
    parser.add_argument('--preset', choices=['smoke', 'full'], default='full',
                        help='smoke uses a small matrix for validating the environment')
    parser.add_argument('--metadata_output', type=str, default=None)

    # 扫描范围
    parser.add_argument('--cfg_scales', nargs='+', type=float,
                        default=[1.0, 3.0, 7.5, 15.0, 25.0])
    parser.add_argument('--steps', nargs='+', type=int,
                        default=[10, 20, 50, 100])
    parser.add_argument('--samplers', nargs='+', type=str,
                        default=['DDIM', 'Euler-A', 'DPM++ 2M'])
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    args = parser.parse_args()
    started = time.time()

    if args.preset == 'smoke':
        args.cfg_scales = [1.0, 7.5]
        args.steps = [5, 10]
        args.samplers = ['DDIM']

    if any(value % 8 for value in (args.height, args.width)):
        raise ValueError('--height and --width must be divisible by 8')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading {args.model_id}...")
    load_kwargs = {'torch_dtype': dtype}
    if args.model_revision:
        load_kwargs['revision'] = args.model_revision
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, **load_kwargs).to(device)
    pipe.safety_checker = None  # 简化，建议自己使用时也禁用
    if hasattr(pipe, 'enable_attention_slicing'):
        pipe.enable_attention_slicing()
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
                   num_inference_steps=50 if args.preset == 'full' else 5,
                   guidance_scale=cfg, height=args.height, width=args.width,
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
                   height=args.height, width=args.width,
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
        try:
            scheduler_cls = SAMPLER_REGISTRY[sampler_name]
        except KeyError as exc:
            raise ValueError(f"unknown sampler {sampler_name!r}; choose from {sorted(SAMPLER_REGISTRY)}") from exc
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
        gen = torch.Generator(device=device).manual_seed(args.seed)
        img = pipe(args.prompt, negative_prompt=args.negative_prompt,
                   num_inference_steps=30 if args.preset == 'full' else 5,
                   guidance_scale=7.5, height=args.height, width=args.width,
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
    if args.preset == 'full':
        cfgs_2d = [3.0, 7.5, 15.0]
        steps_2d = [10, 20, 50]
    else:
        cfgs_2d = [3.0, 7.5]
        steps_2d = [5, 10]
    fig, axes = plt.subplots(len(cfgs_2d), len(steps_2d),
                              figsize=(3 * len(steps_2d), 3 * len(cfgs_2d)),
                              squeeze=False)
    for i, cfg in enumerate(cfgs_2d):
        for j, steps in enumerate(steps_2d):
            gen = torch.Generator(device=device).manual_seed(args.seed)
            img = pipe(args.prompt, negative_prompt=args.negative_prompt,
                       num_inference_steps=steps, guidance_scale=cfg,
                       height=args.height, width=args.width,
                       generator=gen).images[0]
            axes[i][j].imshow(img)
            axes[i][j].set_title(f"CFG={cfg}, steps={steps}")
            axes[i][j].axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'grid_2d.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  saved {output_dir / 'grid_2d.png'}")

    metadata_path = Path(args.metadata_output) if args.metadata_output else output_dir / 'metadata.json'
    metadata = runtime_metadata(seed=args.seed, model_id=args.model_id,
                                model_revision=args.model_revision)
    metadata.update({
        'git_commit': git_commit(),
        'preset': args.preset,
        'prompt': args.prompt,
        'negative_prompt': args.negative_prompt,
        'cfg_scales': args.cfg_scales,
        'steps': args.steps,
        'samplers': args.samplers,
        'height': args.height,
        'width': args.width,
        'elapsed_seconds': round(time.time() - started, 3) if 'started' in locals() else None,
        'files': sorted(str(p.relative_to(output_dir)) for p in output_dir.glob('*.png')),
    })
    add_file_hashes(metadata, [output_dir / name for name in metadata['files']])
    write_json(metadata_path, metadata)
    print(f"\n[done] All outputs in {output_dir}")
    print(f"[saved] metadata -> {metadata_path}")


if __name__ == '__main__':
    main()
