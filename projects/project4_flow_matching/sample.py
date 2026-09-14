"""Project 4: Flow Matching Sampling

含 TODO 17 (Euler ODE sampler) 和 TODO 18 (CFG for FM)
"""
import argparse
import os

import torch
import torchvision
import yaml

from train import build_model


@torch.no_grad()
def euler_sample(model, n_samples, n_steps, num_classes, class_label, image_size=32, device='cuda'):
    """
    ============================================================
    TODO 17: 实现 Euler ODE sampler
    ============================================================

    输入:
        model: trained FM model, model(x, t, y) -> v
        n_samples: 生成多少张图
        n_steps: ODE 积分步数
        num_classes: 总类别数（最后一个 index 是 null）
        class_label: 目标 class（int），生成该类别的图
        image_size: 32 for CIFAR-10
        device: 'cuda' 或 'cpu'

    要求:
        1. 初始化 x = torch.randn(n_samples, 3, image_size, image_size)
        2. y = torch.full((n_samples,), class_label, dtype=long)
        3. timesteps = torch.linspace(0, 1, n_steps + 1)  # [0, ..., 1]
        4. For i in range(n_steps):
             t = timesteps[i]
             dt = timesteps[i+1] - timesteps[i]
             v = model(x, t.repeat(n_samples), y)
             x = x + v * dt
        5. 返回 x（此时 x ≈ x_1，clamp 到 [-1, 1] 后转为图像）

    Hints:
        - t.repeat(n_samples) 让 model 接收 (B,) shape 的 t
        - 注意是 + v * dt（FM 是 t=0 → t=1 前向积分）
    ============================================================
    """
    if n_samples <= 0 or n_steps <= 0:
        raise ValueError("n_samples and n_steps must be positive")
    x = torch.randn(n_samples, 3, image_size, image_size, device=device)
    y = torch.full((n_samples,), class_label, dtype=torch.long, device=device)
    timesteps = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    for i in range(n_steps):
        t = timesteps[i]
        dt = timesteps[i + 1] - t
        velocity = model(x, t.expand(n_samples), y)
        x = x + velocity * dt
    return x


@torch.no_grad()
def heun_sample(model, n_samples, n_steps, num_classes, class_label, image_size=32, device='cuda'):
    """Second-order Heun integration for the FM velocity field."""
    if n_samples <= 0 or n_steps <= 0:
        raise ValueError("n_samples and n_steps must be positive")
    x = torch.randn(n_samples, 3, image_size, image_size, device=device)
    y = torch.full((n_samples,), class_label, dtype=torch.long, device=device)
    timesteps = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    for i in range(n_steps):
        t0, t1 = timesteps[i], timesteps[i + 1]
        dt = t1 - t0
        v0 = model(x, t0.expand(n_samples), y)
        x_euler = x + v0 * dt
        v1 = model(x_euler, t1.expand(n_samples), y)
        x = x + 0.5 * (v0 + v1) * dt
    return x


@torch.no_grad()
def euler_sample_cfg(model, n_samples, n_steps, num_classes, class_label, cfg_scale,
                     image_size=32, device='cuda'):
    """
    ============================================================
    TODO 18: 实现 CFG for Flow Matching
    ============================================================

    输入:
        cfg_scale: guidance scale (e.g., 3.0, 7.5)
        其他同 euler_sample

    要求:
        1. 在每步 ODE 中，做两次 model forward:
           v_cond = model(x, t, y_cond)
           v_uncond = model(x, t, y_null)  # y_null = num_classes
        2. v = v_uncond + cfg_scale * (v_cond - v_uncond)
        3. 用 v 走 Euler 一步

    Hints:
        - 工程优化：把 x, y 都 cat 成 batch=2*n 一次 forward 完成
          x_in = torch.cat([x, x], 0)
          y_in = torch.cat([y_null, y_cond], 0)
          v_uncond, v_cond = model(x_in, t.repeat(2*n_samples), y_in).chunk(2)
    ============================================================
    """
    if n_samples <= 0 or n_steps <= 0:
        raise ValueError("n_samples and n_steps must be positive")
    x = torch.randn(n_samples, 3, image_size, image_size, device=device)
    y_cond = torch.full((n_samples,), class_label, dtype=torch.long, device=device)
    y_null = torch.full_like(y_cond, num_classes)
    timesteps = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    for i in range(n_steps):
        t = timesteps[i]
        dt = timesteps[i + 1] - t
        x_in = torch.cat([x, x], dim=0)
        y_in = torch.cat([y_null, y_cond], dim=0)
        velocity_uncond, velocity_cond = model(
            x_in, t.expand(2 * n_samples), y_in
        ).chunk(2, dim=0)
        velocity = velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)
        x = x + velocity * dt
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to checkpoint')
    parser.add_argument('--output', default='samples/')
    parser.add_argument('--nfe', type=int, nargs='+', default=[4, 8, 16, 32, 50])
    parser.add_argument('--cfg', type=float, nargs='+', default=[0.0, 1.0, 3.0, 7.5])
    parser.add_argument('--n_samples', type=int, default=64)
    parser.add_argument('--class_label', type=int, default=0)
    parser.add_argument('--no_ema', action='store_true', help='use raw model weights')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.model, map_location=device)
    cfg = ckpt['cfg']
    model = build_model(cfg).to(device)

    state = ckpt['model'] if args.no_ema else ckpt.get('ema', ckpt['model'])
    model.load_state_dict(state)
    model.eval()
    num_classes = cfg['model']['num_classes']

    os.makedirs(args.output, exist_ok=True)

    for nfe in args.nfe:
        for cfg_scale in args.cfg:
            print(f"Sampling NFE={nfe}, CFG={cfg_scale}...")
            if cfg_scale == 0.0:
                # Unconditional (no CFG)
                samples = euler_sample(model, args.n_samples, nfe,
                                       num_classes, args.class_label, device=device)
            else:
                samples = euler_sample_cfg(model, args.n_samples, nfe,
                                           num_classes, args.class_label, cfg_scale, device=device)

            samples = ((samples.clamp(-1, 1) + 1) / 2).cpu()
            grid = torchvision.utils.make_grid(samples, nrow=8)
            fp = os.path.join(args.output, f'nfe{nfe}_cfg{cfg_scale}_class{args.class_label}.png')
            torchvision.utils.save_image(grid, fp)
            print(f"  Saved to {fp}")


if __name__ == '__main__':
    main()
