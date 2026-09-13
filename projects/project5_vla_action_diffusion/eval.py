"""
Closed-loop evaluation for Project 5. Contains: TODO 21 (eval loop).
"""
import argparse
import os

import numpy as np
import torch
import yaml

from env import Reach2DEnv
from model import DiffusionPolicy
from obs_utils import state_dim_for, state_from_obs
from train import DDPMScheduler, load_config


@torch.no_grad()
def ddpm_sample_action(model, scheduler, image, state, horizon=32, action_dim=2,
                       n_steps=20, device="cuda"):
    """DDPM ancestral sampling for action chunk."""
    model.eval()
    B = image.shape[0]
    a = torch.randn(B, horizon, action_dim, device=device)
    T = scheduler.T
    step = T // n_steps
    timesteps = list(range(0, T, step))[::-1]
    for i, ti in enumerate(timesteps):
        t_tensor = torch.full((B,), ti, device=device, dtype=torch.long)
        eps_pred = model(a, t_tensor, image=image, state=state)
        # Reconstruct x_0
        ac_t = scheduler.alpha_cum[ti]
        x_0 = (a - (1 - ac_t).sqrt() * eps_pred) / ac_t.sqrt()
        x_0 = x_0.clamp(-1.5, 1.5)
        # Move to previous step
        if i + 1 < len(timesteps):
            ti_prev = timesteps[i + 1]
            ac_prev = scheduler.alpha_cum[ti_prev]
            a = ac_prev.sqrt() * x_0 + (1 - ac_prev).sqrt() * eps_pred
        else:
            a = x_0
    return a  # (B, H, A)


# =============================================================================
# TODO 21: Implement closed-loop evaluation
# =============================================================================
def evaluate(model, scheduler, cfg, device, n_episodes=100, exec_steps=10,
             chunk_size=32, n_sample_steps=20, verbose=False):
    """
    Closed-loop policy evaluation.

    For each of n_episodes:
        1) env.reset() → obs
        2) Loop until done:
            a) Every `exec_steps` step (or step 0), sample fresh action chunk
            b) Execute the next action from the chunk
            c) Update obs, check info['success']/'collision']
        3) Record success / collision / steps

    Args:
        model: trained DiffusionPolicy
        scheduler: DDPMScheduler
        cfg: config dict
        device: cuda or cpu
        n_episodes: number of evaluation episodes
        exec_steps: how many actions from each generated chunk to execute before re-planning
        chunk_size: action chunk length (must match training cfg['chunk_size'])
        n_sample_steps: DDPM sampling steps

    Returns:
        dict with keys 'success_rate', 'collision_rate', 'avg_steps', 'episodes'

    可以直接用的零件（不用自己拼条件表示）：
        env = Reach2DEnv(n_distractors=n_dist, seed=1000 + ep)   # 固定 seed 才可复现
        obs = env.reset()
        image = torch.from_numpy(obs["image"]).permute(2, 0, 1).float() / 127.5 - 1.0
        image = image.unsqueeze(0).to(device)                    # (1, 3, 64, 64)
        state = torch.from_numpy(state_from_obs(obs, use_vision)).unsqueeze(0).to(device)
        chunk = ddpm_sample_action(model, scheduler, image, state,
                                   horizon=chunk_size, n_steps=n_sample_steps,
                                   device=device)[0].cpu().numpy()   # (H, 2)

    ⚠️ 两个最容易错的点：
    1. image 的归一化必须和 dataset.py 里一模一样（/127.5 - 1.0），
       否则训练分布和评估分布对不上，成功率会莫名其妙地低。
    2. 重规划的计数：`exec_steps` 是"执行几步后重新采样"，
       从 chunk 里取的下标是 step % exec_steps，不是 step % chunk_size。
       README §阶段 4 的伪代码给的是前者。

    评估用的 seed 要和训练数据的 seed 区间错开（collect_demos 用的是 0..2*n_demos），
    否则你测的是训练集，成功率虚高。
    """
    n_dist = cfg["n_distractors"]
    use_vision = cfg["use_vision"]

    # ============================================================
    # TODO 21: Implement closed-loop evaluation (≈ 25-40 lines)
    # ============================================================
    raise NotImplementedError(
        "TODO 21: Implement closed-loop evaluation. See README §阶段 4."
    )
    # ============================================================
    # END TODO 21
    # ============================================================


# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--exec_steps", type=int, default=10)
    parser.add_argument("--n_sample_steps", type=int, default=20)
    parser.add_argument("--use_ema", action="store_true", default=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiffusionPolicy(
        horizon=cfg["chunk_size"],
        action_dim=2,
        state_dim=state_dim_for(cfg["use_vision"]),
        image_size=64,
        vision_out_dim=cfg["vision_out_dim"],
        hidden=cfg["hidden"],
        use_vision=cfg["use_vision"],
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["ema"] if args.use_ema and "ema" in ckpt else ckpt["model"]
    model.load_state_dict(state_dict)
    print(f"Loaded {'EMA' if args.use_ema else 'main'} model from {args.ckpt}")

    scheduler = DDPMScheduler(T=cfg["diffusion_steps"], device=device)

    results = evaluate(
        model, scheduler, cfg, device,
        n_episodes=args.n_episodes,
        exec_steps=args.exec_steps,
        chunk_size=cfg["chunk_size"],
        n_sample_steps=args.n_sample_steps,
        verbose=True,
    )

    print("\n=== Evaluation Results ===")
    print(f"Success rate:   {results['success_rate']*100:.1f}%")
    print(f"Collision rate: {results['collision_rate']*100:.1f}%")
    print(f"Avg steps:      {results['avg_steps']:.1f}")
    print(f"Total episodes: {results['episodes']}")


if __name__ == "__main__":
    main()
