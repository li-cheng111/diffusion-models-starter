"""
Project 5 training script. Contains: TODO 19 (action chunk DDPM loss)
"""
import argparse
import os
import time
from copy import deepcopy

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from dataset import DemoDataset, collect_demos
from model import DiffusionPolicy
from obs_utils import state_dim_for, state_from_batch


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
# DDPM utility
# -----------------------------------------------------------------------------
class DDPMScheduler:
    def __init__(self, T=100, beta_min=1e-4, beta_max=0.02, device="cpu"):
        self.T = T
        betas = torch.linspace(beta_min, beta_max, T, device=device)
        alphas = 1 - betas
        self.alpha_cum = alphas.cumprod(0)
        self.sqrt_ac = self.alpha_cum.sqrt()
        self.sqrt_1mac = (1 - self.alpha_cum).sqrt()

    def add_noise(self, x_0, t, noise):
        """x_t = sqrt(ac_t) * x_0 + sqrt(1 - ac_t) * noise."""
        # x_0 shape (B, H, A), t shape (B,)
        shape = [t.shape[0]] + [1] * (x_0.ndim - 1)
        sa = self.sqrt_ac[t].reshape(shape)
        s1ma = self.sqrt_1mac[t].reshape(shape)
        return sa * x_0 + s1ma * noise


# -----------------------------------------------------------------------------
# TODO 19: Implement action chunk DDPM loss
# -----------------------------------------------------------------------------
def diffusion_loss(model, batch, scheduler, device, use_vision=True):
    """
    Compute DDPM noise prediction loss on action chunks.

    Args:
        model: DiffusionPolicy, callable as model(a_noisy, t, image, state) -> eps_pred
        batch: dict with keys image (B, 3, H, W), state (B, 2), goal (B, 2), action (B, H, A)
        scheduler: DDPMScheduler instance
        device: cuda or cpu
        use_vision: 决定条件用什么 state（见 obs_utils.py）

    Returns: scalar loss

    Steps:
        1) Sample timestep t ~ U(0, T) per-batch
        2) Sample noise eps ~ N(0, I) with shape of action
        3) Add noise to action: a_t = scheduler.add_noise(action, t, eps)
        4) Forward model: eps_pred = model(a_t, t, image, state)
        5) Return MSE(eps_pred, eps)

    注意 t 的取值范围是 [0, T)，别写成 [1, T] —— scheduler 的表是 0-indexed。
    """
    image = batch["image"].to(device)
    # use_vision=False 时 state 是 concat(agent_pos, target_pos)，见 obs_utils
    state = state_from_batch(batch, use_vision).to(device)
    action = batch["action"].to(device)
    B = action.shape[0]

    # ============================================================
    # TODO 19: Implement DDPM loss for action chunk (≈ 5 lines)
    # ============================================================
    raise NotImplementedError(
        "TODO 19: Implement action chunk DDPM loss. See README §阶段 2."
    )
    # ============================================================
    # END TODO 19
    # ============================================================


# -----------------------------------------------------------------------------
# EMA helper
# -----------------------------------------------------------------------------
@torch.no_grad()
def ema_update(ema_model, model, decay=0.999):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--collect", action="store_true", help="Collect demo data first")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    # Collect demos if needed
    demo_path = cfg["demo_path"]
    if args.collect or not os.path.exists(demo_path):
        collect_demos(
            n_demos=cfg["n_demos"],
            chunk_size=cfg["chunk_size"],
            n_distractors=cfg["n_distractors"],
            save_path=demo_path,
            seed=cfg.get("seed", 0),
        )

    # Dataset
    ds = DemoDataset(demo_path)
    loader = DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg.get("num_workers", 2), drop_last=True, pin_memory=True,
    )
    print(f"Demo dataset: {len(ds)} samples")

    # Model
    use_vision = cfg["use_vision"]
    model = DiffusionPolicy(
        horizon=cfg["chunk_size"],
        action_dim=2,
        state_dim=state_dim_for(use_vision),
        image_size=64,
        vision_out_dim=cfg["vision_out_dim"],
        hidden=cfg["hidden"],
        use_vision=use_vision,
    ).to(device)
    print(f"条件模式: {'image + agent_pos' if use_vision else 'agent_pos + target_pos（无视觉）'}"
          f"  state_dim={state_dim_for(use_vision)}")
    ema_model = deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg.get("weight_decay", 0.0))
    scheduler = DDPMScheduler(T=cfg["diffusion_steps"], device=device)
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Training
    model.train()
    step = 0
    losses = []
    t_start = time.time()
    while step < cfg["max_steps"]:
        for batch in loader:
            loss = diffusion_loss(model, batch, scheduler, device, use_vision)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema_update(ema_model, model, decay=cfg.get("ema_decay", 0.999))

            losses.append(loss.item())
            step += 1
            if step % cfg["log_every"] == 0:
                avg = sum(losses[-100:]) / min(len(losses), 100)
                print(f"step {step}/{cfg['max_steps']} | loss {loss.item():.4f} "
                      f"(avg100 {avg:.4f}) | {step / (time.time() - t_start):.1f}/s")
            if step % cfg["save_every"] == 0:
                p = os.path.join(cfg["ckpt_dir"], f"model_{step:06d}.pt")
                torch.save(
                    {"model": model.state_dict(), "ema": ema_model.state_dict(),
                     "step": step, "config": cfg},
                    p,
                )
                print(f"Saved {p}")
            if step >= cfg["max_steps"]:
                break

    final = os.path.join(cfg["ckpt_dir"], "model_final.pt")
    torch.save({"model": model.state_dict(), "ema": ema_model.state_dict(),
                "step": step, "config": cfg}, final)
    print(f"Done. Final ckpt: {final}")


if __name__ == "__main__":
    main()
