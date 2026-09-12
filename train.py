"""Config-driven DDPM training with EMA, AMP, logging, and checkpoints."""

from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torchvision.utils as vutils
import yaml
from torch.cuda.amp import GradScaler, autocast

from dataset import denormalize, get_dataloader
from diffusion import p_losses, p_sample_loop
from model import UNet
from schedule import DDPMSchedule


class EMA:
    """Exponential moving average of model parameters and buffers."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_parameter, parameter in zip(
            self.ema_model.parameters(), model.parameters()
        ):
            ema_parameter.mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)
        for ema_buffer, buffer in zip(self.ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "model": self.ema_model.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "model" in state:
            self.ema_model.load_state_dict(state["model"])
            self.decay = float(state.get("decay", self.decay))
        else:
            self.ema_model.load_state_dict(state)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_loss_history(history: list[Dict[str, float]], output_dir: Path) -> None:
    csv_path = output_dir / "loss_history.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "epoch", "loss", "lr"])
        writer.writeheader()
        writer.writerows(history)

    # Plotting happens only when the training script is actually run; this
    # function is intentionally not invoked during repository preparation.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not history:
        return
    plt.figure(figsize=(8, 4))
    plt.plot([row["step"] for row in history], [row["loss"] for row in history])
    plt.xlabel("Training step")
    plt.ylabel("Noise prediction MSE")
    plt.title("DDPM training loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    ema: Optional[EMA],
    scaler: GradScaler,
    cfg: Dict[str, Any],
    epoch: int,
    global_step: int,
) -> None:
    state: Dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
    }
    if lr_scheduler is not None:
        state["lr_scheduler"] = lr_scheduler.state_dict()
    if ema is not None:
        state["ema"] = ema.state_dict()
    if scaler.is_enabled():
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)


def train(cfg: Dict[str, Any], resume: Optional[str] = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)
    (output_dir / "ckpt").mkdir(exist_ok=True)

    _set_seed(int(cfg.get("seed", 42)))
    with (output_dir / "config.yaml").open("w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    loader = get_dataloader(
        name=cfg["dataset"]["name"],
        batch_size=cfg["dataset"]["batch_size"],
        root=cfg["dataset"].get("root", "./data"),
        image_size=cfg["dataset"].get("image_size"),
        num_workers=cfg["dataset"].get("num_workers", 4),
    )
    model = UNet(**cfg["model"]).to(device)
    schedule = DDPMSchedule(**cfg["diffusion"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)),
        betas=(0.9, 0.999),
    )

    warmup_steps = int(cfg["optimizer"].get("warmup_steps", 0))
    lr_scheduler = None
    if warmup_steps > 0:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min((step + 1) / warmup_steps, 1.0)
        )

    ema_decay = float(cfg.get("ema_decay", 0.0))
    ema = EMA(model, ema_decay) if ema_decay > 0 else None
    precision = cfg.get("mixed_precision", "no")
    use_amp = precision in {"fp16", "bf16"} and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    start_epoch = 0
    global_step = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lr_scheduler is not None and "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if ema is not None and "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        if scaler.is_enabled() and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))

    wandb_enabled = bool(cfg.get("wandb", {}).get("enabled", False))
    wandb = None
    if wandb_enabled:
        import wandb as wandb_module

        wandb = wandb_module
        wandb.init(
            project=cfg["wandb"].get("project", "ddpm-course"),
            name=cfg["wandb"].get("run_name"),
            config=cfg,
        )

    training_cfg = cfg["training"]
    log_every = int(training_cfg.get("log_every", 50))
    sample_every = int(training_cfg.get("sample_every", 1000))
    ckpt_every = int(training_cfg.get("ckpt_every", 5000))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    history: list[Dict[str, float]] = []
    start_time = time.time()

    model.train()
    for epoch in range(start_epoch, int(training_cfg["num_epochs"])):
        for x0 in loader:
            x0 = x0.to(device, non_blocking=True)
            t = torch.randint(0, schedule.T, (x0.shape[0],), device=device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp, dtype=amp_dtype):
                loss = p_losses(model, x0, t, schedule)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if ema is not None:
                ema.update(model)

            global_step += 1
            lr_now = float(optimizer.param_groups[0]["lr"])
            row = {
                "step": float(global_step),
                "epoch": float(epoch),
                "loss": float(loss.detach().cpu()),
                "lr": lr_now,
            }
            history.append(row)
            if global_step % log_every == 0:
                print(
                    f"[epoch {epoch:03d} step {global_step:06d}] "
                    f"loss={row['loss']:.5f} lr={lr_now:.3e}"
                )
                if wandb is not None:
                    wandb.log({"loss": row["loss"], "lr": lr_now}, step=global_step)

            if sample_every > 0 and global_step % sample_every == 0:
                sample_model = ema.ema_model if ema is not None else model
                was_training = sample_model.training
                sample_model.eval()
                samples = p_sample_loop(
                    sample_model,
                    (
                        16,
                        cfg["model"]["in_channels"],
                        cfg["model"]["image_size"],
                        cfg["model"]["image_size"],
                    ),
                    schedule,
                    device=device,
                )
                vutils.save_image(
                    denormalize(samples),
                    output_dir / "samples" / f"step_{global_step:06d}.png",
                    nrow=4,
                )
                if was_training:
                    sample_model.train()

            if ckpt_every > 0 and global_step % ckpt_every == 0:
                _save_checkpoint(
                    output_dir / "ckpt" / f"step_{global_step:06d}.pt",
                    model,
                    optimizer,
                    lr_scheduler,
                    ema,
                    scaler,
                    cfg,
                    epoch,
                    global_step,
                )

    _save_checkpoint(
        output_dir / "ckpt" / "final.pt",
        model,
        optimizer,
        lr_scheduler,
        ema,
        scaler,
        cfg,
        int(training_cfg["num_epochs"]) - 1,
        global_step,
    )
    _save_loss_history(history, output_dir)
    if wandb is not None:
        wandb.finish()
    print(f"Training complete in {(time.time() - start_time) / 60.0:.1f} minutes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a from-scratch DDPM")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override training.num_epochs without editing the YAML config.",
    )
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.num_epochs is not None:
        if args.num_epochs <= 0:
            parser.error("--num_epochs must be positive")
        cfg["training"]["num_epochs"] = args.num_epochs
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
