"""Project 4: Flow Matching Training

含 TODO 16: 实现 Rectified Flow loss
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg):
    """根据 config 构建 model（UNet 或 DiT）。

    约定：两个 backbone 的 `num_classes` 都传**真实类别数**（CIFAR-10 是 10），
    null token 由模型内部多分配一个 embedding，索引固定为 num_classes。
    """
    m = cfg['model']
    if m['type'] == 'unet':
        from model.unet import ConditionalUNet
        return ConditionalUNet(
            in_channels=3,
            out_channels=3,
            base_channels=m['base_channels'],
            num_classes=m['num_classes'],
        )
    elif m['type'] == 'dit':
        from model.dit import SimpleDiT
        return SimpleDiT(
            img_size=32,
            patch_size=m.get('patch_size', 2),
            in_ch=3,
            embed_dim=m['embed_dim'],
            depth=m['depth'],
            num_heads=m['num_heads'],
            num_classes=m['num_classes'],
        )
    else:
        raise ValueError(f"未知的 model.type: {m['type']}（只支持 'unet' / 'dit'）")


def compute_fm_loss(model, x_1, y, cond_drop_prob, num_classes):
    """
    ============================================================
    TODO 16: 实现 Rectified Flow / Conditional Flow Matching loss
    ============================================================

    输入:
        model: 接受 (x, t, y) 输出 velocity 预测 v_pred (B, 3, 32, 32)
        x_1: 真实图像 (B, 3, 32, 32)
        y: 类别 label (B,) ∈ {0, ..., num_classes-1}
        cond_drop_prob: 训练时 y 替换为 null 的概率（CFG）
        num_classes: 总类别数（最后一个 index 用作 null）

    要求:
        1. 采样 t ~ U[0, 1]，shape = (B,)
        2. 采样 epsilon ~ N(0, I)，shape = x_1.shape
        3. 构造 x_t = (1 - t) * epsilon + t * x_1
        4. Target = x_1 - epsilon
        5. 做 conditional dropout: 以 cond_drop_prob 概率把 y 替换为 num_classes（null token）
        6. 调用 model(x_t, t, y) → v_pred
        7. 返回 MSE(v_pred, target)

    Hints:
        - t.view(-1, 1, 1, 1) 让 t 能与 (B, C, H, W) broadcast
        - y_null = torch.full_like(y, num_classes)
        - drop_mask = torch.rand(B) < cond_drop_prob
        - y_in = torch.where(drop_mask, y_null, y)
    ============================================================
    """
    B = x_1.shape[0]
    device = x_1.device

    # === Your code here ===
    raise NotImplementedError("TODO 16: implement Rectified Flow loss")
    # === End ===


def train(cfg, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    # Data
    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # to [-1, 1]
    ])
    ds = datasets.CIFAR10(cfg['data']['root'], train=True, download=True, transform=tf)
    loader = DataLoader(
        ds,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        drop_last=True,
        pin_memory=True,
    )

    # Model
    model = build_model(cfg).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # EMA
    ema = {k: v.clone().detach() for k, v in model.state_dict().items()}
    ema_decay = cfg['train'].get('ema_decay', 0.9999)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay'],
    )

    num_classes = cfg['model']['num_classes']  # 10 for CIFAR-10
    cond_drop_prob = cfg['train'].get('cond_drop_prob', 0.1)

    step = 0
    losses = []
    t0 = time.time()

    while step < cfg['train']['max_steps']:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            loss = compute_fm_loss(model, x, y, cond_drop_prob, num_classes)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train'].get('grad_clip', 1.0))
            opt.step()

            # EMA update
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(ema_decay).add_(v.detach(), alpha=1 - ema_decay)

            losses.append(loss.item())
            step += 1

            if step % cfg['train'].get('log_every', 100) == 0:
                mean = sum(losses[-100:]) / min(len(losses), 100)
                elapsed = time.time() - t0
                print(f"step {step:6d} | loss {mean:.4f} | {step/elapsed:.1f} step/s")

            if step % cfg['train']['save_every'] == 0:
                ckpt = {
                    'step': step,
                    'model': model.state_dict(),
                    'ema': ema,
                    'opt': opt.state_dict(),
                    'cfg': cfg,
                }
                torch.save(ckpt, os.path.join(output_dir, f'step_{step}.pt'))
                torch.save(ckpt, os.path.join(output_dir, 'latest.pt'))

            if step >= cfg['train']['max_steps']:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, args.output)
