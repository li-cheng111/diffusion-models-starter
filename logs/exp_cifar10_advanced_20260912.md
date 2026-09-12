# CIFAR-10 进阶档实验日志

## 实验信息

- Experiment ID: `exp_20260912_cifar10_advanced`
- Date: `2026-09-12`
- Host / GPU: AutoDL / NVIDIA GeForce RTX 5090
- Source version: GitHub `3ebafc3`（AutoDL working tree synced to this version）
- Config: `configs/cifar10.yaml`
- Seed: `42`

## 环境

- Python: `3.12.3`
- PyTorch: `2.8.0+cu128`
- CUDA: `12.8`
- Dataset root: `./data` on AutoDL data disk

## 配置

- Dataset: CIFAR-10 train split
- Resolution: 32x32
- Batch size: 128
- Epochs: 200
- Training steps: 78,000
- Optimizer: AdamW
- Learning rate: 2e-4
- Warmup steps: 5,000
- Diffusion steps T: 1,000
- Schedule: linear
- EMA decay: 0.9999
- Mixed precision: fp16
- Augmentation: random horizontal flip during training only
- Training time: 98.2 minutes

## 结果

- Final logged loss: `0.01938` at step `78,000`
- EMA FID: `19.2879` using 5,000 generated images vs 5,000 train images
- Raw FID: `28.7464` using 5,000 generated images vs 5,000 train images
- Raw - EMA: `+9.4586`
- Target FID ≤ 15: not reached in this run
- EMA grid: `runs/exp_cifar10_advanced/samples_inference_ema/grid.png`
- Raw grid: `runs/exp_cifar10_advanced/samples_inference_raw/grid.png`
- Checkpoint: `runs/exp_cifar10_advanced/ckpt/final.pt`

## 观察

EMA 权重的 FID 比 raw 权重低 9.4586，说明 EMA 对本次训练的采样质量有明显帮助。该结论仅适用于当前配置、seed 和 5,000 样本评估；由于 EMA FID 仍为 19.2879，后续需要单独调参或重训才能继续冲击 FID ≤ 15。
