# CIFAR-10 进阶档实验日志

> 本模板只记录计划和待填字段。实际训练完成后再填写结果，不预先虚构 FID、loss 或样本质量。

## 实验信息

- Experiment ID: `exp_YYYYMMDD_cifar10_advanced`
- Date:
- Host / GPU:
- Git commit:
- Config: `configs/cifar10.yaml`
- Seed: `42`

## 环境

- Python:
- PyTorch:
- CUDA:
- Dataset root:

## 配置

- Dataset: CIFAR-10 train split
- Resolution: 32x32
- Batch size: 128
- Epochs: 200
- Expected training steps: 78,000
- Optimizer: AdamW
- Learning rate: 2e-4
- Warmup steps: 5,000
- Diffusion steps T: 1,000
- Schedule: linear
- EMA decay: 0.9999
- Mixed precision: fp16
- Augmentation: random horizontal flip during training only

## 结果（待实际运行后填写）

- Training time:
- Final loss:
- EMA FID @ 5,000 generated / 5,000 train images:
- Raw FID @ 5,000 generated / 5,000 train images:
- Raw - EMA:
- Sample grid path:
- Checkpoint path:

## 观察与问题

- EMA 与 raw 样本的视觉差异：待填写
- 训练稳定性 / NaN：待填写
- 其他实际问题及解决方式：待填写
