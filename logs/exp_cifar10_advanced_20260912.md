# CIFAR-10 进阶档实验日志

## 实验信息

- 实验编号：`exp_20260912_cifar10_advanced`
- 日期：`2026-09-12`
- 主机 / GPU：AutoDL / NVIDIA GeForce RTX 5090
- 源码版本：GitHub `3ebafc3`（AutoDL 工作目录已同步到该版本）
- 配置：`configs/cifar10.yaml`
- 随机种子：`42`

## 环境

- Python：`3.12.3`
- PyTorch：`2.8.0+cu128`
- CUDA：`12.8`
- 数据集根目录：AutoDL 数据盘上的 `./data`

## 配置

- 数据集：CIFAR-10 训练集
- 分辨率：32x32
- Batch size：128
- 训练轮数：200
- 训练步数：78,000
- 优化器：AdamW
- 学习率：2e-4
- Warmup 步数：5,000
- 扩散步数 T：1,000
- 调度策略：linear
- EMA 衰减：0.9999
- 混合精度：fp16
- 数据增强：仅训练阶段随机水平翻转
- 训练时间：98.2 分钟

## 结果

- 最后记录的 loss：第 `78,000` 步为 `0.01938`
- EMA FID：使用 5,000 张生成图对比 5,000 张训练图时为 `19.2879`
- Raw FID：使用 5,000 张生成图对比 5,000 张训练图时为 `28.7464`
- Raw - EMA（原始权重减 EMA）：`+9.4586`
- 目标 FID ≤ 15：本次运行未达到
- EMA 网格：`runs/exp_cifar10_advanced/samples_inference_ema/grid.png`
- Raw 网格：`runs/exp_cifar10_advanced/samples_inference_raw/grid.png`
- Checkpoint：`runs/exp_cifar10_advanced/ckpt/final.pt`

## 观察

EMA 权重的 FID 比 raw 权重低 9.4586，说明 EMA 对本次训练的采样质量有明显帮助。该结论仅适用于当前配置、seed 和 5,000 样本评估；由于 EMA FID 仍为 19.2879，后续需要单独调参或重训才能继续冲击 FID ≤ 15。
