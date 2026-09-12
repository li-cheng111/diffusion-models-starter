# Project 1：从零实现 DDPM

## 当前状态

本仓库已完成 Project 1 基础档要求对应的源码、MNIST 配置、静态测试和实验文档。MNIST 基线已在 AutoDL 的 NVIDIA GeForce RTX 5090 上完成真实训练、采样和 EMA FID 评估。本报告只记录实际得到的结果，不对单个 FID 数值作超出实验范围的结论。

## 方法概述

DDPM 使用固定的前向过程逐步向图像加入高斯噪声：

\[
x_t=\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon.
\]

训练时随机采样时间步 `t`，使用闭合形式直接得到 `x_t`，再让 U-Net 预测加入的噪声。优化目标是：

\[
\mathcal L_{\text{simple}}
=\|\epsilon-\epsilon_\theta(x_t,t)\|^2.
\]

反向采样从标准高斯噪声开始，按 `T-1` 到 `0` 的顺序执行采样步骤。最后一步不再添加随机噪声。

## 代码结构

- `schedule.py` 计算 beta、alpha、累积 alpha 和 posterior variance。
- `diffusion.py` 实现前向加噪、训练损失和完整反向采样。
- `model/embedding.py` 实现 sinusoidal timestep embedding。
- `model/unet.py` 实现带时间条件的残差 U-Net。
- `dataset.py` 支持 MNIST 和 CIFAR-10，并将输入归一化到 `[-1,1]`。
- `train.py` 支持 AdamW、warmup、EMA、AMP、梯度裁剪、checkpoint 和 loss history。

## 自查问题

1. `q_sample` 中的线性组合对应 `q(x_t|x_0)` 的闭合形式。
2. `(B,)` 的时间系数要 reshape 为 `(B,1,1,1)`，这样才能对每个 batch 样本的所有通道和像素广播；否则维度可能无法匹配或产生错误广播。
3. schedule 使用 `register_buffer`，因为这些张量不是可学习参数，但必须随模型一起移动到 CPU 或 GPU，并进入 checkpoint 状态。
4. 随机采样 `t` 是对所有时间步期望的 Monte Carlo 估计；每个 batch 都能覆盖不同噪声强度。
5. `t=0` 时 posterior variance 为零，继续添加噪声会破坏最终的干净样本。

## 待完成实验

- MNIST 50 epoch：已完成训练和推理；结果见下方“MNIST 实验结果”。
- CIFAR-10 200 epoch：待实际运行后补充 EMA/非 EMA 采样对比和 FID。

## MNIST 实验结果

| 项目 | 实际值 |
|---|---|
| 硬件 | NVIDIA GeForce RTX 5090 |
| 训练配置 | MNIST，50 epochs，23,400 steps |
| 训练时间 | 24.8 minutes |
| 最后一次日志 loss | 0.01151 |
| FID | 32.8913（EMA，5,000 samples） |

结果文件：

- `runs/exp_mnist_baseline/loss_history.csv`
- `runs/exp_mnist_baseline/loss_curve.png`
- `runs/exp_mnist_baseline/samples_inference_ema/grid.png`
- `runs/exp_mnist_baseline/ckpt/final.pt`
- `runs/exp_mnist_baseline/ckpt/fid_5000_EMA.txt`

## 基础档验收状态

基础档的 8 个代码实现点已经写入仓库；MNIST 50 epoch 已完成真实训练、推理和 EMA FID 评估。CIFAR-10 训练及 EMA/非 EMA 对比仍待后续实验。
