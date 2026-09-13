# 项目 1 实验日志

> AutoDL 上真实 MNIST 基线运行的完成记录。

## 实验编号

`exp_20260912_01_mnist_autodl`

## 实验目标

验证从零实现的无条件 DDPM 基线能否在 GPU 实例上完成训练并生成可用的 MNIST 样本。

## 环境

| 项目 | 值 |
|---|---|
| 日期 | 2026-09-12 |
| 硬件 | NVIDIA GeForce RTX 5090 |
| Python / PyTorch | Python 3.12.3 / PyTorch 2.8.0+cu128 |
| CUDA | 12.8 |
| Git 提交 | `99955cac0d37da993a7ede479086baa92fc8be65` |
| 分支 | `main` |

## 配置

- 数据集：MNIST
- 图像尺寸：32
- Batch size：128
- 训练轮数：50
- 扩散步数 `T`：1000
- Beta 调度策略：linear，0.0001 → 0.02
- 学习率：2.0e-4
- EMA 衰减：0.9999
- 混合精度：否
- 随机种子：42

## 运行命令

```text
python train.py --config configs/mnist.yaml
```

## 结果

 - 最后记录的 loss：第 23,400 步为 0.01151
 - 训练时间：24.8 分钟
 - FID：使用 EMA 权重和 5,000 个样本时为 32.8913
 - 样本路径：`runs/exp_mnist_baseline/samples_inference_ema/grid.png`

## 结论

MNIST 基线已成功完成。最终 checkpoint、loss 历史、loss 曲线、生成样本网格和
FID 记录均包含在实验产物中。

## 实际遇到的问题

GitHub 克隆在 AutoDL 上最初返回 HTTP 503；改用浅克隆重试后成功。训练代码没有
出现运行时问题。具体操作记录见 `debug_log.md`。
