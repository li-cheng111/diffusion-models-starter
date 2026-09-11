# Project 1：从零实现 DDPM

本仓库提交 Project 1 基础档的完整源码、配置、静态测试和实验文档模板。实现目标是用 PyTorch 手写一个不依赖 `diffusers` 或 `lucidrains` 的 unconditional DDPM。

## 当前提交状态

本次提交只包含程序和文档内容，尚未执行 Python、测试、数据下载、训练或采样。因此仓库暂不包含 loss 曲线、生成样本、FID 和 checkpoint 等运行产物。

## 基础档完成清单

- [x] linear beta schedule 和 DDPM 系数预计算
- [x] 闭合形式前向加噪 `q_sample`
- [x] simplified noise-prediction loss `p_losses`
- [x] 单步反向采样 `p_sample`
- [x] 完整反向采样循环 `p_sample_loop`
- [x] sinusoidal timestep embedding
- [x] ResBlock 的时间 embedding 广播注入
- [x] MNIST 50 epoch 配置和训练入口
- [x] 64 张样本生成与最终产物的后续命令
- [ ] 实际运行 MNIST 训练并补充 loss 曲线、样本网格和 checkpoint

## 目录

```text
schedule.py       # beta schedule 与 DDPM 系数
diffusion.py      # q_sample、训练损失和反向采样
dataset.py        # MNIST / CIFAR-10 加载与 [-1, 1] 归一化
train.py          # 配置驱动训练、EMA、AMP、checkpoint
sample.py         # checkpoint 采样
evaluate.py       # FID 评估
model/            # sinusoidal embedding、ResBlock、U-Net
configs/          # MNIST 和 CIFAR-10 配置
tests/            # 尚未执行的单元测试源码
report.md         # 理论和实现说明，实验结果待补充
debug_log.md      # 实际运行后填写的调试记录模板
logs/             # 实验日志模板
```

## 核心公式

前向过程使用闭合形式：

```text
x_t = sqrt(alpha_bar_t) * x_0
    + sqrt(1 - alpha_bar_t) * epsilon
```

模型预测加入的噪声，训练目标为噪声 MSE。采样时从标准高斯噪声开始，依次执行 `T-1` 到 `0` 的反向步骤；最后一步不加入随机扰动。

## 后续运行方式

安装依赖后，可在未来运行：

```bash
python train.py --config configs/mnist.yaml
python sample.py --ckpt runs/exp_mnist_baseline/ckpt/final.pt --num_samples 64 --save_grid
```

CIFAR-10 的长训练使用：

```bash
python train.py --config configs/cifar10.yaml
python evaluate.py --ckpt runs/exp_cifar10_baseline/ckpt/final.pt --num_samples 5000
```

运行结果应在真实实验完成后再提交，并同步更新 `report.md`、`debug_log.md` 和 `logs/`。
