# Project 1：从零实现 DDPM

本仓库提交 Project 1 基础档和进阶档的源码、配置、静态测试与实验文档模板。实现目标是用 PyTorch 手写一个不依赖 `diffusers` 或 `lucidrains` 的 unconditional DDPM。

## 当前提交状态

基础档源码已经提交；MNIST 实验已在 AutoDL RTX 5090 上完成训练、采样和 EMA FID 评估。进阶档 CIFAR-10 200 epoch 训练和 EMA/raw 对比也已完成，真实结果已整理到 `runs/exp_cifar10_advanced/`。本次 EMA FID 为 19.2879，尚未达到作业目标 15。

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
- [x] 实际运行 MNIST 训练并补充最终 loss 曲线、样本网格和 checkpoint

## 进阶档实现状态

- [x] CIFAR-10 200 epoch 配置（`configs/cifar10.yaml`）
- [x] 训练过程维护 raw 与 EMA 两套权重
- [x] FID 默认使用 5,000 张无增强训练图作为 real split
- [x] `evaluate.py --compare_ema` 一键生成 EMA/raw 对比记录
- [x] CIFAR-10 实际训练、EMA/raw 样本和 FID 对比报告
- [ ] FID ≤ 15（本次 EMA FID 为 19.2879，仍需调参或重训）

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
monitor.py        # 本地只读实时训练进度监控
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

安装依赖后，可运行：

```bash
python train.py --config configs/mnist.yaml
python sample.py --ckpt runs/exp_mnist_baseline/ckpt/final.pt --num_samples 64 --save_grid
```

训练进行中时，可以启动本地只读监控页面。页面每 2 秒读取一次样本、checkpoint 和 loss 文件：

```bash
python monitor.py --run-dir runs/exp_mnist_baseline \
    --total-steps 23400 --train-pid <训练进程 PID> --port 8765
```

然后打开 <http://127.0.0.1:8765/>。步数会随着最新持久化产物更新；`loss_history.csv` 出现后会自动绘制 loss 曲线。

CIFAR-10 的长训练使用：

```bash
python train.py --config configs/cifar10.yaml
python sample.py --ckpt runs/exp_cifar10_advanced/ckpt/final.pt \
    --num_samples 64 --save_grid
python evaluate.py --ckpt runs/exp_cifar10_advanced/ckpt/final.pt \
    --num_samples 5000 --batch_size 64 --real_split train --compare_ema
```

运行结果应在真实实验完成后再提交，并同步更新 `report.md`、`debug_log.md` 和 `logs/`。
