# 项目 2 实验日志

## 实现验证

目标：验证父目录项目 1 的导入、有限的采样输出和正确的 NFE。

- 环境：Windows，Python 3.14.0，PyTorch 2.13.0+cu126，NVIDIA RTX 4060
  Laptop GPU。
- 命令：`python check_compat.py --ckpt ../runs/exp_cifar10_advanced/ckpt/final.pt`、
  `python -m unittest discover -s tests -v`，以及两个 sampler 模块自测。
- 结果：11 个标准库单元测试通过；兼容性检查成功加载嵌套 EMA 的项目 1
  checkpoint；DDIM 和 DPM-Solver 模块冒烟测试通过。由于本地没有安装
  `torchmetrics`，正式 FID 改在 AutoDL 上运行。
- 定性冒烟：生成并检查了 MNIST DDIM 50 步、CIFAR-10 DDIM 10 步和
  DPM-Solver 5 步/9 NFE 样本网格，输出有限且具有结构。这些不是最终 FID。

## FID-NFE 正式实验

假设：在相近 NFE 下，DPM-Solver-2 的 FID-计算量折中应优于一阶 DDIM。

- 控制条件：seed44 linear EMA checkpoint、生成 seed42，每个配置严格使用
  5,000 张无增强真实图和 5,000 张生成图。
- 命令：见 `README.md` 中的“AutoDL 正式实验”。
- AutoDL：NVIDIA RTX 4090，Python 3.12.3，PyTorch 2.8.0+cu128，CUDA 12.8。
- Git 提交：`e8f8af6698570485b9909ca1755f0266a09e4da8`。
- checkpoint SHA256：`937853559a1377660f7d4cbd1dd3f7c6cf022aed4ab84e9daa918aa6e34541412`。
- 完整矩阵已完成：每个配置使用固定的 5,000 张无增强真实图和 5,000 张
  生成图。绝对 FID 最佳为 DDPM 的 19.290（1,000 NFE）；低 NFE 最佳为
  DPM-Solver-2 的 21.074（19 NFE）。完整数值见 `runs/benchmark_all.json`
  和 Pareto 图。

## DDIM 反演实验

假设：中高步数可以降低离散化误差，但模型预测误差可能使趋势不单调。

- 控制图像：CIFAR-10 测试集索引 0–63。
- 步数：10、20、50、100、250。
- 结果见 `runs/inversion_results.json` 和 `runs/inversion_errors.png`。
  平均 L2 从 10 步的 18.2820 单调下降到 250 步的 0.9316，PSNR 从
  15.891 dB 上升到 41.836 dB。

## 运行后检查

- `runs/benchmark_all.json`：15 个配置，每个配置有 5,000 张生成图和对应的
  5,000 张真实图。
- `runs/benchmark_ddim.json`：包含全部五个要求步数的 DDIM 专项结果。
- `runs/trajectory_comparison.json`：`same_initial_noise=true`；DPM-Solver-2
  在 50 个外层步时报告 99 NFE。
- AutoDL 进程和 detached 监控会话均正常结束；结果下载后可以停止只读仪表盘。
