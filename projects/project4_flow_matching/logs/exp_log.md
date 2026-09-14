# Project 4 Flow Matching 实验日志

## 环境

- 日期：2026-09-14
- AutoDL：NVIDIA GeForce RTX 4090 24GB
- Python 3.12.3；PyTorch 2.8.0+cu128；TorchVision 0.23.0+cu128；TorchMetrics 1.9.0
- seed=42；precision=BF16 autocast；DiT-S 32.62M 参数
- 正式配置：batch=128，steps=200000，EMA decay=0.9999，conditional dropout=0.1

## 实现与验证

- `check_setup.py`：DiT/UNet 前向、TODO 16 loss、Euler、CFG null token index=10 全部通过。
- debug UNet 500 steps：loss 约 0.87 降至 0.30，checkpoint 保存与恢复通过。
- BF16 batch 128 预检：峰值显存约 7.47 GiB，约 9.15 step/s。
- 正式训练：最终 step=200000，loss=0.1676，稳定约 8.81 step/s，耗时约 6h18m。
- 采样：Euler NFE=4/8/16/32/50、CFG=0/1/3/7.5 采样图生成成功；Heun CPU 冒烟测试成功。
- FID：FM NFE 扫描与 CFG 扫描均以 5000 samples 完成，结果保存到 `results/fm_nfe.json` 和 `results/fm_cfg.json`。
- 对比图：`results/nfe_fid_curve.png` 已生成，使用 Project 2 `benchmark_ddim.json` 基线。

## 结果摘要

FM Euler FID：NFE 4/8/16/32/50 = 47.377/28.305/22.121/19.311/18.299。

CFG NFE=20：scale 1/2/3/5/7.5 = 20.988/18.349/23.693/35.316/43.962，最佳 scale=2。
