# Project 4 调试记录

## 1. 误从仓库根目录运行自检

- 现象：在 monorepo 根目录运行 `python3 check_setup.py` 报找不到文件。
- 原因：Project4 的入口位于 `projects/project4_flow_matching/check_setup.py`。
- 修复：固定进入 Project4 目录后运行命令。

## 2. CIFAR-10 下载速度过慢

- 现象：TorchVision 默认镜像下载约 30–50KB/s，预计超过 1 小时。
- 原因：实例中已有完整 `cifar-10-batches-py` 缓存。
- 修复：将 Project4 的 `data` 指向已有数据目录，避免重复下载。

## 3. GitHub 完整克隆触发限速

- 现象：普通浅克隆在大型历史对象处长时间停滞。
- 修复：复用 AutoDL 上已有目标仓库对象，建立稀疏工作树。
- 结果：Project4 源码和 Project 2 基线可用。

## 4. BF16 训练预检

- 现象：需要确认 DiT-S 是否适合 24GB 显存。
- 结果：batch 128 BF16 峰值约 7.47 GiB，约 9.15 step/s。
- 决策：正式训练使用 batch 128、BF16，不需要梯度累积。

## 5. 后处理脚本相对路径错误

- 现象：FID 和采样完成后，NFE 曲线绘图找不到 DDPM baseline。
- 原因：脚本把 `projects/project2_samplers/...` 错拼成了 Project4 子目录下的路径。
- 修复：从 `projects/project4_flow_matching` 执行时使用 `../project2_samplers/runs/benchmark_ddim.json`，随后图表生成成功。
