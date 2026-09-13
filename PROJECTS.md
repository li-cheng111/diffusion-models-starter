# 仓库项目说明

本仓库将项目 1 保留在根目录，以兼容已有提交历史和 Git LFS 路径。
项目 2 独立放在自己的子目录中。

| 项目 | 位置 | 上游来源 |
|---|---|---|
| 项目 1：从零实现 DDPM | 仓库根目录 | 现有仓库历史 |
| 项目 2：采样器对比 | [`project2-samplers/`](project2-samplers/) | [Qi-StarterTrain/diffusion-project2-samplers](https://github.com/Qi-StarterTrain/diffusion-project2-samplers)，从 `main` 的 `774d640f3915c3396e034068beff318fbf421719` 提交导入 |

## 仓库规则

- `project2-samplers/` 是普通的受 Git 跟踪目录，不是嵌套 Git 仓库。
- 数据集、checkpoint、Inception 权重和缓存不提交。
- 项目 2 命令从 `project2-samplers/` 目录运行；兼容层从父目录导入
  项目 1 模块。
- 实验 JSON、图表、样本网格、日志和报告统一放在 `project2-samplers/`
  下，避免与项目 1 产物冲突。
