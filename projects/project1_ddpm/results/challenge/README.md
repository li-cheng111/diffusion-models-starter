# 项目 1 挑战档结果

本目录记录已完成的 CIFAR-10 挑战档测量结果；本次矩阵已完成，不是待填模板。

- 调度策略：`linear`、`cosine`
- 随机种子：`42`、`43`、`44`
- 训练预算：每次运行 `200` 轮
- FID 样本数：`5,000`
- 真实数据划分：CIFAR-10 训练集，不使用随机增强

`summary.md` 包含每个随机种子的数值以及均值 ± 标准差。每次运行目录中的小型
`grid.png` 展示定性样本。大型训练 checkpoint、中间样图以及 MNIST 临时产物通过
[`challenge-v1 Release`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1)
保存，不存储在本 Git 仓库中。

挑战档 EMA FID 均值为：linear `19.2926 ± 0.3357`，cosine `137.5132 ± 8.0166`。
FID 的真实数据为 5,000 张不使用随机增强的 CIFAR-10 训练图，每次生成 5,000 张图。
