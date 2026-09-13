# 数据、权重与实验产物策略

`.local/` 用于本地或 AutoDL 的数据集、模型权重、Hugging Face 缓存和临时运行目录，
默认被 Git 忽略。项目目录中的 `results/`、`samples/` 和 `logs/` 只保存可复核的小型
产物。

已有历史 checkpoint 不在本次目录整理中删除或改写历史；迁移后新增的大型权重统一放入
`.local/checkpoints/`，报告通过 SHA256 和相对路径引用。课程明确要求提交的小型 LoRA
或 Project 5 模型需要先检查大小，再通过普通 Git 或 Git LFS 单独加入。
