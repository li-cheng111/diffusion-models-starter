# 五个 Project 的仓库规划

本仓库是课程五个 Project 的统一 monorepo。上游课程材料仍是只读教材库；本仓库
只保存个人实现、实验产物和报告。课程总览见
[diffusion-models-starter-materials](https://github.com/Qi-StarterTrain/diffusion-models-starter-materials)。

## 项目映射

| 项目 | 本地目录 | 上游模板 | 主要交付物 |
|---|---|---|---|
| 项目 1 | `projects/project1_ddpm/` | [diffusion-project1-ddpm](https://github.com/Qi-StarterTrain/diffusion-project1-ddpm) | DDPM 训练、MNIST/CIFAR-10、EMA、FID、挑战实验 |
| 项目 2 | `projects/project2_samplers/` | [diffusion-project2-samplers](https://github.com/Qi-StarterTrain/diffusion-project2-samplers) | DDIM、Euler、DPM-Solver-2、FID-NFE、反演 |
| 项目 3 | `projects/project3_stable_diffusion/` | [diffusion-project3-stable-diffusion](https://github.com/Qi-StarterTrain/diffusion-project3-stable-diffusion) | SD 推理解剖、参数扫描、VAE、LoRA |
| 项目 4 | `projects/project4_flow_matching/` | [diffusion-project4-flow-matching](https://github.com/Qi-StarterTrain/diffusion-project4-flow-matching) | Rectified Flow、Euler/Heun、CFG、NFE-FID |
| 项目 5 | `projects/project5_vla_action_diffusion/` | [diffusion-project5-vla-action-diffusion](https://github.com/Qi-StarterTrain/diffusion-project5-vla-action-diffusion) | Action Diffusion、视觉编码、闭环成功率 |

## 依赖关系

```text
项目 1（DDPM） ───────► 项目 2（采样器）
       │                       │
       └──── DDPM/FID 基线 ────┴──► 项目 4（FM 对比）

项目 3（Stable Diffusion）独立使用 Hugging Face 预训练模型
项目 4（Flow Matching） ───────► 项目 5 的 FM action-head 加分项（可选）
```

项目 2 可以直接导入项目 1 的 `model/`、`schedule.py` 和 checkpoint 格式；项目 4
只读取项目 1/2 的标准化结果，不直接依赖其网络实现；项目 5 的基础档不依赖项目 4。

## 共用代码边界

允许放入 `shared/` 的内容：路径解析、seed、环境记录、SHA256、JSON 结果格式、
FID 输入约定、通用图片网格和曲线绘图。

不放入 `shared/` 的内容：具体 U-Net、DiT、Stable Diffusion pipeline、VLA policy、
项目专属 scheduler 和训练循环。这样每个 Project 的 TODO、实验假设和报告仍然可独立
阅读和评分。

## 目录和 Git 规则

- 仓库中只允许根目录存在一个 `.git`。
- 五个 Project 都使用下划线目录名，便于 `python -m` 和包导入。
- 训练原始目录放在各项目的 `runs/` 或根目录 `.local/temporary_runs/`，不提交。
- 数据集、checkpoint、Inception 权重和缓存不提交；报告只记录路径和 SHA256。
- 不使用 submodule；上游快照和提交哈希记录在 [`UPSTREAMS.md`](UPSTREAMS.md)。
- 提交时使用明确的路径，不使用根目录 `git add .`。
