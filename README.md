# Diffusion Models 五项目总仓库

本仓库把课程的五个 Project 整理为一个 monorepo。每个项目在
`projects/` 下保持独立的代码、配置、测试、实验日志、结果和报告；只有稳定且
与具体模型无关的工具放在 `shared/`。

## 项目状态

| 项目 | 本地目录 | 当前状态 | 主要依赖 |
|---|---|---|---|
| 项目 1：DDPM | `projects/project1_ddpm/` | 已完成基础档、进阶档和挑战实验，目录迁移已完成，待安装 pytest 做完整回归 | 无 |
| 项目 2：采样器对比 | `projects/project2_samplers/` | 已完成 DDIM、Euler、DPM-Solver-2、反演和 FID 实验 | 项目 1 |
| 项目 3：Stable Diffusion 解剖 | `projects/project3_stable_diffusion/` | 已导入 starter，待实现 | `diffusers`、`transformers`、`peft` |
| 项目 4：Flow Matching | `projects/project4_flow_matching/` | 已导入 starter，待实现 | PyTorch、CIFAR-10、FID |
| 项目 5：VLA Action Diffusion | `projects/project5_vla_action_diffusion/` | 已导入 starter，待实现 | PyTorch、玩具 2D 环境 |

课程要求和上游固定版本见 [`PROJECTS.md`](PROJECTS.md) 与 [`UPSTREAMS.md`](UPSTREAMS.md)。

## 目录约定

- `projects/projectN_*/`：项目自己的源码和交付物。
- `shared/`：路径、随机种子、checkpoint 校验、实验元数据、评估和绘图工具。
- `results/`：可复核的 JSON/CSV/图表，提交到 Git。
- `samples/`：精选样本网格，提交到 Git。
- `logs/`：实验日志，提交到 Git。
- `runs/`：原始运行目录，原则上不提交。
- `.local/`：数据集、模型权重、Hugging Face 缓存和临时文件，整个目录不提交。

## 环境安装

先安装跨项目基础依赖，再按项目安装额外依赖：

```bash
pip install -r requirements/base.txt
pip install -r requirements/project1-2.txt  # 项目 1/2
```

项目 3、4、5 分别使用 `requirements/project3.txt`、
`requirements/project4.txt`、`requirements/project5.txt`。由于 CUDA 和
Hugging Face 依赖经常随机器变化，正式实验应在 AutoDL 上记录完整版本信息。

## 从仓库根目录运行

```bash
python -m pytest projects/project1_ddpm/tests -v
python -m unittest discover -s projects/project2_samplers/tests -v
python -m projects.project1_ddpm.challenge summarize \
  --output_root runs/challenge
python -m projects.project2_samplers.check_compat
python scripts/check_repo.py
```

项目 3–5 完成 starter 自检后，再分别按其 README 执行训练和评估。长时间训练
不放入 GitHub Actions；CI 只运行 CPU 冒烟测试、导入检查和仓库结构检查。

## 数据与权重

默认路径由 `shared.paths` 解析：

```text
.local/
├── datasets/
├── checkpoints/
├── hf_cache/
└── temporary_runs/
```

报告中必须记录数据划分、随机种子、软件环境和 checkpoint SHA256，但不提交大型
数据集或模型权重。Project 3 的 LoRA 权重和 Project 5 的小型最终模型只有在符合
课程提交要求且经过大小检查后才允许单独加入 Git/LFS。

## 五项目推进顺序

1. 完成项目 1/2 目录迁移和回归测试。
2. 完成项目 3 的 SD 推理、参数扫描、VAE 和 LoRA。
3. 完成项目 4 的 FM loss、Euler/Heun、CFG 和 NFE-FID 对比。
4. 完成项目 5 的 action diffusion、视觉条件和闭环评估。
5. 统一五个项目的报告、结果格式、实验日志和 GitHub 提交检查。
