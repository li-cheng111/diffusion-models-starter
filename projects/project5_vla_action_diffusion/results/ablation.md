# Project 5 ablation（AutoDL 实测）

所有闭环指标均为 100 个 episode，`seed=10000..10099`，`exec_steps=4`；DDPM 使用 20 个采样步，FM 使用 10 个 Euler 步。除特别注明外，训练 seed 为 42、10,000 steps、1,000 条 expert demos。

| 实验 | H | 条件/方法 | 成功率 | 碰撞率 | 超时率 | 平均步数 | 末端距离 |
|---|---:|---|---:|---:|---:|---:|---:|
| Pure BC | 16 | image + agent state / MSE | 58% | 12% | 30% | 50.73 | 0.3570 |
| DDPM baseline | 16 | image + agent state / global pooling | 54% | 6% | 40% | 68.93 | 0.4114 |
| DDPM + spatial vision（最终） | 16 | image + agent state / 4×4 spatial pooling | **71%** | 3% | 26% | **48.70** | **0.1643** |
| Flow Matching | 16 | image + agent state / 10-step Euler | 57% | 9% | 34% | 52.85 | 0.4804 |
| DDPM chunk ablation | 32 | image + agent state | 14% | 1% | 85% | 95.03 | 0.6302 |
| DDPM state-only | 16 | agent + target state | 81% | 12% | 7% | 55.89 | 0.1075 |

`eval_vision.json` 是最终空间视觉模型的结果；`eval_vision_ddpm_baseline.json` 与 `eval_vision_exec1.json` 保留了失败/对照实验。`results/loss_curve.png` 展示了四个训练曲线。

## action chunk padding

按训练 demo 的 1,000 条成功 expert 轨迹（长度中位数 26，范围 10–72）计算“重复末动作”槽位占比：

| H | 8 | 16（最终） | 32 | 64 |
|---:|---:|---:|---:|---:|
| padding ratio | 12.9% | 27.5% | 51.7% | 74.9% |

H=32 的闭环退化与超过一半的重复尾部监督一致；因此最终选择 H=16，而不是盲目延长 chunk。

## 结论

全局平均池化会抹掉目标/障碍的二维位置。保留 4×4 特征网格后，视觉 DDPM 从 54% 提升到 71%，达到任务的 70% 目标。状态-only 的 81% 不能替代视觉结果，因为它直接得到 target 坐标；它主要作为可学习性上限和条件信息消融。

## 加分项：多模态、预训练视觉与动态障碍

以下结果均为 AutoDL RTX 4090 上独立的 100 集闭环评估，`exec_steps=4`、DDPM 20 个采样步；结果 JSON 与训练日志随仓库提交。

| 加分实验 | 训练/评估设置 | 成功率 | 碰撞率 | 超时率 | 备注 |
|---|---|---:|---:|---:|---|
| 双目标多模态 | H=16，小型 spatial CNN；目标 `(-0.65,0.55)` / `(0.65,0.55)`；seed 20000–20099 | **97%** | 3% | 0% | mode 0: 96.2%（50/52），mode 1: 97.9%（47/48） |
| ImageNet ResNet18 | 预训练 ResNet18，5,000 steps；seed 30000–30099 | 3% | 8% | 89% | 11.48M 参数；toy 数据/预算下明显不如 0.59M spatial CNN |
| ImageNet ResNet18（长训） | 同上，15,000 steps；seed 30000–30099 | 2% | 18% | 80% | 延长训练未改善，保留作为真实负结果 |
| 移动障碍泛化 | 静态障碍训练的最终 checkpoint；障碍速度 0.025；seed 40000–40099 | **72%** | 20% | 8% | 只改变测试环境，未重新训练 |

多模态实验不仅汇报总体成功率，还按目标模式统计，确认模型对左右两个 target 都能工作。ResNet18 对照使用同一观测和评估协议；负结果说明在 64×64 toy 图像与小数据上，直接微调大 backbone 需要更合适的学习率、冻结策略或更长数据规模，不能仅凭“预训练”假设一定更好。动态障碍结果显示静态训练策略具备一定迁移能力，但碰撞率上升是清晰的泛化代价。

汇总图：[results/bonus_summary.png](bonus_summary.png)；复现实验入口：[run_bonus_experiments.py](../run_bonus_experiments.py)。
