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
