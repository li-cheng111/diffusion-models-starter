# Project 5：VLA Action Diffusion 实验报告

## 摘要

本项目在 64×64 RGB 的 2D reaching 环境中实现了一个简化 VLA：视觉编码器读取 agent、target 和 distractors，条件 action head 一次生成一段连续二维动作，并在闭环中周期性重规划。完成了 TODO 19（action-chunk DDPM loss）、TODO 20（vision encoder）和 TODO 21（100 集闭环评估），并额外实现了 Flow Matching 与 Pure BC 对照。AutoDL RTX 4090 上最终模型使用 H=16、DDPM 20 步采样和每 4 步重规划，在固定测试集上取得 **71% 成功率、3% 碰撞率、26% 超时率**，达到题目要求的 70% 目标。

## 1. Design note

### 1.1 Action representation 与 chunk size

动作是环境坐标系中的 `(Δx, Δy) ∈ [-0.1, 0.1]^2`。模型不只预测下一步，而是预测 H 步 action chunk，评估时执行前 `exec_steps` 步后重新观察并采样。这样可以把短期运动的相关性作为一个整体学习，同时减少每个控制周期的采样开销。训练 demo 的轨迹中位数为 26 步；在本项目的重复末动作 padding 策略下，H=16 的 padding 为 27.5%，而 H=32 为 51.7%，因此最终选择 H=16。H=32 的实测成功率只有 14%，验证了过长 chunk 会稀释监督。

### 1.2 Vision encoder 与条件融合

视觉编码器是三层 stride=2 的 CNN（3→32→64→128），每层使用 GroupNorm + SiLU。第一版使用 `AdaptiveAvgPool2d(1)`，但它几乎消除了 target 和障碍的空间位置信息，视觉 DDPM 只有 54% 成功率。最终版本保留 `4×4` 特征网格，再用 Linear(2048, 128) 和 LayerNorm 投影；它仍然是小于 1M 参数的 toy encoder，但能表达“目标在左/右、障碍在何处”。视觉特征与 agent 自身的 2D state、时间 embedding 拼接后输入 action denoiser。

### 1.3 Scheduler 与训练目标

主模型使用 T=100 的 DDPM。给定 clean action chunk `a`、噪声 `ε` 和 `ᾱ_t`，训练构造

```text
a_t = sqrt(ᾱ_t) a + sqrt(1-ᾱ_t) ε
L_DDPM = || ε_θ(a_t, t, image, state) - ε ||_2^2
```

采样时从高斯噪声开始，用 20 个反向步得到长度 H 的 action chunk。作为加分项，Flow Matching 采用 `x_t=(1-t)ε+t a`、回归向量场 `a-ε`，用 10 步 Euler 积分；Pure BC 则固定零 action 输入，直接回归 clean action。

## 2. AutoDL 实验设置与可复现性

| 项目 | 设置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 24GB |
| CPU/RAM | 16 vCPU / 120GB |
| Python / PyTorch | 3.12.3 / 2.8.0+cu128 |
| demos | 1,000 条成功 expert 轨迹，训练 seed=42 |
| optimizer | Adam，lr=1e-3，weight decay=0，EMA=0.999 |
| steps / batch | 10,000 / 256 |
| evaluation | 100 episodes，seed 10000–10099，闭环 exec_steps=4 |

AutoDL 上的启动命令和只读 dashboard 见项目 README。训练、评估和 GPU 状态通过 tmux 与 `monitor_dashboard.py` 持续写入 `status.json`、`metrics.jsonl`；本地 SSH 隧道访问 `http://127.0.0.1:18765/`。原始训练曲线保存在 `results/metrics/`，曲线图为 [results/loss_curve.png](results/loss_curve.png)。

## 3. Ablation 结果

| 实验 | H | 成功率 | 碰撞率 | 超时率 | 平均步数 | 平均末端距离 |
|---|---:|---:|---:|---:|---:|---:|
| Pure BC | 16 | 58% | 12% | 30% | 50.73 | 0.3570 |
| DDPM + global pooling | 16 | 54% | 6% | 40% | 68.93 | 0.4114 |
| **DDPM + 4×4 spatial pooling（最终）** | **16** | **71%** | **3%** | **26%** | **48.70** | **0.1643** |
| Flow Matching + vision | 16 | 57% | 9% | 34% | 52.85 | 0.4804 |
| DDPM + vision，H=32 | 32 | 14% | 1% | 85% | 95.03 | 0.6302 |
| DDPM state-only | 16 | 81% | 12% | 7% | 55.89 | 0.1075 |

每一行都是同一个 100 集 seed 区间，不是训练集重放。最终 `eval_vision.json`、`eval_state.json`、`eval_bc.json` 和 `eval_fm.json` 为可直接读取的 JSON；`results/rollouts/` 中保留了 4 张成功轨迹图。

训练曲线显示 spatial encoder 的 moving-average loss 在 10k 步约 0.097，低于 global pooling 版本约 0.118。单独把 DDPM 每步重规划（exec_steps=1）并没有提升成功率（54%），但碰撞降至 2%，代价是推理调用次数约增加 4 倍；这说明视觉表征而非采样频率是本任务的主要瓶颈。

## 4. 自查问题

1. **为什么 action chunk？** 连续控制相邻动作高度相关，一次预测 H 步能学习局部轨迹并减少采样次数；每步预测虽然反馈最及时，但会增加约 H/`exec_steps` 倍的模型调用。`exec_steps=1` 能更快纠偏和降低碰撞，却牺牲实时性。

2. **多模态问题？** 遇到障碍时左绕和右绕都是合理解。Pure BC 的 MSE 会把两条轨迹平均成穿过障碍的动作；diffusion 从噪声采样，能够表示多峰的 action chunk 分布，并在闭环重规划中选择不同模式。

3. **条件与差值？** state-only 的 81% 比视觉 baseline 的 54% 高 27 个百分点，但 state-only 直接拿到 target 坐标且看不到障碍，不能把这 27 点解释成“视觉增益”。真正有意义的对照是相同 agent state 下 global pooling 与 spatial encoder 的 54%→71%；如果把 distractor 坐标加入 state，视觉 ablation 就失去辨识障碍视觉能力的意义。

4. **padding？** 轨迹中位数 26 步，H=32 时 51.7% 槽位是重复末动作，H=64 时 74.9%。过多重复监督会把策略拉向小/恒定动作，造成超时。改进方法是选择 H≈任务时间尺度（本实验 H=16），使用 mask/weighted loss，或按轨迹长度自适应 chunk。

5. **FM vs DDPM？** FM 直接学习从噪声到动作的连续向量场，推理只需少量 Euler 步，适合实时控制；DDPM 的离散反向链通常需要更多网络调用、但训练和调试更成熟。本实验 FM 10 步的成功率为 57%，尚未超过最终 DDPM，体现了表征与采样质量之间的 trade-off。

## 5. Reflection：从 toy VLA 到 Pi-0/OpenVLA 部署

这个 toy 环境没有体现工业机器人最昂贵的部分：相机和 proprioception 的时间同步、标定漂移、遮挡与光照变化、动作延迟、关节/力矩/速度约束、碰撞安全层、失败恢复和数据闭环。Pi-0/OpenVLA 还需要处理语言目标 grounding、预训练视觉 token 的 domain gap、长时任务的技能切换，以及在有限算力上的量化、缓存和延迟预算。真正部署时不能只看平均成功率；应记录每个场景的风险、置信度和 OOD 检测，给策略配一个可验证的安全控制器，并通过仿真回放、少量真实数据和人工接管逐步扩大覆盖。这个项目最有价值的经验是：空间表征、chunk 时间尺度和闭环评估必须一起设计，低 loss 本身并不等于可执行策略。

## 6. 结论

TODO 19–21 已完成并通过自检/测试。最终小型 spatial CNN + H=16 DDPM 在 100 集未见测试 seed 上达到 71%，提交 checkpoint 为 `ckpts/model_final.pt`。所有对照、失败实验、训练曲线和成功 rollout 均已保留，能够从仓库脚本和配置复现实验。
