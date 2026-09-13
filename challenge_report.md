# 项目 1 挑战档技术报告（八页结构稿）

> 本文按八页技术报告组织，可在实验完成后导出为 PDF。当前版本只完成方法、实验协议和结果表结构；由于本轮明确不运行训练，所有经验结果、均值、标准差和失败案例均保持待填，不虚构数值。

## 第 1 页：摘要与研究问题

### 研究问题

在 U-Net、优化器、数据增强、训练轮数和随机种子集合保持一致的前提下，比较 linear 与 cosine beta 调度策略对 CIFAR-10 无条件 DDPM 的影响。主要指标是使用 EMA 权重生成的 FID，辅助指标是 raw 权重 FID、训练 loss 曲线和 64 张样本网格。

### 可复现性声明

挑战实验矩阵默认是两个调度策略和三个随机种子：`42、43、44`，每个组合训练 200 轮；`challenge.py` 也支持显式加入 50 轮控制组。每个组合评估 5,000 张生成图像与 5,000 张无增强 CIFAR-10 训练图像，并写入 `results.csv` 和 `summary.md`。

### 当前状态

实现已准备好，但实验尚未执行。因此不能回答“哪个调度策略更好”、不能填写均值 ± 标准差，也不能把预期现象写成失败案例。

## 第 2 页：DDPM 数学背景

前向过程使用固定的高斯转移：

\[
q(x_t\mid x_{t-1})=\mathcal N(\sqrt{1-\beta_t}x_{t-1},\beta_t I).
\]

令 \(\alpha_t=1-\beta_t\)，\(\bar\alpha_t=\prod_{s=1}^{t}\alpha_s\)，则可以用闭合形式直接采样：

\[
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\quad\epsilon\sim\mathcal N(0,I).
\]

模型 \(\epsilon_\theta(x_t,t)\) 预测噪声，训练目标为：

\[
\mathcal L_{simple}=\mathbb E_{x_0,t,\epsilon}
\left[\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert^2\right].
\]

调度策略不改变模型结构，而是改变每个时间步的噪声强度以及训练样本在不同信噪比区域的分布。

## 第 3 页：linear 与 cosine 调度策略实现

### linear 调度策略

`linear_beta_schedule` 在 `[beta_start, beta_end]` 之间等间隔生成 `T` 个 beta。它直接对应原始 DDPM 常用基线，优点是简单、易解释；缺点是时间步上的信噪比变化不一定均衡。

### cosine 调度策略

`cosine_beta_schedule` 先构造：

\[
\bar\alpha(t)=\cos^2\left(\frac{t/T+s}{1+s}\frac{\pi}{2}\right),
\]

再通过相邻累积量之比得到：

\[
\beta_t=1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}.
\]

实现对 beta 做有限性和 `(0,1)` 范围检查，并使用数值截断避免极端时间步造成不稳定。两种调度策略都进入同一个 `DDPMSchedule`，因此训练和采样逻辑无需分叉。

### 实现自查

- `betas`、`alphas_cumprod` 和 posterior 系数注册为 buffer。
- `t` 从 0 开始索引。
- batch 时间系数 reshape 为 `(B, 1, 1, 1)` 后再广播到图像张量。
- `t=0` 的 posterior variance 显式设为 0，最后一步不添加随机扰动。

## 第 4 页：实验设计

### 控制变量

| 项目 | 固定值 |
|---|---|
| 数据集 | CIFAR-10 训练集 |
| 分辨率 | 32 × 32 |
| Batch size | 128 |
| 训练轮数 | 200 |
| 模型 | base 128，channel mult 1/2/2/2 |
| 优化器 | AdamW |
| 学习率 | 2e-4 |
| Warmup | 5,000 步 |
| EMA | 衰减 0.9999 |
| 精度 | fp16 |
| 数据增强 | 仅训练阶段随机水平翻转 |
| FID 真实图 | 5,000 张无增强训练图 |
| FID 生成图 | 每次运行 5,000 张 |

唯一实验变量是 `beta_schedule`。两个调度策略各运行随机种子 `42、43、44`，并用同一评估随机种子规则生成可比较的随机样本。

### 运行方式

```bash
python challenge.py run \
  --schedules linear cosine \
  --seeds 42 43 44 \
  --output_root runs/challenge
```

若要同时完成 50 轮/200 轮的调度策略自查：

```bash
python challenge.py run \
  --schedules linear cosine \
  --epoch_budgets 50 200 \
  --seeds 42 43 44 \
  --output_root runs/challenge
```

每个实验独立保存 checkpoint、loss history、loss curve、EMA 样本网格和 raw/EMA FID。若训练中断，可保留已完成目录，修复后使用单独的 `train.py --resume` 命令恢复；恢复前应在日志中记录原因。

## 第 5 页：结果表与统计方法

本页必须由真实实验产物生成，不能手工估计。

### 每个随机种子的结果

| 调度策略 | 随机种子 | 最终 loss | EMA FID | Raw FID | 训练时间 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| linear | 42 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| linear | 43 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| linear | 44 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| cosine | 42 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| cosine | 43 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| cosine | 44 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

### 均值 ± 标准差

`challenge.py summarize` 使用三个随机种子的算术平均值和样本标准差：

\[
\bar x=\frac{1}{n}\sum_i x_i,\qquad
s=\sqrt{\frac{1}{n-1}\sum_i(x_i-\bar x)^2}.
\]

最终应报告 linear EMA、cosine EMA、linear raw、cosine raw 四组统计，并明确 FID 的真实数据划分、生成样本数和评估随机种子。

## 第 6 页：训练曲线与样本质量分析

本页待实验后插入：

- `runs/challenge/cifar10_linear_*ep_seed*/loss_curve.png`
- `runs/challenge/cifar10_cosine_*ep_seed*/loss_curve.png`
- 每个调度策略至少一个 EMA 样本网格
- 可选：同一随机种子下 linear/cosine 的并排样本网格

分析应区分训练 loss 和生成质量：loss 更低不必然意味着 FID 更低；样本网格应观察颜色、轮廓、多样性、重复样本和明显伪影。EMA/raw 的差异也应结合 FID 和图像，而不是只凭单张样本下结论。

对于“50 轮与 200 轮哪个调度策略更好”的自查问题，本仓库只提供 200 轮挑战矩阵。若要回答 50 轮，必须新增同样的 50 轮控制实验，不能从 200 轮结果外推。

## 第 7 页：失败案例与威胁有效性

### 真实失败案例

本页只允许写入实际发生且有日志证据的问题，例如：数据下载中断、显存不足、NaN、checkpoint 恢复错位、FID 输入范围错误或某个调度策略的训练失败。每条记录都应包含命令、日志片段或文件证据、根因和修复验证，格式见 `logs/challenge_experiment_log_template.md` 和 `debug_log.md`。

当前不能预填失败案例，因为本轮尚未运行挑战实验。已有的 AutoDL 数据下载慢等问题属于前序 CIFAR-10 进阶实验，不能冒充挑战档中的调度策略失败。

### 威胁有效性

- 三个随机种子仍然是小样本，均值 ± 标准差不能等同于统计显著性检验。
- FID 对 Inception 实现、输入范围和 real split 敏感。
- 同一训练时长不代表两个调度策略达到相同优化程度。
- 仅比较一个 U-Net 容量和一个学习率，结论只适用于本实验设置。
- 评估样本数为 5,000，结果可能比大规模评估有更高方差。

## 第 8 页：结论与复现清单

### 结论模板

实验完成后应将下面的“待填写”替换为真实结论：

> 在 CIFAR-10、200 轮、相同 U-Net/优化器和随机种子 `42/43/44` 的条件下，`[linear/cosine]` 的 EMA FID 为 `[均值 ± 标准差]`，`[优于/不优于]` 另一调度策略的 `[均值 ± 标准差]`。该结论仅适用于本实验协议。EMA 相对 raw 的变化为 `[填写]`。

### 复现清单

- [ ] `configs/cifar10_linear.yaml` 与 `configs/cifar10_cosine.yaml` 已固定控制变量
- [ ] 两个调度策略均完成随机种子 42、43、44（200 轮；若回答 50 轮自查则再完成 50 轮）
- [ ] 六个 checkpoint 的 MD5 或文件信息已记录（完整 50+200 矩阵为十二个）
- [ ] 六份 loss history 和 loss curve 已保存（完整 50+200 矩阵为十二份）
- [ ] 六个 FID 结果均为 5,000 对 5,000（完整 50+200 矩阵为十二个）
- [ ] `python challenge.py summarize` 生成 `results.csv` 和 `summary.md`
- [ ] 失败案例全部有证据，未将推测写成事实
- [ ] 报告中的图、表和结论与产物路径一致
