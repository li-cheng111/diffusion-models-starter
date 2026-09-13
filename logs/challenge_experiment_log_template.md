# 项目 1 挑战档实验日志

> 本文件是挑战档的真实实验记录模板。未实际运行前，不填写训练时间、loss、FID、样本质量或失败原因。

## 实验矩阵

默认记录 200 轮矩阵；若执行 `--epoch_budgets 50 200`，将每行复制为 50 轮和 200 轮两行，并在路径中保留 epoch 标记。

| 调度策略 | 训练轮数 | 随机种子 | 输出目录 | 状态 | 最终 loss | EMA FID | Raw FID | 时间 |
|---|---:|---:|---|---|---:|---:|---:|---:|
| linear | 200 | 42 | `runs/challenge/cifar10_linear_200ep_seed42` | 待填写 |  |  |  |  |
| linear | 200 | 43 | `runs/challenge/cifar10_linear_200ep_seed43` | 待填写 |  |  |  |  |
| linear | 200 | 44 | `runs/challenge/cifar10_linear_200ep_seed44` | 待填写 |  |  |  |  |
| cosine | 200 | 42 | `runs/challenge/cifar10_cosine_200ep_seed42` | 待填写 |  |  |  |  |
| cosine | 200 | 43 | `runs/challenge/cifar10_cosine_200ep_seed43` | 待填写 |  |  |  |  |
| cosine | 200 | 44 | `runs/challenge/cifar10_cosine_200ep_seed44` | 待填写 |  |  |  |  |

## 固定环境

- 日期：
- 主机 / GPU：
- Python / PyTorch / CUDA：
- Git 提交：
- 数据集压缩包和校验和：
- 数据集根目录：

## 固定训练配置

- 数据集：CIFAR-10 训练集
- 分辨率：32x32
- Batch size：128
- 训练轮数：200
- 扩散步数 `T`：1000
- 优化器：AdamW
- 学习率：`2e-4`
- Warmup 步数：5,000
- EMA 衰减：`0.9999`
- 混合精度：fp16
- 数据增强：仅训练阶段随机水平翻转
- FID 真实划分：5,000 张无增强训练图
- 每次 FID 运行的生成图：5,000 张

## 命令

```bash
python challenge.py run \
  --schedules linear cosine \
  --seeds 42 43 44 \
  --output_root runs/challenge
```

如果所有训练都已完成而只需要重新汇总：

```bash
python challenge.py summarize \
  --schedules linear cosine \
  --seeds 42 43 44 \
  --output_root runs/challenge
```

## 结果汇总

- linear EMA FID 均值 ± 标准差：
- cosine EMA FID 均值 ± 标准差：
- linear raw FID 均值 ± 标准差：
- cosine raw FID 均值 ± 标准差：
- 200 轮下哪个调度策略更好：
- 统计学注意事项：

## 真实失败案例记录

每条记录至少包含：现象、最小复现命令、证据、根因、修复、修复后的验证。

### 失败案例 1

- 状态：
- 现象：
- 证据：
- 根因：
- 修复：
- 验证：

### 失败案例 2

- 状态：
- 现象：
- 证据：
- 根因：
- 修复：
- 验证：

### 失败案例 3

- 状态：
- 现象：
- 证据：
- 根因：
- 修复：
- 验证：
