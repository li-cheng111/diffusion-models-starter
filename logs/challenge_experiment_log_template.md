# Project 1 挑战档实验日志

> 本文件是挑战档的真实实验记录模板。未实际运行前，不填写训练时间、loss、FID、样本质量或失败原因。

## 实验矩阵

默认记录 200 epoch 矩阵；若执行 `--epoch_budgets 50 200`，将每行复制为 50 epoch 和 200 epoch 两行，并在路径中保留 epoch 标记。

| Schedule | Epochs | Seed | Output directory | Status | Final loss | EMA FID | Raw FID | Time |
|---|---:|---:|---|---|---:|---:|---:|---:|
| linear | 200 | 42 | `runs/challenge/cifar10_linear_200ep_seed42` | pending |  |  |  |  |
| linear | 200 | 43 | `runs/challenge/cifar10_linear_200ep_seed43` | pending |  |  |  |  |
| linear | 200 | 44 | `runs/challenge/cifar10_linear_200ep_seed44` | pending |  |  |  |  |
| cosine | 200 | 42 | `runs/challenge/cifar10_cosine_200ep_seed42` | pending |  |  |  |  |
| cosine | 200 | 43 | `runs/challenge/cifar10_cosine_200ep_seed43` | pending |  |  |  |  |
| cosine | 200 | 44 | `runs/challenge/cifar10_cosine_200ep_seed44` | pending |  |  |  |  |

## 固定环境

- Date:
- Host / GPU:
- Python / PyTorch / CUDA:
- Git commit:
- Dataset archive and checksum:
- Dataset root:

## 固定训练配置

- Dataset: CIFAR-10 train split
- Resolution: 32x32
- Batch size: 128
- Epochs: 200
- Diffusion steps `T`: 1000
- Optimizer: AdamW
- Learning rate: `2e-4`
- Warmup steps: 5,000
- EMA decay: `0.9999`
- Mixed precision: fp16
- Augmentation: random horizontal flip during training only
- FID real split: 5,000 unaugmented train images
- Generated images per FID run: 5,000

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

- Linear EMA FID mean ± std:
- Cosine EMA FID mean ± std:
- Linear raw FID mean ± std:
- Cosine raw FID mean ± std:
- Which schedule is better at 200 epochs:
- Statistical caveat:

## 真实失败案例记录

每条记录至少包含：现象、最小复现命令、证据、根因、修复、修复后的验证。

### Failure 1

- Status:
- Symptom:
- Evidence:
- Root cause:
- Fix:
- Verification:

### Failure 2

- Status:
- Symptom:
- Evidence:
- Root cause:
- Fix:
- Verification:

### Failure 3

- Status:
- Symptom:
- Evidence:
- Root cause:
- Fix:
- Verification:
