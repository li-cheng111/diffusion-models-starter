# Project 2: sampler comparison

This directory contains the complete Project 2 implementation imported from
the course starter at commit `774d640f3915c3396e034068beff318fbf421719`
and adapted to this repository's monorepo layout.

It reuses the trained Project 1 epsilon-prediction model without retraining and
implements or evaluates:

- DDPM ancestral sampling as a 1000-NFE reference;
- DDIM with arbitrary step skipping and `eta` stochasticity;
- Euler sampling;
- second-order singlestep DPM-Solver;
- DDIM inversion and deterministic reconstruction;
- reproducible FID-versus-NFE and trajectory comparisons.

## Layout and data policy

Project 1 remains in the parent directory. `project1_path.py` checks that parent
first and imports its `schedule.py`, `dataset.py`, and `model/` package.

Do not put a `.git` directory below this folder. Model checkpoints, datasets,
Inception weights, and caches are intentionally ignored. Commit only source,
configuration, JSON measurements, curated plots/sample grids, logs, and reports.

The selected formal checkpoint is the linear-schedule seed-44 EMA model. Keep it
outside Git, for example:

```text
repository-root/local_checkpoints/cifar10_linear_200ep_seed44_final.pt
```

From this directory, its path is
`../local_checkpoints/cifar10_linear_200ep_seed44_final.pt`.

## Setup and checks

```bash
cd project2-samplers
pip install -r requirements.txt
python check_compat.py
python -m unittest discover -s tests -v
python -m samplers.ddim
python -m samplers.dpm_solver
```

Generate a qualitative grid without installing the FID dependencies:

```bash
python sample_grid.py --ckpt ../runs/exp_mnist_baseline/ckpt/final.pt \
  --sampler ddim --steps 50 --output samples/smoke/mnist_ddim50.png
```

To validate the real checkpoint format as well:

```bash
python check_compat.py \
  --ckpt ../local_checkpoints/cifar10_linear_200ep_seed44_final.pt
```

## Quick validation

Run a small benchmark before committing expensive AutoDL time:

```bash
python benchmark.py \
  --ckpt ../local_checkpoints/cifar10_linear_200ep_seed44_final.pt \
  --sampler ddim dpm-solver --steps 5 10 \
  --num_samples 128 --batch_size 32 \
  --output runs/smoke_benchmark.json \
  --plot runs/smoke_pareto.png
```

Small-sample FID is only a pipeline check and must not be reported as the final
metric.

## Formal AutoDL experiments

### 实时可视化监控

`monitor_dashboard.py` 是只读监控服务，会每 2 秒刷新 GPU、benchmark 进程、配置进度、已完成 FID、预览文件和日志。AutoDL 上启动后，通过 SSH 本地端口转发访问：

```bash
python monitor_dashboard.py --host 127.0.0.1 --port 18765
# 本地另开终端：
ssh -N -L 18765:127.0.0.1:18765 -p 23398 root@connect.bjb1.seetacloud.com
```

然后打开 `http://127.0.0.1:18765/`。正式实验的远端页面当前运行在该端口；只要 SSH 转发保持连接，浏览器页面就会自动更新。

The full preset uses the assignment matrix: DDPM 1000 steps; DDIM and Euler at
10/20/50/100/250 steps; DPM-Solver-2 at 5/10/25/50 outer steps. DPM-Solver's
reported NFE is `2 * outer_steps - 1` because the terminal step needs one model
call.

```bash
python benchmark.py \
  --ckpt ../local_checkpoints/cifar10_linear_200ep_seed44_final.pt \
  --preset full --num_samples 5000 --batch_size 64 --seed 42 \
  --data_root ../data \
  --output runs/benchmark_all.json \
  --plot runs/pareto_fid_nfe.png
```

The required DDIM-only artifact can be produced independently (use the same
checkpoint, seed, real-data policy, and batch size):

```bash
python benchmark.py \
  --ckpt ../local_checkpoints/cifar10_linear_200ep_seed44_final.pt \
  --sampler ddim --steps 10 20 50 100 250 \
  --num_samples 5000 --batch_size 64 --seed 42 \
  --data_root ../data \
  --output runs/benchmark_ddim.json \
  --plot runs/pareto_ddim.png
```

All configurations use the same 5,000 unaugmented CIFAR-10 training images and
restart the generated-noise stream from seed 42.

Trajectory comparison:

```bash
python visualize_trajectories.py \
  --ckpt ../local_checkpoints/cifar10_linear_200ep_seed44_final.pt \
  --samplers ddpm ddim euler dpm-solver --steps 50 \
  --output runs/trajectory_comparison.png \
  --metrics_output runs/trajectory_comparison.json
```

DDIM inversion:

```bash
python evaluate_inversion.py \
  --ckpt ../local_checkpoints/cifar10_linear_200ep_seed44_final.pt \
  --steps 10 20 50 100 250 --num_images 64 \
  --data_root ../data \
  --output runs/inversion_results.json \
  --plot runs/inversion_errors.png
```

## Completion status

- [x] Project 1 discovery and checkpoint compatibility
- [x] DDIM implementation
- [x] reproducible FID/NFE benchmark and Pareto plotting
- [x] shared-noise trajectory comparison
- [x] DPM-Solver-2 implementation with true NFE counting
- [x] DDIM inversion and reconstruction evaluation
- [x] unit and compatibility tests
- [x] formal 5,000-sample AutoDL measurements
- [x] replace report result placeholders with measured values
