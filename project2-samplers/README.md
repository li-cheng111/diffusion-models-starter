# 项目 2：采样器对比

本目录包含完整的项目 2 实现。代码从课程 starter 的
`774d640f3915c3396e034068beff318fbf421719` 提交导入，并适配到本仓库的
monorepo 目录结构。

实现不重新训练，直接复用项目 1 已训练的 epsilon 预测模型，并完成：

- DDPM ancestral sampling 1000 NFE 基线；
- 支持任意跳步和 `eta` 随机性的 DDIM；
- Euler 采样；
- 二阶单步 DPM-Solver；
- DDIM 反演和确定性重构；
- 可复现的 FID-NFE 对比和轨迹对比。

## 目录和数据规则

项目 1 保留在父目录。`project1_path.py` 优先检查父目录，并导入其中的
`schedule.py`、`dataset.py` 和 `model/` 包。

本目录下不得放置 `.git` 目录。模型 checkpoint、数据集、Inception 权重和
缓存均会被忽略。只提交源码、配置、JSON 测量结果、整理后的图表/样本网格、
日志和报告。

正式实验使用 linear 调度策略的随机种子 44 EMA 模型。checkpoint 应放在 Git 仓库
之外，例如：

```text
repository-root/local_checkpoints/cifar10_linear_200ep_seed44_final.pt
```

从本目录运行时，对应路径为
`../local_checkpoints/cifar10_linear_200ep_seed44_final.pt`。

## 环境配置和检查

```bash
cd project2-samplers
pip install -r requirements.txt
python check_compat.py
python -m unittest discover -s tests -v
python -m samplers.ddim
python -m samplers.dpm_solver
```

无需安装 FID 依赖即可生成定性样本网格：

```bash
python sample_grid.py --ckpt ../runs/exp_mnist_baseline/ckpt/final.pt \
  --sampler ddim --steps 50 --output samples/smoke/mnist_ddim50.png
```

如需同时验证真实 checkpoint 格式：

```bash
python check_compat.py \
  --ckpt ../local_checkpoints/cifar10_linear_200ep_seed44_final.pt
```

## 快速验证

在占用 AutoDL 进行长时间计算前，先运行小规模 benchmark：

```bash
python benchmark.py \
  --ckpt ../local_checkpoints/cifar10_linear_200ep_seed44_final.pt \
  --sampler ddim dpm-solver --steps 5 10 \
  --num_samples 128 --batch_size 32 \
  --output runs/smoke_benchmark.json \
  --plot runs/smoke_pareto.png
```

小样本 FID 仅用于检查流程，不能作为最终指标报告。

## AutoDL 正式实验

### 实时可视化监控

`monitor_dashboard.py` 是只读监控服务，会每 2 秒刷新 GPU、benchmark 进程、配置进度、已完成 FID、预览文件和日志。AutoDL 上启动后，通过 SSH 本地端口转发访问：

```bash
python monitor_dashboard.py --host 127.0.0.1 --port 18765
# 本地另开终端：
ssh -N -L 18765:127.0.0.1:18765 -p 23398 root@connect.bjb1.seetacloud.com
```

然后打开 `http://127.0.0.1:18765/`。正式实验的远端页面当前运行在该端口；只要 SSH 转发保持连接，浏览器页面就会自动更新。

`full` 预设使用作业要求的实验矩阵：DDPM 1000 步；DDIM 和 Euler 使用
10/20/50/100/250 步；DPM-Solver-2 使用 5/10/25/50 个外层步。DPM-Solver
报告的 NFE 为 `2 * outer_steps - 1`，因为最后一个终止步只需要一次模型调用。

```bash
python benchmark.py \
  --ckpt ../local_checkpoints/cifar10_linear_200ep_seed44_final.pt \
  --preset full --num_samples 5000 --batch_size 64 --seed 42 \
  --data_root ../data \
  --output runs/benchmark_all.json \
  --plot runs/pareto_fid_nfe.png
```

DDIM 专项结果可以独立生成（必须使用相同 checkpoint、随机种子、真实数据规则和
批大小）：

```bash
python benchmark.py \
  --ckpt ../local_checkpoints/cifar10_linear_200ep_seed44_final.pt \
  --sampler ddim --steps 10 20 50 100 250 \
  --num_samples 5000 --batch_size 64 --seed 42 \
  --data_root ../data \
  --output runs/benchmark_ddim.json \
  --plot runs/pareto_ddim.png
```

所有配置都使用相同的 5,000 张无增强 CIFAR-10 训练图，并在每个配置开始时将
生成噪声流重置为随机种子 42。

轨迹对比：

```bash
python visualize_trajectories.py \
  --ckpt ../local_checkpoints/cifar10_linear_200ep_seed44_final.pt \
  --samplers ddpm ddim euler dpm-solver --steps 50 \
  --output runs/trajectory_comparison.png \
  --metrics_output runs/trajectory_comparison.json
```

DDIM 反演：

```bash
python evaluate_inversion.py \
  --ckpt ../local_checkpoints/cifar10_linear_200ep_seed44_final.pt \
  --steps 10 20 50 100 250 --num_images 64 \
  --data_root ../data \
  --output runs/inversion_results.json \
  --plot runs/inversion_errors.png
```

## 完成状态

- [x] 项目 1 路径发现和 checkpoint 兼容性
- [x] DDIM 实现
- [x] 可复现的 FID/NFE benchmark 和 Pareto 图
- [x] 共享初始噪声的轨迹对比
- [x] DPM-Solver-2 实现和真实 NFE 计数
- [x] DDIM 反演和重构评估
- [x] 单元测试和兼容性测试
- [x] AutoDL 5,000 样本正式测量
- [x] 用实测值替换报告中的占位内容
