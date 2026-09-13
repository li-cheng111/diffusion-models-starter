# exp_20260913_01_smoke_validation

## 目标

验证 Project 3 A–E 的代码入口、依赖、固定 revision、产物元数据和 adapter 重载链路；本次不是 AutoDL full run。

## 环境

| 项目 | 值 |
|---|---|
| 平台 | Windows 11，本机 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8 GB |
| Python | 3.14.0 |
| PyTorch/CUDA | 2.13.0+cu126 / CUDA 12.6 |
| diffusers / transformers | 0.40.0 / 5.15.1 |
| accelerate / peft | 1.14.0 / 0.20.0 |
| Git commit | `43b9308`（代码修复；notebook A/C 已在前一提交执行） |
| model revision | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` |

## 命令与结果

1. `check_env.py --model_revision 451f4fe... --json_output <external>/env_check.json`：依赖、HF 模型通道通过；报告 8 GB 显存警告。
2. `01_inference_walkthrough.ipynb`：nbconvert 执行无 error；50-step 手写循环、latent 统计和图片输出已嵌入 notebook。
3. `02_parameter_sweep.py --preset smoke --height 256 --width 256`：输出 cfg/steps/sampler/grid 共 9 张 PNG 和 metadata；运行约 34.345 s。
4. `05_vae_anatomy.ipynb`：无 error；latent `(1,4,64,64)`，MSE `1.6900175e-4`，PSNR `37.7211 dB`。
5. `03_lora_finetune.py --num_train_steps 2 --checkpointing_steps 1 --num_workers 0 --validation_prompts ...`：loss `0.2498 → 0.2942`，无 NaN/Inf；adapter 约 6.4 MB；`validation/step-0001_prompt-00.png` 和 `step-0002_prompt-00.png` 均生成。
6. `evaluate_lora.py --steps 2`：全新 pipeline 载入 base/checkpoint-0001/checkpoint-0002 成功；修复 loader 后三张图 SHA256 不同。
7. `06_cross_attention_visualization.py --steps 5 --tokens cat wizard hat forest`：生成图、4 张 token overlay；metadata 报告 16×16/32×32 map 和有效 token index `[3,7,8,12]`。

## 失败/阻塞

- Commons 旧根分类直接筛选只有 19 个 public-domain 文件；下载器已改为按 `Paintings_by_Vincent_van_Gogh_by_title` 递归子分类并加入 429 重试。当前 IP 在抓取 20 张时触发过 429，AutoDL 应重新执行并保留 manifest。
- ControlNet 权重在本机开始下载但受网络速度限制，缓存停在约 1.45 GB 未完成，因此本次不记录 ControlNet 图像为成功结果；AutoDL 需先完成权重下载再执行 notebook。

## 下一次（AutoDL 4090）

先做所有 smoke，再运行 full sweep、A/C notebook、800-step LoRA（每 200 步验证/保存）、ControlNet 四 prompt+冲突、cross-attention 30 steps。结束后记录 `nvidia-smi` 峰值、loss finite 检查、文件数量、单文件大小和 `git status`，然后只提交 Project 3 产物。
