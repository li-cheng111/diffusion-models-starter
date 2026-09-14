# AutoDL full run（2026-09-13）

## 环境

- GPU：NVIDIA GeForce RTX 4090，24,564 MiB；Python 3.12.3；PyTorch 2.8.0+cu128；CUDA 12.8。
- Diffusers 0.40.0、Transformers 5.15.1、Accelerate 1.14.0、PEFT 0.20.0。
- HF endpoint 无法连接；基础模型使用 ModelScope `AI-ModelScope/stable-diffusion-v1-5@master`，
  ControlNet 使用 `lllyasviel/sd-controlnet-canny@master`，权重 SHA256 见
  `outputs/autodl_environment.json`。

## 冒烟与全量

- Project 3 pytest：`5 passed`。
- 手写推理：50-step、seed 42，保存 `outputs/manual/`。
- 参数扫描：CFG `[1,3,7.5,15,25]`、steps `[10,20,50,100]`、DDIM/Euler-A/DPM++ 2M，
  512×512，耗时 36.642 秒，16 张单图 + 4 张总图/网格。
- VAE：posterior mode，输入和重构 `(1,3,512,512)`，latent `(1,4,64,64)`，MSE
  `0.0019791808`，PSNR `27.0351 dB`。
- LoRA：20 张镜像数据、rank/alpha 8、lr `1e-4`、batch 1、gradient checkpointing、
  fp16、800 steps，耗时 209.47 秒；loss `0.2810367 → 0.1527308`，last-50 mean
  `0.2282097`，最终 adapter 6.4 MB。200/400/600/800 验证图与独立评估图均已保存。
- ControlNet：Canny 阈值 `[100,200]`，4 个风格 prompt + 1 个结构/文本冲突 prompt，
  结果和 SHA256 在 `outputs/controlnet/metadata.json`。
- Cross-attention：30 steps，conditional CFG 分支，32 层、16×16/32×32 map，
  `cat/wizard/hat/forest` 四 token 热力图和 metadata 已保存。

## 实际 debug

Wikimedia Commons API 在 AutoDL 返回 `OSError: [Errno 99] Cannot assign requested address`；
为不伪造来源，训练图改用 ModelScope 公共 `huggan/vangogh2photo` 镜像的 `imageA`，
manifest 标明镜像来源和许可证。Commons downloader 仍保留在代码中，原图不进 Git。
