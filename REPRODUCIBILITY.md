# Project 1–4 复现文件

代码和实验结果在 `main` 分支；大型权重放在 GitHub Releases，避免把数百 MB 的二进制文件塞进 Git 历史。Project 1 的 challenge Release 提供六个 CIFAR-10 checkpoint 和压缩后的样本附件；Project 2–4 Release 另外提供各项目复现所需的权重。

| 项目 | Release | 权重 | 大小 |
|---|---|---|---:|
| Project1 DDPM challenge | [challenge-v1](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1) | 6 个 `cifar10_{linear,cosine}_200ep_seed{42,43,44}_final.pt` | 2,757,235,046 B |
| Project2 采样器 | [project2-v1.0](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project2-v1.0) | `project2-input-ddpm-cifar10-linear-seed44.pt` | 459,538,769 B |
| Project3 Stable Diffusion | [project3-v1.0](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project3-v1.0) | `project3-lora-vangogh-full.safetensors` | 6,414,448 B |
| Project4 Flow Matching | [project4-v1.0](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project4-v1.0) | `project4-fm-dit-s-step200000.pt` | 522,163,965 B |

在 AutoDL 上从仓库根目录下载并校验：

```bash
mkdir -p .local/checkpoints/project1 .local/checkpoints/project2 .local/checkpoints/project3 .local/checkpoints/project4

for asset in \
  cifar10_linear_200ep_seed42_final.pt \
  cifar10_linear_200ep_seed43_final.pt \
  cifar10_linear_200ep_seed44_final.pt \
  cifar10_cosine_200ep_seed42_final.pt \
  cifar10_cosine_200ep_seed43_final.pt \
  cifar10_cosine_200ep_seed44_final.pt; do
  wget -O ".local/checkpoints/project1/$asset" \
    "https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/challenge-v1/$asset"
done

wget -O .local/checkpoints/project2/project2-input-ddpm-cifar10-linear-seed44.pt \
  https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project2-v1.0/project2-input-ddpm-cifar10-linear-seed44.pt
wget -O .local/checkpoints/project3/project3-lora-vangogh-full.safetensors \
  https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project3-v1.0/project3-lora-vangogh-full.safetensors
wget -O .local/checkpoints/project4/project4-fm-dit-s-step200000.pt \
  https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project4-v1.0/project4-fm-dit-s-step200000.pt

# Release 页面中的清单或下载后的文件可用 SHA256 直接核对：
sha256sum .local/checkpoints/project1/*.pt
sha256sum .local/checkpoints/project2/project2-input-ddpm-cifar10-linear-seed44.pt
sha256sum .local/checkpoints/project3/project3-lora-vangogh-full.safetensors
sha256sum .local/checkpoints/project4/project4-fm-dit-s-step200000.pt
```

Project2 可复用 Project1 challenge-v1 中的 CIFAR-10 DDPM checkpoint；Project3 需要另外从 Hugging Face 下载基础模型 `stable-diffusion-v1-5/stable-diffusion-v1-5`，Release 只提供几 MB 的 LoRA adapter；Project4 使用 200,000 steps 的 DiT-S EMA checkpoint。下载后，分别按各项目 README 中的 AutoDL 命令运行 benchmark、LoRA evaluation 和 NFE/FID 扫描。
