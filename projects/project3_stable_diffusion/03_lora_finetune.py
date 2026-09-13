"""
SD LoRA 微调脚手架

任务：用一组小数据集（10-30 张特定风格/对象图）微调 SD UNet 的 attention 层。

只微调 LoRA 适配器（rank=8），不动原模型权重。

依赖：
    pip install peft accelerate

用法：
    python 03_lora_finetune.py \
        --train_data_dir ./my_dataset \
        --instance_prompt "a photo of sks dog" \
        --output_dir ./lora_output \
        --num_train_steps 500
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffusers import (
    AutoencoderKL, UNet2DConditionModel,
    DDPMScheduler,
)
from transformers import CLIPTokenizer, CLIPTextModel


class InstanceDataset(Dataset):
    """每张图配同一 instance prompt 的简单数据集."""
    def __init__(self, data_dir, prompt, tokenizer, size=512):
        self.data_dir = Path(data_dir)
        self.image_paths = list(self.data_dir.glob('*.jpg')) + \
                           list(self.data_dir.glob('*.png')) + \
                           list(self.data_dir.glob('*.jpeg'))
        assert len(self.image_paths) > 0, f"No images found in {data_dir}"
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        token_ids = self.tokenizer(
            self.prompt, padding='max_length', max_length=77,
            truncation=True, return_tensors='pt',
        ).input_ids[0]
        return img, token_ids


# ============================================================
# TODO 13: 用 peft 给 UNet 添加 LoRA adapter
# ============================================================
def add_lora_to_unet(unet, rank=8):
    """
    给 UNet 的 attention 层添加 LoRA。

    提示：
    - 用 `peft.LoraConfig` 指定 target_modules
    - SD UNet 的 attention 模块叫 'to_q', 'to_k', 'to_v', 'to_out.0'
    - 这些 module 分布在多个 cross/self-attn 块中

    返回：增加了 LoRA 的 UNet（仅 LoRA params 可训练）
    """
    from peft import LoraConfig, get_peft_model  # noqa

    # ─────── TODO 13 ───────
    # lora_config = LoraConfig(
    #     r=rank,
    #     lora_alpha=rank,
    #     init_lora_weights="gaussian",
    #     target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    # )
    # unet = get_peft_model(unet, lora_config)
    # unet.print_trainable_parameters()
    # return unet
    raise NotImplementedError("TODO 13: 配置 LoRA adapter")


# ============================================================
# TODO 14: 训练循环
# ============================================================
def train_one_step(batch, unet, vae, text_encoder, scheduler, optimizer,
                   device, weight_dtype, autocast_dtype, scaler):
    """
    单步训练。

    流程：
    1. img → vae.encode → latent (注意 sample + scale 0.18215)
    2. 加噪：t ~ U[0, T], latent_t = sqrt(α̅_t) latent + sqrt(1-α̅_t) noise
    3. text → text_encoder → text_emb
    4. UNet 预测 noise: eps_pred = unet(latent_t, t, text_emb)
    5. Loss = MSE(eps_pred, noise)
    6. Backward + step

    ⚠️ 混合精度的正确姿势（这一点最容易写崩）：
    - vae / text_encoder 是冻结的，用 weight_dtype（fp16）跑，省显存
    - **UNet 与 LoRA 参数保持 fp32**，前向用 autocast 包起来
    - fp16 下必须用 GradScaler，否则梯度下溢，loss 会一路 NaN
    - latent 送进 UNet 前要 `.to(autocast_dtype)`；算 loss 时两边都 `.float()`

    Args:
        weight_dtype: 冻结模块（vae/text_encoder）的 dtype
        autocast_dtype: autocast 用的 dtype；mixed_precision='no' 时为 None
        scaler: torch.amp.GradScaler；fp32 时 enabled=False，可以照常调用
    """
    images, token_ids = batch
    images = images.to(device, dtype=weight_dtype)
    token_ids = token_ids.to(device)
    B = images.shape[0]

    # autocast 上下文：mixed_precision='no' 时退化为空操作
    amp = torch.autocast(device_type=device.type, dtype=autocast_dtype,
                         enabled=autocast_dtype is not None)

    # ─────── TODO 14 ───────
    # 1. encode 图像到 latent（vae 冻结，no_grad）
    # with torch.no_grad():
    #     latents = vae.encode(images).latent_dist.sample() * 0.18215
    #     latents = latents.float()          # 回到 fp32，后面交给 autocast
    #
    # 2. 采样 t 和 noise
    # t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()
    # noise = torch.randn_like(latents)
    # noisy_latents = scheduler.add_noise(latents, noise, t)
    #
    # 3. 文本编码（冻结，no_grad）
    # with torch.no_grad():
    #     text_emb = text_encoder(token_ids)[0]
    #
    # 4. UNet 预测 + loss，都放进 autocast
    # with amp:
    #     noise_pred = unet(noisy_latents, t,
    #                       encoder_hidden_states=text_emb.to(noisy_latents.dtype)).sample
    #     loss = F.mse_loss(noise_pred.float(), noise.float())
    #
    # 5. Backward（经过 scaler）
    # scaler.scale(loss).backward()
    # scaler.step(optimizer)
    # scaler.update()
    # optimizer.zero_grad(set_to_none=True)
    #
    # return loss.item()
    raise NotImplementedError("TODO 14: 实现单步训练")


# ============================================================
# TODO 15: 训练后保存 LoRA 权重
# ============================================================
def save_lora_weights(unet, output_dir):
    """
    保存 LoRA 权重（只存 LoRA 部分，非常小，几 MB）。

    提示：用 `unet.save_pretrained(output_dir)` 或
        `from peft import get_peft_model_state_dict; ...`
    """
    # ─────── TODO 15 ───────
    # unet.save_pretrained(output_dir)
    # print(f"[saved] LoRA weights → {output_dir}")
    raise NotImplementedError("TODO 15: 保存 LoRA 权重")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str, required=True)
    parser.add_argument('--instance_prompt', type=str, required=True,
                        help='e.g., "a photo of sks dog"')
    parser.add_argument('--output_dir', type=str, default='./lora_output')
    parser.add_argument('--model_id', type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument('--num_train_steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--mixed_precision', type=str, default='fp16',
                        choices=['no', 'fp16', 'bf16'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # CPU 上没有可用的 fp16 训练路径，直接退回 fp32
    if device.type != 'cuda' and args.mixed_precision != 'no':
        print("[warn] 非 CUDA 设备，mixed_precision 强制改为 'no'")
        args.mixed_precision = 'no'

    # 冻结模块（vae / text_encoder）用低精度省显存
    weight_dtype = {'no': torch.float32,
                    'fp16': torch.float16,
                    'bf16': torch.bfloat16}[args.mixed_precision]
    # 前向 autocast 的 dtype；fp32 时为 None（不开 autocast）
    autocast_dtype = None if args.mixed_precision == 'no' else weight_dtype

    # ─────── Load components ───────
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder").to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device, dtype=weight_dtype)
    # ⚠️ UNet 保持 fp32：LoRA 参数要在 fp32 下更新，混合精度靠 autocast + GradScaler
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").to(device, dtype=torch.float32)
    scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # ─────── Add LoRA ───────
    unet = add_lora_to_unet(unet, rank=args.rank)

    # ─────── Data ───────
    dataset = InstanceDataset(args.train_data_dir, args.instance_prompt, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # ─────── Optimizer ───────
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    assert trainable_params, "没有可训练参数——TODO 13 的 LoRA 是不是没接上？"
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # fp16 需要 loss scaling；bf16 / fp32 下 enabled=False，调用方式不变
    scaler = torch.amp.GradScaler(device.type, enabled=(args.mixed_precision == 'fp16'))

    print(f"Training: {len(trainable_params)} param groups, "
          f"total {sum(p.numel() for p in trainable_params)/1e6:.2f}M params")

    # ─────── Train ───────
    unet.train()
    step = 0
    pbar = tqdm(total=args.num_train_steps)
    while step < args.num_train_steps:
        for batch in loader:
            loss = train_one_step(batch, unet, vae, text_encoder, scheduler,
                                   optimizer, device, weight_dtype,
                                   autocast_dtype, scaler)
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss:.4f}")
            if step >= args.num_train_steps:
                break
    pbar.close()

    # ─────── Save ───────
    save_lora_weights(unet, args.output_dir)


if __name__ == '__main__':
    main()
