"""Project 4: FID 评估（NFE-FID 曲线）

对训好的 FM 模型扫描不同 NFE / CFG scale，计算 FID，输出 JSON + Pareto 曲线。

用法：
    # 扫 NFE（无 CFG）
    python eval_fid.py --model checkpoints/fm/latest.pt --nfe 4 8 16 32 50

    # 扫 CFG scale（固定 NFE）
    python eval_fid.py --model checkpoints/fm/latest.pt --nfe 20 --cfg 1 2 3 5 7.5

依赖：
    pip install torchmetrics[image]

注意：
- FID 数值只在**同一份实现、同一 num_samples**下可比。别拿本脚本的数
  和论文里的数直接比较，也别混用不同 num_samples 的结果画同一条曲线。
- 5000 样本是教学折中；论文一般用 50000。样本数越少 FID 越高且方差越大。
- NFE 口径：Euler 每步 1 次 forward。开 CFG 后每步要跑 cond + uncond，
  实际网络前向次数翻倍——本脚本按论文惯例记 NFE = 积分步数，
  但在结果里同时给出 `forward_passes` 方便你诚实地做对比。
"""

import argparse
import json
from pathlib import Path

import torch

from train import build_model
from sample import euler_sample, euler_sample_cfg


@torch.no_grad()
def generate(model, n_samples, batch_size, nfe, num_classes, cfg_scale, device):
    """按 batch 生成 n_samples 张图，类别均匀铺满 0..num_classes-1，返回 uint8 tensor."""
    out = []
    done = 0
    while done < n_samples:
        bs = min(batch_size, n_samples - done)
        # 类别轮转，保证每类样本数大致相同
        label = (done // max(bs, 1)) % num_classes
        if cfg_scale and cfg_scale > 0:
            x = euler_sample_cfg(model, bs, nfe, num_classes, label, cfg_scale, device=device)
        else:
            x = euler_sample(model, bs, nfe, num_classes, label, device=device)
        x = ((x.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
        out.append(x.cpu())
        done += bs
        print(f"    generated {done}/{n_samples}", end='\r')
    print()
    return torch.cat(out, 0)


def compute_fid(fake_uint8, data_root, num_samples, batch_size, device):
    """用 torchmetrics 算 FID（真实样本取 CIFAR-10 训练集）."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchvision import datasets, transforms

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    for i in range(0, fake_uint8.shape[0], batch_size):
        fid.update(fake_uint8[i:i + batch_size].to(device), real=False)

    ds = datasets.CIFAR10(data_root, train=True, download=True,
                          transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                         num_workers=4)
    n_real = 0
    for x, _ in loader:
        if n_real >= num_samples:
            break
        fid.update((x * 255).to(torch.uint8).to(device), real=True)
        n_real += x.shape[0]

    return fid.compute().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='checkpoint 路径')
    parser.add_argument('--nfe', type=int, nargs='+', default=[4, 8, 16, 32, 50])
    parser.add_argument('--cfg', type=float, nargs='+', default=[0.0],
                        help='CFG scale；0 表示不开 CFG')
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=125)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output', type=str, default='results/fid.json')
    parser.add_argument('--no_ema', action='store_true',
                        help='用原始权重而非 EMA（默认用 EMA）')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.model, map_location=device)
    cfg = ckpt['cfg']
    model = build_model(cfg).to(device)
    state = ckpt['model'] if args.no_ema else ckpt.get('ema', ckpt['model'])
    model.load_state_dict(state)
    model.eval()
    num_classes = cfg['model']['num_classes']
    print(f"[loaded] {args.model} (step {ckpt.get('step', '?')}, "
          f"{'model' if args.no_ema else 'ema'} weights)")

    results = []
    for nfe in args.nfe:
        for cfg_scale in args.cfg:
            print(f"\n[eval] NFE={nfe}, CFG={cfg_scale}")
            fake = generate(model, args.num_samples, args.batch_size,
                            nfe, num_classes, cfg_scale, device)
            fid = compute_fid(fake, args.data_root, args.num_samples,
                              args.batch_size, device)
            row = {
                'nfe': nfe,
                'cfg_scale': cfg_scale,
                # 开 CFG 后每个积分步要跑 cond + uncond 两次
                'forward_passes': nfe * (2 if cfg_scale and cfg_scale > 0 else 1),
                'fid': fid,
                'num_samples': args.num_samples,
                'weights': 'model' if args.no_ema else 'ema',
            }
            print(f"  FID = {fid:.3f}  (forward passes = {row['forward_passes']})")
            results.append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {out}")

    print("\n" + "=" * 52)
    print(f"{'NFE':>6} {'CFG':>7} {'fwd':>6} {'FID':>10}")
    print("=" * 52)
    for r in sorted(results, key=lambda r: (r['cfg_scale'], r['nfe'])):
        print(f"{r['nfe']:>6} {r['cfg_scale']:>7} {r['forward_passes']:>6} {r['fid']:>10.3f}")


if __name__ == '__main__':
    main()
