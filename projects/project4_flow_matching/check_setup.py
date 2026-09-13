"""配置与模型自检：确认 config、backbone、TODO 状态都对得上。

在开跑 200K 步之前先跑一遍（几秒钟）：

    python check_setup.py                                # 检查所有 configs/
    python check_setup.py --config configs/cifar10_fm.yaml

它会检查：
1. config 的键与 train.py 读取的路径是否一一对应
2. 按 config 能否成功 build_model，参数量多少
3. 模型前向 shape 对不对，null token（索引 num_classes）会不会越界
4. TODO 16/17/18 填了没

TODO 没填时显示 ⏳ 是预期的——那正是你要写的部分。
"""

import argparse
import sys
from pathlib import Path

import torch

from train import build_model, load_config, compute_fm_loss
from sample import euler_sample, euler_sample_cfg

# train.py / sample.py 实际会读的 config 路径
REQUIRED_KEYS = [
    ('data', 'root'), ('data', 'num_workers'),
    ('model', 'type'), ('model', 'num_classes'),
    ('train', 'batch_size'), ('train', 'lr'), ('train', 'weight_decay'),
    ('train', 'max_steps'), ('train', 'save_every'),
]
# 按 model.type 额外需要的键
TYPE_KEYS = {
    'dit': ['embed_dim', 'depth', 'num_heads'],
    'unet': ['base_channels'],
}


def check_config(path):
    cfg = load_config(path)
    missing = [f"{a}.{b}" for a, b in REQUIRED_KEYS
               if not isinstance(cfg.get(a), dict) or b not in cfg[a]]

    mtype = cfg.get('model', {}).get('type')
    for k in TYPE_KEYS.get(mtype, []):
        if k not in cfg.get('model', {}):
            missing.append(f"model.{k}  (type={mtype} 需要)")

    if missing:
        print(f"   ❌ {path.name}: 缺少 {', '.join(missing)}")
        return None
    print(f"   ✅ {path.name}: 键齐全 (type={mtype})")
    return cfg


def check_model(cfg, name):
    try:
        model = build_model(cfg)
    except Exception as e:
        print(f"   ❌ {name}: build_model 失败 → {type(e).__name__}: {e}")
        return None

    n = sum(p.numel() for p in model.parameters()) / 1e6
    num_classes = cfg['model']['num_classes']

    B = 2
    x = torch.randn(B, 3, 32, 32)
    t = torch.rand(B)
    # 关键：把 null token（索引 num_classes）也喂进去，越界的话这里就会炸
    y = torch.tensor([0, num_classes])

    try:
        with torch.no_grad():
            v = model(x, t, y)
    except Exception as e:
        print(f"   ❌ {name}: 前向失败 → {type(e).__name__}: {e}")
        print(f"      （y 里含 null token index={num_classes}，"
              f"embedding 大小要是 num_classes+1）")
        return None

    if v.shape != x.shape:
        print(f"   ❌ {name}: 输出 shape {tuple(v.shape)} != 输入 {tuple(x.shape)}")
        return None

    print(f"   ✅ {name}: {n:.2f}M params, 前向 OK, null token 索引 {num_classes} 可用")
    return model


def check_todos(model, cfg):
    num_classes = cfg['model']['num_classes']
    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, num_classes, (2,))

    todos = [
        ('TODO 16 (compute_fm_loss)',
         lambda: compute_fm_loss(model, x, y, 0.1, num_classes)),
        ('TODO 17 (euler_sample)',
         lambda: euler_sample(model, 2, 4, num_classes, 0, device='cpu')),
        ('TODO 18 (euler_sample_cfg)',
         lambda: euler_sample_cfg(model, 2, 4, num_classes, 0, 3.0, device='cpu')),
    ]

    all_done = True
    for label, fn in todos:
        try:
            out = fn()
            if torch.is_tensor(out) and out.numel() > 0:
                extra = f"loss={out.item():.4f}" if out.ndim == 0 else f"shape={tuple(out.shape)}"
                print(f"   ✅ {label}: {extra}")
            else:
                print(f"   ⚠️  {label}: 返回了 {type(out).__name__}，检查一下返回值")
                all_done = False
        except NotImplementedError:
            print(f"   ⏳ {label}: 未完成（预期）")
            all_done = False
        except Exception as e:
            print(f"   ❌ {label}: {type(e).__name__}: {e}")
            all_done = False
    return all_done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='只检查这一个配置；默认检查 configs/ 下全部')
    args = parser.parse_args()

    paths = ([Path(args.config)] if args.config
             else sorted(Path('configs').glob('*.yaml')))
    if not paths:
        print("configs/ 下没找到 yaml")
        return 1

    print("[1/3] config 键检查")
    cfgs = [(p, check_config(p)) for p in paths]
    if any(c is None for _, c in cfgs):
        return 1

    print("\n[2/3] 按 config 构建模型并前向")
    models = [(p, cfg, check_model(cfg, p.name)) for p, cfg in cfgs]
    if any(m is None for _, _, m in models):
        return 1

    print("\n[3/3] TODO 状态（用第一个 config 的模型试跑）")
    _, cfg0, model0 = models[0]
    done = check_todos(model0, cfg0)

    print()
    if done:
        print("全部就绪 ✅ 可以开跑训练了")
    else:
        print("配置与模型 OK ✅，TODO 还没写完 —— 去动手吧")
    return 0


if __name__ == '__main__':
    sys.exit(main())
