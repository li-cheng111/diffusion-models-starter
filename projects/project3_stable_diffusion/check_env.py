"""
环境自检：确认 Project 3 需要的依赖、显存和模型下载通道都就绪。

SD 权重有 ~7 GB，跑到一半才发现装错库或下不动很浪费时间。动手前先跑：

    python check_env.py

只做轻量 HEAD 请求探测模型是否可达，**不会**下载权重。

国内网络访问 HuggingFace 不稳定时，用镜像：

    export HF_ENDPOINT=https://hf-mirror.com
    python check_env.py
"""

import argparse
import importlib
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    # AutoDL is UTF-8; Windows local consoles may still default to GBK.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# (import 名, pip 名, 是哪个任务需要的)
PACKAGES = [
    ('torch', 'torch', '全部'),
    ('diffusers', 'diffusers', '全部'),
    ('transformers', 'transformers', '全部'),
    ('matplotlib', 'matplotlib', '任务 B'),
    ('accelerate', 'accelerate', '任务 D'),
    ('peft', 'peft', '任务 D（LoRA）'),
    ('cv2', 'opencv-python', '任务 E / ControlNet demo'),
]

DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
CONTROLNET_ID = "lllyasviel/sd-controlnet-canny"


def check_packages():
    missing = []
    for mod, pip_name, need_for in PACKAGES:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, '__version__', '?')
            print(f"   ✅ {pip_name:<16} {ver:<12} ({need_for})")
        except ImportError:
            missing.append((pip_name, need_for))
            print(f"   ❌ {pip_name:<16} {'未安装':<12} ({need_for})")
    if missing:
        print("\n   安装缺失的包：")
        print("     pip install " + " ".join(p for p, _ in missing))
    return not missing


def check_device():
    try:
        import torch
    except ImportError:
        print("   ⏭  跳过（torch 没装）")
        return False

    if not torch.cuda.is_available():
        print("   ⚠️  没有可用 CUDA 设备。SD 推理在 CPU 上一张图要几分钟，"
              "任务 D 基本跑不动。\n"
              "      没显卡的话用 Colab / AutoDL，见 README「环境」。")
        return False

    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"   ✅ {name}，显存 {vram:.1f} GB")

    if vram < 8:
        print("   ⚠️  显存 < 8 GB：推理需要开 enable_attention_slicing()，"
              "任务 D（LoRA）很可能 OOM")
    elif vram < 12:
        print("   ℹ️  显存 8-12 GB：推理没问题；任务 D 需要 fp16 + "
              "gradient checkpointing")
    return True


def check_model_reachable(model_id, model_revision=DEFAULT_MODEL_REVISION):
    """只发 HEAD 请求探测配置文件，不下载权重（用标准库，不依赖 requests）."""
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
    print(f"   endpoint: {endpoint}")

    ok = True
    for repo, fname in [(model_id, 'model_index.json'),
                        (CONTROLNET_ID, 'config.json')]:
        revision = model_revision if repo == model_id and model_revision else 'main'
        url = f"{endpoint}/{repo}/resolve/{revision}/{fname}"
        try:
            req = Request(url, method='HEAD')
            with urlopen(req, timeout=15) as r:
                if r.status == 200:
                    print(f"   ✅ {repo}")
                else:
                    ok = False
                    print(f"   ❌ {repo} → HTTP {r.status}")
        except HTTPError as e:
            ok = False
            print(f"   ❌ {repo} → HTTP {e.code}")
        except (URLError, OSError) as e:
            ok = False
            print(f"   ❌ {repo} → 连不上（{e.__class__.__name__}）")

    if not ok:
        print("\n   连不上的话试试镜像：")
        print("     export HF_ENDPOINT=https://hf-mirror.com")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument('--model_revision', type=str, default=DEFAULT_MODEL_REVISION)
    parser.add_argument('--json_output', type=str, default=None)
    args = parser.parse_args()

    print("[1/3] 依赖包")
    ok_pkg = check_packages()

    print("\n[2/3] 计算设备")
    check_device()   # 没有 GPU 只是警告，不算失败

    print("\n[3/3] 模型下载通道")
    ok_net = check_model_reachable(args.model_id, args.model_revision)

    print()
    if ok_pkg and ok_net:
        print("环境就绪 ✅ 可以开始任务 A 了")
        if args.json_output:
            from pathlib import Path
            try:
                Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.json_output).write_text(json.dumps({
                    "packages_ok": True,
                    "model_reachable": True,
                    "model_id": args.model_id,
                    "model_revision": args.model_revision,
                }, indent=2) + "\n", encoding='utf-8')
            except OSError as exc:
                print(f"   ⚠️  无法写入 JSON 检查结果：{exc}")
        return 0
    print("有检查未通过 ❌ 按上面的提示处理后重跑")
    return 1


if __name__ == '__main__':
    sys.exit(main())
