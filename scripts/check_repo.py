"""仓库结构检查：防嵌套 Git、检查五个项目和大型文件策略。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTS = (
    "project1_ddpm",
    "project2_samplers",
    "project3_stable_diffusion",
    "project4_flow_matching",
    "project5_vla_action_diffusion",
)
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
ALLOWED_LORA_ADAPTER = Path(
    "projects/project3_stable_diffusion/outputs/lora/full/pytorch_lora_weights.safetensors"
)
ALLOWED_PROJECT5_CHECKPOINT = Path(
    "projects/project5_vla_action_diffusion/ckpts/model_final.pt"
)
MAX_LORA_ADAPTER_BYTES = 25 * 1024 * 1024
MAX_PROJECT5_CHECKPOINT_BYTES = 25 * 1024 * 1024


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
    )
    return [REPO_ROOT / item for item in output.decode().split("\0") if item]


def main() -> int:
    errors: list[str] = []
    projects_root = REPO_ROOT / "projects"
    for project in PROJECTS:
        path = projects_root / project
        if not path.is_dir():
            errors.append(f"missing project directory: {path.relative_to(REPO_ROOT)}")
        elif not (path / "README.md").exists():
            errors.append(f"missing project README: {path.relative_to(REPO_ROOT)}")

    for git_path in projects_root.rglob(".git") if projects_root.exists() else ():
        errors.append(f"nested git metadata: {git_path.relative_to(REPO_ROOT)}")

    for path in tracked_files():
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            relative = path.relative_to(REPO_ROOT)
            if relative == ALLOWED_PROJECT5_CHECKPOINT:
                if path.stat().st_size > MAX_PROJECT5_CHECKPOINT_BYTES:
                    errors.append(
                        f"Project 5 checkpoint exceeds 25 MiB: {relative} ({path.stat().st_size} bytes)"
                    )
            elif relative != ALLOWED_LORA_ADAPTER:
                errors.append(f"tracked checkpoint-like file: {relative}")
            elif path.stat().st_size > MAX_LORA_ADAPTER_BYTES:
                errors.append(
                    f"LoRA adapter exceeds 25 MiB: {relative} ({path.stat().st_size} bytes)"
                )

    if errors:
        print("仓库检查失败：")
        for error in errors:
            print(f"- {error}")
        return 1
    print("仓库结构检查通过：五个项目存在，未发现嵌套 Git、违规 checkpoint 或超大 LoRA adapter。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
