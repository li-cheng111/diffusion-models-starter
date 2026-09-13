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
            errors.append(f"tracked checkpoint-like file: {path.relative_to(REPO_ROOT)}")

    if errors:
        print("仓库检查失败：")
        for error in errors:
            print(f"- {error}")
        return 1
    print("仓库结构检查通过：五个项目存在，未发现嵌套 Git 或违规 checkpoint。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
