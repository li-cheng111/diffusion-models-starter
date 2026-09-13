"""统一解析仓库、数据集、checkpoint 和缓存路径。"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = REPO_ROOT / ".local"
DATA_ROOT = LOCAL_ROOT / "datasets"
CHECKPOINT_ROOT = LOCAL_ROOT / "checkpoints"
HF_CACHE_ROOT = LOCAL_ROOT / "hf_cache"
TEMPORARY_RUN_ROOT = LOCAL_ROOT / "temporary_runs"


def project_root(name: str) -> Path:
    """Return a project directory by its import-safe name."""

    path = REPO_ROOT / "projects" / name
    if not path.is_dir():
        raise FileNotFoundError(f"Unknown project directory: {path}")
    return path
