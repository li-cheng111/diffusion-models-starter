"""跨项目视觉评估约定，不负责下载数据。"""

from __future__ import annotations


DEFAULT_FID_COUNT = 5_000


def fixed_indices(count: int = DEFAULT_FID_COUNT) -> list[int]:
    """Return the deterministic prefix used by the CIFAR-10 FID protocol."""

    if count <= 0:
        raise ValueError("count must be positive")
    return list(range(count))
