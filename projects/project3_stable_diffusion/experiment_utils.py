"""Small, dependency-light helpers shared by Project 3 experiments.

The project intentionally keeps all large model/data files outside Git.  These
helpers make the committed result files self-describing instead: every run can
record its seed, model revision, software stack, hardware and output hashes.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def set_seed(seed: int) -> None:
    """Seed Python, NumPy (when installed) and PyTorch."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def runtime_metadata(
    *,
    seed: int | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Return JSON-serializable metadata for a run."""
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "model_id": model_id,
        "model_revision": model_revision,
        "command": command or " ".join(sys.argv),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _version(name)
            for name in (
                "torch",
                "torchvision",
                "diffusers",
                "transformers",
                "peft",
                "accelerate",
                "safetensors",
                "matplotlib",
            )
        },
    }
    try:
        import torch

        metadata["torch_cuda"] = torch.version.cuda
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            metadata["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            metadata["gpu_memory_bytes"] = int(props.total_memory)
            metadata["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as exc:  # pragma: no cover - diagnostics must not fail a run
        metadata["torch_error"] = repr(exc)
    return metadata


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_file_hashes(metadata: dict[str, Any], paths: Iterable[str | os.PathLike[str]]) -> dict[str, Any]:
    metadata["output_sha256"] = {
        str(Path(path)): sha256_file(path)
        for path in paths
        if Path(path).is_file()
    }
    return metadata


def write_json(path: str | os.PathLike[str], payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_commit() -> str | None:
    """Return the current commit when invoked inside the monorepo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
