"""Checkpoint compatibility helpers shared by Project 2 command-line tools."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import subprocess

import torch


def repository_commit(start: Path | None = None) -> str | None:
    """Return the enclosing Git commit without changing repository state."""

    cwd = str(start or Path(__file__).resolve().parent)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def model_state_from_checkpoint(checkpoint: Mapping, use_ema: bool = True) -> Mapping:
    """Return a model state dict from flat or nested Project 1 checkpoints."""

    if use_ema and "ema" in checkpoint:
        state = checkpoint["ema"]
        if isinstance(state, Mapping) and isinstance(state.get("model"), Mapping):
            state = state["model"]
    else:
        if "model" not in checkpoint:
            raise KeyError("Checkpoint contains neither usable EMA nor model weights")
        state = checkpoint["model"]

    if not isinstance(state, Mapping):
        raise TypeError(f"Expected a state dict mapping, got {type(state).__name__}")
    return state


def load_model_weights(
    model: torch.nn.Module,
    checkpoint: Mapping,
    use_ema: bool = True,
) -> str:
    """Load weights and return the human-readable weight type."""

    model.load_state_dict(model_state_from_checkpoint(checkpoint, use_ema=use_ema))
    return "EMA" if use_ema and "ema" in checkpoint else "raw"
