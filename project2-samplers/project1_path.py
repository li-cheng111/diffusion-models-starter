"""Locate Project 1 and put it at the front of ``sys.path``.

This repository uses a monorepo layout: Project 1 remains at the Git root and
Project 2 lives in ``project2-samplers/``. The parent directory is therefore
checked first. The original sibling-repository discovery rules remain as
fallbacks so this directory can still be copied into a Classroom repository.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REQUIRED = ("schedule.py", "dataset.py", "model")
_REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CANDIDATES = (
    "..",  # selected monorepo layout
    "../diffusion-project1-ddpm",
    "../project1_ddpm",
    "./project1_ddpm",
)


def _is_project1(path: Path) -> bool:
    return path.is_dir() and all((path / name).exists() for name in REQUIRED)


def find_project1(explicit: str | None = None) -> Path:
    """Return the Project 1 source directory or raise ``FileNotFoundError``."""

    tried: list[str] = []
    for candidate in (explicit, os.environ.get("PROJECT1_PATH")):
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        tried.append(str(path))
        if _is_project1(path):
            return path

    for candidate in DEFAULT_CANDIDATES:
        path = (_REPO_ROOT / candidate).resolve()
        tried.append(str(path))
        if _is_project1(path):
            return path

    for pattern in ("diffusion-ddpm-*", "project1-ddpm-*", "diffusion-project1-*"):
        for path in sorted(_REPO_ROOT.parent.glob(pattern)):
            resolved = path.resolve()
            tried.append(str(resolved))
            if _is_project1(resolved):
                return resolved

    raise FileNotFoundError(
        "Cannot find Project 1. Pass --project1_path or set PROJECT1_PATH.\n"
        f"The directory must contain: {', '.join(REQUIRED)}\n"
        "Tried:\n  " + "\n  ".join(tried)
    )


def add_project1_to_path(explicit: str | None = None) -> Path:
    """Add Project 1 to ``sys.path`` and return its resolved directory."""

    path = find_project1(explicit)
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)
    return path


if __name__ == "__main__":
    print(f"Project 1 directory: {find_project1()}")
