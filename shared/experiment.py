"""统一实验结果 JSON 的最小元数据接口。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1


def metadata(project: str, experiment: str, seed: int, **extra: Any) -> dict[str, Any]:
    """Build a serializable record header shared by all project results."""

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project": project,
        "experiment": experiment,
        "seed": int(seed),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    result.update(extra)
    return result


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a UTF-8, indented JSON result and return its resolved path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target
