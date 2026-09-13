"""检查跨项目实验结果是否包含最小可复现字段。"""

from __future__ import annotations

from collections.abc import Mapping


REQUIRED_FIELDS = {"schema_version", "project", "experiment", "seed"}


def validate_record(record: Mapping[str, object]) -> None:
    """Raise ValueError when a result record misses required metadata."""

    missing = sorted(REQUIRED_FIELDS.difference(record))
    if missing:
        raise ValueError("missing result metadata: " + ", ".join(missing))
