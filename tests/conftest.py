from __future__ import annotations

import json
from pathlib import Path


def write_test_json(
    path: Path,
    payload: dict,
    *,
    indent: int | None = None,
    sort_keys: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=indent, sort_keys=sort_keys)
    if indent is not None:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def write_test_jsonl(
    path: Path,
    rows: list[dict],
    *,
    sort_keys: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=sort_keys) + "\n" for row in rows),
        encoding="utf-8",
    )
