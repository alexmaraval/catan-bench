from __future__ import annotations

from pathlib import Path


def iter_run_directory_candidates(base_dir: str | Path) -> tuple[Path, ...]:
    """Return direct children and one extra nested level under *base_dir*."""
    base_path = Path(base_dir)
    if not base_path.exists() or not base_path.is_dir():
        return ()

    candidates: list[Path] = []
    seen: set[Path] = set()
    for child in sorted(base_path.iterdir()):
        if not child.is_dir():
            continue
        if child not in seen:
            candidates.append(child)
            seen.add(child)
        for grandchild in sorted(child.iterdir()):
            if not grandchild.is_dir() or grandchild in seen:
                continue
            candidates.append(grandchild)
            seen.add(grandchild)
    return tuple(candidates)
