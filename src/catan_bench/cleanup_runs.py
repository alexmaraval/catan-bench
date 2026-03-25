from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


RUN_MARKER_FILES = ("metadata.json", "public_history.jsonl", "public_state_trace.jsonl")


def is_run_directory(path: Path) -> bool:
    return path.is_dir() and all((path / file_name).exists() for file_name in RUN_MARKER_FILES)


def is_finished_run_directory(path: Path) -> bool:
    return is_run_directory(path) and (path / "result.json").exists()


def discover_incomplete_run_directories(base_run_dir: str | Path) -> tuple[Path, ...]:
    base_path = Path(base_run_dir)
    if is_run_directory(base_path):
        return () if is_finished_run_directory(base_path) else (base_path,)
    if not base_path.is_dir():
        return ()

    candidates = [
        path
        for path in base_path.iterdir()
        if is_run_directory(path) and not is_finished_run_directory(path)
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return tuple(candidates)


def cleanup_incomplete_run_directories(
    base_run_dir: str | Path,
    *,
    delete: bool = False,
) -> tuple[Path, ...]:
    incomplete_runs = discover_incomplete_run_directories(base_run_dir)
    if delete:
        for run_dir in incomplete_runs:
            shutil.rmtree(run_dir)
    return incomplete_runs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List or remove unfinished run directories (missing result.json)."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="runs",
        help="Run base directory or a specific run directory. Defaults to ./runs",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually remove the unfinished run directories. Without this flag, the command is a dry run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    target = Path(args.run_dir)
    incomplete_runs = cleanup_incomplete_run_directories(target, delete=args.delete)

    if not incomplete_runs:
        print(f"No unfinished runs found under {target}.")
        return 0

    action = "Removed" if args.delete else "Would remove"
    for run_dir in incomplete_runs:
        print(f"{action}: {run_dir}")
    print(f"{action} {len(incomplete_runs)} unfinished run(s).")
    if not args.delete:
        print("Re-run with --delete to remove them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
