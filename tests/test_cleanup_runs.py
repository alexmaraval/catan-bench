from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from catan_bench.cleanup_runs import (
    cleanup_incomplete_run_directories,
    discover_incomplete_run_directories,
    is_finished_run_directory,
    is_run_directory,
)


class CleanupRunsTests(unittest.TestCase):
    @staticmethod
    def _make_run_dir(path: Path, *, finished: bool) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "metadata.json").write_text("{}\n", encoding="utf-8")
        (path / "public_history.jsonl").write_text("", encoding="utf-8")
        (path / "public_state_trace.jsonl").write_text("", encoding="utf-8")
        if finished:
            (path / "result.json").write_text("{}\n", encoding="utf-8")

    def test_discover_incomplete_run_directories_finds_only_unfinished_runs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            finished = base / "finished-run"
            unfinished = base / "unfinished-run"
            not_a_run = base / "notes"
            self._make_run_dir(finished, finished=True)
            self._make_run_dir(unfinished, finished=False)
            not_a_run.mkdir()

            discovered = discover_incomplete_run_directories(base)

            self.assertEqual(discovered, (unfinished,))
            self.assertTrue(is_run_directory(unfinished))
            self.assertFalse(is_finished_run_directory(unfinished))
            self.assertTrue(is_finished_run_directory(finished))

    def test_cleanup_incomplete_run_directories_deletes_only_unfinished_runs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            finished = base / "finished-run"
            unfinished = base / "unfinished-run"
            self._make_run_dir(finished, finished=True)
            self._make_run_dir(unfinished, finished=False)

            removed = cleanup_incomplete_run_directories(base, delete=True)

            self.assertEqual(removed, (unfinished,))
            self.assertFalse(unfinished.exists())
            self.assertTrue(finished.exists())

    def test_discover_incomplete_run_directories_accepts_single_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "unfinished-run"
            self._make_run_dir(run_dir, finished=False)

            discovered = discover_incomplete_run_directories(run_dir)

            self.assertEqual(discovered, (run_dir,))

    def test_discover_incomplete_run_directories_walks_version_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            unfinished = base / "1.1.1" / "tags-unstable-run"
            finished = base / "1.1.1" / "tags-finished-run"
            self._make_run_dir(unfinished, finished=False)
            self._make_run_dir(finished, finished=True)

            discovered = discover_incomplete_run_directories(base)

            self.assertEqual(discovered, (unfinished,))


if __name__ == "__main__":
    unittest.main()
