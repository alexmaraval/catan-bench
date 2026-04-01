from __future__ import annotations

import json
import tempfile
import unittest
from functools import partial
from pathlib import Path

from catan_bench.replay import build_replay_timeline, export_replay_html
from conftest import write_test_jsonl

_write_jsonl = partial(write_test_jsonl, sort_keys=True)


class ReplayTests(unittest.TestCase):
    def test_build_replay_timeline_reads_public_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            self._write_jsonl(
                run_dir / "public_history.jsonl",
                [
                    {
                        "history_index": 1,
                        "kind": "trade_offered",
                        "payload": {"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
                        "turn_index": 1,
                        "phase": "play_turn",
                        "decision_index": 0,
                        "actor_player_id": "RED",
                    }
                ],
            )

            timeline = build_replay_timeline(run_dir)

            self.assertEqual(len(timeline), 1)
            self.assertEqual(timeline[0].history_index, 1)
            self.assertEqual(timeline[0].event_kind, "trade_offered")
            self.assertIn("Trade offered", timeline[0].title)

    def test_export_replay_html_writes_simple_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            self._write_jsonl(
                run_dir / "public_history.jsonl",
                [
                    {
                        "history_index": 1,
                        "kind": "dice_rolled",
                        "payload": {"result": [3, 4]},
                        "turn_index": 1,
                        "phase": "play_turn",
                        "decision_index": 0,
                        "actor_player_id": "RED",
                    }
                ],
            )

            output_path = export_replay_html(run_dir)

            self.assertTrue(output_path.exists())
            self.assertIn("catan-bench replay", output_path.read_text(encoding="utf-8"))

    @classmethod
    def _write_jsonl(cls, path: Path, payloads: list[dict]) -> None:
        _write_jsonl(path, payloads)


if __name__ == "__main__":
    unittest.main()
