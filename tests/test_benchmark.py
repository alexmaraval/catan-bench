from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from catan_bench.benchmark import collect_game_records, main as benchmark_main


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class BenchmarkCliTests(unittest.TestCase):
    def _make_minimal_completed_run(
        self,
        run_dir: Path,
        *,
        red_model: str,
        blue_model: str,
        winner: str = "RED",
    ) -> None:
        _write_json(
            run_dir / "metadata.json",
            {
                "player_ids": ["RED", "BLUE"],
                "player_adapter_types": {"RED": "LLMPlayer", "BLUE": "LLMPlayer"},
            },
        )
        _write_json(
            run_dir / "result.json",
            {
                "game_id": run_dir.name,
                "winner_ids": [winner],
                "metadata": {
                    "num_turns": 10,
                    "players": {
                        "RED": {"actual_victory_points": 5},
                        "BLUE": {"actual_victory_points": 3},
                    },
                },
            },
        )
        _write_jsonl(run_dir / "public_history.jsonl", [])
        _write_jsonl(
            run_dir / "public_state_trace.jsonl",
            [
                {
                    "history_index": 0,
                    "turn_index": 0,
                    "phase": "initial",
                    "decision_index": None,
                    "public_state": {"players": {"RED": {}, "BLUE": {}}},
                }
            ],
        )
        _write_jsonl(
            run_dir / "players" / "RED" / "prompt_trace.jsonl",
            [
                {
                    "player_id": "RED",
                    "history_index": 0,
                    "turn_index": 0,
                    "phase": "play_turn",
                    "decision_index": 0,
                    "stage": "choose_action",
                    "model": red_model,
                    "temperature": 0.3,
                    "attempts": [],
                }
            ],
        )
        _write_jsonl(
            run_dir / "players" / "BLUE" / "prompt_trace.jsonl",
            [
                {
                    "player_id": "BLUE",
                    "history_index": 0,
                    "turn_index": 0,
                    "phase": "play_turn",
                    "decision_index": 0,
                    "stage": "choose_action",
                    "model": blue_model,
                    "temperature": 0.3,
                    "attempts": [],
                }
            ],
        )

    def test_collect_game_records_accepts_flat_runs_base_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            run_dir = base_dir / "0.4.0-dev-game-a"
            self._make_minimal_completed_run(
                run_dir,
                red_model="provider/red-model",
                blue_model="provider/blue-model",
            )

            records = collect_game_records(base_dir)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].run_dir, run_dir)

    def test_benchmark_main_accepts_base_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            self._make_minimal_completed_run(
                base_dir / "0.4.0-dev-game-a",
                red_model="provider/red-model",
                blue_model="provider/blue-model",
                winner="RED",
            )
            self._make_minimal_completed_run(
                base_dir / "0.4.0-dev-game-b",
                red_model="provider/red-model",
                blue_model="provider/blue-model",
                winner="BLUE",
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = benchmark_main([str(base_dir), "--json-only"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["games_scanned"], 2)
            self.assertEqual(len(payload["leaderboard"]), 2)


if __name__ == "__main__":
    unittest.main()
