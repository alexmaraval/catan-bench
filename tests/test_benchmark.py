from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from catan_bench.benchmark import (
    GameRecord,
    collect_game_records,
    compute_elo_ratings,
    main as benchmark_main,
)
from conftest import write_test_json as _write_json, write_test_jsonl as _write_jsonl


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

    def test_collect_game_records_accepts_versioned_runs_base_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            run_dir = base_dir / "1.2.0" / "tags-dev-game-a"
            self._make_minimal_completed_run(
                run_dir,
                red_model="provider/red-model",
                blue_model="provider/blue-model",
            )

            records = collect_game_records(base_dir)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].run_dir, run_dir)

    def test_collect_game_records_uses_deterministic_path_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            run_a = base_dir / "1.2.0" / "game-01"
            run_b = base_dir / "1.2.0" / "game-02"
            self._make_minimal_completed_run(
                run_a,
                red_model="provider/red-model",
                blue_model="provider/blue-model",
            )
            self._make_minimal_completed_run(
                run_b,
                red_model="provider/red-model",
                blue_model="provider/blue-model",
            )

            # Deliberately invert mtimes: order should still follow the path names.
            os.utime(run_a, (100.0, 100.0))
            os.utime(run_b, (200.0, 200.0))

            records = collect_game_records(base_dir)

            self.assertEqual(
                [record.run_dir.name for record in records],
                ["game-01", "game-02"],
            )

    def test_compute_elo_ratings_is_invariant_to_player_order_within_game(self) -> None:
        first_order = GameRecord(
            game_id="game-a",
            run_dir=Path("run-a"),
            num_turns=10,
            player_models={"RED": "A", "BLUE": "B", "ORANGE": "C", "WHITE": "D"},
            winner_ids=["RED"],
            player_vps={"RED": 10, "BLUE": 7, "ORANGE": 6, "WHITE": 5},
            analysis=None,
        )
        reversed_order = GameRecord(
            game_id="game-a",
            run_dir=Path("run-a"),
            num_turns=10,
            player_models={"WHITE": "D", "ORANGE": "C", "BLUE": "B", "RED": "A"},
            winner_ids=["RED"],
            player_vps={"RED": 10, "BLUE": 7, "ORANGE": 6, "WHITE": 5},
            analysis=None,
        )

        first_ratings = compute_elo_ratings([first_order]).ratings
        reversed_ratings = compute_elo_ratings([reversed_order]).ratings

        self.assertEqual(first_ratings.keys(), reversed_ratings.keys())
        for model in first_ratings:
            self.assertAlmostEqual(first_ratings[model], reversed_ratings[model])

    def test_benchmark_main_accepts_base_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            self._make_minimal_completed_run(
                base_dir / "1.2.0" / "tags-dev-game-a",
                red_model="provider/red-model",
                blue_model="provider/blue-model",
                winner="RED",
            )
            self._make_minimal_completed_run(
                base_dir / "1.2.0" / "tags-dev-game-b",
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
