from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from catan_bench.dashboard import load_dashboard_snapshot


class DashboardTests(unittest.TestCase):
    def test_load_dashboard_snapshot_reads_live_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            self._write_json(
                run_dir / "metadata.json",
                {
                    "game_id": "live-game",
                    "player_ids": ["RED", "BLUE"],
                    "player_adapter_types": {"RED": "LLMPlayer", "BLUE": "RandomLegalPlayer"},
                },
            )
            self._write_jsonl(
                run_dir / "public_history.jsonl",
                [
                    {
                        "kind": "trade_offered",
                        "payload": {
                            "offering_player_id": "RED",
                            "offer": {"WOOD": 1},
                            "request": {"BRICK": 1},
                        },
                        "turn_index": 2,
                        "phase": "play_turn",
                        "decision_index": 4,
                        "actor_player_id": "RED",
                    }
                ],
            )
            self._write_jsonl(
                run_dir / "players" / "RED" / "private_history.jsonl",
                [
                    {
                        "kind": "player_decision",
                        "payload": {
                            "decision_prompt": "Offer a trade.",
                            "action": {"action_type": "OFFER_TRADE", "payload": {}},
                        },
                        "turn_index": 2,
                        "phase": "play_turn",
                        "decision_index": 4,
                        "actor_player_id": "RED",
                    }
                ],
            )
            self._write_jsonl(run_dir / "players" / "BLUE" / "private_history.jsonl", [])
            self._write_json(run_dir / "players" / "RED" / "memory.json", {"memory": {"plan": "trade"}})
            self._write_json(run_dir / "players" / "BLUE" / "memory.json", {"memory": None})
            self._write_jsonl(
                run_dir / "players" / "RED" / "prompt_trace.jsonl",
                [
                    {
                        "player_id": "RED",
                        "turn_index": 2,
                        "phase": "play_turn",
                        "decision_index": 4,
                        "stage": "act",
                        "model": "fake-model",
                        "temperature": 0.2,
                        "attempts": [],
                    }
                ],
            )
            self._write_jsonl(run_dir / "players" / "BLUE" / "prompt_trace.jsonl", [])

            snapshot = load_dashboard_snapshot(run_dir)

            self.assertEqual(snapshot.player_ids, ("RED", "BLUE"))
            self.assertEqual(snapshot.memories_by_player["RED"], {"plan": "trade"})
            self.assertIsNone(snapshot.result)
            self.assertEqual(len(snapshot.public_timeline), 1)
            self.assertEqual(snapshot.public_timeline[0].title, "RED · Trade Offered")
            self.assertEqual(len(snapshot.player_timelines["RED"]), 2)
            self.assertEqual(len(snapshot.prompt_traces_by_player["RED"]), 1)

    def test_load_dashboard_snapshot_discovers_players_without_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            self._write_json(run_dir / "players" / "WHITE" / "memory.json", {"memory": None})
            self._write_json(run_dir / "players" / "ORANGE" / "memory.json", {"memory": {"note": "x"}})

            snapshot = load_dashboard_snapshot(run_dir)

            self.assertEqual(snapshot.player_ids, ("ORANGE", "WHITE"))
            self.assertEqual(snapshot.memories_by_player["ORANGE"], {"note": "x"})
            self.assertEqual(snapshot.public_timeline, [])

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def _write_jsonl(cls, path: Path, payloads: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(payload, sort_keys=True) + "\n" for payload in payloads),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
