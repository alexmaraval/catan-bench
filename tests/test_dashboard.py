from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from catan_bench.dashboard import (
    build_board_svg,
    build_player_timelines,
    discover_run_directories,
    load_dashboard_snapshot,
    _jump_history_to_next_turn,
    _jump_history_to_previous_turn,
    _turn_index_for_cursor,
)


class DashboardTests(unittest.TestCase):
    def test_load_dashboard_snapshot_reads_simplified_run_artifacts(self) -> None:
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
                        "history_index": 1,
                        "kind": "trade_offered",
                        "payload": {"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
                        "turn_index": 2,
                        "phase": "play_turn",
                        "decision_index": 4,
                        "actor_player_id": "RED",
                    }
                ],
            )
            self._write_jsonl(
                run_dir / "public_state_trace.jsonl",
                [
                    {
                        "history_index": 0,
                        "turn_index": 0,
                        "phase": "initial",
                        "public_state": {"turn": {"turn_player_id": "RED"}},
                    },
                    {
                        "history_index": 1,
                        "turn_index": 2,
                        "phase": "play_turn",
                        "decision_index": 4,
                        "public_state": {"turn": {"turn_player_id": "RED"}},
                    },
                ],
            )
            self._write_jsonl(
                run_dir / "players" / "RED" / "memory_trace.jsonl",
                [
                    {
                        "player_id": "RED",
                        "history_index": 1,
                        "turn_index": 2,
                        "phase": "play_turn",
                        "decision_index": 4,
                        "stage": "turn_start",
                        "memory": {"long_term": {"goal": "trade"}, "short_term": {"plan": "offer"}},
                    }
                ],
            )
            self._write_jsonl(run_dir / "players" / "BLUE" / "memory_trace.jsonl", [])
            self._write_jsonl(
                run_dir / "players" / "RED" / "prompt_trace.jsonl",
                [
                    {
                        "player_id": "RED",
                        "history_index": 1,
                        "turn_index": 2,
                        "phase": "play_turn",
                        "decision_index": 4,
                        "stage": "choose_action",
                        "model": "fake-model",
                        "temperature": 0.2,
                        "attempts": [],
                    }
                ],
            )
            self._write_jsonl(run_dir / "players" / "BLUE" / "prompt_trace.jsonl", [])

            snapshot = load_dashboard_snapshot(run_dir)

            self.assertEqual(snapshot.player_ids, ("RED", "BLUE"))
            self.assertEqual(snapshot.max_history_index, 1)
            self.assertEqual(snapshot.memory_traces_by_player["RED"][0].memory.long_term, {"goal": "trade"})
            self.assertEqual(snapshot.prompt_traces_by_player["RED"][0].stage, "choose_action")
            self.assertEqual(snapshot.public_events[0].kind, "trade_offered")

    def test_load_dashboard_snapshot_discovers_players_without_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            self._write_json(run_dir / "players" / "WHITE" / "memory.json", {"memory": {"long_term": None, "short_term": None}})
            self._write_json(run_dir / "players" / "ORANGE" / "memory.json", {"memory": {"long_term": {"note": "x"}, "short_term": None}})

            snapshot = load_dashboard_snapshot(run_dir)

            self.assertEqual(snapshot.player_ids, ("ORANGE", "WHITE"))
            self.assertEqual(snapshot.public_events, ())

    def test_discover_run_directories_returns_newest_child_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            older = base_dir / "older-run"
            newer = base_dir / "newer-run"
            older.mkdir()
            newer.mkdir()
            self._write_json(older / "metadata.json", {"game_id": "older"})
            self._write_json(newer / "metadata.json", {"game_id": "newer"})
            older_history = older / "public_history.jsonl"
            newer_history = newer / "public_history.jsonl"
            self._write_jsonl(older_history, [])
            self._write_jsonl(newer_history, [])
            os.utime(older_history, (1000, 1000))
            os.utime(newer_history, (2000, 2000))

            runs = discover_run_directories(base_dir)

            self.assertEqual(runs, (newer, older))

    def test_build_player_timelines_groups_owner_trade_discussion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            self._write_json(
                run_dir / "metadata.json",
                {
                    "game_id": "trade-game",
                    "player_ids": ["RED", "BLUE"],
                },
            )
            self._write_jsonl(
                run_dir / "public_history.jsonl",
                [
                    {
                        "history_index": 1,
                        "kind": "trade_chat_opened",
                        "payload": {
                            "owner_player_id": "RED",
                            "requested_resources": {"BRICK": 1},
                            "attempt_index": 1,
                        },
                        "turn_index": 4,
                        "phase": "play_turn",
                        "decision_index": 8,
                        "actor_player_id": "RED",
                    },
                    {
                        "history_index": 2,
                        "kind": "trade_chat_message",
                        "payload": {
                            "owner_player_id": "RED",
                            "speaker_player_id": "BLUE",
                            "message": "I can trade.",
                            "offer": {"WOOD": 1},
                            "request": {"BRICK": 1},
                            "attempt_index": 1,
                        },
                        "turn_index": 4,
                        "phase": "play_turn",
                        "decision_index": 8,
                        "actor_player_id": "BLUE",
                    },
                    {
                        "history_index": 3,
                        "kind": "trade_chat_quote_selected",
                        "payload": {
                            "owner_player_id": "RED",
                            "selected_player_id": "BLUE",
                            "offer": {"WOOD": 1},
                            "request": {"BRICK": 1},
                            "attempt_index": 1,
                        },
                        "turn_index": 4,
                        "phase": "play_turn",
                        "decision_index": 8,
                        "actor_player_id": "RED",
                    },
                    {
                        "history_index": 4,
                        "kind": "trade_chat_closed",
                        "payload": {
                            "owner_player_id": "RED",
                            "selected_player_id": "BLUE",
                            "outcome": "selected",
                            "attempt_index": 1,
                        },
                        "turn_index": 4,
                        "phase": "play_turn",
                        "decision_index": 8,
                        "actor_player_id": "RED",
                    },
                    {
                        "history_index": 5,
                        "kind": "trade_confirmed",
                        "payload": {
                            "offering_player_id": "RED",
                            "accepting_player_id": "BLUE",
                            "offer": {"WOOD": 1},
                            "request": {"BRICK": 1},
                        },
                        "turn_index": 4,
                        "phase": "decide_acceptees",
                        "decision_index": 9,
                        "actor_player_id": "RED",
                    },
                ],
            )
            self._write_jsonl(
                run_dir / "public_state_trace.jsonl",
                [
                    {
                        "history_index": 5,
                        "turn_index": 4,
                        "phase": "decide_acceptees",
                        "decision_index": 9,
                        "public_state": {"turn": {"turn_player_id": "RED"}},
                    }
                ],
            )
            self._write_jsonl(
                run_dir / "players" / "RED" / "memory_trace.jsonl",
                [
                    {
                        "player_id": "RED",
                        "history_index": 5,
                        "turn_index": 4,
                        "phase": "play_turn",
                        "decision_index": 9,
                        "stage": "turn_end",
                        "memory": {"long_term": {"goal": "trade"}, "short_term": None},
                    }
                ],
            )
            self._write_jsonl(run_dir / "players" / "BLUE" / "memory_trace.jsonl", [])
            self._write_jsonl(run_dir / "players" / "RED" / "prompt_trace.jsonl", [])
            self._write_jsonl(run_dir / "players" / "BLUE" / "prompt_trace.jsonl", [])

            snapshot = load_dashboard_snapshot(run_dir)
            timelines = build_player_timelines(snapshot, cursor=5)

            red_turns = timelines["RED"]
            self.assertEqual(len(red_turns), 1)
            self.assertEqual(len(red_turns[0].trade_artifacts), 1)
            self.assertEqual(
                [event.kind for event in red_turns[0].trade_artifacts[0].events],
                [
                    "trade_chat_opened",
                    "trade_chat_message",
                    "trade_chat_quote_selected",
                    "trade_chat_closed",
                    "trade_confirmed",
                ],
            )

    def test_build_board_svg_renders_tiles_roads_and_buildings(self) -> None:
        svg = build_board_svg(
            {
                "robber_coordinate": [0, 0, 0],
                "tiles": [
                    {
                        "coordinate": [0, 0, 0],
                        "tile": {"id": 0, "type": "RESOURCE_TILE", "resource": "WOOD", "number": 8},
                    }
                ],
                "nodes": {
                    "1": {
                        "id": 1,
                        "tile_coordinate": [0, 0, 0],
                        "direction": "NORTH",
                        "building": "SETTLEMENT",
                        "color": "RED",
                    },
                    "2": {
                        "id": 2,
                        "tile_coordinate": [0, 0, 0],
                        "direction": "NORTHEAST",
                        "building": None,
                        "color": None,
                    },
                },
                "edges": [
                    {
                        "id": [1, 2],
                        "tile_coordinate": [0, 0, 0],
                        "direction": "NORTHEAST",
                        "color": "BLUE",
                    }
                ],
            },
            height=240,
        )

        self.assertIn("<svg", svg)
        self.assertIn("🥷", svg)
        self.assertIn("polygon", svg)
        self.assertIn("line", svg)
        self.assertIn("circle", svg)

    def test_turn_jump_helpers_follow_turn_markers(self) -> None:
        turn_markers = ((0, 0), (1, 3), (2, 7), (3, 12))

        self.assertEqual(_turn_index_for_cursor(0, turn_markers), 0)
        self.assertEqual(_turn_index_for_cursor(5, turn_markers), 1)
        self.assertEqual(_turn_index_for_cursor(12, turn_markers), 3)
        self.assertEqual(_jump_history_to_previous_turn(7, turn_markers), 3)
        self.assertEqual(_jump_history_to_previous_turn(2, turn_markers), 0)
        self.assertEqual(
            _jump_history_to_next_turn(4, turn_markers, max_history_index=12),
            7,
        )
        self.assertEqual(
            _jump_history_to_next_turn(12, turn_markers, max_history_index=12),
            12,
        )

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
