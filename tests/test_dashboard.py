from __future__ import annotations

import os
import tempfile
import unittest
from functools import partial
from pathlib import Path

from conftest import write_test_json, write_test_jsonl

from catan_bench.dashboard import (
    DashboardSnapshot,
    _colorize_player_mentions_html,
    _event_body,
    _infer_turn_owner_player_id,
    _public_chat_panel_html,
    _render_public_chat_panel,
    _render_analysis_tab,
    _render_player_summary_table,
    build_board_svg,
    build_player_timelines,
    discover_run_directories,
    load_dashboard_snapshot,
    _recent_turn_event_sections,
    _jump_history_to_next_turn,
    _jump_history_to_previous_turn,
    _turn_event_digest_body,
    _turn_index_for_cursor,
)
from catan_bench.schemas import Event, PromptTrace, PublicStateSnapshot

_write_json = partial(write_test_json, indent=2, sort_keys=True)
_write_jsonl = partial(write_test_jsonl, sort_keys=True)


class DashboardTests(unittest.TestCase):
    def test_render_analysis_tab_supports_live_incomplete_run(self) -> None:
        class FakeColumn:
            def __init__(self) -> None:
                self.metrics: list[tuple[str, object]] = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def metric(self, label: str, value: object) -> None:
                self.metrics.append((label, value))

            def altair_chart(self, *args, **kwargs) -> None:
                return None

            def markdown(self, *args, **kwargs) -> None:
                return None

            def info(self, *args, **kwargs) -> None:
                return None

        class FakeStreamlit:
            def __init__(self) -> None:
                self.info_calls: list[str] = []
                self.caption_calls: list[str] = []
                self.subheader_calls: list[str] = []

            def info(self, body: str) -> None:
                self.info_calls.append(body)

            def caption(self, body: str) -> None:
                self.caption_calls.append(body)

            def subheader(self, body: str) -> None:
                self.subheader_calls.append(body)

            def columns(self, n: int):
                return [FakeColumn() for _ in range(n)]

            def metric(self, label: str, value: object) -> None:
                return None

            def button(self, *args, **kwargs) -> bool:
                return False

            def dataframe(self, *args, **kwargs) -> None:
                return None

            def altair_chart(self, *args, **kwargs) -> None:
                return None

            def plotly_chart(self, *args, **kwargs) -> None:
                return None

            def markdown(self, *args, **kwargs) -> None:
                return None

            def expander(self, *args, **kwargs):
                class _Dummy:
                    def __enter__(self_inner):
                        return self_inner

                    def __exit__(self_inner, exc_type, exc, tb):
                        return False

                return _Dummy()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            self._write_json(
                run_dir / "metadata.json",
                {
                    "game_id": "live-game",
                    "player_ids": ["RED", "BLUE"],
                },
            )
            self._write_jsonl(
                run_dir / "public_history.jsonl",
                [
                    {
                        "history_index": 1,
                        "turn_index": 1,
                        "phase": "play_turn",
                        "kind": "public_chat_message",
                        "payload": {
                            "speaker_player_id": "RED",
                            "message": "Watching BLUE.",
                            "target_player_id": "BLUE",
                        },
                        "actor_player_id": "RED",
                    }
                ],
            )
            self._write_jsonl(
                run_dir / "public_state_trace.jsonl",
                [
                    {
                        "history_index": 1,
                        "turn_index": 1,
                        "phase": "play_turn",
                        "decision_index": 1,
                        "public_state": {
                            "players": {
                                "RED": {
                                    "visible_victory_points": 2,
                                    "dev_victory_points": 0,
                                    "longest_road_length": 1,
                                    "played_knights": 0,
                                },
                                "BLUE": {
                                    "visible_victory_points": 1,
                                    "dev_victory_points": 0,
                                    "longest_road_length": 1,
                                    "played_knights": 0,
                                },
                            },
                            "board": {
                                "tiles": {},
                                "nodes": {},
                                "edges": [],
                                "adjacent_tiles": {},
                                "robber_coordinate": [0, 0, 0],
                            },
                        },
                    }
                ],
            )
            self._write_jsonl(run_dir / "players" / "RED" / "prompt_trace.jsonl", [])
            self._write_jsonl(run_dir / "players" / "BLUE" / "prompt_trace.jsonl", [])
            self._write_jsonl(run_dir / "players" / "RED" / "memory_trace.jsonl", [])
            self._write_jsonl(run_dir / "players" / "BLUE" / "memory_trace.jsonl", [])

            snapshot = load_dashboard_snapshot(run_dir)
            st = FakeStreamlit()

            _render_analysis_tab(st, snapshot)

            self.assertTrue(
                any(
                    "live provisional analysis" in body.lower()
                    for body in st.info_calls
                )
            )
            self.assertTrue(
                any("provisional" in body.lower() for body in st.caption_calls)
            )

    def test_event_body_formats_public_chat_message(self) -> None:
        body = _event_body(
            Event(
                kind="public_chat_message",
                payload={
                    "speaker_player_id": "RED",
                    "message": "BLUE, don't trade them ore.",
                    "target_player_id": "BLUE",
                },
                history_index=10,
                turn_index=2,
                phase="play_turn",
                decision_index=3,
                actor_player_id="RED",
            )
        )

        self.assertEqual(body, "To BLUE (public): BLUE, don't trade them ore.")

    def test_public_chat_panel_html_groups_messages_by_turn(self) -> None:
        html = _public_chat_panel_html(
            (
                Event(
                    kind="public_chat_message",
                    payload={
                        "speaker_player_id": "RED",
                        "message": "I can help on this roll.",
                    },
                    history_index=5,
                    turn_index=2,
                    phase="play_turn",
                    decision_index=1,
                    actor_player_id="RED",
                ),
                Event(
                    kind="public_chat_message",
                    payload={
                        "speaker_player_id": "BLUE",
                        "target_player_id": "RED",
                        "message": "Hold your wheat for me.",
                    },
                    history_index=8,
                    turn_index=3,
                    phase="play_turn",
                    decision_index=2,
                    actor_player_id="BLUE",
                ),
            ),
            current_turn=3,
        )

        self.assertIn("Turn 2", html)
        self.assertIn("Turn 3", html)
        self.assertIn("public-chat-group-current", html)
        self.assertIn("BLUE \u2192 RED", html)
        self.assertIn("Hold your wheat for me.", html)
        self.assertNotIn("To RED (public):", html)

    def test_render_public_chat_panel_renders_turn_sections_incrementally(self) -> None:
        class FakeStreamlit:
            def __init__(self) -> None:
                self.markdown_calls: list[tuple[str, bool]] = []
                self.caption_calls: list[str] = []
                self.container_calls: list[dict[str, object]] = []

            def markdown(self, body: str, unsafe_allow_html: bool = False) -> None:
                self.markdown_calls.append((body, unsafe_allow_html))

            def caption(self, body: str) -> None:
                self.caption_calls.append(body)

            def container(self, **kwargs):
                self.container_calls.append(kwargs)
                return self

        st = FakeStreamlit()

        _render_public_chat_panel(
            st,
            (
                Event(
                    kind="public_chat_message",
                    payload={
                        "speaker_player_id": "RED",
                        "message": "I can help on this roll.",
                    },
                    history_index=5,
                    turn_index=2,
                    phase="play_turn",
                    decision_index=1,
                    actor_player_id="RED",
                ),
                Event(
                    kind="public_chat_message",
                    payload={
                        "speaker_player_id": "BLUE",
                        "target_player_id": "RED",
                        "message": "Hold your wheat for me.",
                    },
                    history_index=8,
                    turn_index=3,
                    phase="play_turn",
                    decision_index=2,
                    actor_player_id="BLUE",
                ),
            ),
            current_turn=3,
        )

        self.assertEqual(st.caption_calls, ["2 message(s) through history 8"])
        self.assertEqual(st.markdown_calls[0], ("**📣 Table Chat**", False))
        self.assertEqual(st.container_calls, [{"height": 576, "border": True}])
        self.assertEqual(len(st.markdown_calls), 3)
        self.assertTrue(st.markdown_calls[1][1])
        self.assertIn("Turn 2", st.markdown_calls[1][0])
        self.assertIn("I can help on this roll.", st.markdown_calls[1][0])
        self.assertTrue(st.markdown_calls[2][1])
        self.assertIn("Turn 3", st.markdown_calls[2][0])
        self.assertIn("public-chat-group-current", st.markdown_calls[2][0])
        self.assertIn("BLUE → RED", st.markdown_calls[2][0])

    def test_infer_turn_owner_prefers_earliest_turn_event_actor(self) -> None:
        snapshot = DashboardSnapshot(
            run_dir=Path("/tmp/fake-run"),
            metadata={},
            player_ids=("WHITE", "BLUE"),
            public_events=(
                Event(
                    "dice_rolled",
                    {"result": [6, 4]},
                    history_index=110,
                    turn_index=18,
                    phase="play_turn",
                    decision_index=77,
                    actor_player_id="WHITE",
                ),
            ),
            public_state_snapshots=(
                PublicStateSnapshot(
                    history_index=110,
                    turn_index=18,
                    phase="play_turn",
                    decision_index=77,
                    public_state={
                        "turn": {
                            "turn_player_id": "BLUE",
                            "current_player_id": "BLUE",
                        }
                    },
                ),
            ),
            memory_traces_by_player={"WHITE": (), "BLUE": ()},
            prompt_traces_by_player={
                "WHITE": (
                    PromptTrace(
                        player_id="WHITE",
                        history_index=110,
                        turn_index=18,
                        phase="play_turn",
                        decision_index=78,
                        stage="turn_start",
                        model="fake-model",
                        temperature=0.0,
                        attempts=(),
                    ),
                ),
                "BLUE": (),
            },
            result=None,
        )

        owner = _infer_turn_owner_player_id(
            snapshot,
            turn_index=18,
            cursor=110,
            current_state=snapshot.public_state_snapshots[0],
            turn_events=snapshot.public_events,
        )

        self.assertEqual(owner, "WHITE")

    def test_render_player_summary_table_includes_army_column(self) -> None:
        class FakeStreamlit:
            def __init__(self) -> None:
                self.markdown_calls: list[tuple[str, bool]] = []
                self.caption_calls: list[str] = []

            def markdown(self, body: str, unsafe_allow_html: bool = False) -> None:
                self.markdown_calls.append((body, unsafe_allow_html))

            def caption(self, body: str) -> None:
                self.caption_calls.append(body)

        st = FakeStreamlit()

        _render_player_summary_table(
            st,
            {
                "RED": {
                    "visible_victory_points": 3,
                    "resource_card_count": 4,
                    "dev_victory_points": 2,
                    "longest_road_length": 5,
                    "played_knights": 3,
                    "has_longest_road": True,
                    "has_largest_army": True,
                }
            },
            winner_ids=["RED"],
        )

        self.assertEqual(len(st.markdown_calls), 1)
        html, unsafe = st.markdown_calls[0]
        self.assertTrue(unsafe)
        self.assertIn("<th>Dev VP</th>", html)
        self.assertIn("<th>L. Army</th>", html)
        self.assertIn("<th>L. Road</th>", html)
        self.assertIn("<th>Total VP</th>", html)
        # Board VP = visible(3) - road(2) - army(2) = -1 → but that's fine for this test data
        self.assertIn("2</td>", html)  # dev_vp
        self.assertIn("3 ⚔️</td>", html)  # army with emoji
        self.assertIn("5 🛤️</td>", html)  # road with emoji
        # Trophy only appears for an explicit recorded winner.
        self.assertIn("5 🏆</td>", html)  # winner gets trophy
        self.assertEqual(st.caption_calls, [])

    def test_render_player_summary_table_only_marks_recorded_winner(self) -> None:
        class FakeStreamlit:
            def __init__(self) -> None:
                self.markdown_calls: list[tuple[str, bool]] = []

            def markdown(self, body: str, unsafe_allow_html: bool = False) -> None:
                self.markdown_calls.append((body, unsafe_allow_html))

        st = FakeStreamlit()

        _render_player_summary_table(
            st,
            {
                "BLUE": {
                    "visible_victory_points": 2,
                    "dev_victory_points": 0,
                    "longest_road_length": 1,
                    "played_knights": 0,
                },
                "ORANGE": {
                    "visible_victory_points": 2,
                    "dev_victory_points": 0,
                    "longest_road_length": 1,
                    "played_knights": 0,
                },
                "RED": {
                    "visible_victory_points": 2,
                    "dev_victory_points": 0,
                    "longest_road_length": 1,
                    "played_knights": 0,
                },
                "WHITE": {
                    "visible_victory_points": 2,
                    "dev_victory_points": 0,
                    "longest_road_length": 1,
                    "played_knights": 0,
                },
            },
            winner_ids=["BLUE"],
        )

        self.assertEqual(len(st.markdown_calls), 1)
        html, unsafe = st.markdown_calls[0]
        self.assertTrue(unsafe)
        self.assertEqual(html.count("🏆"), 1)
        self.assertIn("BLUE</span></td><td style='text-align:center'>2</td>", html)
        self.assertIn("2 🏆</td>", html)

    def test_load_dashboard_snapshot_reads_simplified_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            self._write_json(
                run_dir / "metadata.json",
                {
                    "game_id": "live-game",
                    "player_ids": ["RED", "BLUE"],
                    "player_adapter_types": {
                        "RED": "LLMPlayer",
                        "BLUE": "RandomLegalPlayer",
                    },
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
                        "memory": {
                            "long_term": {"goal": "trade"},
                            "short_term": {"plan": "offer"},
                        },
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
            self.assertEqual(
                snapshot.memory_traces_by_player["RED"][0].memory.long_term,
                {"goal": "trade"},
            )
            self.assertEqual(
                snapshot.prompt_traces_by_player["RED"][0].stage, "choose_action"
            )
            self.assertEqual(snapshot.public_events[0].kind, "trade_offered")

    def test_load_dashboard_snapshot_discovers_players_without_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            self._write_json(
                run_dir / "players" / "WHITE" / "memory.json",
                {"memory": {"long_term": None, "short_term": None}},
            )
            self._write_json(
                run_dir / "players" / "ORANGE" / "memory.json",
                {"memory": {"long_term": {"note": "x"}, "short_term": None}},
            )

            snapshot = load_dashboard_snapshot(run_dir)

            self.assertEqual(snapshot.player_ids, ("ORANGE", "WHITE"))
            self.assertEqual(snapshot.public_events, ())

    def test_discover_run_directories_returns_newest_child_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            older = base_dir / "1.0.0" / "tags-older-run"
            newer = base_dir / "1.2.0" / "tags-newer-run"
            older.mkdir(parents=True)
            newer.mkdir(parents=True)
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
                        "tile": {
                            "id": 0,
                            "type": "RESOURCE_TILE",
                            "resource": "WOOD",
                            "number": 8,
                        },
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

    def test_turn_event_digest_body_formats_compact_summaries(self) -> None:
        self.assertEqual(
            _turn_event_digest_body(
                Event(
                    "dice_rolled",
                    {"result": [5, 5]},
                    history_index=1,
                    turn_index=6,
                    phase="play_turn",
                    actor_player_id="WHITE",
                )
            ),
            "rolled 10 (5+5)",
        )
        self.assertEqual(
            _turn_event_digest_body(
                Event(
                    "trade_confirmed",
                    {
                        "accepting_player_id": "RED",
                        "offer": {"WOOD": 1},
                        "request": {"BRICK": 1},
                    },
                    history_index=2,
                    turn_index=6,
                    phase="play_turn",
                    actor_player_id="WHITE",
                )
            ),
            "↔ RED: 1 WOOD for 1 BRICK",
        )
        self.assertEqual(
            _turn_event_digest_body(
                Event(
                    "trade_chat_opened",
                    {
                        "message": "Looking to trade 1×ORE for 1×BRICK to build roads.",
                        "requested_resources": {"BRICK": 1},
                    },
                    history_index=3,
                    turn_index=6,
                    phase="play_turn",
                    actor_player_id="WHITE",
                )
            ),
            "opened trade chat: Looking to trade 1×ORE for 1×BRICK to build roads.",
        )
        self.assertEqual(
            _turn_event_digest_body(
                Event(
                    "action_taken",
                    {
                        "action": {
                            "action_type": "MARITIME_TRADE",
                            "description": "Trade 3 WOOD to the bank for 1 BRICK.",
                        }
                    },
                    history_index=3,
                    turn_index=6,
                    phase="play_turn",
                    actor_player_id="WHITE",
                )
            ),
            "Trade 3 WOOD to the bank for 1 BRICK.",
        )

    def test_colorize_player_mentions_html_styles_in_text_names(self) -> None:
        self.assertEqual(
            _colorize_player_mentions_html("↔ RED: 1 ORE, 1 SHEEP for 1 WOOD"),
            "↔ <span style='color:#dc2626'>RED</span>: 1 ORE, 1 SHEEP for 1 WOOD",
        )
        self.assertEqual(
            _colorize_player_mentions_html("WHITE ↔ RED"),
            "<span style='color:#ffffff'>WHITE</span> ↔ <span style='color:#dc2626'>RED</span>",
        )
        self.assertEqual(
            _colorize_player_mentions_html("trade cancelled"),
            "trade cancelled",
        )

    def test_recent_turn_event_sections_group_recent_turns_and_filter_chat_noise(
        self,
    ) -> None:
        snapshot = DashboardSnapshot(
            run_dir=Path("/tmp/fake-run"),
            metadata={},
            player_ids=("WHITE", "RED", "BLUE"),
            public_events=(
                Event(
                    "road_built",
                    {"edge": [17, 39]},
                    history_index=1,
                    turn_index=5,
                    phase="play_turn",
                    actor_player_id="BLUE",
                ),
                Event(
                    "trade_chat_opened",
                    {
                        "message": "Looking to trade 1×WOOD for 1×BRICK.",
                        "requested_resources": {"BRICK": 1},
                    },
                    history_index=2,
                    turn_index=6,
                    phase="play_turn",
                    actor_player_id="WHITE",
                ),
                Event(
                    "trade_chat_message",
                    {"message": "I'll give you wood for brick."},
                    history_index=3,
                    turn_index=6,
                    phase="play_turn",
                    actor_player_id="RED",
                ),
                Event(
                    "dice_rolled",
                    {"result": [5, 5]},
                    history_index=4,
                    turn_index=6,
                    phase="play_turn",
                    actor_player_id="WHITE",
                ),
                Event(
                    "trade_confirmed",
                    {
                        "accepting_player_id": "RED",
                        "offer": {"WOOD": 1},
                        "request": {"BRICK": 1},
                    },
                    history_index=5,
                    turn_index=6,
                    phase="play_turn",
                    actor_player_id="WHITE",
                ),
                Event(
                    "turn_ended",
                    {},
                    history_index=6,
                    turn_index=6,
                    phase="play_turn",
                    actor_player_id="WHITE",
                ),
            ),
            public_state_snapshots=(),
            memory_traces_by_player={},
            prompt_traces_by_player={},
            result=None,
        )

        sections = _recent_turn_event_sections(snapshot, cursor=6, max_turns=2)

        self.assertEqual([turn_index for turn_index, _label, _rows in sections], [5, 6])
        self.assertEqual(sections[0][1], "Turn 5")
        self.assertEqual(sections[0][2], (("BLUE", "road on [17, 39]"),))
        self.assertEqual(
            sections[1][2],
            (
                ("WHITE", "opened trade chat: Looking to trade 1×WOOD for 1×BRICK."),
                ("WHITE", "rolled 10 (5+5)"),
                ("WHITE", "↔ RED: 1 WOOD for 1 BRICK"),
                ("WHITE", "End the current turn."),
            ),
        )

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        _write_json(path, payload)

    @classmethod
    def _write_jsonl(cls, path: Path, payloads: list[dict]) -> None:
        _write_jsonl(path, payloads)


if __name__ == "__main__":
    unittest.main()
