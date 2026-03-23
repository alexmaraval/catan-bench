from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from catan_bench import (
    Action,
    DecisionPoint,
    Event,
    GameOrchestrator,
    MemoryResponse,
    PlayerResponse,
    RecallObservation,
    ReflectionObservation,
    ScriptedPlayer,
    TransitionResult,
    build_player_replay_timeline,
    build_replay_timeline,
    export_replay_html,
)


class ReplayMockTradeEngine:
    def __init__(self) -> None:
        self._game_id = "mock-game-1"
        self._player_ids = ("RED", "BLUE")
        self._decision_index = 0
        self._terminal = False

    @property
    def game_id(self) -> str:
        return self._game_id

    @property
    def player_ids(self) -> tuple[str, str]:
        return self._player_ids

    def is_terminal(self) -> bool:
        return self._terminal

    def current_decision(self) -> DecisionPoint:
        if self._decision_index == 0:
            return DecisionPoint(
                acting_player_id="RED",
                turn_index=1,
                phase="offer_trade",
                decision_index=0,
                prompt="Offer a trade to BLUE.",
                legal_actions=(
                    Action(
                        "OFFER_TRADE",
                        payload={
                            "to": ["BLUE"],
                            "give": {"WOOD": 1},
                            "want": {"BRICK": 1},
                        },
                    ),
                ),
            )

        return DecisionPoint(
            acting_player_id="BLUE",
            turn_index=1,
            phase="decide_trade",
            decision_index=1,
            prompt="Respond to RED's trade offer.",
            legal_actions=(
                Action(
                    "ACCEPT_TRADE",
                    payload={
                        "from": "RED",
                        "give": {"BRICK": 1},
                        "want": {"WOOD": 1},
                    },
                ),
            ),
        )

    def public_state(self):
        return {"scores": {"RED": 2, "BLUE": 2}}

    def private_state(self, player_id: str):
        if player_id == "RED":
            return {"resources": {"WOOD": 1, "BRICK": 0}}
        return {"resources": {"WOOD": 0, "BRICK": 1}}

    def apply_action(self, action: Action) -> TransitionResult:
        if self._decision_index == 0:
            self._decision_index = 1
            return TransitionResult(
                public_events=(
                    Event(
                        kind="trade_offered",
                        payload={
                            "offering_player_id": "RED",
                            "offer": {"WOOD": 1},
                            "request": {"BRICK": 1},
                            "action": {
                                "action_type": "OFFER_TRADE",
                                "payload": {"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
                                "description": "Offer 1 wood for 1 brick.",
                            },
                        },
                        turn_index=1,
                        phase="offer_trade",
                        decision_index=0,
                        actor_player_id="RED",
                    ),
                ),
                private_events_by_player={
                    "BLUE": (
                        Event(
                            kind="trade_offer_received",
                            payload={"from": "RED"},
                            turn_index=1,
                            phase="offer_trade",
                            decision_index=0,
                            actor_player_id="RED",
                        ),
                    )
                },
            )

        self._terminal = True
        return TransitionResult(
            public_events=(
                Event(
                    kind="trade_accepted",
                    payload={
                        "offering_player_id": "RED",
                        "responding_player_id": "BLUE",
                        "action": {
                            "action_type": "ACCEPT_TRADE",
                            "payload": {"offering_player_id": "RED"},
                            "description": "Accept RED's trade.",
                        },
                    },
                    turn_index=1,
                    phase="decide_trade",
                    decision_index=1,
                    actor_player_id="BLUE",
                ),
                Event(
                    kind="trade_confirmed",
                    payload={
                        "offering_player_id": "RED",
                        "accepting_player_id": "BLUE",
                        "offer": {"WOOD": 1},
                        "request": {"BRICK": 1},
                    },
                    turn_index=1,
                    phase="decide_trade",
                    decision_index=1,
                    actor_player_id="BLUE",
                ),
            ),
            private_events_by_player={},
            terminal=True,
            result_metadata={"winner_ids": ["BLUE"], "final_vp": {"RED": 6, "BLUE": 10}},
        )

    def result(self):
        return {"winner_ids": ["BLUE"], "final_vp": {"RED": 6, "BLUE": 10}, "num_turns": 1}


class PhasedScriptedPlayer:
    def __init__(
        self,
        *,
        recall_memories: list[object | None],
        responses: list[PlayerResponse | Action],
        reflect_memories: list[object | None],
    ) -> None:
        self._recall_memories = list(recall_memories)
        self._responses = list(responses)
        self._reflect_memories = list(reflect_memories)

    def recall(self, observation: RecallObservation) -> MemoryResponse:
        return MemoryResponse(memory=self._recall_memories.pop(0))

    def respond(self, observation) -> PlayerResponse:
        next_response = self._responses.pop(0)
        if isinstance(next_response, PlayerResponse):
            return next_response
        return PlayerResponse(action=next_response)

    def reflect(self, observation: ReflectionObservation) -> MemoryResponse:
        return MemoryResponse(memory=self._reflect_memories.pop(0))


class ReplayTests(unittest.TestCase):
    def test_build_replay_timeline_formats_semantic_and_generic_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            self._write_json(
                run_dir / "metadata.json",
                {
                    "game_id": "fallback-game",
                    "player_ids": ["RED", "BLUE"],
                    "player_adapter_types": {"RED": "ScriptedPlayer", "BLUE": "ScriptedPlayer"},
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
                        "turn_index": 1,
                        "phase": "offer_trade",
                        "decision_index": 0,
                        "actor_player_id": "RED",
                    },
                    {
                        "kind": "offer_trade",
                        "payload": {
                            "action": {
                                "action_type": "OFFER_TRADE",
                                "payload": {"offer": {"SHEEP": 1}, "request": {"ORE": 1}},
                                "description": "Offer 1 sheep for 1 ore.",
                            }
                        },
                        "turn_index": 2,
                        "phase": "play_turn",
                        "decision_index": 3,
                        "actor_player_id": "BLUE",
                    },
                ],
            )
            self._write_json(
                run_dir / "result.json",
                {
                    "game_id": "fallback-game",
                    "winner_ids": ["RED"],
                    "total_decisions": 4,
                    "metadata": {"num_turns": 2},
                },
            )

            timeline = build_replay_timeline(run_dir)

            self.assertEqual(len(timeline), 2)
            self.assertEqual(timeline[0].title, "RED · Trade Offered")
            self.assertEqual(timeline[0].body, "RED offered 1 wood for 1 brick.")
            self.assertEqual(timeline[0].variant, "red")
            self.assertEqual(timeline[1].speaker_type, "player")
            self.assertEqual(timeline[1].body, "Offer 1 sheep for 1 ore.")

    def test_export_replay_html_creates_transcript_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = GameOrchestrator(
                ReplayMockTradeEngine(),
                players={
                    "RED": PhasedScriptedPlayer(
                        recall_memories=[{"plans": ["scan for brick trades"]}],
                        responses=[
                            PlayerResponse(
                                action=Action(
                                    "OFFER_TRADE",
                                    payload={
                                        "to": ["BLUE"],
                                        "give": {"WOOD": 1},
                                        "want": {"BRICK": 1},
                                    },
                                ),
                            )
                        ],
                        reflect_memories=[{"plans": ["push the wood-for-brick trade early"]}],
                    ),
                    "BLUE": PhasedScriptedPlayer(
                        recall_memories=[{"beliefs": ["RED is offering for brick"]}],
                        responses=[
                            PlayerResponse(
                                action=Action(
                                    "ACCEPT_TRADE",
                                    payload={
                                        "from": "RED",
                                        "give": {"BRICK": 1},
                                        "want": {"WOOD": 1},
                                    },
                                ),
                            )
                        ],
                        reflect_memories=[{"beliefs": ["RED is prioritizing brick"]}],
                    ),
                },
                run_dir=tmpdir,
            )

            orchestrator.run()
            run_dir = orchestrator.run_dir
            self.assertIsNotNone(run_dir)
            assert run_dir is not None
            output_path = export_replay_html(run_dir)
            html = output_path.read_text(encoding="utf-8")
            red_private_html = Path(run_dir, "players", "RED", "replay.html").read_text(
                encoding="utf-8"
            )
            blue_private_html = Path(run_dir, "players", "BLUE", "replay.html").read_text(
                encoding="utf-8"
            )

            self.assertEqual(output_path, run_dir / "replay.html")
            self.assertIn("mock-game-1", html)
            self.assertIn("RED offered 1 wood for 1 brick.", html)
            self.assertIn("RED Personal View", html)
            self.assertIn("Trade Confirmed", html)
            self.assertLess(html.index("RED · Trade Offered"), html.index("Trade Confirmed"))
            self.assertIn("Public Replay", red_private_html)
            self.assertIn("Private Trade Alert", blue_private_html)
            self.assertIn("Received a trade offer from RED.", blue_private_html)
            self.assertIn("RED Memory (reflect)", red_private_html)
            self.assertIn("Player Decision", red_private_html)
            self.assertIn("push the wood-for-brick trade early", red_private_html)

    def test_export_replay_html_handles_in_progress_run_without_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            self._write_json(
                run_dir / "metadata.json",
                {
                    "game_id": "live-game",
                    "player_ids": ["RED", "BLUE"],
                    "player_adapter_types": {"RED": "LLMPlayer", "BLUE": "LLMPlayer"},
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
                        "turn_index": 4,
                        "phase": "play_turn",
                        "decision_index": 12,
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
                        "turn_index": 4,
                        "phase": "play_turn",
                        "decision_index": 12,
                        "actor_player_id": "RED",
                    }
                ],
            )
            self._write_json(run_dir / "players" / "RED" / "memory.json", {"memory": None})
            self._write_jsonl(run_dir / "players" / "BLUE" / "private_history.jsonl", [])
            self._write_json(run_dir / "players" / "BLUE" / "memory.json", {"memory": None})

            output_path = export_replay_html(run_dir)
            public_html = output_path.read_text(encoding="utf-8")
            red_html = Path(run_dir, "players", "RED", "replay.html").read_text(
                encoding="utf-8"
            )

            self.assertEqual(output_path, run_dir / "replay.html")
            self.assertIn("in-progress game", public_html)
            self.assertIn("Decisions:</strong> 13", public_html)
            self.assertIn("while the game is still in progress", red_html)

    def test_build_player_replay_timeline_includes_public_private_and_memory_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = GameOrchestrator(
                ReplayMockTradeEngine(),
                players={
                    "RED": PhasedScriptedPlayer(
                        recall_memories=[{"plans": ["offer early trade"]}],
                        responses=[
                            PlayerResponse(
                                action=Action(
                                    "OFFER_TRADE",
                                    payload={
                                        "to": ["BLUE"],
                                        "give": {"WOOD": 1},
                                        "want": {"BRICK": 1},
                                    },
                                ),
                            )
                        ],
                        reflect_memories=[{"plans": ["trade before road build"]}],
                    ),
                    "BLUE": PhasedScriptedPlayer(
                        recall_memories=[{"beliefs": ["RED wants brick"]}],
                        responses=[
                            PlayerResponse(
                                action=Action(
                                    "ACCEPT_TRADE",
                                    payload={
                                        "from": "RED",
                                        "give": {"BRICK": 1},
                                        "want": {"WOOD": 1},
                                    },
                                )
                            )
                        ],
                        reflect_memories=[{"beliefs": ["RED will trade again"]}],
                    ),
                },
                run_dir=tmpdir,
            )

            orchestrator.run()
            run_dir = orchestrator.run_dir
            self.assertIsNotNone(run_dir)
            assert run_dir is not None
            timeline = build_player_replay_timeline(run_dir, "BLUE")

            self.assertEqual(
                [item.stream for item in timeline],
                ["public", "private", "public", "public", "private", "memory", "memory"],
            )
            self.assertEqual(timeline[1].title, "Private Trade Alert")
            self.assertEqual(timeline[1].body, "Received a trade offer from RED.")
            self.assertEqual(timeline[4].title, "Player Decision")
            self.assertIn("BLUE Memory", timeline[-1].title)

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def _write_jsonl(cls, path: Path, payloads: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(payload, sort_keys=True) for payload in payloads]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
