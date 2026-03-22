from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from catan_bench import (
    Action,
    DecisionPoint,
    Event,
    GameOrchestrator,
    InvalidActionError,
    PlayerResponse,
    ScriptedPlayer,
    TransitionResult,
)


class MockTradeEngine:
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
                        description="Offer 1 wood for 1 brick to BLUE.",
                    ),
                    Action("END_TURN", description="Skip trading."),
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
                    description="Accept RED's trade.",
                ),
                Action(
                    "REJECT_TRADE",
                    payload={"from": "RED"},
                    description="Reject RED's trade.",
                ),
            ),
        )

    def public_state(self):
        return {
            "scores": {"RED": 2, "BLUE": 2},
            "current_trade_offer": (
                None
                if self._decision_index == 0
                else {"from": "RED", "to": ["BLUE"], "give": {"WOOD": 1}, "want": {"BRICK": 1}}
            ),
        }

    def private_state(self, player_id: str):
        if player_id == "RED":
            return {"resources": {"WOOD": 1, "BRICK": 0}}
        return {"resources": {"WOOD": 0, "BRICK": 1}}

    def apply_action(self, action: Action) -> TransitionResult:
        if self._decision_index == 0:
            if action.action_type != "OFFER_TRADE":
                raise AssertionError(f"Unexpected action in offer step: {action!r}")
            self._decision_index = 1
            return TransitionResult(
                public_events=(
                    Event(
                        kind="trade_offered",
                        payload={
                            "from": "RED",
                            "to": ["BLUE"],
                            "give": {"WOOD": 1},
                            "want": {"BRICK": 1},
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

        if action.action_type != "ACCEPT_TRADE":
            raise AssertionError(f"Unexpected action in response step: {action!r}")
        self._terminal = True
        return TransitionResult(
            public_events=(
                Event(
                    kind="trade_accepted",
                    payload={"from": "RED", "by": "BLUE"},
                    turn_index=1,
                    phase="decide_trade",
                    decision_index=1,
                    actor_player_id="BLUE",
                ),
                Event(
                    kind="trade_confirmed",
                    payload={
                        "from": "RED",
                        "to": "BLUE",
                        "give": {"WOOD": 1},
                        "want": {"BRICK": 1},
                    },
                    turn_index=1,
                    phase="decide_trade",
                    decision_index=1,
                    actor_player_id="BLUE",
                ),
            ),
            private_events_by_player={
                "RED": (
                    Event(
                        kind="resource_delta",
                        payload={"wood": -1, "brick": 1},
                        turn_index=1,
                        phase="decide_trade",
                        decision_index=1,
                        actor_player_id="BLUE",
                    ),
                ),
                "BLUE": (
                    Event(
                        kind="resource_delta",
                        payload={"wood": 1, "brick": -1},
                        turn_index=1,
                        phase="decide_trade",
                        decision_index=1,
                        actor_player_id="BLUE",
                    ),
                ),
            },
            terminal=True,
            result_metadata={"winner_ids": ["BLUE"], "final_vp": {"RED": 6, "BLUE": 10}},
        )

    def result(self):
        if not self._terminal:
            raise AssertionError("result() called before terminal state.")
        return {"winner_ids": ["BLUE"], "final_vp": {"RED": 6, "BLUE": 10}}


class OrchestratorTests(unittest.TestCase):
    def test_orchestrator_tracks_public_private_and_memory_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = GameOrchestrator(
                MockTradeEngine(),
                players={
                    "RED": ScriptedPlayer(
                        [
                            PlayerResponse(
                                action=Action(
                                    "OFFER_TRADE",
                                    payload={
                                        "to": ["BLUE"],
                                        "give": {"WOOD": 1},
                                        "want": {"BRICK": 1},
                                    },
                                ),
                                memory_write={
                                    "plans": ["trade into brick before building a road"]
                                },
                            )
                        ]
                    ),
                    "BLUE": ScriptedPlayer(
                        [
                            PlayerResponse(
                                action=Action(
                                    "ACCEPT_TRADE",
                                    payload={
                                        "from": "RED",
                                        "give": {"BRICK": 1},
                                        "want": {"WOOD": 1},
                                    },
                                ),
                                memory_write={"beliefs": ["RED values brick highly"]},
                            )
                        ]
                    ),
                },
                run_dir=tmpdir,
            )

            result = orchestrator.run()

            self.assertEqual(result.game_id, "mock-game-1")
            self.assertEqual(result.winner_ids, ("BLUE",))
            self.assertEqual(result.total_decisions, 2)
            self.assertEqual(result.public_event_count, 3)
            self.assertEqual(result.private_event_count, 3)
            self.assertEqual(result.memory_writes, 2)

            self.assertEqual(len(orchestrator.memory_store.get("RED")), 1)
            self.assertEqual(len(orchestrator.memory_store.get("BLUE")), 1)

            blue_player = orchestrator.players["BLUE"]
            self.assertEqual(len(blue_player.observations), 1)
            blue_observation = blue_player.observations[0]
            self.assertEqual(blue_observation.recent_public_events[0].kind, "trade_offered")
            self.assertEqual(
                blue_observation.recent_private_events[0].kind, "trade_offer_received"
            )

            self.assertTrue(Path(tmpdir, "public_history.jsonl").exists())
            self.assertTrue(Path(tmpdir, "players", "RED", "memory.jsonl").exists())
            self.assertTrue(Path(tmpdir, "players", "BLUE", "private_history.jsonl").exists())
            self.assertTrue(Path(tmpdir, "result.json").exists())

    def test_invalid_action_is_rejected_before_engine_apply(self) -> None:
        orchestrator = GameOrchestrator(
            MockTradeEngine(),
            players={
                "RED": ScriptedPlayer([Action("BUILD_ROAD", payload={"edge_id": 7})]),
                "BLUE": ScriptedPlayer([]),
            },
        )

        with self.assertRaises(InvalidActionError):
            orchestrator.step()


if __name__ == "__main__":
    unittest.main()
