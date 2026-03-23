from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from catan_bench import (
    Action,
    DecisionPoint,
    Event,
    EventLog,
    GameOrchestrator,
    InvalidActionError,
    LLMPlayer,
    MemoryStore,
    ObservationBuilder,
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


class FakeLLMClient:
    def __init__(self, content: dict[str, object]) -> None:
        self.content = content

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        temperature: float,
    ) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(self.content),
                    }
                }
            ]
        }


class OrchestratorTests(unittest.TestCase):
    def test_observation_builder_distinguishes_full_history_from_recent_window(self) -> None:
        class StubEngine:
            game_id = "stub-game"

            @staticmethod
            def public_state():
                return {"scores": {"RED": 1, "BLUE": 1}}

            @staticmethod
            def private_state(player_id: str):
                return {"player_id": player_id}

        event_log = EventLog()
        event_log.reset(("RED", "BLUE"))
        event_log.append_public(
            (
                Event(kind="event-1"),
                Event(kind="event-2"),
                Event(kind="event-3"),
            )
        )
        event_log.append_private(
            "RED",
            (
                Event(kind="private-1"),
                Event(kind="private-2"),
            ),
        )

        observation = ObservationBuilder(recent_event_window=1).build(
            engine=StubEngine(),
            decision=DecisionPoint(
                acting_player_id="RED",
                turn_index=3,
                phase="play_turn",
                decision_index=4,
                legal_actions=(Action("END_TURN"),),
            ),
            event_log=event_log,
            memory_store=MemoryStore(),
        )

        self.assertEqual(
            tuple(event.kind for event in observation.public_history),
            ("event-1", "event-2", "event-3"),
        )
        self.assertEqual(
            tuple(event.kind for event in observation.private_history),
            ("private-1", "private-2"),
        )
        self.assertEqual(
            tuple(event.kind for event in observation.recent_public_events),
            ("event-3",),
        )
        self.assertEqual(
            tuple(event.kind for event in observation.recent_private_events),
            ("private-2",),
        )

    def test_orchestrator_persists_llm_prompt_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = GameOrchestrator(
                MockTradeEngine(),
                players={
                    "RED": LLMPlayer(
                        client=FakeLLMClient(
                            {
                                "action_index": 0,
                                "private_reasoning": "Offer wood to turn this hand into road tempo.",
                                "private_memory_write": {
                                    "plans": ["Offer wood for brick when it unlocks roads."]
                                },
                            }
                        ),
                        model="fake-model",
                        temperature=0.1,
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
                                )
                            )
                        ]
                    ),
                },
                run_dir=tmpdir,
            )

            orchestrator.step()

            prompt_trace_lines = Path(
                tmpdir, "players", "RED", "prompt_trace.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(prompt_trace_lines), 1)
            prompt_trace = json.loads(prompt_trace_lines[0])
            self.assertEqual(prompt_trace["player_id"], "RED")
            self.assertEqual(prompt_trace["model"], "fake-model")
            self.assertEqual(prompt_trace["attempts"][0]["response"]["action_index"], 0)
            self.assertIn(
                "legal_actions",
                prompt_trace["attempts"][0]["messages"][1]["content"],
            )

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
                                reasoning=(
                                    "I need brick to turn this hand into tempo. Offering one wood "
                                    "is acceptable because the road timing matters more than "
                                    "holding a balanced hand right now."
                                ),
                                memory_write={
                                    "plans": ["Trade wood into brick when it enables road tempo."]
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
                                reasoning=(
                                    "I can spare the brick because my near-term build is slower. "
                                    "Taking wood improves flexibility, and I can still pressure "
                                    "RED later through placement rather than this trade."
                                ),
                                memory_write={"beliefs": ["RED will pay to secure brick tempo."]},
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
            self.assertEqual(result.private_event_count, 5)
            self.assertEqual(result.memory_writes, 2)
            self.assertEqual(result.metadata["benchmark"]["trade_metrics"]["offers"], 1)
            self.assertEqual(result.metadata["benchmark"]["trade_metrics"]["accepted"], 1)

            self.assertEqual(len(orchestrator.memory_store.get("RED")), 1)
            self.assertEqual(len(orchestrator.memory_store.get("BLUE")), 1)

            blue_player = orchestrator.players["BLUE"]
            self.assertEqual(len(blue_player.observations), 1)
            blue_observation = blue_player.observations[0]
            self.assertEqual(blue_observation.decision_prompt, "Respond to RED's trade offer.")
            self.assertGreater(len(blue_observation.game_rules or ""), 0)
            self.assertEqual(blue_observation.recent_public_events[0].kind, "trade_offered")
            self.assertEqual(
                blue_observation.recent_private_events[0].kind, "trade_offer_received"
            )

            self.assertTrue(Path(tmpdir, "public_history.jsonl").exists())
            self.assertTrue(Path(tmpdir, "players", "RED", "memory.jsonl").exists())
            self.assertTrue(Path(tmpdir, "players", "BLUE", "private_history.jsonl").exists())
            self.assertTrue(Path(tmpdir, "players", "RED", "prompt_trace.jsonl").exists())
            self.assertTrue(Path(tmpdir, "result.json").exists())

            red_private_history = Path(
                tmpdir, "players", "RED", "private_history.jsonl"
            ).read_text(encoding="utf-8")
            self.assertIn('"kind": "player_decision"', red_private_history)
            self.assertIn(
                '"reasoning": "I need brick to turn this hand into tempo. Offering one wood is acceptable because the road timing matters more than holding a balanced hand right now."',
                red_private_history,
            )
            self.assertEqual(
                Path(tmpdir, "players", "RED", "prompt_trace.jsonl").read_text(encoding="utf-8"),
                "",
            )

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
