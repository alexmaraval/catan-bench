from __future__ import annotations

import io
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
    MemoryResponse,
    MemoryStore,
    ObservationBuilder,
    PlayerResponse,
    RecallObservation,
    ReflectionObservation,
    ScriptedPlayer,
    TradeChatObservation,
    TradeChatOpenResponse,
    TradeChatReplyResponse,
    TradeChatSelectionResponse,
    TransitionResult,
)
from catan_bench.reporter import DebugTerminalReporter


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
    def __init__(self, content: list[dict[str, object]] | dict[str, object]) -> None:
        self.content = content

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        temperature: float,
        top_p: float | None = None,
        reasoning_enabled: bool | None = None,
    ) -> dict[str, object]:
        if isinstance(self.content, list):
            if not self.content:
                raise AssertionError("FakeLLMClient ran out of scripted completions.")
            payload = self.content.pop(0)
        else:
            payload = self.content
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}


class OwnerTradeChatPlayer:
    def __init__(
        self,
        *,
        action_responses: list[PlayerResponse | Action],
        open_responses: list[TradeChatOpenResponse],
        selection_responses: list[TradeChatSelectionResponse],
    ) -> None:
        self._action_responses = list(action_responses)
        self._open_responses = list(open_responses)
        self._selection_responses = list(selection_responses)
        self.open_calls = 0
        self.selection_calls = 0

    def open_trade_chat(self, observation: TradeChatObservation) -> TradeChatOpenResponse:
        self.open_calls += 1
        return self._open_responses.pop(0)

    def select_trade_chat_offer(
        self, observation: TradeChatObservation
    ) -> TradeChatSelectionResponse:
        self.selection_calls += 1
        return self._selection_responses.pop(0)

    def respond(self, observation) -> PlayerResponse:
        next_response = self._action_responses.pop(0)
        if isinstance(next_response, PlayerResponse):
            return next_response
        return PlayerResponse(action=next_response)


class QuoteReplyPlayer:
    def __init__(self, reply: TradeChatReplyResponse) -> None:
        self.reply = reply
        self.chat_observations: list[TradeChatObservation] = []

    def respond_trade_chat(self, observation: TradeChatObservation) -> TradeChatReplyResponse:
        self.chat_observations.append(observation)
        return self.reply

    def respond(self, observation) -> PlayerResponse:
        raise AssertionError("Trade responses should be auto-resolved by the orchestrator.")


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
        self.recall_observations: list[RecallObservation] = []
        self.observations = []
        self.reflection_observations: list[ReflectionObservation] = []

    def recall(self, observation: RecallObservation) -> MemoryResponse:
        self.recall_observations.append(observation)
        return MemoryResponse(memory=self._recall_memories.pop(0))

    def respond(self, observation) -> PlayerResponse:
        self.observations.append(observation)
        next_response = self._responses.pop(0)
        if isinstance(next_response, PlayerResponse):
            return next_response
        return PlayerResponse(action=next_response)

    def reflect(self, observation: ReflectionObservation) -> MemoryResponse:
        self.reflection_observations.append(observation)
        return MemoryResponse(memory=self._reflect_memories.pop(0))


class OwnerLedTradeChatEngine:
    def __init__(self) -> None:
        self._game_id = "owner-led-trade"
        self._player_ids = ("RED", "WHITE", "ORANGE")
        self._decision_index = 0
        self._terminal = False

    @property
    def game_id(self) -> str:
        return self._game_id

    @property
    def player_ids(self) -> tuple[str, str, str]:
        return self._player_ids

    def is_terminal(self) -> bool:
        return self._terminal

    def current_decision(self) -> DecisionPoint:
        if self._decision_index == 0:
            return DecisionPoint(
                acting_player_id="RED",
                turn_index=7,
                phase="play_turn",
                decision_index=0,
                prompt="Choose an action for your turn.",
                legal_actions=(
                    Action("OFFER_TRADE", payload={"offer": {}, "request": {}}),
                    Action("END_TURN"),
                ),
            )
        if self._decision_index == 1:
            return DecisionPoint(
                acting_player_id="WHITE",
                turn_index=7,
                phase="decide_trade",
                decision_index=1,
                prompt="Respond to RED's trade offer.",
                legal_actions=(
                    Action(
                        "ACCEPT_TRADE",
                        payload={
                            "offer": {"SHEEP": 1},
                            "request": {"WOOD": 1},
                            "offering_player_id": "RED",
                        },
                    ),
                    Action(
                        "REJECT_TRADE",
                        payload={
                            "offer": {"SHEEP": 1},
                            "request": {"WOOD": 1},
                            "offering_player_id": "RED",
                        },
                    ),
                ),
            )
        if self._decision_index == 2:
            return DecisionPoint(
                acting_player_id="ORANGE",
                turn_index=7,
                phase="decide_trade",
                decision_index=2,
                prompt="Respond to RED's trade offer.",
                legal_actions=(
                    Action(
                        "ACCEPT_TRADE",
                        payload={
                            "offer": {"SHEEP": 1},
                            "request": {"WOOD": 1},
                            "offering_player_id": "RED",
                        },
                    ),
                    Action(
                        "REJECT_TRADE",
                        payload={
                            "offer": {"SHEEP": 1},
                            "request": {"WOOD": 1},
                            "offering_player_id": "RED",
                        },
                    ),
                ),
            )
        if self._decision_index == 3:
            return DecisionPoint(
                acting_player_id="RED",
                turn_index=7,
                phase="decide_acceptees",
                decision_index=3,
                prompt="Choose which accepting player to trade with.",
                legal_actions=(
                    Action(
                        "CONFIRM_TRADE",
                        payload={
                            "offer": {"SHEEP": 1},
                            "request": {"WOOD": 1},
                            "accepting_player_id": "ORANGE",
                        },
                    ),
                    Action("CANCEL_TRADE"),
                ),
            )

        return DecisionPoint(
            acting_player_id="RED",
            turn_index=7,
            phase="play_turn",
            decision_index=4,
            prompt="Choose an action for your turn.",
            legal_actions=(Action("END_TURN"),),
        )

    def public_state(self):
        return {"scores": {"RED": 4, "WHITE": 3, "ORANGE": 3}}

    def private_state(self, player_id: str):
        if player_id == "RED":
            return {"resources": {"SHEEP": 2, "WOOD": 0}}
        if player_id == "WHITE":
            return {"resources": {"WOOD": 1, "SHEEP": 0}}
        return {"resources": {"WOOD": 1, "SHEEP": 0}}

    def apply_action(self, action: Action) -> TransitionResult:
        if self._decision_index == 0:
            self._decision_index = 1
            return TransitionResult(
                public_events=(
                    Event(
                        kind="trade_offered",
                        payload={
                            "offering_player_id": "RED",
                            "offer": {"SHEEP": 1},
                            "request": {"WOOD": 1},
                        },
                        turn_index=7,
                        phase="play_turn",
                        decision_index=0,
                        actor_player_id="RED",
                    ),
                ),
            )
        if self._decision_index == 1:
            self._decision_index = 2
            return TransitionResult(
                public_events=(
                    Event(
                        kind="trade_rejected",
                        payload={"offering_player_id": "RED"},
                        turn_index=7,
                        phase="decide_trade",
                        decision_index=1,
                        actor_player_id="WHITE",
                    ),
                ),
            )
        if self._decision_index == 2:
            self._decision_index = 3
            return TransitionResult(
                public_events=(
                    Event(
                        kind="trade_accepted",
                        payload={"offering_player_id": "RED"},
                        turn_index=7,
                        phase="decide_trade",
                        decision_index=2,
                        actor_player_id="ORANGE",
                    ),
                ),
            )
        if self._decision_index == 3:
            self._decision_index = 4
            return TransitionResult(
                public_events=(
                    Event(
                        kind="trade_confirmed",
                        payload={
                            "offering_player_id": "RED",
                            "accepting_player_id": "ORANGE",
                            "offer": {"SHEEP": 1},
                            "request": {"WOOD": 1},
                        },
                        turn_index=7,
                        phase="decide_acceptees",
                        decision_index=3,
                        actor_player_id="RED",
                    ),
                ),
            )

        self._terminal = True
        return TransitionResult(
            public_events=(
                Event(
                    kind="turn_ended",
                    payload={},
                    turn_index=7,
                    phase="play_turn",
                    decision_index=4,
                    actor_player_id="RED",
                ),
            ),
            terminal=True,
            result_metadata={"winner_ids": ["RED"], "num_turns": 7},
        )

    def result(self):
        return {"winner_ids": ["RED"], "num_turns": 7}


class FailedTradeAttemptEngine:
    def __init__(self) -> None:
        self._game_id = "failed-trade-attempts"
        self._player_ids = ("RED", "WHITE")
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
                turn_index=11,
                phase="play_turn",
                decision_index=0,
                prompt="Choose an action for your turn.",
                legal_actions=(
                    Action("OFFER_TRADE", payload={"offer": {}, "request": {}}),
                    Action("BUILD_ROAD", payload={"edge": [1, 2]}),
                ),
            )
        return DecisionPoint(
            acting_player_id="RED",
            turn_index=11,
            phase="play_turn",
            decision_index=1,
            prompt="Choose an action for your turn.",
            legal_actions=(
                Action("OFFER_TRADE", payload={"offer": {}, "request": {}}),
                Action("END_TURN"),
            ),
        )

    def public_state(self):
        return {"scores": {"RED": 4, "WHITE": 4}}

    def private_state(self, player_id: str):
        return {"resources": {"WOOD": 0, "SHEEP": 1}}

    def apply_action(self, action: Action) -> TransitionResult:
        if self._decision_index == 0:
            self._decision_index = 1
            return TransitionResult(
                public_events=(
                    Event(
                        kind="road_built",
                        payload={"edge": [1, 2]},
                        turn_index=11,
                        phase="play_turn",
                        decision_index=0,
                        actor_player_id="RED",
                    ),
                ),
            )
        self._terminal = True
        return TransitionResult(
            public_events=(
                Event(
                    kind="turn_ended",
                    payload={},
                    turn_index=11,
                    phase="play_turn",
                    decision_index=1,
                    actor_player_id="RED",
                ),
            ),
            terminal=True,
            result_metadata={"winner_ids": ["WHITE"], "num_turns": 11},
        )

    def result(self):
        return {"winner_ids": ["WHITE"], "num_turns": 11}


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

    def test_orchestrator_persists_recall_act_reflect_prompt_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = GameOrchestrator(
                MockTradeEngine(),
                players={
                    "RED": LLMPlayer(
                        client=FakeLLMClient(
                            [
                                {"private_memory": {"plan": "Trade wood into brick."}},
                                {
                                    "action_index": 0,
                                    "private_reasoning": "Offer wood to unlock road tempo.",
                                },
                                {"private_memory": {"plan": "BLUE accepted wood-for-brick."}},
                            ]
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
            run_dir = orchestrator.run_dir
            self.assertIsNotNone(run_dir)
            assert run_dir is not None

            prompt_trace_lines = Path(
                run_dir, "players", "RED", "prompt_trace.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(prompt_trace_lines), 3)
            stages = [json.loads(line)["stage"] for line in prompt_trace_lines]
            self.assertEqual(stages, ["recall", "act", "reflect"])
            act_trace = json.loads(prompt_trace_lines[1])
            self.assertEqual(act_trace["player_id"], "RED")
            self.assertEqual(act_trace["model"], "fake-model")
            self.assertEqual(act_trace["attempts"][0]["response"]["action_index"], 0)
            self.assertIn("response_text", act_trace["attempts"][0])
            self.assertIn("legal_actions", act_trace["attempts"][0]["messages"][1]["content"])

    def test_orchestrator_streams_prompt_traces_to_debug_reporter(self) -> None:
        output = io.StringIO()
        debug_input = io.StringIO("n\nn\nn\n")

        orchestrator = GameOrchestrator(
            MockTradeEngine(),
            players={
                "RED": LLMPlayer(
                    client=FakeLLMClient(
                        [
                            {"private_memory": {"plan": "Trade wood into brick."}},
                            {
                                "action_index": 0,
                                "private_reasoning": "Offer wood to unlock road tempo.",
                            },
                            {"private_memory": {"plan": "BLUE accepted wood-for-brick."}},
                        ]
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
            reporter=DebugTerminalReporter(file=output, input_file=debug_input),
        )

        orchestrator.step()

        rendered = output.getvalue()
        self.assertIn("stage=recall", rendered)
        self.assertIn("stage=act", rendered)
        self.assertIn("stage=reflect", rendered)
        self.assertIn('"action_index": 0', rendered)
        self.assertIn("PROMPT", rendered)
        self.assertIn("ANSWER", rendered)
        self.assertIn("Press N then Enter to continue.", rendered)

    def test_orchestrator_runs_recall_and_reflect_memory_flow(self) -> None:
        red_player = PhasedScriptedPlayer(
            recall_memories=[{"plan": "Offer wood for brick."}],
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
                    reasoning="Offer wood now to convert into road tempo.",
                )
            ],
            reflect_memories=[{"plan": "BLUE may accept wood-for-brick again."}],
        )
        blue_player = PhasedScriptedPlayer(
            recall_memories=[{"belief": "RED is pushing brick tempo."}],
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
                    reasoning="Wood flexibility is worth the brick here.",
                )
            ],
            reflect_memories=[{"belief": "RED will keep paying for brick tempo."}],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = GameOrchestrator(
                MockTradeEngine(),
                players={"RED": red_player, "BLUE": blue_player},
                run_dir=tmpdir,
            )

            result = orchestrator.run()
            run_dir = orchestrator.run_dir
            self.assertIsNotNone(run_dir)
            assert run_dir is not None

            self.assertEqual(result.game_id, "mock-game-1")
            self.assertEqual(result.winner_ids, ("BLUE",))
            self.assertEqual(result.total_decisions, 2)
            self.assertEqual(result.public_event_count, 3)
            self.assertEqual(result.private_event_count, 5)
            self.assertEqual(result.memory_writes, 4)
            self.assertEqual(result.metadata["benchmark"]["trade_metrics"]["offers"], 1)
            self.assertEqual(result.metadata["benchmark"]["trade_metrics"]["accepted"], 1)

            red_memory = orchestrator.memory_store.get("RED")
            blue_memory = orchestrator.memory_store.get("BLUE")
            self.assertIsNotNone(red_memory)
            self.assertIsNotNone(blue_memory)
            assert red_memory is not None and blue_memory is not None
            self.assertEqual(red_memory.content, {"plan": "BLUE may accept wood-for-brick again."})
            self.assertEqual(blue_memory.content, {"belief": "RED will keep paying for brick tempo."})

            self.assertEqual(len(blue_player.recall_observations), 1)
            self.assertEqual(
                tuple(event.kind for event in blue_player.recall_observations[0].public_events_since_last_turn),
                ("trade_offered",),
            )
            self.assertEqual(
                tuple(event.kind for event in blue_player.recall_observations[0].private_events_since_last_turn),
                ("trade_offer_received",),
            )
            self.assertEqual(
                blue_player.observations[0].memory.content,
                {"belief": "RED is pushing brick tempo."},
            )
            self.assertEqual(
                tuple(event.kind for event in red_player.reflection_observations[0].public_events_this_turn),
                ("trade_offered",),
            )

            self.assertTrue(Path(run_dir, "public_history.jsonl").exists())
            self.assertTrue(Path(run_dir, "players", "RED", "memory.json").exists())
            self.assertTrue(Path(run_dir, "players", "RED", "memory_trace.jsonl").exists())
            self.assertTrue(Path(run_dir, "players", "BLUE", "private_history.jsonl").exists())
            self.assertTrue(Path(run_dir, "result.json").exists())

            red_memory_trace_lines = Path(
                run_dir, "players", "RED", "memory_trace.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(red_memory_trace_lines), 2)
            self.assertEqual(json.loads(red_memory_trace_lines[0])["update_kind"], "recall")
            self.assertEqual(json.loads(red_memory_trace_lines[1])["update_kind"], "reflect")

            red_private_history = Path(
                run_dir, "players", "RED", "private_history.jsonl"
            ).read_text(encoding="utf-8")
            self.assertIn('"kind": "player_decision"', red_private_history)
            self.assertIn('"reasoning": "Offer wood now to convert into road tempo."', red_private_history)

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

    def test_owner_led_trade_chat_selects_best_public_quote_and_auto_resolves_bilateral_trade(
        self,
    ) -> None:
        red_player = OwnerTradeChatPlayer(
            action_responses=[Action("END_TURN")],
            open_responses=[
                TradeChatOpenResponse(
                    open_chat=True,
                    message="I need 1 wood. What is your market?",
                    requested_resources={"WOOD": 1},
                )
            ],
            selection_responses=[
                TradeChatSelectionResponse(
                    selected_player_id="ORANGE",
                    message="Taking ORANGE's quote.",
                )
            ],
        )
        white_player = QuoteReplyPlayer(
            TradeChatReplyResponse(
                message="I can do 1 wood for 2 sheep.",
                owner_gives={"SHEEP": 2},
                owner_gets={"WOOD": 1},
            )
        )
        orange_player = QuoteReplyPlayer(
            TradeChatReplyResponse(
                message="I can do 1 wood for 1 sheep.",
                owner_gives={"SHEEP": 1},
                owner_gets={"WOOD": 1},
            )
        )

        orchestrator = GameOrchestrator(
            OwnerLedTradeChatEngine(),
            players={"RED": red_player, "WHITE": white_player, "ORANGE": orange_player},
            trading_chat_enabled=True,
            run_dir=None,
        )

        first_transition = orchestrator.step()
        self.assertEqual(
            [event.kind for event in first_transition.public_events],
            [
                "trade_chat_opened",
                "trade_chat_message",
                "trade_chat_message",
                "trade_chat_message",
                "trade_chat_quote_selected",
                "trade_chat_closed",
                "trade_offered",
            ],
        )
        self.assertEqual(white_player.chat_observations[0].transcript[0].kind, "trade_chat_opened")
        self.assertEqual(
            orange_player.chat_observations[0].transcript[-1].payload["speaker_player_id"],
            "WHITE",
        )

        second_transition = orchestrator.step()
        third_transition = orchestrator.step()
        fourth_transition = orchestrator.step()

        self.assertEqual(second_transition.public_events[0].kind, "trade_rejected")
        self.assertEqual(third_transition.public_events[0].kind, "trade_accepted")
        self.assertEqual(fourth_transition.public_events[0].kind, "trade_confirmed")
        self.assertEqual(orchestrator.engine.current_decision().acting_player_id, "RED")
        self.assertEqual(orchestrator.engine.current_decision().phase, "play_turn")
        self.assertEqual(red_player.open_calls, 1)
        self.assertEqual(red_player.selection_calls, 1)

    def test_failed_trade_chat_attempt_limit_blocks_more_trade_openings_same_turn(self) -> None:
        red_player = OwnerTradeChatPlayer(
            action_responses=[Action("BUILD_ROAD", payload={"edge": [1, 2]}), Action("END_TURN")],
            open_responses=[
                TradeChatOpenResponse(
                    open_chat=True,
                    message="Anyone selling 1 wood?",
                    requested_resources={"WOOD": 1},
                )
            ],
            selection_responses=[TradeChatSelectionResponse(selected_player_id=None)],
        )
        white_player = QuoteReplyPlayer(TradeChatReplyResponse())

        orchestrator = GameOrchestrator(
            FailedTradeAttemptEngine(),
            players={"RED": red_player, "WHITE": white_player},
            trading_chat_enabled=True,
            trading_chat_max_failed_attempts_per_turn=1,
        )

        first_transition = orchestrator.step()
        second_transition = orchestrator.step()

        self.assertIn("trade_chat_no_deal", [event.kind for event in first_transition.public_events])
        self.assertEqual(first_transition.public_events[-1].kind, "road_built")
        self.assertEqual(second_transition.public_events[0].kind, "turn_ended")
        self.assertEqual(red_player.open_calls, 1)


if __name__ == "__main__":
    unittest.main()
