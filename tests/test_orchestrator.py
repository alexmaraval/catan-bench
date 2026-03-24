from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from catan_bench import (
    Action,
    ActionDecision,
    DecisionPoint,
    Event,
    GameOrchestrator,
    InvalidActionError,
    OpeningStrategyResponse,
    ScriptedPlayer,
    TradeChatOpenResponse,
    TradeChatReplyResponse,
    TradeChatSelectionResponse,
    TransitionResult,
    TurnEndResponse,
    TurnStartResponse,
)


class MockTurnEngine:
    def __init__(self) -> None:
        self._game_id = "mock-game"
        self._player_ids = ("RED", "BLUE")
        self._step = 0
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
        match self._step:
            case 0:
                return DecisionPoint(
                    acting_player_id="RED",
                    turn_index=1,
                    phase="play_turn",
                    decision_index=0,
                    prompt="Roll the dice.",
                    legal_actions=(Action("ROLL"),),
                )
            case 1:
                return DecisionPoint(
                    acting_player_id="RED",
                    turn_index=1,
                    phase="play_turn",
                    decision_index=1,
                    prompt="Choose an action.",
                    legal_actions=(
                        Action("OFFER_TRADE", payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}}),
                        Action("END_TURN"),
                    ),
                )
            case 2:
                return DecisionPoint(
                    acting_player_id="BLUE",
                    turn_index=1,
                    phase="decide_trade",
                    decision_index=2,
                    prompt="Respond to RED's trade offer.",
                    legal_actions=(Action("REJECT_TRADE"),),
                )
            case 3:
                return DecisionPoint(
                    acting_player_id="RED",
                    turn_index=1,
                    phase="play_turn",
                    decision_index=3,
                    prompt="Choose an action.",
                    legal_actions=(Action("END_TURN"),),
                )
            case _:
                raise RuntimeError("No more decisions.")

    def public_state(self):
        turn_player_id = "RED"
        if self._step >= 4:
            turn_player_id = "BLUE"
        return {
            "turn": {
                "turn_player_id": turn_player_id,
                "current_player_id": self.current_decision().acting_player_id if not self._terminal else "BLUE",
            },
            "players": {
                "RED": {"vp": 2},
                "BLUE": {"vp": 2},
            },
            "board": {"robber_coordinate": [0, 0, 0]},
            "trade_state": {},
            "bank": {},
        }

    def private_state(self, player_id: str):
        return {
            "player_id": player_id,
            "resources": {"WOOD": 1 if player_id == "RED" else 0, "BRICK": 1 if player_id == "BLUE" else 0},
        }

    def apply_action(self, action: Action) -> TransitionResult:
        match self._step:
            case 0:
                self._step = 1
                return TransitionResult(
                    public_events=(
                        Event(
                            kind="dice_rolled",
                            payload={"result": [3, 4]},
                            turn_index=1,
                            phase="play_turn",
                            decision_index=0,
                            actor_player_id="RED",
                        ),
                    )
                )
            case 1:
                self._step = 2
                return TransitionResult(
                    public_events=(
                        Event(
                            kind="trade_offered",
                            payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
                            turn_index=1,
                            phase="play_turn",
                            decision_index=1,
                            actor_player_id="RED",
                        ),
                    )
                )
            case 2:
                self._step = 3
                return TransitionResult(
                    public_events=(
                        Event(
                            kind="trade_rejected",
                            payload={"offering_player_id": "RED"},
                            turn_index=1,
                            phase="decide_trade",
                            decision_index=2,
                            actor_player_id="BLUE",
                        ),
                    )
                )
            case 3:
                self._step = 4
                self._terminal = True
                return TransitionResult(
                    public_events=(
                        Event(
                            kind="turn_ended",
                            payload={},
                            turn_index=1,
                            phase="play_turn",
                            decision_index=3,
                            actor_player_id="RED",
                        ),
                    ),
                    terminal=True,
                    result_metadata={"winner_ids": ["RED"], "num_turns": 1},
                )
            case _:
                raise RuntimeError("Unexpected action.")

    def result(self):
        return {"winner_ids": ["RED"], "num_turns": 1}


class MockTradeChatEngine:
    def __init__(self) -> None:
        self._game_id = "trade-chat-game"
        self._player_ids = ("RED", "BLUE")
        self._step = 0
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
        match self._step:
            case 0:
                return DecisionPoint("RED", 1, "play_turn", (Action("ROLL"),), 0, "Roll.")
            case 1:
                return DecisionPoint(
                    "RED",
                    1,
                    "play_turn",
                    (
                        Action("OFFER_TRADE", payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}}),
                        Action("END_TURN"),
                    ),
                    1,
                    "Choose an action.",
                )
            case 2:
                return DecisionPoint("BLUE", 1, "decide_trade", (Action("ACCEPT_TRADE"),), 2, "Respond.")
            case 3:
                return DecisionPoint(
                    "RED",
                    1,
                    "decide_acceptees",
                    (Action("CONFIRM_TRADE", payload={"accepting_player_id": "BLUE"}),),
                    3,
                    "Confirm trade.",
                )
            case 4:
                return DecisionPoint("RED", 1, "play_turn", (Action("END_TURN"),), 4, "Choose an action.")
            case _:
                raise RuntimeError("No more decisions.")

    def public_state(self):
        return {
            "turn": {
                "turn_player_id": "RED" if self._step < 5 else "BLUE",
                "current_player_id": self.current_decision().acting_player_id if not self._terminal else "BLUE",
            },
            "players": {"RED": {"vp": 2}, "BLUE": {"vp": 2}},
            "board": {"robber_coordinate": [0, 0, 0]},
            "trade_state": {},
            "bank": {},
        }

    def private_state(self, player_id: str):
        return {"player_id": player_id, "resources": {"WOOD": 1, "BRICK": 1}}

    def apply_action(self, action: Action) -> TransitionResult:
        match self._step:
            case 0:
                self._step = 1
                return TransitionResult(
                    public_events=(Event("dice_rolled", {"result": [2, 5]}, turn_index=1, phase="play_turn", decision_index=0, actor_player_id="RED"),)
                )
            case 1:
                self._step = 2
                return TransitionResult(
                    public_events=(Event("trade_offered", {"offer": {"WOOD": 1}, "request": {"BRICK": 1}}, turn_index=1, phase="play_turn", decision_index=1, actor_player_id="RED"),)
                )
            case 2:
                self._step = 3
                return TransitionResult(
                    public_events=(Event("trade_accepted", {"offering_player_id": "RED"}, turn_index=1, phase="decide_trade", decision_index=2, actor_player_id="BLUE"),)
                )
            case 3:
                self._step = 4
                return TransitionResult(
                    public_events=(Event("trade_confirmed", {"offering_player_id": "RED", "accepting_player_id": "BLUE"}, turn_index=1, phase="decide_acceptees", decision_index=3, actor_player_id="RED"),)
                )
            case 4:
                self._step = 5
                self._terminal = True
                return TransitionResult(
                    public_events=(Event("turn_ended", {}, turn_index=1, phase="play_turn", decision_index=4, actor_player_id="RED"),),
                    terminal=True,
                    result_metadata={"winner_ids": ["RED"], "num_turns": 1},
                )
            case _:
                raise RuntimeError("Unexpected action.")

    def result(self):
        return {"winner_ids": ["RED"], "num_turns": 1}


class MockMultiEventEngine:
    def __init__(self) -> None:
        self._game_id = "multi-event-game"
        self._player_ids = ("RED",)
        self._step = 0
        self._terminal = False

    @property
    def game_id(self) -> str:
        return self._game_id

    @property
    def player_ids(self) -> tuple[str, ...]:
        return self._player_ids

    def is_terminal(self) -> bool:
        return self._terminal

    def current_decision(self) -> DecisionPoint:
        if self._step == 0:
            return DecisionPoint(
                acting_player_id="RED",
                turn_index=1,
                phase="play_turn",
                decision_index=0,
                prompt="Choose an action.",
                legal_actions=(Action("END_TURN"),),
            )
        raise RuntimeError("No more decisions.")

    def public_state(self):
        roads = 0 if self._step == 0 else 1
        return {
            "turn": {"turn_player_id": "RED", "current_player_id": "RED"},
            "players": {"RED": {"vp": 2}},
            "board": {"robber_coordinate": [0, 0, 0], "roads_built": roads},
            "trade_state": {},
            "bank": {},
        }

    def private_state(self, player_id: str):
        return {"player_id": player_id, "resources": {"WOOD": 1, "BRICK": 1}}

    def apply_action(self, action: Action) -> TransitionResult:
        if self._step != 0:
            raise RuntimeError("Unexpected action.")
        self._step = 1
        self._terminal = True
        return TransitionResult(
            public_events=(
                Event(
                    kind="road_built",
                    payload={"edge": [1, 2]},
                    turn_index=1,
                    phase="play_turn",
                    decision_index=0,
                    actor_player_id="RED",
                ),
                Event(
                    kind="turn_ended",
                    payload={},
                    turn_index=1,
                    phase="play_turn",
                    decision_index=0,
                    actor_player_id="RED",
                ),
            ),
            terminal=True,
            result_metadata={"winner_ids": ["RED"], "num_turns": 1},
        )

    def result(self):
        return {"winner_ids": ["RED"], "num_turns": 1}


class MockCounterOfferEngine:
    def __init__(self) -> None:
        self._game_id = "counter-offer-game"
        self._player_ids = ("WHITE", "BLUE")
        self._step = 0
        self._terminal = False

    @property
    def game_id(self) -> str:
        return self._game_id

    @property
    def player_ids(self) -> tuple[str, str]:
        return self._player_ids

    @property
    def turn_owner_id(self) -> str:
        return "WHITE"

    def is_terminal(self) -> bool:
        return self._terminal

    def current_decision(self) -> DecisionPoint:
        if self._step == 0:
            return DecisionPoint(
                acting_player_id="BLUE",
                turn_index=1,
                phase="decide_trade",
                decision_index=0,
                prompt="Respond to WHITE's trade offer.",
                legal_actions=(
                    Action(
                        "REJECT_TRADE",
                        payload={
                            "offer": {"SHEEP": 1},
                            "request": {"BRICK": 1},
                            "offering_player_id": "WHITE",
                        },
                    ),
                    Action(
                        "COUNTER_OFFER",
                        payload={"offer": {}, "request": {}},
                        description="Counter the trade with different terms.",
                    ),
                ),
            )
        raise RuntimeError("No more decisions.")

    def public_state(self):
        return {
            "turn": {"turn_player_id": "WHITE", "current_player_id": "BLUE"},
            "players": {"WHITE": {"vp": 2}, "BLUE": {"vp": 2}},
            "board": {"robber_coordinate": [0, 0, 0]},
            "trade_state": {
                "is_resolving_trade": True,
                "offering_player_id": "WHITE",
                "offer": {"SHEEP": 1},
                "request": {"BRICK": 1},
                "acceptees": [],
            },
            "bank": {},
        }

    def private_state(self, player_id: str):
        return {"player_id": player_id, "resources": {"BRICK": 1 if player_id == "BLUE" else 0}}

    def apply_action(self, action: Action) -> TransitionResult:
        if self._step != 0:
            raise RuntimeError("Unexpected action.")
        if action.action_type != "REJECT_TRADE":
            raise RuntimeError(f"Expected REJECT_TRADE, got {action.action_type}")
        self._step = 1
        self._terminal = True
        return TransitionResult(
            public_events=(
                Event(
                    kind="trade_rejected",
                    payload={"offering_player_id": "WHITE"},
                    turn_index=1,
                    phase="decide_trade",
                    decision_index=0,
                    actor_player_id="BLUE",
                ),
            ),
            terminal=True,
            result_metadata={"winner_ids": ["WHITE"], "num_turns": 1},
        )

    def result(self):
        return {"winner_ids": ["WHITE"], "num_turns": 1}


class MockRepeatRejectedTradeEngine:
    def __init__(self) -> None:
        self._game_id = "repeat-rejected-trade-game"
        self._player_ids = ("RED", "BLUE")
        self._step = 0
        self._terminal = False

    @property
    def game_id(self) -> str:
        return self._game_id

    @property
    def player_ids(self) -> tuple[str, str]:
        return self._player_ids

    @property
    def turn_owner_id(self) -> str:
        return "RED"

    def is_terminal(self) -> bool:
        return self._terminal

    def current_decision(self) -> DecisionPoint:
        match self._step:
            case 0:
                return DecisionPoint("RED", 1, "play_turn", (Action("ROLL"),), 0, "Roll.")
            case 1:
                return DecisionPoint(
                    "RED",
                    1,
                    "play_turn",
                    (
                        Action("OFFER_TRADE", payload={"offer": {}, "request": {}}),
                        Action("END_TURN"),
                    ),
                    1,
                    "Offer or end.",
                )
            case 2:
                return DecisionPoint("BLUE", 1, "decide_trade", (Action("REJECT_TRADE"),), 2, "Respond.")
            case 3:
                return DecisionPoint(
                    "RED",
                    1,
                    "play_turn",
                    (
                        Action("OFFER_TRADE", payload={"offer": {}, "request": {}}),
                        Action("END_TURN"),
                    ),
                    3,
                    "Offer again or end.",
                )
            case _:
                raise RuntimeError("No more decisions.")

    def public_state(self):
        return {
            "turn": {
                "turn_player_id": "RED",
                "current_player_id": (
                    self.current_decision().acting_player_id if not self._terminal else "RED"
                ),
            },
            "players": {"RED": {"vp": 2}, "BLUE": {"vp": 2}},
            "board": {"robber_coordinate": [0, 0, 0]},
            "trade_state": {},
            "bank": {},
        }

    def private_state(self, player_id: str):
        if player_id == "RED":
            return {"player_id": player_id, "resources": {"ORE": 2}}
        return {"player_id": player_id, "resources": {"WOOD": 1}}

    def apply_action(self, action: Action) -> TransitionResult:
        match self._step:
            case 0:
                self._step = 1
                return TransitionResult(
                    public_events=(Event("dice_rolled", {"result": [4, 3]}, turn_index=1, phase="play_turn", decision_index=0, actor_player_id="RED"),)
                )
            case 1:
                self._step = 2
                return TransitionResult(
                    public_events=(
                        Event(
                            "trade_offered",
                            {"offer": dict(action.payload.get("offer", {})), "request": dict(action.payload.get("request", {}))},
                            turn_index=1,
                            phase="play_turn",
                            decision_index=1,
                            actor_player_id="RED",
                        ),
                    )
                )
            case 2:
                self._step = 3
                return TransitionResult(
                    public_events=(
                        Event(
                            "trade_rejected",
                            {"offering_player_id": "RED"},
                            turn_index=1,
                            phase="decide_trade",
                            decision_index=2,
                            actor_player_id="BLUE",
                        ),
                    )
                )
            case 3:
                self._step = 4
                self._terminal = True
                if action.action_type == "OFFER_TRADE":
                    return TransitionResult(
                        public_events=(
                            Event(
                                "trade_offered",
                                {"offer": dict(action.payload.get("offer", {})), "request": dict(action.payload.get("request", {}))},
                                turn_index=1,
                                phase="play_turn",
                                decision_index=3,
                                actor_player_id="RED",
                            ),
                        ),
                        terminal=True,
                        result_metadata={"winner_ids": ["RED"], "num_turns": 1},
                    )
                return TransitionResult(
                    public_events=(Event("turn_ended", {}, turn_index=1, phase="play_turn", decision_index=3, actor_player_id="RED"),),
                    terminal=True,
                    result_metadata={"winner_ids": ["RED"], "num_turns": 1},
                )
            case _:
                raise RuntimeError("Unexpected action.")

    def result(self):
        return {"winner_ids": ["RED"], "num_turns": 1}


class MockRepeatAfterCounterTradeEngine(MockRepeatRejectedTradeEngine):
    def __init__(self) -> None:
        super().__init__()
        self._game_id = "repeat-after-counter-trade-game"

    def current_decision(self) -> DecisionPoint:
        match self._step:
            case 2:
                return DecisionPoint(
                    "BLUE",
                    1,
                    "decide_trade",
                    (
                        Action("REJECT_TRADE"),
                        Action(
                            "COUNTER_OFFER",
                            payload={"offer": {}, "request": {}},
                            description="Counter the trade.",
                        ),
                    ),
                    2,
                    "Respond.",
                )
            case _:
                return super().current_decision()


class MockSelfTradeResponseEngine:
    def __init__(self) -> None:
        self._game_id = "self-trade-response-game"
        self._player_ids = ("BLUE", "WHITE")
        self._step = 0
        self._terminal = False

    @property
    def game_id(self) -> str:
        return self._game_id

    @property
    def player_ids(self) -> tuple[str, str]:
        return self._player_ids

    @property
    def turn_owner_id(self) -> str:
        return "BLUE"

    def is_terminal(self) -> bool:
        return self._terminal

    def current_decision(self) -> DecisionPoint:
        if self._step == 0:
            return DecisionPoint(
                "BLUE",
                1,
                "decide_trade",
                (
                    Action(
                        "REJECT_TRADE",
                        payload={
                            "offer": {"SHEEP": 1},
                            "request": {"BRICK": 1},
                            "offering_player_id": "BLUE",
                        },
                    ),
                ),
                0,
                "Respond to BLUE's trade offer.",
            )
        raise RuntimeError("No more decisions.")

    def public_state(self):
        return {
            "turn": {"turn_player_id": "BLUE", "current_player_id": "BLUE"},
            "players": {"BLUE": {"vp": 2}, "WHITE": {"vp": 2}},
            "board": {"robber_coordinate": [0, 0, 0]},
            "trade_state": {
                "is_resolving_trade": True,
                "offering_player_id": "BLUE",
                "offer": {"SHEEP": 1},
                "request": {"BRICK": 1},
                "acceptees": [],
            },
            "bank": {},
        }

    def private_state(self, player_id: str):
        return {"player_id": player_id, "resources": {}}

    def apply_action(self, action: Action) -> TransitionResult:
        if self._step != 0:
            raise RuntimeError("Unexpected action.")
        self._step = 1
        self._terminal = True
        return TransitionResult(
            public_events=(
                Event(
                    "trade_rejected",
                    {
                        "offering_player_id": "BLUE",
                        "offer": {"SHEEP": 1},
                        "request": {"BRICK": 1},
                        "responding_player_id": "BLUE",
                    },
                    turn_index=1,
                    phase="decide_trade",
                    decision_index=0,
                    actor_player_id="BLUE",
                ),
            ),
            terminal=True,
            result_metadata={"winner_ids": ["WHITE"], "num_turns": 1},
        )

    def result(self):
        return {"winner_ids": ["WHITE"], "num_turns": 1}


class GameOrchestratorTests(unittest.TestCase):
    def test_opening_strategy_phase_runs_for_all_players_before_first_turn_start(self) -> None:
        red = ScriptedPlayer(
            opening_strategy_responses=[
                OpeningStrategyResponse(long_term="Open on wood-brick expansion first.")
            ],
            action_responses=[
                Action(
                    "OFFER_TRADE",
                    payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
                )
            ],
        )
        blue = ScriptedPlayer(
            opening_strategy_responses=[
                OpeningStrategyResponse(long_term="Keep ore for an early development card.")
            ],
            reactive_responses=[Action("REJECT_TRADE")],
        )

        orchestrator = GameOrchestrator(
            MockTurnEngine(),
            {"RED": red, "BLUE": blue},
        )

        orchestrator.step()

        self.assertEqual(len(red.opening_strategy_observations), 1)
        self.assertEqual(len(blue.opening_strategy_observations), 1)
        self.assertEqual(len(red.start_turn_observations), 0)
        self.assertEqual(
            [snapshot.stage for snapshot in orchestrator.memory_store.history("RED")],
            ["opening_strategy"],
        )
        self.assertEqual(
            [snapshot.stage for snapshot in orchestrator.memory_store.history("BLUE")],
            ["opening_strategy"],
        )
        self.assertEqual(orchestrator.memory_store.history("RED")[0].history_index, 0)
        self.assertEqual(orchestrator.memory_store.history("BLUE")[0].history_index, 0)

        orchestrator.step()

        self.assertEqual(len(red.start_turn_observations), 1)
        self.assertEqual(
            [snapshot.stage for snapshot in orchestrator.memory_store.history("RED")],
            ["opening_strategy", "turn_start", "choose_action"],
        )

    def test_can_resume_mid_turn_from_saved_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            initial = GameOrchestrator(
                MockTurnEngine(),
                {
                    "RED": ScriptedPlayer(
                        action_responses=[
                            Action(
                                "OFFER_TRADE",
                                payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
                            )
                        ]
                    ),
                    "BLUE": ScriptedPlayer(reactive_responses=[Action("REJECT_TRADE")]),
                },
                run_dir=tmpdir,
            )

            initial.step()
            initial.step()

            run_dir = Path(initial.run_dir)
            checkpoint_path = run_dir / "checkpoint.json"
            action_trace_path = run_dir / "action_trace.jsonl"

            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(action_trace_path.exists())
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["total_decisions"], 2)

            resumed_red = ScriptedPlayer(action_responses=[Action("END_TURN")])
            resumed = GameOrchestrator(
                MockTurnEngine(),
                {
                    "RED": resumed_red,
                    "BLUE": ScriptedPlayer(reactive_responses=[Action("REJECT_TRADE")]),
                },
                resume_run_dir=run_dir,
            )

            result = resumed.run()

            self.assertEqual(result.total_decisions, 4)
            self.assertEqual(result.winner_ids, ("RED",))
            self.assertEqual(len(resumed.action_trace_store.entries), 4)
            self.assertEqual(len(resumed_red.start_turn_observations), 0)
            self.assertEqual(len(resumed_red.action_observations), 1)
            self.assertEqual(len(resumed_red.end_turn_observations), 1)

    def test_orchestrator_persists_two_slot_memory_without_private_history(self) -> None:
        red = ScriptedPlayer(
            start_turn_responses=[TurnStartResponse(short_term="Offer trade first.")],
            action_responses=[
                ActionDecision(
                    action=Action("OFFER_TRADE", payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}}),
                    short_term="Wait for BLUE's reply.",
                ),
                ActionDecision(action=Action("END_TURN"), short_term="End turn if trade fails."),
            ],
            end_turn_responses=[TurnEndResponse(long_term={"goal": "Try a different trade next time."})],
        )
        blue = ScriptedPlayer(
            reactive_responses=[Action("REJECT_TRADE")],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = GameOrchestrator(
                MockTurnEngine(),
                {"RED": red, "BLUE": blue},
                run_dir=tmpdir,
            )

            result = orchestrator.run()
            run_dir = Path(orchestrator.run_dir)

            self.assertEqual(result.winner_ids, ("RED",))
            self.assertEqual(
                red.action_observations[1].memory.short_term,
                "Offer trade first.\nWait for BLUE's reply.",
            )
            red_history = orchestrator.memory_store.history("RED")
            self.assertEqual(
                [snapshot.stage for snapshot in red_history],
                [
                    "opening_strategy",
                    "turn_start",
                    "choose_action",
                    "choose_action",
                    "turn_end",
                    "turn_cleanup",
                ],
            )
            self.assertEqual(
                red_history[2].memory.short_term,
                "Offer trade first.\nWait for BLUE's reply.",
            )
            self.assertEqual(
                red_history[3].memory.short_term,
                "Offer trade first.\nWait for BLUE's reply.\nEnd turn if trade fails.",
            )
            self.assertEqual(red_history[-1].memory.short_term, None)
            self.assertEqual(red_history[-2].memory.long_term, {"goal": "Try a different trade next time."})
            self.assertFalse(Path(run_dir, "players", "RED", "private_history.jsonl").exists())
            self.assertTrue(Path(run_dir, "public_state_trace.jsonl").exists())

    def test_orchestrator_keeps_trade_chat_public_and_first_class(self) -> None:
        red = ScriptedPlayer(
            start_turn_responses=[TurnStartResponse(short_term={"plan": "Open trade chat."})],
            action_responses=[ActionDecision(action=Action("END_TURN"), short_term={"plan": "Done."})],
            end_turn_responses=[TurnEndResponse(long_term={"goal": "Trade earlier."})],
            trade_chat_open_responses=[
                TradeChatOpenResponse(
                    open_chat=True,
                    message="Need brick.",
                    requested_resources={"BRICK": 1},
                )
            ],
            trade_chat_selection_responses=[
                TradeChatSelectionResponse(selected_player_id="BLUE", message="Let's do it.")
            ],
        )
        blue = ScriptedPlayer(
            trade_chat_reply_responses=[
                TradeChatReplyResponse(
                    message="I can do that.",
                    owner_gives={"WOOD": 1},
                    owner_gets={"BRICK": 1},
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = GameOrchestrator(
                MockTradeChatEngine(),
                {"RED": red, "BLUE": blue},
                run_dir=tmpdir,
                trading_chat_enabled=True,
            )

            result = orchestrator.run()
            run_dir = Path(orchestrator.run_dir)

            self.assertEqual(result.winner_ids, ("RED",))
            event_kinds = [event.kind for event in orchestrator.event_log.public_events]
            self.assertIn("trade_chat_opened", event_kinds)
            self.assertIn("trade_chat_message", event_kinds)
            self.assertIn("trade_chat_quote_selected", event_kinds)
            self.assertIn("trade_confirmed", event_kinds)
            self.assertFalse(Path(run_dir, "players", "RED", "private_history.jsonl").exists())

    def test_orchestrator_snapshots_only_after_final_event_in_transition(self) -> None:
        red = ScriptedPlayer(
            start_turn_responses=[TurnStartResponse(short_term={"plan": "Build then end."})],
            action_responses=[ActionDecision(action=Action("END_TURN"), short_term={"plan": "Done."})],
            end_turn_responses=[TurnEndResponse(long_term={"goal": "Keep expanding."})],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = GameOrchestrator(
                MockMultiEventEngine(),
                {"RED": red},
                run_dir=tmpdir,
            )

            orchestrator.run()

            snapshots = orchestrator.public_state_store.snapshots
            self.assertEqual([snapshot.history_index for snapshot in snapshots], [0, 2])
            self.assertEqual(snapshots[0].public_state["board"]["roads_built"], 0)
            self.assertEqual(snapshots[1].public_state["board"]["roads_built"], 1)

    def test_counter_offer_records_public_event_and_rejects_underlying_trade(self) -> None:
        blue = ScriptedPlayer(
            reactive_responses=[
                Action(
                    "COUNTER_OFFER",
                    payload={"offer": {"BRICK": 1}, "request": {"WOOD": 1}},
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = GameOrchestrator(
                MockCounterOfferEngine(),
                {"WHITE": ScriptedPlayer(), "BLUE": blue},
                run_dir=tmpdir,
            )

            result = orchestrator.run()

            self.assertEqual(result.winner_ids, ("WHITE",))
            event_kinds = [event.kind for event in orchestrator.event_log.public_events]
            self.assertEqual(event_kinds, ["trade_counter_offered", "trade_rejected"])
            counter_event = orchestrator.event_log.public_events[0]
            self.assertEqual(counter_event.payload["owner_player_id"], "WHITE")
            self.assertEqual(counter_event.payload["offer"], {"BRICK": 1})
            self.assertEqual(counter_event.payload["request"], {"WOOD": 1})
            self.assertEqual(orchestrator.action_trace_store.entries[0].action.action_type, "COUNTER_OFFER")

    def test_same_trade_market_gets_retry_feedback_and_continues_after_rejection(self) -> None:
        red = ScriptedPlayer(
            action_responses=[
                Action("OFFER_TRADE", payload={"offer": {"ORE": 1}, "request": {"WOOD": 1}}),
                Action("OFFER_TRADE", payload={"offer": {"ORE": 1}, "request": {"WOOD": 1}}),
                Action("END_TURN"),
            ]
        )
        blue = ScriptedPlayer(reactive_responses=[Action("REJECT_TRADE")])

        orchestrator = GameOrchestrator(
            MockRepeatRejectedTradeEngine(),
            {"RED": red, "BLUE": blue},
        )

        result = orchestrator.run()

        self.assertEqual(result.winner_ids, ("RED",))
        self.assertEqual(len(red.action_observations), 3)
        self.assertIsNotNone(red.action_observations[-1].decision_prompt)
        assert red.action_observations[-1].decision_prompt is not None
        self.assertIn("Previous action was invalid", red.action_observations[-1].decision_prompt)
        self.assertIn("Cannot repeat the same domestic trade market", red.action_observations[-1].decision_prompt)
        self.assertEqual(
            [event.kind for event in orchestrator.event_log.public_events],
            ["dice_rolled", "trade_offered", "trade_rejected", "turn_ended"],
        )

    def test_same_trade_market_can_repeat_after_counteroffer(self) -> None:
        red = ScriptedPlayer(
            action_responses=[
                Action("OFFER_TRADE", payload={"offer": {"ORE": 1}, "request": {"WOOD": 1}}),
                Action("OFFER_TRADE", payload={"offer": {"ORE": 1}, "request": {"WOOD": 1}}),
            ]
        )
        blue = ScriptedPlayer(
            reactive_responses=[
                Action("COUNTER_OFFER", payload={"offer": {"WOOD": 1}, "request": {"ORE": 1}})
            ]
        )

        orchestrator = GameOrchestrator(
            MockRepeatAfterCounterTradeEngine(),
            {"RED": red, "BLUE": blue},
        )

        orchestrator.step()
        orchestrator.step()
        orchestrator.step()
        transition = orchestrator.step()

        self.assertTrue(any(event.kind == "trade_offered" for event in transition.public_events))
        latest_offer = transition.public_events[0]
        self.assertEqual(latest_offer.payload["offer"], {"ORE": 1})
        self.assertEqual(latest_offer.payload["request"], {"WOOD": 1})

    def test_trade_owner_is_not_prompted_to_respond_to_own_offer(self) -> None:
        blue = ScriptedPlayer()
        orchestrator = GameOrchestrator(
            MockSelfTradeResponseEngine(),
            {"BLUE": blue, "WHITE": ScriptedPlayer()},
        )

        result = orchestrator.run()

        self.assertEqual(result.winner_ids, ("WHITE",))
        self.assertEqual(len(blue.reactive_observations), 0)
        self.assertEqual(len(orchestrator.action_trace_store.entries), 1)
        self.assertEqual(orchestrator.action_trace_store.entries[0].action.action_type, "REJECT_TRADE")


if __name__ == "__main__":
    unittest.main()
