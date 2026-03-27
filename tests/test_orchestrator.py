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
    OpeningStrategyResponse,
    PublicChatDraft,
    ScriptedPlayer,
    TradeChatOpenResponse,
    TradeChatOwnerDecisionResponse,
    TradeChatReplyResponse,
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
                        Action(
                            "OFFER_TRADE",
                            payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
                        ),
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
                "current_player_id": self.current_decision().acting_player_id
                if not self._terminal
                else "BLUE",
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
            "resources": {
                "WOOD": 1 if player_id == "RED" else 0,
                "BRICK": 1 if player_id == "BLUE" else 0,
            },
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


class MockResolveActionValueErrorEngine:
    def __init__(self) -> None:
        self._game_id = "mock-resolve-error-game"
        self._player_ids = ("ORANGE",)
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
                acting_player_id="ORANGE",
                turn_index=133,
                phase="play_turn",
                decision_index=0,
                prompt="Choose an action.",
                legal_actions=(
                    Action(
                        "OFFER_TRADE",
                        payload={"offer": {"ORE": 1}, "request": {"SHEEP": 2}},
                    ),
                    Action("END_TURN"),
                ),
            )
        raise RuntimeError("No more decisions.")

    def resolve_action(
        self, *, proposed_action: Action, legal_actions: tuple[Action, ...]
    ) -> Action:
        del legal_actions
        if proposed_action.action_type == "OFFER_TRADE":
            raise ValueError(
                "Action {'action_type': 'OFFER_TRADE', 'payload': {'offer': {'ORE': 1}, "
                "'request': {'ORE': 1, 'SHEEP': 1}}} is not currently valid in catanatron."
            )
        return proposed_action

    def public_state(self):
        return {
            "turn": {"turn_player_id": "ORANGE", "current_player_id": "ORANGE"},
            "players": {"ORANGE": {"vp": 2}},
            "board": {"robber_coordinate": [0, 0, 0]},
            "trade_state": {},
            "bank": {},
        }

    def private_state(self, player_id: str):
        return {"player_id": player_id, "resources": {"ORE": 1, "SHEEP": 2}}

    def apply_action(self, action: Action) -> TransitionResult:
        self._step = 1
        self._terminal = True
        return TransitionResult(
            public_events=(
                Event(
                    kind="turn_ended",
                    payload={},
                    turn_index=133,
                    phase="play_turn",
                    decision_index=0,
                    actor_player_id="ORANGE",
                ),
            ),
            terminal=True,
            result_metadata={"winner_ids": ["ORANGE"], "num_turns": 133},
        )

    def result(self):
        return {"winner_ids": ["ORANGE"], "num_turns": 133}


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
                return DecisionPoint(
                    "RED", 1, "play_turn", (Action("ROLL"),), 0, "Roll."
                )
            case 1:
                return DecisionPoint(
                    "RED",
                    1,
                    "play_turn",
                    (
                        Action(
                            "OFFER_TRADE",
                            payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
                        ),
                        Action("END_TURN"),
                    ),
                    1,
                    "Choose an action.",
                )
            case 2:
                return DecisionPoint(
                    "BLUE", 1, "decide_trade", (Action("ACCEPT_TRADE"),), 2, "Respond."
                )
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
                return DecisionPoint(
                    "RED", 1, "play_turn", (Action("END_TURN"),), 4, "Choose an action."
                )
            case _:
                raise RuntimeError("No more decisions.")

    def public_state(self):
        return {
            "turn": {
                "turn_player_id": "RED" if self._step < 5 else "BLUE",
                "current_player_id": self.current_decision().acting_player_id
                if not self._terminal
                else "BLUE",
            },
            "players": {"RED": {"vp": 2}, "BLUE": {"vp": 2}},
            "board": {"robber_coordinate": [0, 0, 0]},
            "trade_state": {},
            "bank": {},
        }

    def private_state(self, player_id: str):
        return {
            "player_id": player_id,
            "resources": {"WOOD": 1, "BRICK": 1, "SHEEP": 1},
        }

    def apply_action(self, action: Action) -> TransitionResult:
        match self._step:
            case 0:
                self._step = 1
                return TransitionResult(
                    public_events=(
                        Event(
                            "dice_rolled",
                            {"result": [2, 5]},
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
                            "trade_offered",
                            {"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
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
                            "trade_accepted",
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
                return TransitionResult(
                    public_events=(
                        Event(
                            "trade_confirmed",
                            {
                                "offering_player_id": "RED",
                                "accepting_player_id": "BLUE",
                            },
                            turn_index=1,
                            phase="decide_acceptees",
                            decision_index=3,
                            actor_player_id="RED",
                        ),
                    )
                )
            case 4:
                self._step = 5
                self._terminal = True
                return TransitionResult(
                    public_events=(
                        Event(
                            "turn_ended",
                            {},
                            turn_index=1,
                            phase="play_turn",
                            decision_index=4,
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


class MockTradeChatResumeTimeoutEngine(MockTradeChatEngine):
    def __init__(self) -> None:
        super().__init__()
        self._game_id = "trade-chat-resume-timeout-game"

    def apply_action(self, action: Action) -> TransitionResult:
        if self._step == 1 and action.action_type == "END_TURN":
            self._step = 5
            self._terminal = True
            return TransitionResult(
                public_events=(
                    Event(
                        "turn_ended",
                        {},
                        turn_index=1,
                        phase="play_turn",
                        decision_index=1,
                        actor_player_id="RED",
                    ),
                ),
                terminal=True,
                result_metadata={"winner_ids": ["RED"], "num_turns": 1},
            )
        return super().apply_action(action)


class MockTradeChatInvalidSelectionEngine:
    def __init__(self) -> None:
        self._game_id = "trade-chat-invalid-selection-game"
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
                    "RED", 1, "play_turn", (Action("ROLL"),), 0, "Roll."
                )
            case 1:
                return DecisionPoint(
                    "RED",
                    1,
                    "play_turn",
                    (
                        Action(
                            "OFFER_TRADE",
                            payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
                        ),
                        Action("END_TURN"),
                    ),
                    1,
                    "Choose an action.",
                )
            case 2:
                return DecisionPoint(
                    "BLUE", 1, "decide_trade", (Action("REJECT_TRADE"),), 2, "Respond."
                )
            case 3:
                return DecisionPoint(
                    "RED",
                    1,
                    "decide_acceptees",
                    (Action("CANCEL_TRADE"),),
                    3,
                    "Cancel trade.",
                )
            case 4:
                return DecisionPoint(
                    "RED", 1, "play_turn", (Action("END_TURN"),), 4, "Choose an action."
                )
            case _:
                raise RuntimeError("No more decisions.")

    def public_state(self):
        return {
            "turn": {
                "turn_player_id": "RED" if self._step < 5 else "BLUE",
                "current_player_id": self.current_decision().acting_player_id
                if not self._terminal
                else "BLUE",
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
                    public_events=(
                        Event(
                            "dice_rolled",
                            {"result": [2, 5]},
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
                            "trade_offered",
                            {"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
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
                return TransitionResult(
                    public_events=(
                        Event(
                            "trade_cancelled",
                            {"offering_player_id": "RED"},
                            turn_index=1,
                            phase="decide_acceptees",
                            decision_index=3,
                            actor_player_id="RED",
                        ),
                    )
                )
            case 4:
                self._step = 5
                self._terminal = True
                return TransitionResult(
                    public_events=(
                        Event(
                            "turn_ended",
                            {},
                            turn_index=1,
                            phase="play_turn",
                            decision_index=4,
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


class MockTradeChatInsufficientOwnerResourcesEngine:
    def __init__(self) -> None:
        self._game_id = "trade-chat-insufficient-owner-resources-game"
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
                    "RED", 1, "play_turn", (Action("ROLL"),), 0, "Roll."
                )
            case 1:
                return DecisionPoint(
                    "RED",
                    1,
                    "play_turn",
                    (
                        Action(
                            "OFFER_TRADE",
                            payload={"offer": {"BRICK": 1}, "request": {"WOOD": 1}},
                        ),
                        Action("END_TURN"),
                    ),
                    1,
                    "Choose an action.",
                )
            case 2:
                raise RuntimeError("No more decisions.")
            case _:
                raise RuntimeError("Unexpected step.")

    def public_state(self):
        return {
            "turn": {
                "turn_player_id": "RED" if not self._terminal else "BLUE",
                "current_player_id": self.current_decision().acting_player_id
                if not self._terminal
                else "BLUE",
            },
            "players": {"RED": {"vp": 2}, "BLUE": {"vp": 2}},
            "board": {"robber_coordinate": [0, 0, 0]},
            "trade_state": {},
            "bank": {},
        }

    def private_state(self, player_id: str):
        if player_id == "RED":
            return {"player_id": player_id, "resources": {"WOOD": 0, "BRICK": 1}}
        return {"player_id": player_id, "resources": {"WOOD": 1, "BRICK": 1}}

    def apply_action(self, action: Action) -> TransitionResult:
        match self._step:
            case 0:
                self._step = 1
                return TransitionResult(
                    public_events=(
                        Event(
                            "dice_rolled",
                            {"result": [4, 3]},
                            turn_index=1,
                            phase="play_turn",
                            decision_index=0,
                            actor_player_id="RED",
                        ),
                    )
                )
            case 1:
                if action.action_type != "END_TURN":
                    raise RuntimeError(
                        f"Expected END_TURN after no-deal, got {action.action_type}"
                    )
                self._step = 2
                self._terminal = True
                return TransitionResult(
                    public_events=(
                        Event(
                            "turn_ended",
                            {},
                            turn_index=1,
                            phase="play_turn",
                            decision_index=1,
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
        return {
            "player_id": player_id,
            "resources": {"BRICK": 1 if player_id == "BLUE" else 0},
        }

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
                return DecisionPoint(
                    "RED", 1, "play_turn", (Action("ROLL"),), 0, "Roll."
                )
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
                return DecisionPoint(
                    "BLUE", 1, "decide_trade", (Action("REJECT_TRADE"),), 2, "Respond."
                )
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
                    self.current_decision().acting_player_id
                    if not self._terminal
                    else "RED"
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
                    public_events=(
                        Event(
                            "dice_rolled",
                            {"result": [4, 3]},
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
                            "trade_offered",
                            {
                                "offer": dict(action.payload.get("offer", {})),
                                "request": dict(action.payload.get("request", {})),
                            },
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
                                {
                                    "offer": dict(action.payload.get("offer", {})),
                                    "request": dict(action.payload.get("request", {})),
                                },
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
                    public_events=(
                        Event(
                            "turn_ended",
                            {},
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


class MockPublicChatEngine:
    def __init__(self) -> None:
        self._game_id = "mock-public-chat-game"
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
        if self._step == 0:
            return DecisionPoint(
                acting_player_id="RED",
                turn_index=1,
                phase="play_turn",
                decision_index=0,
                prompt="Roll.",
                legal_actions=(Action("ROLL"),),
            )
        if self._step == 1:
            return DecisionPoint(
                acting_player_id="RED",
                turn_index=1,
                phase="play_turn",
                decision_index=1,
                prompt="Choose an action.",
                legal_actions=(Action("END_TURN"),),
            )
        raise RuntimeError("No more decisions.")

    def public_state(self):
        return {
            "turn": {
                "turn_player_id": "RED",
                "current_player_id": "BLUE" if self._terminal else "RED",
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
        return {"player_id": player_id, "resources": {"WOOD": 1}}

    def apply_action(self, action: Action) -> TransitionResult:
        if self._step == 0:
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
        if self._step == 1:
            self._step = 2
            self._terminal = True
            return TransitionResult(
                public_events=(
                    Event(
                        kind="turn_ended",
                        payload={},
                        turn_index=1,
                        phase="play_turn",
                        decision_index=1,
                        actor_player_id="RED",
                    ),
                ),
                terminal=True,
                result_metadata={"winner_ids": ["RED"], "num_turns": 1},
            )
        raise RuntimeError("Unexpected action.")

    def result(self):
        return {"winner_ids": ["RED"], "num_turns": 1}


class GameOrchestratorTests(unittest.TestCase):
    def test_can_resume_after_timeout_during_trade_chat_owner_decision(self) -> None:
        class TimeoutOwnerDecisionPlayer(ScriptedPlayer):
            def decide_trade_chat(self, observation):
                self.trade_chat_observations.append(observation)
                raise TimeoutError("owner decision timed out")

        red = TimeoutOwnerDecisionPlayer(
            action_responses=[
                ActionDecision(
                    action=Action(
                        "OFFER_TRADE",
                        payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
                    )
                )
            ],
            trade_chat_open_responses=[
                TradeChatOpenResponse(
                    open_chat=True,
                    message="Need brick.",
                    requested_resources={"BRICK": 1},
                )
            ],
        )
        blue = ScriptedPlayer(
            trade_chat_reply_responses=[
                TradeChatReplyResponse(
                    message="I can do that for sheep.",
                    owner_gives={"SHEEP": 1},
                    owner_gets={"BRICK": 1},
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = GameOrchestrator(
                MockTradeChatResumeTimeoutEngine(),
                {"RED": red, "BLUE": blue},
                run_dir=tmpdir,
                trading_chat_enabled=True,
            )

            orchestrator.step()
            with self.assertRaisesRegex(TimeoutError, "timed out"):
                orchestrator.step()

            run_dir = Path(orchestrator.run_dir)
            checkpoint = json.loads(
                (run_dir / "checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["total_decisions"], 1)
            self.assertEqual(
                checkpoint["current_history_index"],
                len(orchestrator.event_log.public_events),
            )
            self.assertEqual(
                [event.kind for event in orchestrator.event_log.public_events],
                [
                    "dice_rolled",
                    "trade_chat_opened",
                    "trade_chat_message",
                    "trade_chat_message",
                ],
            )

            resumed_red = ScriptedPlayer(action_responses=[Action("END_TURN")])
            resumed = GameOrchestrator(
                MockTradeChatResumeTimeoutEngine(),
                {"RED": resumed_red, "BLUE": ScriptedPlayer()},
                resume_run_dir=run_dir,
                trading_chat_enabled=True,
            )

            result = resumed.run()

            self.assertEqual(result.winner_ids, ("RED",))
            self.assertEqual(result.total_decisions, 2)
            self.assertEqual(
                resumed.action_trace_store.entries[-1].action.action_type, "END_TURN"
            )

    def test_opening_strategy_phase_runs_for_all_players_before_first_turn_start(
        self,
    ) -> None:
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
                OpeningStrategyResponse(
                    long_term="Keep ore for an early development card."
                )
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

    def test_resume_rejects_checkpoint_history_index_mismatch(self) -> None:
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
            history_path = run_dir / "public_history.jsonl"
            state_path = run_dir / "public_state_trace.jsonl"
            history_lines = history_path.read_text(encoding="utf-8").splitlines()
            state_lines = state_path.read_text(encoding="utf-8").splitlines()
            history_path.write_text(history_lines[0] + "\n", encoding="utf-8")
            state_path.write_text("\n".join(state_lines[:2]) + "\n", encoding="utf-8")

            resumed = GameOrchestrator(
                MockTurnEngine(),
                {
                    "RED": ScriptedPlayer(action_responses=[Action("END_TURN")]),
                    "BLUE": ScriptedPlayer(reactive_responses=[Action("REJECT_TRADE")]),
                },
                resume_run_dir=run_dir,
            )

            with self.assertRaisesRegex(RuntimeError, "current_history_index"):
                resumed.step()

    def test_resume_rejects_public_state_snapshot_mismatch(self) -> None:
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
            history_path = run_dir / "public_history.jsonl"
            state_path = run_dir / "public_state_trace.jsonl"
            checkpoint_path = run_dir / "checkpoint.json"
            history_lines = history_path.read_text(encoding="utf-8").splitlines()
            state_lines = state_path.read_text(encoding="utf-8").splitlines()
            history_path.write_text(history_lines[0] + "\n", encoding="utf-8")
            state_path.write_text("\n".join(state_lines[:2]) + "\n", encoding="utf-8")

            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["current_history_index"] = 1
            checkpoint_path.write_text(
                json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            resumed = GameOrchestrator(
                MockTurnEngine(),
                {
                    "RED": ScriptedPlayer(action_responses=[Action("END_TURN")]),
                    "BLUE": ScriptedPlayer(reactive_responses=[Action("REJECT_TRADE")]),
                },
                resume_run_dir=run_dir,
            )

            with self.assertRaisesRegex(RuntimeError, "public state snapshot"):
                resumed.step()

    def test_orchestrator_persists_two_slot_memory_without_private_history(
        self,
    ) -> None:
        red = ScriptedPlayer(
            start_turn_responses=[TurnStartResponse(short_term="Offer trade first.")],
            action_responses=[
                ActionDecision(
                    action=Action(
                        "OFFER_TRADE",
                        payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
                    ),
                    short_term="Wait for BLUE's reply.",
                ),
                ActionDecision(
                    action=Action("END_TURN"), short_term="End turn if trade fails."
                ),
            ],
            end_turn_responses=[
                TurnEndResponse(long_term={"goal": "Try a different trade next time."})
            ],
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
            self.assertEqual(
                red_history[-2].memory.long_term,
                {"goal": "Try a different trade next time."},
            )
            self.assertFalse(
                Path(run_dir, "players", "RED", "private_history.jsonl").exists()
            )
            self.assertTrue(Path(run_dir, "public_state_trace.jsonl").exists())

    def test_orchestrator_keeps_trade_chat_public_and_first_class(self) -> None:
        red = ScriptedPlayer(
            start_turn_responses=[
                TurnStartResponse(short_term={"plan": "Open trade chat."})
            ],
            action_responses=[
                ActionDecision(
                    action=Action(
                        "OFFER_TRADE",
                        payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
                    ),
                    short_term={"plan": "Try the table first."},
                ),
                ActionDecision(action=Action("END_TURN"), short_term={"plan": "Done."}),
            ],
            end_turn_responses=[TurnEndResponse(long_term={"goal": "Trade earlier."})],
            trade_chat_open_responses=[
                TradeChatOpenResponse(
                    open_chat=True,
                    message="Need brick.",
                    requested_resources={"BRICK": 1},
                )
            ],
            trade_chat_owner_decision_responses=[
                TradeChatOwnerDecisionResponse(
                    decision="continue", message="I can offer sheep instead."
                ),
                TradeChatOwnerDecisionResponse(
                    decision="select",
                    selected_proposal_id="attempt-1-round-2-proposal-1",
                    message="Let's do it.",
                ),
            ],
        )
        blue = ScriptedPlayer(
            trade_chat_reply_responses=[
                TradeChatReplyResponse(
                    message="Not for wood.",
                    owner_gives={},
                    owner_gets={},
                ),
                TradeChatReplyResponse(
                    message="I can do that for sheep.",
                    owner_gives={"SHEEP": 1},
                    owner_gets={"BRICK": 1},
                ),
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
            events = orchestrator.event_log.public_events
            event_kinds = [event.kind for event in events]
            self.assertEqual(
                event_kinds,
                [
                    "dice_rolled",
                    "trade_chat_opened",
                    "trade_chat_message",
                    "trade_chat_message",
                    "trade_chat_message",
                    "trade_chat_message",
                    "trade_chat_quote_selected",
                    "trade_chat_closed",
                    "trade_confirmed",
                    "turn_ended",
                ],
            )
            selected_event = next(
                event for event in events if event.kind == "trade_chat_quote_selected"
            )
            self.assertEqual(selected_event.payload["selected_player_id"], "BLUE")
            self.assertEqual(
                selected_event.payload["selected_proposal_id"],
                "attempt-1-round-2-proposal-1",
            )
            reply_events = [
                event
                for event in events
                if event.kind == "trade_chat_message"
                and event.actor_player_id == "BLUE"
            ]
            self.assertEqual(
                [event.payload["round_index"] for event in reply_events], [1, 2]
            )
            self.assertEqual(reply_events[-1].payload["offer"], {"SHEEP": 1})
            self.assertEqual(reply_events[-1].payload["request"], {"BRICK": 1})
            self.assertEqual(len(red.action_observations), 2)
            self.assertTrue(red.action_observations[0].trade_chat_enabled)
            self.assertEqual(
                red.action_observations[0].trade_chat_attempts_remaining, 5
            )
            self.assertIsNone(red.action_observations[1].trade_chat_attempts_remaining)
            self.assertEqual(red.action_observations[1].decision_index, 4)
            self.assertFalse(
                Path(run_dir, "players", "RED", "private_history.jsonl").exists()
            )

    def test_trade_chat_observations_include_recent_persistent_public_chat(self) -> None:
        red = ScriptedPlayer(
            start_turn_responses=[
                TurnStartResponse(
                    short_term={"plan": "Open trade chat."},
                    public_chat=PublicChatDraft(
                        message="BLUE, WHITE is the real threat.",
                        target_player_id="BLUE",
                    ),
                )
            ],
            action_responses=[
                ActionDecision(
                    action=Action(
                        "OFFER_TRADE",
                        payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
                    ),
                    short_term={"plan": "Try the table first."},
                ),
                ActionDecision(action=Action("END_TURN"), short_term={"plan": "Done."}),
            ],
            trade_chat_open_responses=[
                TradeChatOpenResponse(
                    open_chat=True,
                    message="Need brick.",
                    requested_resources={"BRICK": 1},
                )
            ],
            trade_chat_owner_decision_responses=[
                TradeChatOwnerDecisionResponse(
                    decision="select",
                    selected_proposal_id="attempt-1-round-1-proposal-1",
                    message="Deal.",
                ),
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

        orchestrator = GameOrchestrator(
            MockTradeChatEngine(),
            {"RED": red, "BLUE": blue},
            public_chat_enabled=True,
            trading_chat_enabled=True,
        )

        orchestrator.run()

        owner_trade_obs = red.trade_chat_observations[0]
        counterparty_trade_obs = blue.trade_chat_observations[0]
        self.assertEqual(
            owner_trade_obs.public_chat_transcript[0].kind, "public_chat_message"
        )
        self.assertEqual(
            counterparty_trade_obs.public_chat_transcript[0].kind,
            "public_chat_message",
        )
        self.assertEqual(
            owner_trade_obs.public_chat_transcript[0].payload["message"],
            "BLUE, WHITE is the real threat.",
        )
        self.assertEqual(
            counterparty_trade_obs.public_chat_transcript[0].payload["message"],
            "BLUE, WHITE is the real threat.",
        )

    def test_orchestrator_records_persistent_public_chat_and_surfaces_transcript(
        self,
    ) -> None:
        red = ScriptedPlayer(
            start_turn_responses=[
                TurnStartResponse(
                    short_term={"plan": "End quietly."},
                    public_chat=PublicChatDraft(
                        message="BLUE, do not feed RED the road.",
                        target_player_id="BLUE",
                    ),
                )
            ],
            action_responses=[
                ActionDecision(
                    action=Action("END_TURN"),
                    short_term={"plan": "Done."},
                    public_chat=PublicChatDraft(message="I am done here."),
                )
            ],
            end_turn_responses=[
                TurnEndResponse(
                    long_term={"goal": "Keep pressure on BLUE."},
                    public_chat=PublicChatDraft(
                        message="BLUE, your move.",
                        target_player_id="BLUE",
                    ),
                )
            ],
        )
        blue = ScriptedPlayer()

        orchestrator = GameOrchestrator(
            MockPublicChatEngine(),
            {"RED": red, "BLUE": blue},
            public_chat_enabled=True,
        )

        result = orchestrator.run()

        self.assertEqual(result.winner_ids, ("RED",))
        events = orchestrator.event_log.public_events
        self.assertEqual(
            [event.kind for event in events],
            [
                "dice_rolled",
                "public_chat_message",
                "public_chat_message",
                "turn_ended",
                "public_chat_message",
            ],
        )
        self.assertEqual(events[1].payload["source_stage"], "turn_start")
        self.assertEqual(events[1].payload["target_player_id"], "BLUE")
        self.assertEqual(events[2].payload["source_stage"], "choose_action")
        self.assertEqual(events[4].payload["source_stage"], "turn_end")
        self.assertEqual(
            red.action_observations[0].public_chat_transcript[0].kind,
            "public_chat_message",
        )
        self.assertEqual(len(red.end_turn_observations[0].public_chat_transcript), 2)

    def test_orchestrator_handles_invalid_selected_trade_without_crashing(self) -> None:
        red = ScriptedPlayer(
            start_turn_responses=[
                TurnStartResponse(short_term={"plan": "Open trade chat."})
            ],
            action_responses=[
                ActionDecision(
                    action=Action(
                        "OFFER_TRADE",
                        payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
                    ),
                    short_term={"plan": "Try the table first."},
                ),
                ActionDecision(action=Action("END_TURN"), short_term={"plan": "Done."}),
            ],
            end_turn_responses=[
                TurnEndResponse(long_term={"goal": "Try a different trade next time."})
            ],
            trade_chat_open_responses=[
                TradeChatOpenResponse(
                    open_chat=True,
                    message="Need brick.",
                    requested_resources={"BRICK": 1},
                )
            ],
            trade_chat_owner_decision_responses=[
                TradeChatOwnerDecisionResponse(
                    decision="select",
                    selected_proposal_id="attempt-1-round-1-proposal-1",
                    message="Let's do it.",
                ),
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
                MockTradeChatInvalidSelectionEngine(),
                {"RED": red, "BLUE": blue},
                run_dir=tmpdir,
                trading_chat_enabled=True,
            )

            result = orchestrator.run()

            self.assertEqual(result.winner_ids, ("RED",))
            self.assertEqual(
                [event.kind for event in orchestrator.event_log.public_events],
                [
                    "dice_rolled",
                    "trade_chat_opened",
                    "trade_chat_message",
                    "trade_chat_message",
                    "trade_chat_quote_selected",
                    "trade_chat_closed",
                    "turn_ended",
                ],
            )

    def test_orchestrator_recovers_trade_chat_selection_from_invalid_hint(self) -> None:
        red = ScriptedPlayer(
            start_turn_responses=[
                TurnStartResponse(short_term={"plan": "Open trade chat."})
            ],
            action_responses=[
                ActionDecision(
                    action=Action(
                        "OFFER_TRADE",
                        payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
                    ),
                    short_term={"plan": "Try the table first."},
                ),
                ActionDecision(action=Action("END_TURN"), short_term={"plan": "Done."}),
            ],
            end_turn_responses=[TurnEndResponse(long_term={"goal": "Trade earlier."})],
            trade_chat_open_responses=[
                TradeChatOpenResponse(
                    open_chat=True,
                    message="Need brick.",
                    requested_resources={"BRICK": 1},
                )
            ],
            trade_chat_owner_decision_responses=[
                TradeChatOwnerDecisionResponse(
                    decision="select",
                    selected_proposal_id="BLUE_WOOD_FOR_BRICK",
                    message="Let's do it.",
                ),
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

        orchestrator = GameOrchestrator(
            MockTradeChatEngine(),
            {"RED": red, "BLUE": blue},
            trading_chat_enabled=True,
        )

        result = orchestrator.run()

        self.assertEqual(result.winner_ids, ("RED",))
        selected_event = next(
            event
            for event in orchestrator.event_log.public_events
            if event.kind == "trade_chat_quote_selected"
        )
        self.assertEqual(selected_event.payload["selected_player_id"], "BLUE")
        self.assertEqual(
            selected_event.payload["selected_proposal_id"],
            "attempt-1-round-1-proposal-1",
        )
        self.assertFalse(
            any(
                event.kind == "trade_chat_no_deal"
                for event in orchestrator.event_log.public_events
            )
        )

    def test_orchestrator_filters_trade_chat_proposals_owner_cannot_pay(self) -> None:
        red = ScriptedPlayer(
            start_turn_responses=[
                TurnStartResponse(short_term={"plan": "Open trade chat."})
            ],
            action_responses=[
                ActionDecision(
                    action=Action(
                        "OFFER_TRADE",
                        payload={"offer": {"BRICK": 1}, "request": {"WOOD": 1}},
                    ),
                    short_term={"plan": "Try the table first."},
                ),
                ActionDecision(action=Action("END_TURN"), short_term={"plan": "Done."}),
            ],
            end_turn_responses=[
                TurnEndResponse(long_term={"goal": "Wait for better resources."})
            ],
            trade_chat_open_responses=[
                TradeChatOpenResponse(
                    open_chat=True,
                    message="Need wood.",
                    requested_resources={"WOOD": 1},
                )
            ],
            trade_chat_owner_decision_responses=[
                TradeChatOwnerDecisionResponse(
                    decision="select",
                    selected_proposal_id="attempt-1-round-1-proposal-1",
                    message="Let's do it.",
                ),
            ],
        )
        blue = ScriptedPlayer(
            trade_chat_reply_responses=[
                TradeChatReplyResponse(
                    message="I can do wood for brick.",
                    owner_gives={"WOOD": 1},
                    owner_gets={"BRICK": 1},
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = GameOrchestrator(
                MockTradeChatInsufficientOwnerResourcesEngine(),
                {"RED": red, "BLUE": blue},
                run_dir=tmpdir,
                trading_chat_enabled=True,
            )

            result = orchestrator.run()

            self.assertEqual(result.winner_ids, ("RED",))
            self.assertEqual(
                [event.kind for event in orchestrator.event_log.public_events],
                [
                    "dice_rolled",
                    "trade_chat_opened",
                    "trade_chat_message",
                    "trade_chat_message",
                    "trade_chat_no_deal",
                    "trade_chat_closed",
                    "turn_ended",
                ],
            )
            self.assertFalse(
                any(
                    event.kind == "trade_confirmed"
                    for event in orchestrator.event_log.public_events
                )
            )

    def test_orchestrator_rejects_duplicate_trade_chat_room_in_same_turn(self) -> None:
        red = ScriptedPlayer(
            start_turn_responses=[
                TurnStartResponse(short_term={"plan": "Ask for wood once."})
            ],
            action_responses=[
                ActionDecision(
                    action=Action(
                        "OFFER_TRADE",
                        payload={"offer": {"BRICK": 1}, "request": {"WOOD": 1}},
                    ),
                    short_term={"plan": "Try the table."},
                ),
                ActionDecision(
                    action=Action(
                        "OFFER_TRADE",
                        payload={"offer": {"BRICK": 1}, "request": {"WOOD": 1}},
                    ),
                    short_term={"plan": "Try the exact same room again."},
                ),
                ActionDecision(
                    action=Action("END_TURN"), short_term={"plan": "Move on."}
                ),
            ],
            end_turn_responses=[
                TurnEndResponse(long_term={"goal": "Change the ask next time."})
            ],
            trade_chat_open_responses=[
                TradeChatOpenResponse(
                    open_chat=True,
                    message="Need wood.",
                    requested_resources={"WOOD": 1},
                ),
                TradeChatOpenResponse(
                    open_chat=True,
                    message="Need wood.",
                    requested_resources={"WOOD": 1},
                ),
            ],
            trade_chat_owner_decision_responses=[
                TradeChatOwnerDecisionResponse(decision="close", message="No deal."),
            ],
        )
        blue = ScriptedPlayer()

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = GameOrchestrator(
                MockTradeChatInsufficientOwnerResourcesEngine(),
                {"RED": red, "BLUE": blue},
                run_dir=tmpdir,
                trading_chat_enabled=True,
            )

            result = orchestrator.run()

            self.assertEqual(result.winner_ids, ("RED",))
            self.assertEqual(
                [event.kind for event in orchestrator.event_log.public_events],
                [
                    "dice_rolled",
                    "trade_chat_opened",
                    "trade_chat_message",
                    "trade_chat_no_deal",
                    "trade_chat_closed",
                    "turn_ended",
                ],
            )
            self.assertEqual(red.action_observations[-1].decision_index, 1)
            self.assertEqual(red.trade_chat_observations[0].attempt_index, 1)
            self.assertEqual(
                orchestrator._trade_chat_turn_state.opened_attempts
                if orchestrator._trade_chat_turn_state
                else None,
                1,
            )

    def test_orchestrator_falls_back_after_repeated_invalid_duplicate_trade_chat_retries(
        self,
    ) -> None:
        red = ScriptedPlayer(
            start_turn_responses=[
                TurnStartResponse(short_term={"plan": "Ask for wood once."})
            ],
            action_responses=[
                ActionDecision(
                    action=Action(
                        "OFFER_TRADE",
                        payload={"offer": {"BRICK": 1}, "request": {"WOOD": 1}},
                    ),
                    short_term={"plan": "Try the table."},
                ),
                ActionDecision(
                    action=Action(
                        "OFFER_TRADE",
                        payload={"offer": {"BRICK": 1}, "request": {"WOOD": 1}},
                    ),
                    short_term={"plan": "Try the same room again."},
                ),
                ActionDecision(
                    action=Action(
                        "OFFER_TRADE",
                        payload={"offer": {"BRICK": 1}, "request": {"WOOD": 1}},
                    ),
                    short_term={"plan": "Still trying the same room."},
                ),
            ],
            trade_chat_open_responses=[
                TradeChatOpenResponse(
                    open_chat=True,
                    message="Need wood.",
                    requested_resources={"WOOD": 1},
                ),
                TradeChatOpenResponse(
                    open_chat=True,
                    message="Need wood.",
                    requested_resources={"WOOD": 1},
                ),
                TradeChatOpenResponse(
                    open_chat=True,
                    message="Need wood.",
                    requested_resources={"WOOD": 1},
                ),
            ],
            trade_chat_owner_decision_responses=[
                TradeChatOwnerDecisionResponse(decision="close", message="No deal."),
            ],
        )
        blue = ScriptedPlayer()

        orchestrator = GameOrchestrator(
            MockTradeChatInsufficientOwnerResourcesEngine(),
            {"RED": red, "BLUE": blue},
            trading_chat_enabled=True,
        )

        result = orchestrator.run()

        self.assertEqual(result.winner_ids, ("RED",))
        self.assertEqual(
            [event.kind for event in orchestrator.event_log.public_events],
            [
                "dice_rolled",
                "trade_chat_opened",
                "trade_chat_message",
                "trade_chat_no_deal",
                "trade_chat_closed",
                "turn_ended",
            ],
        )
        self.assertEqual(len(red.action_observations), 3)
        self.assertIn(
            "Cannot reopen the same public trade room",
            red.action_observations[-1].decision_prompt or "",
        )

    def test_orchestrator_snapshots_only_after_final_event_in_transition(self) -> None:
        red = ScriptedPlayer(
            start_turn_responses=[
                TurnStartResponse(short_term={"plan": "Build then end."})
            ],
            action_responses=[
                ActionDecision(action=Action("END_TURN"), short_term={"plan": "Done."})
            ],
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

    def test_counter_offer_records_public_event_and_rejects_underlying_trade(
        self,
    ) -> None:
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
            self.assertEqual(
                orchestrator.action_trace_store.entries[0].action.action_type,
                "COUNTER_OFFER",
            )

    def test_same_trade_market_gets_retry_feedback_and_continues_after_rejection(
        self,
    ) -> None:
        red = ScriptedPlayer(
            action_responses=[
                Action(
                    "OFFER_TRADE", payload={"offer": {"ORE": 1}, "request": {"WOOD": 1}}
                ),
                Action(
                    "OFFER_TRADE", payload={"offer": {"ORE": 1}, "request": {"WOOD": 1}}
                ),
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
        self.assertIn(
            "Previous action was invalid", red.action_observations[-1].decision_prompt
        )
        self.assertIn(
            "Cannot repeat the same domestic trade market",
            red.action_observations[-1].decision_prompt,
        )
        self.assertEqual(
            [event.kind for event in orchestrator.event_log.public_events],
            ["dice_rolled", "trade_offered", "trade_rejected", "turn_ended"],
        )

    def test_engine_resolve_action_value_error_retries_instead_of_crashing(
        self,
    ) -> None:
        orange = ScriptedPlayer(
            action_responses=[
                Action(
                    "OFFER_TRADE",
                    payload={"offer": {"ORE": 1}, "request": {"ORE": 1, "SHEEP": 1}},
                ),
                Action("END_TURN"),
            ]
        )

        orchestrator = GameOrchestrator(
            MockResolveActionValueErrorEngine(),
            {"ORANGE": orange},
        )

        result = orchestrator.run()

        self.assertEqual(result.winner_ids, ("ORANGE",))
        self.assertEqual(len(orange.action_observations), 2)
        self.assertIsNotNone(orange.action_observations[-1].decision_prompt)
        assert orange.action_observations[-1].decision_prompt is not None
        self.assertIn(
            "Previous action was invalid",
            orange.action_observations[-1].decision_prompt,
        )
        self.assertIn(
            "is not currently valid in catanatron",
            orange.action_observations[-1].decision_prompt,
        )
        self.assertEqual(
            [event.kind for event in orchestrator.event_log.public_events],
            ["turn_ended"],
        )

    def test_same_trade_market_can_repeat_after_counteroffer(self) -> None:
        red = ScriptedPlayer(
            action_responses=[
                Action(
                    "OFFER_TRADE", payload={"offer": {"ORE": 1}, "request": {"WOOD": 1}}
                ),
                Action(
                    "OFFER_TRADE", payload={"offer": {"ORE": 1}, "request": {"WOOD": 1}}
                ),
            ]
        )
        blue = ScriptedPlayer(
            reactive_responses=[
                Action(
                    "COUNTER_OFFER",
                    payload={"offer": {"WOOD": 1}, "request": {"ORE": 1}},
                )
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

        self.assertTrue(
            any(event.kind == "trade_offered" for event in transition.public_events)
        )
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
        self.assertEqual(
            orchestrator.action_trace_store.entries[0].action.action_type,
            "REJECT_TRADE",
        )


if __name__ == "__main__":
    unittest.main()
