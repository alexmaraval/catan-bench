from __future__ import annotations

import json
import unittest

from catan_bench.players import LLMPlayer
from catan_bench.prompting import PromptRenderer
from catan_bench.prompts import CATAN_RULES_SUMMARY
from catan_bench.schemas import (
    Action,
    ActionObservation,
    OpeningStrategyObservation,
    PlayerMemory,
    ReactiveObservation,
    TradeChatObservation,
    TradeChatProposal,
    TurnEndObservation,
    TurnStartObservation,
)


class FakeLLMClient:
    def __init__(self, *payloads: dict[str, object]) -> None:
        self._payloads = list(payloads)

    def complete(
        self, *, model, messages, temperature, top_p=None, reasoning_enabled=None
    ):
        payload = self._payloads.pop(0)
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}


class FailingLLMClient:
    def __init__(self, error: RuntimeError) -> None:
        self._error = error

    def complete(
        self, *, model, messages, temperature, top_p=None, reasoning_enabled=None
    ):
        raise self._error


class RawCompletionClient:
    def __init__(self, *contents: str) -> None:
        self._contents = list(contents)

    def complete(
        self, *, model, messages, temperature, top_p=None, reasoning_enabled=None
    ):
        content = self._contents.pop(0)
        return {"choices": [{"message": {"content": content}}]}


class StructuredCompletionClient:
    def __init__(self, *completions: dict[str, object]) -> None:
        self._completions = list(completions)

    def complete(
        self, *, model, messages, temperature, top_p=None, reasoning_enabled=None
    ):
        return self._completions.pop(0)


class CapturingRenderer:
    def __init__(self) -> None:
        self.last_payload: dict[str, object] | None = None

    def render_messages(self, *, system_template, user_template, payload, **context):
        self.last_payload = dict(payload)
        return [
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_template},
        ]


class LLMPlayerTests(unittest.TestCase):
    def test_llm_player_plan_opening_strategy_records_trace(self) -> None:
        player = LLMPlayer(
            client=FakeLLMClient(
                {
                    "long_term": "Prioritize expansion from the brick-wheat side, then look for a city line."
                }
            ),
            model="fake-model",
        )

        response = player.plan_opening_strategy(
            OpeningStrategyObservation(
                game_id="game-1",
                player_id="RED",
                history_index=0,
                turn_index=6,
                phase="opening_strategy",
                decision_index=17,
                public_state={"turn": {"turn_player_id": "RED"}},
                private_state={"resources": {"WOOD": 1, "BRICK": 1, "WHEAT": 1}},
                public_history=(),
                game_rules="Rules",
                memory=PlayerMemory(),
            )
        )

        self.assertEqual(
            response.long_term,
            "Prioritize expansion from the brick-wheat side, then look for a city line.",
        )
        trace = player.take_last_prompt_trace()
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(trace.stage, "opening_strategy")
        self.assertEqual(trace.history_index, 0)

    def test_llm_player_turn_lifecycle_returns_memory_and_traces(self) -> None:
        player = LLMPlayer(
            client=FakeLLMClient(
                {"short_term": {"plan": "Trade first."}},
                {
                    "action_index": 0,
                    "short_term": {"plan": "End now."},
                    "private_reasoning": "Done.",
                },
                {"long_term": {"focus": "Watch BLUE's brick demand."}},
            ),
            model="fake-model",
        )

        start_response = player.start_turn(
            TurnStartObservation(
                game_id="game-1",
                player_id="RED",
                history_index=2,
                turn_index=3,
                phase="play_turn",
                decision_index=5,
                public_state={"turn": {"turn_player_id": "RED"}},
                private_state={"resources": {"WOOD": 1}},
                public_history_since_last_turn=(),
                game_rules="Rules",
                memory=PlayerMemory(long_term={"belief": "BLUE wants brick"}),
            )
        )
        action_response = player.choose_action(
            ActionObservation(
                game_id="game-1",
                player_id="RED",
                history_index=2,
                turn_index=3,
                phase="play_turn",
                decision_index=5,
                public_state={"turn": {"turn_player_id": "RED"}},
                private_state={"resources": {"WOOD": 1}},
                public_history=(),
                turn_public_events=(),
                legal_actions=(Action("END_TURN"),),
                decision_prompt="Choose an action.",
                game_rules="Rules",
                memory=PlayerMemory(
                    long_term={"belief": "BLUE wants brick"},
                    short_term={"plan": "Trade first."},
                ),
            )
        )
        end_response = player.end_turn(
            TurnEndObservation(
                game_id="game-1",
                player_id="RED",
                history_index=3,
                turn_index=3,
                phase="play_turn",
                decision_index=6,
                public_state={"turn": {"turn_player_id": "RED"}},
                private_state={"resources": {"WOOD": 1}},
                turn_public_events=(),
                game_rules="Rules",
                memory=PlayerMemory(
                    long_term={"belief": "BLUE wants brick"},
                    short_term={"plan": "End now."},
                ),
            )
        )

        self.assertEqual(start_response.short_term, {"plan": "Trade first."})
        self.assertEqual(action_response.action.action_type, "END_TURN")
        self.assertEqual(action_response.short_term, {"plan": "End now."})
        self.assertEqual(
            end_response.long_term, {"focus": "Watch BLUE's brick demand."}
        )

        traces = player.take_prompt_traces()
        self.assertEqual(
            [trace.stage for trace in traces],
            ["turn_start", "choose_action", "turn_end"],
        )
        self.assertEqual([trace.history_index for trace in traces], [2, 2, 3])

    def test_llm_player_repairs_illegal_reactive_action(self) -> None:
        player = LLMPlayer(
            client=FakeLLMClient(
                {"action_index": 9, "private_reasoning": "Bad index."},
                {"action_index": 0, "private_reasoning": "Fixed."},
            ),
            model="fake-model",
        )

        response = player.respond_reactive(
            ReactiveObservation(
                game_id="game-1",
                player_id="BLUE",
                history_index=4,
                turn_index=3,
                phase="decide_trade",
                decision_index=7,
                public_state={"turn": {"turn_player_id": "RED"}},
                private_state={"resources": {"BRICK": 1}},
                public_history=(),
                legal_actions=(Action("REJECT_TRADE"),),
                decision_prompt="Respond to the trade.",
                game_rules="Rules",
                memory=PlayerMemory(long_term={"belief": "RED is fishing for brick"}),
            )
        )

        self.assertEqual(response.action.action_type, "REJECT_TRADE")
        trace = player.take_last_prompt_trace()
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(trace.stage, "reactive_action")
        self.assertEqual(len(trace.attempts), 2)

    def test_llm_player_falls_back_when_reactive_repair_is_still_invalid(self) -> None:
        player = LLMPlayer(
            client=FakeLLMClient(
                {"action_index": 9, "private_reasoning": "Bad index."},
                {"action_index": 3, "private_reasoning": "Still bad."},
            ),
            model="fake-model",
        )

        response = player.respond_reactive(
            ReactiveObservation(
                game_id="game-1",
                player_id="BLUE",
                history_index=4,
                turn_index=3,
                phase="decide_trade",
                decision_index=7,
                public_state={"turn": {"turn_player_id": "RED"}},
                private_state={"resources": {"BRICK": 1}},
                public_history=(),
                legal_actions=(Action("REJECT_TRADE"),),
                decision_prompt="Respond to the trade.",
                game_rules="Rules",
                memory=PlayerMemory(long_term={"belief": "RED is fishing for brick"}),
            )
        )

        self.assertEqual(response.action.action_type, "REJECT_TRADE")
        self.assertEqual(response.reasoning, "Still bad.")
        trace = player.take_last_prompt_trace()
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(trace.stage, "reactive_action")
        self.assertEqual(len(trace.attempts), 2)

    def test_llm_player_reactive_discard_selection_preserves_chosen_payload(
        self,
    ) -> None:
        player = LLMPlayer(
            client=FakeLLMClient(
                {
                    "action_index": 1,
                    "private_reasoning": "Keep ore for the city race.",
                }
            ),
            model="fake-model",
        )

        response = player.respond_reactive(
            ReactiveObservation(
                game_id="game-1",
                player_id="RED",
                history_index=9,
                turn_index=4,
                phase="discard",
                decision_index=12,
                public_state={
                    "turn": {"turn_player_id": "BLUE"},
                    "players": {},
                    "board": {},
                    "bank": {},
                },
                private_state={
                    "resources": {"WOOD": 2, "BRICK": 1, "SHEEP": 1, "ORE": 2},
                    "development_cards": {},
                    "pieces": {"roads": 13, "settlements": 3, "cities": 4},
                    "victory_points": {"visible": 4, "actual": 4},
                    "discard_requirement": {"count": 2, "legal_options": 2},
                },
                public_history=(),
                legal_actions=(
                    Action(
                        "DISCARD",
                        payload={"resources": {"ORE": 2}},
                        description="Discard 2×ORE for the robber event.",
                    ),
                    Action(
                        "DISCARD",
                        payload={"resources": {"BRICK": 1, "SHEEP": 1}},
                        description="Discard 1×BRICK, 1×SHEEP for the robber event.",
                    ),
                ),
                decision_prompt="Choose which 2 resource cards to discard for the robber event.",
                game_rules="Rules",
                memory=PlayerMemory(
                    long_term={"goal": "Keep ore and wheat for a city."}
                ),
            )
        )

        self.assertEqual(response.action.action_type, "DISCARD")
        self.assertEqual(
            response.action.payload,
            {"resources": {"BRICK": 1, "SHEEP": 1}},
        )
        self.assertEqual(response.reasoning, "Keep ore for the city race.")

    def test_llm_player_falls_back_when_choose_action_repair_is_still_invalid(
        self,
    ) -> None:
        player = LLMPlayer(
            client=FakeLLMClient(
                {"action_index": 7, "private_reasoning": "Bad index."},
                {"action_index": 4, "private_reasoning": "Still bad."},
            ),
            model="fake-model",
        )

        response = player.choose_action(
            ActionObservation(
                game_id="game-1",
                player_id="RED",
                history_index=2,
                turn_index=3,
                phase="play_turn",
                decision_index=5,
                public_state={"turn": {"turn_player_id": "RED"}},
                private_state={"resources": {"WOOD": 1}},
                public_history=(),
                turn_public_events=(),
                legal_actions=(Action("END_TURN"),),
                decision_prompt="Choose an action.",
                game_rules="Rules",
                memory=PlayerMemory(
                    long_term={"belief": "BLUE wants brick"},
                    short_term={"plan": "Trade first."},
                ),
            )
        )

        self.assertEqual(response.action.action_type, "END_TURN")
        self.assertIsNone(response.short_term)
        self.assertEqual(response.reasoning, "Still bad.")
        trace = player.take_last_prompt_trace()
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(trace.stage, "choose_action")
        self.assertEqual(len(trace.attempts), 2)

    def test_llm_player_trade_chat_methods_parse_payloads(self) -> None:
        player = LLMPlayer(
            client=FakeLLMClient(
                {
                    "open_chat": True,
                    "message": "Need brick.",
                    "requested_resources": {"BRICK": 1},
                },
                {
                    "message": "I can do that.",
                    "owner_gives": {"WOOD": 1},
                    "owner_gets": {"BRICK": 1},
                },
                {
                    "decision": "select",
                    "selected_proposal_id": "attempt-1-round-1-proposal-1",
                    "message": "Deal.",
                },
            ),
            model="fake-model",
        )
        observation = TradeChatObservation(
            game_id="game-1",
            player_id="RED",
            owner_player_id="RED",
            history_index=5,
            turn_index=3,
            phase="play_turn",
            decision_index=8,
            stage="open",
            attempt_index=1,
            round_index=1,
            public_state={"turn": {"turn_player_id": "RED"}},
            private_state={"resources": {"WOOD": 1}},
            transcript=(),
            requested_resources={"BRICK": 1},
            other_player_ids=("BLUE",),
            proposals=(
                TradeChatProposal(
                    proposal_id="attempt-1-round-1-proposal-1",
                    player_id="BLUE",
                    round_index=1,
                    owner_gives={"WOOD": 1},
                    owner_gets={"BRICK": 1},
                ),
            ),
            game_rules="Rules",
            memory=PlayerMemory(long_term={"goal": "trade"}),
            message_char_limit=120,
        )

        open_response = player.open_trade_chat(observation)
        reply_response = player.respond_trade_chat(observation)
        select_response = player.decide_trade_chat(observation)

        self.assertTrue(open_response.open_chat)
        self.assertEqual(open_response.requested_resources, {"BRICK": 1})
        self.assertEqual(reply_response.owner_gives, {"WOOD": 1})
        self.assertEqual(reply_response.owner_gets, {"BRICK": 1})
        self.assertEqual(select_response.decision, "select")
        self.assertEqual(
            select_response.selected_proposal_id, "attempt-1-round-1-proposal-1"
        )

    def test_llm_player_trade_chat_select_recovers_invalid_proposal_hint(self) -> None:
        player = LLMPlayer(
            client=FakeLLMClient(
                {
                    "decision": "select",
                    "selected_proposal_id": "BLUE_WOOD_FOR_BRICK",
                    "message": "Deal.",
                }
            ),
            model="fake-model",
        )
        observation = TradeChatObservation(
            game_id="game-1",
            player_id="RED",
            owner_player_id="RED",
            history_index=5,
            turn_index=3,
            phase="play_turn",
            decision_index=8,
            stage="owner_decision",
            attempt_index=1,
            round_index=1,
            public_state={"turn": {"turn_player_id": "RED"}},
            private_state={"resources": {"WOOD": 1}},
            transcript=(),
            requested_resources={"BRICK": 1},
            other_player_ids=("BLUE",),
            proposals=(
                TradeChatProposal(
                    proposal_id="attempt-1-round-1-proposal-1",
                    player_id="BLUE",
                    round_index=1,
                    owner_gives={"WOOD": 1},
                    owner_gets={"BRICK": 1},
                ),
            ),
            game_rules="Rules",
            memory=PlayerMemory(long_term={"goal": "trade"}),
            message_char_limit=120,
        )

        select_response = player.decide_trade_chat(observation)

        self.assertEqual(select_response.decision, "select")
        self.assertEqual(
            select_response.selected_proposal_id, "attempt-1-round-1-proposal-1"
        )

    def test_trade_chat_reply_falls_back_to_responder_relative_offer_request_when_needed(
        self,
    ) -> None:
        player = LLMPlayer(
            client=FakeLLMClient(
                {
                    "message": "I can offer 1 wheat for 1 brick.",
                    "offer": {"WHEAT": 1},
                    "request": {"BRICK": 1},
                }
            ),
            model="fake-model",
        )
        observation = TradeChatObservation(
            game_id="game-1",
            player_id="BLUE",
            owner_player_id="RED",
            history_index=5,
            turn_index=3,
            phase="play_turn",
            decision_index=8,
            stage="reply",
            attempt_index=1,
            round_index=1,
            public_state={"turn": {"turn_player_id": "RED"}},
            private_state={"resources": {"WHEAT": 1}},
            transcript=(),
            requested_resources={"BRICK": 1},
            other_player_ids=("RED", "ORANGE"),
            proposals=(),
            game_rules="Rules",
            memory=PlayerMemory(long_term={"goal": "trade"}),
            message_char_limit=120,
        )

        reply_response = player.respond_trade_chat(observation)

        self.assertEqual(reply_response.owner_gives, {"BRICK": 1})
        self.assertEqual(reply_response.owner_gets, {"WHEAT": 1})

    def test_trade_chat_open_prompt_includes_requested_resources_context(self) -> None:
        renderer = CapturingRenderer()
        player = LLMPlayer(
            client=FakeLLMClient({"open_chat": False}),
            model="fake-model",
            renderer=renderer,
        )
        observation = TradeChatObservation(
            game_id="game-1",
            player_id="RED",
            owner_player_id="RED",
            history_index=5,
            turn_index=3,
            phase="play_turn",
            decision_index=8,
            stage="open",
            attempt_index=1,
            round_index=0,
            public_state={"turn": {"turn_player_id": "RED"}},
            private_state={"resources": {"WOOD": 1}},
            transcript=(),
            requested_resources={"BRICK": 1},
            other_player_ids=("BLUE",),
            proposals=(),
            game_rules="Rules",
            memory=PlayerMemory(long_term={"goal": "trade"}),
            message_char_limit=120,
        )

        messages = player._messages_for_trade_chat_open(observation)

        self.assertEqual(messages[1]["role"], "user")
        self.assertIsNotNone(renderer.last_payload)
        assert renderer.last_payload is not None
        self.assertEqual(renderer.last_payload["requested_resources"], {"BRICK": 1})

    def test_choose_action_prompt_renders_payloads_and_trade_template_constraints(
        self,
    ) -> None:
        player = LLMPlayer(
            client=FakeLLMClient({"action_index": 1, "short_term": None}),
            model="fake-model",
        )
        observation = ActionObservation(
            game_id="game-1",
            player_id="RED",
            history_index=2,
            turn_index=3,
            phase="play_turn",
            decision_index=5,
            public_state={
                "turn": {"turn_player_id": "RED"},
                "board": {},
                "players": {},
                "bank": {},
            },
            private_state={
                "resources": {"WOOD": 1},
                "development_cards": {},
                "pieces": {"roads": 15, "settlements": 5, "cities": 4},
                "victory_points": {"visible": 2, "actual": 2},
            },
            public_history=(),
            turn_public_events=(),
            legal_actions=(
                Action(
                    "OFFER_TRADE",
                    payload={"offer": {}, "request": {}},
                    description="Offer a domestic trade.",
                ),
                Action(
                    "MOVE_ROBBER",
                    payload={"coordinate": [1, -1, 0], "victim": "BLUE"},
                    description="Move the robber to [1, -1, 0] and steal from BLUE.",
                ),
            ),
            decision_prompt="Choose an action.",
            game_rules="Rules",
            memory=PlayerMemory(short_term={"plan": "Pressure BLUE."}),
        )

        messages = player._messages_for_action(observation)
        user_prompt = messages[1]["content"]

        self.assertIn("[0] OFFER_TRADE", user_prompt)
        self.assertIn('payload: `{"offer": {}, "request": {}}`', user_prompt)
        self.assertIn(
            "requires full `action` object; do not use `action_index` alone",
            user_prompt,
        )
        self.assertIn("[1] MOVE_ROBBER", user_prompt)
        self.assertIn(
            'payload: `{"coordinate": [1, -1, 0], "victim": "BLUE"}`', user_prompt
        )

    def test_choose_action_system_prompt_warns_against_circular_trades(self) -> None:
        player = LLMPlayer(
            client=FakeLLMClient({"action_index": 0, "short_term": None}),
            model="fake-model",
        )
        observation = ActionObservation(
            game_id="game-1",
            player_id="RED",
            history_index=2,
            turn_index=3,
            phase="play_turn",
            decision_index=5,
            public_state={
                "turn": {"turn_player_id": "RED"},
                "board": {},
                "players": {},
                "bank": {},
            },
            private_state={
                "resources": {"WOOD": 1, "SHEEP": 1},
                "development_cards": {},
                "pieces": {"roads": 15, "settlements": 5, "cities": 4},
                "victory_points": {"visible": 2, "actual": 2},
            },
            public_history=(),
            turn_public_events=(),
            legal_actions=(
                Action(
                    "OFFER_TRADE",
                    payload={"offer": {"WOOD": 1}, "request": {"SHEEP": 1}},
                    description="Offer a domestic trade.",
                ),
                Action("END_TURN"),
            ),
            decision_prompt="Choose an action.",
            game_rules=CATAN_RULES_SUMMARY,
            memory=PlayerMemory(short_term={"plan": "Trade first."}),
        )

        messages = player._messages_for_action(observation)
        system_prompt = messages[0]["content"]

        self.assertIn("Avoid circular same-turn trades", system_prompt)
        self.assertIn("unless you intentionally want that reversal", system_prompt)

    def test_reactive_discard_prompt_surfaces_requirement_and_strategy_hint(
        self,
    ) -> None:
        player = LLMPlayer(
            client=FakeLLMClient({"action_index": 0}),
            model="fake-model",
        )
        observation = ReactiveObservation(
            game_id="game-1",
            player_id="RED",
            history_index=9,
            turn_index=4,
            phase="discard",
            decision_index=12,
            public_state={
                "turn": {"turn_player_id": "BLUE"},
                "players": {},
                "board": {},
                "bank": {},
            },
            private_state={
                "resources": {"WOOD": 2, "BRICK": 1, "SHEEP": 1, "ORE": 2},
                "development_cards": {},
                "pieces": {"roads": 13, "settlements": 3, "cities": 4},
                "victory_points": {"visible": 4, "actual": 4},
                "discard_requirement": {"count": 2, "legal_options": 2},
            },
            public_history=(),
            legal_actions=(
                Action(
                    "DISCARD",
                    payload={"resources": {"ORE": 2}},
                    description="Discard 2×ORE for the robber event.",
                ),
                Action(
                    "DISCARD",
                    payload={"resources": {"BRICK": 1, "SHEEP": 1}},
                    description="Discard 1×BRICK, 1×SHEEP for the robber event.",
                ),
            ),
            decision_prompt="Choose which 2 resource cards to discard for the robber event.",
            game_rules="Rules",
            memory=PlayerMemory(long_term={"goal": "Keep ore and wheat for a city."}),
        )

        messages = player._messages_for_reactive(observation)
        user_prompt = messages[1]["content"]

        self.assertIn("If this is a discard decision, keep the hand", user_prompt)
        self.assertIn("Must discard: 2 cards", user_prompt)
        self.assertIn("Legal discard options: 2", user_prompt)
        self.assertIn("[0] DISCARD", user_prompt)
        self.assertIn('payload: `{"resources": {"ORE": 2}}`', user_prompt)

    def test_llm_player_accepts_common_trade_action_shape_without_payload_wrapper(
        self,
    ) -> None:
        player = LLMPlayer(
            client=FakeLLMClient(
                {
                    "action_index": 2,
                    "short_term": "Offer sheep for brick to enable a road.",
                    "action": {
                        "action": "OFFER_TRADE",
                        "offer": {"SHEEP": 1},
                        "request": {"BRICK": 1},
                    },
                }
            ),
            model="fake-model",
        )

        response = player.choose_action(
            ActionObservation(
                game_id="game-1",
                player_id="BLUE",
                history_index=2,
                turn_index=7,
                phase="play_turn",
                decision_index=20,
                public_state={"turn": {"turn_player_id": "BLUE"}},
                private_state={
                    "resources": {"WOOD": 1, "SHEEP": 1, "WHEAT": 1, "ORE": 2}
                },
                public_history=(),
                turn_public_events=(),
                legal_actions=(
                    Action("END_TURN"),
                    Action("BUY_DEVELOPMENT_CARD"),
                    Action(
                        "OFFER_TRADE",
                        payload={"offer": {}, "request": {}},
                        description="Offer a domestic trade.",
                    ),
                ),
                decision_prompt="Choose an action.",
                game_rules="Rules",
                memory=PlayerMemory(short_term="Trade for brick if possible."),
            )
        )

        self.assertEqual(response.action.action_type, "OFFER_TRADE")
        self.assertEqual(
            response.action.payload, {"offer": {"SHEEP": 1}, "request": {"BRICK": 1}}
        )
        trace = player.take_last_prompt_trace()
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(len(trace.attempts), 1)

    def test_llm_player_accepts_counter_offer_payload_action(self) -> None:
        player = LLMPlayer(
            client=FakeLLMClient(
                {
                    "action_index": 1,
                    "action": {
                        "action_type": "COUNTER_OFFER",
                        "payload": {"offer": {"BRICK": 1}, "request": {"WOOD": 1}},
                    },
                    "private_reasoning": "Counter instead of rejecting outright.",
                }
            ),
            model="fake-model",
        )

        response = player.respond_reactive(
            ReactiveObservation(
                game_id="game-1",
                player_id="BLUE",
                history_index=4,
                turn_index=3,
                phase="decide_trade",
                decision_index=7,
                public_state={"turn": {"turn_player_id": "WHITE"}},
                private_state={"resources": {"BRICK": 1}},
                public_history=(),
                legal_actions=(
                    Action("REJECT_TRADE"),
                    Action(
                        "COUNTER_OFFER",
                        payload={"offer": {}, "request": {}},
                        description="Counter the current trade.",
                    ),
                ),
                decision_prompt="Respond to the trade.",
                game_rules="Rules",
                memory=PlayerMemory(
                    long_term={"belief": "WHITE may still want a deal"}
                ),
            )
        )

        self.assertEqual(response.action.action_type, "COUNTER_OFFER")
        self.assertEqual(
            response.action.payload,
            {"offer": {"BRICK": 1}, "request": {"WOOD": 1}},
        )

    def test_game_context_renders_structured_memory_as_json_text(self) -> None:
        renderer = PromptRenderer()
        rendered = renderer.render(
            "partials/game_context.jinja",
            payload={
                "turn_index": 6,
                "player_id": "WHITE",
                "private_state": {
                    "resources": {"WOOD": 2},
                    "development_cards": {},
                    "pieces": {"roads": 13, "settlements": 3, "cities": 4},
                    "victory_points": {"visible": 2, "actual": 2},
                },
                "public_state": {"players": {}, "board": {}, "bank": {}},
                "memory": {
                    "short_term": {"action": "build road", "target_edge": "[17, 18]"},
                    "long_term": {"goal": "contest longest road"},
                },
            },
        )

        self.assertIn(
            'Current turn scratchpad: {"action": "build road", "target_edge": "[17, 18]"}',
            rendered,
        )
        self.assertIn('{"goal": "contest longest road"}', rendered)

    def test_turn_start_contract_asks_for_plain_text_memory(self) -> None:
        renderer = PromptRenderer()
        rendered = renderer.render("partials/turn_start_contract.jinja")
        self.assertIn("scratchpad entry", rendered)
        self.assertIn("not an action log", rendered)

    def test_opening_strategy_contract_asks_for_plain_text_memory(self) -> None:
        renderer = PromptRenderer()
        rendered = renderer.render("partials/opening_strategy_contract.jinja")
        self.assertIn("brief plain-text opening strategy note", rendered)

    def test_turn_end_contract_asks_for_plain_text_memory(self) -> None:
        renderer = PromptRenderer()
        rendered = renderer.render("partials/turn_end_contract.jinja")
        self.assertIn("brief plain-text memory note", rendered)

    def test_action_contract_includes_payload_example(self) -> None:
        renderer = PromptRenderer()
        rendered = renderer.render("partials/action_contract.jinja")
        self.assertIn('"action_type": "OFFER_TRADE"', rendered)
        self.assertIn(
            '"payload": {"offer": {"SHEEP": 1}, "request": {"BRICK": 1}}', rendered
        )
        self.assertIn("not as a description of the chosen action", rendered)

    def test_start_turn_raises_runtime_error_and_records_failed_attempt(self) -> None:
        player = LLMPlayer(
            client=FailingLLMClient(RuntimeError("local model request failed")),
            model="fake-model",
        )

        with self.assertRaisesRegex(RuntimeError, "local model request failed"):
            player.start_turn(
                TurnStartObservation(
                    game_id="game-1",
                    player_id="RED",
                    history_index=2,
                    turn_index=3,
                    phase="play_turn",
                    decision_index=5,
                    public_state={"turn": {"turn_player_id": "RED"}},
                    private_state={"resources": {"WOOD": 1}},
                    public_history_since_last_turn=(),
                    game_rules="Rules",
                    memory=PlayerMemory(long_term={"belief": "BLUE wants brick"}),
                )
            )

        trace = player.take_last_prompt_trace()
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(trace.stage, "turn_start")
        self.assertEqual(len(trace.attempts), 1)
        self.assertEqual(
            trace.attempts[0].response,
            {
                "error": {
                    "type": "llm_request_failed",
                    "message": "local model request failed",
                }
            },
        )

    def test_start_turn_retries_once_after_invalid_json_response(self) -> None:
        player = LLMPlayer(
            client=RawCompletionClient(
                '{"short_term":"unterminated', '{"short_term":"Trade first."}'
            ),
            model="fake-model",
        )

        response = player.start_turn(
            TurnStartObservation(
                game_id="game-1",
                player_id="RED",
                history_index=2,
                turn_index=3,
                phase="play_turn",
                decision_index=5,
                public_state={"turn": {"turn_player_id": "RED"}},
                private_state={"resources": {"WOOD": 1}},
                public_history_since_last_turn=(),
                game_rules="Rules",
                memory=PlayerMemory(long_term={"belief": "BLUE wants brick"}),
            )
        )

        self.assertEqual(response.short_term, "Trade first.")
        trace = player.take_last_prompt_trace()
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(len(trace.attempts), 2)
        self.assertEqual(
            trace.attempts[0].response["error"]["type"],  # type: ignore[index]
            "invalid_response",
        )

    def test_start_turn_retries_once_after_empty_response(self) -> None:
        player = LLMPlayer(
            client=RawCompletionClient("", '{"short_term":"Build safely."}'),
            model="fake-model",
        )

        response = player.start_turn(
            TurnStartObservation(
                game_id="game-1",
                player_id="RED",
                history_index=2,
                turn_index=3,
                phase="play_turn",
                decision_index=5,
                public_state={"turn": {"turn_player_id": "RED"}},
                private_state={"resources": {"WOOD": 1}},
                public_history_since_last_turn=(),
                game_rules="Rules",
                memory=PlayerMemory(long_term={"belief": "BLUE wants brick"}),
            )
        )

        self.assertEqual(response.short_term, "Build safely.")
        trace = player.take_last_prompt_trace()
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(len(trace.attempts), 2)
        self.assertEqual(
            trace.attempts[0].response["error"]["type"],  # type: ignore[index]
            "invalid_response",
        )

    def test_reactive_stage_salvages_reasoning_only_output_via_repair(self) -> None:
        player = LLMPlayer(
            client=StructuredCompletionClient(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "reasoning": "Thinking through setup.",
                            },
                            "finish_reason": "length",
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {"content": "", "reasoning": "Still thinking."},
                            "finish_reason": "length",
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"action_index": 0, "private_reasoning": "Recovered."}'
                            },
                            "finish_reason": "stop",
                        }
                    ]
                },
            ),
            model="fake-model",
        )

        response = player.respond_reactive(
            ReactiveObservation(
                game_id="game-1",
                player_id="BLUE",
                history_index=4,
                turn_index=3,
                phase="build_initial_settlement",
                decision_index=7,
                public_state={"turn": {"turn_player_id": "BLUE"}},
                private_state={"resources": {}},
                public_history=(),
                legal_actions=(
                    Action("BUILD_INITIAL_SETTLEMENT", payload={"node_id": 4}),
                ),
                decision_prompt="Place your opening settlement.",
                game_rules="Rules",
                memory=PlayerMemory(),
            )
        )

        self.assertEqual(response.action.action_type, "BUILD_INITIAL_SETTLEMENT")
        self.assertEqual(response.reasoning, "Recovered.")
        trace = player.take_last_prompt_trace()
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(trace.stage, "reactive_action")
        self.assertEqual(len(trace.attempts), 3)


if __name__ == "__main__":
    unittest.main()
