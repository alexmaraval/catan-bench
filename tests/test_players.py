from __future__ import annotations

import json
import unittest

from catan_bench.players import LLMPlayer, _normalize_offer_trade_payload
from catan_bench.prompting import PromptRenderer
from catan_bench.prompts import CATAN_RULES_SUMMARY
from catan_bench.schemas import (
    Action,
    ActionObservation,
    Event,
    OpeningStrategyObservation,
    PlayerMemory,
    PostGameChatObservation,
    PublicChatDraft,
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
        self,
        *,
        model,
        messages,
        temperature,
        top_p=None,
        reasoning_enabled=None,
        reasoning_effort=None,
    ):
        payload = self._payloads.pop(0)
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}


class FailingLLMClient:
    def __init__(self, error: RuntimeError) -> None:
        self._error = error

    def complete(
        self,
        *,
        model,
        messages,
        temperature,
        top_p=None,
        reasoning_enabled=None,
        reasoning_effort=None,
    ):
        raise self._error


class RawCompletionClient:
    def __init__(self, *contents: str) -> None:
        self._contents = list(contents)

    def complete(
        self,
        *,
        model,
        messages,
        temperature,
        top_p=None,
        reasoning_enabled=None,
        reasoning_effort=None,
    ):
        content = self._contents.pop(0)
        return {"choices": [{"message": {"content": content}}]}


class StructuredCompletionClient:
    def __init__(self, *completions: dict[str, object]) -> None:
        self._completions = list(completions)

    def complete(
        self,
        *,
        model,
        messages,
        temperature,
        top_p=None,
        reasoning_enabled=None,
        reasoning_effort=None,
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

    def test_llm_player_post_game_chat_records_trace_and_payload(self) -> None:
        renderer = CapturingRenderer()
        player = LLMPlayer(
            client=FakeLLMClient(
                {
                    "public_chat": {
                        "message": "Good game, BLUE. That city timing was sharp.",
                        "target_player_id": "blue",
                    }
                }
            ),
            model="fake-model",
            renderer=renderer,
        )

        response = player.post_game_chat(
            PostGameChatObservation(
                game_id="game-1",
                player_id="RED",
                history_index=7,
                turn_index=4,
                phase="post_game_chat",
                decision_index=1,
                public_state={"turn": {"turn_player_id": "RED"}},
                private_state={"resources": {"WOOD": 1}},
                public_history=(
                    Event(
                        kind="turn_ended",
                        payload={},
                        history_index=7,
                        turn_index=3,
                        phase="play_turn",
                        decision_index=6,
                        actor_player_id="RED",
                    ),
                ),
                public_chat_enabled=True,
                public_chat_transcript=(
                    Event(
                        kind="public_chat_message",
                        payload={
                            "speaker_player_id": "BLUE",
                            "message": "Well played.",
                        },
                        history_index=6,
                        turn_index=3,
                        phase="turn_end",
                        actor_player_id="BLUE",
                    ),
                ),
                public_chat_message_char_limit=500,
                result={"winner_ids": ["RED"], "num_turns": 3},
                game_rules="Rules",
                memory=PlayerMemory(long_term={"goal": "Win the race cleanly"}),
            )
        )

        self.assertEqual(
            response.public_chat,
            PublicChatDraft(
                message="Good game, BLUE. That city timing was sharp.",
                target_player_id="BLUE",
            ),
        )
        assert renderer.last_payload is not None
        self.assertEqual(renderer.last_payload["result"]["winner_ids"], ["RED"])
        self.assertEqual(renderer.last_payload["public_history"][0]["kind"], "turn_ended")
        self.assertEqual(
            renderer.last_payload["public_chat_transcript"][0]["kind"],
            "public_chat_message",
        )
        trace = player.take_last_prompt_trace()
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(trace.stage, "post_game_chat")
        self.assertEqual(trace.phase, "post_game_chat")

    def test_llm_player_parses_optional_public_chat(self) -> None:
        player = LLMPlayer(
            client=FakeLLMClient(
                {
                    "short_term": {"plan": "Apply pressure."},
                    "public_chat": {
                        "message": "BLUE, don't feed RED the road.",
                        "target_player_id": "blue",
                    },
                },
                {
                    "action_index": 0,
                    "short_term": {"plan": "Close the turn."},
                    "public_chat": {"message": "I am ending here."},
                },
                {
                    "long_term": {"focus": "Keep WHITE boxed in."},
                    "public_chat": {
                        "message": "WHITE is the real threat.",
                        "target_player_id": "WHITE",
                    },
                },
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
                public_chat_enabled=True,
                public_chat_transcript=(),
                public_chat_message_char_limit=500,
                game_rules="Rules",
                memory=PlayerMemory(),
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
                public_chat_enabled=True,
                public_chat_transcript=(),
                public_chat_message_char_limit=500,
                legal_actions=(Action("END_TURN"),),
                decision_prompt="Choose an action.",
                game_rules="Rules",
                memory=PlayerMemory(short_term={"plan": "Apply pressure."}),
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
                public_chat_enabled=True,
                public_chat_transcript=(),
                public_chat_message_char_limit=500,
                game_rules="Rules",
                memory=PlayerMemory(short_term={"plan": "Close the turn."}),
            )
        )

        self.assertEqual(
            start_response.public_chat,
            PublicChatDraft(
                message="BLUE, don't feed RED the road.", target_player_id="BLUE"
            ),
        )
        self.assertEqual(
            action_response.public_chat,
            PublicChatDraft(message="I am ending here."),
        )
        self.assertEqual(
            end_response.public_chat,
            PublicChatDraft(
                message="WHITE is the real threat.", target_player_id="WHITE"
            ),
        )

    def test_turn_start_prompt_payload_includes_public_chat_transcript(self) -> None:
        renderer = CapturingRenderer()
        player = LLMPlayer(
            client=FakeLLMClient({"short_term": None}),
            model="fake-model",
            renderer=renderer,
        )

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
                public_chat_enabled=True,
                public_chat_transcript=(
                    Event(
                        kind="public_chat_message",
                        payload={
                            "speaker_player_id": "BLUE",
                            "message": "RED is ahead on road pace.",
                            "target_player_id": "WHITE",
                        },
                        history_index=2,
                        turn_index=2,
                        phase="play_turn",
                        actor_player_id="BLUE",
                    ),
                ),
                public_chat_message_char_limit=500,
                game_rules="Rules",
                memory=PlayerMemory(),
            )
        )

        assert renderer.last_payload is not None
        self.assertTrue(renderer.last_payload["public_chat_enabled"])
        self.assertEqual(renderer.last_payload["public_chat_message_char_limit"], 500)
        transcript = renderer.last_payload["public_chat_transcript"]
        self.assertEqual(len(transcript), 1)
        self.assertEqual(transcript[0]["kind"], "public_chat_message")

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

    def test_trade_chat_reply_accepts_responder_relative_keys(self) -> None:
        player = LLMPlayer(
            client=FakeLLMClient(
                {
                    "message": "I'll trade 1 brick for 1 wood.",
                    "you_give": {"BRICK": 1},
                    "you_get": {"WOOD": 1},
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
            private_state={"resources": {"BRICK": 1}},
            transcript=(),
            requested_resources={"BRICK": 1},
            other_player_ids=("RED", "ORANGE"),
            proposals=(),
            game_rules="Rules",
            memory=PlayerMemory(long_term={"goal": "trade"}),
            message_char_limit=120,
        )

        reply_response = player.respond_trade_chat(observation)

        self.assertEqual(reply_response.owner_gives, {"WOOD": 1})
        self.assertEqual(reply_response.owner_gets, {"BRICK": 1})

    def test_trade_chat_reply_error_identifies_player_model_and_stage(self) -> None:
        player = LLMPlayer(
            client=StructuredCompletionClient(
                {
                    "choices": [
                        {
                            "message": {
                                "content": {"type": "text", "text": "{}"},
                            }
                        }
                    ]
                }
            ),
            model="openrouter/fake-model",
            invalid_response_retries=0,
        )
        observation = TradeChatObservation(
            game_id="game-1",
            player_id="RED",
            owner_player_id="RED",
            history_index=5,
            turn_index=3,
            phase="play_turn",
            decision_index=8,
            stage="reply",
            attempt_index=1,
            round_index=1,
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

        with self.assertRaisesRegex(
            RuntimeError,
            "player 'RED'.*model 'openrouter/fake-model'.*stage 'trade_chat_reply'.*content.*dict",
        ):
            player.respond_trade_chat(observation)

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

    def test_trade_chat_owner_decision_downgrades_select_when_no_proposals(self) -> None:
        player = LLMPlayer(
            client=FakeLLMClient(
                {
                    "decision": "select",
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
            proposals=(),
            game_rules="Rules",
            memory=PlayerMemory(long_term={"goal": "trade"}),
            message_char_limit=120,
        )

        select_response = player.decide_trade_chat(observation)

        self.assertEqual(select_response.decision, "close")
        self.assertIsNone(select_response.selected_proposal_id)

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
        player = LLMPlayer(
            client=FakeLLMClient({"open_chat": False}),
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
            round_index=0,
            public_state={
                "turn": {"turn_player_id": "RED"},
                "players": {},
                "board": {},
                "bank": {},
            },
            private_state={
                "resources": {"WOOD": 1},
                "development_cards": {},
                "pieces": {"roads": 15, "settlements": 5, "cities": 4},
                "victory_points": {"visible": 2, "actual": 2},
            },
            public_history=(
                Event(
                    kind="dice_rolled",
                    payload={"result": [4, 3]},
                    history_index=4,
                    turn_index=3,
                    phase="play_turn",
                    decision_index=0,
                    actor_player_id="RED",
                ),
            ),
            transcript=(),
            public_chat_transcript=(
                Event(
                    kind="public_chat_message",
                    payload={
                        "speaker_player_id": "BLUE",
                        "message": "RED is pushing for road tempo.",
                        "target_player_id": "WHITE",
                    },
                    history_index=4,
                    turn_index=3,
                    phase="play_turn",
                    actor_player_id="BLUE",
                ),
            ),
            requested_resources={"BRICK": 1},
            other_player_ids=("BLUE",),
            proposals=(),
            game_rules="Rules",
            memory=PlayerMemory(long_term={"goal": "trade"}),
            message_char_limit=120,
        )

        messages = player._messages_for_trade_chat_open(observation)
        user_prompt = messages[1]["content"]

        self.assertIn("### Identity", user_prompt)
        self.assertIn("You are player RED.", user_prompt)
        self.assertIn("### Recent Game History", user_prompt)
        self.assertIn("RED rolled [4, 3]", user_prompt)
        self.assertIn("### Main public chat", user_prompt)
        self.assertIn(
            "Quoted table talk from named speakers. Read it as public evidence, not as your voice.",
            user_prompt,
        )
        self.assertIn("BLUE to WHITE (public): RED is pushing for road tempo.", user_prompt)

    def test_trade_chat_reply_prompt_includes_recent_game_history(self) -> None:
        player = LLMPlayer(
            client=FakeLLMClient({"message": "Pass."}),
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
            public_state={
                "turn": {"turn_player_id": "RED"},
                "players": {},
                "board": {},
                "bank": {},
            },
            private_state={
                "resources": {"WHEAT": 1},
                "development_cards": {},
                "pieces": {"roads": 15, "settlements": 5, "cities": 4},
                "victory_points": {"visible": 2, "actual": 2},
            },
            public_history=(
                Event(
                    kind="dice_rolled",
                    payload={"result": [4, 3]},
                    history_index=3,
                    turn_index=3,
                    phase="play_turn",
                    decision_index=0,
                    actor_player_id="RED",
                ),
            ),
            transcript=(
                Event(
                    kind="trade_chat_opened",
                    payload={
                        "owner_player_id": "RED",
                        "requested_resources": {"BRICK": 1},
                        "attempt_index": 1,
                    },
                    history_index=4,
                    turn_index=3,
                    phase="play_turn",
                    decision_index=8,
                    actor_player_id="RED",
                ),
                Event(
                    kind="trade_chat_message",
                    payload={
                        "owner_player_id": "RED",
                        "speaker_player_id": "RED",
                        "message": "Need brick.",
                        "attempt_index": 1,
                        "round_index": 0,
                    },
                    history_index=5,
                    turn_index=3,
                    phase="play_turn",
                    decision_index=8,
                    actor_player_id="RED",
                ),
            ),
            requested_resources={"BRICK": 1},
            proposals=(),
            game_rules="Rules",
            memory=PlayerMemory(long_term={"goal": "Only trade up."}),
            message_char_limit=120,
        )

        messages = player._messages_for_trade_chat_reply(observation)
        user_prompt = messages[1]["content"]

        self.assertIn("### Identity", user_prompt)
        self.assertIn("You are player BLUE.", user_prompt)
        self.assertIn("### Recent Game History", user_prompt)
        self.assertIn("RED rolled [4, 3]", user_prompt)
        self.assertIn("### Trade Chat", user_prompt)
        self.assertIn("RED opened trade chat requesting 1×BRICK", user_prompt)
        self.assertIn("RED: Need brick.", user_prompt)
        self.assertIn(
            "Quoted table talk from named speakers. Read it as public evidence, not as your voice.",
            user_prompt,
        )
        self.assertIn("A concrete proposal usually means `you_give` matches or includes that request.", user_prompt)
        self.assertIn("If your message says \"I'll give you X for Y\", then JSON must be `you_give = X` and `you_get = Y`.", user_prompt)

    def test_trade_chat_owner_decision_prompt_omits_select_when_no_proposals(self) -> None:
        player = LLMPlayer(
            client=FakeLLMClient({"decision": "close"}),
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
            public_state={
                "turn": {"turn_player_id": "RED"},
                "players": {},
                "board": {},
                "bank": {},
            },
            private_state={
                "resources": {"WOOD": 1},
                "development_cards": {},
                "pieces": {"roads": 15, "settlements": 5, "cities": 4},
                "victory_points": {"visible": 2, "actual": 2},
            },
            transcript=(
                Event(
                    kind="trade_chat_opened",
                    payload={
                        "owner_player_id": "RED",
                        "requested_resources": {"BRICK": 1},
                        "attempt_index": 1,
                    },
                    history_index=4,
                    turn_index=3,
                    phase="play_turn",
                    decision_index=8,
                    actor_player_id="RED",
                ),
            ),
            requested_resources={"BRICK": 1},
            proposals=(),
            game_rules="Rules",
            memory=PlayerMemory(long_term={"goal": "trade"}),
            message_char_limit=120,
        )

        messages = player._messages_for_trade_chat_owner_decision(observation)
        user_prompt = messages[1]["content"]

        self.assertIn("There are no valid concrete proposals to select right now.", user_prompt)
        self.assertIn("- `decision`: one of `continue` or `close`.", user_prompt)
        self.assertIn("Do not return `select` because there are no concrete proposals in this room.", user_prompt)

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

        self.assertIn("### Identity", user_prompt)
        self.assertIn("You are player RED.", user_prompt)
        self.assertIn("[0] OFFER_TRADE", user_prompt)
        self.assertIn("### Recent Game History", user_prompt)
        self.assertIn('payload: `{"offer": {}, "request": {}}`', user_prompt)
        self.assertIn(
            "requires full `action` object; do not use `action_index` alone",
            user_prompt,
        )
        self.assertIn("[1] MOVE_ROBBER", user_prompt)
        self.assertIn(
            'payload: `{"coordinate": [1, -1, 0], "victim": "BLUE"}`', user_prompt
        )

    def test_repair_prompt_includes_identity_perspective_block(self) -> None:
        player = LLMPlayer(
            client=FakeLLMClient({"action_index": 0}),
            model="fake-model",
        )
        observation = ActionObservation(
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

        messages = player._repair_messages(observation, {"action_index": 99})
        user_prompt = messages[1]["content"]

        self.assertIn("Stage: Repair action.", user_prompt)
        self.assertIn("### Identity", user_prompt)
        self.assertIn("You are player RED.", user_prompt)
        self.assertIn("Current turn owner: RED. That is you.", user_prompt)
        self.assertIn(
            "Only use information and legal actions that belong to RED.",
            user_prompt,
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

        self.assertIn("You are player RED in this game.", system_prompt)
        self.assertIn(
            "Interpret all first-person and second-person references only from RED's point of view.",
            system_prompt,
        )
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
        self.assertIn("### Recent Game History", user_prompt)
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
            "### Short-term strategy",
            rendered,
        )
        self.assertIn(
            '- {"action": "build road", "target_edge": "[17, 18]"}',
            rendered,
        )
        self.assertIn("### Long-term strategy", rendered)
        self.assertIn('- {"goal": "contest longest road"}', rendered)

    def test_game_context_renders_explicit_hand_visibility_and_vp_goal(self) -> None:
        renderer = PromptRenderer()
        rendered = renderer.render(
            "partials/game_context.jinja",
            payload={
                "turn_index": 9,
                "player_id": "WHITE",
                "private_state": {
                    "resources": {
                        "WOOD": 4,
                        "BRICK": 3,
                        "SHEEP": 2,
                        "WHEAT": 6,
                        "ORE": 2,
                    },
                    "development_cards": {"KNIGHT": 1, "VICTORY_POINT": 1},
                    "pieces": {"roads": 0, "settlements": 0, "cities": 3},
                    "victory_points": {"visible": 9, "actual": 9},
                    "ports": ["ANY", "BRICK", "ORE"],
                },
                "public_state": {
                    "turn": {"turn_player_id": "WHITE", "vps_to_win": 10},
                    "players": {
                        "WHITE": {
                            "vp": 9,
                            "resource_card_count": 17,
                            "development_card_count": 2,
                            "roads_left": 0,
                            "settlements_left": 0,
                            "cities_left": 3,
                            "longest_road_length": 6,
                            "played_knights": 2,
                            "has_longest_road": True,
                            "has_largest_army": False,
                        }
                    },
                    "board": {},
                    "bank": {},
                },
                "memory": {"short_term": None, "long_term": None},
            },
        )

        self.assertIn(
            "Resources (17 cards total; count public, exact types private): 4×WOOD, 3×BRICK, 2×SHEEP, 6×WHEAT, 2×ORE",
            rendered,
        )
        self.assertIn(
            "Development (played knights public; unplayed cards private, including VP cards): 2 played knights; 1×KNIGHT, 1×VICTORY_POINT",
            rendered,
        )
        self.assertIn(
            "Pieces left to play (public): 0 Roads / 0 Settlements / 3 Cities",
            rendered,
        )
        self.assertIn("VP: 9 (9 public, 0 private)", rendered)
        self.assertIn("VP remaining to win: 1 to reach 10 VP", rendered)
        self.assertIn(
            "Ports (public): ANY (3:1), BRICK (2:1), ORE (2:1)",
            rendered,
        )

    def test_game_context_preserves_zero_played_knights_in_hand_summary(self) -> None:
        renderer = PromptRenderer()
        rendered = renderer.render(
            "partials/game_context.jinja",
            payload={
                "turn_index": 4,
                "player_id": "RED",
                "private_state": {
                    "resources": {"WOOD": 1},
                    "development_cards": {},
                    "pieces": {"roads": 13, "settlements": 3, "cities": 4},
                    "victory_points": {"visible": 2, "actual": 2},
                },
                "public_state": {
                    "turn": {"turn_player_id": "RED", "vps_to_win": 10},
                    "players": {
                        "RED": {
                            "vp": 2,
                            "resource_card_count": 1,
                            "development_card_count": 0,
                            "roads_left": 13,
                            "settlements_left": 3,
                            "cities_left": 4,
                            "longest_road_length": 1,
                            "played_knights": 0,
                        }
                    },
                    "board": {},
                    "bank": {},
                },
                "memory": {"short_term": None, "long_term": None},
            },
        )

        self.assertIn("### Identity", rendered)
        self.assertIn("You are player RED.", rendered)
        self.assertIn("Current turn owner: RED. That is you.", rendered)
        self.assertIn(
            "Development (played knights public; unplayed cards private, including VP cards): 0 played knights; no unplayed development cards",
            rendered,
        )

    def test_game_context_renders_longest_road_and_largest_army_status(self) -> None:
        renderer = PromptRenderer()
        rendered = renderer.render(
            "partials/game_context.jinja",
            payload={
                "turn_index": 6,
                "player_id": "WHITE",
                "private_state": {
                    "resources": {"WOOD": 2},
                    "development_cards": {"KNIGHT": 2},
                    "pieces": {"roads": 13, "settlements": 3, "cities": 4},
                    "victory_points": {"visible": 2, "actual": 2},
                },
                "public_state": {
                    "players": {
                        "WHITE": {
                            "vp": 2,
                            "resource_card_count": 4,
                            "development_card_count": 2,
                            "roads_left": 13,
                            "settlements_left": 3,
                            "cities_left": 4,
                            "longest_road_length": 4,
                            "played_knights": 1,
                            "has_longest_road": False,
                            "has_largest_army": False,
                        },
                        "BLUE": {
                            "vp": 4,
                            "resource_card_count": 5,
                            "development_card_count": 1,
                            "roads_left": 10,
                            "settlements_left": 3,
                            "cities_left": 3,
                            "longest_road_length": 6,
                            "played_knights": 3,
                            "has_longest_road": True,
                            "has_largest_army": True,
                        },
                    },
                    "board": {},
                    "bank": {},
                },
                "memory": {"short_term": None, "long_term": None},
            },
        )

        self.assertIn("Public Longest Road status:", rendered)
        self.assertIn(
            "WHITE [this is you]: 2 Victory Points  4 resource cards  2 unused development cards  pieces left 13R/3S/4C  longest road 4  played knights 1",
            rendered,
        )
        self.assertIn(
            "BLUE: 4 Victory Points  5 resource cards  1 unused development card  pieces left 10R/3S/3C  longest road 6  played knights 3  [Longest Road]  [Largest Army]",
            rendered,
        )
        self.assertIn("BLUE currently holds Longest Road at length 6.", rendered)
        self.assertIn("Your current longest road is 4.", rendered)
        self.assertIn("Public Largest Army status:", rendered)
        self.assertIn(
            "BLUE currently holds Largest Army with 3 played knights.", rendered
        )
        self.assertIn("Your played knights count is 1.", rendered)
        self.assertIn("Unused knights in hand do not count yet.", rendered)

    def test_game_context_renders_rich_board_summary_with_other_players(self) -> None:
        renderer = PromptRenderer()
        rendered = renderer.render(
            "partials/game_context.jinja",
            payload={
                "turn_index": 39,
                "player_id": "BLUE",
                "private_state": {
                    "resources": {"WOOD": 2, "WHEAT": 2},
                    "development_cards": {},
                    "pieces": {"roads": 9, "settlements": 3, "cities": 4},
                    "victory_points": {"visible": 2, "actual": 2},
                },
                "public_state": {
                    "turn": {"turn_player_id": "BLUE", "vps_to_win": 10},
                    "players": {
                        "RED": {
                            "vp": 2,
                            "resource_card_count": 7,
                            "development_card_count": 2,
                            "roads_left": 12,
                            "settlements_left": 3,
                            "cities_left": 4,
                            "longest_road_length": 2,
                            "played_knights": 0,
                        },
                        "BLUE": {
                            "vp": 2,
                            "resource_card_count": 4,
                            "development_card_count": 0,
                            "roads_left": 9,
                            "settlements_left": 3,
                            "cities_left": 4,
                            "longest_road_length": 4,
                            "played_knights": 1,
                        },
                        "WHITE": {
                            "vp": 2,
                            "resource_card_count": 5,
                            "development_card_count": 0,
                            "roads_left": 10,
                            "settlements_left": 3,
                            "cities_left": 4,
                            "longest_road_length": 4,
                            "played_knights": 0,
                        },
                    },
                    "board": {
                        "robber_coordinate": [0, 1, -1],
                        "robber_tile_summary": "ORE@9",
                        "your_network": {
                            "adjacent_tiles": [
                                "SHEEP@11",
                                "BRICK@5",
                                "WHEAT@4",
                                "ORE@9",
                                "WHEAT@5",
                                "BRICK@2",
                            ],
                            "buildings": [
                                {
                                    "node_id": 0,
                                    "building": "SETTLEMENT",
                                    "adjacent_tiles": [
                                        "SHEEP@11",
                                        "BRICK@5",
                                        "WHEAT@4",
                                    ],
                                },
                                {
                                    "node_id": 7,
                                    "building": "SETTLEMENT",
                                    "adjacent_tiles": ["ORE@9", "WHEAT@5", "BRICK@2"],
                                    "ports": ["ANY"],
                                },
                            ],
                            "roads": [
                                {"edge": [0, 1]},
                                {"edge": [1, 2]},
                                {"edge": [2, 3]},
                            ],
                        },
                        "other_player_networks": [
                            {
                                "player_id": "RED",
                                "roads_built": 3,
                                "buildings": [
                                    {
                                        "node_id": 12,
                                        "building": "SETTLEMENT",
                                        "adjacent_tiles": [
                                            "WOOD@6",
                                            "BRICK@3",
                                            "SHEEP@11",
                                        ],
                                    }
                                ],
                            },
                            {
                                "player_id": "WHITE",
                                "roads_built": 4,
                                "buildings": [
                                    {
                                        "node_id": 31,
                                        "building": "CITY",
                                        "adjacent_tiles": [
                                            "WOOD@6",
                                            "WHEAT@9",
                                            "SHEEP@4",
                                        ],
                                        "ports": ["ANY"],
                                    }
                                ],
                            },
                        ],
                        "settlement_candidates": [
                            {
                                "action_index": 12,
                                "node_id": 19,
                                "adjacent_tiles": ["WOOD@6", "BRICK@9", "SHEEP@3"],
                            }
                        ],
                        "road_candidates": [
                            {
                                "action_index": 7,
                                "edge": [7, 11],
                                "adjacent_tiles": ["ORE@9", "WHEAT@5"],
                            }
                        ],
                    },
                    "bank": {"resources": {"WOOD": 17, "BRICK": 12}},
                },
                "memory": {"short_term": None, "long_term": None},
            },
        )

        self.assertIn("Robber: [0, 1, -1] (on ORE@9)", rendered)
        self.assertIn("Your position:", rendered)
        self.assertIn(
            "Tiles: SHEEP@11, BRICK@5, WHEAT@4, ORE@9, WHEAT@5, BRICK@2",
            rendered,
        )
        self.assertIn(
            "SETTLEMENT at node 0 (SHEEP@11, BRICK@5, WHEAT@4)",
            rendered,
        )
        self.assertIn(
            "SETTLEMENT at node 7 (ORE@9, WHEAT@5, BRICK@2) [port: ANY]",
            rendered,
        )
        self.assertIn("Other players on board:", rendered)
        self.assertIn(
            "- RED: 3 roads built; SETTLEMENT at node 12 (WOOD@6, BRICK@3, SHEEP@11)",
            rendered,
        )
        self.assertIn(
            "- WHITE: 4 roads built; CITY at node 31 (WOOD@6, WHEAT@9, SHEEP@4) [port: ANY]",
            rendered,
        )
        self.assertIn("Road candidates:", rendered)
        self.assertIn("Bank: 17×WOOD, 12×BRICK", rendered)

    def test_board_summary_mentions_ports_for_own_and_other_buildings(self) -> None:
        renderer = PromptRenderer()
        rendered = renderer.render(
            "partials/board_context_rich.jinja",
            payload={
                "public_state": {
                    "board": {
                        "robber_coordinate": [0, 1, -1],
                        "robber_tile_summary": "ORE@9",
                        "your_network": {
                            "adjacent_tiles": ["ORE@9", "WHEAT@5", "BRICK@2"],
                            "buildings": [
                                {
                                    "node_id": 7,
                                    "building": "SETTLEMENT",
                                    "adjacent_tiles": ["ORE@9", "WHEAT@5", "BRICK@2"],
                                    "ports": ["ANY"],
                                }
                            ],
                            "roads": [{"edge": [7, 11]}],
                        },
                        "other_player_networks": [
                            {
                                "player_id": "ORANGE",
                                "roads_built": 2,
                                "buildings": [
                                    {
                                        "node_id": 27,
                                        "building": "SETTLEMENT",
                                        "adjacent_tiles": ["ORE@6", "WHEAT@11", "BRICK@4"],
                                        "ports": ["ORE"],
                                    }
                                ],
                            }
                        ],
                    },
                    "bank": {},
                }
            },
        )

        self.assertIn(
            "SETTLEMENT at node 7 (ORE@9, WHEAT@5, BRICK@2) [port: ANY]",
            rendered,
        )
        self.assertIn(
            "- ORANGE: 2 roads built; SETTLEMENT at node 27 (ORE@6, WHEAT@11, BRICK@4) [port: ORE]",
            rendered,
        )

    def test_trade_chat_open_prompt_renders_rich_board_and_marks_viewer(self) -> None:
        renderer = PromptRenderer()
        rendered = renderer.render(
            "trade_chat_open_user.jinja",
            payload={
                "turn_index": 12,
                "player_id": "BLUE",
                "private_state": {
                    "resources": {"WOOD": 1},
                    "development_cards": {},
                    "pieces": {"roads": 13, "settlements": 3, "cities": 4},
                    "victory_points": {"visible": 2, "actual": 2},
                },
                "public_state": {
                    "turn": {"turn_player_id": "BLUE", "vps_to_win": 10},
                    "players": {
                        "BLUE": {
                            "vp": 2,
                            "resource_card_count": 1,
                            "development_card_count": 0,
                            "roads_left": 13,
                            "settlements_left": 3,
                            "cities_left": 4,
                            "longest_road_length": 2,
                            "played_knights": 0,
                        },
                        "RED": {
                            "vp": 3,
                            "resource_card_count": 5,
                            "development_card_count": 1,
                            "roads_left": 11,
                            "settlements_left": 2,
                            "cities_left": 4,
                            "longest_road_length": 3,
                            "played_knights": 1,
                        },
                    },
                    "board": {
                        "robber_coordinate": [0, 1, -1],
                        "robber_tile_summary": "ORE@9",
                        "your_network": {
                            "adjacent_tiles": ["ORE@9", "WHEAT@5"],
                            "buildings": [
                                {
                                    "node_id": 7,
                                    "building": "SETTLEMENT",
                                    "adjacent_tiles": ["ORE@9", "WHEAT@5", "BRICK@2"],
                                }
                            ],
                            "roads": [{"edge": [7, 11]}],
                        },
                        "other_player_networks": [
                            {
                                "player_id": "RED",
                                "roads_built": 3,
                                "buildings": [
                                    {
                                        "node_id": 12,
                                        "building": "SETTLEMENT",
                                        "adjacent_tiles": ["WOOD@6", "BRICK@3"],
                                    }
                                ],
                            }
                        ],
                    },
                    "bank": {"resources": {"WOOD": 17}},
                },
                "memory": {"short_term": None, "long_term": None},
                "requested_resources": {"ORE": 1},
                "other_player_ids": ["RED", "WHITE", "ORANGE"],
                "message_char_limit": 500,
                "public_history": (),
                "public_chat_transcript": (),
            },
        )

        self.assertIn(
            "BLUE [this is you]: 2 Victory Points  1 resource card  0 unused development cards  pieces left 13R/3S/4C  longest road 2  played knights 0",
            rendered,
        )
        self.assertIn("Robber: [0, 1, -1] (on ORE@9)", rendered)
        self.assertIn("Your position:", rendered)
        self.assertIn(
            "- RED: 3 roads built; SETTLEMENT at node 12 (WOOD@6, BRICK@3)",
            rendered,
        )
        self.assertIn("Bank: 17×WOOD", rendered)

    def test_turn_end_prompt_uses_rich_board_context(self) -> None:
        renderer = PromptRenderer()
        rendered = renderer.render(
            "turn_end_user.jinja",
            payload={
                "turn_index": 12,
                "player_id": "BLUE",
                "private_state": {
                    "resources": {"WOOD": 1},
                    "development_cards": {},
                    "pieces": {"roads": 13, "settlements": 3, "cities": 4},
                    "victory_points": {"visible": 2, "actual": 2},
                },
                "public_state": {
                    "turn": {"turn_player_id": "BLUE", "vps_to_win": 10},
                    "players": {
                        "BLUE": {
                            "vp": 2,
                            "resource_card_count": 1,
                            "development_card_count": 0,
                            "roads_left": 13,
                            "settlements_left": 3,
                            "cities_left": 4,
                            "longest_road_length": 2,
                            "played_knights": 0,
                        }
                    },
                    "board": {
                        "robber_coordinate": [0, 1, -1],
                        "robber_tile_summary": "ORE@9",
                        "your_network": {
                            "adjacent_tiles": ["ORE@9"],
                            "buildings": [],
                            "roads": [],
                        },
                        "other_player_networks": [
                            {
                                "player_id": "RED",
                                "roads_built": 3,
                                "buildings": [
                                    {
                                        "node_id": 12,
                                        "building": "SETTLEMENT",
                                        "adjacent_tiles": ["WOOD@6", "BRICK@3"],
                                    }
                                ],
                            }
                        ],
                    },
                    "bank": {"resources": {"WOOD": 17}},
                },
                "memory": {"short_term": None, "long_term": None},
                "turn_public_events": (),
                "public_chat_enabled": False,
                "public_chat_transcript": (),
                "public_chat_message_char_limit": 500,
            },
        )

        self.assertIn("Robber: [0, 1, -1] (on ORE@9)", rendered)
        self.assertIn("Bank: 17×WOOD", rendered)
        self.assertIn("Your position:", rendered)
        self.assertIn("Tiles: ORE@9", rendered)
        self.assertIn("Buildings: none", rendered)
        self.assertIn("Other players on board:", rendered)
        self.assertIn(
            "- RED: 3 roads built; SETTLEMENT at node 12 (WOOD@6, BRICK@3)",
            rendered,
        )

    def test_post_game_prompt_uses_rich_board_context(self) -> None:
        renderer = PromptRenderer()
        rendered = renderer.render(
            "post_game_chat_user.jinja",
            payload={
                "turn_index": 12,
                "player_id": "BLUE",
                "private_state": {
                    "resources": {"WOOD": 1},
                    "development_cards": {},
                    "pieces": {"roads": 13, "settlements": 3, "cities": 4},
                    "victory_points": {"visible": 2, "actual": 2},
                },
                "public_state": {
                    "turn": {"turn_player_id": "BLUE", "vps_to_win": 10},
                    "players": {
                        "BLUE": {
                            "vp": 2,
                            "resource_card_count": 1,
                            "development_card_count": 0,
                            "roads_left": 13,
                            "settlements_left": 3,
                            "cities_left": 4,
                            "longest_road_length": 2,
                            "played_knights": 0,
                        }
                    },
                    "board": {
                        "robber_coordinate": [0, 1, -1],
                        "robber_tile_summary": "ORE@9",
                        "your_network": {
                            "adjacent_tiles": ["ORE@9"],
                            "buildings": [],
                            "roads": [],
                        },
                        "other_player_networks": [
                            {
                                "player_id": "RED",
                                "roads_built": 3,
                                "buildings": [],
                            }
                        ],
                    },
                    "bank": {"resources": {"WOOD": 17}},
                },
                "memory": {"short_term": None, "long_term": None},
                "public_history": (),
                "public_chat_enabled": False,
                "public_chat_transcript": (),
                "public_chat_message_char_limit": 500,
                "result": {"winner_ids": ["WHITE"], "num_turns": 12},
            },
        )

        self.assertIn("Robber: [0, 1, -1] (on ORE@9)", rendered)
        self.assertIn("Bank: 17×WOOD", rendered)
        self.assertIn("Your position:", rendered)
        self.assertIn("Tiles: ORE@9", rendered)
        self.assertIn("Other players on board:", rendered)
        self.assertIn("- RED: 3 roads built; no public buildings", rendered)

    def test_rules_summary_explains_board_notation(self) -> None:
        self.assertIn(
            "Board notation: hex coordinates like [q, r, s] identify tiles only.",
            CATAN_RULES_SUMMARY,
        )
        self.assertIn(
            "Roads are identified by edge endpoint pairs like [18, 23]",
            CATAN_RULES_SUMMARY,
        )

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

    def test_public_chat_context_frames_table_talk_purpose(self) -> None:
        renderer = PromptRenderer()
        rendered = renderer.render(
            "partials/public_chat_context.jinja",
            payload={
                "public_chat_enabled": True,
                "public_chat_message_char_limit": 500,
                "public_chat_transcript": (),
            },
        )

        self.assertIn("Treat it like real table talk", rendered)
        self.assertIn("influence what other players think or do", rendered)
        self.assertIn("not as a private notebook", rendered)
        self.assertIn("Stay silent if you have nothing strategically useful", rendered)

    def test_public_chat_contract_discourages_plan_dumping(self) -> None:
        renderer = PromptRenderer()
        rendered = renderer.render(
            "partials/public_chat_contract.jinja",
            payload={"public_chat_enabled": True},
        )

        self.assertIn("Good uses: warnings, promises, requests", rendered)
        self.assertIn("Avoid using it just to narrate your internal strategy", rendered)
        self.assertIn("Nobody should feed ORANGE ore right now.", rendered)

    def test_normalize_offer_trade_payload_drops_non_mapping_request(self) -> None:
        normalized = _normalize_offer_trade_payload(
            {"offer": {"WOOD": 1}, "request": "ORE"}
        )

        self.assertEqual(normalized, {"offer": {"WOOD": 1}, "request": {}})

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
