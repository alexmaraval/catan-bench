from __future__ import annotations

import json
import unittest

from catan_bench.players import LLMPlayer
from catan_bench.schemas import (
    Action,
    ActionObservation,
    PlayerMemory,
    ReactiveObservation,
    TradeChatObservation,
    TradeChatQuote,
    TurnEndObservation,
    TurnStartObservation,
)


class FakeLLMClient:
    def __init__(self, *payloads: dict[str, object]) -> None:
        self._payloads = list(payloads)

    def complete(self, *, model, messages, temperature, top_p=None, reasoning_enabled=None):
        payload = self._payloads.pop(0)
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}


class FailingLLMClient:
    def __init__(self, error: RuntimeError) -> None:
        self._error = error

    def complete(self, *, model, messages, temperature, top_p=None, reasoning_enabled=None):
        raise self._error


class RawCompletionClient:
    def __init__(self, *contents: str) -> None:
        self._contents = list(contents)

    def complete(self, *, model, messages, temperature, top_p=None, reasoning_enabled=None):
        content = self._contents.pop(0)
        return {"choices": [{"message": {"content": content}}]}


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
    def test_llm_player_turn_lifecycle_returns_memory_and_traces(self) -> None:
        player = LLMPlayer(
            client=FakeLLMClient(
                {"short_term": {"plan": "Trade first."}},
                {"action_index": 0, "short_term": {"plan": "End now."}, "private_reasoning": "Done."},
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
        self.assertEqual(end_response.long_term, {"focus": "Watch BLUE's brick demand."})

        traces = player.take_prompt_traces()
        self.assertEqual([trace.stage for trace in traces], ["turn_start", "choose_action", "turn_end"])
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

    def test_llm_player_falls_back_when_choose_action_repair_is_still_invalid(self) -> None:
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
        self.assertEqual(response.short_term, {"plan": "Trade first."})
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
                    "selected_player_id": "BLUE",
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
            public_state={"turn": {"turn_player_id": "RED"}},
            private_state={"resources": {"WOOD": 1}},
            transcript=(),
            requested_resources={"BRICK": 1},
            other_player_ids=("BLUE",),
            quotes=(TradeChatQuote(player_id="BLUE", owner_gives={"WOOD": 1}, owner_gets={"BRICK": 1}),),
            game_rules="Rules",
            memory=PlayerMemory(long_term={"goal": "trade"}),
            message_char_limit=120,
        )

        open_response = player.open_trade_chat(observation)
        reply_response = player.respond_trade_chat(observation)
        select_response = player.select_trade_chat_offer(observation)

        self.assertTrue(open_response.open_chat)
        self.assertEqual(open_response.requested_resources, {"BRICK": 1})
        self.assertEqual(reply_response.owner_gives, {"WOOD": 1})
        self.assertEqual(reply_response.owner_gets, {"BRICK": 1})
        self.assertEqual(select_response.selected_player_id, "BLUE")

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
            public_state={"turn": {"turn_player_id": "RED"}},
            private_state={"resources": {"WOOD": 1}},
            transcript=(),
            requested_resources={"BRICK": 1},
            other_player_ids=("BLUE",),
            quotes=(),
            game_rules="Rules",
            memory=PlayerMemory(long_term={"goal": "trade"}),
            message_char_limit=120,
        )

        messages = player._messages_for_trade_chat_open(observation)

        self.assertEqual(messages[1]["role"], "user")
        self.assertIsNotNone(renderer.last_payload)
        assert renderer.last_payload is not None
        self.assertEqual(renderer.last_payload["requested_resources"], {"BRICK": 1})

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
            {"error": {"type": "llm_request_failed", "message": "local model request failed"}},
        )

    def test_start_turn_retries_once_after_invalid_json_response(self) -> None:
        player = LLMPlayer(
            client=RawCompletionClient('{"short_term":"unterminated', '{"short_term":"Trade first."}'),
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


if __name__ == "__main__":
    unittest.main()
