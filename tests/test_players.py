from __future__ import annotations

import json
import unittest

from catan_bench import (
    Action,
    Event,
    LLMPlayer,
    MemoryEntry,
    Observation,
    RecallObservation,
    ReflectionObservation,
    TradeChatObservation,
    TradeChatQuote,
)
from catan_bench.llm import LLMRequestTooLargeError


class FakeLLMClient:
    def __init__(
        self,
        content: (
            dict[str, object]
            | str
            | list[dict[str, object] | str | dict[str, object]]
        ),
    ) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        temperature: float,
        top_p: float | None = None,
        reasoning_enabled: bool | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "reasoning_enabled": reasoning_enabled,
            }
        )
        if isinstance(self.content, list):
            if not self.content:
                raise AssertionError("FakeLLMClient ran out of scripted completions.")
            content = self.content.pop(0)
        else:
            content = self.content
        if isinstance(content, dict) and "choices" in content:
            return content
        return {
            "choices": [
                {
                    "message": {
                        "content": content if isinstance(content, str) else json.dumps(content),
                    }
                }
            ]
        }


class OversizeThenSuccessClient:
    def __init__(self, success_payload: dict[str, object]) -> None:
        self.success_payload = success_payload
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        temperature: float,
        top_p: float | None = None,
        reasoning_enabled: bool | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "reasoning_enabled": reasoning_enabled,
            }
        )
        if len(self.calls) == 1:
            raise LLMRequestTooLargeError("first attempt was too large")
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(self.success_payload),
                    }
                }
            ]
        }


class LLMPlayerTests(unittest.TestCase):
    def test_llm_player_recall_returns_consolidated_memory(self) -> None:
        client = FakeLLMClient({"private_memory": {"belief": "BLUE is short on wood"}})
        player = LLMPlayer(client=client, model="fake-model", temperature=0.1)

        response = player.recall(
            RecallObservation(
                game_id="game-1",
                player_id="RED",
                turn_index=3,
                phase="play_turn",
                decision_index=12,
                game_rules="Rules",
                public_events_since_last_turn=(
                    Event(kind="dice_rolled", payload={"roll": 8}, turn_index=3, phase="play_turn"),
                ),
                private_events_since_last_turn=(
                    Event(
                        kind="trade_offer_received",
                        payload={"from": "BLUE"},
                        turn_index=3,
                        phase="play_turn",
                    ),
                ),
                memory=MemoryEntry(
                    player_id="RED",
                    content={"belief": "BLUE needs wood"},
                    turn_index=2,
                    phase="play_turn",
                    decision_index=8,
                ),
            )
        )

        prompt_trace = player.take_last_prompt_trace()
        payload = json.loads(client.calls[0]["messages"][1]["content"])

        self.assertEqual(response.memory, {"belief": "BLUE is short on wood"})
        self.assertEqual(payload["current_memory"], {"belief": "BLUE needs wood"})
        self.assertIn("public_events_since_last_turn", payload)
        self.assertIsNotNone(prompt_trace)
        assert prompt_trace is not None
        self.assertEqual(prompt_trace.stage, "recall")

    def test_llm_player_returns_action_and_reasoning(self) -> None:
        client = FakeLLMClient(
            {
                "action_index": 0,
                "private_reasoning": "Trading for brick unlocks a road now.",
            }
        )
        player = LLMPlayer(client=client, model="fake-model", temperature=0.1)

        observation = Observation(
            game_id="game-1",
            player_id="RED",
            turn_index=3,
            phase="play_turn",
            decision_index=12,
            public_state={"scores": {"RED": 3, "BLUE": 2}},
            private_state={"resources": {"WOOD": 2, "BRICK": 0}},
            game_rules="Rules",
            decision_prompt="Choose an action for your turn.",
            legal_actions=(
                Action(
                    "OFFER_TRADE",
                    payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
                ),
                Action("END_TURN"),
            ),
            memory=MemoryEntry(
                player_id="RED",
                content={"belief": "BLUE needs wood"},
                turn_index=2,
                phase="play_turn",
                decision_index=8,
            ),
        )

        response = player.respond(observation)
        prompt_trace = player.take_last_prompt_trace()
        payload = json.loads(client.calls[0]["messages"][1]["content"])

        self.assertEqual(response.action.action_type, "OFFER_TRADE")
        self.assertEqual(response.action.payload["offer"], {"WOOD": 1})
        self.assertEqual(response.reasoning, "Trading for brick unlocks a road now.")
        self.assertEqual(payload["private_memory"], {"belief": "BLUE needs wood"})
        self.assertNotIn("public_history", payload)
        self.assertIsNotNone(prompt_trace)
        assert prompt_trace is not None
        self.assertEqual(prompt_trace.stage, "act")
        self.assertEqual(
            prompt_trace.attempts[0].response_text,
            json.dumps(
                {
                    "action_index": 0,
                    "private_reasoning": "Trading for brick unlocks a road now.",
                }
            ),
        )

    def test_llm_player_repairs_one_illegal_action_attempt(self) -> None:
        client = FakeLLMClient(
            [
                {
                    "action": {
                        "action_type": "BUILD_SETTLEMENT",
                        "payload": {"node_id": 19},
                    },
                    "private_reasoning": "Node 19 would be strong if it were legal.",
                },
                {
                    "action_index": 1,
                    "private_reasoning": "Node 22 is the best remaining legal ore setup.",
                },
            ]
        )
        player = LLMPlayer(client=client, model="fake-model", temperature=0.1)

        observation = Observation(
            game_id="game-1",
            player_id="RED",
            turn_index=0,
            phase="build_initial_settlement",
            decision_index=0,
            public_state={},
            private_state={},
            legal_actions=(
                Action("BUILD_SETTLEMENT", payload={"node_id": 10}),
                Action("BUILD_SETTLEMENT", payload={"node_id": 22}),
            ),
        )

        response = player.respond(observation)
        prompt_trace = player.take_last_prompt_trace()

        self.assertEqual(response.action.payload, {"node_id": 22})
        self.assertIn("best remaining legal", response.reasoning or "")
        self.assertEqual(len(client.calls), 2)
        self.assertIn("illegal action", client.calls[1]["messages"][-1]["content"])
        self.assertIsNotNone(prompt_trace)
        assert prompt_trace is not None
        self.assertEqual(prompt_trace.stage, "act")
        self.assertEqual(len(prompt_trace.attempts), 2)

    def test_llm_player_repairs_malformed_action_shape(self) -> None:
        client = FakeLLMClient(
            [
                {
                    "private_reasoning": "Take a strong opening spot.",
                    "note": "missing action fields",
                },
                {
                    "action_index": 1,
                    "private_reasoning": "Choose the legal fallback opening.",
                },
            ]
        )
        player = LLMPlayer(client=client, model="fake-model", temperature=0.1)

        observation = Observation(
            game_id="game-1",
            player_id="RED",
            turn_index=0,
            phase="build_initial_settlement",
            decision_index=0,
            public_state={},
            private_state={},
            legal_actions=(
                Action("BUILD_SETTLEMENT", payload={"node_id": 10}),
                Action("BUILD_SETTLEMENT", payload={"node_id": 22}),
            ),
        )

        response = player.respond(observation)
        prompt_trace = player.take_last_prompt_trace()

        self.assertEqual(response.action.payload, {"node_id": 22})
        self.assertIsNotNone(prompt_trace)
        assert prompt_trace is not None
        self.assertEqual(len(prompt_trace.attempts), 2)
        self.assertIn(
            "must include an `action_index` or an `action` object",
            client.calls[1]["messages"][-1]["content"],
        )

    def test_llm_player_repairs_trade_template_selected_by_index(self) -> None:
        client = FakeLLMClient(
            [
                {
                    "action_index": 0,
                    "private_reasoning": "I want to trade.",
                },
                {
                    "action_index": 1,
                    "private_reasoning": "No concrete trade is available, so end the turn.",
                },
            ]
        )
        player = LLMPlayer(client=client, model="fake-model", temperature=0.1)

        observation = Observation(
            game_id="game-1",
            player_id="WHITE",
            turn_index=5,
            phase="play_turn",
            decision_index=18,
            public_state={},
            private_state={"resources": {"WOOD": 0, "BRICK": 0}},
            legal_actions=(
                Action("OFFER_TRADE", payload={"offer": {}, "request": {}}),
                Action("END_TURN"),
            ),
        )

        response = player.respond(observation)
        prompt_trace = player.take_last_prompt_trace()

        self.assertEqual(response.action.action_type, "END_TURN")
        self.assertEqual(len(client.calls), 2)
        self.assertIsNotNone(prompt_trace)
        assert prompt_trace is not None
        self.assertEqual(len(prompt_trace.attempts), 2)

    def test_llm_player_parses_json_wrapped_in_markdown_fences(self) -> None:
        client = FakeLLMClient(
            '```json\n{"action_index": 1, "private_reasoning": "Take the safe action."}\n```'
        )
        player = LLMPlayer(client=client, model="fake-model", temperature=0.1)

        observation = Observation(
            game_id="game-1",
            player_id="RED",
            turn_index=1,
            phase="play_turn",
            decision_index=2,
            public_state={},
            private_state={},
            legal_actions=(
                Action("BUILD_ROAD", payload={"edge_id": 3}),
                Action("END_TURN"),
            ),
        )

        response = player.respond(observation)
        self.assertEqual(response.action.action_type, "END_TURN")
        self.assertEqual(response.reasoning, "Take the safe action.")

    def test_llm_player_accepts_integer_like_float_action_index(self) -> None:
        client = FakeLLMClient(
            {
                "action_index": 1.0,
                "private_reasoning": "The second action is the intended choice.",
            }
        )
        player = LLMPlayer(client=client, model="fake-model", temperature=0.1)

        observation = Observation(
            game_id="game-1",
            player_id="RED",
            turn_index=1,
            phase="play_turn",
            decision_index=3,
            public_state={},
            private_state={},
            legal_actions=(
                Action("BUILD_ROAD", payload={"edge_id": 3}),
                Action("END_TURN"),
            ),
        )

        response = player.respond(observation)
        self.assertEqual(response.action.action_type, "END_TURN")

    def test_llm_player_accepts_top_level_action_type_payload_shape(self) -> None:
        client = FakeLLMClient(
            {
                "action_type": "BUILD_ROAD",
                "payload": {"edge_id": 3},
                "reasoning": "The top-level shape is still clear enough.",
            }
        )
        player = LLMPlayer(client=client, model="fake-model", temperature=0.1)

        observation = Observation(
            game_id="game-1",
            player_id="RED",
            turn_index=1,
            phase="play_turn",
            decision_index=3,
            public_state={},
            private_state={},
            legal_actions=(
                Action("BUILD_ROAD", payload={"edge_id": 3}),
                Action("END_TURN"),
            ),
        )

        response = player.respond(observation)

        self.assertEqual(response.action.action_type, "BUILD_ROAD")
        self.assertEqual(response.reasoning, "The top-level shape is still clear enough.")

    def test_llm_player_falls_back_to_first_legal_action_after_failed_repair(self) -> None:
        client = FakeLLMClient(
            [
                {
                    "action": {
                        "action_type": "BUILD_SETTLEMENT",
                        "payload": {"node_id": 99},
                    },
                    "private_reasoning": "Try an aggressive but illegal settlement.",
                },
                {
                    "action": {
                        "action_type": "BUILD_SETTLEMENT",
                        "payload": {"node_id": 77},
                    },
                    "private_reasoning": "Second attempt is still illegal.",
                },
            ]
        )
        player = LLMPlayer(client=client, model="fake-model", temperature=0.1)

        observation = Observation(
            game_id="game-1",
            player_id="RED",
            turn_index=0,
            phase="build_initial_settlement",
            decision_index=0,
            public_state={},
            private_state={},
            legal_actions=(
                Action("BUILD_SETTLEMENT", payload={"node_id": 10}),
                Action("BUILD_SETTLEMENT", payload={"node_id": 22}),
            ),
        )

        response = player.respond(observation)

        self.assertEqual(response.action.payload, {"node_id": 10})
        self.assertEqual(response.reasoning, "Second attempt is still illegal.")
        self.assertIsNone(response.memory_write)

    def test_llm_player_passes_optional_sampling_and_reasoning_flags(self) -> None:
        client = FakeLLMClient({"action_index": 0, "private_reasoning": "Safe opening."})
        player = LLMPlayer(
            client=client,
            model="fake-model",
            temperature=0.6,
            top_p=0.95,
            reasoning_enabled=False,
        )

        observation = Observation(
            game_id="game-1",
            player_id="ORANGE",
            turn_index=0,
            phase="build_initial_settlement",
            decision_index=0,
            public_state={},
            private_state={},
            legal_actions=(Action("END_TURN"),),
        )

        response = player.respond(observation)

        self.assertEqual(response.action.action_type, "END_TURN")
        self.assertEqual(client.calls[0]["top_p"], 0.95)
        self.assertFalse(client.calls[0]["reasoning_enabled"])

    def test_llm_player_raises_clear_error_for_reasoning_only_response(self) -> None:
        client = FakeLLMClient(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning": "Long hidden analysis with no final answer.",
                        },
                        "finish_reason": "length",
                    }
                ]
            }
        )
        player = LLMPlayer(client=client, model="fake-model", temperature=0.1)

        observation = Observation(
            game_id="game-1",
            player_id="ORANGE",
            turn_index=0,
            phase="build_initial_settlement",
            decision_index=0,
            public_state={},
            private_state={},
            legal_actions=(Action("END_TURN"),),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "reasoning-only output with finish_reason='length'",
        ):
            player.respond(observation)

    def test_llm_player_bounds_recall_event_window(self) -> None:
        client = FakeLLMClient({"private_memory": {"note": "keep the latest events"}})
        player = LLMPlayer(
            client=client,
            model="fake-model",
            temperature=0.1,
            prompt_history_limit=2,
        )

        player.recall(
            RecallObservation(
                game_id="game-1",
                player_id="RED",
                turn_index=4,
                phase="play_turn",
                decision_index=7,
                public_events_since_last_turn=tuple(
                    Event(kind=f"public-{index}", turn_index=index, phase="play_turn")
                    for index in range(5)
                ),
                private_events_since_last_turn=tuple(
                    Event(kind=f"private-{index}", turn_index=index, phase="play_turn")
                    for index in range(4)
                ),
                memory=MemoryEntry(
                    player_id="RED",
                    content={"note": "older-memory"},
                    turn_index=3,
                    phase="play_turn",
                    decision_index=6,
                ),
            )
        )

        payload = json.loads(client.calls[0]["messages"][1]["content"])
        self.assertEqual(
            [event["kind"] for event in payload["public_events_since_last_turn"]],
            ["public-0", "public-1", "public-2", "public-3", "public-4"],
        )
        self.assertEqual(payload["current_memory"], {"note": "older-memory"})

    def test_llm_player_retries_recall_with_compact_prompt_after_oversized_request(self) -> None:
        client = OversizeThenSuccessClient({"private_memory": {"plan": "Compact retry worked."}})
        player = LLMPlayer(
            client=client,
            model="fake-model",
            temperature=0.1,
            prompt_history_limit=6,
        )

        response = player.recall(
            RecallObservation(
                game_id="game-1",
                player_id="RED",
                turn_index=6,
                phase="play_turn",
                decision_index=11,
                public_events_since_last_turn=tuple(
                    Event(kind=f"public-{index}", turn_index=index, phase="play_turn")
                    for index in range(8)
                ),
                private_events_since_last_turn=tuple(
                    Event(kind=f"private-{index}", turn_index=index, phase="play_turn")
                    for index in range(8)
                ),
            )
        )
        prompt_trace = player.take_last_prompt_trace()

        self.assertEqual(response.memory, {"plan": "Compact retry worked."})
        self.assertEqual(len(client.calls), 2)
        first_payload = json.loads(client.calls[0]["messages"][1]["content"])
        second_payload = json.loads(client.calls[1]["messages"][1]["content"])
        self.assertEqual(first_payload["context_window"]["public_events_included"], 8)
        self.assertEqual(second_payload["context_window"]["public_events_included"], 3)
        self.assertTrue(second_payload["context_window"]["compact_retry"])
        self.assertIsNotNone(prompt_trace)
        assert prompt_trace is not None
        self.assertEqual(prompt_trace.stage, "recall")
        self.assertEqual(prompt_trace.attempts[0].response["error"]["type"], "request_too_large")

    def test_llm_player_reflect_returns_final_memory(self) -> None:
        client = FakeLLMClient({"private_memory": {"plan": "Trade first, build next."}})
        player = LLMPlayer(client=client, model="fake-model", temperature=0.1)

        response = player.reflect(
            ReflectionObservation(
                game_id="game-1",
                player_id="RED",
                turn_index=3,
                phase="play_turn",
                decision_index=14,
                public_events_this_turn=(
                    Event(kind="trade_offered", turn_index=3, phase="play_turn"),
                    Event(kind="trade_confirmed", turn_index=3, phase="play_turn"),
                ),
                private_events_this_turn=(
                    Event(kind="player_decision", turn_index=3, phase="play_turn"),
                ),
                memory=MemoryEntry(
                    player_id="RED",
                    content={"plan": "Look for brick trades."},
                    turn_index=3,
                    phase="play_turn",
                    decision_index=12,
                    update_kind="recall",
                ),
            )
        )
        prompt_trace = player.take_last_prompt_trace()

        self.assertEqual(response.memory, {"plan": "Trade first, build next."})
        self.assertIsNotNone(prompt_trace)
        assert prompt_trace is not None
        self.assertEqual(prompt_trace.stage, "reflect")

    def test_llm_player_open_trade_chat_returns_request(self) -> None:
        client = FakeLLMClient(
            {
                "open_chat": True,
                "message": "I am looking for 1 wood. What is your market?",
                "requested_resources": {"WOOD": 1},
                "private_reasoning": "Need wood to unlock the road line.",
            }
        )
        player = LLMPlayer(client=client, model="fake-model", temperature=0.1)

        response = player.open_trade_chat(
            TradeChatObservation(
                game_id="game-1",
                player_id="RED",
                owner_player_id="RED",
                turn_index=5,
                phase="play_turn",
                decision_index=9,
                stage="open",
                attempt_index=1,
                public_state={"scores": {"RED": 3, "WHITE": 2, "ORANGE": 2}},
                private_state={"resources": {"SHEEP": 2}},
                other_player_ids=("WHITE", "ORANGE"),
                message_char_limit=120,
                memory=MemoryEntry(
                    player_id="RED",
                    content={"plan": "Buy the missing wood if needed"},
                    turn_index=4,
                    phase="play_turn",
                    decision_index=7,
                ),
            )
        )

        prompt_trace = player.take_last_prompt_trace()
        payload = json.loads(client.calls[0]["messages"][1]["content"])

        self.assertTrue(response.open_chat)
        self.assertEqual(response.requested_resources, {"WOOD": 1})
        self.assertEqual(response.message, "I am looking for 1 wood. What is your market?")
        self.assertEqual(payload["other_player_ids"], ["WHITE", "ORANGE"])
        self.assertEqual(payload["private_memory"], {"plan": "Buy the missing wood if needed"})
        self.assertIsNotNone(prompt_trace)
        assert prompt_trace is not None
        self.assertEqual(prompt_trace.stage, "trade_chat_open")

    def test_llm_player_trade_chat_reply_and_selection_parse_quotes(self) -> None:
        client = FakeLLMClient(
            [
                {
                    "message": "I can do 1 wood for 1 sheep.",
                    "owner_gives": {"SHEEP": 1},
                    "owner_gets": {"WOOD": 1},
                    "private_reasoning": "Competitive quote.",
                },
                {
                    "selected_player_id": "ORANGE",
                    "message": "Taking ORANGE's offer.",
                    "private_reasoning": "Best price on the table.",
                },
            ]
        )
        player = LLMPlayer(client=client, model="fake-model", temperature=0.1)

        reply = player.respond_trade_chat(
            TradeChatObservation(
                game_id="game-1",
                player_id="ORANGE",
                owner_player_id="RED",
                turn_index=5,
                phase="play_turn",
                decision_index=9,
                stage="reply",
                attempt_index=1,
                public_state={},
                private_state={"resources": {"WOOD": 1}},
                requested_resources={"WOOD": 1},
                transcript=(
                    Event(
                        kind="trade_chat_message",
                        payload={"message": "Looking for 1 wood.", "speaker_player_id": "RED"},
                        turn_index=5,
                        phase="play_turn",
                    ),
                ),
            )
        )
        reply_trace = player.take_last_prompt_trace()

        selection = player.select_trade_chat_offer(
            TradeChatObservation(
                game_id="game-1",
                player_id="RED",
                owner_player_id="RED",
                turn_index=5,
                phase="play_turn",
                decision_index=9,
                stage="select",
                attempt_index=1,
                public_state={},
                private_state={"resources": {"SHEEP": 2}},
                requested_resources={"WOOD": 1},
                quotes=(
                    TradeChatQuote(
                        player_id="WHITE",
                        message="I can do 1 wood for 2 sheep.",
                        owner_gives={"SHEEP": 2},
                        owner_gets={"WOOD": 1},
                    ),
                    TradeChatQuote(
                        player_id="ORANGE",
                        message="I can do 1 wood for 1 sheep.",
                        owner_gives={"SHEEP": 1},
                        owner_gets={"WOOD": 1},
                    ),
                ),
            )
        )
        selection_trace = player.take_last_prompt_trace()

        self.assertEqual(reply.owner_gives, {"SHEEP": 1})
        self.assertEqual(reply.owner_gets, {"WOOD": 1})
        self.assertEqual(selection.selected_player_id, "ORANGE")
        self.assertEqual(selection.message, "Taking ORANGE's offer.")
        self.assertIsNotNone(reply_trace)
        self.assertIsNotNone(selection_trace)
        assert reply_trace is not None and selection_trace is not None
        self.assertEqual(reply_trace.stage, "trade_chat_reply")
        self.assertEqual(selection_trace.stage, "trade_chat_select")


if __name__ == "__main__":
    unittest.main()
