from __future__ import annotations

import json
import unittest

from catan_bench import Action, Event, LLMPlayer, MemoryEntry, Observation
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
    def test_llm_player_returns_action_and_private_memory_write(self) -> None:
        client = FakeLLMClient(
            {
                "action_index": 0,
                "action": {
                    "action_type": "OFFER_TRADE",
                    "payload": {"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
                },
                "private_reasoning": (
                    "Trading for brick unlocks a road now, and keeping extra wood is less "
                    "useful than expanding before BLUE can contest the lane."
                ),
                "private_memory_write": {"belief": "BLUE is short on wood"},
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
            public_history=(
                Event(kind="dice_rolled", payload={"roll": 8}, turn_index=3, phase="play_turn"),
            ),
            private_history=(
                Event(
                    kind="private_state_changed",
                    payload={"resources": {"WOOD": 2}},
                    turn_index=3,
                    phase="play_turn",
                ),
            ),
            legal_actions=(
                Action(
                    "OFFER_TRADE",
                    payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
                ),
                Action("END_TURN"),
            ),
            memory=(
                MemoryEntry(
                    player_id="RED",
                    content={"belief": "BLUE needs wood"},
                    turn_index=2,
                    phase="play_turn",
                    decision_index=8,
                ),
            ),
        )

        response = player.respond(observation)
        prompt_trace = player.take_last_prompt_trace()

        self.assertEqual(response.action.action_type, "OFFER_TRADE")
        self.assertEqual(response.action.payload["offer"], {"WOOD": 1})
        self.assertIn("unlocks a road now", response.reasoning or "")
        self.assertEqual(response.memory_write, {"belief": "BLUE is short on wood"})
        self.assertEqual(client.calls[0]["model"], "fake-model")
        self.assertIn("public_history", client.calls[0]["messages"][1]["content"])
        self.assertIsNotNone(prompt_trace)
        self.assertEqual(len(prompt_trace.attempts), 1)
        self.assertEqual(prompt_trace.attempts[0].response["action_index"], 0)

    def test_llm_player_repairs_one_illegal_action_attempt(self) -> None:
        client = FakeLLMClient(
            [
                {
                    "action": {
                        "action_type": "BUILD_SETTLEMENT",
                        "payload": {"node_id": 19},
                    },
                    "private_reasoning": "Node 19 would be strong if it were legal.",
                    "private_memory_write": {"plan": "Prioritize strong ore spots."},
                },
                {
                    "action_index": 1,
                    "private_reasoning": "Node 22 is the best remaining legal ore setup.",
                    "private_memory_write": {"plan": "Claim ore access early."},
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
        self.assertEqual(len(prompt_trace.attempts), 2)
        self.assertEqual(
            prompt_trace.attempts[0].response["action"]["payload"],
            {"node_id": 19},
        )
        self.assertEqual(prompt_trace.attempts[1].response["action_index"], 1)

    def test_llm_player_repairs_trade_template_selected_by_index(self) -> None:
        client = FakeLLMClient(
            [
                {
                    "action_index": 0,
                    "private_reasoning": "I want to trade.",
                    "private_memory_write": {"plan": "Look for profitable trades."},
                },
                {
                    "action_index": 1,
                    "private_reasoning": "No concrete trade is available, so end the turn.",
                    "private_memory_write": {"plan": "Only trade with concrete offers."},
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
        self.assertEqual(len(prompt_trace.attempts), 2)
        self.assertEqual(prompt_trace.attempts[0].response["action_index"], 0)
        self.assertEqual(prompt_trace.attempts[1].response["action_index"], 1)

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
                    "private_memory_write": {"plan": "Fallback gracefully if needed."},
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
        self.assertEqual(response.memory_write, {"plan": "Fallback gracefully if needed."})

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

    def test_llm_player_bounds_prompt_history_and_memory(self) -> None:
        client = FakeLLMClient({"action_index": 0, "private_reasoning": "Take the legal move."})
        player = LLMPlayer(
            client=client,
            model="fake-model",
            temperature=0.1,
            prompt_history_limit=2,
            prompt_memory_limit=1,
        )

        observation = Observation(
            game_id="game-1",
            player_id="RED",
            turn_index=4,
            phase="play_turn",
            decision_index=7,
            public_state={},
            private_state={},
            public_history=tuple(
                Event(kind=f"public-{index}", turn_index=index, phase="play_turn")
                for index in range(5)
            ),
            private_history=tuple(
                Event(kind=f"private-{index}", turn_index=index, phase="play_turn")
                for index in range(4)
            ),
            legal_actions=(Action("END_TURN"),),
            memory=tuple(
                MemoryEntry(
                    player_id="RED",
                    content={"note": f"memory-{index}"},
                    turn_index=index,
                    phase="play_turn",
                    decision_index=index,
                )
                for index in range(3)
            ),
        )

        player.respond(observation)

        payload = json.loads(client.calls[0]["messages"][1]["content"])
        self.assertEqual(
            [event["kind"] for event in payload["public_history"]],
            ["public-3", "public-4"],
        )
        self.assertEqual(
            [event["kind"] for event in payload["private_history"]],
            ["private-2", "private-3"],
        )
        self.assertEqual(len(payload["private_memory"]), 1)
        self.assertEqual(payload["private_memory"][0]["content"], {"note": "memory-2"})
        self.assertEqual(payload["context_window"]["public_history_available"], 5)
        self.assertEqual(payload["context_window"]["public_history_included"], 2)

    def test_llm_player_retries_with_compact_prompt_after_oversized_request(self) -> None:
        client = OversizeThenSuccessClient(
            {"action_index": 0, "private_reasoning": "Compact retry succeeded."}
        )
        player = LLMPlayer(
            client=client,
            model="fake-model",
            temperature=0.1,
            prompt_history_limit=6,
            prompt_memory_limit=5,
        )

        observation = Observation(
            game_id="game-1",
            player_id="RED",
            turn_index=6,
            phase="play_turn",
            decision_index=11,
            public_state={},
            private_state={},
            public_history=tuple(
                Event(kind=f"public-{index}", turn_index=index, phase="play_turn")
                for index in range(8)
            ),
            private_history=tuple(
                Event(kind=f"private-{index}", turn_index=index, phase="play_turn")
                for index in range(8)
            ),
            legal_actions=(Action("END_TURN"),),
            memory=tuple(
                MemoryEntry(
                    player_id="RED",
                    content={"note": f"memory-{index}"},
                    turn_index=index,
                    phase="play_turn",
                    decision_index=index,
                )
                for index in range(7)
            ),
        )

        response = player.respond(observation)
        prompt_trace = player.take_last_prompt_trace()

        self.assertEqual(response.action.action_type, "END_TURN")
        self.assertEqual(len(client.calls), 2)
        first_payload = json.loads(client.calls[0]["messages"][1]["content"])
        second_payload = json.loads(client.calls[1]["messages"][1]["content"])
        self.assertEqual(first_payload["context_window"]["public_history_included"], 6)
        self.assertEqual(second_payload["context_window"]["public_history_included"], 3)
        self.assertEqual(second_payload["context_window"]["private_memory_included"], 2)
        self.assertTrue(second_payload["context_window"]["compact_retry"])
        self.assertIsNotNone(prompt_trace)
        assert prompt_trace is not None
        self.assertEqual(prompt_trace.attempts[0].response["error"]["type"], "request_too_large")
        self.assertEqual(prompt_trace.attempts[1].response["action_index"], 0)


if __name__ == "__main__":
    unittest.main()
