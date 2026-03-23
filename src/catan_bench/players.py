from __future__ import annotations

import json
import math
import random
from collections import deque
from typing import Iterable, Protocol

from .llm import LLMClient
from .schemas import Action, JsonValue, Observation, PlayerResponse, PromptTrace, PromptTraceAttempt


class Player(Protocol):
    def respond(self, observation: Observation) -> PlayerResponse:
        ...


class ScriptedPlayer:
    """Small utility adapter for tests and scripted demos."""

    def __init__(self, responses: Iterable[PlayerResponse | Action]) -> None:
        self._responses = deque(responses)
        self.observations: list[Observation] = []

    def respond(self, observation: Observation) -> PlayerResponse:
        self.observations.append(observation)
        if not self._responses:
            raise RuntimeError("ScriptedPlayer ran out of scripted responses.")

        next_response = self._responses.popleft()
        if isinstance(next_response, PlayerResponse):
            return next_response
        return PlayerResponse(action=next_response)


def _is_trade_template(action: Action) -> bool:
    return action.action_type == "OFFER_TRADE" and action.payload == {
        "offer": {},
        "request": {},
    }


def _materialize_default_trade_offer() -> Action:
    return Action(
        "OFFER_TRADE",
        payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
    )


class FirstLegalPlayer:
    """Simple deterministic baseline that always picks the first legal action."""

    def __init__(
        self, *, allow_trade_offers: bool = False, record_observations: bool = False
    ) -> None:
        self.allow_trade_offers = allow_trade_offers
        self.record_observations = record_observations
        self.observations: list[Observation] = []

    def respond(self, observation: Observation) -> PlayerResponse:
        if self.record_observations:
            self.observations.append(observation)
        for action in observation.legal_actions:
            if not _is_trade_template(action):
                return PlayerResponse(action=action)

        if self.allow_trade_offers:
            return PlayerResponse(action=_materialize_default_trade_offer())

        raise RuntimeError("FirstLegalPlayer found no concrete legal actions to take.")


class RandomLegalPlayer:
    """Random baseline that samples from the current legal concrete actions."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        allow_trade_offers: bool = False,
        record_observations: bool = False,
    ) -> None:
        self._rng = random.Random(seed)
        self.allow_trade_offers = allow_trade_offers
        self.record_observations = record_observations
        self.observations: list[Observation] = []

    def respond(self, observation: Observation) -> PlayerResponse:
        if self.record_observations:
            self.observations.append(observation)
        concrete_actions = [
            action for action in observation.legal_actions if not _is_trade_template(action)
        ]
        if concrete_actions:
            return PlayerResponse(action=self._rng.choice(concrete_actions))

        if self.allow_trade_offers:
            return PlayerResponse(action=_materialize_default_trade_offer())

        raise RuntimeError("RandomLegalPlayer found no concrete legal actions to sample.")


class LLMPlayer:
    """LLM-backed player that returns an action, private reasoning, and memory note."""

    def __init__(
        self,
        *,
        client: LLMClient,
        model: str,
        temperature: float = 0.2,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.observations: list[Observation] = []
        self._last_prompt_trace: PromptTrace | None = None

    def respond(self, observation: Observation) -> PlayerResponse:
        self.observations.append(observation)
        trace_attempts: list[PromptTraceAttempt] = []
        messages = self._messages_for(observation)
        response_payload = self._complete_and_trace(
            messages=messages,
            trace_attempts=trace_attempts,
        )
        response = self._response_from_payload(observation, response_payload)
        if self._is_legal_response_action(response.action, observation.legal_actions):
            self._last_prompt_trace = self._prompt_trace_for(
                observation=observation,
                attempts=trace_attempts,
            )
            return response

        repaired_messages = messages + self._repair_messages(
            observation=observation,
            attempted_response=response_payload,
        )
        repaired_payload = self._complete_and_trace(
            messages=repaired_messages,
            trace_attempts=trace_attempts,
        )
        repaired_response = self._response_from_payload(observation, repaired_payload)
        if self._is_legal_response_action(repaired_response.action, observation.legal_actions):
            self._last_prompt_trace = self._prompt_trace_for(
                observation=observation,
                attempts=trace_attempts,
            )
            return repaired_response

        self._last_prompt_trace = self._prompt_trace_for(
            observation=observation,
            attempts=trace_attempts,
        )
        return PlayerResponse(
            action=self._fallback_legal_action(observation),
            memory_write=repaired_response.memory_write,
            reasoning=repaired_response.reasoning,
        )

    def take_last_prompt_trace(self) -> PromptTrace | None:
        trace = self._last_prompt_trace
        self._last_prompt_trace = None
        return trace

    def _messages_for(self, observation: Observation) -> list[dict[str, object]]:
        system_prompt = "\n\n".join(
            part
            for part in (
                observation.game_rules,
                (
                    "Return strict JSON with keys `action_index`, `private_reasoning`, and "
                    "`private_memory_write`. `action_index` should be the integer index of one "
                    "entry from the provided `legal_actions` list and is the preferred way to "
                    "choose an action. You may also include `action` with `action_type` and "
                    "`payload`, but use the exact legal action when you do. If a legal action is "
                    "marked as requiring a concrete payload, do not choose it by `action_index`; "
                    "instead return an `action` object with a concrete payload. `private_reasoning` "
                    "should contain your full private reasoning for this decision, including how "
                    "you are refining your strategy; it will be saved only in your private history. "
                    "`private_memory_write` should be a short distilled note worth keeping for "
                    "future turns."
                ),
            )
            if part
        )
        indexed_legal_actions = [
            {
                "index": index,
                "selectable_by_index": not _is_trade_template(action),
                "requires_concrete_payload": _is_trade_template(action),
                **action.to_dict(),
            }
            for index, action in enumerate(observation.legal_actions)
        ]
        user_payload = {
            "player_id": observation.player_id,
            "turn_index": observation.turn_index,
            "phase": observation.phase,
            "decision_index": observation.decision_index,
            "decision_prompt": observation.decision_prompt,
            "public_state": observation.public_state,
            "private_state": observation.private_state,
            "public_history": [event.to_dict() for event in observation.public_history],
            "private_history": [event.to_dict() for event in observation.private_history],
            "private_memory": [entry.to_dict() for entry in observation.memory],
            "legal_actions": indexed_legal_actions,
            "response_contract": {
                "action_index": "preferred integer index into legal_actions",
                "action": (
                    "optional exact copy of the chosen legal action object; required for any "
                    "action that needs a concrete payload"
                ),
                "private_reasoning": (
                    "required private explanation of this move and any strategy updates; "
                    "kept in your private history"
                ),
                "private_memory_write": (
                    "optional short summary worth remembering on future turns; null if none"
                ),
            },
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
        ]

    def _response_from_payload(
        self, observation: Observation, response_payload: dict[str, object]
    ) -> PlayerResponse:
        action = self._action_from_payload(observation, response_payload)

        memory_write = response_payload.get("private_memory_write")
        reasoning = response_payload.get("private_reasoning")
        if reasoning is not None and not isinstance(reasoning, str):
            raise RuntimeError(
                "LLM response field `private_reasoning` must be a string when present."
            )

        return PlayerResponse(
            action=action,
            memory_write=memory_write,
            reasoning=reasoning,
        )

    @staticmethod
    def _action_from_payload(
        observation: Observation, response_payload: dict[str, object]
    ) -> Action:
        action_index = response_payload.get("action_index")
        if isinstance(action_index, str) and action_index.isdigit():
            action_index = int(action_index)
        elif isinstance(action_index, float) and math.isfinite(action_index):
            rounded_index = round(action_index)
            if action_index == rounded_index:
                action_index = int(rounded_index)
        if isinstance(action_index, int):
            if not 0 <= action_index < len(observation.legal_actions):
                raise RuntimeError(
                    f"LLM action_index {action_index} is out of range for legal actions."
                )
            return observation.legal_actions[action_index]

        action_payload = response_payload.get("action")
        if not isinstance(action_payload, dict):
            raise RuntimeError(
                "LLM response must include an `action_index` or an `action` object."
            )

        action_type = action_payload.get("action_type")
        if not isinstance(action_type, str):
            raise RuntimeError("LLM action must include string field `action_type`.")

        payload = action_payload.get("payload", {})
        if not isinstance(payload, dict):
            raise RuntimeError("LLM action payload must be a JSON object.")

        return Action(action_type=action_type, payload=payload)

    @staticmethod
    def _is_legal_response_action(
        action: Action, legal_actions: tuple[Action, ...]
    ) -> bool:
        if _is_trade_template(action):
            return False
        if any(legal_action.matches(action) for legal_action in legal_actions):
            return True

        matching_types = [
            legal_action
            for legal_action in legal_actions
            if legal_action.action_type == action.action_type
        ]
        if action.action_type in {"ACCEPT_TRADE", "REJECT_TRADE", "CANCEL_TRADE"}:
            return len(matching_types) == 1
        if action.action_type == "CONFIRM_TRADE":
            accepting_player_id = action.payload.get("accepting_player_id")
            # The canonical schema uses `accepting_player_id`; keep legality checks strict.
            return any(
                legal_action.payload.get("accepting_player_id") == accepting_player_id
                for legal_action in matching_types
            )
        return False

    @staticmethod
    def _fallback_legal_action(observation: Observation) -> Action:
        for action in observation.legal_actions:
            if not _is_trade_template(action):
                return action
        if any(_is_trade_template(action) for action in observation.legal_actions):
            return _materialize_default_trade_offer()
        raise RuntimeError(
            "LLM returned an illegal action after repair, and no fallback legal action was available."
        )

    @staticmethod
    def _repair_messages(
        *,
        observation: Observation,
        attempted_response: dict[str, object],
    ) -> list[dict[str, object]]:
        repair_payload = {
            "error": (
                "Your previous response selected an illegal action for the current decision."
            ),
            "previous_response": attempted_response,
            "instruction": (
                "Choose one legal action by returning only `action_index`, "
                "`private_reasoning`, and `private_memory_write`. The action_index must be one "
                "of the provided legal action indexes."
            ),
            "legal_actions": [
                {"index": index, **action.to_dict()}
                for index, action in enumerate(observation.legal_actions)
            ],
        }
        return [
            {
                "role": "system",
                "content": "Repair the previous invalid move by selecting one legal action.",
            },
            {
                "role": "user",
                "content": json.dumps(repair_payload, sort_keys=True),
            },
        ]

    def _complete_and_trace(
        self,
        *,
        messages: list[dict[str, object]],
        trace_attempts: list[PromptTraceAttempt],
    ) -> dict[str, object]:
        completion = self.client.complete(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
        )
        response_payload = self._extract_response_json(completion)
        trace_attempts.append(
            PromptTraceAttempt(
                messages=tuple(self._json_safe_message(message) for message in messages),
                response=self._json_safe_object(response_payload),
            )
        )
        return response_payload

    def _prompt_trace_for(
        self,
        *,
        observation: Observation,
        attempts: list[PromptTraceAttempt],
    ) -> PromptTrace:
        return PromptTrace(
            player_id=observation.player_id,
            turn_index=observation.turn_index,
            phase=observation.phase,
            decision_index=observation.decision_index,
            model=self.model,
            temperature=self.temperature,
            attempts=tuple(attempts),
        )

    @staticmethod
    def _json_safe_message(message: dict[str, object]) -> dict[str, JsonValue]:
        return LLMPlayer._json_safe_object(message)

    @staticmethod
    def _json_safe_object(payload: object) -> dict[str, JsonValue]:
        safe_payload = json.loads(json.dumps(payload, sort_keys=True))
        if not isinstance(safe_payload, dict):
            raise RuntimeError("Expected a JSON object while recording prompt traces.")
        return safe_payload

    @staticmethod
    def _extract_response_json(completion: dict[str, object]) -> dict[str, object]:
        choices = completion.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LLM completion did not include any choices.")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("LLM completion choice payload had an unexpected shape.")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("LLM completion did not include a message payload.")
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("LLM completion message content must be a string.")
        parsed = json.loads(LLMPlayer._strip_markdown_fences(content))
        if not isinstance(parsed, dict):
            raise RuntimeError("LLM response JSON must decode to an object.")
        return parsed

    @staticmethod
    def _strip_markdown_fences(content: str) -> str:
        stripped = content.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1] == "```":
            return "\n".join(lines[1:-1]).strip()
        return stripped
