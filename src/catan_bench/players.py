from __future__ import annotations

import json
import math
import random
from collections import deque
from typing import Iterable, Protocol

from .llm import LLMClient, LLMRequestTooLargeError
from .schemas import (
    Action,
    Event,
    JsonValue,
    MemoryResponse,
    Observation,
    PlayerResponse,
    PromptTrace,
    PromptTraceAttempt,
    RecallObservation,
    ReflectionObservation,
    TradeChatObservation,
    TradeChatOpenResponse,
    TradeChatReplyResponse,
    TradeChatSelectionResponse,
)


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

    def __init__(self, *, record_observations: bool = False) -> None:
        self.record_observations = record_observations
        self.observations: list[Observation] = []

    def respond(self, observation: Observation) -> PlayerResponse:
        if self.record_observations:
            self.observations.append(observation)
        for action in observation.legal_actions:
            if not _is_trade_template(action):
                return PlayerResponse(action=action)

        return PlayerResponse(action=_materialize_default_trade_offer())


class RandomLegalPlayer:
    """Random baseline that samples from the current legal concrete actions."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        record_observations: bool = False,
    ) -> None:
        self._rng = random.Random(seed)
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

        return PlayerResponse(action=_materialize_default_trade_offer())


class LLMPlayer:
    """LLM-backed player with phased recall, action selection, and reflection."""

    def __init__(
        self,
        *,
        client: LLMClient,
        model: str,
        temperature: float = 0.2,
        top_p: float | None = None,
        reasoning_enabled: bool | None = None,
        prompt_history_limit: int | None = 12,
        prompt_memory_limit: int | None = 8,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_enabled = reasoning_enabled
        self.prompt_history_limit = prompt_history_limit
        self.prompt_memory_limit = prompt_memory_limit
        self.observations: list[Observation] = []
        self._prompt_traces: deque[PromptTrace] = deque()

    def recall(self, observation: RecallObservation) -> MemoryResponse:
        trace_attempts: list[PromptTraceAttempt] = []
        messages = self._messages_for_recall(observation)
        try:
            response_payload = self._complete_and_trace(
                messages=messages,
                trace_attempts=trace_attempts,
            )
        except LLMRequestTooLargeError as exc:
            self._record_failed_attempt(
                messages=messages,
                trace_attempts=trace_attempts,
                error_type="request_too_large",
                error_message=str(exc),
            )
            messages = self._messages_for_recall(
                observation,
                event_limit=self._compact_limit(self.prompt_history_limit, fallback=6),
                compact=True,
            )
            response_payload = self._complete_and_trace(
                messages=messages,
                trace_attempts=trace_attempts,
            )

        response = MemoryResponse(memory=response_payload.get("private_memory"))
        self._prompt_traces.append(
            self._prompt_trace_for(
                player_id=observation.player_id,
                turn_index=observation.turn_index,
                phase=observation.phase,
                decision_index=observation.decision_index,
                stage="recall",
                attempts=trace_attempts,
            )
        )
        return response

    def respond(self, observation: Observation) -> PlayerResponse:
        self.observations.append(observation)
        trace_attempts: list[PromptTraceAttempt] = []
        messages = self._messages_for_action(observation)
        try:
            response_payload = self._complete_and_trace(
                messages=messages,
                trace_attempts=trace_attempts,
            )
        except LLMRequestTooLargeError as exc:
            self._record_failed_attempt(
                messages=messages,
                trace_attempts=trace_attempts,
                error_type="request_too_large",
                error_message=str(exc),
            )
            messages = self._messages_for_action(
                observation,
                compact=True,
            )
            response_payload = self._complete_and_trace(
                messages=messages,
                trace_attempts=trace_attempts,
            )

        try:
            response = self._response_from_payload(observation, response_payload)
        except RuntimeError as exc:
            response = None
            repair_reason = str(exc)
        else:
            if self._is_legal_response_action(response.action, observation.legal_actions):
                self._prompt_traces.append(
                    self._prompt_trace_for(
                        player_id=observation.player_id,
                        turn_index=observation.turn_index,
                        phase=observation.phase,
                        decision_index=observation.decision_index,
                        stage="act",
                        attempts=trace_attempts,
                    )
                )
                return response
            repair_reason = "Your previous response selected an illegal action for the current decision."

        repaired_messages = messages + self._repair_messages(
            observation=observation,
            attempted_response=response_payload,
            error_message=repair_reason,
        )
        repaired_payload = self._complete_and_trace(
            messages=repaired_messages,
            trace_attempts=trace_attempts,
        )
        try:
            repaired_response = self._response_from_payload(observation, repaired_payload)
        except RuntimeError:
            repaired_response = PlayerResponse(
                action=self._fallback_legal_action(observation),
                reasoning=self._coerce_reasoning(repaired_payload),
            )
        self._prompt_traces.append(
            self._prompt_trace_for(
                player_id=observation.player_id,
                turn_index=observation.turn_index,
                phase=observation.phase,
                decision_index=observation.decision_index,
                stage="act",
                attempts=trace_attempts,
            )
        )
        if self._is_legal_response_action(repaired_response.action, observation.legal_actions):
            return repaired_response

        return PlayerResponse(
            action=self._fallback_legal_action(observation),
            reasoning=repaired_response.reasoning,
        )

    def reflect(self, observation: ReflectionObservation) -> MemoryResponse:
        trace_attempts: list[PromptTraceAttempt] = []
        messages = self._messages_for_reflection(observation)
        try:
            response_payload = self._complete_and_trace(
                messages=messages,
                trace_attempts=trace_attempts,
            )
        except LLMRequestTooLargeError as exc:
            self._record_failed_attempt(
                messages=messages,
                trace_attempts=trace_attempts,
                error_type="request_too_large",
                error_message=str(exc),
            )
            messages = self._messages_for_reflection(
                observation,
                event_limit=self._compact_limit(self.prompt_history_limit, fallback=6),
                compact=True,
            )
            response_payload = self._complete_and_trace(
                messages=messages,
                trace_attempts=trace_attempts,
            )

        response = MemoryResponse(memory=response_payload.get("private_memory"))
        self._prompt_traces.append(
            self._prompt_trace_for(
                player_id=observation.player_id,
                turn_index=observation.turn_index,
                phase=observation.phase,
                decision_index=observation.decision_index,
                stage="reflect",
                attempts=trace_attempts,
            )
        )
        return response

    def open_trade_chat(self, observation: TradeChatObservation) -> TradeChatOpenResponse:
        trace_attempts: list[PromptTraceAttempt] = []
        messages = self._messages_for_trade_chat_open(observation)
        response_payload = self._complete_and_trace(
            messages=messages,
            trace_attempts=trace_attempts,
        )
        response = self._trade_chat_open_response_from_payload(response_payload)
        self._prompt_traces.append(
            self._prompt_trace_for(
                player_id=observation.player_id,
                turn_index=observation.turn_index,
                phase=observation.phase,
                decision_index=observation.decision_index,
                stage="trade_chat_open",
                attempts=trace_attempts,
            )
        )
        return response

    def respond_trade_chat(self, observation: TradeChatObservation) -> TradeChatReplyResponse:
        trace_attempts: list[PromptTraceAttempt] = []
        messages = self._messages_for_trade_chat_reply(observation)
        response_payload = self._complete_and_trace(
            messages=messages,
            trace_attempts=trace_attempts,
        )
        response = self._trade_chat_reply_response_from_payload(response_payload)
        self._prompt_traces.append(
            self._prompt_trace_for(
                player_id=observation.player_id,
                turn_index=observation.turn_index,
                phase=observation.phase,
                decision_index=observation.decision_index,
                stage="trade_chat_reply",
                attempts=trace_attempts,
            )
        )
        return response

    def select_trade_chat_offer(
        self, observation: TradeChatObservation
    ) -> TradeChatSelectionResponse:
        trace_attempts: list[PromptTraceAttempt] = []
        messages = self._messages_for_trade_chat_select(observation)
        response_payload = self._complete_and_trace(
            messages=messages,
            trace_attempts=trace_attempts,
        )
        response = self._trade_chat_selection_response_from_payload(response_payload)
        self._prompt_traces.append(
            self._prompt_trace_for(
                player_id=observation.player_id,
                turn_index=observation.turn_index,
                phase=observation.phase,
                decision_index=observation.decision_index,
                stage="trade_chat_select",
                attempts=trace_attempts,
            )
        )
        return response

    def take_last_prompt_trace(self) -> PromptTrace | None:
        if not self._prompt_traces:
            return None
        return self._prompt_traces.pop()

    def take_prompt_traces(self) -> tuple[PromptTrace, ...]:
        traces = tuple(self._prompt_traces)
        self._prompt_traces.clear()
        return traces

    def _messages_for_recall(
        self,
        observation: RecallObservation,
        *,
        event_limit: int | None = None,
        compact: bool = False,
    ) -> list[dict[str, object]]:
        system_prompt = "\n\n".join(
            part
            for part in (
                observation.game_rules,
                (
                    "Phase 1 - Recall. Rewrite your entire private memory as one consolidated "
                    "JSON document based only on the newly observed events and your prior "
                    "memory. Return strict JSON with a single key `private_memory`."
                ),
            )
            if part
        )
        public_events = self._tail_events(
            observation.public_events_since_last_turn,
            event_limit,
        )
        private_events = self._tail_events(
            observation.private_events_since_last_turn,
            event_limit,
        )
        user_payload = {
            "player_id": observation.player_id,
            "turn_index": observation.turn_index,
            "phase": observation.phase,
            "decision_index": observation.decision_index,
            "current_memory": self._memory_content(observation.memory),
            "public_events_since_last_turn": [event.to_dict() for event in public_events],
            "private_events_since_last_turn": [event.to_dict() for event in private_events],
            "context_window": {
                "compact_retry": compact,
                "public_events_available": len(observation.public_events_since_last_turn),
                "public_events_included": len(public_events),
                "private_events_available": len(observation.private_events_since_last_turn),
                "private_events_included": len(private_events),
            },
            "response_contract": {
                "private_memory": (
                    "the full consolidated private memory to carry into the action phase; "
                    "return null if there is nothing worth keeping"
                ),
            },
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
        ]

    def _messages_for_action(
        self,
        observation: Observation,
        *,
        compact: bool = False,
    ) -> list[dict[str, object]]:
        system_prompt = "\n\n".join(
            part
            for part in (
                observation.game_rules,
                (
                    "Phase 2 - Act. Return strict JSON with keys `action_index` and "
                    "`private_reasoning`. `action_index` should be the integer index of one "
                    "entry from the provided `legal_actions` list and is the preferred way to "
                    "choose an action. You may also include `action` with `action_type` and "
                    "`payload`, but use the exact legal action when you do. If a legal action is "
                    "marked as requiring a concrete payload, do not choose it by `action_index`; "
                    "instead return an `action` object with a concrete payload. "
                    "`private_reasoning` should be a concise private summary under 30 words."
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
            "private_memory": self._memory_content(observation.memory),
            "context_window": {
                "compact_retry": compact,
            },
            "recent_public_events": [event.to_dict() for event in observation.recent_public_events],
            "recent_private_events": [event.to_dict() for event in observation.recent_private_events],
            "legal_actions": indexed_legal_actions,
            "response_contract": {
                "action_index": "preferred integer index into legal_actions",
                "action": (
                    "optional exact copy of the chosen legal action object; required for any "
                    "action that needs a concrete payload"
                ),
                "private_reasoning": (
                    "required concise private explanation under 30 words; kept in your private "
                    "history"
                ),
            },
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
        ]

    def _messages_for_trade_chat_open(
        self, observation: TradeChatObservation
    ) -> list[dict[str, object]]:
        system_prompt = "\n\n".join(
            part
            for part in (
                observation.game_rules,
                (
                    "Public negotiation stage. You are the active player and may choose whether "
                    "to open a public trade room for one bilateral trade attempt. Return strict "
                    "JSON with keys `open_chat`, `message`, `requested_resources`, and optional "
                    "`private_reasoning`. If you do not want to trade now, return "
                    "`{\"open_chat\": false}`. If you open the room, keep `message` short and "
                    "public, and set `requested_resources` to the exact resources you want to "
                    "receive from another player."
                ),
            )
            if part
        )
        user_payload = {
            "player_id": observation.player_id,
            "owner_player_id": observation.owner_player_id,
            "turn_index": observation.turn_index,
            "phase": observation.phase,
            "decision_index": observation.decision_index,
            "attempt_index": observation.attempt_index,
            "public_state": observation.public_state,
            "private_state": observation.private_state,
            "private_memory": self._memory_content(observation.memory),
            "other_player_ids": list(observation.other_player_ids),
            "transcript": [event.to_dict() for event in observation.transcript],
            "message_char_limit": observation.message_char_limit,
            "response_contract": {
                "open_chat": "boolean",
                "message": "required short public request if open_chat is true",
                "requested_resources": "required resource-count map if open_chat is true",
                "private_reasoning": "optional concise private summary under 30 words",
            },
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
        ]

    def _messages_for_trade_chat_reply(
        self, observation: TradeChatObservation
    ) -> list[dict[str, object]]:
        system_prompt = "\n\n".join(
            part
            for part in (
                observation.game_rules,
                (
                    "Public negotiation stage. The owner has requested a bilateral trade. "
                    "Return strict JSON with keys `message`, `owner_gives`, `owner_gets`, and "
                    "optional `private_reasoning`. To decline, return empty resource maps and an "
                    "optional short message. Any non-empty quote must be bilateral with the "
                    "owner only, where `owner_gives` are resources the owner pays and "
                    "`owner_gets` are resources you give the owner."
                ),
            )
            if part
        )
        user_payload = {
            "player_id": observation.player_id,
            "owner_player_id": observation.owner_player_id,
            "turn_index": observation.turn_index,
            "phase": observation.phase,
            "decision_index": observation.decision_index,
            "attempt_index": observation.attempt_index,
            "public_state": observation.public_state,
            "private_state": observation.private_state,
            "private_memory": self._memory_content(observation.memory),
            "requested_resources": observation.requested_resources,
            "transcript": [event.to_dict() for event in observation.transcript],
            "message_char_limit": observation.message_char_limit,
            "response_contract": {
                "message": "optional short public quote or decline",
                "owner_gives": "resource-count map the owner would pay",
                "owner_gets": "resource-count map the owner would receive",
                "private_reasoning": "optional concise private summary under 30 words",
            },
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
        ]

    def _messages_for_trade_chat_select(
        self, observation: TradeChatObservation
    ) -> list[dict[str, object]]:
        system_prompt = "\n\n".join(
            part
            for part in (
                observation.game_rules,
                (
                    "Public negotiation stage. Review the quotes and select exactly one player "
                    "to trade with, or choose no deal. Return strict JSON with keys "
                    "`selected_player_id`, `message`, and optional `private_reasoning`. Use "
                    "null for `selected_player_id` if no quote is acceptable."
                ),
            )
            if part
        )
        user_payload = {
            "player_id": observation.player_id,
            "owner_player_id": observation.owner_player_id,
            "turn_index": observation.turn_index,
            "phase": observation.phase,
            "decision_index": observation.decision_index,
            "attempt_index": observation.attempt_index,
            "public_state": observation.public_state,
            "private_state": observation.private_state,
            "private_memory": self._memory_content(observation.memory),
            "requested_resources": observation.requested_resources,
            "quotes": [quote.to_dict() for quote in observation.quotes],
            "transcript": [event.to_dict() for event in observation.transcript],
            "message_char_limit": observation.message_char_limit,
            "response_contract": {
                "selected_player_id": "player id from the provided quotes, or null for no deal",
                "message": "optional short public selection message",
                "private_reasoning": "optional concise private summary under 30 words",
            },
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
        ]

    def _messages_for_reflection(
        self,
        observation: ReflectionObservation,
        *,
        event_limit: int | None = None,
        compact: bool = False,
    ) -> list[dict[str, object]]:
        system_prompt = "\n\n".join(
            part
            for part in (
                observation.game_rules,
                (
                    "Phase 3 - Reflect. Rewrite your entire private memory as one consolidated "
                    "JSON document using the starting memory and what happened during this turn. "
                    "Return strict JSON with a single key `private_memory`."
                ),
            )
            if part
        )
        public_events = self._tail_events(observation.public_events_this_turn, event_limit)
        private_events = self._tail_events(observation.private_events_this_turn, event_limit)
        user_payload = {
            "player_id": observation.player_id,
            "turn_index": observation.turn_index,
            "phase": observation.phase,
            "decision_index": observation.decision_index,
            "starting_memory": self._memory_content(observation.memory),
            "public_events_this_turn": [event.to_dict() for event in public_events],
            "private_events_this_turn": [event.to_dict() for event in private_events],
            "context_window": {
                "compact_retry": compact,
                "public_events_available": len(observation.public_events_this_turn),
                "public_events_included": len(public_events),
                "private_events_available": len(observation.private_events_this_turn),
                "private_events_included": len(private_events),
            },
            "response_contract": {
                "private_memory": (
                    "the final consolidated private memory to carry into the next turn; "
                    "return null if there is nothing worth keeping"
                ),
            },
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
        ]

    @staticmethod
    def _tail_events(items: tuple[Event, ...], limit: int | None) -> tuple[Event, ...]:
        if limit is None:
            return items
        if limit <= 0:
            return ()
        return items[-limit:]

    @staticmethod
    def _compact_limit(limit: int | None, *, fallback: int) -> int:
        if limit is None:
            return fallback
        return max(1, min(limit, max(1, limit // 2)))

    @staticmethod
    def _memory_content(memory) -> JsonValue | None:
        if memory is None:
            return None
        if isinstance(memory, tuple):
            if not memory:
                return None
            return memory[-1].content
        return memory.content

    def _response_from_payload(
        self, observation: Observation, response_payload: dict[str, object]
    ) -> PlayerResponse:
        action = self._action_from_payload(observation, response_payload)
        reasoning = self._coerce_reasoning(response_payload)
        return PlayerResponse(action=action, reasoning=reasoning)

    def _trade_chat_open_response_from_payload(
        self, response_payload: dict[str, object]
    ) -> TradeChatOpenResponse:
        open_chat = bool(response_payload.get("open_chat", False))
        message = self._coerce_public_message(response_payload.get("message"))
        requested_resources = self._coerce_resource_map(
            response_payload.get("requested_resources")
            or response_payload.get("request")
            or response_payload.get("owner_gets")
        )
        if not open_chat or not requested_resources:
            return TradeChatOpenResponse(open_chat=False, reasoning=self._coerce_reasoning(response_payload))
        return TradeChatOpenResponse(
            open_chat=True,
            message=message,
            requested_resources=requested_resources,
            reasoning=self._coerce_reasoning(response_payload),
        )

    def _trade_chat_reply_response_from_payload(
        self, response_payload: dict[str, object]
    ) -> TradeChatReplyResponse:
        return TradeChatReplyResponse(
            message=self._coerce_public_message(response_payload.get("message")),
            owner_gives=self._coerce_resource_map(
                response_payload.get("owner_gives")
                or response_payload.get("offer")
                or response_payload.get("give")
            ),
            owner_gets=self._coerce_resource_map(
                response_payload.get("owner_gets")
                or response_payload.get("request")
                or response_payload.get("want")
            ),
            reasoning=self._coerce_reasoning(response_payload),
        )

    def _trade_chat_selection_response_from_payload(
        self, response_payload: dict[str, object]
    ) -> TradeChatSelectionResponse:
        selected_player_id = response_payload.get("selected_player_id")
        if not isinstance(selected_player_id, str):
            selected_player_id = response_payload.get("player_id")
        if isinstance(selected_player_id, str):
            selected_player_id = selected_player_id.upper()
        else:
            selected_player_id = None
        return TradeChatSelectionResponse(
            selected_player_id=selected_player_id,
            message=self._coerce_public_message(response_payload.get("message")),
            reasoning=self._coerce_reasoning(response_payload),
        )

    @staticmethod
    def _action_from_payload(
        observation: Observation, response_payload: dict[str, object]
    ) -> Action:
        action_index = None
        for key in (
            "action_index",
            "chosen_action_index",
            "selected_action_index",
            "legal_action_index",
            "index",
        ):
            candidate = response_payload.get(key)
            if candidate is not None:
                action_index = candidate
                break
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
        if isinstance(action_payload, int):
            if not 0 <= action_payload < len(observation.legal_actions):
                raise RuntimeError(
                    f"LLM action index {action_payload} is out of range for legal actions."
                )
            return observation.legal_actions[action_payload]
        if isinstance(action_payload, str) and action_payload.isdigit():
            indexed_action = int(action_payload)
            if not 0 <= indexed_action < len(observation.legal_actions):
                raise RuntimeError(
                    f"LLM action index {indexed_action} is out of range for legal actions."
                )
            return observation.legal_actions[indexed_action]
        if not isinstance(action_payload, dict):
            for key in ("chosen_action", "selected_action", "move"):
                candidate = response_payload.get(key)
                if isinstance(candidate, dict):
                    action_payload = candidate
                    break
        if not isinstance(action_payload, dict) and isinstance(
            response_payload.get("action_type"), str
        ):
            action_payload = {
                "action_type": response_payload["action_type"],
                "payload": response_payload.get("payload", {}),
            }
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
    def _coerce_reasoning(response_payload: dict[str, object]) -> str | None:
        for key in ("private_reasoning", "reasoning", "thought", "summary"):
            reasoning = response_payload.get(key)
            if reasoning is None:
                continue
            if not isinstance(reasoning, str):
                raise RuntimeError(
                    f"LLM response field `{key}` must be a string when present."
                )
            return reasoning
        return None

    @staticmethod
    def _coerce_public_message(value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise RuntimeError("Public chat message fields must be strings when present.")
        stripped = value.strip()
        return stripped or None

    @staticmethod
    def _coerce_resource_map(value: object) -> dict[str, JsonValue]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise RuntimeError("Resource maps must decode to JSON objects.")
        result: dict[str, JsonValue] = {}
        for resource, amount in value.items():
            if not isinstance(resource, str):
                raise RuntimeError("Resource map keys must be strings.")
            if not isinstance(amount, int):
                raise RuntimeError("Resource map values must be integers.")
            if amount <= 0:
                continue
            result[resource] = amount
        return result

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
        error_message: str,
    ) -> list[dict[str, object]]:
        repair_payload = {
            "error": error_message,
            "previous_response": attempted_response,
            "instruction": (
                "Choose one legal action by returning only `action_index` and "
                "`private_reasoning`. The action_index must be one of the provided legal "
                "action indexes."
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
            top_p=self.top_p,
            reasoning_enabled=self.reasoning_enabled,
        )
        response_payload = self._extract_response_json(completion)
        trace_attempts.append(
            PromptTraceAttempt(
                messages=tuple(self._json_safe_message(message) for message in messages),
                response=self._json_safe_object(response_payload),
            )
        )
        return response_payload

    def _record_failed_attempt(
        self,
        *,
        messages: list[dict[str, object]],
        trace_attempts: list[PromptTraceAttempt],
        error_type: str,
        error_message: str,
    ) -> None:
        trace_attempts.append(
            PromptTraceAttempt(
                messages=tuple(self._json_safe_message(message) for message in messages),
                response=self._json_safe_object(
                    {
                        "error": {
                            "type": error_type,
                            "message": error_message,
                        }
                    }
                ),
            )
        )

    def _prompt_trace_for(
        self,
        *,
        player_id: str,
        turn_index: int,
        phase: str,
        decision_index: int,
        stage: str,
        attempts: list[PromptTraceAttempt],
    ) -> PromptTrace:
        return PromptTrace(
            player_id=player_id,
            turn_index=turn_index,
            phase=phase,
            decision_index=decision_index,
            stage=stage,
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
        stripped_content = LLMPlayer._strip_markdown_fences(content)
        if not stripped_content:
            reasoning = message.get("reasoning")
            finish_reason = first_choice.get("finish_reason")
            if isinstance(reasoning, str) and reasoning.strip():
                raise RuntimeError(
                    "LLM completion did not include JSON content. The provider returned "
                    f"reasoning-only output with finish_reason={finish_reason!r}."
                )
            raise RuntimeError("LLM completion message content was empty.")
        parsed = json.loads(stripped_content)
        if not isinstance(parsed, dict):
            raise RuntimeError("LLM response JSON must decode to an object.")
        return parsed

    @staticmethod
    def _strip_markdown_fences(content: str) -> str:
        stripped = content.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if not lines:
            return stripped
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
