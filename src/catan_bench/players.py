from __future__ import annotations

import json
import math
import random
import re
from collections import deque
from typing import Callable, Iterable, Protocol

from .llm import LLMClient, LLMRequestTooLargeError
from .prompting import PromptRenderer
from .schemas import (
    Action,
    ActionDecision,
    ActionObservation,
    Event,
    JsonValue,
    OpeningStrategyObservation,
    OpeningStrategyResponse,
    PostGameChatObservation,
    PostGameChatResponse,
    PublicChatDraft,
    PromptTrace,
    PromptTraceAttempt,
    ReactiveObservation,
    TradeChatObservation,
    TradeChatOpenResponse,
    TradeChatOwnerDecisionResponse,
    TradeChatReplyResponse,
    TurnEndObservation,
    TurnEndResponse,
    TurnStartObservation,
    TurnStartResponse,
)

_TRADE_CHAT_ROOM_EVENT_KINDS = frozenset(
    {
        "trade_chat_opened",
        "trade_chat_message",
        "trade_chat_proposal_rejected",
        "trade_chat_quote_selected",
        "trade_chat_no_deal",
        "trade_chat_closed",
    }
)


class Player(Protocol):
    def plan_opening_strategy(
        self, observation: OpeningStrategyObservation
    ) -> OpeningStrategyResponse: ...

    def start_turn(self, observation: TurnStartObservation) -> TurnStartResponse: ...

    def choose_action(self, observation: ActionObservation) -> ActionDecision: ...

    def end_turn(self, observation: TurnEndObservation) -> TurnEndResponse: ...

    def respond_reactive(self, observation: ReactiveObservation) -> ActionDecision: ...

    def post_game_chat(
        self, observation: PostGameChatObservation
    ) -> PostGameChatResponse: ...


def _is_resource_swap_template(action: Action) -> bool:
    return action.action_type in {
        "OFFER_TRADE",
        "COUNTER_OFFER",
    } and action.payload == {
        "offer": {},
        "request": {},
    }


def _materialize_default_trade_offer() -> Action:
    return Action(
        "OFFER_TRADE",
        payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
    )


def _normalize_offer_trade_payload(payload: dict) -> dict:
    """Accept LLM field-name variants for trade-like actions and canonicalize to offer/request."""
    result = dict(payload)
    if "offer" not in result and "give" in result:
        result["offer"] = result.pop("give")
    if "request" not in result and "take" in result:
        result["request"] = result.pop("take")
    if "request" not in result and "want" in result:
        result["request"] = result.pop("want")
    if "offer" in result and not isinstance(result["offer"], dict):
        result["offer"] = {}
    if "request" in result and not isinstance(result["request"], dict):
        result["request"] = {}
    # Drop extraneous keys the LLM sometimes adds (e.g. "player").
    return {k: v for k, v in result.items() if k in {"offer", "request"}}


def _coerce_trade_chat_selected_proposal_id(
    observation: TradeChatObservation,
    selected_proposal_id: object,
) -> str | None:
    if isinstance(selected_proposal_id, str):
        for proposal in observation.proposals:
            if proposal.proposal_id == selected_proposal_id:
                return selected_proposal_id

        hint_tokens = {
            token
            for token in re.split(r"[^A-Z0-9]+", selected_proposal_id.upper())
            if token
        }
        matching_players = {
            proposal.player_id
            for proposal in observation.proposals
            if proposal.player_id.upper() in hint_tokens
        }
        if len(matching_players) == 1:
            matching_player_id = next(iter(matching_players))
            for proposal in reversed(observation.proposals):
                if proposal.player_id == matching_player_id:
                    return proposal.proposal_id

    if len(observation.proposals) == 1:
        return observation.proposals[0].proposal_id
    return None


class ScriptedPlayer:
    """Small utility adapter for tests and scripted demos."""

    def __init__(
        self,
        *,
        opening_strategy_responses: Iterable[
            OpeningStrategyResponse | JsonValue | None
        ] = (),
        start_turn_responses: Iterable[TurnStartResponse | JsonValue | None] = (),
        action_responses: Iterable[ActionDecision | Action] = (),
        end_turn_responses: Iterable[TurnEndResponse | JsonValue | None] = (),
        post_game_chat_responses: Iterable[
            PostGameChatResponse | PublicChatDraft | None
        ] = (),
        reactive_responses: Iterable[ActionDecision | Action] = (),
        trade_chat_open_responses: Iterable[TradeChatOpenResponse] = (),
        trade_chat_reply_responses: Iterable[TradeChatReplyResponse] = (),
        trade_chat_owner_decision_responses: Iterable[
            TradeChatOwnerDecisionResponse
        ] = (),
    ) -> None:
        self._opening_strategy_responses = deque(opening_strategy_responses)
        self._start_turn_responses = deque(start_turn_responses)
        self._action_responses = deque(action_responses)
        self._end_turn_responses = deque(end_turn_responses)
        self._post_game_chat_responses = deque(post_game_chat_responses)
        self._reactive_responses = deque(reactive_responses)
        self._trade_chat_open_responses = deque(trade_chat_open_responses)
        self._trade_chat_reply_responses = deque(trade_chat_reply_responses)
        self._trade_chat_owner_decision_responses = deque(
            trade_chat_owner_decision_responses
        )
        self.opening_strategy_observations: list[OpeningStrategyObservation] = []
        self.start_turn_observations: list[TurnStartObservation] = []
        self.action_observations: list[ActionObservation] = []
        self.end_turn_observations: list[TurnEndObservation] = []
        self.post_game_chat_observations: list[PostGameChatObservation] = []
        self.reactive_observations: list[ReactiveObservation] = []
        self.trade_chat_observations: list[TradeChatObservation] = []

    def plan_opening_strategy(
        self, observation: OpeningStrategyObservation
    ) -> OpeningStrategyResponse:
        self.opening_strategy_observations.append(observation)
        if not self._opening_strategy_responses:
            return OpeningStrategyResponse(long_term=observation.memory.long_term)
        response = self._opening_strategy_responses.popleft()
        if isinstance(response, OpeningStrategyResponse):
            return response
        return OpeningStrategyResponse(long_term=response)

    def start_turn(self, observation: TurnStartObservation) -> TurnStartResponse:
        self.start_turn_observations.append(observation)
        if not self._start_turn_responses:
            return TurnStartResponse(short_term=None)
        response = self._start_turn_responses.popleft()
        if isinstance(response, TurnStartResponse):
            return response
        return TurnStartResponse(short_term=response)

    def choose_action(self, observation: ActionObservation) -> ActionDecision:
        self.action_observations.append(observation)
        if not self._action_responses:
            raise RuntimeError("ScriptedPlayer ran out of action responses.")
        response = self._action_responses.popleft()
        if isinstance(response, ActionDecision):
            return response
        return ActionDecision(action=response, short_term=None)

    def end_turn(self, observation: TurnEndObservation) -> TurnEndResponse:
        self.end_turn_observations.append(observation)
        if not self._end_turn_responses:
            return TurnEndResponse(long_term=observation.memory.long_term)
        response = self._end_turn_responses.popleft()
        if isinstance(response, TurnEndResponse):
            return response
        return TurnEndResponse(long_term=response)

    def respond_reactive(self, observation: ReactiveObservation) -> ActionDecision:
        self.reactive_observations.append(observation)
        if not self._reactive_responses:
            raise RuntimeError("ScriptedPlayer ran out of reactive responses.")
        response = self._reactive_responses.popleft()
        if isinstance(response, ActionDecision):
            return response
        return ActionDecision(action=response, short_term=None)

    def post_game_chat(
        self, observation: PostGameChatObservation
    ) -> PostGameChatResponse:
        self.post_game_chat_observations.append(observation)
        if not self._post_game_chat_responses:
            return PostGameChatResponse()
        response = self._post_game_chat_responses.popleft()
        if isinstance(response, PostGameChatResponse):
            return response
        return PostGameChatResponse(public_chat=response)

    def open_trade_chat(
        self, observation: TradeChatObservation
    ) -> TradeChatOpenResponse:
        self.trade_chat_observations.append(observation)
        if not self._trade_chat_open_responses:
            return TradeChatOpenResponse()
        return self._trade_chat_open_responses.popleft()

    def respond_trade_chat(
        self, observation: TradeChatObservation
    ) -> TradeChatReplyResponse:
        self.trade_chat_observations.append(observation)
        if not self._trade_chat_reply_responses:
            return TradeChatReplyResponse()
        return self._trade_chat_reply_responses.popleft()

    def decide_trade_chat(
        self, observation: TradeChatObservation
    ) -> TradeChatOwnerDecisionResponse:
        self.trade_chat_observations.append(observation)
        if not self._trade_chat_owner_decision_responses:
            return TradeChatOwnerDecisionResponse()
        return self._trade_chat_owner_decision_responses.popleft()

    def select_trade_chat_offer(
        self, observation: TradeChatObservation
    ) -> TradeChatOwnerDecisionResponse:
        return self.decide_trade_chat(observation)


class FirstLegalPlayer:
    """Simple deterministic baseline that always picks the first legal action."""

    def __init__(self, *, record_observations: bool = False) -> None:
        self.record_observations = record_observations
        self.opening_strategy_observations: list[OpeningStrategyObservation] = []
        self.start_turn_observations: list[TurnStartObservation] = []
        self.action_observations: list[ActionObservation] = []
        self.end_turn_observations: list[TurnEndObservation] = []
        self.post_game_chat_observations: list[PostGameChatObservation] = []
        self.reactive_observations: list[ReactiveObservation] = []

    def plan_opening_strategy(
        self, observation: OpeningStrategyObservation
    ) -> OpeningStrategyResponse:
        if self.record_observations:
            self.opening_strategy_observations.append(observation)
        return OpeningStrategyResponse(long_term=observation.memory.long_term)

    def start_turn(self, observation: TurnStartObservation) -> TurnStartResponse:
        if self.record_observations:
            self.start_turn_observations.append(observation)
        return TurnStartResponse(short_term=None)

    def choose_action(self, observation: ActionObservation) -> ActionDecision:
        if self.record_observations:
            self.action_observations.append(observation)
        return ActionDecision(
            action=self._first_legal_action(observation.legal_actions),
            short_term=None,
        )

    def end_turn(self, observation: TurnEndObservation) -> TurnEndResponse:
        if self.record_observations:
            self.end_turn_observations.append(observation)
        return TurnEndResponse(long_term=observation.memory.long_term)

    def respond_reactive(self, observation: ReactiveObservation) -> ActionDecision:
        if self.record_observations:
            self.reactive_observations.append(observation)
        return ActionDecision(
            action=self._first_legal_action(observation.legal_actions),
            short_term=None,
        )

    def post_game_chat(
        self, observation: PostGameChatObservation
    ) -> PostGameChatResponse:
        if self.record_observations:
            self.post_game_chat_observations.append(observation)
        return PostGameChatResponse()

    @staticmethod
    def _first_legal_action(legal_actions: tuple[Action, ...]) -> Action:
        for action in legal_actions:
            if not _is_resource_swap_template(action):
                return action
        return _materialize_default_trade_offer()


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
        self.opening_strategy_observations: list[OpeningStrategyObservation] = []
        self.start_turn_observations: list[TurnStartObservation] = []
        self.action_observations: list[ActionObservation] = []
        self.end_turn_observations: list[TurnEndObservation] = []
        self.post_game_chat_observations: list[PostGameChatObservation] = []
        self.reactive_observations: list[ReactiveObservation] = []

    def plan_opening_strategy(
        self, observation: OpeningStrategyObservation
    ) -> OpeningStrategyResponse:
        if self.record_observations:
            self.opening_strategy_observations.append(observation)
        return OpeningStrategyResponse(long_term=observation.memory.long_term)

    def start_turn(self, observation: TurnStartObservation) -> TurnStartResponse:
        if self.record_observations:
            self.start_turn_observations.append(observation)
        return TurnStartResponse(short_term=None)

    def choose_action(self, observation: ActionObservation) -> ActionDecision:
        if self.record_observations:
            self.action_observations.append(observation)
        return ActionDecision(
            action=self._sample_action(observation.legal_actions),
            short_term=None,
        )

    def end_turn(self, observation: TurnEndObservation) -> TurnEndResponse:
        if self.record_observations:
            self.end_turn_observations.append(observation)
        return TurnEndResponse(long_term=observation.memory.long_term)

    def respond_reactive(self, observation: ReactiveObservation) -> ActionDecision:
        if self.record_observations:
            self.reactive_observations.append(observation)
        return ActionDecision(
            action=self._sample_action(observation.legal_actions),
            short_term=None,
        )

    def post_game_chat(
        self, observation: PostGameChatObservation
    ) -> PostGameChatResponse:
        if self.record_observations:
            self.post_game_chat_observations.append(observation)
        return PostGameChatResponse()

    def _sample_action(self, legal_actions: tuple[Action, ...]) -> Action:
        concrete_actions = [
            action for action in legal_actions if not _is_resource_swap_template(action)
        ]
        if concrete_actions:
            return self._rng.choice(concrete_actions)
        return _materialize_default_trade_offer()


class LLMPlayer:
    """LLM-backed player with explicit start-turn, act, reactive, and end-turn stages."""

    def __init__(
        self,
        *,
        client: LLMClient,
        model: str,
        temperature: float = 0.2,
        top_p: float | None = None,
        reasoning_enabled: bool | None = None,
        reasoning_effort: str | None = None,
        prompt_history_limit: int | None = 12,
        invalid_response_retries: int = 1,
        renderer: PromptRenderer | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_effort = reasoning_effort
        self.prompt_history_limit = prompt_history_limit
        self.invalid_response_retries = max(0, invalid_response_retries)
        self.renderer = renderer or PromptRenderer()
        self._prompt_traces: deque[PromptTrace] = deque()

    def _traced_llm_call(
        self,
        *,
        observation,
        stage: str,
        build_messages: Callable,
        compact_retry: bool = True,
        salvage_on_error: bool = False,
    ) -> tuple[dict[str, object], list[dict[str, object]], list[PromptTraceAttempt]]:
        trace_attempts: list[PromptTraceAttempt] = []
        messages = build_messages(observation)
        response_payload: dict[str, object] = {}
        try:
            response_payload = self._complete_and_trace(
                messages=messages,
                trace_attempts=trace_attempts,
                player_id=observation.player_id,
                stage=stage,
            )
        except LLMRequestTooLargeError as exc:
            if not compact_retry:
                raise
            self._record_failed_attempt(
                messages=messages,
                trace_attempts=trace_attempts,
                error_type="request_too_large",
                error_message=str(exc),
            )
            messages = build_messages(observation, compact=True)
            try:
                response_payload = self._complete_and_trace(
                    messages=messages,
                    trace_attempts=trace_attempts,
                    player_id=observation.player_id,
                    stage=stage,
                )
            except RuntimeError:
                if not (
                    salvage_on_error
                    and self._should_salvage_invalid_response(trace_attempts)
                ):
                    self._append_prompt_trace_entry(
                        player_id=observation.player_id,
                        history_index=observation.history_index,
                        turn_index=observation.turn_index,
                        phase=observation.phase,
                        decision_index=observation.decision_index,
                        stage=stage,
                        attempts=trace_attempts,
                    )
                    raise
        except RuntimeError:
            if not (
                salvage_on_error
                and self._should_salvage_invalid_response(trace_attempts)
            ):
                self._append_prompt_trace_entry(
                    player_id=observation.player_id,
                    history_index=observation.history_index,
                    turn_index=observation.turn_index,
                    phase=observation.phase,
                    decision_index=observation.decision_index,
                    stage=stage,
                    attempts=trace_attempts,
                )
                raise
        return response_payload, messages, trace_attempts

    def _traced_call_and_record(
        self,
        *,
        observation,
        stage: str,
        build_messages: Callable,
        compact_retry: bool = True,
    ) -> dict[str, object]:
        payload, _, attempts = self._traced_llm_call(
            observation=observation,
            stage=stage,
            build_messages=build_messages,
            compact_retry=compact_retry,
        )
        self._append_prompt_trace_entry(
            player_id=observation.player_id,
            history_index=observation.history_index,
            turn_index=observation.turn_index,
            phase=observation.phase,
            decision_index=observation.decision_index,
            stage=stage,
            attempts=attempts,
        )
        return payload

    def plan_opening_strategy(
        self, observation: OpeningStrategyObservation
    ) -> OpeningStrategyResponse:
        payload = self._traced_call_and_record(
            observation=observation,
            stage="opening_strategy",
            build_messages=self._messages_for_opening_strategy,
        )
        return OpeningStrategyResponse(
            long_term=self._coerce_memory_field(payload, "long_term")
        )

    def start_turn(self, observation: TurnStartObservation) -> TurnStartResponse:
        payload = self._traced_call_and_record(
            observation=observation,
            stage="turn_start",
            build_messages=self._messages_for_turn_start,
        )
        return TurnStartResponse(
            short_term=self._coerce_memory_field(payload, "short_term"),
            public_chat=self._coerce_public_chat_draft(payload),
        )

    def choose_action(self, observation: ActionObservation) -> ActionDecision:
        payload, messages, attempts = self._traced_llm_call(
            observation=observation,
            stage="choose_action",
            build_messages=self._messages_for_action,
            salvage_on_error=True,
        )
        return self._resolve_action_decision(
            observation=observation,
            initial_messages=messages,
            initial_payload=payload,
            trace_attempts=attempts,
            stage="choose_action",
            parse_decision=self._action_decision_from_payload,
        )

    def end_turn(self, observation: TurnEndObservation) -> TurnEndResponse:
        payload = self._traced_call_and_record(
            observation=observation,
            stage="turn_end",
            build_messages=self._messages_for_turn_end,
        )
        return TurnEndResponse(
            long_term=self._coerce_memory_field(payload, "long_term"),
            public_chat=self._coerce_public_chat_draft(payload),
        )

    def respond_reactive(self, observation: ReactiveObservation) -> ActionDecision:
        payload, messages, attempts = self._traced_llm_call(
            observation=observation,
            stage="reactive_action",
            build_messages=self._messages_for_reactive,
            salvage_on_error=True,
        )
        return self._resolve_action_decision(
            observation=observation,
            initial_messages=messages,
            initial_payload=payload,
            trace_attempts=attempts,
            stage="reactive_action",
            parse_decision=self._reactive_decision_from_payload,
        )

    def post_game_chat(
        self, observation: PostGameChatObservation
    ) -> PostGameChatResponse:
        payload = self._traced_call_and_record(
            observation=observation,
            stage="post_game_chat",
            build_messages=self._messages_for_post_game_chat,
            compact_retry=False,
        )
        return PostGameChatResponse(
            public_chat=self._coerce_public_chat_draft(payload),
            reasoning=self._coerce_reasoning(payload),
        )

    def open_trade_chat(
        self, observation: TradeChatObservation
    ) -> TradeChatOpenResponse:
        payload = self._traced_call_and_record(
            observation=observation,
            stage="trade_chat_open",
            build_messages=self._messages_for_trade_chat_open,
            compact_retry=False,
        )
        return self._trade_chat_open_response_from_payload(payload)

    def respond_trade_chat(
        self, observation: TradeChatObservation
    ) -> TradeChatReplyResponse:
        payload = self._traced_call_and_record(
            observation=observation,
            stage="trade_chat_reply",
            build_messages=self._messages_for_trade_chat_reply,
            compact_retry=False,
        )
        return self._trade_chat_reply_response_from_payload(
            observation=observation,
            response_payload=payload,
        )

    def decide_trade_chat(
        self, observation: TradeChatObservation
    ) -> TradeChatOwnerDecisionResponse:
        payload = self._traced_call_and_record(
            observation=observation,
            stage="trade_chat_owner_decision",
            build_messages=self._messages_for_trade_chat_owner_decision,
            compact_retry=False,
        )
        return self._trade_chat_owner_decision_response_from_payload(
            observation=observation,
            response_payload=payload,
        )

    def select_trade_chat_offer(
        self, observation: TradeChatObservation
    ) -> TradeChatOwnerDecisionResponse:
        return self.decide_trade_chat(observation)

    def take_last_prompt_trace(self) -> PromptTrace | None:
        if not self._prompt_traces:
            return None
        return self._prompt_traces.pop()

    def take_prompt_traces(self) -> tuple[PromptTrace, ...]:
        traces = tuple(self._prompt_traces)
        self._prompt_traces.clear()
        return traces

    def _messages_for_turn_start(
        self,
        observation: TurnStartObservation,
        *,
        compact: bool = False,
    ) -> list[dict[str, object]]:
        payload = {
            "history_index": observation.history_index,
            "player_id": observation.player_id,
            "turn_index": observation.turn_index,
            "phase": observation.phase,
            "decision_index": observation.decision_index,
            "public_state": observation.public_state,
            "private_state": observation.private_state,
            "public_history_since_last_turn": [
                event.to_dict()
                for event in self._tail_events(
                    observation.public_history_since_last_turn,
                    compact=compact,
                )
            ],
            "memory": observation.memory.to_dict(),
            "public_chat_enabled": observation.public_chat_enabled,
            "public_chat_transcript": [
                event.to_dict()
                for event in self._tail_events(
                    observation.public_chat_transcript, compact=compact
                )
            ],
            "public_chat_message_char_limit": observation.public_chat_message_char_limit,
            "context_window": {
                "compact_retry": compact,
                "history_limit": self._effective_history_limit(compact=compact),
            },
        }
        return self.renderer.render_messages(
            system_template="turn_start_system.jinja",
            user_template="turn_start_user.jinja",
            payload=payload,
            game_rules=observation.game_rules,
        )

    def _messages_for_opening_strategy(
        self,
        observation: OpeningStrategyObservation,
        *,
        compact: bool = False,
    ) -> list[dict[str, object]]:
        payload = {
            "history_index": observation.history_index,
            "player_id": observation.player_id,
            "turn_index": observation.turn_index,
            "phase": observation.phase,
            "decision_index": observation.decision_index,
            "public_state": observation.public_state,
            "private_state": observation.private_state,
            "public_history": [
                event.to_dict()
                for event in self._tail_events(
                    observation.public_history, compact=compact
                )
            ],
            "memory": observation.memory.to_dict(),
            "context_window": {
                "compact_retry": compact,
                "history_limit": self._effective_history_limit(compact=compact),
            },
        }
        return self.renderer.render_messages(
            system_template="opening_strategy_system.jinja",
            user_template="opening_strategy_user.jinja",
            payload=payload,
            game_rules=observation.game_rules,
        )

    def _messages_for_action(
        self,
        observation: ActionObservation,
        *,
        compact: bool = False,
    ) -> list[dict[str, object]]:
        placement_candidates_key = self._placement_candidates_key(observation)
        payload = {
            "history_index": observation.history_index,
            "player_id": observation.player_id,
            "turn_index": observation.turn_index,
            "phase": observation.phase,
            "decision_index": observation.decision_index,
            "public_state": observation.public_state,
            "private_state": observation.private_state,
            "public_history": [
                event.to_dict()
                for event in self._tail_events(
                    observation.public_history, compact=compact
                )
            ],
            "turn_public_events": [
                event.to_dict()
                for event in self._tail_events(
                    observation.turn_public_events, compact=compact
                )
            ],
            "memory": observation.memory.to_dict(),
            "public_chat_enabled": observation.public_chat_enabled,
            "public_chat_transcript": [
                event.to_dict()
                for event in self._tail_events(
                    observation.public_chat_transcript, compact=compact
                )
            ],
            "public_chat_message_char_limit": observation.public_chat_message_char_limit,
            "decision_prompt": observation.decision_prompt,
            "trade_chat_enabled": observation.trade_chat_enabled,
            "trade_chat_attempts_remaining": observation.trade_chat_attempts_remaining,
            "context_window": {
                "compact_retry": compact,
                "history_limit": self._effective_history_limit(compact=compact),
            },
        }
        if placement_candidates_key is None:
            payload["legal_actions"] = self._action_prompt_entries(
                observation.legal_actions
            )
        else:
            payload["candidate_source"] = (
                f"public_state.board.{placement_candidates_key}"
            )
        return self.renderer.render_messages(
            system_template="choose_action_system.jinja",
            user_template="choose_action_user.jinja",
            payload=payload,
            game_rules=observation.game_rules,
        )

    def _messages_for_turn_end(
        self,
        observation: TurnEndObservation,
        *,
        compact: bool = False,
    ) -> list[dict[str, object]]:
        payload = {
            "history_index": observation.history_index,
            "player_id": observation.player_id,
            "turn_index": observation.turn_index,
            "phase": observation.phase,
            "decision_index": observation.decision_index,
            "public_state": observation.public_state,
            "private_state": observation.private_state,
            "turn_public_events": [
                event.to_dict()
                for event in self._tail_events(
                    observation.turn_public_events, compact=compact
                )
            ],
            "memory": observation.memory.to_dict(),
            "public_chat_enabled": observation.public_chat_enabled,
            "public_chat_transcript": [
                event.to_dict()
                for event in self._tail_events(
                    observation.public_chat_transcript, compact=compact
                )
            ],
            "public_chat_message_char_limit": observation.public_chat_message_char_limit,
            "context_window": {
                "compact_retry": compact,
                "history_limit": self._effective_history_limit(compact=compact),
            },
        }
        return self.renderer.render_messages(
            system_template="turn_end_system.jinja",
            user_template="turn_end_user.jinja",
            payload=payload,
            game_rules=observation.game_rules,
        )

    def _messages_for_reactive(
        self,
        observation: ReactiveObservation,
        *,
        compact: bool = False,
    ) -> list[dict[str, object]]:
        placement_candidates_key = self._placement_candidates_key(observation)
        payload = {
            "history_index": observation.history_index,
            "player_id": observation.player_id,
            "turn_index": observation.turn_index,
            "phase": observation.phase,
            "decision_index": observation.decision_index,
            "public_state": observation.public_state,
            "private_state": observation.private_state,
            "public_history": [
                event.to_dict()
                for event in self._tail_events(
                    observation.public_history, compact=compact
                )
            ],
            "memory": observation.memory.to_dict(),
            "public_chat_enabled": observation.public_chat_enabled,
            "public_chat_transcript": [
                event.to_dict()
                for event in self._tail_events(
                    observation.public_chat_transcript, compact=compact
                )
            ],
            "public_chat_message_char_limit": observation.public_chat_message_char_limit,
            "decision_prompt": observation.decision_prompt,
            "context_window": {
                "compact_retry": compact,
                "history_limit": self._effective_history_limit(compact=compact),
            },
        }
        if placement_candidates_key is None:
            payload["legal_actions"] = self._action_prompt_entries(
                observation.legal_actions
            )
        else:
            payload["candidate_source"] = (
                f"public_state.board.{placement_candidates_key}"
            )
        return self.renderer.render_messages(
            system_template="reactive_action_system.jinja",
            user_template="reactive_action_user.jinja",
            payload=payload,
            game_rules=observation.game_rules,
        )

    def _messages_for_post_game_chat(
        self,
        observation: PostGameChatObservation,
        *,
        compact: bool = False,
    ) -> list[dict[str, object]]:
        payload = {
            "history_index": observation.history_index,
            "player_id": observation.player_id,
            "turn_index": observation.turn_index,
            "phase": observation.phase,
            "decision_index": observation.decision_index,
            "public_state": observation.public_state,
            "private_state": observation.private_state,
            "public_history": [
                event.to_dict()
                for event in self._tail_events(
                    observation.public_history, compact=compact
                )
            ],
            "memory": observation.memory.to_dict(),
            "public_chat_enabled": observation.public_chat_enabled,
            "public_chat_transcript": [
                event.to_dict()
                for event in self._tail_events(
                    observation.public_chat_transcript, compact=compact
                )
            ],
            "public_chat_message_char_limit": observation.public_chat_message_char_limit,
            "result": dict(observation.result),
            "context_window": {
                "compact_retry": compact,
                "history_limit": self._effective_history_limit(compact=compact),
            },
        }
        return self.renderer.render_messages(
            system_template="post_game_chat_system.jinja",
            user_template="post_game_chat_user.jinja",
            payload=payload,
            game_rules=observation.game_rules,
        )

    def _messages_for_trade_chat_open(
        self, observation: TradeChatObservation
    ) -> list[dict[str, object]]:
        visible_public_history = self._tail_events(
            observation.public_history, compact=False
        )
        payload = {
            "history_index": observation.history_index,
            "player_id": observation.player_id,
            "owner_player_id": observation.owner_player_id,
            "turn_index": observation.turn_index,
            "phase": observation.phase,
            "decision_index": observation.decision_index,
            "attempt_index": observation.attempt_index,
            "round_index": observation.round_index,
            "public_state": observation.public_state,
            "private_state": observation.private_state,
            "public_history": [event.to_dict() for event in visible_public_history],
            "previous_trade_chat_attempts": self._previous_trade_chat_attempts(
                visible_public_history,
                turn_index=observation.turn_index,
                current_attempt_index=observation.attempt_index,
            ),
            "memory": observation.memory.to_dict(),
            "requested_resources": observation.requested_resources,
            "other_player_ids": list(observation.other_player_ids),
            "public_chat_transcript": [
                event.to_dict() for event in observation.public_chat_transcript
            ],
            "transcript": [
                event.to_dict()
                for event in self._tail_events(observation.transcript, compact=False)
            ],
            "message_char_limit": observation.message_char_limit,
        }
        return self.renderer.render_messages(
            system_template="trade_chat_open_system.jinja",
            user_template="trade_chat_open_user.jinja",
            payload=payload,
            game_rules=observation.game_rules,
        )

    def _previous_trade_chat_attempts(
        self,
        public_history: tuple[Event, ...],
        *,
        turn_index: int,
        current_attempt_index: int,
    ) -> list[dict[str, object]]:
        attempts: dict[int, list[Event]] = {}
        for event in public_history:
            if (
                event.turn_index != turn_index
                or event.kind not in _TRADE_CHAT_ROOM_EVENT_KINDS
            ):
                continue
            attempt_index = event.payload.get("attempt_index")
            if (
                not isinstance(attempt_index, int)
                or attempt_index >= current_attempt_index
            ):
                continue
            attempts.setdefault(attempt_index, []).append(event)
        return [
            {
                "attempt_index": attempt_index,
                "events": [event.to_dict() for event in events],
            }
            for attempt_index, events in sorted(attempts.items())
        ]

    def _messages_for_trade_chat_reply(
        self, observation: TradeChatObservation
    ) -> list[dict[str, object]]:
        payload = {
            "history_index": observation.history_index,
            "player_id": observation.player_id,
            "owner_player_id": observation.owner_player_id,
            "turn_index": observation.turn_index,
            "phase": observation.phase,
            "decision_index": observation.decision_index,
            "attempt_index": observation.attempt_index,
            "round_index": observation.round_index,
            "public_state": observation.public_state,
            "private_state": observation.private_state,
            "public_history": [
                event.to_dict()
                for event in self._tail_events(
                    observation.public_history, compact=False
                )
            ],
            "memory": observation.memory.to_dict(),
            "requested_resources": observation.requested_resources,
            "proposals": [proposal.to_dict() for proposal in observation.proposals],
            "public_chat_transcript": [
                event.to_dict() for event in observation.public_chat_transcript
            ],
            "transcript": [
                event.to_dict()
                for event in self._tail_events(observation.transcript, compact=False)
            ],
            "message_char_limit": observation.message_char_limit,
        }
        return self.renderer.render_messages(
            system_template="trade_chat_reply_system.jinja",
            user_template="trade_chat_reply_user.jinja",
            payload=payload,
            game_rules=observation.game_rules,
        )

    def _messages_for_trade_chat_owner_decision(
        self, observation: TradeChatObservation
    ) -> list[dict[str, object]]:
        payload = {
            "history_index": observation.history_index,
            "player_id": observation.player_id,
            "owner_player_id": observation.owner_player_id,
            "turn_index": observation.turn_index,
            "phase": observation.phase,
            "decision_index": observation.decision_index,
            "attempt_index": observation.attempt_index,
            "round_index": observation.round_index,
            "public_state": observation.public_state,
            "private_state": observation.private_state,
            "public_history": [
                event.to_dict()
                for event in self._tail_events(
                    observation.public_history, compact=False
                )
            ],
            "memory": observation.memory.to_dict(),
            "requested_resources": observation.requested_resources,
            "proposals": [proposal.to_dict() for proposal in observation.proposals],
            "public_chat_transcript": [
                event.to_dict() for event in observation.public_chat_transcript
            ],
            "transcript": [
                event.to_dict()
                for event in self._tail_events(observation.transcript, compact=False)
            ],
            "message_char_limit": observation.message_char_limit,
        }
        return self.renderer.render_messages(
            system_template="trade_chat_owner_decision_system.jinja",
            user_template="trade_chat_owner_decision_user.jinja",
            payload=payload,
            game_rules=observation.game_rules,
        )

    def _action_decision_from_payload(
        self, observation: ActionObservation, response_payload: dict[str, object]
    ) -> ActionDecision:
        return ActionDecision(
            action=self._action_from_payload(
                observation.legal_actions, response_payload
            ),
            short_term=self._coerce_memory_field(response_payload, "short_term"),
            public_chat=self._coerce_public_chat_draft(response_payload),
            reasoning=self._coerce_reasoning(response_payload),
        )

    def _reactive_decision_from_payload(
        self, observation: ReactiveObservation, response_payload: dict[str, object]
    ) -> ActionDecision:
        return ActionDecision(
            action=self._action_from_payload(
                observation.legal_actions, response_payload
            ),
            short_term=None,
            public_chat=self._coerce_public_chat_draft(response_payload),
            reasoning=self._coerce_reasoning(response_payload),
        )

    @staticmethod
    def _coerce_memory_field(
        response_payload: dict[str, object],
        expected_key: str,
        *,
        default: JsonValue | None = None,
    ) -> JsonValue | None:
        candidates = [expected_key]
        if expected_key == "short_term":
            candidates.extend(["turn_strategy", "strategy", "plan"])
        if expected_key == "long_term":
            candidates.extend(["game_memory", "memory", "long_term_memory"])
        for key in candidates:
            if key in response_payload:
                return response_payload.get(key)
        return default

    @staticmethod
    def _action_from_payload(
        legal_actions: tuple[Action, ...],
        response_payload: dict[str, object],
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
            if not 0 <= action_index < len(legal_actions):
                raise RuntimeError(
                    f"LLM action_index {action_index} is out of range for legal actions."
                )
            indexed_action = legal_actions[action_index]
            # When action_index selects a trade template, the LLM may have also
            # provided a concrete payload in the `action` field — fall through so
            # that payload is picked up instead of the empty template.
            if not _is_resource_swap_template(indexed_action):
                return indexed_action

        action_payload = response_payload.get("action")
        if isinstance(action_payload, int):
            if not 0 <= action_payload < len(legal_actions):
                raise RuntimeError(
                    f"LLM action index {action_payload} is out of range for legal actions."
                )
            return legal_actions[action_payload]
        if isinstance(action_payload, str) and action_payload.isdigit():
            indexed_action = int(action_payload)
            if not 0 <= indexed_action < len(legal_actions):
                raise RuntimeError(
                    f"LLM action index {indexed_action} is out of range for legal actions."
                )
            return legal_actions[indexed_action]
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

        action_type = (
            action_payload.get("action_type")
            or action_payload.get("type")
            or action_payload.get("action")
        )
        if not isinstance(action_type, str):
            raise RuntimeError("LLM action must include string field `action_type`.")

        payload = action_payload.get("payload")
        if payload is None:
            payload = {
                key: value
                for key, value in action_payload.items()
                if key not in {"action_type", "type", "action", "description"}
            }
        if not isinstance(payload, dict):
            raise RuntimeError("LLM action payload must be a JSON object.")

        # Normalize common trade field-name variants the LLM may produce.
        if action_type in {"OFFER_TRADE", "COUNTER_OFFER"}:
            payload = _normalize_offer_trade_payload(payload)

        return Action(action_type=action_type, payload=payload)

    @staticmethod
    def _coerce_reasoning(response_payload: dict[str, object]) -> str | None:
        for key in ("private_reasoning", "reasoning", "thought", "summary"):
            value = response_payload.get(key)
            if value is None:
                continue
            if not isinstance(value, str):
                raise RuntimeError(
                    f"LLM response field `{key}` must be a string when present."
                )
            return value
        return None

    @staticmethod
    def _coerce_public_message(value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise RuntimeError(
                "Public chat message fields must be strings when present."
            )
        stripped = value.strip()
        return stripped or None

    @classmethod
    def _coerce_public_chat_draft(
        cls, response_payload: dict[str, object]
    ) -> PublicChatDraft | None:
        public_chat_payload = response_payload.get("public_chat")
        if public_chat_payload is None:
            message = cls._coerce_public_message(response_payload.get("public_message"))
            target_player_id = response_payload.get("public_target_player_id")
            if target_player_id is None:
                target_player_id = response_payload.get("target_player_id")
        else:
            if not isinstance(public_chat_payload, dict):
                raise RuntimeError("`public_chat` must be an object or null.")
            message = cls._coerce_public_message(public_chat_payload.get("message"))
            target_player_id = public_chat_payload.get("target_player_id")
        if message is None:
            return None
        if target_player_id is not None and not isinstance(target_player_id, str):
            raise RuntimeError(
                "`public_chat.target_player_id` must be a string when present."
            )
        normalized_target = (
            target_player_id.strip().upper()
            if isinstance(target_player_id, str) and target_player_id.strip()
            else None
        )
        return PublicChatDraft(
            message=message,
            target_player_id=normalized_target,
        )

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
        if not open_chat or (message is None and not requested_resources):
            return TradeChatOpenResponse(
                open_chat=False, reasoning=self._coerce_reasoning(response_payload)
            )
        return TradeChatOpenResponse(
            open_chat=True,
            message=message,
            requested_resources=requested_resources,
            reasoning=self._coerce_reasoning(response_payload),
        )

    def _trade_chat_reply_response_from_payload(
        self,
        *,
        observation: TradeChatObservation,
        response_payload: dict[str, object],
    ) -> TradeChatReplyResponse:
        owner_gives: dict[str, JsonValue]
        owner_gets: dict[str, JsonValue]
        if "you_give" in response_payload or "you_get" in response_payload:
            owner_gives = self._coerce_resource_map(response_payload.get("you_get"))
            owner_gets = self._coerce_resource_map(response_payload.get("you_give"))
        elif "owner_gives" in response_payload or "owner_gets" in response_payload:
            owner_gives = self._coerce_resource_map(response_payload.get("owner_gives"))
            owner_gets = self._coerce_resource_map(response_payload.get("owner_gets"))
        else:
            generic_offer = self._coerce_resource_map(
                response_payload.get("offer") or response_payload.get("give")
            )
            generic_request = self._coerce_resource_map(
                response_payload.get("request")
                or response_payload.get("want")
                or response_payload.get("take")
            )
            owner_gives, owner_gets = self._normalize_trade_chat_terms(
                observation=observation,
                generic_offer=generic_offer,
                generic_request=generic_request,
            )
        return TradeChatReplyResponse(
            message=self._coerce_public_message(response_payload.get("message")),
            owner_gives=owner_gives,
            owner_gets=owner_gets,
            reasoning=self._coerce_reasoning(response_payload),
        )

    def _normalize_trade_chat_terms(
        self,
        *,
        observation: TradeChatObservation,
        generic_offer: dict[str, JsonValue],
        generic_request: dict[str, JsonValue],
    ) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
        if not generic_offer and not generic_request:
            return {}, {}

        owner_relative = (dict(generic_offer), dict(generic_request))
        responder_relative = (dict(generic_request), dict(generic_offer))
        if self._score_trade_chat_candidate(
            observation=observation, candidate=owner_relative
        ) >= self._score_trade_chat_candidate(
            observation=observation,
            candidate=responder_relative,
        ):
            return owner_relative
        return responder_relative

    def _score_trade_chat_candidate(
        self,
        *,
        observation: TradeChatObservation,
        candidate: tuple[dict[str, JsonValue], dict[str, JsonValue]],
    ) -> int:
        owner_gives, owner_gets = candidate
        if not owner_gives and not owner_gets:
            return -1000

        score = 0
        player_resources = self._coerce_resource_map(
            observation.private_state.get("resources")
        )
        if owner_gets:
            if self._resource_map_is_subset(player_resources, owner_gets):
                score += 100
            else:
                score -= 100

        requested_resources = self._coerce_resource_map(observation.requested_resources)
        if requested_resources:
            score += 10 * self._resource_map_overlap(owner_gets, requested_resources)
            if owner_gets == requested_resources:
                score += 5

        if owner_gives and owner_gets:
            score += 1
        return score

    @staticmethod
    def _resource_map_is_subset(
        owned: dict[str, JsonValue],
        required: dict[str, JsonValue],
    ) -> bool:
        for resource, amount in required.items():
            if not isinstance(amount, int):
                return False
            if int(owned.get(resource, 0)) < amount:
                return False
        return True

    @staticmethod
    def _resource_map_overlap(
        left: dict[str, JsonValue],
        right: dict[str, JsonValue],
    ) -> int:
        overlap = 0
        for resource, left_amount in left.items():
            right_amount = right.get(resource)
            if isinstance(left_amount, int) and isinstance(right_amount, int):
                overlap += min(left_amount, right_amount)
        return overlap

    def _trade_chat_owner_decision_response_from_payload(
        self,
        *,
        observation: TradeChatObservation,
        response_payload: dict[str, object],
    ) -> TradeChatOwnerDecisionResponse:
        decision = response_payload.get("decision")
        if not isinstance(decision, str):
            if response_payload.get("selected_proposal_id") is not None:
                decision = "select"
            elif response_payload.get("selected_player_id") is not None:
                decision = "select"
            elif bool(response_payload.get("continue_chat")):
                decision = "continue"
            else:
                decision = "close"
        decision = decision.lower()
        if decision not in {"continue", "select", "close"}:
            decision = "close"
        if decision == "select" and not observation.proposals:
            decision = "close"

        selected_proposal_id = response_payload.get("selected_proposal_id")
        if not isinstance(selected_proposal_id, str):
            selected_player_id = response_payload.get("selected_player_id")
            if not isinstance(selected_player_id, str):
                selected_player_id = response_payload.get("player_id")
            if isinstance(selected_player_id, str):
                selected_player_id = selected_player_id.upper()
                for proposal in reversed(observation.proposals):
                    if proposal.player_id == selected_player_id:
                        selected_proposal_id = proposal.proposal_id
                        break
        selected_proposal_id = _coerce_trade_chat_selected_proposal_id(
            observation,
            selected_proposal_id,
        )

        return TradeChatOwnerDecisionResponse(
            decision=decision,
            selected_proposal_id=selected_proposal_id,
            message=self._coerce_public_message(response_payload.get("message")),
            reasoning=self._coerce_reasoning(response_payload),
        )

    @staticmethod
    def _is_legal_response_action(
        action: Action, legal_actions: tuple[Action, ...]
    ) -> bool:
        if _is_resource_swap_template(action):
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
        if action.action_type in {"OFFER_TRADE", "COUNTER_OFFER"}:
            # Any trade-like action with a non-empty offer and request is accepted
            # here; the orchestrator / engine adapter performs the deeper handling.
            return (
                len(matching_types) > 0
                and bool(action.payload.get("offer"))
                and bool(action.payload.get("request"))
            )
        return False

    @staticmethod
    def _fallback_legal_action(legal_actions: tuple[Action, ...]) -> Action:
        for action in legal_actions:
            if not _is_resource_swap_template(action):
                return action
        if any(_is_resource_swap_template(action) for action in legal_actions):
            return _materialize_default_trade_offer()
        raise RuntimeError("No fallback legal action was available.")

    def _repair_messages(
        self,
        observation: ActionObservation | ReactiveObservation,
        attempted_response: dict[str, object],
    ) -> list[dict[str, object]]:
        placement_candidates_key = self._placement_candidates_key(observation)
        payload: dict[str, object] = {
            "player_id": observation.player_id,
            "public_state": observation.public_state,
            "error": "Your previous response selected an illegal action for the current decision.",
            "previous_response": attempted_response,
        }
        if placement_candidates_key is None:
            payload["legal_actions"] = self._action_prompt_entries(
                observation.legal_actions
            )
        else:
            payload["candidate_source"] = (
                f"Use one action_index from public_state.board.{placement_candidates_key}."
            )
        return self.renderer.render_messages(
            system_template="repair_action_system.jinja",
            user_template="repair_action_user.jinja",
            payload=payload,
            game_rules="",
        )

    def _resolve_action_decision(
        self,
        *,
        observation: ActionObservation | ReactiveObservation,
        initial_messages: list[dict[str, object]],
        initial_payload: dict[str, object],
        trace_attempts: list[PromptTraceAttempt],
        stage: str,
        parse_decision: Callable[
            [ActionObservation | ReactiveObservation, dict[str, object]],
            ActionDecision,
        ],
    ) -> ActionDecision:
        decision = self._parse_action_decision(
            parse_decision, observation, initial_payload
        )
        if decision is not None and self._is_legal_response_action(
            decision.action, observation.legal_actions
        ):
            self._record_action_trace(
                observation=observation,
                stage=stage,
                attempts=trace_attempts,
            )
            return decision

        repair_messages = initial_messages + self._repair_messages(
            observation, initial_payload
        )
        repaired_payload: dict[str, object] | None = None
        repaired_decision: ActionDecision | None = None
        try:
            repaired_payload = self._complete_and_trace(
                messages=repair_messages,
                trace_attempts=trace_attempts,
                player_id=observation.player_id,
                stage=f"{stage}_repair",
            )
            repaired_decision = self._parse_action_decision(
                parse_decision,
                observation,
                repaired_payload,
            )
        except LLMRequestTooLargeError as exc:
            self._record_failed_attempt(
                messages=repair_messages,
                trace_attempts=trace_attempts,
                error_type="request_too_large",
                error_message=str(exc),
            )
        except RuntimeError:
            repaired_decision = None

        self._record_action_trace(
            observation=observation,
            stage=stage,
            attempts=trace_attempts,
        )
        if repaired_decision is not None and self._is_legal_response_action(
            repaired_decision.action, observation.legal_actions
        ):
            return repaired_decision

        return ActionDecision(
            action=self._fallback_legal_action(observation.legal_actions),
            short_term=None,
            reasoning=self._best_effort_reasoning(
                repaired_payload,
                fallback=decision.reasoning if decision is not None else None,
            ),
        )

    @staticmethod
    def _parse_action_decision(
        parse_decision: Callable[
            [ActionObservation | ReactiveObservation, dict[str, object]],
            ActionDecision,
        ],
        observation: ActionObservation | ReactiveObservation,
        response_payload: dict[str, object],
    ) -> ActionDecision | None:
        try:
            return parse_decision(observation, response_payload)
        except RuntimeError:
            return None

    @classmethod
    def _best_effort_reasoning(
        cls,
        response_payload: dict[str, object] | None,
        *,
        fallback: str | None,
    ) -> str | None:
        if response_payload is None:
            return fallback
        try:
            return cls._coerce_reasoning(response_payload)
        except RuntimeError:
            return fallback

    @staticmethod
    def _should_salvage_invalid_response(
        trace_attempts: list[PromptTraceAttempt],
    ) -> bool:
        if not trace_attempts:
            return False
        last_response = trace_attempts[-1].response
        error_payload = (
            last_response.get("error") if isinstance(last_response, dict) else None
        )
        if not isinstance(error_payload, dict):
            return False
        return error_payload.get("type") == "invalid_response"

    def _complete_and_trace(
        self,
        *,
        messages: list[dict[str, object]],
        trace_attempts: list[PromptTraceAttempt],
        player_id: str,
        stage: str,
    ) -> dict[str, object]:
        last_error: RuntimeError | None = None
        for attempt_index in range(self.invalid_response_retries + 1):
            try:
                completion = self.client.complete(
                    model=self.model,
                    temperature=self.temperature,
                    messages=messages,
                    top_p=self.top_p,
                    reasoning_enabled=self.reasoning_enabled,
                    reasoning_effort=self.reasoning_effort,
                )
            except LLMRequestTooLargeError:
                raise
            except RuntimeError as exc:
                self._record_failed_attempt(
                    messages=messages,
                    trace_attempts=trace_attempts,
                    error_type="llm_request_failed",
                    error_message=str(exc),
                )
                raise

            try:
                raw_response_text = self._extract_response_text(
                    completion,
                    player_id=player_id,
                    stage=stage,
                )
            except RuntimeError as exc:
                trace_attempts.append(
                    PromptTraceAttempt(
                        messages=tuple(
                            self._json_safe_message(message) for message in messages
                        ),
                        response=self._json_safe_object(
                            {
                                "error": {
                                    "type": "invalid_response",
                                    "message": str(exc),
                                    "retry_attempt": attempt_index + 1,
                                }
                            }
                        ),
                        response_text=None,
                    )
                )
                last_error = exc
                if attempt_index < self.invalid_response_retries:
                    continue
                raise
            try:
                response_payload = self._parse_response_json(
                    raw_response_text=raw_response_text,
                    completion=completion,
                )
            except RuntimeError as exc:
                trace_attempts.append(
                    PromptTraceAttempt(
                        messages=tuple(
                            self._json_safe_message(message) for message in messages
                        ),
                        response=self._json_safe_object(
                            {
                                "error": {
                                    "type": "invalid_response",
                                    "message": str(exc),
                                    "retry_attempt": attempt_index + 1,
                                }
                            }
                        ),
                        response_text=raw_response_text,
                    )
                )
                last_error = exc
                if attempt_index < self.invalid_response_retries:
                    continue
                raise

            trace_attempts.append(
                PromptTraceAttempt(
                    messages=tuple(
                        self._json_safe_message(message) for message in messages
                    ),
                    response=self._json_safe_object(response_payload),
                    response_text=raw_response_text,
                )
            )
            return response_payload

        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM completion failed without returning a payload.")

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
                messages=tuple(
                    self._json_safe_message(message) for message in messages
                ),
                response=self._json_safe_object(
                    {"error": {"type": error_type, "message": error_message}}
                ),
                response_text=None,
            )
        )

    def _record_action_trace(
        self,
        *,
        observation: ActionObservation | ReactiveObservation,
        stage: str,
        attempts: list[PromptTraceAttempt],
    ) -> None:
        self._append_prompt_trace_entry(
            player_id=observation.player_id,
            history_index=observation.history_index,
            turn_index=observation.turn_index,
            phase=observation.phase,
            decision_index=observation.decision_index,
            stage=stage,
            attempts=attempts,
        )

    def _append_prompt_trace_entry(
        self,
        *,
        player_id: str,
        history_index: int,
        turn_index: int,
        phase: str,
        decision_index: int,
        stage: str,
        attempts: list[PromptTraceAttempt],
    ) -> None:
        self._prompt_traces.append(
            self._prompt_trace_for(
                player_id=player_id,
                history_index=history_index,
                turn_index=turn_index,
                phase=phase,
                decision_index=decision_index,
                stage=stage,
                attempts=attempts,
            )
        )

    def _prompt_trace_for(
        self,
        *,
        player_id: str,
        history_index: int,
        turn_index: int,
        phase: str,
        decision_index: int,
        stage: str,
        attempts: list[PromptTraceAttempt],
    ) -> PromptTrace:
        return PromptTrace(
            player_id=player_id,
            history_index=history_index,
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

    def _extract_response_text(
        self,
        completion: dict[str, object],
        *,
        player_id: str,
        stage: str,
    ) -> str:
        choices = completion.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(
                self._completion_shape_error(
                    player_id=player_id,
                    stage=stage,
                    detail="did not include any choices.",
                )
            )
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError(
                self._completion_shape_error(
                    player_id=player_id,
                    stage=stage,
                    detail=(
                        "had a first choice payload with unexpected type "
                        f"{type(first_choice).__name__}."
                    ),
                )
            )
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError(
                self._completion_shape_error(
                    player_id=player_id,
                    stage=stage,
                    detail="did not include a message payload.",
                )
            )
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(
                self._completion_shape_error(
                    player_id=player_id,
                    stage=stage,
                    detail=f"returned non-string `message.content` ({type(content).__name__}).",
                )
            )
        return content

    def _completion_shape_error(
        self,
        *,
        player_id: str,
        stage: str,
        detail: str,
    ) -> str:
        return (
            "LLM completion for "
            f"player {player_id!r} using model {self.model!r} during stage {stage!r} "
            f"{detail}"
        )

    @staticmethod
    def _parse_response_json(
        *, raw_response_text: str, completion: dict[str, object]
    ) -> dict[str, object]:
        choices = completion.get("choices")
        first_choice = choices[0] if isinstance(choices, list) and choices else None
        message = (
            first_choice.get("message") if isinstance(first_choice, dict) else None
        )
        stripped_content = LLMPlayer._strip_markdown_fences(raw_response_text)
        if not stripped_content:
            reasoning = message.get("reasoning") if isinstance(message, dict) else None
            finish_reason = (
                first_choice.get("finish_reason")
                if isinstance(first_choice, dict)
                else None
            )
            if isinstance(reasoning, str) and reasoning.strip():
                raise RuntimeError(
                    "LLM completion did not include JSON content. "
                    f"The provider returned reasoning-only output with finish_reason={finish_reason!r}."
                )
            raise RuntimeError("LLM completion returned an empty response.")
        try:
            response_payload = json.loads(stripped_content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"LLM completion did not decode as JSON: {exc.msg}"
            ) from exc
        if not isinstance(response_payload, dict):
            raise RuntimeError("LLM completion must decode to a JSON object.")
        return response_payload

    @staticmethod
    def _strip_markdown_fences(content: str) -> str:
        stripped = content.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if not lines:
            return stripped
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _tail_events(
        self, items: tuple[Event, ...], *, compact: bool
    ) -> tuple[Event, ...]:
        limit = self._effective_history_limit(compact=compact)
        if limit is None:
            return items
        return items[-limit:]

    def _effective_history_limit(self, *, compact: bool) -> int | None:
        if self.prompt_history_limit is None:
            return None if not compact else 6
        if not compact:
            return self.prompt_history_limit
        return max(
            1, min(self.prompt_history_limit, max(1, self.prompt_history_limit // 2))
        )

    @staticmethod
    def _placement_candidates_key(
        observation: ActionObservation | ReactiveObservation,
    ) -> str | None:
        action_types = {action.action_type for action in observation.legal_actions}
        if action_types == {"BUILD_SETTLEMENT"}:
            candidates_key = "settlement_candidates"
        elif action_types == {"BUILD_ROAD"}:
            candidates_key = "road_candidates"
        elif action_types == {"BUILD_CITY"}:
            candidates_key = "city_candidates"
        else:
            return None

        board = observation.public_state.get("board")
        if not isinstance(board, dict):
            return None
        candidates = board.get(candidates_key)
        if not isinstance(candidates, list) or len(candidates) != len(
            observation.legal_actions
        ):
            return None
        if any(
            not isinstance(candidate, dict)
            or not isinstance(candidate.get("action_index"), int)
            for candidate in candidates
        ):
            return None
        return candidates_key

    @staticmethod
    def _action_prompt_entries(
        legal_actions: tuple[Action, ...],
    ) -> list[dict[str, JsonValue]]:
        return [
            LLMPlayer._action_prompt_entry(index=index, action=action)
            for index, action in enumerate(legal_actions)
        ]

    @staticmethod
    def _action_prompt_entry(*, index: int, action: Action) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "index": index,
            "action_type": action.action_type,
        }
        if action.description is not None:
            payload["description"] = action.description
        if action.payload or _is_resource_swap_template(action):
            payload["payload"] = action.payload
        if _is_resource_swap_template(action):
            payload["selectable_by_index"] = False
            payload["requires_concrete_payload"] = True
        return payload
