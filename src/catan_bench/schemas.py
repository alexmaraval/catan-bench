from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class Action:
    action_type: str
    payload: dict[str, JsonValue] = field(default_factory=dict)
    description: str | None = None

    def matches(self, other: "Action") -> bool:
        return self.action_type == other.action_type and self.payload == other.payload

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "action_type": self.action_type,
            "payload": self.payload,
        }
        if self.description is not None:
            payload["description"] = self.description
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "Action":
        return cls(
            action_type=str(data["action_type"]),
            payload=dict(data.get("payload") or {}),
            description=str(data["description"])
            if data.get("description") is not None
            else None,
        )


@dataclass(frozen=True, slots=True)
class Event:
    kind: str
    payload: dict[str, JsonValue] = field(default_factory=dict)
    history_index: int = 0
    turn_index: int = 0
    phase: str = "unknown"
    decision_index: int | None = None
    actor_player_id: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "kind": self.kind,
            "payload": self.payload,
            "history_index": self.history_index,
            "turn_index": self.turn_index,
            "phase": self.phase,
        }
        if self.decision_index is not None:
            payload["decision_index"] = self.decision_index
        if self.actor_player_id is not None:
            payload["actor_player_id"] = self.actor_player_id
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "Event":
        return cls(
            kind=str(data["kind"]),
            payload=dict(data.get("payload") or {}),
            history_index=int(data.get("history_index", 0)),
            turn_index=int(data.get("turn_index", 0)),
            phase=str(data.get("phase", "unknown")),
            decision_index=int(data["decision_index"])
            if data.get("decision_index") is not None
            else None,
            actor_player_id=(
                str(data["actor_player_id"])
                if data.get("actor_player_id") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class PlayerMemory:
    long_term: JsonValue | None = None
    short_term: JsonValue | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "long_term": self.long_term,
            "short_term": self.short_term,
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue] | None) -> "PlayerMemory":
        if data is None:
            return cls()
        return cls(
            long_term=data.get("long_term"),
            short_term=data.get("short_term"),
        )


@dataclass(frozen=True, slots=True)
class PublicChatDraft:
    message: str
    target_player_id: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {"message": self.message}
        if self.target_player_id is not None:
            payload["target_player_id"] = self.target_player_id
        return payload


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    player_id: str
    history_index: int
    turn_index: int
    phase: str
    decision_index: int
    stage: str
    memory: PlayerMemory

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "player_id": self.player_id,
            "history_index": self.history_index,
            "turn_index": self.turn_index,
            "phase": self.phase,
            "decision_index": self.decision_index,
            "stage": self.stage,
            "memory": self.memory.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "MemorySnapshot":
        return cls(
            player_id=str(data["player_id"]),
            history_index=int(data.get("history_index", 0)),
            turn_index=int(data.get("turn_index", 0)),
            phase=str(data.get("phase", "unknown")),
            decision_index=int(data.get("decision_index", 0)),
            stage=str(data.get("stage", "unknown")),
            memory=PlayerMemory.from_dict(
                dict(data.get("memory") or {})
                if isinstance(data.get("memory"), dict)
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class PublicStateSnapshot:
    history_index: int
    turn_index: int
    phase: str
    decision_index: int | None
    public_state: dict[str, JsonValue]

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "history_index": self.history_index,
            "turn_index": self.turn_index,
            "phase": self.phase,
            "public_state": self.public_state,
        }
        if self.decision_index is not None:
            payload["decision_index"] = self.decision_index
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "PublicStateSnapshot":
        return cls(
            history_index=int(data.get("history_index", 0)),
            turn_index=int(data.get("turn_index", 0)),
            phase=str(data.get("phase", "unknown")),
            decision_index=int(data["decision_index"])
            if data.get("decision_index") is not None
            else None,
            public_state=dict(data.get("public_state") or {}),
        )


@dataclass(frozen=True, slots=True)
class DecisionPoint:
    acting_player_id: str
    turn_index: int
    phase: str
    legal_actions: tuple[Action, ...]
    decision_index: int = 0
    prompt: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "acting_player_id": self.acting_player_id,
            "turn_index": self.turn_index,
            "phase": self.phase,
            "decision_index": self.decision_index,
            "legal_actions": [action.to_dict() for action in self.legal_actions],
        }
        if self.prompt is not None:
            payload["prompt"] = self.prompt
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "DecisionPoint":
        return cls(
            acting_player_id=str(data["acting_player_id"]),
            turn_index=int(data["turn_index"]),
            phase=str(data["phase"]),
            legal_actions=tuple(
                Action.from_dict(action) for action in data.get("legal_actions", ())
            ),
            decision_index=int(data.get("decision_index", 0)),
            prompt=str(data["prompt"]) if data.get("prompt") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class TurnStartObservation:
    game_id: str
    player_id: str
    history_index: int
    turn_index: int
    phase: str
    decision_index: int
    public_state: dict[str, JsonValue]
    private_state: dict[str, JsonValue]
    public_history_since_last_turn: tuple[Event, ...] = ()
    public_chat_enabled: bool = False
    public_chat_transcript: tuple[Event, ...] = ()
    public_chat_message_char_limit: int = 500
    game_rules: str | None = None
    memory: PlayerMemory = field(default_factory=PlayerMemory)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "game_id": self.game_id,
            "player_id": self.player_id,
            "history_index": self.history_index,
            "turn_index": self.turn_index,
            "phase": self.phase,
            "decision_index": self.decision_index,
            "public_state": self.public_state,
            "private_state": self.private_state,
            "public_history_since_last_turn": [
                event.to_dict() for event in self.public_history_since_last_turn
            ],
            "public_chat_enabled": self.public_chat_enabled,
            "public_chat_transcript": [
                event.to_dict() for event in self.public_chat_transcript
            ],
            "public_chat_message_char_limit": self.public_chat_message_char_limit,
            "game_rules": self.game_rules,
            "memory": self.memory.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class OpeningStrategyObservation:
    game_id: str
    player_id: str
    history_index: int
    turn_index: int
    phase: str
    decision_index: int
    public_state: dict[str, JsonValue]
    private_state: dict[str, JsonValue]
    public_history: tuple[Event, ...] = ()
    game_rules: str | None = None
    memory: PlayerMemory = field(default_factory=PlayerMemory)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "game_id": self.game_id,
            "player_id": self.player_id,
            "history_index": self.history_index,
            "turn_index": self.turn_index,
            "phase": self.phase,
            "decision_index": self.decision_index,
            "public_state": self.public_state,
            "private_state": self.private_state,
            "public_history": [event.to_dict() for event in self.public_history],
            "game_rules": self.game_rules,
            "memory": self.memory.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ActionObservation:
    game_id: str
    player_id: str
    history_index: int
    turn_index: int
    phase: str
    decision_index: int
    public_state: dict[str, JsonValue]
    private_state: dict[str, JsonValue]
    public_history: tuple[Event, ...] = ()
    turn_public_events: tuple[Event, ...] = ()
    public_chat_enabled: bool = False
    public_chat_transcript: tuple[Event, ...] = ()
    public_chat_message_char_limit: int = 500
    legal_actions: tuple[Action, ...] = ()
    decision_prompt: str | None = None
    trade_chat_enabled: bool = False
    trade_chat_attempts_remaining: int | None = None
    game_rules: str | None = None
    memory: PlayerMemory = field(default_factory=PlayerMemory)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "game_id": self.game_id,
            "player_id": self.player_id,
            "history_index": self.history_index,
            "turn_index": self.turn_index,
            "phase": self.phase,
            "decision_index": self.decision_index,
            "public_state": self.public_state,
            "private_state": self.private_state,
            "public_history": [event.to_dict() for event in self.public_history],
            "turn_public_events": [
                event.to_dict() for event in self.turn_public_events
            ],
            "public_chat_enabled": self.public_chat_enabled,
            "public_chat_transcript": [
                event.to_dict() for event in self.public_chat_transcript
            ],
            "public_chat_message_char_limit": self.public_chat_message_char_limit,
            "legal_actions": [action.to_dict() for action in self.legal_actions],
            "decision_prompt": self.decision_prompt,
            "trade_chat_enabled": self.trade_chat_enabled,
            "trade_chat_attempts_remaining": self.trade_chat_attempts_remaining,
            "game_rules": self.game_rules,
            "memory": self.memory.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TurnEndObservation:
    game_id: str
    player_id: str
    history_index: int
    turn_index: int
    phase: str
    decision_index: int
    public_state: dict[str, JsonValue]
    private_state: dict[str, JsonValue]
    turn_public_events: tuple[Event, ...] = ()
    public_chat_enabled: bool = False
    public_chat_transcript: tuple[Event, ...] = ()
    public_chat_message_char_limit: int = 500
    game_rules: str | None = None
    memory: PlayerMemory = field(default_factory=PlayerMemory)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "game_id": self.game_id,
            "player_id": self.player_id,
            "history_index": self.history_index,
            "turn_index": self.turn_index,
            "phase": self.phase,
            "decision_index": self.decision_index,
            "public_state": self.public_state,
            "private_state": self.private_state,
            "turn_public_events": [
                event.to_dict() for event in self.turn_public_events
            ],
            "public_chat_enabled": self.public_chat_enabled,
            "public_chat_transcript": [
                event.to_dict() for event in self.public_chat_transcript
            ],
            "public_chat_message_char_limit": self.public_chat_message_char_limit,
            "game_rules": self.game_rules,
            "memory": self.memory.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PostGameChatObservation:
    game_id: str
    player_id: str
    history_index: int
    turn_index: int
    phase: str
    decision_index: int
    public_state: dict[str, JsonValue]
    private_state: dict[str, JsonValue]
    public_history: tuple[Event, ...] = ()
    public_chat_enabled: bool = False
    public_chat_transcript: tuple[Event, ...] = ()
    public_chat_message_char_limit: int = 500
    result: dict[str, JsonValue] = field(default_factory=dict)
    game_rules: str | None = None
    memory: PlayerMemory = field(default_factory=PlayerMemory)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "game_id": self.game_id,
            "player_id": self.player_id,
            "history_index": self.history_index,
            "turn_index": self.turn_index,
            "phase": self.phase,
            "decision_index": self.decision_index,
            "public_state": self.public_state,
            "private_state": self.private_state,
            "public_history": [event.to_dict() for event in self.public_history],
            "public_chat_enabled": self.public_chat_enabled,
            "public_chat_transcript": [
                event.to_dict() for event in self.public_chat_transcript
            ],
            "public_chat_message_char_limit": self.public_chat_message_char_limit,
            "result": self.result,
            "game_rules": self.game_rules,
            "memory": self.memory.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ReactiveObservation:
    game_id: str
    player_id: str
    history_index: int
    turn_index: int
    phase: str
    decision_index: int
    public_state: dict[str, JsonValue]
    private_state: dict[str, JsonValue]
    public_history: tuple[Event, ...] = ()
    public_chat_enabled: bool = False
    public_chat_transcript: tuple[Event, ...] = ()
    public_chat_message_char_limit: int = 500
    legal_actions: tuple[Action, ...] = ()
    decision_prompt: str | None = None
    game_rules: str | None = None
    memory: PlayerMemory = field(default_factory=PlayerMemory)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "game_id": self.game_id,
            "player_id": self.player_id,
            "history_index": self.history_index,
            "turn_index": self.turn_index,
            "phase": self.phase,
            "decision_index": self.decision_index,
            "public_state": self.public_state,
            "private_state": self.private_state,
            "public_history": [event.to_dict() for event in self.public_history],
            "public_chat_enabled": self.public_chat_enabled,
            "public_chat_transcript": [
                event.to_dict() for event in self.public_chat_transcript
            ],
            "public_chat_message_char_limit": self.public_chat_message_char_limit,
            "legal_actions": [action.to_dict() for action in self.legal_actions],
            "decision_prompt": self.decision_prompt,
            "game_rules": self.game_rules,
            "memory": self.memory.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TradeChatProposal:
    proposal_id: str
    player_id: str
    round_index: int
    message: str | None = None
    owner_gives: dict[str, JsonValue] = field(default_factory=dict)
    owner_gets: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "proposal_id": self.proposal_id,
            "player_id": self.player_id,
            "round_index": self.round_index,
            "owner_gives": self.owner_gives,
            "owner_gets": self.owner_gets,
        }
        if self.message is not None:
            payload["message"] = self.message
        return payload


# Backward-compatible alias for older callers that still use the quote name.
TradeChatQuote = TradeChatProposal


@dataclass(frozen=True, slots=True)
class TradeChatObservation:
    game_id: str
    player_id: str
    owner_player_id: str
    history_index: int
    turn_index: int
    phase: str
    decision_index: int
    stage: str
    attempt_index: int
    round_index: int
    public_state: dict[str, JsonValue]
    private_state: dict[str, JsonValue]
    transcript: tuple[Event, ...] = ()
    public_chat_transcript: tuple[Event, ...] = ()
    requested_resources: dict[str, JsonValue] = field(default_factory=dict)
    other_player_ids: tuple[str, ...] = ()
    proposals: tuple[TradeChatProposal, ...] = ()
    game_rules: str | None = None
    memory: PlayerMemory = field(default_factory=PlayerMemory)
    message_char_limit: int = 500

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "game_id": self.game_id,
            "player_id": self.player_id,
            "owner_player_id": self.owner_player_id,
            "history_index": self.history_index,
            "turn_index": self.turn_index,
            "phase": self.phase,
            "decision_index": self.decision_index,
            "stage": self.stage,
            "attempt_index": self.attempt_index,
            "round_index": self.round_index,
            "public_state": self.public_state,
            "private_state": self.private_state,
            "transcript": [event.to_dict() for event in self.transcript],
            "public_chat_transcript": [
                event.to_dict() for event in self.public_chat_transcript
            ],
            "requested_resources": self.requested_resources,
            "other_player_ids": list(self.other_player_ids),
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "game_rules": self.game_rules,
            "memory": self.memory.to_dict(),
            "message_char_limit": self.message_char_limit,
        }


@dataclass(frozen=True, slots=True)
class TurnStartResponse:
    short_term: JsonValue | None = None
    public_chat: PublicChatDraft | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {"short_term": self.short_term}
        if self.public_chat is not None:
            payload["public_chat"] = self.public_chat.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class OpeningStrategyResponse:
    long_term: JsonValue | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        return {"long_term": self.long_term}


@dataclass(frozen=True, slots=True)
class ActionDecision:
    action: Action
    short_term: JsonValue | None = None
    public_chat: PublicChatDraft | None = None
    reasoning: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {"action": self.action.to_dict()}
        if self.short_term is not None:
            payload["short_term"] = self.short_term
        if self.public_chat is not None:
            payload["public_chat"] = self.public_chat.to_dict()
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        return payload


@dataclass(frozen=True, slots=True)
class ActionTraceEntry:
    acting_player_id: str
    turn_index: int
    phase: str
    decision_index: int
    action: Action

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "acting_player_id": self.acting_player_id,
            "turn_index": self.turn_index,
            "phase": self.phase,
            "decision_index": self.decision_index,
            "action": self.action.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "ActionTraceEntry":
        return cls(
            acting_player_id=str(data["acting_player_id"]),
            turn_index=int(data.get("turn_index", 0)),
            phase=str(data.get("phase", "unknown")),
            decision_index=int(data.get("decision_index", 0)),
            action=Action.from_dict(dict(data.get("action") or {})),
        )


@dataclass(frozen=True, slots=True)
class TurnEndResponse:
    long_term: JsonValue | None = None
    public_chat: PublicChatDraft | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {"long_term": self.long_term}
        if self.public_chat is not None:
            payload["public_chat"] = self.public_chat.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class PostGameChatResponse:
    public_chat: PublicChatDraft | None = None
    reasoning: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {}
        if self.public_chat is not None:
            payload["public_chat"] = self.public_chat.to_dict()
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        return payload


@dataclass(frozen=True, slots=True)
class TradeChatOpenResponse:
    open_chat: bool = False
    message: str | None = None
    requested_resources: dict[str, JsonValue] = field(default_factory=dict)
    reasoning: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "open_chat": self.open_chat,
            "requested_resources": self.requested_resources,
        }
        if self.message is not None:
            payload["message"] = self.message
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        return payload


@dataclass(frozen=True, slots=True)
class TradeChatReplyResponse:
    message: str | None = None
    owner_gives: dict[str, JsonValue] = field(default_factory=dict)
    owner_gets: dict[str, JsonValue] = field(default_factory=dict)
    reasoning: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "owner_gives": self.owner_gives,
            "owner_gets": self.owner_gets,
        }
        if self.message is not None:
            payload["message"] = self.message
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        return payload


@dataclass(frozen=True, slots=True)
class TradeChatOwnerDecisionResponse:
    decision: str = "close"
    selected_proposal_id: str | None = None
    message: str | None = None
    reasoning: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {"decision": self.decision}
        if self.selected_proposal_id is not None:
            payload["selected_proposal_id"] = self.selected_proposal_id
        if self.message is not None:
            payload["message"] = self.message
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        return payload


@dataclass(frozen=True, slots=True)
class TransitionResult:
    public_events: tuple[Event, ...] = ()
    terminal: bool = False
    result_metadata: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "public_events": [event.to_dict() for event in self.public_events],
            "terminal": self.terminal,
            "result_metadata": self.result_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "TransitionResult":
        return cls(
            public_events=tuple(
                Event.from_dict(event) for event in data.get("public_events", ())
            ),
            terminal=bool(data.get("terminal", False)),
            result_metadata=dict(data.get("result_metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class PromptTraceAttempt:
    messages: tuple[dict[str, JsonValue], ...]
    response: dict[str, JsonValue]
    response_text: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "messages": [dict(message) for message in self.messages],
            "response": self.response,
        }
        if self.response_text is not None:
            payload["response_text"] = self.response_text
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "PromptTraceAttempt":
        return cls(
            messages=tuple(
                dict(message)
                for message in data.get("messages", ())
                if isinstance(message, dict)
            ),
            response=dict(data.get("response") or {}),
            response_text=str(data["response_text"])
            if data.get("response_text") is not None
            else None,
        )


@dataclass(frozen=True, slots=True)
class PromptTrace:
    player_id: str
    history_index: int
    turn_index: int
    phase: str
    decision_index: int
    stage: str
    model: str
    temperature: float
    attempts: tuple[PromptTraceAttempt, ...]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "player_id": self.player_id,
            "history_index": self.history_index,
            "turn_index": self.turn_index,
            "phase": self.phase,
            "decision_index": self.decision_index,
            "stage": self.stage,
            "model": self.model,
            "temperature": self.temperature,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "PromptTrace":
        return cls(
            player_id=str(data["player_id"]),
            history_index=int(data.get("history_index", 0)),
            turn_index=int(data.get("turn_index", 0)),
            phase=str(data.get("phase", "unknown")),
            decision_index=int(data.get("decision_index", 0)),
            stage=str(data.get("stage", "unknown")),
            model=str(data.get("model", "")),
            temperature=float(data.get("temperature", 0.0)),
            attempts=tuple(
                PromptTraceAttempt.from_dict(attempt)
                for attempt in data.get("attempts", ())
                if isinstance(attempt, dict)
            ),
        )


@dataclass(frozen=True, slots=True)
class GameResult:
    game_id: str
    winner_ids: tuple[str, ...]
    total_decisions: int
    public_event_count: int
    memory_writes: int
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "game_id": self.game_id,
            "winner_ids": list(self.winner_ids),
            "total_decisions": self.total_decisions,
            "public_event_count": self.public_event_count,
            "memory_writes": self.memory_writes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "GameResult":
        return cls(
            game_id=str(data["game_id"]),
            winner_ids=tuple(
                str(player_id) for player_id in data.get("winner_ids", ())
            ),
            total_decisions=int(data["total_decisions"]),
            public_event_count=int(data["public_event_count"]),
            memory_writes=int(data["memory_writes"]),
            metadata=dict(data.get("metadata") or {}),
        )
