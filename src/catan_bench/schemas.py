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
        data: dict[str, JsonValue] = {
            "action_type": self.action_type,
            "payload": self.payload,
        }
        if self.description is not None:
            data["description"] = self.description
        return data

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "Action":
        return cls(
            action_type=str(data["action_type"]),
            payload=dict(data.get("payload") or {}),
            description=data.get("description") if "description" in data else None,
        )


@dataclass(frozen=True, slots=True)
class Event:
    kind: str
    payload: dict[str, JsonValue] = field(default_factory=dict)
    turn_index: int = 0
    phase: str = "unknown"
    decision_index: int | None = None
    actor_player_id: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        data: dict[str, JsonValue] = {
            "kind": self.kind,
            "payload": self.payload,
            "turn_index": self.turn_index,
            "phase": self.phase,
        }
        if self.decision_index is not None:
            data["decision_index"] = self.decision_index
        if self.actor_player_id is not None:
            data["actor_player_id"] = self.actor_player_id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "Event":
        return cls(
            kind=str(data["kind"]),
            payload=dict(data.get("payload") or {}),
            turn_index=int(data.get("turn_index", 0)),
            phase=str(data.get("phase", "unknown")),
            decision_index=data.get("decision_index"),
            actor_player_id=data.get("actor_player_id"),
        )


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    player_id: str
    content: JsonValue
    turn_index: int
    phase: str
    decision_index: int
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "player_id": self.player_id,
            "content": self.content,
            "turn_index": self.turn_index,
            "phase": self.phase,
            "decision_index": self.decision_index,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "MemoryEntry":
        tags = data.get("tags", ())
        return cls(
            player_id=str(data["player_id"]),
            content=data["content"],
            turn_index=int(data["turn_index"]),
            phase=str(data["phase"]),
            decision_index=int(data["decision_index"]),
            tags=tuple(str(tag) for tag in tags) if tags else (),
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
        data: dict[str, JsonValue] = {
            "acting_player_id": self.acting_player_id,
            "turn_index": self.turn_index,
            "phase": self.phase,
            "decision_index": self.decision_index,
            "legal_actions": [action.to_dict() for action in self.legal_actions],
        }
        if self.prompt is not None:
            data["prompt"] = self.prompt
        return data

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
            prompt=data.get("prompt") if "prompt" in data else None,
        )


@dataclass(frozen=True, slots=True)
class Observation:
    game_id: str
    player_id: str
    turn_index: int
    phase: str
    decision_index: int
    public_state: dict[str, JsonValue]
    private_state: dict[str, JsonValue]
    recent_public_events: tuple[Event, ...] = ()
    recent_private_events: tuple[Event, ...] = ()
    legal_actions: tuple[Action, ...] = ()
    memory: tuple[MemoryEntry, ...] = ()

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "game_id": self.game_id,
            "player_id": self.player_id,
            "turn_index": self.turn_index,
            "phase": self.phase,
            "decision_index": self.decision_index,
            "public_state": self.public_state,
            "private_state": self.private_state,
            "recent_public_events": [event.to_dict() for event in self.recent_public_events],
            "recent_private_events": [
                event.to_dict() for event in self.recent_private_events
            ],
            "legal_actions": [action.to_dict() for action in self.legal_actions],
            "memory": [entry.to_dict() for entry in self.memory],
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "Observation":
        return cls(
            game_id=str(data["game_id"]),
            player_id=str(data["player_id"]),
            turn_index=int(data["turn_index"]),
            phase=str(data["phase"]),
            decision_index=int(data["decision_index"]),
            public_state=dict(data.get("public_state") or {}),
            private_state=dict(data.get("private_state") or {}),
            recent_public_events=tuple(
                Event.from_dict(e) for e in data.get("recent_public_events", ())
            ),
            recent_private_events=tuple(
                Event.from_dict(e) for e in data.get("recent_private_events", ())
            ),
            legal_actions=tuple(
                Action.from_dict(a) for a in data.get("legal_actions", ())
            ),
            memory=tuple(
                MemoryEntry.from_dict(m) for m in data.get("memory", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class PlayerResponse:
    action: Action
    memory_write: JsonValue | None = None
    summary: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        data: dict[str, JsonValue] = {"action": self.action.to_dict()}
        if self.memory_write is not None:
            data["memory_write"] = self.memory_write
        if self.summary is not None:
            data["summary"] = self.summary
        return data

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "PlayerResponse":
        return cls(
            action=Action.from_dict(data["action"]),
            memory_write=data.get("memory_write"),
            summary=data.get("summary"),
        )


@dataclass(frozen=True, slots=True)
class TransitionResult:
    public_events: tuple[Event, ...] = ()
    private_events_by_player: dict[str, tuple[Event, ...]] = field(default_factory=dict)
    terminal: bool = False
    result_metadata: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "public_events": [event.to_dict() for event in self.public_events],
            "private_events_by_player": {
                player_id: [event.to_dict() for event in events]
                for player_id, events in self.private_events_by_player.items()
            },
            "terminal": self.terminal,
            "result_metadata": self.result_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "TransitionResult":
        return cls(
            public_events=tuple(
                Event.from_dict(e) for e in data.get("public_events", ())
            ),
            private_events_by_player={
                str(pid): tuple(Event.from_dict(e) for e in events)
                for pid, events in (data.get("private_events_by_player") or {}).items()
            },
            terminal=bool(data.get("terminal", False)),
            result_metadata=dict(data.get("result_metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class GameResult:
    game_id: str
    winner_ids: tuple[str, ...]
    total_decisions: int
    public_event_count: int
    private_event_count: int
    memory_writes: int
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "game_id": self.game_id,
            "winner_ids": list(self.winner_ids),
            "total_decisions": self.total_decisions,
            "public_event_count": self.public_event_count,
            "private_event_count": self.private_event_count,
            "memory_writes": self.memory_writes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "GameResult":
        return cls(
            game_id=str(data["game_id"]),
            winner_ids=tuple(str(w) for w in data.get("winner_ids", ())),
            total_decisions=int(data["total_decisions"]),
            public_event_count=int(data["public_event_count"]),
            private_event_count=int(data["private_event_count"]),
            memory_writes=int(data["memory_writes"]),
            metadata=dict(data.get("metadata") or {}),
        )
