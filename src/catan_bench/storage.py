from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TextIO

from .schemas import (
    ActionTraceEntry,
    Event,
    JsonValue,
    MemorySnapshot,
    PlayerMemory,
    PromptTrace,
    PublicStateSnapshot,
)


def write_json(path: Path, payload: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, JsonValue] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _open_jsonl_handle(path: Path, *, truncate: bool) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w" if truncate else "a", encoding="utf-8")


def _write_jsonl(handle: TextIO, payload: dict[str, JsonValue]) -> None:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()


def read_jsonl(path: Path) -> list[dict[str, JsonValue]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class EventLog:
    """Append-only public event log with optional JSONL persistence."""

    def __init__(self, run_dir: str | Path | None = None) -> None:
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.public_events: list[Event] = []
        self._handle: TextIO | None = None

    @property
    def current_history_index(self) -> int:
        return len(self.public_events)

    def reset(self) -> None:
        self.close()
        self.public_events = []
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._handle = _open_jsonl_handle(self.run_dir / "public_history.jsonl", truncate=True)

    def hydrate(self) -> None:
        self.close()
        self.public_events = []
        if self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.public_events = [
            Event.from_dict(entry)
            for entry in read_jsonl(self.run_dir / "public_history.jsonl")
        ]
        self._handle = _open_jsonl_handle(self.run_dir / "public_history.jsonl", truncate=False)

    def append(self, events: Iterable[Event]) -> tuple[Event, ...]:
        stored_events: list[Event] = []
        for event in events:
            stored_event = Event(
                kind=event.kind,
                payload=dict(event.payload),
                history_index=self.current_history_index + 1,
                turn_index=event.turn_index,
                phase=event.phase,
                decision_index=event.decision_index,
                actor_player_id=event.actor_player_id,
            )
            self.public_events.append(stored_event)
            stored_events.append(stored_event)
            if self.run_dir is not None:
                handle = self._handle
                if handle is None:
                    handle = _open_jsonl_handle(
                        self.run_dir / "public_history.jsonl",
                        truncate=False,
                    )
                    self._handle = handle
                _write_jsonl(handle, stored_event.to_dict())
        return tuple(stored_events)

    def recent(self, limit: int | None = None) -> tuple[Event, ...]:
        if limit is None:
            return tuple(self.public_events)
        return tuple(self.public_events[-limit:])

    def since(self, history_index: int, limit: int | None = None) -> tuple[Event, ...]:
        events = tuple(event for event in self.public_events if event.history_index > history_index)
        if limit is None:
            return events
        return events[-limit:]

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - best effort only.
        self.close()


class PublicStateStore:
    """Append-only public state snapshots keyed by history index."""

    def __init__(self, run_dir: str | Path | None = None) -> None:
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.snapshots: list[PublicStateSnapshot] = []
        self._handle: TextIO | None = None

    def reset(self, initial_snapshot: PublicStateSnapshot) -> None:
        self.close()
        self.snapshots = [initial_snapshot]
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._handle = _open_jsonl_handle(
                self.run_dir / "public_state_trace.jsonl",
                truncate=True,
            )
            _write_jsonl(self._handle, initial_snapshot.to_dict())

    def hydrate(self) -> None:
        self.close()
        self.snapshots = []
        if self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots = [
            PublicStateSnapshot.from_dict(entry)
            for entry in read_jsonl(self.run_dir / "public_state_trace.jsonl")
        ]
        self._handle = _open_jsonl_handle(
            self.run_dir / "public_state_trace.jsonl",
            truncate=False,
        )

    def append(self, snapshot: PublicStateSnapshot) -> None:
        self.snapshots.append(snapshot)
        if self.run_dir is not None:
            handle = self._handle
            if handle is None:
                handle = _open_jsonl_handle(
                    self.run_dir / "public_state_trace.jsonl",
                    truncate=False,
                )
                self._handle = handle
            _write_jsonl(handle, snapshot.to_dict())

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - best effort only.
        self.close()


class MemoryStore:
    """Per-player two-slot memory snapshots keyed by history index."""

    def __init__(self, run_dir: str | Path | None = None) -> None:
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self._current_by_player: dict[str, PlayerMemory] = {}
        self._history_by_player: dict[str, list[MemorySnapshot]] = {}
        self._trace_handles_by_player: dict[str, TextIO] = {}

    def reset(self, player_ids: Iterable[str]) -> None:
        player_ids = tuple(player_ids)
        self.close()
        self._current_by_player = {player_id: PlayerMemory() for player_id in player_ids}
        self._history_by_player = {player_id: [] for player_id in player_ids}
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            for player_id in player_ids:
                player_dir = self.run_dir / "players" / player_id
                player_dir.mkdir(parents=True, exist_ok=True)
                write_json(player_dir / "memory.json", {"memory": PlayerMemory().to_dict()})
                self._trace_handles_by_player[player_id] = _open_jsonl_handle(
                    player_dir / "memory_trace.jsonl",
                    truncate=True,
                )

    def hydrate(self, player_ids: Iterable[str]) -> None:
        player_ids = tuple(player_ids)
        self.close()
        self._current_by_player = {}
        self._history_by_player = {}
        if self.run_dir is None:
            for player_id in player_ids:
                self._current_by_player[player_id] = PlayerMemory()
                self._history_by_player[player_id] = []
            return

        self.run_dir.mkdir(parents=True, exist_ok=True)
        for player_id in player_ids:
            player_dir = self.run_dir / "players" / player_id
            player_dir.mkdir(parents=True, exist_ok=True)
            history = [
                MemorySnapshot.from_dict(entry)
                for entry in read_jsonl(player_dir / "memory_trace.jsonl")
            ]
            self._history_by_player[player_id] = history
            if history:
                current = history[-1].memory
            else:
                payload = read_json(player_dir / "memory.json") or {}
                raw_memory = payload.get("memory")
                current = PlayerMemory.from_dict(raw_memory if isinstance(raw_memory, dict) else None)
            self._current_by_player[player_id] = current
            self._trace_handles_by_player[player_id] = _open_jsonl_handle(
                player_dir / "memory_trace.jsonl",
                truncate=False,
            )

    def get(self, player_id: str) -> PlayerMemory:
        return self._current_by_player.get(player_id, PlayerMemory())

    def write(
        self,
        *,
        player_id: str,
        memory: PlayerMemory,
        history_index: int,
        turn_index: int,
        phase: str,
        decision_index: int,
        stage: str,
    ) -> MemorySnapshot:
        snapshot = MemorySnapshot(
            player_id=player_id,
            history_index=history_index,
            turn_index=turn_index,
            phase=phase,
            decision_index=decision_index,
            stage=stage,
            memory=memory,
        )
        self._current_by_player[player_id] = memory
        self._history_by_player.setdefault(player_id, []).append(snapshot)
        if self.run_dir is not None:
            player_dir = self.run_dir / "players" / player_id
            write_json(player_dir / "memory.json", snapshot.to_dict())
            handle = self._trace_handles_by_player.get(player_id)
            if handle is None:
                handle = _open_jsonl_handle(player_dir / "memory_trace.jsonl", truncate=False)
                self._trace_handles_by_player[player_id] = handle
            _write_jsonl(handle, snapshot.to_dict())
        return snapshot

    def set_short_term(
        self,
        *,
        player_id: str,
        short_term: JsonValue | None,
        history_index: int,
        turn_index: int,
        phase: str,
        decision_index: int,
        stage: str,
    ) -> MemorySnapshot:
        current = self.get(player_id)
        return self.write(
            player_id=player_id,
            memory=PlayerMemory(long_term=current.long_term, short_term=short_term),
            history_index=history_index,
            turn_index=turn_index,
            phase=phase,
            decision_index=decision_index,
            stage=stage,
        )

    def set_long_term(
        self,
        *,
        player_id: str,
        long_term: JsonValue | None,
        history_index: int,
        turn_index: int,
        phase: str,
        decision_index: int,
        stage: str,
    ) -> MemorySnapshot:
        current = self.get(player_id)
        return self.write(
            player_id=player_id,
            memory=PlayerMemory(long_term=long_term, short_term=current.short_term),
            history_index=history_index,
            turn_index=turn_index,
            phase=phase,
            decision_index=decision_index,
            stage=stage,
        )

    def clear_short_term(
        self,
        *,
        player_id: str,
        history_index: int,
        turn_index: int,
        phase: str,
        decision_index: int,
        stage: str,
    ) -> MemorySnapshot:
        current = self.get(player_id)
        return self.write(
            player_id=player_id,
            memory=PlayerMemory(long_term=current.long_term, short_term=None),
            history_index=history_index,
            turn_index=turn_index,
            phase=phase,
            decision_index=decision_index,
            stage=stage,
        )

    def history(self, player_id: str) -> tuple[MemorySnapshot, ...]:
        return tuple(self._history_by_player.get(player_id, ()))

    def count(self) -> int:
        return sum(len(snapshots) for snapshots in self._history_by_player.values())

    def close(self) -> None:
        for handle in self._trace_handles_by_player.values():
            handle.close()
        self._trace_handles_by_player = {}

    def __del__(self) -> None:  # pragma: no cover - best effort only.
        self.close()


class PromptTraceStore:
    """Append-only prompt/response traces keyed by player."""

    def __init__(self, run_dir: str | Path | None = None) -> None:
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self._traces_by_player: dict[str, list[PromptTrace]] = {}
        self._handles_by_player: dict[str, TextIO] = {}

    def reset(self, player_ids: Iterable[str]) -> None:
        player_ids = tuple(player_ids)
        self.close()
        self._traces_by_player = {player_id: [] for player_id in player_ids}
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            for player_id in player_ids:
                player_dir = self.run_dir / "players" / player_id
                player_dir.mkdir(parents=True, exist_ok=True)
                self._handles_by_player[player_id] = _open_jsonl_handle(
                    player_dir / "prompt_trace.jsonl",
                    truncate=True,
                )

    def hydrate(self, player_ids: Iterable[str]) -> None:
        player_ids = tuple(player_ids)
        self.close()
        self._traces_by_player = {}
        if self.run_dir is None:
            for player_id in player_ids:
                self._traces_by_player[player_id] = []
            return

        self.run_dir.mkdir(parents=True, exist_ok=True)
        for player_id in player_ids:
            player_dir = self.run_dir / "players" / player_id
            player_dir.mkdir(parents=True, exist_ok=True)
            self._traces_by_player[player_id] = [
                PromptTrace.from_dict(entry)
                for entry in read_jsonl(player_dir / "prompt_trace.jsonl")
            ]
            self._handles_by_player[player_id] = _open_jsonl_handle(
                player_dir / "prompt_trace.jsonl",
                truncate=False,
            )

    def append(self, trace: PromptTrace) -> None:
        self._traces_by_player.setdefault(trace.player_id, []).append(trace)
        if self.run_dir is not None:
            handle = self._handles_by_player.get(trace.player_id)
            if handle is None:
                handle = _open_jsonl_handle(
                    self.run_dir / "players" / trace.player_id / "prompt_trace.jsonl",
                    truncate=False,
                )
                self._handles_by_player[trace.player_id] = handle
            _write_jsonl(handle, trace.to_dict())

    def get(self, player_id: str) -> tuple[PromptTrace, ...]:
        return tuple(self._traces_by_player.get(player_id, ()))

    def close(self) -> None:
        for handle in self._handles_by_player.values():
            handle.close()
        self._handles_by_player = {}

    def __del__(self) -> None:  # pragma: no cover - best effort only.
        self.close()


class ActionTraceStore:
    """Append-only canonical action trace for deterministic replay/resume."""

    def __init__(self, run_dir: str | Path | None = None) -> None:
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.entries: list[ActionTraceEntry] = []
        self._handle: TextIO | None = None

    def reset(self) -> None:
        self.close()
        self.entries = []
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._handle = _open_jsonl_handle(self.run_dir / "action_trace.jsonl", truncate=True)

    def hydrate(self) -> None:
        self.close()
        self.entries = []
        if self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.entries = [
            ActionTraceEntry.from_dict(entry)
            for entry in read_jsonl(self.run_dir / "action_trace.jsonl")
        ]
        self._handle = _open_jsonl_handle(self.run_dir / "action_trace.jsonl", truncate=False)

    def append(self, entry: ActionTraceEntry) -> None:
        self.entries.append(entry)
        if self.run_dir is not None:
            handle = self._handle
            if handle is None:
                handle = _open_jsonl_handle(self.run_dir / "action_trace.jsonl", truncate=False)
                self._handle = handle
            _write_jsonl(handle, entry.to_dict())

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - best effort only.
        self.close()
