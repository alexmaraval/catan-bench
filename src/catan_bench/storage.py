from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schemas import Event, JsonValue, MemoryEntry


def write_json(path: Path, payload: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


class EventLog:
    """Append-only public/private event log with optional JSONL persistence."""

    def __init__(self, run_dir: str | Path | None = None) -> None:
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.public_events: list[Event] = []
        self.private_events_by_player: dict[str, list[Event]] = {}

    def reset(self, player_ids: Iterable[str]) -> None:
        player_ids = tuple(player_ids)
        self.public_events = []
        self.private_events_by_player = {player_id: [] for player_id in player_ids}

        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "public_history.jsonl").write_text("", encoding="utf-8")
            for player_id in player_ids:
                players_dir = self.run_dir / "players" / player_id
                players_dir.mkdir(parents=True, exist_ok=True)
                (players_dir / "private_history.jsonl").write_text("", encoding="utf-8")

    def append_public(self, events: Iterable[Event]) -> None:
        for event in events:
            self.public_events.append(event)
            if self.run_dir is not None:
                _append_jsonl(self.run_dir / "public_history.jsonl", event.to_dict())

    def append_private(self, player_id: str, events: Iterable[Event]) -> None:
        player_events = self.private_events_by_player.setdefault(player_id, [])
        for event in events:
            player_events.append(event)
            if self.run_dir is not None:
                _append_jsonl(
                    self.run_dir / "players" / player_id / "private_history.jsonl",
                    event.to_dict(),
                )

    def recent_public(self, limit: int | None = None) -> tuple[Event, ...]:
        if limit is None:
            return tuple(self.public_events)
        return tuple(self.public_events[-limit:])

    def recent_private(self, player_id: str, limit: int | None = None) -> tuple[Event, ...]:
        events = self.private_events_by_player.get(player_id, [])
        if limit is None:
            return tuple(events)
        return tuple(events[-limit:])


class MemoryStore:
    """Append-only private memory store with optional JSONL persistence."""

    def __init__(self, run_dir: str | Path | None = None) -> None:
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self._entries_by_player: dict[str, list[MemoryEntry]] = {}

    def reset(self, player_ids: Iterable[str]) -> None:
        player_ids = tuple(player_ids)
        self._entries_by_player = {player_id: [] for player_id in player_ids}

        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            for player_id in player_ids:
                players_dir = self.run_dir / "players" / player_id
                players_dir.mkdir(parents=True, exist_ok=True)
                (players_dir / "memory.jsonl").write_text("", encoding="utf-8")

    def append(self, entry: MemoryEntry) -> None:
        self._entries_by_player.setdefault(entry.player_id, []).append(entry)
        if self.run_dir is not None:
            _append_jsonl(
                self.run_dir / "players" / entry.player_id / "memory.jsonl",
                entry.to_dict(),
            )

    def get(self, player_id: str) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries_by_player.get(player_id, []))

    def count(self) -> int:
        return sum(len(entries) for entries in self._entries_by_player.values())
