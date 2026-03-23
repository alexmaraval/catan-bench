from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TextIO

from .schemas import Event, JsonValue, MemoryEntry, PromptTrace


def write_json(path: Path, payload: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _open_jsonl_handle(path: Path, *, truncate: bool) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if truncate else "a"
    return path.open(mode, encoding="utf-8")


def _write_jsonl(handle: TextIO, payload: dict[str, JsonValue]) -> None:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()


class EventLog:
    """Append-only public/private event log with optional JSONL persistence."""

    def __init__(self, run_dir: str | Path | None = None) -> None:
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.public_events: list[Event] = []
        self.private_events_by_player: dict[str, list[Event]] = {}
        self._public_handle: TextIO | None = None
        self._private_handles: dict[str, TextIO] = {}

    def reset(self, player_ids: Iterable[str]) -> None:
        player_ids = tuple(player_ids)
        self.close()
        self.public_events = []
        self.private_events_by_player = {player_id: [] for player_id in player_ids}

        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._public_handle = _open_jsonl_handle(
                self.run_dir / "public_history.jsonl",
                truncate=True,
            )
            for player_id in player_ids:
                players_dir = self.run_dir / "players" / player_id
                players_dir.mkdir(parents=True, exist_ok=True)
                self._private_handles[player_id] = _open_jsonl_handle(
                    players_dir / "private_history.jsonl",
                    truncate=True,
                )

    def append_public(self, events: Iterable[Event]) -> None:
        for event in events:
            self.public_events.append(event)
            if self.run_dir is not None:
                handle = self._public_handle
                if handle is None:
                    handle = _open_jsonl_handle(
                        self.run_dir / "public_history.jsonl",
                        truncate=False,
                    )
                    self._public_handle = handle
                _write_jsonl(handle, event.to_dict())

    def append_private(self, player_id: str, events: Iterable[Event]) -> None:
        player_events = self.private_events_by_player.setdefault(player_id, [])
        for event in events:
            player_events.append(event)
            if self.run_dir is not None:
                handle = self._private_handles.get(player_id)
                if handle is None:
                    handle = _open_jsonl_handle(
                        self.run_dir / "players" / player_id / "private_history.jsonl",
                        truncate=False,
                    )
                    self._private_handles[player_id] = handle
                _write_jsonl(handle, event.to_dict())

    def recent_public(self, limit: int | None = None) -> tuple[Event, ...]:
        if limit is None:
            return tuple(self.public_events)
        return tuple(self.public_events[-limit:])

    def recent_private(self, player_id: str, limit: int | None = None) -> tuple[Event, ...]:
        events = self.private_events_by_player.get(player_id, [])
        if limit is None:
            return tuple(events)
        return tuple(events[-limit:])

    def close(self) -> None:
        if self._public_handle is not None:
            self._public_handle.close()
            self._public_handle = None
        for handle in self._private_handles.values():
            handle.close()
        self._private_handles = {}

    def __del__(self) -> None:  # pragma: no cover - cleanup best effort only.
        self.close()


class MemoryStore:
    """Append-only private memory store with optional JSONL persistence."""

    def __init__(self, run_dir: str | Path | None = None) -> None:
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self._entries_by_player: dict[str, list[MemoryEntry]] = {}
        self._handles_by_player: dict[str, TextIO] = {}

    def reset(self, player_ids: Iterable[str]) -> None:
        player_ids = tuple(player_ids)
        self.close()
        self._entries_by_player = {player_id: [] for player_id in player_ids}

        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            for player_id in player_ids:
                players_dir = self.run_dir / "players" / player_id
                players_dir.mkdir(parents=True, exist_ok=True)
                self._handles_by_player[player_id] = _open_jsonl_handle(
                    players_dir / "memory.jsonl",
                    truncate=True,
                )

    def append(self, entry: MemoryEntry) -> None:
        self._entries_by_player.setdefault(entry.player_id, []).append(entry)
        if self.run_dir is not None:
            handle = self._handles_by_player.get(entry.player_id)
            if handle is None:
                handle = _open_jsonl_handle(
                    self.run_dir / "players" / entry.player_id / "memory.jsonl",
                    truncate=False,
                )
                self._handles_by_player[entry.player_id] = handle
            _write_jsonl(handle, entry.to_dict())

    def get(self, player_id: str) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries_by_player.get(player_id, []))

    def count(self) -> int:
        return sum(len(entries) for entries in self._entries_by_player.values())

    def close(self) -> None:
        for handle in self._handles_by_player.values():
            handle.close()
        self._handles_by_player = {}

    def __del__(self) -> None:  # pragma: no cover - cleanup best effort only.
        self.close()


class PromptTraceStore:
    """Append-only prompt/response trace store with optional JSONL persistence."""

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
                players_dir = self.run_dir / "players" / player_id
                players_dir.mkdir(parents=True, exist_ok=True)
                self._handles_by_player[player_id] = _open_jsonl_handle(
                    players_dir / "prompt_trace.jsonl",
                    truncate=True,
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

    def __del__(self) -> None:  # pragma: no cover - cleanup best effort only.
        self.close()
