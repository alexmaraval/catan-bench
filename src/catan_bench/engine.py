from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from .schemas import Action, DecisionPoint, JsonValue, TransitionResult


class EngineAdapter(Protocol):
    """Minimal interface the orchestrator expects from a game engine."""

    @property
    def game_id(self) -> str: ...

    @property
    def player_ids(self) -> Sequence[str]: ...

    def is_terminal(self) -> bool: ...

    def current_decision(self) -> DecisionPoint:
        """Return the current decision point for the next acting player."""

    def public_state(self) -> Mapping[str, JsonValue]:
        """Return the public state visible to all players."""

    def private_state(self, player_id: str) -> Mapping[str, JsonValue]:
        """Return the private state visible only to one player."""

    def apply_action(self, action: Action) -> TransitionResult:
        """Apply a validated canonical action to the underlying engine."""

    def result(self) -> Mapping[str, JsonValue]:
        """Return final structured metadata once the game reaches a terminal state."""
