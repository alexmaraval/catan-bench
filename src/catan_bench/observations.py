from __future__ import annotations

from typing import TYPE_CHECKING

from .schemas import Observation

if TYPE_CHECKING:
    from .engine import EngineAdapter
    from .storage import EventLog, MemoryStore
    from .schemas import DecisionPoint


class ObservationBuilder:
    """Builds a player-scoped observation from the engine and stored logs."""

    def __init__(self, recent_event_window: int | None = 25) -> None:
        self.recent_event_window = recent_event_window

    def build(
        self,
        *,
        engine: EngineAdapter,
        decision: DecisionPoint,
        event_log: EventLog,
        memory_store: MemoryStore,
    ) -> Observation:
        player_id = decision.acting_player_id
        return Observation(
            game_id=engine.game_id,
            player_id=player_id,
            turn_index=decision.turn_index,
            phase=decision.phase,
            decision_index=decision.decision_index,
            public_state=dict(engine.public_state()),
            private_state=dict(engine.private_state(player_id)),
            recent_public_events=event_log.recent_public(self.recent_event_window),
            recent_private_events=event_log.recent_private(
                player_id, self.recent_event_window
            ),
            legal_actions=tuple(decision.legal_actions),
            memory=memory_store.get(player_id),
        )
