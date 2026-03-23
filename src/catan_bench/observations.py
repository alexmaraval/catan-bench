from __future__ import annotations

from typing import TYPE_CHECKING

from .prompts import CATAN_RULES_SUMMARY
from .schemas import Observation

if TYPE_CHECKING:
    from .engine import EngineAdapter
    from .storage import EventLog, MemoryStore
    from .schemas import DecisionPoint


class ObservationBuilder:
    """Builds a player-scoped observation from the engine and stored logs."""

    def __init__(
        self,
        recent_event_window: int | None = 25,
        *,
        game_rules: str = CATAN_RULES_SUMMARY,
    ) -> None:
        self.recent_event_window = recent_event_window
        self.game_rules = game_rules

    def build(
        self,
        *,
        engine: EngineAdapter,
        decision: DecisionPoint,
        event_log: EventLog,
        memory_store: MemoryStore,
    ) -> Observation:
        player_id = decision.acting_player_id
        public_history = event_log.recent_public(None)
        private_history = event_log.recent_private(player_id, None)
        recent_public_events = event_log.recent_public(self.recent_event_window)
        recent_private_events = event_log.recent_private(
            player_id,
            self.recent_event_window,
        )
        return Observation(
            game_id=engine.game_id,
            player_id=player_id,
            turn_index=decision.turn_index,
            phase=decision.phase,
            decision_index=decision.decision_index,
            public_state=dict(engine.public_state()),
            private_state=dict(engine.private_state(player_id)),
            game_rules=self.game_rules,
            decision_prompt=decision.prompt,
            public_history=public_history,
            private_history=private_history,
            recent_public_events=recent_public_events,
            recent_private_events=recent_private_events,
            legal_actions=tuple(decision.legal_actions),
            memory=memory_store.get(player_id),
        )
