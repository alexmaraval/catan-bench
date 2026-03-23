from __future__ import annotations

from typing import TYPE_CHECKING

from .prompts import CATAN_RULES_SUMMARY
from .schemas import Observation, RecallObservation, ReflectionObservation

if TYPE_CHECKING:
    from .engine import EngineAdapter
    from .storage import EventLog, MemoryStore
    from .schemas import DecisionPoint


class ObservationBuilder:
    """Builds player-scoped observations for recall, action, and reflection phases."""

    def __init__(
        self,
        recent_event_window: int | None = 25,
        *,
        game_rules: str = CATAN_RULES_SUMMARY,
    ) -> None:
        self.recent_event_window = recent_event_window
        self.game_rules = game_rules

    def build_action(
        self,
        *,
        engine: EngineAdapter,
        decision: DecisionPoint,
        memory_store: MemoryStore,
    ) -> Observation:
        player_id = decision.acting_player_id
        return Observation(
            game_id=engine.game_id,
            player_id=player_id,
            turn_index=decision.turn_index,
            phase=decision.phase,
            decision_index=decision.decision_index,
            public_state=self._public_state_for_decision(
                engine=engine,
                decision=decision,
                player_id=player_id,
            ),
            private_state=self._private_state_for_decision(
                engine=engine,
                decision=decision,
                player_id=player_id,
            ),
            game_rules=self.game_rules,
            decision_prompt=decision.prompt,
            legal_actions=tuple(decision.legal_actions),
            memory=memory_store.get(player_id),
        )

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
            public_state=self._public_state_for_decision(
                engine=engine,
                decision=decision,
                player_id=player_id,
            ),
            private_state=self._private_state_for_decision(
                engine=engine,
                decision=decision,
                player_id=player_id,
            ),
            game_rules=self.game_rules,
            decision_prompt=decision.prompt,
            public_history=event_log.recent_public(None),
            private_history=event_log.recent_private(player_id, None),
            recent_public_events=event_log.recent_public(self.recent_event_window),
            recent_private_events=event_log.recent_private(
                player_id,
                self.recent_event_window,
            ),
            legal_actions=tuple(decision.legal_actions),
            memory=memory_store.get(player_id),
        )

    def build_recall(
        self,
        *,
        engine: EngineAdapter,
        decision: DecisionPoint,
        event_log: EventLog,
        memory_store: MemoryStore,
        public_event_start: int,
        private_event_start: int,
    ) -> RecallObservation:
        player_id = decision.acting_player_id
        public_events = event_log.recent_public(None)[public_event_start:]
        private_events = event_log.recent_private(player_id, None)[private_event_start:]
        return RecallObservation(
            game_id=engine.game_id,
            player_id=player_id,
            turn_index=decision.turn_index,
            phase=decision.phase,
            decision_index=decision.decision_index,
            game_rules=self.game_rules,
            public_events_since_last_turn=public_events,
            private_events_since_last_turn=private_events,
            memory=memory_store.get(player_id),
        )

    def build_reflection(
        self,
        *,
        engine: EngineAdapter,
        player_id: str,
        turn_index: int,
        phase: str,
        decision_index: int,
        event_log: EventLog,
        memory_store: MemoryStore,
        public_event_start: int,
        private_event_start: int,
    ) -> ReflectionObservation:
        public_events = event_log.recent_public(None)[public_event_start:]
        private_events = event_log.recent_private(player_id, None)[private_event_start:]
        return ReflectionObservation(
            game_id=engine.game_id,
            player_id=player_id,
            turn_index=turn_index,
            phase=phase,
            decision_index=decision_index,
            game_rules=self.game_rules,
            public_events_this_turn=public_events,
            private_events_this_turn=private_events,
            memory=memory_store.get(player_id),
        )

    @staticmethod
    def _public_state_for_decision(
        *,
        engine: EngineAdapter,
        decision: DecisionPoint,
        player_id: str,
    ) -> dict:
        builder = getattr(engine, "public_state_for_decision", None)
        if callable(builder):
            return dict(
                builder(
                    player_id=player_id,
                    phase=decision.phase,
                    legal_actions=tuple(decision.legal_actions),
                )
            )
        return dict(engine.public_state())

    @staticmethod
    def _private_state_for_decision(
        *,
        engine: EngineAdapter,
        decision: DecisionPoint,
        player_id: str,
    ) -> dict:
        builder = getattr(engine, "private_state_for_decision", None)
        if callable(builder):
            return dict(
                builder(
                    player_id=player_id,
                    phase=decision.phase,
                    legal_actions=tuple(decision.legal_actions),
                )
            )
        return dict(engine.private_state(player_id))
