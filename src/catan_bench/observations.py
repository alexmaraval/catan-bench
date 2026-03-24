from __future__ import annotations

from typing import TYPE_CHECKING

from .prompts import CATAN_RULES_SUMMARY
from .schemas import (
    ActionObservation,
    OpeningStrategyObservation,
    ReactiveObservation,
    TradeChatObservation,
    TradeChatQuote,
    TurnEndObservation,
    TurnStartObservation,
)

if TYPE_CHECKING:
    from .engine import EngineAdapter
    from .storage import EventLog, MemoryStore
    from .schemas import DecisionPoint


class ObservationBuilder:
    """Builds compact player-scoped observations for the simplified lifecycle."""

    def __init__(
        self,
        recent_event_window: int | None = 25,
        *,
        game_rules: str = CATAN_RULES_SUMMARY,
    ) -> None:
        self.recent_event_window = recent_event_window
        self.game_rules = game_rules

    def build_turn_start(
        self,
        *,
        engine: EngineAdapter,
        decision: DecisionPoint,
        event_log: EventLog,
        memory_store: MemoryStore,
        last_turn_history_index: int,
    ) -> TurnStartObservation:
        player_id = decision.acting_player_id
        return TurnStartObservation(
            game_id=engine.game_id,
            player_id=player_id,
            history_index=event_log.current_history_index,
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
            public_history_since_last_turn=self._tail_events(
                event_log.since(last_turn_history_index),
            ),
            game_rules=self.game_rules,
            memory=memory_store.get(player_id),
        )

    def build_opening_strategy(
        self,
        *,
        engine: EngineAdapter,
        player_id: str,
        turn_index: int,
        decision_index: int,
        event_log: EventLog,
        memory_store: MemoryStore,
    ) -> OpeningStrategyObservation:
        phase = "opening_strategy"
        return OpeningStrategyObservation(
            game_id=engine.game_id,
            player_id=player_id,
            history_index=event_log.current_history_index,
            turn_index=turn_index,
            phase=phase,
            decision_index=decision_index,
            public_state=self._compact_public_state(engine=engine, player_id=player_id, phase=phase),
            private_state=self._compact_private_state(engine=engine, player_id=player_id, phase=phase),
            public_history=self._tail_events(event_log.recent()),
            game_rules=self.game_rules,
            memory=memory_store.get(player_id),
        )

    def build_action(
        self,
        *,
        engine: EngineAdapter,
        decision: DecisionPoint,
        event_log: EventLog,
        memory_store: MemoryStore,
        turn_start_history_index: int,
    ) -> ActionObservation:
        player_id = decision.acting_player_id
        return ActionObservation(
            game_id=engine.game_id,
            player_id=player_id,
            history_index=event_log.current_history_index,
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
            public_history=self._tail_events(event_log.recent()),
            turn_public_events=event_log.since(turn_start_history_index),
            legal_actions=tuple(decision.legal_actions),
            decision_prompt=decision.prompt,
            game_rules=self.game_rules,
            memory=memory_store.get(player_id),
        )

    def build_turn_end(
        self,
        *,
        engine: EngineAdapter,
        player_id: str,
        turn_index: int,
        phase: str,
        decision_index: int,
        event_log: EventLog,
        memory_store: MemoryStore,
        turn_start_history_index: int,
    ) -> TurnEndObservation:
        return TurnEndObservation(
            game_id=engine.game_id,
            player_id=player_id,
            history_index=event_log.current_history_index,
            turn_index=turn_index,
            phase=phase,
            decision_index=decision_index,
            public_state=self._compact_public_state(engine=engine, player_id=player_id, phase=phase),
            private_state=self._compact_private_state(engine=engine, player_id=player_id, phase=phase),
            turn_public_events=event_log.since(turn_start_history_index),
            game_rules=self.game_rules,
            memory=memory_store.get(player_id),
        )

    def build_reactive(
        self,
        *,
        engine: EngineAdapter,
        decision: DecisionPoint,
        event_log: EventLog,
        memory_store: MemoryStore,
    ) -> ReactiveObservation:
        player_id = decision.acting_player_id
        return ReactiveObservation(
            game_id=engine.game_id,
            player_id=player_id,
            history_index=event_log.current_history_index,
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
            public_history=self._tail_events(event_log.recent()),
            legal_actions=tuple(decision.legal_actions),
            decision_prompt=decision.prompt,
            game_rules=self.game_rules,
            memory=memory_store.get(player_id),
        )

    def build_trade_chat(
        self,
        *,
        engine: EngineAdapter,
        player_id: str,
        owner_player_id: str,
        decision: DecisionPoint,
        stage: str,
        attempt_index: int,
        requested_resources: dict,
        quotes: tuple[TradeChatQuote, ...],
        event_log: EventLog,
        memory_store: MemoryStore,
        message_char_limit: int,
        transcript_limit: int | None,
    ) -> TradeChatObservation:
        transcript = tuple(
            event
            for event in self._tail_events(event_log.recent(transcript_limit))
            if event.turn_index == decision.turn_index
        )
        return TradeChatObservation(
            game_id=engine.game_id,
            player_id=player_id,
            owner_player_id=owner_player_id,
            history_index=event_log.current_history_index,
            turn_index=decision.turn_index,
            phase=decision.phase,
            decision_index=decision.decision_index,
            stage=stage,
            attempt_index=attempt_index,
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
            transcript=transcript,
            requested_resources=dict(requested_resources),
            other_player_ids=tuple(
                other_player_id
                for other_player_id in engine.player_ids
                if other_player_id != player_id
            ),
            quotes=quotes,
            game_rules=self.game_rules,
            memory=memory_store.get(player_id),
            message_char_limit=message_char_limit,
        )

    def _tail_events(self, events, limit: int | None = None):
        effective_limit = self.recent_event_window if limit is None else limit
        events = tuple(events)
        if effective_limit is None:
            return events
        if effective_limit <= 0:
            return ()
        return events[-effective_limit:]

    @staticmethod
    def _compact_public_state(*, engine: "EngineAdapter", player_id: str, phase: str) -> dict:
        """Compact public state for prompts that have no active decision (turn_end)."""
        builder = getattr(engine, "public_state_for_decision", None)
        if callable(builder):
            return dict(builder(player_id=player_id, phase=phase, legal_actions=()))
        return dict(engine.public_state())

    @staticmethod
    def _compact_private_state(*, engine: "EngineAdapter", player_id: str, phase: str) -> dict:
        """Compact private state for prompts that have no active decision (turn_end)."""
        builder = getattr(engine, "private_state_for_decision", None)
        if callable(builder):
            return dict(builder(player_id=player_id, phase=phase, legal_actions=()))
        return dict(engine.private_state(player_id))

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
