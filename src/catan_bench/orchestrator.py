from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

from .observations import ObservationBuilder
from .players import Player
from .schemas import Action, DecisionPoint, Event, GameResult, MemoryEntry, PlayerResponse
from .storage import EventLog, MemoryStore, PromptTraceStore, write_json


class MissingPlayerError(KeyError):
    """Raised when the orchestrator has no adapter for an engine player."""


class InvalidActionError(ValueError):
    """Raised when a player returns an action that is not currently legal."""


logger = logging.getLogger(__name__)


class GameOrchestrator:
    """Coordinates the engine, player adapters, and append-only logs."""

    def __init__(
        self,
        engine,
        players: Mapping[str, Player],
        *,
        observation_builder: ObservationBuilder | None = None,
        event_log: EventLog | None = None,
        memory_store: MemoryStore | None = None,
        prompt_trace_store: PromptTraceStore | None = None,
        run_dir: str | Path | None = None,
        max_decisions: int = 10_000,
    ) -> None:
        self.engine = engine
        self.players = dict(players)
        self.observation_builder = observation_builder or ObservationBuilder()
        self.event_log = event_log or EventLog(run_dir)
        self.memory_store = memory_store or MemoryStore(run_dir)
        self.prompt_trace_store = prompt_trace_store or PromptTraceStore(run_dir)
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.max_decisions = max_decisions
        self._decision_phase_counts: dict[str, int] = {}

    def run(self) -> GameResult:
        self._prepare_run()

        total_decisions = 0
        while not self.engine.is_terminal():
            if total_decisions >= self.max_decisions:
                logger.warning(
                    "Stopping game %s after %s decisions because max_decisions=%s was reached.",
                    self.engine.game_id,
                    total_decisions,
                    self.max_decisions,
                )
                raise RuntimeError(
                    f"Stopped after {self.max_decisions} decisions without a terminal state."
                )
            self.step()
            total_decisions += 1

        metadata = dict(self.engine.result())
        metadata["benchmark"] = self._benchmark_metadata(total_decisions=total_decisions)
        result = GameResult(
            game_id=self.engine.game_id,
            winner_ids=self._winner_ids_from_metadata(metadata),
            total_decisions=total_decisions,
            public_event_count=len(self.event_log.public_events),
            private_event_count=sum(
                len(events) for events in self.event_log.private_events_by_player.values()
            ),
            memory_writes=self.memory_store.count(),
            metadata=metadata,
        )

        if self.run_dir is not None:
            write_json(self.run_dir / "result.json", result.to_dict())

        return result

    def step(self):
        if self.engine.is_terminal():
            raise RuntimeError("Cannot step a terminal game.")

        decision = self.engine.current_decision()
        player = self.players.get(decision.acting_player_id)
        if player is None:
            raise MissingPlayerError(
                f"No player adapter registered for {decision.acting_player_id!r}."
            )

        observation = self.observation_builder.build(
            engine=self.engine,
            decision=decision,
            event_log=self.event_log,
            memory_store=self.memory_store,
        )
        self._decision_phase_counts[decision.phase] = (
            self._decision_phase_counts.get(decision.phase, 0) + 1
        )
        response = player.respond(observation)
        self._append_player_prompt_trace(player)
        engine_resolve_action = getattr(self.engine, "resolve_action", None)
        if callable(engine_resolve_action):
            action = engine_resolve_action(
                proposed_action=response.action,
                legal_actions=decision.legal_actions,
            )
        else:
            action = self._resolve_action(
                proposed_action=response.action,
                legal_actions=decision.legal_actions,
            )

        transition = self.engine.apply_action(action)
        self.event_log.append_public(transition.public_events)
        for player_id, events in transition.private_events_by_player.items():
            self.event_log.append_private(player_id, events)

        self.event_log.append_private(
            decision.acting_player_id,
            (self._private_decision_event(decision=decision, action=action, response=response),),
        )

        if response.memory_write is not None:
            self.memory_store.append(
                MemoryEntry(
                    player_id=decision.acting_player_id,
                    content=response.memory_write,
                    turn_index=decision.turn_index,
                    phase=decision.phase,
                    decision_index=decision.decision_index,
                )
            )

        return transition

    def _prepare_run(self) -> None:
        missing_player_ids = [
            player_id for player_id in self.engine.player_ids if player_id not in self.players
        ]
        if missing_player_ids:
            raise MissingPlayerError(
                "Missing player adapters for: " + ", ".join(sorted(missing_player_ids))
            )

        self.event_log.reset(self.engine.player_ids)
        self.memory_store.reset(self.engine.player_ids)
        self.prompt_trace_store.reset(self.engine.player_ids)
        self._decision_phase_counts: dict[str, int] = {}

        if self.run_dir is not None:
            write_json(
                self.run_dir / "metadata.json",
                {
                    "game_id": self.engine.game_id,
                    "player_ids": list(self.engine.player_ids),
                    "player_adapter_types": {
                        player_id: type(self.players[player_id]).__name__
                        for player_id in self.engine.player_ids
                    },
                },
            )

    @staticmethod
    def _resolve_action(
        *, proposed_action: Action, legal_actions: tuple[Action, ...]
    ) -> Action:
        for legal_action in legal_actions:
            if legal_action.matches(proposed_action):
                return legal_action
        raise InvalidActionError(
            f"Action {proposed_action.to_dict()} is not in the current legal action set."
        )

    @staticmethod
    def _winner_ids_from_metadata(metadata) -> tuple[str, ...]:
        winner_ids = metadata.get("winner_ids")
        if isinstance(winner_ids, list) and all(
            isinstance(player_id, str) for player_id in winner_ids
        ):
            return tuple(winner_ids)

        winner_id = metadata.get("winner_id")
        if isinstance(winner_id, str):
            return (winner_id,)

        return ()

    def _benchmark_metadata(self, *, total_decisions: int) -> dict[str, object]:
        public_events = self.event_log.public_events
        trade_kinds = {
            "trade_offered": "offers",
            "trade_accepted": "accepted",
            "trade_rejected": "rejected",
            "trade_confirmed": "confirmed",
            "trade_cancelled": "cancelled",
        }
        trade_metrics = {
            metric: 0 for metric in ("offers", "accepted", "rejected", "confirmed", "cancelled")
        }
        for event in public_events:
            metric_name = trade_kinds.get(event.kind)
            if metric_name is not None:
                trade_metrics[metric_name] += 1

        return {
            "history_window": self.observation_builder.recent_event_window,
            "decision_phase_counts": dict(sorted(self._decision_phase_counts.items())),
            "trade_metrics": trade_metrics,
            "trade_event_share": (
                0.0 if total_decisions == 0 else round(sum(trade_metrics.values()) / total_decisions, 4)
            ),
        }

    @staticmethod
    def _private_decision_event(
        *, decision: DecisionPoint, action: Action, response: PlayerResponse
    ) -> Event:
        payload = {
            "decision_prompt": decision.prompt,
            "action": action.to_dict(),
        }
        if response.reasoning is not None:
            payload["reasoning"] = response.reasoning
        if response.memory_write is not None:
            payload["memory_write"] = response.memory_write

        return Event(
            kind="player_decision",
            payload=payload,
            turn_index=decision.turn_index,
            phase=decision.phase,
            decision_index=decision.decision_index,
            actor_player_id=decision.acting_player_id,
        )

    def _append_player_prompt_trace(self, player: Player) -> None:
        take_last_prompt_trace = getattr(player, "take_last_prompt_trace", None)
        if not callable(take_last_prompt_trace):
            return
        prompt_trace = take_last_prompt_trace()
        if prompt_trace is None:
            return
        self.prompt_trace_store.append(prompt_trace)
