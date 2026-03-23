from __future__ import annotations

import logging
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from .engine import EngineAdapter
from .observations import ObservationBuilder
from .players import Player
from .schemas import (
    Action,
    DecisionPoint,
    Event,
    GameResult,
    JsonValue,
    MemoryEntry,
    PlayerResponse,
    TradeChatObservation,
    TradeChatOpenResponse,
    TradeChatQuote,
    TradeChatReplyResponse,
    TradeChatSelectionResponse,
    TransitionResult,
)
from .storage import EventLog, MemoryStore, PromptTraceStore, write_json

if TYPE_CHECKING:
    from .reporter import TerminalReporter

logger = logging.getLogger(__name__)


class MissingPlayerError(KeyError):
    """Raised when the orchestrator has no adapter for an engine player."""


class InvalidActionError(ValueError):
    """Raised when a player returns an action that is not currently legal."""


def _slugify_path_component(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "game"


def _resolve_run_dir(base_run_dir: str | Path | None, *, game_id: str) -> Path | None:
    if base_run_dir is None:
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = secrets.token_hex(4)
    run_name = f"{_slugify_path_component(game_id)}-{timestamp}-{token}"
    return Path(base_run_dir) / run_name


@dataclass(slots=True)
class _ActiveControlWindow:
    player_id: str
    turn_index: int
    phase: str
    decision_index: int
    public_event_start: int
    private_event_start: int


@dataclass(slots=True)
class _TradeChatTurnState:
    owner_player_id: str
    turn_index: int
    failed_attempts: int = 0


@dataclass(slots=True)
class _PendingTradeChatSelection:
    owner_player_id: str
    counterparty_player_id: str
    owner_gives: dict[str, JsonValue]
    owner_gets: dict[str, JsonValue]
    turn_index: int


class GameOrchestrator:
    """Coordinates the engine, player adapters, and run-scoped logs."""

    def __init__(
        self,
        engine: EngineAdapter,
        players: Mapping[str, Player],
        *,
        observation_builder: ObservationBuilder | None = None,
        event_log: EventLog | None = None,
        memory_store: MemoryStore | None = None,
        prompt_trace_store: PromptTraceStore | None = None,
        run_dir: str | Path | None = None,
        max_decisions: int = 10_000,
        trading_chat_enabled: bool = False,
        trading_chat_max_failed_attempts_per_turn: int = 4,
        trading_chat_message_chars: int = 160,
        trading_chat_history_limit: int | None = 16,
        reporter: TerminalReporter | None = None,
    ) -> None:
        resolved_run_dir = _resolve_run_dir(run_dir, game_id=engine.game_id)
        self.engine = engine
        self.players = dict(players)
        self.observation_builder = observation_builder or ObservationBuilder()
        self.event_log = event_log or EventLog(resolved_run_dir)
        self.memory_store = memory_store or MemoryStore(resolved_run_dir)
        self.prompt_trace_store = prompt_trace_store or PromptTraceStore(resolved_run_dir)
        self.run_dir = resolved_run_dir
        self.max_decisions = max_decisions
        self.trading_chat_enabled = trading_chat_enabled
        self.trading_chat_max_failed_attempts_per_turn = (
            trading_chat_max_failed_attempts_per_turn
        )
        self.trading_chat_message_chars = trading_chat_message_chars
        self.trading_chat_history_limit = trading_chat_history_limit
        self.reporter = reporter
        self._decision_phase_counts: dict[str, int] = {}
        self._active_control_window: _ActiveControlWindow | None = None
        self._public_event_cursor_by_player: dict[str, int] = {}
        self._private_event_cursor_by_player: dict[str, int] = {}
        self._trade_chat_turn_state: _TradeChatTurnState | None = None
        self._pending_trade_chat_selection: _PendingTradeChatSelection | None = None

    def run(self) -> GameResult:
        self._prepare_run()
        logger.info("Starting game %s with players %s", self.engine.game_id, list(self.players))
        if self.reporter is not None:
            self.reporter.on_game_start(self.engine.game_id, list(self.engine.player_ids))

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

        logger.info("Game %s finished after %d decisions", self.engine.game_id, total_decisions)
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

        if self.reporter is not None:
            self.reporter.on_game_end(result)

        return result

    def step(self):
        if self.engine.is_terminal():
            raise RuntimeError("Cannot step a terminal game.")

        decision = self.engine.current_decision()
        self._refresh_trade_chat_turn_state(decision)
        logger.debug(
            "Decision %d: player=%s phase=%s actions=%d",
            decision.decision_index,
            decision.acting_player_id,
            decision.phase,
            len(decision.legal_actions),
        )
        player = self.players.get(decision.acting_player_id)
        if player is None:
            raise MissingPlayerError(
                f"No player adapter registered for {decision.acting_player_id!r}."
            )

        self._ensure_control_window(player=player, decision=decision)
        self._decision_phase_counts[decision.phase] = (
            self._decision_phase_counts.get(decision.phase, 0) + 1
        )
        chat_events: tuple[Event, ...] = ()
        response = self._forced_trade_response(decision)
        action_decision = self._decision_for_player_action(decision)

        if response is None and self._can_offer_trade_chat(decision):
            chat_events, selected_action = self._run_trade_chat(player=player, decision=decision)
            if selected_action is not None:
                response = PlayerResponse(action=selected_action)
                action_decision = decision

        if response is None:
            observation = self.observation_builder.build_action(
                engine=self.engine,
                decision=action_decision,
                memory_store=self.memory_store,
                event_log=self.event_log,
            )
            response = player.respond(observation)
            self._append_player_prompt_traces(player)

        engine_resolve_action = getattr(self.engine, "resolve_action", None)
        if callable(engine_resolve_action):
            action = engine_resolve_action(
                proposed_action=response.action,
                legal_actions=action_decision.legal_actions,
            )
        else:
            action = self._resolve_action(
                proposed_action=response.action,
                legal_actions=action_decision.legal_actions,
            )

        engine_transition = self.engine.apply_action(action)
        combined_public_events = chat_events + engine_transition.public_events
        if engine_transition.public_events:
            self.event_log.append_public(engine_transition.public_events)
        for player_id, events in engine_transition.private_events_by_player.items():
            self.event_log.append_private(player_id, events)

        self.event_log.append_private(
            decision.acting_player_id,
            (self._private_decision_event(decision=decision, action=action, response=response),),
        )

        transition = TransitionResult(
            public_events=combined_public_events,
            private_events_by_player=engine_transition.private_events_by_player,
            terminal=engine_transition.terminal,
            result_metadata=engine_transition.result_metadata,
        )
        self._update_trade_chat_after_action(
            decision=decision,
            action=action,
            transition_terminal=engine_transition.terminal,
        )

        if self.reporter is not None:
            self.reporter.on_step(
                decision=decision,
                action=action,
                response=response,
                transition=transition,
            )

        next_decision = None
        if not engine_transition.terminal and not self.engine.is_terminal():
            next_decision = self.engine.current_decision()
        if self._should_end_control_window(
            decision=decision,
            action=action,
            transition_terminal=engine_transition.terminal,
            next_decision=next_decision,
        ):
            self._reflect_active_window(player=player, decision=decision)

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
        self._decision_phase_counts = {}
        self._active_control_window = None
        self._public_event_cursor_by_player = {
            player_id: 0 for player_id in self.engine.player_ids
        }
        self._private_event_cursor_by_player = {
            player_id: 0 for player_id in self.engine.player_ids
        }
        self._trade_chat_turn_state = None
        self._pending_trade_chat_selection = None

        if self.run_dir is not None:
            write_json(
                self.run_dir / "metadata.json",
                {
                    "game_id": self.engine.game_id,
                    "run_directory": str(self.run_dir),
                    "player_ids": list(self.engine.player_ids),
                    "player_adapter_types": {
                        player_id: type(self.players[player_id]).__name__
                        for player_id in self.engine.player_ids
                    },
                    "trading_chat": {
                        "enabled": self.trading_chat_enabled,
                        "max_failed_attempts_per_turn": (
                            self.trading_chat_max_failed_attempts_per_turn
                        ),
                        "message_chars": self.trading_chat_message_chars,
                        "history_limit": self.trading_chat_history_limit,
                    },
                },
            )

    def _ensure_control_window(self, *, player: Player, decision: DecisionPoint) -> None:
        active = self._active_control_window
        if active is not None and active.player_id == decision.acting_player_id:
            return

        player_id = decision.acting_player_id
        public_cursor = self._public_event_cursor_by_player.get(player_id, 0)
        private_cursor = self._private_event_cursor_by_player.get(player_id, 0)
        recall = getattr(player, "recall", None)
        if callable(recall):
            recall_observation = self.observation_builder.build_recall(
                engine=self.engine,
                decision=decision,
                event_log=self.event_log,
                memory_store=self.memory_store,
                public_event_start=public_cursor,
                private_event_start=private_cursor,
            )
            recall_response = recall(recall_observation)
            self._append_player_prompt_traces(player)
            self._write_memory(
                player_id=player_id,
                memory=recall_response.memory,
                turn_index=decision.turn_index,
                phase=decision.phase,
                decision_index=decision.decision_index,
                update_kind="recall",
            )

        self._active_control_window = _ActiveControlWindow(
            player_id=player_id,
            turn_index=decision.turn_index,
            phase=decision.phase,
            decision_index=decision.decision_index,
            public_event_start=len(self.event_log.public_events),
            private_event_start=len(
                self.event_log.private_events_by_player.get(player_id, ())
            ),
        )

    def _reflect_active_window(self, *, player: Player, decision: DecisionPoint) -> None:
        active = self._active_control_window
        if active is None:
            return

        reflect = getattr(player, "reflect", None)
        if callable(reflect):
            reflection_observation = self.observation_builder.build_reflection(
                engine=self.engine,
                player_id=active.player_id,
                turn_index=decision.turn_index,
                phase=decision.phase,
                decision_index=decision.decision_index,
                event_log=self.event_log,
                memory_store=self.memory_store,
                public_event_start=active.public_event_start,
                private_event_start=active.private_event_start,
            )
            reflection_response = reflect(reflection_observation)
            self._append_player_prompt_traces(player)
            self._write_memory(
                player_id=active.player_id,
                memory=reflection_response.memory,
                turn_index=decision.turn_index,
                phase=decision.phase,
                decision_index=decision.decision_index,
                update_kind="reflect",
            )

        self._public_event_cursor_by_player[active.player_id] = len(self.event_log.public_events)
        self._private_event_cursor_by_player[active.player_id] = len(
            self.event_log.private_events_by_player.get(active.player_id, ())
        )
        self._active_control_window = None

    def _write_memory(
        self,
        *,
        player_id: str,
        memory,
        turn_index: int,
        phase: str,
        decision_index: int,
        update_kind: str,
    ) -> None:
        self.memory_store.set(
            MemoryEntry(
                player_id=player_id,
                content=memory,
                turn_index=turn_index,
                phase=phase,
                decision_index=decision_index,
                update_kind=update_kind,
            )
        )

    @staticmethod
    def _should_end_control_window(
        *,
        decision: DecisionPoint,
        action: Action,
        transition_terminal: bool,
        next_decision: DecisionPoint | None,
    ) -> bool:
        if transition_terminal or action.action_type == "END_TURN":
            return True
        if next_decision is None:
            return True
        return next_decision.acting_player_id != decision.acting_player_id

    @staticmethod
    def _resolve_action(
        *, proposed_action: Action, legal_actions: tuple[Action, ...]
    ) -> Action:
        for legal_action in legal_actions:
            if legal_action.matches(proposed_action):
                return legal_action
        if proposed_action.action_type == "OFFER_TRADE" and any(
            action.action_type == "OFFER_TRADE" for action in legal_actions
        ):
            return proposed_action
        matching_actions = [
            action for action in legal_actions if action.action_type == proposed_action.action_type
        ]
        if proposed_action.action_type in {"ACCEPT_TRADE", "REJECT_TRADE", "CANCEL_TRADE"}:
            if len(matching_actions) == 1:
                return matching_actions[0]
        if proposed_action.action_type == "CONFIRM_TRADE":
            accepting_player_id = proposed_action.payload.get("accepting_player_id")
            if len(matching_actions) == 1 and accepting_player_id is None:
                return matching_actions[0]
            if isinstance(accepting_player_id, str):
                for legal_action in matching_actions:
                    if legal_action.payload.get("accepting_player_id") == accepting_player_id:
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
        trade_chat_metrics = {
            metric: 0
            for metric in (
                "chat_windows_opened",
                "chat_messages",
                "quotes_selected",
                "no_deals",
            )
        }
        for event in public_events:
            metric_name = trade_kinds.get(event.kind)
            if metric_name is not None:
                trade_metrics[metric_name] += 1
            if event.kind == "trade_chat_opened":
                trade_chat_metrics["chat_windows_opened"] += 1
            elif event.kind == "trade_chat_message":
                trade_chat_metrics["chat_messages"] += 1
            elif event.kind == "trade_chat_quote_selected":
                trade_chat_metrics["quotes_selected"] += 1
            elif event.kind == "trade_chat_no_deal":
                trade_chat_metrics["no_deals"] += 1

        return {
            "history_window": self.observation_builder.recent_event_window,
            "run_directory": None if self.run_dir is None else str(self.run_dir),
            "decision_phase_counts": dict(sorted(self._decision_phase_counts.items())),
            "trade_metrics": trade_metrics,
            "trade_chat_metrics": trade_chat_metrics,
            "trade_event_share": (
                0.0
                if total_decisions == 0
                else round(sum(trade_metrics.values()) / total_decisions, 4)
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

        return Event(
            kind="player_decision",
            payload=payload,
            turn_index=decision.turn_index,
            phase=decision.phase,
            decision_index=decision.decision_index,
            actor_player_id=decision.acting_player_id,
        )

    def _refresh_trade_chat_turn_state(self, decision: DecisionPoint) -> None:
        if decision.phase != "play_turn":
            return
        state = self._trade_chat_turn_state
        if (
            state is None
            or state.owner_player_id != decision.acting_player_id
            or state.turn_index != decision.turn_index
        ):
            self._trade_chat_turn_state = _TradeChatTurnState(
                owner_player_id=decision.acting_player_id,
                turn_index=decision.turn_index,
            )

    def _can_offer_trade_chat(self, decision: DecisionPoint) -> bool:
        if not self.trading_chat_enabled:
            return False
        if decision.phase != "play_turn":
            return False
        if any(action.action_type == "OFFER_TRADE" for action in decision.legal_actions) is False:
            return False
        state = self._trade_chat_turn_state
        if state is None:
            return False
        return state.failed_attempts < self.trading_chat_max_failed_attempts_per_turn

    def _decision_for_player_action(self, decision: DecisionPoint) -> DecisionPoint:
        if not self._can_offer_trade_chat(decision):
            return decision
        filtered_actions = tuple(
            action for action in decision.legal_actions if action.action_type != "OFFER_TRADE"
        )
        return DecisionPoint(
            acting_player_id=decision.acting_player_id,
            turn_index=decision.turn_index,
            phase=decision.phase,
            legal_actions=filtered_actions,
            decision_index=decision.decision_index,
            prompt=decision.prompt,
        )

    def _forced_trade_response(self, decision: DecisionPoint) -> PlayerResponse | None:
        pending = self._pending_trade_chat_selection
        if pending is None or pending.turn_index != decision.turn_index:
            return None

        if decision.phase == "decide_trade":
            if decision.acting_player_id == pending.counterparty_player_id:
                return PlayerResponse(action=Action("ACCEPT_TRADE"))
            if decision.acting_player_id != pending.owner_player_id:
                return PlayerResponse(action=Action("REJECT_TRADE"))
            return None

        if (
            decision.phase == "decide_acceptees"
            and decision.acting_player_id == pending.owner_player_id
        ):
            return PlayerResponse(
                action=Action(
                    "CONFIRM_TRADE",
                    payload={"accepting_player_id": pending.counterparty_player_id},
                )
            )

        return None

    def _run_trade_chat(
        self, *, player: Player, decision: DecisionPoint
    ) -> tuple[tuple[Event, ...], Action | None]:
        open_response = self._open_trade_chat(player=player, decision=decision)
        if not open_response.open_chat or not open_response.requested_resources:
            return (), None

        attempt_index = 1
        state = self._trade_chat_turn_state
        if state is not None:
            attempt_index = state.failed_attempts + 1

        events: list[Event] = [
            Event(
                kind="trade_chat_opened",
                payload={
                    "owner_player_id": decision.acting_player_id,
                    "requested_resources": open_response.requested_resources,
                    "message": self._truncate_chat_message(open_response.message),
                    "attempt_index": attempt_index,
                },
                turn_index=decision.turn_index,
                phase=decision.phase,
                decision_index=decision.decision_index,
                actor_player_id=decision.acting_player_id,
            )
        ]
        if open_response.message:
            events.append(
                self._trade_chat_message_event(
                    owner_player_id=decision.acting_player_id,
                    speaker_player_id=decision.acting_player_id,
                    message=open_response.message,
                    turn_index=decision.turn_index,
                    phase=decision.phase,
                    decision_index=decision.decision_index,
                    attempt_index=attempt_index,
                )
            )

        self.event_log.append_public(tuple(events))
        quotes: list[TradeChatQuote] = []
        for other_player_id in self._trade_chat_participants(decision.acting_player_id):
            reply = self._reply_trade_chat(
                player_id=other_player_id,
                decision=decision,
                attempt_index=attempt_index,
                requested_resources=open_response.requested_resources,
                quotes=tuple(quotes),
            )
            if reply.message is None and (not reply.owner_gives or not reply.owner_gets):
                continue
            quote = TradeChatQuote(
                player_id=other_player_id,
                message=self._truncate_chat_message(reply.message),
                owner_gives=reply.owner_gives,
                owner_gets=reply.owner_gets,
            )
            quotes.append(quote)
            message_event = self._trade_chat_message_event(
                owner_player_id=decision.acting_player_id,
                speaker_player_id=other_player_id,
                message=reply.message,
                turn_index=decision.turn_index,
                phase=decision.phase,
                decision_index=decision.decision_index,
                attempt_index=attempt_index,
                quote=quote,
            )
            events.append(message_event)
            self.event_log.append_public((message_event,))

        selection = self._select_trade_chat_offer(
            player=player,
            decision=decision,
            attempt_index=attempt_index,
            requested_resources=open_response.requested_resources,
            quotes=tuple(quotes),
        )
        selected_quote = next(
            (quote for quote in quotes if quote.player_id == selection.selected_player_id),
            None,
        )

        if selected_quote is None:
            no_deal_event = Event(
                kind="trade_chat_no_deal",
                payload={
                    "owner_player_id": decision.acting_player_id,
                    "attempt_index": attempt_index,
                    "message": self._truncate_chat_message(selection.message),
                },
                turn_index=decision.turn_index,
                phase=decision.phase,
                decision_index=decision.decision_index,
                actor_player_id=decision.acting_player_id,
            )
            close_event = Event(
                kind="trade_chat_closed",
                payload={
                    "owner_player_id": decision.acting_player_id,
                    "attempt_index": attempt_index,
                    "outcome": "no_deal",
                },
                turn_index=decision.turn_index,
                phase=decision.phase,
                decision_index=decision.decision_index,
                actor_player_id=decision.acting_player_id,
            )
            events.extend((no_deal_event, close_event))
            self.event_log.append_public((no_deal_event, close_event))
            self._increment_failed_trade_chat_attempt()
            return tuple(events), None

        selected_event = Event(
            kind="trade_chat_quote_selected",
            payload={
                "owner_player_id": decision.acting_player_id,
                "selected_player_id": selected_quote.player_id,
                "offer": selected_quote.owner_gives,
                "request": selected_quote.owner_gets,
                "message": self._truncate_chat_message(selection.message),
                "attempt_index": attempt_index,
            },
            turn_index=decision.turn_index,
            phase=decision.phase,
            decision_index=decision.decision_index,
            actor_player_id=decision.acting_player_id,
        )
        close_event = Event(
            kind="trade_chat_closed",
            payload={
                "owner_player_id": decision.acting_player_id,
                "attempt_index": attempt_index,
                "outcome": "selected",
                "selected_player_id": selected_quote.player_id,
            },
            turn_index=decision.turn_index,
            phase=decision.phase,
            decision_index=decision.decision_index,
            actor_player_id=decision.acting_player_id,
        )
        events.extend((selected_event, close_event))
        self.event_log.append_public((selected_event, close_event))
        self._pending_trade_chat_selection = _PendingTradeChatSelection(
            owner_player_id=decision.acting_player_id,
            counterparty_player_id=selected_quote.player_id,
            owner_gives={key: int(value) for key, value in selected_quote.owner_gives.items()},
            owner_gets={key: int(value) for key, value in selected_quote.owner_gets.items()},
            turn_index=decision.turn_index,
        )
        return (
            tuple(events),
            Action(
                "OFFER_TRADE",
                payload={
                    "offer": dict(selected_quote.owner_gives),
                    "request": dict(selected_quote.owner_gets),
                },
            ),
        )

    def _open_trade_chat(self, *, player: Player, decision: DecisionPoint) -> TradeChatOpenResponse:
        open_trade_chat = getattr(player, "open_trade_chat", None)
        if not callable(open_trade_chat):
            return TradeChatOpenResponse(open_chat=False)
        observation = self._trade_chat_observation(
            player_id=decision.acting_player_id,
            owner_player_id=decision.acting_player_id,
            decision=decision,
            stage="open",
            attempt_index=(self._trade_chat_turn_state.failed_attempts + 1)
            if self._trade_chat_turn_state is not None
            else 1,
            requested_resources={},
            quotes=(),
        )
        response = open_trade_chat(observation)
        self._append_player_prompt_traces(player)
        return TradeChatOpenResponse(
            open_chat=response.open_chat,
            message=self._truncate_chat_message(response.message),
            requested_resources=self._normalize_resource_map(response.requested_resources),
            reasoning=response.reasoning,
        )

    def _reply_trade_chat(
        self,
        *,
        player_id: str,
        decision: DecisionPoint,
        attempt_index: int,
        requested_resources: dict[str, JsonValue],
        quotes: tuple[TradeChatQuote, ...],
    ) -> TradeChatReplyResponse:
        player = self.players[player_id]
        respond_trade_chat = getattr(player, "respond_trade_chat", None)
        if not callable(respond_trade_chat):
            return TradeChatReplyResponse()
        observation = self._trade_chat_observation(
            player_id=player_id,
            owner_player_id=decision.acting_player_id,
            decision=decision,
            stage="reply",
            attempt_index=attempt_index,
            requested_resources=requested_resources,
            quotes=quotes,
        )
        response = respond_trade_chat(observation)
        self._append_player_prompt_traces(player)
        return TradeChatReplyResponse(
            message=self._truncate_chat_message(response.message),
            owner_gives=self._normalize_resource_map(response.owner_gives),
            owner_gets=self._normalize_resource_map(response.owner_gets),
            reasoning=response.reasoning,
        )

    def _select_trade_chat_offer(
        self,
        *,
        player: Player,
        decision: DecisionPoint,
        attempt_index: int,
        requested_resources: dict[str, JsonValue],
        quotes: tuple[TradeChatQuote, ...],
    ) -> TradeChatSelectionResponse:
        select_trade_chat_offer = getattr(player, "select_trade_chat_offer", None)
        if not callable(select_trade_chat_offer):
            return TradeChatSelectionResponse()
        observation = self._trade_chat_observation(
            player_id=decision.acting_player_id,
            owner_player_id=decision.acting_player_id,
            decision=decision,
            stage="select",
            attempt_index=attempt_index,
            requested_resources=requested_resources,
            quotes=quotes,
        )
        response = select_trade_chat_offer(observation)
        self._append_player_prompt_traces(player)
        selected_player_id = response.selected_player_id
        if isinstance(selected_player_id, str):
            selected_player_id = selected_player_id.upper()
        return TradeChatSelectionResponse(
            selected_player_id=selected_player_id,
            message=self._truncate_chat_message(response.message),
            reasoning=response.reasoning,
        )

    def _trade_chat_observation(
        self,
        *,
        player_id: str,
        owner_player_id: str,
        decision: DecisionPoint,
        stage: str,
        attempt_index: int,
        requested_resources: dict[str, JsonValue],
        quotes: tuple[TradeChatQuote, ...],
    ) -> TradeChatObservation:
        transcript = self._trade_chat_transcript(decision.turn_index)
        return TradeChatObservation(
            game_id=self.engine.game_id,
            player_id=player_id,
            owner_player_id=owner_player_id,
            turn_index=decision.turn_index,
            phase=decision.phase,
            decision_index=decision.decision_index,
            stage=stage,
            attempt_index=attempt_index,
            public_state=dict(self.engine.public_state()),
            private_state=dict(self.engine.private_state(player_id)),
            transcript=transcript,
            requested_resources=dict(requested_resources),
            other_player_ids=tuple(
                other_player_id for other_player_id in self.engine.player_ids if other_player_id != player_id
            ),
            quotes=quotes,
            game_rules=self.observation_builder.game_rules,
            memory=self.memory_store.get(player_id),
            message_char_limit=self.trading_chat_message_chars,
        )

    def _trade_chat_transcript(self, turn_index: int) -> tuple[Event, ...]:
        events = self.event_log.recent_public(self.trading_chat_history_limit)
        return tuple(event for event in events if event.turn_index == turn_index)

    def _trade_chat_participants(self, owner_player_id: str) -> tuple[str, ...]:
        return tuple(player_id for player_id in self.engine.player_ids if player_id != owner_player_id)

    def _trade_chat_message_event(
        self,
        *,
        owner_player_id: str,
        speaker_player_id: str,
        message: str | None,
        turn_index: int,
        phase: str,
        decision_index: int,
        attempt_index: int,
        quote: TradeChatQuote | None = None,
    ) -> Event:
        payload: dict[str, JsonValue] = {
            "owner_player_id": owner_player_id,
            "speaker_player_id": speaker_player_id,
            "attempt_index": attempt_index,
        }
        if message is not None:
            payload["message"] = self._truncate_chat_message(message)
        if quote is not None and quote.owner_gives and quote.owner_gets:
            payload["offer"] = dict(quote.owner_gives)
            payload["request"] = dict(quote.owner_gets)
        return Event(
            kind="trade_chat_message",
            payload=payload,
            turn_index=turn_index,
            phase=phase,
            decision_index=decision_index,
            actor_player_id=speaker_player_id,
        )

    def _increment_failed_trade_chat_attempt(self) -> None:
        state = self._trade_chat_turn_state
        if state is not None:
            state.failed_attempts += 1

    def _update_trade_chat_after_action(
        self,
        *,
        decision: DecisionPoint,
        action: Action,
        transition_terminal: bool,
    ) -> None:
        pending = self._pending_trade_chat_selection
        if pending is None:
            return
        if transition_terminal:
            self._pending_trade_chat_selection = None
            return
        if action.action_type == "CANCEL_TRADE":
            self._increment_failed_trade_chat_attempt()
            self._pending_trade_chat_selection = None
            return
        if action.action_type == "CONFIRM_TRADE":
            self._pending_trade_chat_selection = None
            return
        if decision.turn_index != pending.turn_index:
            self._pending_trade_chat_selection = None

    def _normalize_resource_map(self, value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for resource, amount in value.items():
            if not isinstance(resource, str):
                continue
            if not isinstance(amount, int):
                continue
            if amount <= 0:
                continue
            result[resource] = amount
        return result

    def _truncate_chat_message(self, message: str | None) -> str | None:
        if message is None:
            return None
        stripped = message.strip()
        if not stripped:
            return None
        return stripped[: self.trading_chat_message_chars]

    def _append_player_prompt_traces(self, player: Player) -> None:
        take_prompt_traces = getattr(player, "take_prompt_traces", None)
        if callable(take_prompt_traces):
            for prompt_trace in take_prompt_traces():
                self.prompt_trace_store.append(prompt_trace)
                self._report_prompt_trace(prompt_trace)
            return

        take_last_prompt_trace = getattr(player, "take_last_prompt_trace", None)
        if not callable(take_last_prompt_trace):
            return
        prompt_trace = take_last_prompt_trace()
        if prompt_trace is None:
            return
        self.prompt_trace_store.append(prompt_trace)
        self._report_prompt_trace(prompt_trace)

    def _report_prompt_trace(self, prompt_trace) -> None:
        if self.reporter is None:
            return
        on_prompt_trace = getattr(self.reporter, "on_prompt_trace", None)
        if callable(on_prompt_trace):
            on_prompt_trace(prompt_trace)
