from __future__ import annotations

import json
import logging
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from .engine import EngineAdapter
from .observations import ObservationBuilder
from .players import Player
from .schemas import (
    Action,
    ActionDecision,
    ActionTraceEntry,
    DecisionPoint,
    Event,
    GameResult,
    JsonValue,
    PublicStateSnapshot,
    TradeChatOpenResponse,
    TradeChatOwnerDecisionResponse,
    TradeChatProposal,
    TradeChatReplyResponse,
    TransitionResult,
)
from .storage import (
    ActionTraceStore,
    EventLog,
    MemoryStore,
    PromptTraceStore,
    PublicStateStore,
    read_json,
    write_json,
)

if TYPE_CHECKING:
    from .reporter import TerminalReporter

logger = logging.getLogger(__name__)


class MissingPlayerError(KeyError):
    """Raised when the orchestrator has no adapter for an engine player."""


class InvalidActionError(ValueError):
    """Raised when a player returns an action that is not currently legal."""


def _slugify_path_component(value: str) -> str:
    slug = re.sub(r"[^a-z0-9.]+", "-", value.lower()).strip("-")
    return slug or "game"


def _resolve_run_dir(
    base_run_dir: str | Path | None,
    *,
    game_id: str,
    run_tags: tuple[str, ...] = (),
) -> Path | None:
    if base_run_dir is None:
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = secrets.token_hex(4)
    tag_prefix = "-".join(
        _slugify_path_component(tag) for tag in run_tags if _slugify_path_component(tag)
    )
    if tag_prefix:
        run_name = (
            f"{tag_prefix}-{_slugify_path_component(game_id)}-{timestamp}-{token}"
        )
    else:
        run_name = f"{_slugify_path_component(game_id)}-{timestamp}-{token}"
    return Path(base_run_dir) / run_name


@dataclass(slots=True)
class _ActiveTurnSession:
    owner_player_id: str
    turn_index: int
    start_history_index: int
    start_phase: str
    start_decision_index: int


@dataclass(slots=True)
class _TradeChatTurnState:
    owner_player_id: str
    turn_index: int
    opened_attempts: int = 0
    failed_attempts: int = 0
    opened_room_signatures: set[str] = field(default_factory=set)


@dataclass(slots=True)
class _PendingTradeChatSelection:
    owner_player_id: str
    counterparty_player_id: str
    owner_gives: dict[str, JsonValue]
    owner_gets: dict[str, JsonValue]
    turn_index: int


class GameOrchestrator:
    """Coordinates the engine, players, and run-scoped artifacts."""

    def __init__(
        self,
        engine: EngineAdapter,
        players: Mapping[str, Player],
        *,
        observation_builder: ObservationBuilder | None = None,
        event_log: EventLog | None = None,
        public_state_store: PublicStateStore | None = None,
        memory_store: MemoryStore | None = None,
        prompt_trace_store: PromptTraceStore | None = None,
        action_trace_store: ActionTraceStore | None = None,
        run_dir: str | Path | None = None,
        run_tags: tuple[str, ...] = (),
        resume_run_dir: str | Path | None = None,
        max_decisions: int = 10_000,
        trading_chat_enabled: bool = False,
        trading_chat_max_failed_attempts_per_turn: int = 5,
        trading_chat_max_rooms_per_turn: int = 5,
        trading_chat_max_rounds_per_attempt: int = 3,
        trading_chat_message_chars: int = 160,
        trading_chat_history_limit: int | None = 16,
        reporter: TerminalReporter | None = None,
    ) -> None:
        if run_dir is not None and resume_run_dir is not None:
            raise ValueError("Specify either run_dir or resume_run_dir, not both.")

        resolved_run_dir = (
            Path(resume_run_dir)
            if resume_run_dir is not None
            else _resolve_run_dir(run_dir, game_id=engine.game_id, run_tags=run_tags)
        )
        self.engine = engine
        self.players = dict(players)
        self.observation_builder = observation_builder or ObservationBuilder()
        self.event_log = event_log or EventLog(resolved_run_dir)
        self.public_state_store = public_state_store or PublicStateStore(
            resolved_run_dir
        )
        self.memory_store = memory_store or MemoryStore(resolved_run_dir)
        self.prompt_trace_store = prompt_trace_store or PromptTraceStore(
            resolved_run_dir
        )
        self.action_trace_store = action_trace_store or ActionTraceStore(
            resolved_run_dir
        )
        self.run_dir = resolved_run_dir
        self._run_tags = tuple(str(tag) for tag in run_tags)
        self.max_decisions = max_decisions
        self.trading_chat_enabled = trading_chat_enabled
        self.trading_chat_max_failed_attempts_per_turn = (
            trading_chat_max_failed_attempts_per_turn
        )
        self.trading_chat_max_rooms_per_turn = max(1, trading_chat_max_rooms_per_turn)
        self.trading_chat_max_rounds_per_attempt = max(
            1, trading_chat_max_rounds_per_attempt
        )
        self.trading_chat_message_chars = trading_chat_message_chars
        self.trading_chat_history_limit = trading_chat_history_limit
        self.reporter = reporter
        self._decision_phase_counts: dict[str, int] = {}
        self._active_turn: _ActiveTurnSession | None = None
        self._last_turn_history_index_by_player: dict[str, int] = {}
        self._trade_chat_turn_state: _TradeChatTurnState | None = None
        self._pending_trade_chat_selection: _PendingTradeChatSelection | None = None
        self._opening_strategy_done = False
        self._completed_decisions = 0
        self._resume_run_dir = (
            Path(resume_run_dir) if resume_run_dir is not None else None
        )
        self._resume_checkpoint_history_index: int | None = None
        self._prepared = False

    def run(self) -> GameResult:
        self._ensure_ready_for_play()
        logger.info(
            "Starting game %s with players %s", self.engine.game_id, list(self.players)
        )
        if self.reporter is not None:
            self.reporter.on_game_start(
                self.engine.game_id, list(self.engine.player_ids)
            )

        while not self.engine.is_terminal():
            if self._completed_decisions >= self.max_decisions:
                raise RuntimeError(
                    f"Stopped after {self.max_decisions} decisions without a terminal state."
                )
            self.step()

        self._finalize_active_turn_if_needed()
        metadata = dict(self.engine.result())
        metadata["benchmark"] = self._benchmark_metadata(
            total_decisions=self._completed_decisions
        )
        result = GameResult(
            game_id=self.engine.game_id,
            winner_ids=self._winner_ids_from_metadata(metadata),
            total_decisions=self._completed_decisions,
            public_event_count=len(self.event_log.public_events),
            memory_writes=self.memory_store.count(),
            metadata=metadata,
        )
        if self.run_dir is not None:
            write_json(self.run_dir / "result.json", result.to_dict())
            self._write_checkpoint(terminal=True)
        if self.reporter is not None:
            self.reporter.on_game_end(result)
        return result

    def step(self) -> TransitionResult:
        self._ensure_ready_for_play()
        if self.engine.is_terminal():
            raise RuntimeError("Cannot step a terminal game.")

        decision = self.engine.current_decision()
        turn_owner_id = self._turn_owner_id(decision)
        self._refresh_trade_chat_turn_state(
            turn_owner_id=turn_owner_id, decision=decision
        )
        self._decision_phase_counts[decision.phase] = (
            self._decision_phase_counts.get(decision.phase, 0) + 1
        )
        self._maybe_run_opening_strategy_phase(decision)

        player = self.players.get(decision.acting_player_id)
        if player is None:
            raise MissingPlayerError(
                f"No player adapter registered for {decision.acting_player_id!r}."
            )

        if self._should_auto_roll(decision):
            response = ActionDecision(action=Action("ROLL"))
            transition = self._apply_decision(
                decision=decision,
                proposed_action=response.action,
                response=response,
            )
            self._completed_step(transition)
            return transition

        if self._is_turn_owner_choice(decision, turn_owner_id):
            self._ensure_turn_started(player=player, decision=decision)
            response = self._turn_owner_response(player=player, decision=decision)
            is_turn_owner_choice = True
        else:
            response = self._reactive_response(player=player, decision=decision)
            is_turn_owner_choice = False

        transition = self._apply_decision_with_recovery(
            player=player,
            decision=decision,
            response=response,
            is_turn_owner_choice=is_turn_owner_choice,
        )
        self._completed_step(transition)
        return transition

    def _ensure_ready_for_play(self) -> None:
        if self._prepared:
            return
        if self._resume_run_dir is not None:
            self._prepare_resume()
        else:
            self._prepare_run()
        self._prepared = True

    def _completed_step(self, transition: TransitionResult) -> None:
        self._completed_decisions += 1
        if self.run_dir is not None:
            self._write_checkpoint(terminal=transition.terminal)

    def _prepare_run(self) -> None:
        missing_player_ids = [
            player_id
            for player_id in self.engine.player_ids
            if player_id not in self.players
        ]
        if missing_player_ids:
            raise MissingPlayerError(
                "Missing player adapters for: " + ", ".join(sorted(missing_player_ids))
            )

        self.event_log.reset()
        self.memory_store.reset(self.engine.player_ids)
        self.prompt_trace_store.reset(self.engine.player_ids)
        self.action_trace_store.reset()
        self.public_state_store.reset(
            PublicStateSnapshot(
                history_index=0,
                turn_index=0,
                phase="initial",
                decision_index=None,
                public_state=dict(self.engine.public_state()),
            )
        )
        self._decision_phase_counts = {}
        self._active_turn = None
        self._last_turn_history_index_by_player = {
            player_id: 0 for player_id in self.engine.player_ids
        }
        self._trade_chat_turn_state = None
        self._pending_trade_chat_selection = None
        self._opening_strategy_done = False
        self._completed_decisions = 0
        self._resume_checkpoint_history_index = None

        if self.run_dir is not None:
            write_json(
                self.run_dir / "metadata.json",
                {
                    "game_id": self.engine.game_id,
                    "artifact_version": 3,
                    "run_directory": str(self.run_dir),
                    "run_tags": list(self._run_tags),
                    "player_ids": list(self.engine.player_ids),
                    "player_adapter_types": {
                        player_id: type(self.players[player_id]).__name__
                        for player_id in self.engine.player_ids
                    },
                    "trading_chat": {
                        "enabled": self.trading_chat_enabled,
                        "max_rooms_per_turn": self.trading_chat_max_rooms_per_turn,
                        "max_failed_attempts_per_turn": (
                            self.trading_chat_max_failed_attempts_per_turn
                        ),
                        "max_rounds_per_attempt": self.trading_chat_max_rounds_per_attempt,
                        "message_chars": self.trading_chat_message_chars,
                        "history_limit": self.trading_chat_history_limit,
                    },
                    "resume": {
                        "checkpoint_file": "checkpoint.json",
                        "action_trace_file": "action_trace.jsonl",
                    },
                },
            )
            self._write_checkpoint(terminal=False)

    def _prepare_resume(self) -> None:
        missing_player_ids = [
            player_id
            for player_id in self.engine.player_ids
            if player_id not in self.players
        ]
        if missing_player_ids:
            raise MissingPlayerError(
                "Missing player adapters for: " + ", ".join(sorted(missing_player_ids))
            )
        if self.run_dir is None:
            raise RuntimeError("Resume mode requires a concrete run directory.")

        self.event_log.hydrate()
        self.public_state_store.hydrate()
        self.memory_store.hydrate(self.engine.player_ids)
        self.prompt_trace_store.hydrate(self.engine.player_ids)
        self.action_trace_store.hydrate()
        self._validate_initial_resume_artifacts()
        self._replay_saved_actions()
        checkpoint = read_json(self.run_dir / "checkpoint.json") or {}
        self._restore_checkpoint_state(checkpoint)
        self._validate_resumed_artifacts()

    def _replay_saved_actions(self) -> None:
        for index, entry in enumerate(self.action_trace_store.entries, start=1):
            if self.engine.is_terminal():
                raise RuntimeError(
                    "Resume trace contains more actions than the rebuilt engine can accept."
                )
            decision = self.engine.current_decision()
            self._validate_trace_entry(entry=entry, decision=decision, index=index)
            replay_action = entry.action
            if replay_action.action_type == "COUNTER_OFFER":
                replay_action = Action("REJECT_TRADE")
            self.engine.apply_action(
                self._resolved_action_for_decision(
                    decision=decision,
                    proposed_action=replay_action,
                )
            )

    def _validate_trace_entry(
        self,
        *,
        entry: ActionTraceEntry,
        decision: DecisionPoint,
        index: int,
    ) -> None:
        mismatches: list[str] = []
        if decision.acting_player_id != entry.acting_player_id:
            mismatches.append(
                f"acting_player_id expected {entry.acting_player_id} got {decision.acting_player_id}"
            )
        if decision.turn_index != entry.turn_index:
            mismatches.append(
                f"turn_index expected {entry.turn_index} got {decision.turn_index}"
            )
        if decision.phase != entry.phase:
            mismatches.append(f"phase expected {entry.phase} got {decision.phase}")
        if decision.decision_index != entry.decision_index:
            mismatches.append(
                f"decision_index expected {entry.decision_index} got {decision.decision_index}"
            )
        if mismatches:
            raise RuntimeError(
                "Cannot resume run because the recorded action trace no longer matches the "
                f"rebuilt engine at step {index}: " + "; ".join(mismatches)
            )

    def _restore_checkpoint_state(self, checkpoint: dict[str, JsonValue]) -> None:
        self._decision_phase_counts = {
            str(phase): int(count)
            for phase, count in dict(
                checkpoint.get("decision_phase_counts") or {}
            ).items()
            if isinstance(phase, str)
        }
        self._last_turn_history_index_by_player = {
            str(player_id): int(history_index)
            for player_id, history_index in dict(
                checkpoint.get("last_turn_history_index_by_player") or {}
            ).items()
            if isinstance(player_id, str)
        }
        active_turn = checkpoint.get("active_turn")
        if isinstance(active_turn, dict):
            self._active_turn = _ActiveTurnSession(
                owner_player_id=str(active_turn["owner_player_id"]),
                turn_index=int(active_turn["turn_index"]),
                start_history_index=int(active_turn["start_history_index"]),
                start_phase=str(active_turn["start_phase"]),
                start_decision_index=int(active_turn["start_decision_index"]),
            )
        else:
            self._active_turn = None

        trade_chat_turn_state = checkpoint.get("trade_chat_turn_state")
        if isinstance(trade_chat_turn_state, dict):
            self._trade_chat_turn_state = _TradeChatTurnState(
                owner_player_id=str(trade_chat_turn_state["owner_player_id"]),
                turn_index=int(trade_chat_turn_state["turn_index"]),
                opened_attempts=int(
                    trade_chat_turn_state.get(
                        "opened_attempts",
                        trade_chat_turn_state.get("failed_attempts", 0),
                    )
                ),
                failed_attempts=int(trade_chat_turn_state.get("failed_attempts", 0)),
                opened_room_signatures={
                    str(signature)
                    for signature in trade_chat_turn_state.get(
                        "opened_room_signatures", []
                    )
                    if isinstance(signature, str)
                },
            )
        else:
            self._trade_chat_turn_state = None

        pending_trade_chat_selection = checkpoint.get("pending_trade_chat_selection")
        if isinstance(pending_trade_chat_selection, dict):
            self._pending_trade_chat_selection = _PendingTradeChatSelection(
                owner_player_id=str(pending_trade_chat_selection["owner_player_id"]),
                counterparty_player_id=str(
                    pending_trade_chat_selection["counterparty_player_id"]
                ),
                owner_gives=dict(pending_trade_chat_selection.get("owner_gives") or {}),
                owner_gets=dict(pending_trade_chat_selection.get("owner_gets") or {}),
                turn_index=int(pending_trade_chat_selection["turn_index"]),
            )
        else:
            self._pending_trade_chat_selection = None

        checkpoint_history_index = checkpoint.get("current_history_index")
        if checkpoint_history_index is None:
            self._resume_checkpoint_history_index = None
        else:
            self._resume_checkpoint_history_index = int(checkpoint_history_index)
        self._completed_decisions = int(
            checkpoint.get("total_decisions", len(self.action_trace_store.entries))
        )
        opening_strategy_done = checkpoint.get("opening_strategy_done")
        if isinstance(opening_strategy_done, bool):
            self._opening_strategy_done = opening_strategy_done
        else:
            self._opening_strategy_done = self._infer_opening_strategy_done()
        if not self._last_turn_history_index_by_player:
            self._last_turn_history_index_by_player = {
                player_id: 0 for player_id in self.engine.player_ids
            }

    def _validate_resumed_artifacts(self) -> None:
        if self._completed_decisions != len(self.action_trace_store.entries):
            raise RuntimeError(
                "Resume checkpoint does not match action trace length: "
                f"{self._completed_decisions} decisions recorded in checkpoint versus "
                f"{len(self.action_trace_store.entries)} actions in trace."
            )
        if (
            self._resume_checkpoint_history_index is not None
            and self._resume_checkpoint_history_index
            != self.event_log.current_history_index
        ):
            raise RuntimeError(
                "Resume checkpoint current_history_index does not match public history "
                f"length: checkpoint recorded {self._resume_checkpoint_history_index} "
                f"events versus {self.event_log.current_history_index} in history."
            )
        if self.event_log.current_history_index:
            if (
                self.public_state_store.snapshots[-1].history_index
                != self.event_log.current_history_index
            ):
                raise RuntimeError(
                    "Resume run has inconsistent public history and public state traces."
                )
        if (
            self.public_state_store.snapshots[-1].public_state
            != dict(self.engine.public_state())
        ):
            raise RuntimeError(
                "Resume run final public state snapshot does not match the rebuilt "
                "engine state."
            )

    def _validate_initial_resume_artifacts(self) -> None:
        if not self.public_state_store.snapshots:
            raise RuntimeError(
                "Resume run is missing its initial public state snapshot."
            )
        if (
            self.public_state_store.snapshots[0]
            != self._expected_initial_public_state_snapshot()
        ):
            raise RuntimeError(
                "Resume run initial public state snapshot does not match the rebuilt engine."
            )

    def _expected_initial_public_state_snapshot(self) -> PublicStateSnapshot:
        return PublicStateSnapshot(
            history_index=0,
            turn_index=0,
            phase="initial",
            decision_index=None,
            public_state=dict(self.engine.public_state()),
        )

    def _write_checkpoint(self, *, terminal: bool) -> None:
        if self.run_dir is None:
            return

        active_turn: dict[str, JsonValue] | None = None
        if self._active_turn is not None:
            active_turn = {
                "owner_player_id": self._active_turn.owner_player_id,
                "turn_index": self._active_turn.turn_index,
                "start_history_index": self._active_turn.start_history_index,
                "start_phase": self._active_turn.start_phase,
                "start_decision_index": self._active_turn.start_decision_index,
            }

        trade_chat_turn_state: dict[str, JsonValue] | None = None
        if self._trade_chat_turn_state is not None:
            trade_chat_turn_state = {
                "owner_player_id": self._trade_chat_turn_state.owner_player_id,
                "turn_index": self._trade_chat_turn_state.turn_index,
                "opened_attempts": self._trade_chat_turn_state.opened_attempts,
                "failed_attempts": self._trade_chat_turn_state.failed_attempts,
                "opened_room_signatures": sorted(
                    self._trade_chat_turn_state.opened_room_signatures
                ),
            }

        pending_trade_chat_selection: dict[str, JsonValue] | None = None
        if self._pending_trade_chat_selection is not None:
            pending_trade_chat_selection = {
                "owner_player_id": self._pending_trade_chat_selection.owner_player_id,
                "counterparty_player_id": (
                    self._pending_trade_chat_selection.counterparty_player_id
                ),
                "owner_gives": dict(self._pending_trade_chat_selection.owner_gives),
                "owner_gets": dict(self._pending_trade_chat_selection.owner_gets),
                "turn_index": self._pending_trade_chat_selection.turn_index,
            }

        write_json(
            self.run_dir / "checkpoint.json",
            {
                "artifact_version": 1,
                "total_decisions": self._completed_decisions,
                "current_history_index": self.event_log.current_history_index,
                "terminal": terminal,
                "decision_phase_counts": dict(self._decision_phase_counts),
                "last_turn_history_index_by_player": dict(
                    self._last_turn_history_index_by_player
                ),
                "active_turn": active_turn,
                "opening_strategy_done": self._opening_strategy_done,
                "trade_chat_turn_state": trade_chat_turn_state,
                "pending_trade_chat_selection": pending_trade_chat_selection,
            },
        )

    def _resolved_action_for_decision(
        self,
        *,
        decision: DecisionPoint,
        proposed_action: Action,
    ) -> Action:
        self._validate_trade_offer_turn_constraints(
            decision=decision,
            proposed_action=proposed_action,
        )
        try:
            engine_resolve_action = getattr(self.engine, "resolve_action", None)
            if callable(engine_resolve_action):
                return engine_resolve_action(
                    proposed_action=proposed_action,
                    legal_actions=decision.legal_actions,
                )
            return self._resolve_action(
                proposed_action=proposed_action,
                legal_actions=decision.legal_actions,
            )
        except ValueError as exc:
            raise InvalidActionError(str(exc)) from exc

    def _should_auto_roll(self, decision: DecisionPoint) -> bool:
        if decision.phase != "play_turn":
            return False
        return any(action.action_type == "ROLL" for action in decision.legal_actions)

    def _validate_trade_offer_turn_constraints(
        self,
        *,
        decision: DecisionPoint,
        proposed_action: Action,
    ) -> None:
        if (
            proposed_action.action_type != "OFFER_TRADE"
            or decision.phase != "play_turn"
        ):
            return
        signature = self._trade_market_signature(proposed_action.payload)
        if signature is None:
            return
        blocked_markets = self._blocked_trade_markets_this_turn(
            player_id=decision.acting_player_id,
            turn_index=decision.turn_index,
        )
        if signature in blocked_markets:
            raise InvalidActionError(
                "Cannot repeat the same domestic trade market in the same turn after it was "
                "rejected without any counteroffers."
            )

    def _blocked_trade_markets_this_turn(
        self,
        *,
        player_id: str,
        turn_index: int,
    ) -> set[str]:
        start_history_index = self._turn_start_history_index()
        turn_events = [
            event
            for event in self.event_log.public_events
            if event.turn_index == turn_index
            and event.history_index > start_history_index
        ]
        blocked: set[str] = set()
        current_attempt: dict[str, bool | str] | None = None
        for event in turn_events:
            if event.kind == "trade_offered" and event.actor_player_id == player_id:
                if current_attempt is not None and self._is_blocked_trade_attempt(
                    current_attempt
                ):
                    blocked.add(str(current_attempt["signature"]))
                current_attempt = {
                    "signature": self._trade_market_signature(event.payload) or "",
                    "had_rejected": False,
                    "had_counter": False,
                    "had_accepted": False,
                    "had_confirmed": False,
                }
                continue
            if current_attempt is None:
                continue
            if (
                event.kind == "trade_rejected"
                and event.payload.get("offering_player_id") == player_id
            ):
                current_attempt["had_rejected"] = True
            elif (
                event.kind == "trade_counter_offered"
                and event.payload.get("owner_player_id") == player_id
            ):
                current_attempt["had_counter"] = True
            elif (
                event.kind == "trade_accepted"
                and event.payload.get("offering_player_id") == player_id
            ):
                current_attempt["had_accepted"] = True
            elif (
                event.kind == "trade_confirmed"
                and event.payload.get("offering_player_id") == player_id
            ):
                current_attempt["had_confirmed"] = True
        if current_attempt is not None and self._is_blocked_trade_attempt(
            current_attempt
        ):
            blocked.add(str(current_attempt["signature"]))
        blocked.discard("")
        return blocked

    @staticmethod
    def _is_blocked_trade_attempt(attempt: dict[str, bool | str]) -> bool:
        return bool(
            attempt.get("had_rejected")
            and not attempt.get("had_counter")
            and not attempt.get("had_accepted")
            and not attempt.get("had_confirmed")
        )

    @staticmethod
    def _trade_market_signature(payload: Mapping[str, JsonValue]) -> str | None:
        offer = GameOrchestrator._canonical_trade_resource_map(payload.get("offer"))
        request = GameOrchestrator._canonical_trade_resource_map(payload.get("request"))
        if offer is None or request is None:
            return None
        return f"offer={offer}|request={request}"

    @staticmethod
    def _canonical_trade_resource_map(value: object) -> str | None:
        if not isinstance(value, Mapping):
            return None
        normalized: dict[str, int] = {}
        for resource, amount in value.items():
            if not isinstance(resource, str):
                return None
            if not isinstance(amount, int) or amount <= 0:
                return None
            normalized[resource] = amount
        if not normalized:
            return None
        return ",".join(
            f"{resource}:{normalized[resource]}" for resource in sorted(normalized)
        )

    @staticmethod
    def _canonical_chat_message_signature(value: str | None) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = " ".join(value.lower().split())
        return normalized or None

    @classmethod
    def _trade_chat_room_signature(
        cls,
        *,
        action_payload: Mapping[str, JsonValue],
        requested_resources: Mapping[str, JsonValue],
        opening_message: str | None,
    ) -> str | None:
        market = cls._trade_market_signature(action_payload)
        requested = cls._canonical_trade_resource_map(requested_resources)
        components: list[str] = []
        if market is not None:
            components.append(f"market={market}")
        if requested is not None:
            components.append(f"requested={requested}")
        if not components:
            message = cls._canonical_chat_message_signature(opening_message)
            if message is not None:
                components.append(f"message={message}")
        if not components:
            return None
        return "|".join(components)

    def _maybe_run_opening_strategy_phase(self, decision: DecisionPoint) -> None:
        if self._opening_strategy_done or decision.phase != "play_turn":
            return
        for player_id in self.engine.player_ids:
            player = self.players[player_id]
            observation = self.observation_builder.build_opening_strategy(
                engine=self.engine,
                player_id=player_id,
                turn_index=decision.turn_index,
                decision_index=decision.decision_index,
                event_log=self.event_log,
                memory_store=self.memory_store,
            )
            response = player.plan_opening_strategy(observation)
            self._append_player_prompt_traces(player)
            self.memory_store.set_long_term(
                player_id=player_id,
                long_term=response.long_term,
                history_index=observation.history_index,
                turn_index=observation.turn_index,
                phase=observation.phase,
                decision_index=observation.decision_index,
                stage="opening_strategy",
            )
        self._opening_strategy_done = True

    def _infer_opening_strategy_done(self) -> bool:
        for player_id in self.engine.player_ids:
            if any(
                snapshot.stage == "opening_strategy"
                for snapshot in self.memory_store.history(player_id)
            ):
                return True
        if any(entry.phase == "play_turn" for entry in self.action_trace_store.entries):
            return True
        return any(event.phase == "play_turn" for event in self.event_log.public_events)

    def _is_turn_owner_choice(
        self, decision: DecisionPoint, turn_owner_id: str
    ) -> bool:
        return (
            decision.phase == "play_turn"
            and decision.acting_player_id == turn_owner_id
            and not self._should_auto_roll(decision)
        )

    def _ensure_turn_started(self, *, player: Player, decision: DecisionPoint) -> None:
        active = self._active_turn
        if (
            active is not None
            and active.owner_player_id == decision.acting_player_id
            and active.turn_index == decision.turn_index
        ):
            return

        player_id = decision.acting_player_id
        observation = self.observation_builder.build_turn_start(
            engine=self.engine,
            decision=decision,
            event_log=self.event_log,
            memory_store=self.memory_store,
            last_turn_history_index=self._last_turn_history_index_by_player.get(
                player_id, 0
            ),
        )
        response = player.start_turn(observation)
        self._append_player_prompt_traces(player)
        self.memory_store.set_short_term(
            player_id=player_id,
            short_term=response.short_term,
            history_index=observation.history_index,
            turn_index=decision.turn_index,
            phase=decision.phase,
            decision_index=decision.decision_index,
            stage="turn_start",
        )
        self._active_turn = _ActiveTurnSession(
            owner_player_id=player_id,
            turn_index=decision.turn_index,
            start_history_index=observation.history_index,
            start_phase=decision.phase,
            start_decision_index=decision.decision_index,
        )

    def _turn_owner_response(
        self, *, player: Player, decision: DecisionPoint
    ) -> ActionDecision:
        pending_response: ActionDecision | None = None
        consecutive_invalid_trade_chat_attempts = 0
        while True:
            if pending_response is None:
                observation = self.observation_builder.build_action(
                    engine=self.engine,
                    decision=decision,
                    event_log=self.event_log,
                    memory_store=self.memory_store,
                    turn_start_history_index=self._turn_start_history_index(),
                    trade_chat_enabled=self._trade_chat_available_for_decision(
                        decision
                    ),
                    trade_chat_attempts_remaining=self._trade_chat_attempts_remaining(
                        decision
                    ),
                )
                response = player.choose_action(observation)
                self._append_player_prompt_traces(player)
            else:
                response = pending_response
                pending_response = None
            if not self._should_route_trade_action_to_chat(
                decision=decision, action=response.action
            ):
                return response
            if not self._can_offer_trade_chat(decision):
                pending_response = self._retry_after_invalid_action(
                    player=player,
                    decision=decision,
                    previous_response=response,
                    error_message=(
                        "Public trade room attempts are exhausted for this turn. "
                        "Choose a different legal action instead of OFFER_TRADE."
                    ),
                    is_turn_owner_choice=True,
                )
                if pending_response is None:
                    return ActionDecision(
                        action=self._fallback_non_trade_action(decision.legal_actions),
                        short_term=response.short_term,
                    )
                if not self._should_route_trade_action_to_chat(
                    decision=decision,
                    action=pending_response.action,
                ):
                    return pending_response
                return ActionDecision(
                    action=self._fallback_non_trade_action(decision.legal_actions),
                    short_term=response.short_term,
                )
            try:
                _, selected_action = self._run_trade_chat(
                    player=player,
                    decision=decision,
                    selected_trade_action=response.action,
                )
                consecutive_invalid_trade_chat_attempts = 0
            except InvalidActionError as exc:
                consecutive_invalid_trade_chat_attempts += 1
                if consecutive_invalid_trade_chat_attempts >= 2:
                    return ActionDecision(
                        action=self._fallback_non_trade_action(decision.legal_actions),
                        short_term=response.short_term,
                    )
                pending_response = self._retry_after_invalid_action(
                    player=player,
                    decision=decision,
                    previous_response=response,
                    error_message=str(exc),
                    is_turn_owner_choice=True,
                )
                if pending_response is None:
                    return ActionDecision(
                        action=self._fallback_non_trade_action(decision.legal_actions),
                        short_term=response.short_term,
                    )
                if not self._should_route_trade_action_to_chat(
                    decision=decision,
                    action=pending_response.action,
                ):
                    return pending_response
                continue
            if selected_action is not None:
                return ActionDecision(
                    action=selected_action,
                    short_term=response.short_term,
                    reasoning=response.reasoning,
                )

    def _reactive_response(
        self, *, player: Player, decision: DecisionPoint
    ) -> ActionDecision:
        forced = self._forced_trade_response(decision)
        if forced is not None:
            return forced
        observation = self.observation_builder.build_reactive(
            engine=self.engine,
            decision=decision,
            event_log=self.event_log,
            memory_store=self.memory_store,
        )
        response = player.respond_reactive(observation)
        self._append_player_prompt_traces(player)
        return response

    def _apply_decision(
        self,
        *,
        decision: DecisionPoint,
        proposed_action: Action,
        response: ActionDecision,
        update_short_term: bool = False,
    ) -> TransitionResult:
        if proposed_action.action_type == "COUNTER_OFFER":
            return self._apply_counter_offer_decision(
                decision=decision,
                proposed_action=proposed_action,
                response=response,
            )
        action = self._resolved_action_for_decision(
            decision=decision,
            proposed_action=proposed_action,
        )
        engine_transition = self.engine.apply_action(action)
        transition = self._record_transition(
            decision=decision,
            action=action,
            transition=engine_transition,
        )
        self.action_trace_store.append(
            ActionTraceEntry(
                acting_player_id=decision.acting_player_id,
                turn_index=decision.turn_index,
                phase=decision.phase,
                decision_index=decision.decision_index,
                action=action,
            )
        )
        self._update_trade_chat_after_action(
            decision=decision,
            action=action,
            transition_terminal=transition.terminal,
        )

        if update_short_term:
            self.memory_store.set_short_term(
                player_id=decision.acting_player_id,
                short_term=response.short_term,
                history_index=self.event_log.current_history_index,
                turn_index=decision.turn_index,
                phase=decision.phase,
                decision_index=decision.decision_index,
                stage="choose_action",
            )

        self._maybe_end_turn_session(decision=decision, transition=transition)

        if self.reporter is not None:
            self.reporter.on_step(
                decision=decision,
                action=action,
                response=response,
                transition=transition,
            )

        return transition

    def _apply_decision_with_recovery(
        self,
        *,
        player: Player,
        decision: DecisionPoint,
        response: ActionDecision,
        is_turn_owner_choice: bool,
    ) -> TransitionResult:
        try:
            return self._apply_decision(
                decision=decision,
                proposed_action=response.action,
                response=response,
                update_short_term=is_turn_owner_choice,
            )
        except InvalidActionError as exc:
            logger.info(
                "Player %s selected invalid action %s at turn=%s phase=%s decision=%s: %s",
                decision.acting_player_id,
                response.action.to_dict(),
                decision.turn_index,
                decision.phase,
                decision.decision_index,
                exc,
            )
            retry_response = self._retry_after_invalid_action(
                player=player,
                decision=decision,
                previous_response=response,
                error_message=str(exc),
                is_turn_owner_choice=is_turn_owner_choice,
            )
            if retry_response is not None:
                try:
                    return self._apply_decision(
                        decision=decision,
                        proposed_action=retry_response.action,
                        response=retry_response,
                        update_short_term=is_turn_owner_choice,
                    )
                except InvalidActionError as retry_exc:
                    logger.info(
                        "Retry for player %s remained invalid at turn=%s phase=%s decision=%s: %s",
                        decision.acting_player_id,
                        decision.turn_index,
                        decision.phase,
                        decision.decision_index,
                        retry_exc,
                    )

            fallback_response = ActionDecision(
                action=self._fallback_action_for_decision(decision.legal_actions),
                short_term=None,
            )
            return self._apply_decision(
                decision=decision,
                proposed_action=fallback_response.action,
                response=fallback_response,
                update_short_term=is_turn_owner_choice,
            )

    def _retry_after_invalid_action(
        self,
        *,
        player: Player,
        decision: DecisionPoint,
        previous_response: ActionDecision,
        error_message: str,
        is_turn_owner_choice: bool,
    ) -> ActionDecision | None:
        retry_prompt = self._invalid_action_retry_prompt(
            original_prompt=decision.prompt,
            previous_action=previous_response.action,
            error_message=error_message,
        )
        retry_decision = DecisionPoint(
            acting_player_id=decision.acting_player_id,
            turn_index=decision.turn_index,
            phase=decision.phase,
            legal_actions=decision.legal_actions,
            decision_index=decision.decision_index,
            prompt=retry_prompt,
        )
        try:
            if is_turn_owner_choice:
                observation = self.observation_builder.build_action(
                    engine=self.engine,
                    decision=retry_decision,
                    event_log=self.event_log,
                    memory_store=self.memory_store,
                    turn_start_history_index=self._turn_start_history_index(),
                    trade_chat_enabled=self._trade_chat_available_for_decision(
                        retry_decision
                    ),
                    trade_chat_attempts_remaining=self._trade_chat_attempts_remaining(
                        retry_decision
                    ),
                )
                response = player.choose_action(observation)
            else:
                observation = self.observation_builder.build_reactive(
                    engine=self.engine,
                    decision=retry_decision,
                    event_log=self.event_log,
                    memory_store=self.memory_store,
                )
                response = player.respond_reactive(observation)
            self._append_player_prompt_traces(player)
            return response
        except RuntimeError:
            self._append_player_prompt_traces(player)
            return None

    @staticmethod
    def _invalid_action_retry_prompt(
        *,
        original_prompt: str | None,
        previous_action: Action,
        error_message: str,
    ) -> str:
        base_prompt = original_prompt or "Choose exactly one legal action."
        return (
            f"{base_prompt}\n\n"
            f"Previous action was invalid: {json.dumps(previous_action.to_dict(), sort_keys=True)}\n"
            f"Error: {error_message}\n"
            "Choose a different legal action or, if trading again, a different market."
        )

    def _apply_counter_offer_decision(
        self,
        *,
        decision: DecisionPoint,
        proposed_action: Action,
        response: ActionDecision,
    ) -> TransitionResult:
        action = self._resolved_counter_offer_for_decision(
            decision=decision,
            proposed_action=proposed_action,
        )
        counter_event = Event(
            kind="trade_counter_offered",
            payload={
                "owner_player_id": self._trade_owner_for_decision(decision),
                "offer": dict(action.payload.get("offer", {})),
                "request": dict(action.payload.get("request", {})),
            },
            turn_index=decision.turn_index,
            phase=decision.phase,
            decision_index=decision.decision_index,
            actor_player_id=decision.acting_player_id,
        )
        reject_action = self._resolved_action_for_decision(
            decision=decision,
            proposed_action=Action("REJECT_TRADE"),
        )
        engine_transition = self.engine.apply_action(reject_action)
        transition = self._record_transition(
            decision=decision,
            action=action,
            transition=TransitionResult(
                public_events=(counter_event, *engine_transition.public_events),
                terminal=engine_transition.terminal,
                result_metadata=engine_transition.result_metadata,
            ),
        )
        self.action_trace_store.append(
            ActionTraceEntry(
                acting_player_id=decision.acting_player_id,
                turn_index=decision.turn_index,
                phase=decision.phase,
                decision_index=decision.decision_index,
                action=action,
            )
        )
        self._update_trade_chat_after_action(
            decision=decision,
            action=reject_action,
            transition_terminal=transition.terminal,
        )
        self._maybe_end_turn_session(decision=decision, transition=transition)
        if self.reporter is not None:
            self.reporter.on_step(
                decision=decision,
                action=action,
                response=response,
                transition=transition,
            )
        return transition

    def _store_events(
        self, events: tuple[Event, ...] | list[Event]
    ) -> tuple[Event, ...]:
        """Append events to the log and write the resulting public-state snapshot once.

        We snapshot only after the full transition has been applied. If one engine
        action emits multiple public events, attaching the same post-transition
        state to each event makes the dashboard cursor "look ahead" when stepping
        through those events one by one. Writing the snapshot only at the final
        event keeps earlier cursor positions anchored to the previous known board.
        """
        stored = self.event_log.append(events)
        if stored:
            final_event = stored[-1]
            self.public_state_store.append(
                PublicStateSnapshot(
                    history_index=final_event.history_index,
                    turn_index=final_event.turn_index,
                    phase=final_event.phase,
                    decision_index=final_event.decision_index,
                    public_state=dict(self.engine.public_state()),
                )
            )
        return stored

    def _record_transition(
        self, *, decision: DecisionPoint, action: Action, transition: TransitionResult
    ) -> TransitionResult:
        filtered_events, hidden_internal_step = self._filtered_transition_public_events(
            decision=decision,
            action=action,
            public_events=transition.public_events,
        )
        stored_events = self._store_events(filtered_events)
        result_metadata = dict(transition.result_metadata)
        if hidden_internal_step:
            result_metadata["hidden_internal_step"] = True
        return TransitionResult(
            public_events=stored_events,
            terminal=transition.terminal,
            result_metadata=result_metadata,
        )

    def _filtered_transition_public_events(
        self,
        *,
        decision: DecisionPoint,
        action: Action,
        public_events: tuple[Event, ...],
    ) -> tuple[tuple[Event, ...], bool]:
        pending = self._pending_trade_chat_selection
        if pending is None or pending.turn_index != decision.turn_index:
            return public_events, False

        # After a public chat proposal has been selected, the remaining engine
        # trade-resolution steps are internal plumbing. Keep the chat selection
        # and final confirmation public, but hide the legacy offer/accept/reject
        # events so the transcript matches what players actually saw.
        if (
            decision.phase == "play_turn"
            and action.action_type == "OFFER_TRADE"
            and decision.acting_player_id == pending.owner_player_id
        ):
            return (), True
        if decision.phase == "decide_trade":
            if (
                action.action_type in {"ACCEPT_TRADE", "REJECT_TRADE"}
                and decision.acting_player_id != pending.owner_player_id
            ):
                return (), True
        if (
            decision.phase == "decide_acceptees"
            and decision.acting_player_id == pending.owner_player_id
            and action.action_type == "CANCEL_TRADE"
        ):
            return (), True
        return public_events, False

    @staticmethod
    def _resolved_counter_offer_for_decision(
        *,
        decision: DecisionPoint,
        proposed_action: Action,
    ) -> Action:
        matching_actions = [
            action
            for action in decision.legal_actions
            if action.action_type == "COUNTER_OFFER"
        ]
        if not matching_actions:
            raise InvalidActionError(
                "COUNTER_OFFER is not legal for the current decision."
            )
        offer = proposed_action.payload.get("offer")
        request = proposed_action.payload.get("request")
        if (
            not isinstance(offer, dict)
            or not offer
            or not isinstance(request, dict)
            or not request
        ):
            raise InvalidActionError(
                "COUNTER_OFFER requires non-empty `offer` and `request` resource maps."
            )
        normalized_offer = GameOrchestrator._normalize_counter_offer_map(offer)
        normalized_request = GameOrchestrator._normalize_counter_offer_map(request)
        return Action(
            "COUNTER_OFFER",
            payload={
                "offer": normalized_offer,
                "request": normalized_request,
            },
        )

    @staticmethod
    def _normalize_counter_offer_map(resource_map: dict) -> dict[str, int]:
        normalized: dict[str, int] = {}
        for resource, amount in resource_map.items():
            if not isinstance(resource, str):
                raise InvalidActionError(
                    "COUNTER_OFFER resource names must be strings."
                )
            if not isinstance(amount, int) or amount <= 0:
                raise InvalidActionError(
                    "COUNTER_OFFER resource amounts must be positive integers."
                )
            normalized[resource] = amount
        if not normalized:
            raise InvalidActionError(
                "COUNTER_OFFER resource maps must contain at least one positive quantity."
            )
        return normalized

    def _trade_owner_for_decision(self, decision: DecisionPoint) -> str:
        public_state = dict(self.engine.public_state())
        trade_state = public_state.get("trade_state")
        if isinstance(trade_state, dict):
            owner_player_id = trade_state.get("offering_player_id")
            if isinstance(owner_player_id, str):
                return owner_player_id
        return self._turn_owner_id(decision)

    def _maybe_end_turn_session(
        self, *, decision: DecisionPoint, transition: TransitionResult
    ) -> None:
        active = self._active_turn
        if active is None:
            return
        if transition.terminal:
            self._end_turn_session(
                active, phase=decision.phase, decision_index=decision.decision_index
            )
            return
        next_decision = self.engine.current_decision()
        next_turn_owner_id = self._turn_owner_id(next_decision)
        if (
            next_turn_owner_id != active.owner_player_id
            or next_decision.turn_index != active.turn_index
        ):
            self._end_turn_session(
                active,
                phase=decision.phase,
                decision_index=decision.decision_index,
            )

    def _finalize_active_turn_if_needed(self) -> None:
        active = self._active_turn
        if active is None:
            return
        self._end_turn_session(
            active,
            phase=active.start_phase,
            decision_index=active.start_decision_index,
        )

    def _end_turn_session(
        self,
        active: _ActiveTurnSession,
        *,
        phase: str,
        decision_index: int,
    ) -> None:
        player = self.players[active.owner_player_id]
        observation = self.observation_builder.build_turn_end(
            engine=self.engine,
            player_id=active.owner_player_id,
            turn_index=active.turn_index,
            phase=phase,
            decision_index=decision_index,
            event_log=self.event_log,
            memory_store=self.memory_store,
            turn_start_history_index=active.start_history_index,
        )
        response = player.end_turn(observation)
        self._append_player_prompt_traces(player)
        self.memory_store.set_long_term(
            player_id=active.owner_player_id,
            long_term=response.long_term,
            history_index=self.event_log.current_history_index,
            turn_index=active.turn_index,
            phase=phase,
            decision_index=decision_index,
            stage="turn_end",
        )
        self.memory_store.clear_short_term(
            player_id=active.owner_player_id,
            history_index=self.event_log.current_history_index,
            turn_index=active.turn_index,
            phase=phase,
            decision_index=decision_index,
            stage="turn_cleanup",
        )
        self._last_turn_history_index_by_player[active.owner_player_id] = (
            self.event_log.current_history_index
        )
        self._active_turn = None

    def _turn_owner_id(self, decision: DecisionPoint) -> str:
        turn_owner_id_prop = getattr(self.engine, "turn_owner_id", None)
        if isinstance(turn_owner_id_prop, str):
            return turn_owner_id_prop
        # fallback for engines that don't expose turn_owner_id directly
        public_state = dict(self.engine.public_state())
        turn = public_state.get("turn")
        if isinstance(turn, dict):
            turn_player_id = turn.get("turn_player_id")
            if isinstance(turn_player_id, str):
                return turn_player_id
        return decision.acting_player_id

    def _turn_start_history_index(self) -> int:
        active = self._active_turn
        if active is None:
            return self.event_log.current_history_index
        return active.start_history_index

    def _can_offer_trade_chat(self, decision: DecisionPoint) -> bool:
        if not self.trading_chat_enabled:
            return False
        if decision.phase != "play_turn":
            return False
        if (
            any(
                action.action_type == "OFFER_TRADE" for action in decision.legal_actions
            )
            is False
        ):
            return False
        state = self._trade_chat_turn_state
        if state is None:
            return False
        return (
            state.opened_attempts < self.trading_chat_max_rooms_per_turn
            and state.failed_attempts < self.trading_chat_max_failed_attempts_per_turn
        )

    def _trade_chat_available_for_decision(self, decision: DecisionPoint) -> bool:
        return self._can_offer_trade_chat(decision)

    def _trade_chat_attempts_remaining(self, decision: DecisionPoint) -> int | None:
        if not self._trade_chat_available_for_decision(decision):
            return None
        state = self._trade_chat_turn_state
        if state is None:
            return min(
                self.trading_chat_max_rooms_per_turn,
                self.trading_chat_max_failed_attempts_per_turn,
            )
        return max(
            0,
            min(
                self.trading_chat_max_rooms_per_turn - state.opened_attempts,
                self.trading_chat_max_failed_attempts_per_turn - state.failed_attempts,
            ),
        )

    def _should_route_trade_action_to_chat(
        self, *, decision: DecisionPoint, action: Action
    ) -> bool:
        return (
            self.trading_chat_enabled
            and decision.phase == "play_turn"
            and action.action_type == "OFFER_TRADE"
        )

    @staticmethod
    def _fallback_non_trade_action(legal_actions: tuple[Action, ...]) -> Action:
        for action in legal_actions:
            if action.action_type != "OFFER_TRADE":
                return action
        return GameOrchestrator._fallback_action_for_decision(legal_actions)

    @staticmethod
    def _first_legal_action(
        legal_actions: tuple[Action, ...],
        *,
        action_type: str,
    ) -> Action | None:
        for action in legal_actions:
            if action.action_type == action_type:
                return action
        return None

    def _forced_trade_response(self, decision: DecisionPoint) -> ActionDecision | None:
        pending = self._pending_trade_chat_selection
        if pending is None or pending.turn_index != decision.turn_index:
            trade_owner_id = self._trade_owner_for_decision(decision)
            if (
                decision.phase == "decide_trade"
                and decision.acting_player_id == trade_owner_id
            ):
                reject_action = self._first_legal_action(
                    decision.legal_actions,
                    action_type="REJECT_TRADE",
                )
                if reject_action is not None:
                    return ActionDecision(action=reject_action)
            return None

        if decision.phase == "decide_trade":
            if decision.acting_player_id == pending.counterparty_player_id:
                accept_action = self._first_legal_action(
                    decision.legal_actions,
                    action_type="ACCEPT_TRADE",
                )
                if accept_action is not None:
                    return ActionDecision(action=accept_action)
                reject_action = self._first_legal_action(
                    decision.legal_actions,
                    action_type="REJECT_TRADE",
                )
                if reject_action is not None:
                    return ActionDecision(action=reject_action)
                return None
            if decision.acting_player_id != pending.owner_player_id:
                reject_action = self._first_legal_action(
                    decision.legal_actions,
                    action_type="REJECT_TRADE",
                )
                if reject_action is not None:
                    return ActionDecision(action=reject_action)
            return None

        if (
            decision.phase == "decide_acceptees"
            and decision.acting_player_id == pending.owner_player_id
        ):
            for action in decision.legal_actions:
                if (
                    action.action_type == "CONFIRM_TRADE"
                    and action.payload.get("accepting_player_id")
                    == pending.counterparty_player_id
                ):
                    return ActionDecision(action=action)
            cancel_action = self._first_legal_action(
                decision.legal_actions,
                action_type="CANCEL_TRADE",
            )
            if cancel_action is not None:
                return ActionDecision(action=cancel_action)
        return None

    def _run_trade_chat(
        self,
        *,
        player: Player,
        decision: DecisionPoint,
        selected_trade_action: Action,
    ) -> tuple[tuple[Event, ...], Action | None]:
        initial_requested_resources = self._normalize_resource_map(
            selected_trade_action.payload.get("request")
        )
        open_response = self._open_trade_chat(
            player=player,
            decision=decision,
            requested_resources=initial_requested_resources,
        )
        requested_resources = (
            open_response.requested_resources
            if open_response.requested_resources
            else initial_requested_resources
        )
        room_signature = self._trade_chat_room_signature(
            action_payload=selected_trade_action.payload,
            requested_resources=requested_resources,
            opening_message=open_response.message,
        )
        self._validate_trade_chat_open_turn_constraints(
            decision=decision,
            room_signature=room_signature,
        )

        state = self._trade_chat_turn_state
        attempt_index = state.opened_attempts + 1 if state is not None else 1
        if state is not None:
            state.opened_attempts = attempt_index
            if room_signature is not None:
                state.opened_room_signatures.add(room_signature)
        events: list[Event] = [
            Event(
                kind="trade_chat_opened",
                payload={
                    "owner_player_id": decision.acting_player_id,
                    "requested_resources": requested_resources,
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
                    round_index=0,
                )
            )
        self._record_public_events(events)

        proposals: list[TradeChatProposal] = []
        for round_index in range(1, self.trading_chat_max_rounds_per_attempt + 1):
            for other_player_id in self._trade_chat_participants(
                decision.acting_player_id
            ):
                reply = self._reply_trade_chat(
                    player_id=other_player_id,
                    decision=decision,
                    attempt_index=attempt_index,
                    round_index=round_index,
                    requested_resources=requested_resources,
                    proposals=tuple(proposals),
                )
                proposal: TradeChatProposal | None = None
                if reply.owner_gives and reply.owner_gets:
                    counterparty_can_pay = self._player_has_resources(
                        other_player_id,
                        reply.owner_gets,
                    )
                    owner_can_pay = self._player_has_resources(
                        decision.acting_player_id,
                        reply.owner_gives,
                    )
                    if counterparty_can_pay and owner_can_pay:
                        proposal = TradeChatProposal(
                            proposal_id=self._next_trade_chat_proposal_id(
                                attempt_index=attempt_index,
                                round_index=round_index,
                                proposal_count=len(proposals) + 1,
                            ),
                            player_id=other_player_id,
                            round_index=round_index,
                            message=self._truncate_chat_message(reply.message),
                            owner_gives=reply.owner_gives,
                            owner_gets=reply.owner_gets,
                        )
                        proposals.append(proposal)
                    else:
                        if not counterparty_can_pay:
                            logger.debug(
                                "Skipping proposal from %s: insufficient resources for %s",
                                other_player_id,
                                reply.owner_gets,
                            )
                        if not owner_can_pay:
                            logger.debug(
                                "Skipping proposal from %s: owner %s lacks resources for %s",
                                other_player_id,
                                decision.acting_player_id,
                                reply.owner_gives,
                            )
                if reply.message is None and proposal is None:
                    continue
                message_event = self._trade_chat_message_event(
                    owner_player_id=decision.acting_player_id,
                    speaker_player_id=other_player_id,
                    message=reply.message,
                    turn_index=decision.turn_index,
                    phase=decision.phase,
                    decision_index=decision.decision_index,
                    attempt_index=attempt_index,
                    round_index=round_index,
                    proposal=proposal,
                )
                events.append(message_event)
                self._record_public_events((message_event,))

            owner_decision = self._decide_trade_chat(
                player=player,
                decision=decision,
                attempt_index=attempt_index,
                round_index=round_index,
                requested_resources=requested_resources,
                proposals=tuple(proposals),
            )
            if owner_decision.decision == "continue":
                if owner_decision.message is not None:
                    owner_message_event = self._trade_chat_message_event(
                        owner_player_id=decision.acting_player_id,
                        speaker_player_id=decision.acting_player_id,
                        message=owner_decision.message,
                        turn_index=decision.turn_index,
                        phase=decision.phase,
                        decision_index=decision.decision_index,
                        attempt_index=attempt_index,
                        round_index=round_index,
                    )
                    events.append(owner_message_event)
                    self._record_public_events((owner_message_event,))
                continue

            selected_proposal = None
            if owner_decision.decision == "select":
                selected_proposal = next(
                    (
                        proposal
                        for proposal in proposals
                        if proposal.proposal_id == owner_decision.selected_proposal_id
                    ),
                    None,
                )
                if selected_proposal is not None:
                    owner_can_pay = self._player_has_resources(
                        decision.acting_player_id,
                        selected_proposal.owner_gives,
                    )
                    counterparty_can_pay = self._player_has_resources(
                        selected_proposal.player_id,
                        selected_proposal.owner_gets,
                    )
                    if not owner_can_pay or not counterparty_can_pay:
                        logger.debug(
                            "Selected proposal %s became invalid before execution.",
                            selected_proposal.proposal_id,
                        )
                        selected_proposal = None
            if selected_proposal is None:
                no_deal_event = Event(
                    kind="trade_chat_no_deal",
                    payload={
                        "owner_player_id": decision.acting_player_id,
                        "attempt_index": attempt_index,
                        "round_index": round_index,
                        "message": self._truncate_chat_message(owner_decision.message),
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
                        "round_index": round_index,
                        "outcome": "no_deal",
                    },
                    turn_index=decision.turn_index,
                    phase=decision.phase,
                    decision_index=decision.decision_index,
                    actor_player_id=decision.acting_player_id,
                )
                events.extend((no_deal_event, close_event))
                self._record_public_events((no_deal_event, close_event))
                self._increment_failed_trade_chat_attempt()
                return tuple(events), None

            selected_event = Event(
                kind="trade_chat_quote_selected",
                payload={
                    "owner_player_id": decision.acting_player_id,
                    "selected_player_id": selected_proposal.player_id,
                    "selected_proposal_id": selected_proposal.proposal_id,
                    "offer": selected_proposal.owner_gives,
                    "request": selected_proposal.owner_gets,
                    "message": self._truncate_chat_message(owner_decision.message),
                    "attempt_index": attempt_index,
                    "round_index": round_index,
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
                    "round_index": round_index,
                    "outcome": "selected",
                    "selected_player_id": selected_proposal.player_id,
                    "selected_proposal_id": selected_proposal.proposal_id,
                },
                turn_index=decision.turn_index,
                phase=decision.phase,
                decision_index=decision.decision_index,
                actor_player_id=decision.acting_player_id,
            )
            events.extend((selected_event, close_event))
            self._record_public_events((selected_event, close_event))
            self._pending_trade_chat_selection = _PendingTradeChatSelection(
                owner_player_id=decision.acting_player_id,
                counterparty_player_id=selected_proposal.player_id,
                owner_gives={
                    key: int(value)
                    for key, value in selected_proposal.owner_gives.items()
                },
                owner_gets={
                    key: int(value)
                    for key, value in selected_proposal.owner_gets.items()
                },
                turn_index=decision.turn_index,
            )
            return (
                tuple(events),
                Action(
                    "OFFER_TRADE",
                    payload={
                        "offer": dict(selected_proposal.owner_gives),
                        "request": dict(selected_proposal.owner_gets),
                    },
                ),
            )

        no_deal_event = Event(
            kind="trade_chat_no_deal",
            payload={
                "owner_player_id": decision.acting_player_id,
                "attempt_index": attempt_index,
                "round_index": self.trading_chat_max_rounds_per_attempt,
                "message": "Reached round limit.",
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
                "round_index": self.trading_chat_max_rounds_per_attempt,
                "outcome": "max_rounds",
            },
            turn_index=decision.turn_index,
            phase=decision.phase,
            decision_index=decision.decision_index,
            actor_player_id=decision.acting_player_id,
        )
        events.extend((no_deal_event, close_event))
        self._record_public_events((no_deal_event, close_event))
        self._increment_failed_trade_chat_attempt()
        return tuple(events), None

    def _open_trade_chat(
        self,
        *,
        player: Player,
        decision: DecisionPoint,
        requested_resources: dict[str, JsonValue],
    ) -> TradeChatOpenResponse:
        open_trade_chat = getattr(player, "open_trade_chat", None)
        if not callable(open_trade_chat):
            return TradeChatOpenResponse(
                open_chat=True, requested_resources=requested_resources
            )
        observation = self.observation_builder.build_trade_chat(
            engine=self.engine,
            player_id=decision.acting_player_id,
            owner_player_id=decision.acting_player_id,
            decision=decision,
            stage="open",
            attempt_index=(self._trade_chat_turn_state.opened_attempts + 1)
            if self._trade_chat_turn_state is not None
            else 1,
            round_index=0,
            requested_resources=requested_resources,
            proposals=(),
            event_log=self.event_log,
            memory_store=self.memory_store,
            message_char_limit=self.trading_chat_message_chars,
            transcript_limit=self.trading_chat_history_limit,
        )
        response = open_trade_chat(observation)
        self._append_player_prompt_traces(player)
        return TradeChatOpenResponse(
            open_chat=True,
            message=self._truncate_chat_message(response.message),
            requested_resources=(
                self._normalize_resource_map(response.requested_resources)
                if response.requested_resources
                else requested_resources
            ),
            reasoning=response.reasoning,
        )

    def _reply_trade_chat(
        self,
        *,
        player_id: str,
        decision: DecisionPoint,
        attempt_index: int,
        round_index: int,
        requested_resources: dict[str, JsonValue],
        proposals: tuple[TradeChatProposal, ...],
    ) -> TradeChatReplyResponse:
        player = self.players[player_id]
        respond_trade_chat = getattr(player, "respond_trade_chat", None)
        if not callable(respond_trade_chat):
            return TradeChatReplyResponse()
        observation = self.observation_builder.build_trade_chat(
            engine=self.engine,
            player_id=player_id,
            owner_player_id=decision.acting_player_id,
            decision=decision,
            stage="reply",
            attempt_index=attempt_index,
            round_index=round_index,
            requested_resources=requested_resources,
            proposals=proposals,
            event_log=self.event_log,
            memory_store=self.memory_store,
            message_char_limit=self.trading_chat_message_chars,
            transcript_limit=self.trading_chat_history_limit,
        )
        response = respond_trade_chat(observation)
        self._append_player_prompt_traces(player)
        return TradeChatReplyResponse(
            message=self._truncate_chat_message(response.message),
            owner_gives=self._normalize_resource_map(response.owner_gives),
            owner_gets=self._normalize_resource_map(response.owner_gets),
            reasoning=response.reasoning,
        )

    def _decide_trade_chat(
        self,
        *,
        player: Player,
        decision: DecisionPoint,
        attempt_index: int,
        round_index: int,
        requested_resources: dict[str, JsonValue],
        proposals: tuple[TradeChatProposal, ...],
    ) -> TradeChatOwnerDecisionResponse:
        decide_trade_chat = getattr(player, "decide_trade_chat", None)
        if not callable(decide_trade_chat):
            decide_trade_chat = getattr(player, "select_trade_chat_offer", None)
        if not callable(decide_trade_chat):
            return TradeChatOwnerDecisionResponse(decision="close")
        observation = self.observation_builder.build_trade_chat(
            engine=self.engine,
            player_id=decision.acting_player_id,
            owner_player_id=decision.acting_player_id,
            decision=decision,
            stage="owner_decision",
            attempt_index=attempt_index,
            round_index=round_index,
            requested_resources=requested_resources,
            proposals=proposals,
            event_log=self.event_log,
            memory_store=self.memory_store,
            message_char_limit=self.trading_chat_message_chars,
            transcript_limit=self.trading_chat_history_limit,
        )
        response = decide_trade_chat(observation)
        self._append_player_prompt_traces(player)
        decision_value = getattr(response, "decision", None)
        selected_proposal_id = getattr(response, "selected_proposal_id", None)
        if not isinstance(decision_value, str):
            selected_player_id = getattr(response, "selected_player_id", None)
            if isinstance(selected_player_id, str):
                decision_value = "select"
                selected_proposal_id = self._selected_proposal_id_for_player(
                    selected_player_id=selected_player_id,
                    proposals=proposals,
                )
            else:
                decision_value = "close"
        decision_value = decision_value.lower()
        if decision_value not in {"continue", "select", "close"}:
            decision_value = "close"
        selected_proposal_id = self._coerce_selected_proposal_id_hint(
            proposals=proposals,
            selected_proposal_id=selected_proposal_id,
        )
        return TradeChatOwnerDecisionResponse(
            decision=decision_value,
            selected_proposal_id=selected_proposal_id,
            message=self._truncate_chat_message(getattr(response, "message", None)),
            reasoning=getattr(response, "reasoning", None),
        )

    def _record_public_events(
        self, events: tuple[Event, ...] | list[Event]
    ) -> tuple[Event, ...]:
        return self._store_events(events)

    def _trade_chat_participants(self, owner_player_id: str) -> tuple[str, ...]:
        return tuple(
            player_id
            for player_id in self.engine.player_ids
            if player_id != owner_player_id
        )

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
        round_index: int,
        proposal: TradeChatProposal | None = None,
    ) -> Event:
        payload: dict[str, JsonValue] = {
            "owner_player_id": owner_player_id,
            "speaker_player_id": speaker_player_id,
            "attempt_index": attempt_index,
            "round_index": round_index,
        }
        if message is not None:
            payload["message"] = self._truncate_chat_message(message)
        if proposal is not None and proposal.owner_gives and proposal.owner_gets:
            payload["proposal_id"] = proposal.proposal_id
            payload["offer"] = dict(proposal.owner_gives)
            payload["request"] = dict(proposal.owner_gets)
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

    def _validate_trade_chat_open_turn_constraints(
        self,
        *,
        decision: DecisionPoint,
        room_signature: str | None,
    ) -> None:
        if room_signature is None or decision.phase != "play_turn":
            return
        state = self._trade_chat_turn_state
        if state is None:
            return
        if room_signature in state.opened_room_signatures:
            raise InvalidActionError(
                "Cannot reopen the same public trade room in the same turn. "
                "Choose a meaningfully different trade request or another action."
            )

    @staticmethod
    def _next_trade_chat_proposal_id(
        *, attempt_index: int, round_index: int, proposal_count: int
    ) -> str:
        return f"attempt-{attempt_index}-round-{round_index}-proposal-{proposal_count}"

    @staticmethod
    def _selected_proposal_id_for_player(
        *, selected_player_id: str, proposals: tuple[TradeChatProposal, ...]
    ) -> str | None:
        normalized_player_id = selected_player_id.upper()
        for proposal in reversed(proposals):
            if proposal.player_id == normalized_player_id:
                return proposal.proposal_id
        return None

    @staticmethod
    def _coerce_selected_proposal_id_hint(
        *,
        proposals: tuple[TradeChatProposal, ...],
        selected_proposal_id: str | None,
    ) -> str | None:
        if isinstance(selected_proposal_id, str):
            for proposal in proposals:
                if proposal.proposal_id == selected_proposal_id:
                    return selected_proposal_id

            hint_tokens = {
                token
                for token in re.split(r"[^A-Z0-9]+", selected_proposal_id.upper())
                if token
            }
            matching_players = {
                proposal.player_id
                for proposal in proposals
                if proposal.player_id in hint_tokens
            }
            if len(matching_players) == 1:
                matching_player_id = next(iter(matching_players))
                return GameOrchestrator._selected_proposal_id_for_player(
                    selected_player_id=matching_player_id,
                    proposals=proposals,
                )

        if len(proposals) == 1:
            return proposals[0].proposal_id
        return None

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

    def _refresh_trade_chat_turn_state(
        self, *, turn_owner_id: str, decision: DecisionPoint
    ) -> None:
        if decision.phase != "play_turn":
            return
        state = self._trade_chat_turn_state
        if (
            state is None
            or state.owner_player_id != turn_owner_id
            or state.turn_index != decision.turn_index
        ):
            self._trade_chat_turn_state = _TradeChatTurnState(
                owner_player_id=turn_owner_id,
                turn_index=decision.turn_index,
            )

    def _player_has_resources(
        self, player_id: str, resources: dict[str, JsonValue]
    ) -> bool:
        """Return True if player_id holds at least the given resource amounts."""
        try:
            private = dict(self.engine.private_state(player_id))
            holdings = private.get("resources", {})
            if not isinstance(holdings, dict):
                return True
            return all(
                isinstance(amount, int) and int(holdings.get(resource, 0)) >= amount
                for resource, amount in resources.items()
            )
        except Exception:
            return True

    def _normalize_resource_map(
        self, value: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for resource, amount in value.items():
            if not isinstance(resource, str):
                continue
            if not isinstance(amount, int) or amount <= 0:
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
        if callable(take_last_prompt_trace):
            prompt_trace = take_last_prompt_trace()
            if prompt_trace is not None:
                self.prompt_trace_store.append(prompt_trace)
                self._report_prompt_trace(prompt_trace)

    def _report_prompt_trace(self, prompt_trace) -> None:
        if self.reporter is None:
            return
        hook = getattr(self.reporter, "on_prompt_trace", None)
        if callable(hook):
            hook(prompt_trace)

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
            action
            for action in legal_actions
            if action.action_type == proposed_action.action_type
        ]
        if proposed_action.action_type in {
            "ACCEPT_TRADE",
            "REJECT_TRADE",
            "CANCEL_TRADE",
        }:
            if len(matching_actions) == 1:
                return matching_actions[0]
        if proposed_action.action_type == "CONFIRM_TRADE":
            accepting_player_id = proposed_action.payload.get("accepting_player_id")
            if len(matching_actions) == 1 and accepting_player_id is None:
                return matching_actions[0]
            if isinstance(accepting_player_id, str):
                for legal_action in matching_actions:
                    if (
                        legal_action.payload.get("accepting_player_id")
                        == accepting_player_id
                    ):
                        return legal_action
        raise InvalidActionError(
            f"Action {proposed_action.to_dict()} is not in the current legal action set."
        )

    @staticmethod
    def _fallback_action_for_decision(legal_actions: tuple[Action, ...]) -> Action:
        for action in legal_actions:
            if not (
                action.action_type in {"OFFER_TRADE", "COUNTER_OFFER"}
                and action.payload == {"offer": {}, "request": {}}
            ):
                return action
        if not legal_actions:
            raise InvalidActionError("No legal fallback action was available.")
        return legal_actions[0]

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
        trade_metrics = {
            metric: 0
            for metric in (
                "offers",
                "accepted",
                "rejected",
                "confirmed",
                "cancelled",
                "chat_windows_opened",
                "chat_messages",
                "quotes_selected",
                "no_deals",
            )
        }
        for event in self.event_log.public_events:
            if event.kind == "trade_offered":
                trade_metrics["offers"] += 1
            elif event.kind == "trade_accepted":
                trade_metrics["accepted"] += 1
            elif event.kind == "trade_rejected":
                trade_metrics["rejected"] += 1
            elif event.kind == "trade_confirmed":
                trade_metrics["confirmed"] += 1
            elif event.kind == "trade_cancelled":
                trade_metrics["cancelled"] += 1
            elif event.kind == "trade_chat_opened":
                trade_metrics["chat_windows_opened"] += 1
            elif event.kind == "trade_chat_message":
                trade_metrics["chat_messages"] += 1
            elif event.kind == "trade_chat_quote_selected":
                trade_metrics["quotes_selected"] += 1
            elif event.kind == "trade_chat_no_deal":
                trade_metrics["no_deals"] += 1
        return {
            "history_window": self.observation_builder.recent_event_window,
            "run_directory": None if self.run_dir is None else str(self.run_dir),
            "decision_phase_counts": dict(sorted(self._decision_phase_counts.items())),
            "trade_metrics": trade_metrics,
            "trade_event_share": (
                0.0
                if total_decisions == 0
                else round(sum(trade_metrics.values()) / total_decisions, 4)
            ),
        }
