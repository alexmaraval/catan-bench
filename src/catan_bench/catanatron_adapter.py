from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .schemas import Action, DecisionPoint, Event, JsonValue, TransitionResult

RESOURCE_ORDER = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DEV_CARD_ORDER = (
    "KNIGHT",
    "YEAR_OF_PLENTY",
    "MONOPOLY",
    "ROAD_BUILDING",
    "VICTORY_POINT",
)

try:
    from catanatron import Game
    from catanatron.game import TURNS_LIMIT, is_valid_action as catanatron_is_valid_action
    from catanatron.json import GameEncoder
    from catanatron.models.enums import Action as NativeAction
    from catanatron.models.enums import ActionPrompt, ActionType
    from catanatron.models.player import Color, Player as CatanatronPlayer
    from catanatron.state_functions import player_has_rolled, player_key
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing.
    raise RuntimeError(
        "catanatron is required to use CatanatronEngineAdapter. "
        "Install the GitHub version of catanatron before importing this module."
    ) from exc


class PassiveCatanatronPlayer(CatanatronPlayer):
    """Placeholder player used because the harness drives decisions externally."""

    def decide(self, game, playable_actions):
        raise RuntimeError(
            "PassiveCatanatronPlayer should not be asked to decide; "
            "the benchmark harness should call Game.execute directly."
        )


class CatanatronEngineAdapter:
    """EngineAdapter implementation backed by a live catanatron Game."""

    def __init__(
        self,
        *,
        game: Game | None = None,
        player_ids: Sequence[str] = ("RED", "BLUE", "ORANGE", "WHITE"),
        seed: int | None = None,
        discard_limit: int = 7,
        vps_to_win: int = 10,
        catan_map=None,
    ) -> None:
        if game is None:
            players = [PassiveCatanatronPlayer(Color[player_id]) for player_id in player_ids]
            game = Game(
                players,
                seed=seed,
                discard_limit=discard_limit,
                vps_to_win=vps_to_win,
                catan_map=catan_map,
            )

        self.game = game

    @property
    def game_id(self) -> str:
        return self.game.id

    @property
    def player_ids(self) -> tuple[str, ...]:
        return tuple(color.value for color in self.game.state.colors)

    def is_terminal(self) -> bool:
        return (
            self.game.winning_color() is not None
            or self.game.state.num_turns >= TURNS_LIMIT
        )

    def current_decision(self) -> DecisionPoint:
        state = self.game.state
        legal_actions = tuple(
            self._native_action_to_action(native_action)
            for native_action in self.game.playable_actions
        )
        if self._can_offer_trade():
            legal_actions = legal_actions + (self._offer_trade_template(),)

        return DecisionPoint(
            acting_player_id=state.current_color().value,
            turn_index=state.num_turns,
            phase=self._phase_name(state.current_prompt),
            legal_actions=legal_actions,
            decision_index=len(state.action_records),
            prompt=self._decision_prompt(state.current_prompt),
        )

    def public_state(self) -> Mapping[str, JsonValue]:
        state = self.game.state
        game_json = self._game_json()

        return {
            "players": {
                color.value: self._public_player_summary(color)
                for color in state.colors
            },
            "board": {
                "tiles": game_json["tiles"],
                "nodes": game_json["nodes"],
                "edges": game_json["edges"],
                "adjacent_tiles": game_json["adjacent_tiles"],
                "robber_coordinate": game_json["robber_coordinate"],
            },
            "turn": {
                "current_player_id": state.current_color().value,
                "turn_player_id": state.colors[state.current_turn_index].value,
                "num_turns": state.num_turns,
                "prompt": state.current_prompt.value,
                "is_initial_build_phase": state.is_initial_build_phase,
                "is_discarding": state.is_discarding,
                "is_moving_knight": state.is_moving_knight,
                "is_resolving_trade": state.is_resolving_trade,
            },
            "trade_state": self._trade_state_public(),
            "bank": {
                "resources": self._freqdeck_to_dict(state.resource_freqdeck),
                "development_cards_remaining": len(state.development_listdeck),
            },
        }

    def private_state(self, player_id: str) -> Mapping[str, JsonValue]:
        color = Color[player_id]
        key = player_key(self.game.state, color)
        state = self.game.state

        return {
            "player_id": player_id,
            "resources": self._resource_counts(key),
            "development_cards": self._development_card_counts(key),
            "ports": self._serialize_ports(state.board.get_player_port_resources(color)),
            "pieces": {
                "roads_available": state.player_state[f"{key}_ROADS_AVAILABLE"],
                "settlements_available": state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"],
                "cities_available": state.player_state[f"{key}_CITIES_AVAILABLE"],
            },
            "victory_points": {
                "visible": state.player_state[f"{key}_VICTORY_POINTS"],
                "actual": state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"],
            },
            "status_flags": {
                "has_rolled": bool(state.player_state[f"{key}_HAS_ROLLED"]),
                "has_longest_road": bool(state.player_state[f"{key}_HAS_ROAD"]),
                "has_largest_army": bool(state.player_state[f"{key}_HAS_ARMY"]),
                "played_dev_card_this_turn": bool(
                    state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]
                ),
            },
        }

    def resolve_action(
        self, *, proposed_action: Action, legal_actions: tuple[Action, ...]
    ) -> Action:
        for legal_action in legal_actions:
            if legal_action.matches(proposed_action):
                return legal_action

        if proposed_action.action_type == "OFFER_TRADE" and self._can_offer_trade():
            native_action = self._action_to_native(proposed_action)
            if catanatron_is_valid_action(
                self.game.playable_actions, self.game.state, native_action
            ):
                return self._native_action_to_action(native_action)

        raise ValueError(
            f"Action {proposed_action.to_dict()} is not currently valid in catanatron."
        )

    def apply_action(self, action: Action) -> TransitionResult:
        state_before = self.game.state.copy()
        public_before = self.public_state()
        private_before = {
            player_id: dict(self.private_state(player_id)) for player_id in self.player_ids
        }

        native_action = self._action_to_native(action)
        action_record = self.game.execute(native_action, validate_action=True)

        state_after = self.game.state
        public_after = self.public_state()
        private_after = {
            player_id: dict(self.private_state(player_id)) for player_id in self.player_ids
        }

        public_event = self._public_event_for_action(
            native_action=action_record.action,
            action_result=action_record.result,
            state_before=state_before,
            state_after=state_after,
        )

        private_events_by_player: dict[str, tuple[Event, ...]] = {}
        for player_id in self.player_ids:
            diff = self._dict_diff(private_before[player_id], private_after[player_id])
            if diff:
                private_events_by_player[player_id] = (
                    Event(
                        kind="private_state_changed",
                        payload=diff,
                        turn_index=state_before.num_turns,
                        phase=self._phase_name(state_before.current_prompt),
                        decision_index=len(state_before.action_records),
                        actor_player_id=action_record.action.color.value,
                    ),
                )

        terminal = self.is_terminal()
        result_metadata = dict(self.result()) if terminal else {}

        if public_before != public_after:
            public_events = (public_event,)
        else:
            public_events = ()

        return TransitionResult(
            public_events=public_events,
            private_events_by_player=private_events_by_player,
            terminal=terminal,
            result_metadata=result_metadata,
        )

    def result(self) -> Mapping[str, JsonValue]:
        winner = self.game.winning_color()
        state = self.game.state

        return {
            "winner_id": None if winner is None else winner.value,
            "winner_ids": [] if winner is None else [winner.value],
            "num_turns": state.num_turns,
            "seed": self.game.seed,
            "vps_to_win": self.game.vps_to_win,
            "players": {
                color.value: {
                    "visible_victory_points": self._public_player_summary(color)[
                        "visible_victory_points"
                    ],
                    "actual_victory_points": self._actual_victory_points(color),
                    "resource_card_count": self._public_player_summary(color)[
                        "resource_card_count"
                    ],
                    "development_card_count": self._public_player_summary(color)[
                        "development_card_count"
                    ],
                }
                for color in state.colors
            },
        }

    def _public_player_summary(self, color: Color) -> dict[str, JsonValue]:
        key = player_key(self.game.state, color)
        state = self.game.state
        return {
            "seat_index": state.color_to_index[color],
            "visible_victory_points": state.player_state[f"{key}_VICTORY_POINTS"],
            "resource_card_count": sum(
                state.player_state[f"{key}_{resource}_IN_HAND"]
                for resource in RESOURCE_ORDER
            ),
            "development_card_count": sum(
                state.player_state[f"{key}_{card}_IN_HAND"] for card in DEV_CARD_ORDER
            ),
            "longest_road_length": state.player_state[f"{key}_LONGEST_ROAD_LENGTH"],
            "has_longest_road": bool(state.player_state[f"{key}_HAS_ROAD"]),
            "has_largest_army": bool(state.player_state[f"{key}_HAS_ARMY"]),
        }

    def _resource_counts(self, key: str) -> dict[str, int]:
        return {
            resource: int(self.game.state.player_state[f"{key}_{resource}_IN_HAND"])
            for resource in RESOURCE_ORDER
        }

    def _development_card_counts(self, key: str) -> dict[str, int]:
        return {
            card: int(self.game.state.player_state[f"{key}_{card}_IN_HAND"])
            for card in DEV_CARD_ORDER
        }

    def _actual_victory_points(self, color: Color) -> int:
        key = player_key(self.game.state, color)
        return int(self.game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"])

    def _trade_state_public(self) -> dict[str, JsonValue]:
        state = self.game.state
        acceptees = [
            color.value for color, accepted in zip(state.colors, state.acceptees) if accepted
        ]

        trade_summary: dict[str, JsonValue] = {
            "is_resolving_trade": state.is_resolving_trade,
            "acceptees": acceptees,
        }
        if state.is_resolving_trade:
            offering = self._freqdeck_to_dict(state.current_trade[:5])
            asking = self._freqdeck_to_dict(state.current_trade[5:10])
            offering_player_index = int(state.current_trade[10])
            trade_summary["offer"] = offering
            trade_summary["request"] = asking
            trade_summary["offering_player_id"] = state.colors[offering_player_index].value

        return trade_summary

    def _game_json(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.game, cls=GameEncoder))

    def _offer_trade_template(self) -> Action:
        return Action(
            action_type="OFFER_TRADE",
            payload={"offer": {}, "request": {}},
            description=(
                "Offer a domestic trade by specifying non-overlapping resource-count "
                "maps in `offer` and `request`, for example "
                "`{\"offer\": {\"WOOD\": 1}, \"request\": {\"BRICK\": 1}}`."
            ),
        )

    def _can_offer_trade(self) -> bool:
        state = self.game.state
        return (
            state.current_prompt == ActionPrompt.PLAY_TURN
            and player_has_rolled(state, state.current_color())
        )

    def _native_action_to_action(self, native_action: NativeAction) -> Action:
        action_type = native_action.action_type.value
        payload = self._native_value_to_payload(action_type, native_action.value)
        description = self._description_for_action(action_type, payload)
        return Action(action_type=action_type, payload=payload, description=description)

    def _action_to_native(self, action: Action) -> NativeAction:
        color = self.game.state.current_color()
        action_type = ActionType[action.action_type]
        value = self._payload_to_native_value(action.action_type, action.payload)
        return NativeAction(color, action_type, value)

    def _native_value_to_payload(
        self, action_type: str, value: Any
    ) -> dict[str, JsonValue]:
        if action_type in {
            "ROLL",
            "BUY_DEVELOPMENT_CARD",
            "PLAY_KNIGHT_CARD",
            "PLAY_ROAD_BUILDING",
            "DISCARD",
            "END_TURN",
            "CANCEL_TRADE",
        }:
            return {}

        if action_type in {"BUILD_SETTLEMENT", "BUILD_CITY"}:
            return {"node_id": int(value)}

        if action_type == "BUILD_ROAD":
            return {"edge": list(value)}

        if action_type == "MOVE_ROBBER":
            coordinate, victim = value
            return {
                "coordinate": list(coordinate),
                "victim": None if victim is None else victim.value,
            }

        if action_type == "PLAY_YEAR_OF_PLENTY":
            return {"resources": list(value)}

        if action_type == "PLAY_MONOPOLY":
            return {"resource": value}

        if action_type == "MARITIME_TRADE":
            giving = [resource for resource in value[:4] if resource is not None]
            return {"give": giving, "receive": value[4]}

        if action_type == "OFFER_TRADE":
            return {
                "offer": self._freqdeck_to_dict(value[:5]),
                "request": self._freqdeck_to_dict(value[5:10]),
            }

        if action_type in {"ACCEPT_TRADE", "REJECT_TRADE"}:
            offering_player = self.game.state.colors[int(value[10])].value
            return {
                "offer": self._freqdeck_to_dict(value[:5]),
                "request": self._freqdeck_to_dict(value[5:10]),
                "offering_player_id": offering_player,
            }

        if action_type == "CONFIRM_TRADE":
            return {
                "offer": self._freqdeck_to_dict(value[:5]),
                "request": self._freqdeck_to_dict(value[5:10]),
                "accepting_player_id": value[10].value,
            }

        return {"value": self._json_safe(value)}

    def _payload_to_native_value(self, action_type: str, payload: Mapping[str, JsonValue]) -> Any:
        if action_type in {
            "ROLL",
            "BUY_DEVELOPMENT_CARD",
            "PLAY_KNIGHT_CARD",
            "PLAY_ROAD_BUILDING",
            "DISCARD",
            "END_TURN",
            "CANCEL_TRADE",
        }:
            return None

        if action_type in {"BUILD_SETTLEMENT", "BUILD_CITY"}:
            return int(payload["node_id"])

        if action_type == "BUILD_ROAD":
            edge = payload["edge"]
            if not isinstance(edge, list) or len(edge) != 2:
                raise ValueError("BUILD_ROAD requires payload.edge with two node ids.")
            return tuple(int(node_id) for node_id in edge)

        if action_type == "MOVE_ROBBER":
            coordinate = payload["coordinate"]
            if not isinstance(coordinate, list) or len(coordinate) != 3:
                raise ValueError("MOVE_ROBBER requires payload.coordinate as a 3-item list.")
            victim = payload.get("victim")
            return (
                tuple(int(axis) for axis in coordinate),
                None if victim is None else Color[str(victim)],
            )

        if action_type == "PLAY_YEAR_OF_PLENTY":
            resources = payload["resources"]
            if not isinstance(resources, list) or not 1 <= len(resources) <= 2:
                raise ValueError(
                    "PLAY_YEAR_OF_PLENTY requires payload.resources with one or two resources."
                )
            return tuple(str(resource) for resource in resources)

        if action_type == "PLAY_MONOPOLY":
            return str(payload["resource"])

        if action_type == "MARITIME_TRADE":
            giving = payload["give"]
            receiving = payload["receive"]
            if not isinstance(giving, list) or len(giving) not in {2, 3, 4}:
                raise ValueError(
                    "MARITIME_TRADE requires payload.give with 2, 3, or 4 resources."
                )
            padded = [str(resource) for resource in giving]
            padded.extend([None] * (4 - len(padded)))
            return tuple(padded + [str(receiving)])

        if action_type == "OFFER_TRADE":
            offer = payload.get("offer", {})
            request = payload.get("request", {})
            if not isinstance(offer, Mapping) or not isinstance(request, Mapping):
                raise ValueError("OFFER_TRADE requires mapping payloads for offer and request.")
            return self._resource_map_to_freqdeck(offer) + self._resource_map_to_freqdeck(
                request
            )

        if action_type in {"ACCEPT_TRADE", "REJECT_TRADE"}:
            return self.game.state.current_trade

        if action_type == "CONFIRM_TRADE":
            accepting_player_id = payload.get("accepting_player_id")
            if not isinstance(accepting_player_id, str):
                raise ValueError("CONFIRM_TRADE requires payload.accepting_player_id.")
            return (*self.game.state.current_trade[:10], Color[accepting_player_id])

        value = payload.get("value")
        return value

    @staticmethod
    def _resource_map_to_freqdeck(resource_map: Mapping[str, JsonValue]) -> tuple[int, ...]:
        result = []
        for resource in RESOURCE_ORDER:
            amount = resource_map.get(resource, 0)
            if not isinstance(amount, int):
                raise ValueError(f"Resource count for {resource} must be an integer.")
            if amount < 0:
                raise ValueError(f"Resource count for {resource} cannot be negative.")
            result.append(amount)
        return tuple(result)

    @staticmethod
    def _freqdeck_to_dict(freqdeck: Sequence[Any]) -> dict[str, int]:
        return {
            resource: int(freqdeck[index])
            for index, resource in enumerate(RESOURCE_ORDER)
            if int(freqdeck[index]) > 0
        }

    @staticmethod
    def _serialize_ports(ports: Sequence[Any]) -> list[str]:
        result = []
        for port in ports:
            result.append("ANY" if port is None else str(port))
        return sorted(result)

    @staticmethod
    def _phase_name(prompt: ActionPrompt) -> str:
        return prompt.value.lower()

    @staticmethod
    def _decision_prompt(prompt: ActionPrompt) -> str:
        return {
            ActionPrompt.BUILD_INITIAL_SETTLEMENT: "Place your initial settlement.",
            ActionPrompt.BUILD_INITIAL_ROAD: "Place your initial road.",
            ActionPrompt.PLAY_TURN: "Choose an action for your turn.",
            ActionPrompt.DISCARD: "Discard when the robber has been triggered.",
            ActionPrompt.MOVE_ROBBER: "Move the robber and optionally steal a resource.",
            ActionPrompt.DECIDE_TRADE: "Respond to the current trade offer.",
            ActionPrompt.DECIDE_ACCEPTEES: "Choose which accepting player to trade with.",
        }.get(prompt, "Choose one legal action.")

    def _public_event_for_action(
        self,
        *,
        native_action: NativeAction,
        action_result: Any,
        state_before,
        state_after,
    ) -> Event:
        canonical_action = self._native_action_to_action(native_action)
        actor_player_id = native_action.color.value
        payload = self._public_event_payload(
            action=canonical_action,
            action_result=action_result,
            actor_player_id=actor_player_id,
            state_before=state_before,
            state_after=state_after,
        )
        return Event(
            kind=self._event_kind(canonical_action.action_type),
            payload=payload,
            turn_index=state_before.num_turns,
            phase=self._phase_name(state_before.current_prompt),
            decision_index=len(state_before.action_records),
            actor_player_id=actor_player_id,
        )

    def _public_event_payload(
        self,
        *,
        action: Action,
        action_result: Any,
        actor_player_id: str,
        state_before,
        state_after,
    ) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "action": action.to_dict(),
            "result": self._json_safe(action_result),
            "turn_player_id_before": state_before.colors[
                state_before.current_turn_index
            ].value,
            "turn_player_id_after": state_after.colors[
                state_after.current_turn_index
            ].value,
            "prompt_before": state_before.current_prompt.value,
            "prompt_after": state_after.current_prompt.value,
            "actor_player_id": actor_player_id,
            "trade_state": self._trade_state_summary(state_after),
        }

        action_type = action.action_type
        if action_type == "ROLL":
            payload["dice"] = self._json_safe(action_result)
        elif action_type == "OFFER_TRADE":
            payload["offering_player_id"] = actor_player_id
            payload["offer"] = dict(action.payload.get("offer", {}))
            payload["request"] = dict(action.payload.get("request", {}))
        elif action_type in {"ACCEPT_TRADE", "REJECT_TRADE"}:
            payload["offering_player_id"] = str(action.payload.get("offering_player_id"))
            payload["offer"] = dict(action.payload.get("offer", {}))
            payload["request"] = dict(action.payload.get("request", {}))
            payload["responding_player_id"] = actor_player_id
        elif action_type == "CONFIRM_TRADE":
            payload["offering_player_id"] = self._trade_offering_player_id(state_before)
            payload["accepting_player_id"] = str(action.payload.get("accepting_player_id"))
            payload["offer"] = dict(action.payload.get("offer", {}))
            payload["request"] = dict(action.payload.get("request", {}))
        elif action_type == "CANCEL_TRADE":
            payload["offering_player_id"] = self._trade_offering_player_id(state_before)
        elif action_type == "BUILD_SETTLEMENT":
            payload["node_id"] = action.payload["node_id"]
        elif action_type == "BUILD_CITY":
            payload["node_id"] = action.payload["node_id"]
        elif action_type == "BUILD_ROAD":
            payload["edge"] = list(action.payload["edge"])
        elif action_type == "MOVE_ROBBER":
            payload["coordinate"] = list(action.payload["coordinate"])
            payload["victim"] = action.payload.get("victim")
        elif action_type == "PLAY_YEAR_OF_PLENTY":
            payload["resources"] = list(action.payload["resources"])
        elif action_type == "PLAY_MONOPOLY":
            payload["resource"] = action.payload["resource"]

        return payload

    @staticmethod
    def _event_kind(action_type: str) -> str:
        return {
            "ROLL": "dice_rolled",
            "BUILD_SETTLEMENT": "settlement_built",
            "BUILD_CITY": "city_built",
            "BUILD_ROAD": "road_built",
            "MOVE_ROBBER": "robber_moved",
            "OFFER_TRADE": "trade_offered",
            "ACCEPT_TRADE": "trade_accepted",
            "REJECT_TRADE": "trade_rejected",
            "CONFIRM_TRADE": "trade_confirmed",
            "CANCEL_TRADE": "trade_cancelled",
            "END_TURN": "turn_ended",
            "PLAY_KNIGHT_CARD": "development_card_played",
            "PLAY_YEAR_OF_PLENTY": "development_card_played",
            "PLAY_MONOPOLY": "development_card_played",
            "PLAY_ROAD_BUILDING": "development_card_played",
        }.get(action_type, "action_taken")

    @staticmethod
    def _description_for_action(
        action_type: str, payload: Mapping[str, JsonValue]
    ) -> str | None:
        if action_type == "BUILD_SETTLEMENT":
            return f"Build a settlement on node {payload['node_id']}."
        if action_type == "BUILD_CITY":
            return f"Upgrade node {payload['node_id']} to a city."
        if action_type == "BUILD_ROAD":
            return f"Build a road on edge {payload['edge']}."
        if action_type == "ROLL":
            return "Roll the dice."
        if action_type == "END_TURN":
            return "End the current turn."
        if action_type == "OFFER_TRADE":
            return (
                f"Offer {payload.get('offer', {})} in exchange for "
                f"{payload.get('request', {})}."
            )
        if action_type == "ACCEPT_TRADE":
            return f"Accept the trade offered by {payload.get('offering_player_id')}."
        if action_type == "REJECT_TRADE":
            return f"Reject the trade offered by {payload.get('offering_player_id')}."
        if action_type == "CONFIRM_TRADE":
            return (
                f"Confirm the trade with {payload.get('accepting_player_id')}."
            )
        if action_type == "CANCEL_TRADE":
            return "Cancel the current trade offer."
        return None

    @staticmethod
    def _json_safe(value: Any) -> JsonValue:
        return json.loads(json.dumps(value, cls=GameEncoder))

    def _trade_state_summary(self, state) -> dict[str, JsonValue]:
        acceptees = [
            color.value for color, accepted in zip(state.colors, state.acceptees) if accepted
        ]
        summary: dict[str, JsonValue] = {
            "is_resolving_trade": state.is_resolving_trade,
            "acceptees": acceptees,
        }
        offering_player_id = self._trade_offering_player_id(state)
        if offering_player_id is not None:
            summary["offering_player_id"] = offering_player_id
            summary["offer"] = self._freqdeck_to_dict(state.current_trade[:5])
            summary["request"] = self._freqdeck_to_dict(state.current_trade[5:10])
        return summary

    @staticmethod
    def _trade_offering_player_id(state) -> str | None:
        if not getattr(state, "is_resolving_trade", False):
            return None
        current_trade = getattr(state, "current_trade", None)
        if not current_trade:
            return None
        if len(current_trade) <= 10:
            return None
        offering_player_index = current_trade[10]
        if offering_player_index is None:
            return None
        return state.colors[int(offering_player_index)].value

    @classmethod
    def _dict_diff(
        cls, before: Mapping[str, Any], after: Mapping[str, Any]
    ) -> dict[str, JsonValue]:
        diff: dict[str, JsonValue] = {}
        keys = set(before.keys()) | set(after.keys())
        for key in sorted(keys):
            before_value = before.get(key)
            after_value = after.get(key)
            if before_value == after_value:
                continue

            if isinstance(before_value, Mapping) and isinstance(after_value, Mapping):
                nested = cls._dict_diff(before_value, after_value)
                if nested:
                    diff[key] = nested
            else:
                diff[key] = {
                    "before": cls._json_safe(before_value),
                    "after": cls._json_safe(after_value),
                }
        return diff
