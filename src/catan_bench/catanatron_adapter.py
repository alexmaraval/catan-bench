from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .schemas import Action, DecisionPoint, Event, JsonValue, TransitionResult

logger = logging.getLogger(__name__)

RESOURCE_ORDER = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DEV_CARD_ORDER = (
    "KNIGHT",
    "YEAR_OF_PLENTY",
    "MONOPOLY",
    "ROAD_BUILDING",
    "VICTORY_POINT",
)
TOTAL_ROADS = 15
TOTAL_SETTLEMENTS = 5
TOTAL_CITIES = 4

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


VALID_PLAYER_IDS = ("RED", "BLUE", "ORANGE", "WHITE")


class CatanatronEngineAdapter:
    """EngineAdapter implementation backed by a live catanatron Game."""

    def __init__(
        self,
        *,
        game: Game | None = None,
        player_ids: Sequence[str] = ("RED", "BLUE", "ORANGE", "WHITE"),
        game_id: str | None = None,
        seed: int | None = None,
        discard_limit: int = 7,
        vps_to_win: int = 10,
        catan_map=None,
    ) -> None:
        if game is None:
            for pid in player_ids:
                if pid not in VALID_PLAYER_IDS:
                    raise ValueError(
                        f"Unsupported player id {pid!r}. "
                        f"Catanatron supports: {VALID_PLAYER_IDS}."
                    )
            players = [PassiveCatanatronPlayer(Color[player_id]) for player_id in player_ids]
            game = Game(
                players,
                seed=seed,
                discard_limit=discard_limit,
                vps_to_win=vps_to_win,
                catan_map=catan_map,
            )

        if game_id is not None:
            try:
                game.id = game_id
            except Exception:  # pragma: no cover - depends on catanatron internals.
                logger.debug("Unable to override catanatron game id during resume.", exc_info=True)

        self.game = game
        logger.debug("CatanatronEngineAdapter initialized with %d players", len(self.player_ids))

    @property
    def game_id(self) -> str:
        return self.game.id

    @property
    def player_ids(self) -> tuple[str, ...]:
        return tuple(color.value for color in self.game.state.colors)

    @property
    def turn_owner_id(self) -> str:
        state = self.game.state
        return state.colors[state.current_turn_index].value

    def is_terminal(self) -> bool:
        return (
            self.game.winning_color() is not None
            or self.game.state.num_turns >= TURNS_LIMIT
        )

    def current_decision(self) -> DecisionPoint:
        self._advance_past_self_trade_response()
        state = self.game.state
        legal_actions = tuple(
            self._native_action_to_action(native_action)
            for native_action in self.game.playable_actions
        )
        if self._can_offer_trade():
            legal_actions = legal_actions + (self._offer_trade_template(),)
        if self._can_counter_offer(legal_actions):
            legal_actions = legal_actions + (self._counter_offer_template(),)
        legal_actions = self._ordered_legal_actions(legal_actions)

        return DecisionPoint(
            acting_player_id=state.current_color().value,
            turn_index=state.num_turns,
            phase=self._phase_name(state.current_prompt),
            legal_actions=legal_actions,
            decision_index=len(state.action_records),
            prompt=self._decision_prompt(state.current_prompt, legal_actions=legal_actions),
        )

    def _advance_past_self_trade_response(self) -> None:
        while self._is_self_trade_response_state():
            logger.debug(
                "Auto-skipping self trade response for %s",
                self.game.state.current_color().value,
            )
            self.game.execute(
                NativeAction(
                    self.game.state.current_color(),
                    ActionType.REJECT_TRADE,
                    self.game.state.current_trade,
                ),
                validate_action=True,
            )

    def _is_self_trade_response_state(self) -> bool:
        state = self.game.state
        if state.current_prompt != ActionPrompt.DECIDE_TRADE:
            return False
        offering_player_id = self._trade_offering_player_id(state)
        if offering_player_id is None:
            return False
        return state.current_color().value == offering_player_id

    @staticmethod
    def _ordered_legal_actions(legal_actions: tuple[Action, ...]) -> tuple[Action, ...]:
        """Keep engine order stable, but avoid presenting END_TURN as the easiest default."""
        indexed = list(enumerate(legal_actions))
        indexed.sort(
            key=lambda item: (
                item[1].action_type == "END_TURN",
                item[0],
            )
        )
        return tuple(action for _, action in indexed)

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

    def public_state_for_decision(
        self,
        *,
        player_id: str,
        phase: str,
        legal_actions: Sequence[Action],
    ) -> Mapping[str, JsonValue]:
        state = self.game.state
        game_json = self._game_json()
        return {
            "players": {
                color.value: self._public_player_prompt_summary(color)
                for color in state.colors
            },
            "board": self._board_prompt_view(
                player_id=player_id,
                phase=phase,
                legal_actions=legal_actions,
                game_json=game_json,
            ),
            "turn": {
                "current_player_id": state.current_color().value,
                "turn_player_id": state.colors[state.current_turn_index].value,
                "num_turns": state.num_turns,
                "prompt": state.current_prompt.value,
                "is_discarding": state.is_discarding,
                "is_resolving_trade": state.is_resolving_trade,
            },
            "trade_state": self._trade_state_public(),
            "bank": {
                "resources": self._freqdeck_to_dict(state.resource_freqdeck),
            },
        }

    def private_state_for_decision(
        self,
        *,
        player_id: str,
        phase: str,
        legal_actions: Sequence[Action],
    ) -> Mapping[str, JsonValue]:
        color = Color[player_id]
        key = player_key(self.game.state, color)
        state = self.game.state

        prompt_state: dict[str, JsonValue] = {
            "player_id": player_id,
            "resources": self._resource_counts(key),
            "development_cards": self._development_card_counts(key),
            "pieces": {
                "roads": state.player_state[f"{key}_ROADS_AVAILABLE"],
                "settlements": state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"],
                "cities": state.player_state[f"{key}_CITIES_AVAILABLE"],
            },
            "victory_points": {
                "visible": state.player_state[f"{key}_VICTORY_POINTS"],
                "actual": state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"],
            },
        }
        if self._should_include_ports(phase=phase, legal_actions=legal_actions):
            prompt_state["ports"] = self._serialize_ports(
                state.board.get_player_port_resources(color)
            )
        discard_prompt = self._discard_prompt_summary(legal_actions)
        if discard_prompt is not None:
            prompt_state["discard_requirement"] = discard_prompt
        return prompt_state

    def resolve_action(
        self, *, proposed_action: Action, legal_actions: tuple[Action, ...]
    ) -> Action:
        game = getattr(self, "game", None)
        if (
            proposed_action.action_type == "CONFIRM_TRADE"
            and game is not None
            and proposed_action.payload.get("accepting_player_id")
            == self._trade_offering_player_id(game.state)
        ):
            raise ValueError("CONFIRM_TRADE cannot select the offering player as the counterparty.")
        for legal_action in legal_actions:
            if self._is_trade_template(legal_action):
                continue
            if legal_action.matches(proposed_action):
                return legal_action

        canonical_robber_action = self._canonicalize_move_robber_action(
            proposed_action=proposed_action,
            legal_actions=legal_actions,
        )
        if canonical_robber_action is not None:
            return canonical_robber_action

        canonical_trade_action = self._canonicalize_trade_resolution_action(
            proposed_action=proposed_action,
            legal_actions=legal_actions,
        )
        if canonical_trade_action is not None:
            return canonical_trade_action

        if proposed_action.action_type == "OFFER_TRADE" and self._can_offer_trade():
            offer = proposed_action.payload.get("offer", {})
            if isinstance(offer, Mapping) and not self._current_player_has_resources(offer):
                raise ValueError(
                    "Action "
                    f"{proposed_action.to_dict()} is not currently valid in catanatron: "
                    "offering player lacks the offered resources."
                )
            native_action = self._action_to_native(proposed_action)
            if catanatron_is_valid_action(
                self.game.playable_actions, self.game.state, native_action
            ):
                return self._native_action_to_action(native_action)

        raise ValueError(
            f"Action {proposed_action.to_dict()} is not currently valid in catanatron."
        )

    @staticmethod
    def _is_trade_template(action: Action) -> bool:
        return action.action_type == "OFFER_TRADE" and action.payload == {
            "offer": {},
            "request": {},
        }

    @staticmethod
    def _canonicalize_move_robber_action(
        *, proposed_action: Action, legal_actions: tuple[Action, ...]
    ) -> Action | None:
        if proposed_action.action_type != "MOVE_ROBBER":
            return None

        coordinate = proposed_action.payload.get("coordinate")
        if not isinstance(coordinate, list) or len(coordinate) != 3:
            return None

        matching_coordinates = [
            legal_action
            for legal_action in legal_actions
            if legal_action.action_type == "MOVE_ROBBER"
            and legal_action.payload.get("coordinate") == coordinate
        ]
        if len(matching_coordinates) == 1:
            return matching_coordinates[0]
        return None

    @staticmethod
    def _canonicalize_trade_resolution_action(
        *, proposed_action: Action, legal_actions: tuple[Action, ...]
    ) -> Action | None:
        if proposed_action.action_type not in {
            "ACCEPT_TRADE",
            "REJECT_TRADE",
            "CONFIRM_TRADE",
            "CANCEL_TRADE",
        }:
            return None

        matching_actions = [
            legal_action
            for legal_action in legal_actions
            if legal_action.action_type == proposed_action.action_type
        ]
        if not matching_actions:
            return None

        if len(matching_actions) == 1:
            return matching_actions[0]

        if proposed_action.action_type != "CONFIRM_TRADE":
            return None

        accepting_player_id = proposed_action.payload.get("accepting_player_id")
        if not isinstance(accepting_player_id, str):
            accepting_player_id = proposed_action.payload.get("player_id")
        if not isinstance(accepting_player_id, str):
            accepting_player_id = proposed_action.payload.get("with_player_id")
        if not isinstance(accepting_player_id, str):
            return None

        for legal_action in matching_actions:
            if legal_action.payload.get("accepting_player_id") == accepting_player_id:
                return legal_action
        return None

    def apply_action(self, action: Action) -> TransitionResult:
        logger.debug("Applying action %s for %s", action.action_type, self.game.state.current_color().value)
        state_before = self.game.state.copy()

        native_action = self._action_to_native(action)
        action_record = self.game.execute(native_action, validate_action=True)
        self._recompute_longest_road_state()

        state_after = self.game.state

        public_event = self._public_event_for_action(
            native_action=action_record.action,
            action_result=action_record.result,
            state_before=state_before,
            state_after=state_after,
        )

        terminal = self.is_terminal()
        result_metadata = dict(self.result()) if terminal else {}

        return TransitionResult(
            public_events=() if public_event is None else (public_event,),
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
                    **self._public_player_summary(color),
                    "actual_victory_points": self._actual_victory_points(color),
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
            "dev_victory_points": (
                state.player_state[f"{key}_VICTORY_POINT_IN_HAND"]
                + state.player_state[f"{key}_PLAYED_VICTORY_POINT"]
            ),
            "longest_road_length": state.player_state[f"{key}_LONGEST_ROAD_LENGTH"],
            "played_knights": state.player_state[f"{key}_PLAYED_KNIGHT"],
            "has_longest_road": bool(state.player_state[f"{key}_HAS_ROAD"]),
            "has_largest_army": bool(state.player_state[f"{key}_HAS_ARMY"]),
        }

    def _recompute_longest_road_state(self) -> None:
        state = self.game.state
        board = state.board
        lengths = {
            color: self._longest_road_length_for_color(color)
            for color in state.colors
        }

        previous_holder = next(
            (
                color
                for color in state.colors
                if state.player_state.get(f"{player_key(state, color)}_HAS_ROAD")
            ),
            None,
        )
        candidates = [color for color, length in lengths.items() if length >= 5]
        if candidates:
            max_length = max(lengths[color] for color in candidates)
            leaders = [color for color in candidates if lengths[color] == max_length]
        else:
            max_length = 0
            leaders = []

        if len(leaders) == 1:
            winner = leaders[0]
        elif previous_holder in leaders:
            winner = previous_holder
        else:
            winner = None

        for color in state.colors:
            key = player_key(state, color)
            if state.player_state[f"{key}_HAS_ROAD"]:
                state.player_state[f"{key}_HAS_ROAD"] = False
                state.player_state[f"{key}_VICTORY_POINTS"] -= 2
                state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] -= 2
            state.player_state[f"{key}_LONGEST_ROAD_LENGTH"] = lengths[color]

        if winner is not None:
            winner_key = player_key(state, winner)
            state.player_state[f"{winner_key}_HAS_ROAD"] = True
            state.player_state[f"{winner_key}_VICTORY_POINTS"] += 2
            state.player_state[f"{winner_key}_ACTUAL_VICTORY_POINTS"] += 2

        board.road_lengths = defaultdict(int, lengths)
        board.road_color = winner
        board.road_length = max_length if winner is not None else 0

    def _longest_road_length_for_color(self, color: Color) -> int:
        roads = self.game.state.board.roads
        unique_edges = {
            tuple(sorted(edge))
            for edge, edge_color in roads.items()
            if edge_color == color
        }
        if not unique_edges:
            return 0

        adjacency: dict[int, set[int]] = defaultdict(set)
        for start, end in unique_edges:
            adjacency[start].add(end)
            adjacency[end].add(start)

        best = 0
        for node in adjacency:
            best = max(
                best,
                self._dfs_longest_road_from_node(
                    color=color,
                    adjacency=adjacency,
                    node=node,
                    used_edges=set(),
                    path_length=0,
                ),
            )
        return best

    def _dfs_longest_road_from_node(
        self,
        *,
        color: Color,
        adjacency: Mapping[int, set[int]],
        node: int,
        used_edges: set[tuple[int, int]],
        path_length: int,
    ) -> int:
        best = path_length
        if path_length > 0 and self._is_enemy_building_node(node=node, color=color):
            return best

        for neighbor in adjacency.get(node, ()):
            edge = tuple(sorted((node, neighbor)))
            if edge in used_edges:
                continue
            used_edges.add(edge)
            best = max(
                best,
                self._dfs_longest_road_from_node(
                    color=color,
                    adjacency=adjacency,
                    node=neighbor,
                    used_edges=used_edges,
                    path_length=path_length + 1,
                ),
            )
            used_edges.remove(edge)
        return best

    def _is_enemy_building_node(self, *, node: int, color: Color) -> bool:
        building = self.game.state.board.buildings.get(node)
        return building is not None and building[0] != color

    def _public_player_prompt_summary(self, color: Color) -> dict[str, JsonValue]:
        key = player_key(self.game.state, color)
        state = self.game.state
        return {
            "vp": int(state.player_state[f"{key}_VICTORY_POINTS"]),
            "res_cards": sum(
                state.player_state[f"{key}_{resource}_IN_HAND"]
                for resource in RESOURCE_ORDER
            ),
            "dev_cards": sum(
                state.player_state[f"{key}_{card}_IN_HAND"] for card in DEV_CARD_ORDER
            ),
            "roads": TOTAL_ROADS - int(state.player_state[f"{key}_ROADS_AVAILABLE"]),
            "settlements": (
                TOTAL_SETTLEMENTS - int(state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"])
            ),
            "cities": TOTAL_CITIES - int(state.player_state[f"{key}_CITIES_AVAILABLE"]),
            "longest_road": bool(state.player_state[f"{key}_HAS_ROAD"]),
            "largest_army": bool(state.player_state[f"{key}_HAS_ARMY"]),
        }

    def _resource_counts(self, key: str) -> dict[str, int]:
        return {
            resource: int(self.game.state.player_state[f"{key}_{resource}_IN_HAND"])
            for resource in RESOURCE_ORDER
        }

    def _current_player_has_resources(self, resource_map: Mapping[str, JsonValue]) -> bool:
        color = self.game.state.current_color()
        key = player_key(self.game.state, color)
        for resource, amount in resource_map.items():
            if not isinstance(resource, str) or resource not in RESOURCE_ORDER:
                if isinstance(amount, int) and amount > 0:
                    return False
                continue
            if not isinstance(amount, int) or amount < 0:
                return False
            if int(self.game.state.player_state[f"{key}_{resource}_IN_HAND"]) < amount:
                return False
        return True

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

    def _board_prompt_view(
        self,
        *,
        player_id: str,
        phase: str,
        legal_actions: Sequence[Action],
        game_json: Mapping[str, Any],
    ) -> dict[str, JsonValue]:
        _ = phase
        board_view: dict[str, JsonValue] = {
            "robber_coordinate": list(game_json["robber_coordinate"]),
            "your_network": self._player_network_view(
                player_id=player_id,
                game_json=game_json,
            ),
        }
        robber_targets = self._robber_targets_view(legal_actions)
        if robber_targets:
            board_view["robber_targets"] = robber_targets
        board_view.update(
            self._placement_candidate_views(
                legal_actions=legal_actions,
                game_json=game_json,
            )
        )
        return board_view

    def _player_network_view(
        self,
        *,
        player_id: str,
        game_json: Mapping[str, Any],
    ) -> dict[str, JsonValue]:
        nodes = game_json["nodes"]
        edges = game_json["edges"]
        adjacent_tiles = game_json["adjacent_tiles"]

        owned_buildings = [
            node
            for node in nodes.values()
            if node.get("color") == player_id and node.get("building") is not None
        ]
        owned_buildings.sort(key=lambda node: int(node["id"]))
        owned_roads = [
            edge for edge in edges if edge.get("color") == player_id
        ]
        owned_roads.sort(key=lambda edge: tuple(int(node_id) for node_id in edge["id"]))

        network_tiles: dict[str, None] = {}
        buildings_payload: list[dict[str, JsonValue]] = []
        for node in owned_buildings:
            node_id = int(node["id"])
            tiles = [
                self._tile_prompt_value(tile)
                for tile in adjacent_tiles.get(str(node_id), [])
            ]
            for tile in tiles:
                network_tiles.setdefault(tile, None)
            buildings_payload.append(
                {
                    "node_id": node_id,
                    "building": str(node["building"]),
                    "adjacent_tiles": tiles,
                }
            )

        return {
            "buildings": buildings_payload,
            "roads": [{"edge": list(edge["id"])} for edge in owned_roads],
            "adjacent_tiles": list(network_tiles.keys()),
        }

    @staticmethod
    def _robber_targets_view(legal_actions: Sequence[Action]) -> list[dict[str, JsonValue]]:
        targets: dict[tuple[int, int, int], set[str]] = {}
        for action in legal_actions:
            if action.action_type != "MOVE_ROBBER":
                continue
            coordinate = action.payload.get("coordinate")
            if not isinstance(coordinate, list) or len(coordinate) != 3:
                continue
            key = tuple(int(axis) for axis in coordinate)
            victims = targets.setdefault(key, set())
            victim = action.payload.get("victim")
            if isinstance(victim, str):
                victims.add(victim)

        return [
            {
                "coordinate": list(coordinate),
                "victims": sorted(victims),
            }
            for coordinate, victims in sorted(targets.items())
        ]

    def _placement_candidate_views(
        self,
        *,
        legal_actions: Sequence[Action],
        game_json: Mapping[str, Any],
    ) -> dict[str, JsonValue]:
        candidate_views: dict[str, JsonValue] = {}

        settlement_candidates: list[dict[str, JsonValue]] = []
        city_candidates: list[dict[str, JsonValue]] = []
        road_candidates: list[dict[str, JsonValue]] = []
        for action_index, action in enumerate(legal_actions):
            if action.action_type == "BUILD_SETTLEMENT" and "node_id" in action.payload:
                settlement_candidates.append(
                    self._node_candidate_view(
                        action_index=action_index,
                        node_id=int(action.payload["node_id"]),
                        game_json=game_json,
                    )
                )
            elif action.action_type == "BUILD_CITY" and "node_id" in action.payload:
                city_candidates.append(
                    self._node_candidate_view(
                        action_index=action_index,
                        node_id=int(action.payload["node_id"]),
                        game_json=game_json,
                        include_owner=True,
                    )
                )
            elif action.action_type == "BUILD_ROAD" and "edge" in action.payload:
                road_candidates.append(
                    self._road_candidate_view(
                        action_index=action_index,
                        edge=tuple(int(node_id) for node_id in action.payload["edge"]),
                        game_json=game_json,
                    )
                )

        if settlement_candidates:
            candidate_views["settlement_candidates"] = settlement_candidates
        if city_candidates:
            candidate_views["city_candidates"] = city_candidates
        if road_candidates:
            candidate_views["road_candidates"] = road_candidates

        return candidate_views

    def _node_candidate_view(
        self,
        *,
        action_index: int,
        node_id: int,
        game_json: Mapping[str, Any],
        include_owner: bool = False,
    ) -> dict[str, JsonValue]:
        node = game_json["nodes"][str(node_id)]
        adjacent_tiles = game_json["adjacent_tiles"].get(str(node_id), [])

        tile_summaries: list[str] = []
        ports: list[str] = []
        for tile in adjacent_tiles:
            summary = self._tile_prompt_value(tile)
            if summary.startswith("PORT:"):
                ports.append(summary.removeprefix("PORT:"))
                continue
            tile_summaries.append(summary)

        candidate_view: dict[str, JsonValue] = {
            "action_index": action_index,
            "node_id": node_id,
            "adjacent_tiles": tile_summaries,
        }
        if ports:
            candidate_view["ports"] = sorted(set(ports))
        if include_owner:
            building = node.get("building")
            if building is not None:
                candidate_view["building"] = str(building)
            color = node.get("color")
            if color is not None:
                candidate_view["owner_player_id"] = str(color)
        return candidate_view

    def _road_candidate_view(
        self,
        *,
        action_index: int,
        edge: tuple[int, int],
        game_json: Mapping[str, Any],
    ) -> dict[str, JsonValue]:
        adjacent: dict[str, None] = {}
        for node_id in edge:
            for tile in game_json["adjacent_tiles"].get(str(node_id), []):
                summary = self._tile_prompt_value(tile)
                if not summary.startswith("PORT:") and summary not in ("WATER", "DESERT"):
                    adjacent.setdefault(summary, None)
        result: dict[str, JsonValue] = {
            "action_index": action_index,
            "edge": list(edge),
        }
        if adjacent:
            result["adjacent_tiles"] = list(adjacent.keys())
        return result

    @staticmethod
    def _should_include_ports(
        *,
        phase: str,
        legal_actions: Sequence[Action],
    ) -> bool:
        _ = phase
        return any(action.action_type == "MARITIME_TRADE" for action in legal_actions)

    @staticmethod
    def _tile_prompt_value(tile: Mapping[str, Any]) -> str:
        tile_type = str(tile.get("type", "UNKNOWN"))
        if tile_type == "RESOURCE_TILE":
            return f"{tile['resource']}@{int(tile['number'])}"
        if tile_type == "DESERT":
            return "DESERT"
        if tile_type == "PORT":
            resource = "ANY" if tile.get("resource") is None else str(tile["resource"])
            return f"PORT:{resource}"
        return tile_type

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

    def _counter_offer_template(self) -> Action:
        return Action(
            action_type="COUNTER_OFFER",
            payload={"offer": {}, "request": {}},
            description=(
                "Reject the current trade and propose a different exchange by "
                "specifying what you would give in `offer` and what you want from "
                "the active player in `request`, for example "
                "`{\"offer\": {\"BRICK\": 1}, \"request\": {\"WOOD\": 1}}`."
            ),
        )

    def _can_offer_trade(self) -> bool:
        state = self.game.state
        return (
            state.current_prompt == ActionPrompt.PLAY_TURN
            and player_has_rolled(state, state.current_color())
        )

    def _can_counter_offer(self, legal_actions: Sequence[Action]) -> bool:
        state = self.game.state
        if state.current_prompt != ActionPrompt.DECIDE_TRADE:
            return False
        return any(action.action_type == "REJECT_TRADE" for action in legal_actions)

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
            "END_TURN",
            "CANCEL_TRADE",
        }:
            return {}

        if action_type == "DISCARD":
            resource_map = self._discard_resource_map_from_value(value)
            if resource_map is not None:
                return {"resources": resource_map}
            if value is None:
                return {}
            return {"value": value}

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
            "END_TURN",
            "CANCEL_TRADE",
        }:
            return None

        if action_type == "DISCARD":
            resource_map = self._discard_resource_map_from_payload(payload)
            if resource_map is not None:
                return self._resource_map_to_freqdeck(resource_map)
            value = payload.get("value")
            if isinstance(value, list):
                return tuple(value)
            return value

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
    def _resource_summary(resource_map: Mapping[str, int]) -> str:
        parts = [
            f"{resource_map[resource]}×{resource}"
            for resource in RESOURCE_ORDER
            if resource_map.get(resource)
        ]
        return ", ".join(parts) or "none"

    @staticmethod
    def _resource_name(value: object) -> str | None:
        if isinstance(value, str):
            return value
        enum_value = getattr(value, "value", None)
        if isinstance(enum_value, str):
            return enum_value
        return None

    @classmethod
    def _normalize_resource_map(
        cls, resource_map: Mapping[object, object]
    ) -> dict[str, int] | None:
        normalized: dict[str, int] = {}
        for resource, amount in resource_map.items():
            resource_name = cls._resource_name(resource)
            if resource_name not in RESOURCE_ORDER:
                return None
            if not isinstance(amount, int):
                return None
            if amount <= 0:
                continue
            normalized[resource_name] = amount
        return {
            resource: normalized[resource]
            for resource in RESOURCE_ORDER
            if normalized.get(resource, 0) > 0
        }

    @classmethod
    def _discard_resource_map_from_value(cls, value: Any) -> dict[str, int] | None:
        if isinstance(value, Mapping):
            return cls._normalize_resource_map(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if len(value) == len(RESOURCE_ORDER) and all(isinstance(item, int) for item in value):
                return cls._freqdeck_to_dict(value)
            counts: dict[str, int] = {}
            for item in value:
                resource_name = cls._resource_name(item)
                if resource_name not in RESOURCE_ORDER:
                    return None
                counts[resource_name] = counts.get(resource_name, 0) + 1
            return counts
        return None

    @classmethod
    def _discard_resource_map_from_payload(
        cls, payload: Mapping[str, JsonValue]
    ) -> dict[str, int] | None:
        resources = payload.get("resources")
        if isinstance(resources, Mapping):
            return cls._normalize_resource_map(resources)
        value = payload.get("value")
        return cls._discard_resource_map_from_value(value)

    @classmethod
    def _discard_prompt_summary(
        cls, legal_actions: Sequence[Action]
    ) -> dict[str, JsonValue] | None:
        if not legal_actions or any(action.action_type != "DISCARD" for action in legal_actions):
            return None
        resource_map = cls._discard_resource_map_from_payload(legal_actions[0].payload)
        if resource_map is None:
            return None
        discard_count = sum(resource_map.values())
        if discard_count <= 0:
            return None
        return {
            "count": discard_count,
            "legal_options": len(legal_actions),
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

    @classmethod
    def _decision_prompt(
        cls, prompt: ActionPrompt, *, legal_actions: Sequence[Action] = ()
    ) -> str:
        discard_summary = cls._discard_prompt_summary(legal_actions)
        return {
            ActionPrompt.BUILD_INITIAL_SETTLEMENT: "Place your initial settlement.",
            ActionPrompt.BUILD_INITIAL_ROAD: "Place your initial road.",
            ActionPrompt.PLAY_TURN: "Choose an action for your turn.",
            ActionPrompt.DISCARD: (
                f"Choose which {discard_summary['count']} resource cards to discard "
                "for the robber event."
                if discard_summary is not None
                else "Choose which resources to discard for the robber event."
            ),
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
    ) -> Event | None:
        canonical_action = self._native_action_to_action(native_action)
        actor_player_id = native_action.color.value
        payload = self._public_event_payload(
            action=canonical_action,
            action_result=action_result,
            actor_player_id=actor_player_id,
            state_before=state_before,
            state_after=state_after,
        )
        if (
            canonical_action.action_type in {"ACCEPT_TRADE", "REJECT_TRADE"}
            and payload.get("offering_player_id") == actor_player_id
        ):
            return None
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
        elif action_type == "DISCARD":
            discard_map = self._discard_resource_map_from_payload(action.payload)
            payload["action"] = {"action_type": "DISCARD", "payload": {}}
            if discard_map is not None:
                payload["discarded_count"] = sum(discard_map.values())
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
            "DISCARD": "resources_discarded",
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

    @classmethod
    def _description_for_action(
        cls, action_type: str, payload: Mapping[str, JsonValue]
    ) -> str | None:
        if action_type == "BUILD_SETTLEMENT":
            return f"Build a settlement on node {payload['node_id']}."
        if action_type == "BUILD_CITY":
            return f"Upgrade node {payload['node_id']} to a city."
        if action_type == "BUILD_ROAD":
            return f"Build a road on edge {payload['edge']}."
        if action_type == "ROLL":
            return "Roll the dice."
        if action_type == "BUY_DEVELOPMENT_CARD":
            return "Buy a development card."
        if action_type == "PLAY_KNIGHT_CARD":
            return "Play a knight card; you will then move the robber."
        if action_type == "PLAY_ROAD_BUILDING":
            return "Play Road Building; you will then place two roads."
        if action_type == "DISCARD":
            discard_map = cls._discard_resource_map_from_payload(payload)
            if discard_map:
                return (
                    f"Discard {cls._resource_summary(discard_map)} "
                    "for the robber event."
                )
            return "Discard the required cards for the robber event."
        if action_type == "MOVE_ROBBER":
            coordinate = payload.get("coordinate")
            victim = payload.get("victim")
            if victim is not None:
                return f"Move the robber to {coordinate} and steal from {victim}."
            return f"Move the robber to {coordinate}."
        if action_type == "PLAY_YEAR_OF_PLENTY":
            return f"Take {payload.get('resources', [])} from the bank."
        if action_type == "PLAY_MONOPOLY":
            return f"Claim all {payload.get('resource')} from opponents."
        if action_type == "MARITIME_TRADE":
            return (
                f"Trade {payload.get('give', [])} to the bank for "
                f"{payload.get('receive')}."
            )
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
        if not state.is_resolving_trade:
            return None
        current_trade = state.current_trade
        if not current_trade or len(current_trade) <= 10:
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
