from __future__ import annotations

import json
from pathlib import Path
import re

try:
    from jinja2 import Environment, FileSystemLoader
except ModuleNotFoundError:  # pragma: no cover - exercised in lean test envs.
    Environment = None
    FileSystemLoader = None


_RESOURCE_ORDER = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
_DEV_CARD_ORDER = (
    "KNIGHT",
    "YEAR_OF_PLENTY",
    "MONOPOLY",
    "ROAD_BUILDING",
    "VICTORY_POINT",
)
# Keys that are harness metadata or already in the system prompt — excluded from payload_json
_NOISE_KEYS = frozenset(
    {"game_rules", "game_id", "history_index", "phase", "decision_index"}
)


def fmt_resources(resources: object) -> str:
    """Compact human-readable resource summary, e.g. '2×WOOD, 1×ORE'."""
    if not isinstance(resources, dict):
        return "none"
    parts = [f"{resources[r]}×{r}" for r in _RESOURCE_ORDER if resources.get(r)]
    return ", ".join(parts) or "none"


def fmt_resource_card_count(resources: object) -> str:
    """Total number of resource cards represented by a resource-count mapping."""
    if not isinstance(resources, dict):
        return "?"
    total = 0
    for amount in resources.values():
        if not isinstance(amount, int):
            return "?"
        total += amount
    return str(total)


def fmt_dev_cards(dev_cards: object) -> str:
    """Compact non-zero dev card summary, e.g. '2×KNIGHT, 1×MONOPOLY'."""
    if not isinstance(dev_cards, dict):
        return "none"
    parts = [f"{dev_cards[c]}×{c}" for c in _DEV_CARD_ORDER if dev_cards.get(c)]
    return ", ".join(parts) or "none"


def fmt_ports(ports: object) -> str:
    """Human-readable port summary with trade rates."""
    if not isinstance(ports, (list, tuple)):
        return "none"
    formatted: list[str] = []
    for port in ports:
        if not isinstance(port, str):
            continue
        if port == "ANY":
            formatted.append("ANY (3:1)")
        else:
            formatted.append(f"{port} (2:1)")
    return ", ".join(formatted) or "none"


def fmt_pieces(private_state: object) -> str:
    """'10R / 3S / 4C' from private_state dict, gracefully handling missing keys."""
    if not isinstance(private_state, dict):
        return ""
    pieces = private_state.get("pieces") or {}
    roads = pieces.get("roads", pieces.get("roads_available", "?"))
    settlements = pieces.get("settlements", pieces.get("settlements_available", "?"))
    cities = pieces.get("cities", pieces.get("cities_available", "?"))
    return f"{roads}R / {settlements}S / {cities}C"


def fmt_pieces_long(private_state: object) -> str:
    """'10 Roads / 3 Settlements / 4 Cities' from private_state dict."""
    if not isinstance(private_state, dict):
        return ""
    pieces = private_state.get("pieces") or {}
    roads = pieces.get("roads", pieces.get("roads_available", "?"))
    settlements = pieces.get("settlements", pieces.get("settlements_available", "?"))
    cities = pieces.get("cities", pieces.get("cities_available", "?"))
    return f"{roads} Roads / {settlements} Settlements / {cities} Cities"


def fmt_vp(private_state: object) -> str:
    """'5 (4 public, 1 private)' from private_state dict."""
    if not isinstance(private_state, dict):
        return "?"
    vp = private_state.get("victory_points") or {}
    actual = vp.get("actual", "?")
    visible = vp.get("visible", "?")
    private = "?"
    if isinstance(actual, int) and isinstance(visible, int):
        private = max(actual - visible, 0)
    return f"{actual} ({visible} public, {private} private)"


def fmt_vp_remaining(private_state: object, public_state: object) -> str:
    if not isinstance(private_state, dict):
        return "?"
    target = 10
    if isinstance(public_state, dict):
        turn = public_state.get("turn")
        if isinstance(turn, dict):
            maybe_target = turn.get("vps_to_win")
            if isinstance(maybe_target, int):
                target = maybe_target
        maybe_target = public_state.get("vps_to_win")
        if isinstance(maybe_target, int):
            target = maybe_target
    vp = private_state.get("victory_points") or {}
    actual = vp.get("actual")
    if not isinstance(actual, int):
        return f"? to reach {target} VP"
    return f"{max(target - actual, 0)} to reach {target} VP"


def fmt_payload(payload: object) -> str:
    """Stable JSON rendering for action payload snippets."""
    if payload in ({}, None):
        return "{}"
    return json.dumps(payload, sort_keys=True)


def fmt_memory(value: object) -> str:
    """Render memory naturally when it is text, otherwise as stable JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _fmt_public_tile_summary(tile: object) -> str | None:
    if not isinstance(tile, dict):
        return None
    tile_type = str(tile.get("type", "UNKNOWN"))
    if tile_type == "RESOURCE_TILE":
        resource = tile.get("resource")
        number = tile.get("number")
        if resource is None or number is None:
            return None
        return f"{resource}@{int(number)}"
    if tile_type == "DESERT":
        return "DESERT"
    if tile_type == "PORT":
        resource = tile.get("resource")
        return f"PORT:{'ANY' if resource is None else resource}"
    if tile_type == "WATER":
        return "WATER"
    return tile_type


def fmt_player_standing(
    player_id: object,
    info: object,
    viewer_player_id: object | None = None,
) -> str:
    if not isinstance(info, dict):
        return str(player_id)
    vp = info.get("vp", info.get("visible_victory_points", "?"))
    resource_cards = info.get("resource_card_count", info.get("res_cards", "?"))
    development_cards = info.get("development_card_count", info.get("dev_cards", "?"))
    resource_card_noun = "card" if resource_cards == 1 else "cards"
    development_card_noun = "card" if development_cards == 1 else "cards"
    roads_left = info.get("roads_left", info.get("roads_available", "?"))
    settlements_left = info.get(
        "settlements_left", info.get("settlements_available", "?")
    )
    cities_left = info.get("cities_left", info.get("cities_available", "?"))
    player_label = str(player_id)
    if player_id == viewer_player_id:
        player_label = f"{player_label} [this is you]"
    parts = [
        f"{player_label}: {vp}VP",
        f"{resource_cards} resource {resource_card_noun}",
        f"{development_cards} unused development {development_card_noun}",
        f"pieces left {roads_left}R/{settlements_left}S/{cities_left}C",
        f"longest road {info.get('longest_road_length', 0)}",
        f"played knights {info.get('played_knights', 0)}",
    ]
    if info.get("has_longest_road", info.get("longest_road")):
        parts.append("[Longest Road]")
    if info.get("has_largest_army", info.get("largest_army")):
        parts.append("[Largest Army]")
    return "  ".join(str(part) for part in parts)


def fmt_longest_road_status(players: object, player_id: object) -> str:
    if not isinstance(players, dict):
        return "Longest Road status unavailable."
    own = players.get(player_id, {}) if isinstance(player_id, str) else {}
    own_len = (
        int(dict(own).get("longest_road_length", 0)) if isinstance(own, dict) else 0
    )
    for pid, info in players.items():
        if isinstance(info, dict) and info.get(
            "has_longest_road", info.get("longest_road")
        ):
            return (
                f"{pid} currently holds Longest Road at length "
                f"{int(info.get('longest_road_length', 0))}. "
                f"Your current longest road is {own_len}."
            )
    return (
        f"No one currently holds Longest Road. Your current longest road is {own_len}."
    )


def fmt_largest_army_status(players: object, player_id: object) -> str:
    if not isinstance(players, dict):
        return "Largest Army status unavailable."
    own = players.get(player_id, {}) if isinstance(player_id, str) else {}
    own_knights = (
        int(dict(own).get("played_knights", 0)) if isinstance(own, dict) else 0
    )
    for pid, info in players.items():
        if isinstance(info, dict) and info.get(
            "has_largest_army", info.get("largest_army")
        ):
            return (
                f"{pid} currently holds Largest Army with "
                f"{int(info.get('played_knights', 0))} played knights. "
                f"Your played knights count is {own_knights}. "
                "Unused knights in hand do not count yet."
            )
    return (
        f"No one currently holds Largest Army. Your played knights count is {own_knights}. "
        "Unused knights in hand do not count yet."
    )


def fmt_robber_status(board: object) -> str:
    if not isinstance(board, dict):
        return "?"
    coordinate = board.get("robber_coordinate")
    coordinate_text = str(coordinate) if coordinate is not None else "?"
    tile_summary = board.get("robber_tile_summary")
    if not isinstance(tile_summary, str) or not tile_summary:
        tiles = board.get("tiles")
        if (
            isinstance(coordinate, (list, tuple))
            and len(coordinate) == 3
            and isinstance(tiles, list)
        ):
            target = tuple(int(axis) for axis in coordinate)
            for tile_entry in tiles:
                if not isinstance(tile_entry, dict):
                    continue
                entry_coordinate = tile_entry.get("coordinate")
                if not (
                    isinstance(entry_coordinate, (list, tuple))
                    and len(entry_coordinate) == 3
                ):
                    continue
                if tuple(int(axis) for axis in entry_coordinate) != target:
                    continue
                tile = tile_entry.get("tile")
                summary = _fmt_public_tile_summary(tile)
                if isinstance(summary, str) and summary and summary != "WATER":
                    tile_summary = summary
                break
    if isinstance(tile_summary, str) and tile_summary:
        return f"{coordinate_text} (on {tile_summary})"
    return coordinate_text


def fmt_board_building(building: object) -> str:
    if not isinstance(building, dict):
        return str(building)
    building_name = building.get("building", "?")
    node_id = building.get("node_id", "?")
    adjacent_tiles = building.get("adjacent_tiles")
    ports = building.get("ports")
    result = f"{building_name} at node {node_id}"
    if isinstance(adjacent_tiles, (list, tuple)):
        tiles = [str(tile) for tile in adjacent_tiles if isinstance(tile, str) and tile]
        if tiles:
            result += f" ({', '.join(tiles)})"
    if isinstance(ports, (list, tuple)):
        port_values = [str(port) for port in ports if isinstance(port, str) and port]
        if port_values:
            result += f" [port: {', '.join(port_values)}]"
    return result


def fmt_other_player_network(network: object) -> str:
    if not isinstance(network, dict):
        return str(network)
    player_id = network.get("player_id", "?")
    roads_built = network.get("roads_built", "?")
    road_noun = "road" if roads_built == 1 else "roads"
    parts = [f"{player_id}: {roads_built} {road_noun} built"]
    buildings = network.get("buildings")
    building_parts = []
    if isinstance(buildings, (list, tuple)):
        building_parts = [
            fmt_board_building(building)
            for building in buildings
            if isinstance(building, dict)
        ]
    parts.extend(building_parts or ["no public buildings"])
    return "; ".join(parts)


def fmt_event(event: object) -> str:
    """Single-line prose description of a public event dict."""
    if not isinstance(event, dict):
        return str(event)
    kind = str(event.get("kind", "unknown"))
    actor = str(event.get("actor_player_id") or "SYSTEM")
    p = event.get("payload") or {}
    if not isinstance(p, dict):
        p = {}

    if kind == "dice_rolled":
        return f"{actor} rolled {p.get('result', '?')}"
    if kind == "settlement_built":
        return f"{actor} built settlement at node {p.get('node_id', '?')}"
    if kind == "city_built":
        return f"{actor} upgraded to city at node {p.get('node_id', '?')}"
    if kind == "road_built":
        return f"{actor} built road on edge {p.get('edge', '?')}"
    if kind == "resources_discarded":
        count = p.get("discarded_count", "?")
        noun = "resource" if count == 1 else "resources"
        return f"{actor} discarded {count} {noun} for the robber"
    if kind == "robber_moved":
        coord = p.get("coordinate", "?")
        victim = p.get("victim")
        if victim:
            return f"{actor} moved robber to {coord}, stole from {victim}"
        return f"{actor} moved robber to {coord}"
    if kind == "turn_ended":
        return f"{actor} ended turn"
    if kind == "development_card_played":
        card = p.get("card_type") or p.get("dev_card_type") or "dev card"
        return f"{actor} played {card}"
    if kind == "action_taken":
        return f"{actor}: {p.get('action_type', kind)}"
    if kind == "trade_offered":
        offer = fmt_resources(p.get("offer", {}))
        req = fmt_resources(p.get("request", {}))
        return f"{actor} offered {offer} for {req}"
    if kind == "trade_accepted":
        return f"{actor} accepted trade"
    if kind == "trade_rejected":
        return f"{actor} rejected trade"
    if kind == "trade_counter_offered":
        offer = fmt_resources(p.get("offer", {}))
        req = fmt_resources(p.get("request", {}))
        owner = p.get("owner_player_id", "?")
        return f"{actor} counteroffered to {owner}: gave {offer}, wanted {req}"
    if kind == "trade_confirmed":
        offer = fmt_resources(p.get("offer", {}))
        req = fmt_resources(p.get("request", {}))
        accepting = p.get("accepting_player_id", "?")
        return f"{actor} confirmed trade with {accepting}: gave {offer}, received {req}"
    if kind == "trade_cancelled":
        return f"{actor} cancelled trade"
    if kind == "public_chat_message":
        msg = p.get("message", "")
        target = p.get("target_player_id")
        if isinstance(target, str) and target:
            return f"{actor} to {target} (public): {msg}"
        return f"{actor}: {msg}" if msg else f"{actor} spoke publicly"
    if kind == "trade_chat_opened":
        req = fmt_resources(p.get("requested_resources", {}))
        return f"{actor} opened trade chat requesting {req}"
    if kind == "trade_chat_message":
        msg = p.get("message", "")
        offer = p.get("offer")
        request = p.get("request")
        if offer or request:
            terms = f" [proposal: owner gives {fmt_resources(offer)}, gets {fmt_resources(request)}]"
            return f"{actor}: {msg}{terms}" if msg else f"{actor} proposed trade{terms}"
        return f"{actor}: {msg}" if msg else f"{actor} sent trade quote"
    if kind == "trade_chat_proposal_rejected":
        speaker = p.get("speaker_player_id", actor)
        reason = p.get("reason", "invalid proposal")
        return f"SYSTEM: {speaker}'s proposal was rejected ({reason})"
    if kind == "trade_chat_closed":
        outcome = p.get("outcome", "?")
        return f"{actor} closed trade chat (outcome: {outcome})"
    return f"{actor}: {kind.replace('_', ' ')}"


class PromptRenderer:
    """Renders prompt stages from package-local Jinja templates."""

    def __init__(self, templates_dir: str | Path | None = None) -> None:
        if templates_dir is None:
            templates_dir = Path(__file__).with_name("templates")
        self._templates_dir = Path(templates_dir)
        self._env = None
        if Environment is not None and FileSystemLoader is not None:
            env = Environment(
                loader=FileSystemLoader(str(self._templates_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
            )
            env.globals["fmt_resources"] = fmt_resources
            env.globals["fmt_resource_card_count"] = fmt_resource_card_count
            env.globals["fmt_dev_cards"] = fmt_dev_cards
            env.globals["fmt_ports"] = fmt_ports
            env.globals["fmt_pieces"] = fmt_pieces
            env.globals["fmt_pieces_long"] = fmt_pieces_long
            env.globals["fmt_vp"] = fmt_vp
            env.globals["fmt_vp_remaining"] = fmt_vp_remaining
            env.globals["fmt_payload"] = fmt_payload
            env.globals["fmt_memory"] = fmt_memory
            env.globals["fmt_player_standing"] = fmt_player_standing
            env.globals["fmt_longest_road_status"] = fmt_longest_road_status
            env.globals["fmt_largest_army_status"] = fmt_largest_army_status
            env.globals["fmt_robber_status"] = fmt_robber_status
            env.globals["fmt_board_building"] = fmt_board_building
            env.globals["fmt_other_player_network"] = fmt_other_player_network
            env.globals["fmt_event"] = fmt_event
            self._env = env

    def render_messages(
        self,
        *,
        system_template: str,
        user_template: str,
        payload: dict[str, object],
        **context: object,
    ) -> list[dict[str, object]]:
        clean_payload = {k: v for k, v in payload.items() if k not in _NOISE_KEYS}
        template_context = {
            **context,
            "payload": payload,
            "payload_json": json.dumps(clean_payload, sort_keys=True, indent=2),
        }
        system_content = self.render(system_template, **template_context)
        user_content = self.render(user_template, **template_context)
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def render(self, template_name: str, **context: object) -> str:
        if self._env is not None:
            return self._env.get_template(template_name).render(**context).strip()
        return self._render_without_jinja(template_name, context).strip()

    def _render_without_jinja(
        self, template_name: str, context: dict[str, object]
    ) -> str:
        template_path = self._templates_dir / template_name
        template = template_path.read_text(encoding="utf-8")
        template = re.sub(
            r'{%\s*include\s+"([^"]+)"\s*%}',
            lambda match: self._render_without_jinja(match.group(1), context),
            template,
        )
        template = re.sub(
            r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}",
            lambda match: str(context.get(match.group(1), "")),
            template,
        )
        return template
