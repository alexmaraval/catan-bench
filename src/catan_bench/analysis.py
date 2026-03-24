"""Post-game analysis: compute structured metrics from completed game artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .schemas import Event, JsonValue, PromptTrace, PublicStateSnapshot
from .storage import read_json, read_jsonl, write_json

# ── Constants ────────────────────────────────────────────────────────────────

RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")

# Standard pip counts for each dice number (probability tokens).
PIPS: dict[int, int] = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

TRADE_EVENT_KINDS = {
    "trade_offered", "trade_accepted", "trade_rejected",
    "trade_confirmed", "trade_cancelled",
    "trade_chat_opened", "trade_chat_message",
    "trade_chat_quote_selected", "trade_chat_no_deal", "trade_chat_closed",
}

# ── ANSI helpers (mirroring reporter.py) ─────────────────────────────────────

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_PLAYER_ANSI: dict[str, str] = {
    "RED": "\033[31m",
    "BLUE": "\033[34m",
    "ORANGE": "\033[33m",
    "WHITE": "\033[97m",
    "GREEN": "\033[32m",
    "TEAL": "\033[36m",
}


def _c(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_analysis_artifacts(run_dir: Path) -> dict[str, Any]:
    """Load all artifacts needed for analysis from a completed run directory."""
    result = read_json(run_dir / "result.json")
    if result is None:
        raise FileNotFoundError(f"No result.json found in {run_dir} — game may not have ended.")

    metadata = read_json(run_dir / "metadata.json") or {}

    events = [
        Event.from_dict(entry)
        for entry in read_jsonl(run_dir / "public_history.jsonl")
    ]

    state_snapshots = [
        PublicStateSnapshot.from_dict(entry)
        for entry in read_jsonl(run_dir / "public_state_trace.jsonl")
    ]

    player_ids = [str(pid) for pid in (metadata.get("player_ids") or [])]
    if not player_ids and state_snapshots:
        first_ps = state_snapshots[0].public_state.get("players")
        if isinstance(first_ps, dict):
            player_ids = list(first_ps.keys())

    prompt_traces_by_player: dict[str, list[PromptTrace]] = {}
    for player_id in player_ids:
        trace_path = run_dir / "players" / player_id / "prompt_trace.jsonl"
        raw = read_jsonl(trace_path)
        prompt_traces_by_player[player_id] = [PromptTrace.from_dict(entry) for entry in raw]

    return {
        "result": result,
        "metadata": metadata,
        "events": events,
        "state_snapshots": state_snapshots,
        "prompt_traces_by_player": prompt_traces_by_player,
        "player_ids": player_ids,
    }


# ── Board helpers ────────────────────────────────────────────────────────────

def _build_tile_map(state_snapshots: list[PublicStateSnapshot]) -> dict[str, dict[str, Any]]:
    """Build tile_id -> {resource, number, coordinate} from the first snapshot's board.tiles."""
    if not state_snapshots:
        return {}
    board = state_snapshots[0].public_state.get("board")
    if not isinstance(board, dict):
        return {}
    tiles = board.get("tiles")
    if not isinstance(tiles, dict):
        return {}
    tile_map: dict[str, dict[str, Any]] = {}
    for tile_id, tile_data in tiles.items():
        if not isinstance(tile_data, dict):
            continue
        tile_map[tile_id] = {
            "resource": tile_data.get("resource"),
            "number": tile_data.get("number"),
            "coordinate": tile_data.get("coordinate"),
            "type": tile_data.get("type"),
        }
    return tile_map


def _build_node_adjacent_tiles(state_snapshots: list[PublicStateSnapshot]) -> dict[int, list[dict[str, Any]]]:
    """Build node_id -> list of adjacent tile dicts from the first snapshot."""
    if not state_snapshots:
        return {}
    board = state_snapshots[0].public_state.get("board")
    if not isinstance(board, dict):
        return {}
    adjacent_tiles = board.get("adjacent_tiles")
    if not isinstance(adjacent_tiles, dict):
        return {}
    result: dict[int, list[dict[str, Any]]] = {}
    for node_id_str, tiles in adjacent_tiles.items():
        if not isinstance(tiles, list):
            continue
        result[int(node_id_str)] = tiles
    return result


def _get_robber_coordinate(snapshot: PublicStateSnapshot) -> tuple[int, ...] | None:
    """Extract robber coordinate from a state snapshot."""
    board = snapshot.public_state.get("board")
    if not isinstance(board, dict):
        return None
    coord = board.get("robber_coordinate")
    if isinstance(coord, (list, tuple)):
        return tuple(int(c) for c in coord)
    return None


def _get_player_buildings(snapshot: PublicStateSnapshot, player_id: str) -> tuple[set[int], set[int]]:
    """Return (settlement_node_ids, city_node_ids) for a player from a state snapshot."""
    settlements: set[int] = set()
    cities: set[int] = set()
    board = snapshot.public_state.get("board")
    if not isinstance(board, dict):
        return settlements, cities
    nodes = board.get("nodes")
    if not isinstance(nodes, dict):
        return settlements, cities
    for node_id_str, node_data in nodes.items():
        if not isinstance(node_data, dict):
            continue
        if node_data.get("color") != player_id:
            continue
        building = node_data.get("building")
        if building == "SETTLEMENT":
            settlements.add(int(node_id_str))
        elif building == "CITY":
            cities.add(int(node_id_str))
    return settlements, cities


# ── Game-level metrics ───────────────────────────────────────────────────────

def compute_game_summary(
    *,
    result: dict[str, Any],
    events: list[Event],
    num_turns: int,
    total_decisions: int,
) -> dict[str, Any]:
    """Compute game-level summary metrics."""
    total_events = len(events)
    trade_events = sum(1 for e in events if e.kind in TRADE_EVENT_KINDS)
    trade_offered = sum(1 for e in events if e.kind == "trade_offered")
    trade_confirmed = sum(1 for e in events if e.kind == "trade_confirmed")

    return {
        "winner_ids": list(result.get("winner_ids", [])),
        "num_turns": num_turns,
        "total_events": total_events,
        "total_decisions": total_decisions,
        "events_per_turn": round(total_events / max(num_turns, 1), 2),
        "decisions_per_turn": round(total_decisions / max(num_turns, 1), 2),
        "trade_activity_rate": round(trade_events / max(total_events, 1), 4),
        "trade_efficiency": round(trade_confirmed / max(trade_offered, 1), 4) if trade_offered else 0.0,
    }


# ── Building timeline ────────────────────────────────────────────────────────

def compute_building_timeline(player_id: str, events: list[Event]) -> dict[str, Any]:
    """Extract building events for a player, grouped by type."""
    settlements: list[dict[str, Any]] = []
    cities: list[dict[str, Any]] = []
    roads: list[dict[str, Any]] = []

    for event in events:
        if event.actor_player_id != player_id:
            continue
        if event.kind == "settlement_built":
            settlements.append({
                "turn_index": event.turn_index,
                "node_id": event.payload.get("node_id"),
                "history_index": event.history_index,
            })
        elif event.kind == "city_built":
            cities.append({
                "turn_index": event.turn_index,
                "node_id": event.payload.get("node_id"),
                "history_index": event.history_index,
            })
        elif event.kind == "road_built":
            roads.append({
                "turn_index": event.turn_index,
                "edge": event.payload.get("edge"),
                "history_index": event.history_index,
            })

    return {
        "settlements": settlements,
        "cities": cities,
        "roads": roads,
        "counts": {
            "settlements": len(settlements),
            "cities": len(cities),
            "roads": len(roads),
        },
    }


# ── Robber analysis ──────────────────────────────────────────────────────────

def compute_robber_analysis(player_id: str, events: list[Event]) -> dict[str, Any]:
    """Count robber interactions for a player."""
    times_moved_robber = 0
    times_targeted = 0

    for event in events:
        if event.kind != "robber_moved":
            continue
        if event.actor_player_id == player_id:
            times_moved_robber += 1
        if event.payload.get("victim") == player_id:
            times_targeted += 1

    return {
        "times_moved_robber": times_moved_robber,
        "times_targeted": times_targeted,
    }


# ── Development cards ────────────────────────────────────────────────────────

def compute_dev_card_analysis(
    player_id: str,
    events: list[Event],
    final_snapshot: PublicStateSnapshot | None,
) -> dict[str, Any]:
    """Analyze development card usage for a player."""
    cards_played_by_type: dict[str, int] = {}
    for event in events:
        if event.kind != "development_card_played" or event.actor_player_id != player_id:
            continue
        action = event.payload.get("action")
        if isinstance(action, dict):
            action_type = str(action.get("action_type", "UNKNOWN"))
        else:
            action_type = "UNKNOWN"
        cards_played_by_type[action_type] = cards_played_by_type.get(action_type, 0) + 1

    cards_played = sum(cards_played_by_type.values())
    cards_held_at_end = 0
    if final_snapshot is not None:
        players = final_snapshot.public_state.get("players")
        if isinstance(players, dict):
            player_data = players.get(player_id)
            if isinstance(player_data, dict):
                cards_held_at_end = int(player_data.get("development_card_count", 0))

    return {
        "cards_played": cards_played,
        "cards_held_at_end": cards_held_at_end,
        "cards_played_by_type": cards_played_by_type,
    }


# ── Trade analysis ───────────────────────────────────────────────────────────

def _add_resources(target: dict[str, int], source: dict[str, Any]) -> None:
    """Add resource counts from source into target accumulator."""
    for resource in RESOURCES:
        amount = source.get(resource)
        if isinstance(amount, (int, float)) and amount > 0:
            target[resource] = target.get(resource, 0) + int(amount)


def compute_trade_analysis(player_id: str, events: list[Event]) -> dict[str, Any]:
    """Compute per-player trade metrics from event log."""
    offers_made = 0
    offers_received = 0
    acceptances = 0
    rejections = 0
    confirmations_as_offerer = 0
    confirmations_as_acceptee = 0
    resources_given: dict[str, int] = {}
    resources_received: dict[str, int] = {}

    for event in events:
        if event.kind == "trade_offered":
            if event.actor_player_id == player_id:
                offers_made += 1
            else:
                offers_received += 1

        elif event.kind == "trade_accepted":
            if event.payload.get("responding_player_id") == player_id:
                acceptances += 1

        elif event.kind == "trade_rejected":
            if event.payload.get("responding_player_id") == player_id:
                rejections += 1

        elif event.kind == "trade_confirmed":
            offering_id = event.payload.get("offering_player_id")
            accepting_id = event.payload.get("accepting_player_id")
            offer = event.payload.get("offer", {})
            request = event.payload.get("request", {})
            if not isinstance(offer, dict):
                offer = {}
            if not isinstance(request, dict):
                request = {}

            if offering_id == player_id:
                confirmations_as_offerer += 1
                _add_resources(resources_given, offer)
                _add_resources(resources_received, request)
            elif accepting_id == player_id:
                confirmations_as_acceptee += 1
                # Acceptee gives what offerer requested, receives what offerer offered
                _add_resources(resources_given, request)
                _add_resources(resources_received, offer)

    total_responses = acceptances + rejections
    net_balance: dict[str, int] = {}
    for resource in RESOURCES:
        net = resources_received.get(resource, 0) - resources_given.get(resource, 0)
        if net != 0:
            net_balance[resource] = net

    return {
        "offers_made": offers_made,
        "offers_received": offers_received,
        "acceptances": acceptances,
        "rejections": rejections,
        "acceptance_rate": round(acceptances / max(total_responses, 1), 4) if total_responses else 0.0,
        "confirmations_as_offerer": confirmations_as_offerer,
        "confirmations_as_acceptee": confirmations_as_acceptee,
        "resources_given": resources_given,
        "resources_received": resources_received,
        "net_trade_balance": net_balance,
    }


# ── Resource production estimation ───────────────────────────────────────────

def compute_resource_production(
    player_id: str,
    events: list[Event],
    state_snapshots: list[PublicStateSnapshot],
) -> dict[str, Any]:
    """Estimate resources produced from dice rolls based on building positions.

    For each dice_rolled event, looks up which buildings the player has on nodes
    adjacent to tiles matching the rolled number. Settlements produce 1, cities
    produce 2. Tiles with the robber on them produce nothing.
    """
    if not state_snapshots:
        return {"total": {}, "by_turn": []}

    adjacent_tiles = _build_node_adjacent_tiles(state_snapshots)
    total: dict[str, int] = {}
    by_turn: list[dict[str, Any]] = []

    # Build a lookup: history_index -> snapshot (for finding building positions)
    snapshot_by_history: dict[int, PublicStateSnapshot] = {
        s.history_index: s for s in state_snapshots
    }

    # For each dice roll, find the closest prior snapshot to determine building positions
    sorted_indices = sorted(snapshot_by_history.keys())

    def _find_snapshot_at_or_before(history_index: int) -> PublicStateSnapshot | None:
        best = None
        for idx in sorted_indices:
            if idx <= history_index:
                best = snapshot_by_history[idx]
            else:
                break
        return best

    for event in events:
        if event.kind != "dice_rolled":
            continue

        dice = event.payload.get("dice")
        if isinstance(dice, (list, tuple)):
            roll = sum(int(d) for d in dice)
        elif isinstance(dice, (int, float)):
            roll = int(dice)
        else:
            result = event.payload.get("result")
            if isinstance(result, (list, tuple)):
                roll = sum(int(d) for d in result)
            elif isinstance(result, (int, float)):
                roll = int(result)
            else:
                continue

        if roll == 7:
            continue

        snap = _find_snapshot_at_or_before(event.history_index)
        if snap is None:
            continue

        robber_coord = _get_robber_coordinate(snap)
        settlements, cities = _get_player_buildings(snap, player_id)
        turn_production: dict[str, int] = {}

        all_nodes = settlements | cities
        for node_id in all_nodes:
            multiplier = 2 if node_id in cities else 1
            tiles = adjacent_tiles.get(node_id, [])
            for tile in tiles:
                if not isinstance(tile, dict):
                    continue
                if tile.get("type") != "RESOURCE_TILE":
                    continue
                if tile.get("number") != roll:
                    continue
                # Check robber
                tile_coord = tile.get("coordinate")
                if robber_coord is not None and isinstance(tile_coord, (list, tuple)):
                    if tuple(int(c) for c in tile_coord) == robber_coord:
                        continue
                resource = tile.get("resource")
                if isinstance(resource, str) and resource in RESOURCES:
                    turn_production[resource] = turn_production.get(resource, 0) + multiplier
                    total[resource] = total.get(resource, 0) + multiplier

        if turn_production:
            by_turn.append({"turn_index": event.turn_index, "resources": turn_production})

    return {"total": total, "by_turn": by_turn}


# ── VP progression ───────────────────────────────────────────────────────────

def compute_vp_progression(
    player_id: str,
    state_snapshots: list[PublicStateSnapshot],
) -> list[dict[str, Any]]:
    """Extract VP at each turn boundary, deduplicated to max VP per turn."""
    turn_max_vp: dict[int, int] = {}
    for snap in state_snapshots:
        players = snap.public_state.get("players")
        if not isinstance(players, dict):
            continue
        player_data = players.get(player_id)
        if not isinstance(player_data, dict):
            continue
        vp = int(player_data.get("visible_victory_points", 0))
        turn = snap.turn_index
        if turn not in turn_max_vp or vp > turn_max_vp[turn]:
            turn_max_vp[turn] = vp

    return [{"turn_index": t, "vp": v} for t, v in sorted(turn_max_vp.items())]


# ── Phase analysis ───────────────────────────────────────────────────────────

def compute_phase_analysis(
    player_id: str,
    events: list[Event],
    state_snapshots: list[PublicStateSnapshot],
    vp_progression: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute opening placement analysis and VP milestones."""
    adjacent_tiles = _build_node_adjacent_tiles(state_snapshots)

    # Opening: find initial settlement placements
    initial_nodes: list[int] = []
    for event in events:
        if (
            event.kind == "settlement_built"
            and event.actor_player_id == player_id
            and event.phase in ("build_initial_settlement",)
        ):
            node_id = event.payload.get("node_id")
            if isinstance(node_id, (int, float)):
                initial_nodes.append(int(node_id))

    # Compute resource diversity and pip count for initial placements
    resource_types: set[str] = set()
    tile_numbers: list[int] = []
    pip_count = 0
    for node_id in initial_nodes:
        tiles = adjacent_tiles.get(node_id, [])
        for tile in tiles:
            if not isinstance(tile, dict):
                continue
            if tile.get("type") != "RESOURCE_TILE":
                continue
            resource = tile.get("resource")
            if isinstance(resource, str):
                resource_types.add(resource)
            number = tile.get("number")
            if isinstance(number, (int, float)):
                num = int(number)
                tile_numbers.append(num)
                pip_count += PIPS.get(num, 0)

    # VP milestones
    thresholds = [3, 5, 7, 10]
    milestones: dict[str, int | None] = {}
    for threshold in thresholds:
        key = f"{threshold}vp_turn"
        milestones[key] = None
        for entry in vp_progression:
            if entry["vp"] >= threshold:
                milestones[key] = entry["turn_index"]
                break

    return {
        "opening": {
            "initial_settlement_nodes": initial_nodes,
            "resource_diversity": len(resource_types),
            "resource_types": sorted(resource_types),
            "tile_numbers": sorted(tile_numbers),
            "pip_count": pip_count,
        },
        "vp_milestones": milestones,
    }


# ── Decision quality ─────────────────────────────────────────────────────────

def compute_decision_quality(prompt_traces: list[PromptTrace]) -> dict[str, Any]:
    """Analyze prompt retry counts as a decision quality proxy."""
    total_prompts = len(prompt_traces)
    retries = sum(1 for pt in prompt_traces if len(pt.attempts) > 1)
    max_attempts = max((len(pt.attempts) for pt in prompt_traces), default=0)

    return {
        "total_prompts": total_prompts,
        "retries": retries,
        "retry_rate": round(retries / max(total_prompts, 1), 4) if total_prompts else 0.0,
        "max_attempts_on_single_decision": max_attempts,
    }


# ── Achievements (from final state) ─────────────────────────────────────────

def compute_achievements(
    player_id: str,
    final_snapshot: PublicStateSnapshot | None,
) -> dict[str, Any]:
    """Extract longest road / largest army from the final state snapshot."""
    if final_snapshot is None:
        return {
            "longest_road_length": 0,
            "has_longest_road": False,
            "has_largest_army": False,
        }
    players = final_snapshot.public_state.get("players")
    if not isinstance(players, dict):
        return {
            "longest_road_length": 0,
            "has_longest_road": False,
            "has_largest_army": False,
        }
    player_data = players.get(player_id)
    if not isinstance(player_data, dict):
        return {
            "longest_road_length": 0,
            "has_longest_road": False,
            "has_largest_army": False,
        }
    return {
        "longest_road_length": int(player_data.get("longest_road_length", 0)),
        "has_longest_road": bool(player_data.get("has_longest_road", False)),
        "has_largest_army": bool(player_data.get("has_largest_army", False)),
    }


# ── Per-player orchestration ─────────────────────────────────────────────────

def analyze_player(
    player_id: str,
    *,
    events: list[Event],
    state_snapshots: list[PublicStateSnapshot],
    prompt_traces: list[PromptTrace],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Compute all metrics for a single player."""
    final_snapshot = state_snapshots[-1] if state_snapshots else None
    winner_ids = [str(w) for w in result.get("winner_ids", [])]

    # Get final VP from result metadata
    players_meta = result.get("metadata", {}).get("players", {})
    player_meta = players_meta.get(player_id, {}) if isinstance(players_meta, dict) else {}
    final_vp = int(player_meta.get("actual_victory_points", 0)) if isinstance(player_meta, dict) else 0

    vp_progression = compute_vp_progression(player_id, state_snapshots)

    return {
        "final_vp": final_vp,
        "is_winner": player_id in winner_ids,
        "vp_progression": vp_progression,
        "resource_production": compute_resource_production(player_id, events, state_snapshots),
        "buildings": compute_building_timeline(player_id, events),
        "trade": compute_trade_analysis(player_id, events),
        "robber": compute_robber_analysis(player_id, events),
        "dev_cards": compute_dev_card_analysis(player_id, events, final_snapshot),
        "decision_quality": compute_decision_quality(prompt_traces),
        "achievements": compute_achievements(player_id, final_snapshot),
        "phase_analysis": compute_phase_analysis(player_id, events, state_snapshots, vp_progression),
    }


# ── Top-level orchestration ──────────────────────────────────────────────────

def analyze_game(run_dir: str | Path, *, write: bool = True) -> dict[str, Any]:
    """Main entry point: load artifacts, compute all metrics, optionally write analysis.json."""
    run_dir = Path(run_dir)
    artifacts = load_analysis_artifacts(run_dir)

    result = artifacts["result"]
    events = artifacts["events"]
    state_snapshots = artifacts["state_snapshots"]
    prompt_traces_by_player = artifacts["prompt_traces_by_player"]
    player_ids = artifacts["player_ids"]

    metadata = result.get("metadata", {})
    num_turns = int(metadata.get("num_turns", 0))
    total_decisions = int(result.get("total_decisions", 0))

    game_summary = compute_game_summary(
        result=result,
        events=events,
        num_turns=num_turns,
        total_decisions=total_decisions,
    )

    players: dict[str, Any] = {}
    for player_id in player_ids:
        players[player_id] = analyze_player(
            player_id,
            events=events,
            state_snapshots=state_snapshots,
            prompt_traces=prompt_traces_by_player.get(player_id, []),
            result=result,
        )

    analysis = {
        "game_id": str(result.get("game_id", "")),
        "version": "1",
        "game_summary": game_summary,
        "players": players,
    }

    if write:
        write_json(run_dir / "analysis.json", analysis)

    return analysis


# ── Terminal summary ─────────────────────────────────────────────────────────

def print_terminal_summary(analysis: dict[str, Any], *, file: Any = None) -> None:
    """Print a colored terminal summary of the analysis."""
    out = file or sys.stderr
    use_color = hasattr(out, "isatty") and out.isatty()

    def bold(text: str) -> str:
        return _c(text, _BOLD) if use_color else text

    def dim(text: str) -> str:
        return _c(text, _DIM) if use_color else text

    def player_color(player_id: str, text: str) -> str:
        if not use_color:
            return text
        code = _PLAYER_ANSI.get(player_id, "")
        return _c(text, code) if code else text

    gs = analysis.get("game_summary", {})
    out.write(f"\n{bold('=== Post-Game Analysis ===')}\n\n")

    winner_ids = gs.get("winner_ids", [])
    winner_str = ", ".join(player_color(w, w) for w in winner_ids) if winner_ids else "None"
    out.write(f"  Winner: {winner_str}\n")
    out.write(f"  Turns: {gs.get('num_turns', '?')}  |  Events: {gs.get('total_events', '?')}  |  Decisions: {gs.get('total_decisions', '?')}\n")
    out.write(f"  Events/turn: {gs.get('events_per_turn', '?')}  |  Decisions/turn: {gs.get('decisions_per_turn', '?')}\n")
    out.write(f"  Trade activity: {gs.get('trade_activity_rate', 0):.1%}  |  Trade efficiency: {gs.get('trade_efficiency', 0):.1%}\n")

    players = analysis.get("players", {})
    for player_id, data in players.items():
        out.write(f"\n  {bold(player_color(player_id, player_id))}")
        out.write(f"  {'(WINNER)' if data.get('is_winner') else ''}\n")

        out.write(f"    VP: {data.get('final_vp', '?')}")
        achievements = data.get("achievements", {})
        badges = []
        if achievements.get("has_longest_road"):
            badges.append("Longest Road")
        if achievements.get("has_largest_army"):
            badges.append("Largest Army")
        if badges:
            out.write(f"  [{', '.join(badges)}]")
        out.write("\n")

        # Buildings
        buildings = data.get("buildings", {})
        counts = buildings.get("counts", {})
        out.write(f"    Buildings: {counts.get('settlements', 0)}S  {counts.get('cities', 0)}C  {counts.get('roads', 0)}R\n")

        # Resource production
        production = data.get("resource_production", {}).get("total", {})
        if production:
            res_parts = [f"{r}: {production.get(r, 0)}" for r in RESOURCES if production.get(r, 0)]
            out.write(f"    Production: {', '.join(res_parts)}\n")

        # Trade
        trade = data.get("trade", {})
        if trade.get("offers_made", 0) or trade.get("confirmations_as_offerer", 0) or trade.get("confirmations_as_acceptee", 0):
            out.write(f"    Trades: {trade.get('offers_made', 0)} offered, "
                      f"{trade.get('confirmations_as_offerer', 0) + trade.get('confirmations_as_acceptee', 0)} completed "
                      f"({trade.get('acceptance_rate', 0):.0%} accept rate)\n")

        # Robber
        robber = data.get("robber", {})
        if robber.get("times_moved_robber", 0) or robber.get("times_targeted", 0):
            out.write(f"    Robber: moved {robber.get('times_moved_robber', 0)}x, targeted {robber.get('times_targeted', 0)}x\n")

        # Dev cards
        dev = data.get("dev_cards", {})
        if dev.get("cards_played", 0) or dev.get("cards_held_at_end", 0):
            out.write(f"    Dev cards: {dev.get('cards_played', 0)} played, {dev.get('cards_held_at_end', 0)} held\n")

        # Decision quality
        dq = data.get("decision_quality", {})
        if dq.get("retries", 0):
            out.write(f"    Retries: {dq['retries']}/{dq.get('total_prompts', 0)} ({dq.get('retry_rate', 0):.1%})\n")

        # Opening
        phase = data.get("phase_analysis", {}).get("opening", {})
        if phase.get("pip_count", 0):
            out.write(f"    Opening: {phase.get('resource_diversity', 0)} resource types, {phase.get('pip_count', 0)} pips\n")

    out.write(f"\n{dim('─' * 60)}\n")
    out.flush()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a completed catan-bench game run.",
    )
    parser.add_argument("run_dir", help="Path to the completed run directory.")
    parser.add_argument("--json-only", action="store_true", help="Print JSON to stdout, skip terminal summary.")
    parser.add_argument("--no-write", action="store_true", help="Do not write analysis.json to the run directory.")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory.", file=sys.stderr)
        return 1

    analysis = analyze_game(run_dir, write=not args.no_write)

    if args.json_only:
        print(json.dumps(analysis, indent=2, sort_keys=True))
    else:
        print_terminal_summary(analysis)
        if not args.no_write:
            print(f"Written to: {run_dir / 'analysis.json'}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
