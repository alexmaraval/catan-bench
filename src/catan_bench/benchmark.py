"""Cross-game benchmark evaluation: ELO ratings and rubric scoring."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .run_dirs import iter_run_directory_candidates

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


@dataclass
class GameRecord:
    """A completed game's key data for benchmark evaluation."""

    game_id: str
    run_dir: Path
    num_turns: int
    player_models: dict[str, str]  # color -> model name
    winner_ids: list[str]
    player_vps: dict[str, int]  # color -> actual VP
    analysis: dict[str, Any] | None  # from analysis.json


def identify_model(run_dir: Path, player_id: str) -> str | None:
    """Extract the LLM model name from the first prompt trace entry."""
    trace_path = run_dir / "players" / player_id / "prompt_trace.jsonl"
    if not trace_path.exists():
        return None
    try:
        with trace_path.open() as fh:
            first_line = fh.readline().strip()
            if not first_line:
                return None
            entry = json.loads(first_line)
            return entry.get("model") or None
    except (json.JSONDecodeError, OSError):
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _is_run_directory(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(
        (path / name).exists()
        for name in (
            "metadata.json",
            "public_history.jsonl",
            "public_state_trace.jsonl",
        )
    )


def _discover_run_directories(base_run_dir: Path) -> tuple[Path, ...]:
    base = Path(base_run_dir)
    if _is_run_directory(base):
        return (base,)
    if not base.exists() or not base.is_dir():
        return ()
    # ELO is order-sensitive across games, so keep a deterministic path-based order
    # rather than filesystem mtimes that can change when artifacts are copied/touched.
    return tuple(p for p in iter_run_directory_candidates(base) if _is_run_directory(p))


def collect_game_records(base_run_dir: Path) -> list[GameRecord]:
    """Scan all completed games under *base_run_dir* and return GameRecords."""
    runs = _discover_run_directories(base_run_dir)
    records: list[GameRecord] = []

    for run_dir in runs:
        result = _read_json(run_dir / "result.json")
        if result is None:
            continue  # game not finished

        metadata_file = _read_json(run_dir / "metadata.json")
        result_meta = result.get("metadata", {})
        players_meta = result_meta.get("players", {})
        winner_ids = result.get("winner_ids") or result_meta.get("winner_ids") or []
        num_turns = int(result_meta.get("num_turns", 0))

        # Determine player IDs
        player_ids: list[str] = []
        if metadata_file:
            player_ids = metadata_file.get("player_ids", [])
        if not player_ids:
            player_ids = list(players_meta.keys())

        # Identify model per player
        adapter_types = (metadata_file or {}).get("player_adapter_types", {})
        player_models: dict[str, str] = {}
        for pid in player_ids:
            model = identify_model(run_dir, pid)
            if model:
                player_models[pid] = model
            elif adapter_types.get(pid):
                player_models[pid] = adapter_types[pid]
            # else: skip players with no identifiable model

        if not player_models:
            continue  # no identifiable players

        player_vps: dict[str, int] = {}
        for pid, pdata in players_meta.items():
            player_vps[pid] = int(pdata.get("actual_victory_points", 0))

        analysis = _read_json(run_dir / "analysis.json")

        records.append(
            GameRecord(
                game_id=result.get("game_id", run_dir.name),
                run_dir=run_dir,
                num_turns=num_turns,
                player_models=player_models,
                winner_ids=winner_ids,
                player_vps=player_vps,
                analysis=analysis,
            )
        )

    return records


# ---------------------------------------------------------------------------
# ELO computation
# ---------------------------------------------------------------------------

DEFAULT_ELO = 1500.0
K_FACTOR = 32.0


@dataclass
class EloState:
    """Aggregated ELO state across all games."""

    ratings: dict[str, float] = field(default_factory=dict)
    games_played: dict[str, int] = field(default_factory=dict)
    wins: dict[str, int] = field(default_factory=dict)
    total_vp: dict[str, int] = field(default_factory=dict)
    # history: list of (game_index, {model: rating}) for charting
    history: list[dict[str, float]] = field(default_factory=list)


def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _generate_pairwise_outcomes(
    record: GameRecord,
) -> list[tuple[str, str, float]]:
    """Return (model_a, model_b, score_a) tuples. Skips self-play pairs."""
    # Group colors by model
    model_colors: dict[str, list[str]] = {}
    for color, model in record.player_models.items():
        model_colors.setdefault(model, []).append(color)

    unique_models = list(model_colors.keys())
    outcomes: list[tuple[str, str, float]] = []

    for i, model_a in enumerate(unique_models):
        for model_b in unique_models[i + 1 :]:
            a_won = any(c in record.winner_ids for c in model_colors[model_a])
            b_won = any(c in record.winner_ids for c in model_colors[model_b])
            a_best_vp = max(record.player_vps.get(c, 0) for c in model_colors[model_a])
            b_best_vp = max(record.player_vps.get(c, 0) for c in model_colors[model_b])

            if a_won and not b_won:
                score = 1.0
            elif b_won and not a_won:
                score = 0.0
            elif a_won and b_won:
                score = 0.5
            else:
                if a_best_vp > b_best_vp:
                    score = 1.0
                elif b_best_vp > a_best_vp:
                    score = 0.0
                else:
                    score = 0.5

            outcomes.append((model_a, model_b, score))

    return outcomes


def compute_elo_ratings(
    games: list[GameRecord],
    *,
    k_factor: float = K_FACTOR,
    initial_rating: float = DEFAULT_ELO,
) -> EloState:
    """Compute ELO ratings from all game records."""
    state = EloState()

    for record in games:
        # Ensure all models are initialized
        for model in record.player_models.values():
            if model not in state.ratings:
                state.ratings[model] = initial_rating
                state.games_played[model] = 0
                state.wins[model] = 0
                state.total_vp[model] = 0

        # Track participation and stats
        seen_models: set[str] = set()
        for color, model in record.player_models.items():
            if model not in seen_models:
                state.games_played[model] = state.games_played.get(model, 0) + 1
                seen_models.add(model)
            vp = record.player_vps.get(color, 0)
            state.total_vp[model] = state.total_vp.get(model, 0) + vp
            if color in record.winner_ids:
                state.wins[model] = state.wins.get(model, 0) + 1

        # Pairwise ELO updates
        outcomes = _generate_pairwise_outcomes(record)
        rating_deltas: defaultdict[str, float] = defaultdict(float)
        for model_a, model_b, score_a in outcomes:
            # Apply all pairwise comparisons against the pre-game snapshot so a
            # single game's rating change does not depend on seat ordering.
            ra = state.ratings[model_a]
            rb = state.ratings[model_b]
            ea = _elo_expected(ra, rb)
            eb = 1.0 - ea
            rating_deltas[model_a] += k_factor * (score_a - ea)
            rating_deltas[model_b] += k_factor * ((1.0 - score_a) - eb)

        for model, delta in rating_deltas.items():
            state.ratings[model] += delta

        # Snapshot for history chart
        state.history.append(dict(state.ratings))

    return state


# ---------------------------------------------------------------------------
# Rubric scoring
# ---------------------------------------------------------------------------


def _normalize(
    value: float,
    *,
    min_val: float = 0.0,
    max_val: float = 1.0,
    invert: bool = False,
) -> float:
    """Normalize *value* to 0–100 range."""
    if max_val <= min_val:
        return 50.0
    clamped = max(min_val, min(max_val, value))
    score = (clamped - min_val) / (max_val - min_val) * 100.0
    return 100.0 - score if invert else score


@dataclass
class RubricScores:
    trading: float = 0.0
    strategy: float = 0.0
    manipulation: float = 0.0
    resource_management: float = 0.0
    game_mechanics: float = 0.0

    @property
    def overall(self) -> float:
        return (
            self.trading
            + self.strategy
            + self.manipulation
            + self.resource_management
            + self.game_mechanics
        ) / 5.0

    def as_dict(self) -> dict[str, float]:
        return {
            "Trading": self.trading,
            "Strategy": self.strategy,
            "Manipulation": self.manipulation,
            "Resource Mgmt": self.resource_management,
            "Game Mechanics": self.game_mechanics,
        }


def _score_trading(player_data: dict, num_turns: int) -> float:
    tc = player_data.get("trade_chat", {})
    tr = player_data.get("trade", {})

    negotiation_success = tc.get("negotiation_success_rate", 0.0)
    proposal_accept = tc.get("proposal_acceptance_rate", 0.0)
    trade_accept = tr.get("acceptance_rate", 0.0)

    counterparty_count = len(tc.get("counterparty_frequency", {}))
    counterparty_div = min(counterparty_count / 3.0, 1.0)

    confirmations = tr.get("confirmations_as_offerer", 0) + tr.get(
        "confirmations_as_acceptee", 0
    )
    volume_per_turn = confirmations / max(num_turns, 1) if num_turns else 0.0

    return (
        0.30 * _normalize(negotiation_success, max_val=1.0)
        + 0.25 * _normalize(proposal_accept, max_val=1.0)
        + 0.20 * _normalize(trade_accept, max_val=1.0)
        + 0.15 * _normalize(counterparty_div, max_val=1.0)
        + 0.10 * _normalize(volume_per_turn, max_val=0.5)
    )


def _score_strategy(player_data: dict, num_turns: int) -> float:
    phase = player_data.get("phase_analysis", {}).get("opening", {})
    strategy = player_data.get("strategy", {})
    final_vp = player_data.get("final_vp", 0)

    vp_efficiency = final_vp / max(num_turns, 1)
    pip_count = phase.get("pip_count", 0)
    resource_diversity = phase.get("resource_diversity", 0)
    update_count = strategy.get("strategy_update_count", 0)

    # VP milestone speed: how quickly model reaches halfway to win
    milestones = player_data.get("phase_analysis", {}).get("vp_milestones", {})
    halfway_turn = milestones.get("5vp_turn")
    if halfway_turn is not None and num_turns > 0:
        milestone_speed = 1.0 - (halfway_turn / num_turns)
    else:
        milestone_speed = 0.0

    return (
        0.25 * _normalize(vp_efficiency, max_val=0.2)
        + 0.20 * _normalize(pip_count, max_val=20.0)
        + 0.20 * _normalize(resource_diversity, max_val=5.0)
        + 0.20 * _normalize(update_count, max_val=10.0)
        + 0.15 * _normalize(milestone_speed, max_val=1.0)
    )


def _score_manipulation(player_data: dict, num_turns: int) -> float:
    robber = player_data.get("robber", {})
    tc = player_data.get("trade_chat", {})
    tr = player_data.get("trade", {})

    robber_moved = robber.get("times_moved_robber", 0)
    times_targeted = robber.get("times_targeted", 0)
    rooms_opened = tc.get("rooms_opened", 0)
    avg_rounds = tc.get("avg_rounds_per_room", 3.0)

    # Net trade gain
    net_balance = tr.get("net_trade_balance", {})
    net_gain = sum(max(0, v) for v in net_balance.values()) if net_balance else 0

    target_ratio = robber_moved / max(times_targeted, 1)

    return (
        0.25 * _normalize(robber_moved, max_val=8.0)
        + 0.25 * _normalize(rooms_opened, max_val=20.0)
        + 0.20 * _normalize(target_ratio, max_val=3.0)
        + 0.15 * _normalize(net_gain, max_val=10.0)
        + 0.15 * _normalize(avg_rounds, max_val=5.0, invert=True)
    )


def _score_resource_management(player_data: dict, num_turns: int) -> float:
    rp = player_data.get("resource_production", {}).get("total", {})
    buildings = player_data.get("buildings", {}).get("counts", {})
    discard = player_data.get("discard", {})

    total_production = sum(rp.values()) if rp else 0
    production_per_turn = total_production / max(num_turns, 1)
    resource_types_produced = sum(1 for v in rp.values() if v > 0) if rp else 0

    total_buildings = sum(buildings.values()) if buildings else 0
    buildings_per_turn = total_buildings / max(num_turns, 1)

    total_discarded = discard.get("total_cards_discarded", 0)

    return (
        0.30 * _normalize(production_per_turn, max_val=3.0)
        + 0.25 * _normalize(buildings_per_turn, max_val=0.3)
        + 0.25 * _normalize(total_discarded, max_val=15.0, invert=True)
        + 0.20 * _normalize(resource_types_produced, max_val=5.0)
    )


def _score_game_mechanics(player_data: dict) -> float:
    dev = player_data.get("dev_cards", {})
    achievements = player_data.get("achievements", {})
    dq = player_data.get("decision_quality", {})

    cards_played = dev.get("cards_played", 0)
    has_longest = 100.0 if achievements.get("has_longest_road") else 0.0
    has_largest = 100.0 if achievements.get("has_largest_army") else 0.0
    retry_rate = dq.get("retry_rate", 0.0)

    return (
        0.25 * _normalize(cards_played, max_val=5.0)
        + 0.25 * has_longest
        + 0.25 * has_largest
        + 0.25 * _normalize(retry_rate, max_val=0.5, invert=True)
    )


def _score_player(player_data: dict, num_turns: int) -> RubricScores:
    return RubricScores(
        trading=_score_trading(player_data, num_turns),
        strategy=_score_strategy(player_data, num_turns),
        manipulation=_score_manipulation(player_data, num_turns),
        resource_management=_score_resource_management(player_data, num_turns),
        game_mechanics=_score_game_mechanics(player_data),
    )


def compute_rubric_scores(
    games: list[GameRecord],
) -> dict[str, RubricScores]:
    """Compute rubric scores per model, averaged across all games."""
    # Accumulate per-model scores
    model_scores: dict[str, list[RubricScores]] = {}

    for record in games:
        if record.analysis is None:
            continue
        players_analysis = record.analysis.get("players", {})

        for color, model in record.player_models.items():
            pdata = players_analysis.get(color)
            if pdata is None:
                continue
            scores = _score_player(pdata, record.num_turns)
            model_scores.setdefault(model, []).append(scores)

    # Average across games
    result: dict[str, RubricScores] = {}
    for model, scores_list in model_scores.items():
        n = len(scores_list)
        if n == 0:
            result[model] = RubricScores()
            continue
        result[model] = RubricScores(
            trading=sum(s.trading for s in scores_list) / n,
            strategy=sum(s.strategy for s in scores_list) / n,
            manipulation=sum(s.manipulation for s in scores_list) / n,
            resource_management=sum(s.resource_management for s in scores_list) / n,
            game_mechanics=sum(s.game_mechanics for s in scores_list) / n,
        )

    return result


# ---------------------------------------------------------------------------
# Head-to-head win rates
# ---------------------------------------------------------------------------


def compute_head_to_head(
    games: list[GameRecord],
) -> dict[str, dict[str, dict[str, int]]]:
    """Compute pairwise win/loss/draw counts between models.

    Returns ``{model_a: {model_b: {"wins": W, "losses": L, "draws": D}}}``.
    """
    def _record() -> dict[str, int]:
        return {"wins": 0, "losses": 0, "draws": 0}

    h2h: defaultdict[str, defaultdict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(_record)
    )

    for record in games:
        for model_a, model_b, score_a in _generate_pairwise_outcomes(record):
            if score_a == 1.0:
                h2h[model_a][model_b]["wins"] += 1
                h2h[model_b][model_a]["losses"] += 1
            elif score_a == 0.0:
                h2h[model_a][model_b]["losses"] += 1
                h2h[model_b][model_a]["wins"] += 1
            else:
                h2h[model_a][model_b]["draws"] += 1
                h2h[model_b][model_a]["draws"] += 1

    return {k: dict(v) for k, v in h2h.items()}


def _benchmark_summary_payload(
    games: list[GameRecord], elo: EloState, rubrics: dict[str, RubricScores]
) -> dict[str, Any]:
    ranked_models = sorted(
        elo.ratings.keys(), key=lambda model: elo.ratings[model], reverse=True
    )
    leaderboard = []
    for model in ranked_models:
        games_played = elo.games_played.get(model, 0)
        wins = elo.wins.get(model, 0)
        total_vp = elo.total_vp.get(model, 0)
        leaderboard.append(
            {
                "model": model,
                "elo": round(elo.ratings[model], 2),
                "games_played": games_played,
                "wins": wins,
                "avg_vp": 0.0 if games_played == 0 else total_vp / games_played,
                "rubric_overall": rubrics.get(model, RubricScores()).overall,
            }
        )
    return {
        "games_scanned": len(games),
        "runs": [str(game.run_dir) for game in games],
        "leaderboard": leaderboard,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute ELO rankings and rubric scores from completed catan-bench runs."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="runs",
        help=(
            "Path to a completed run directory or a base directory containing completed child runs. "
            "Defaults to ./runs."
        ),
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print the benchmark summary as JSON.",
    )
    args = parser.parse_args(argv)

    target = Path(args.run_dir)
    if not target.is_dir():
        print(f"Error: {target} is not a directory.", file=sys.stderr)
        return 1

    games = collect_game_records(target)
    if not games:
        print(f"No completed runs found under {target}.", file=sys.stderr)
        return 1

    elo = compute_elo_ratings(games)
    rubrics = compute_rubric_scores(games)
    summary = _benchmark_summary_payload(games, elo, rubrics)

    if args.json_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    print(f"Scanned {summary['games_scanned']} completed run(s) under {target}.")
    for index, row in enumerate(summary["leaderboard"], start=1):
        print(
            f"{index}. {row['model']}  ELO {round(row['elo'])}  "
            f"games {row['games_played']}  wins {row['wins']}  "
            f"avg VP {row['avg_vp']:.2f}  rubric {row['rubric_overall']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
