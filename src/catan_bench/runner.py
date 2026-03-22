from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import GameConfig, PlayerConfig, load_game_config, load_player_configs
from .orchestrator import GameOrchestrator
from .players import FirstLegalPlayer, RandomLegalPlayer

try:
    from .catanatron_adapter import CatanatronEngineAdapter
except RuntimeError as exc:  # pragma: no cover - dependency missing in some environments.
    CatanatronEngineAdapter = None
    _catanatron_import_error = exc
else:
    _catanatron_import_error = None


def build_engine(game_config: GameConfig, players: Sequence[PlayerConfig]):
    if game_config.engine != "catanatron":
        raise ValueError(f"Unsupported engine {game_config.engine!r}.")
    if CatanatronEngineAdapter is None:
        raise RuntimeError(str(_catanatron_import_error))

    return CatanatronEngineAdapter(
        player_ids=[player.id for player in players],
        seed=game_config.seed,
        discard_limit=game_config.discard_limit,
        vps_to_win=game_config.vps_to_win,
    )


def build_players(players: Sequence[PlayerConfig]):
    built_players = {}
    for player_config in players:
        if player_config.type == "random":
            built_players[player_config.id] = RandomLegalPlayer(
                seed=player_config.seed,
                allow_trade_offers=player_config.allow_trade_offers,
            )
        elif player_config.type == "first_legal":
            built_players[player_config.id] = FirstLegalPlayer(
                allow_trade_offers=player_config.allow_trade_offers,
            )
        else:  # pragma: no cover - validated in config loading.
            raise ValueError(f"Unsupported player type {player_config.type!r}.")

    return built_players


def run_from_config_files(
    *, game_config_path: str | Path, players_config_path: str | Path
):
    game_config = load_game_config(game_config_path)
    player_configs = load_player_configs(players_config_path)
    engine = build_engine(game_config, player_configs)
    players = build_players(player_configs)
    orchestrator = GameOrchestrator(
        engine,
        players,
        run_dir=game_config.run_dir,
    )
    return orchestrator.run()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a benchmark game from TOML game and player configs."
    )
    parser.add_argument("--game", required=True, help="Path to game TOML config.")
    parser.add_argument("--players", required=True, help="Path to player TOML config.")
    args = parser.parse_args(argv)

    result = run_from_config_files(
        game_config_path=args.game,
        players_config_path=args.players,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
