from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency in lean test envs.
    def load_dotenv(path: str | Path, override: bool = False):  # type: ignore[no-redef]
        loaded = False
        env_path = Path(path)
        if not env_path.is_file():
            return loaded
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value.strip()
                loaded = True
        return loaded

from .config import GameConfig, PlayerConfig, load_game_config, load_player_configs
from .llm import OpenAICompatibleChatClient
from .observations import ObservationBuilder
from .orchestrator import GameOrchestrator
from .players import FirstLegalPlayer, LLMPlayer, RandomLegalPlayer
from .reporter import TerminalReporter

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


def build_players(players: Sequence[PlayerConfig], game_config: GameConfig | None = None):
    built_players = {}
    for player_config in players:
        if player_config.type == "random":
            built_players[player_config.id] = RandomLegalPlayer(
                seed=player_config.seed,
            )
        elif player_config.type == "first_legal":
            built_players[player_config.id] = FirstLegalPlayer()
        elif player_config.type == "llm":
            built_players[player_config.id] = LLMPlayer(
                client=OpenAICompatibleChatClient(
                    api_base=player_config.api_base,
                    api_key_env=player_config.api_key_env,
                    timeout_seconds=player_config.timeout_seconds,
                ),
                model=player_config.model or "",
                temperature=player_config.temperature,
                top_p=player_config.top_p,
                reasoning_enabled=player_config.reasoning_enabled,
                prompt_history_limit=player_config.prompt_history_limit,
                prompt_memory_limit=player_config.prompt_memory_limit,
            )
        else:  # pragma: no cover - validated in config loading.
            raise ValueError(f"Unsupported player type {player_config.type!r}.")

    return built_players


def run_from_config_files(
    *, game_config_path: str | Path, players_config_path: str | Path
):
    _load_local_env(Path(players_config_path).resolve().parent)
    game_config = load_game_config(game_config_path)
    player_configs = load_player_configs(players_config_path)
    engine = build_engine(game_config, player_configs)
    players = build_players(player_configs, game_config)
    orchestrator = GameOrchestrator(
        engine,
        players,
        observation_builder=ObservationBuilder(
            recent_event_window=game_config.history_window,
        ),
        run_dir=game_config.run_dir,
        trading_chat_enabled=game_config.trading_chat_enabled,
        trading_chat_max_failed_attempts_per_turn=(
            game_config.trading_chat_max_failed_attempts_per_turn
        ),
        trading_chat_message_chars=game_config.trading_chat_message_chars,
        trading_chat_history_limit=game_config.trading_chat_history_limit,
        reporter=TerminalReporter(),
    )
    return orchestrator.run()


def _load_local_env(start_dir: Path) -> None:
    env_path = _find_dotenv(start_dir)
    if env_path is None:
        return
    load_dotenv(env_path, override=False)


def _find_dotenv(start_dir: Path, *, max_depth: int = 4) -> Path | None:
    current = start_dir
    for _ in range(max_depth + 1):
        candidate = current / ".env"
        if candidate.is_file():
            return candidate
        if current.parent == current:
            return None
        current = current.parent
    return None


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
