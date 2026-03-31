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


from .config import (
    GameConfig,
    PlayerConfig,
    load_game_config,
    load_game_config_overrides,
    load_player_configs,
)
from .llm import OpenAICompatibleChatClient
from .observations import ObservationBuilder
from .orchestrator import GameOrchestrator
from .players import FirstLegalPlayer, LLMPlayer, RandomLegalPlayer
from .prompts import build_game_rules
from .reporter import DebugTerminalReporter, TerminalReporter
from .storage import read_json

_RESUME_ARTIFACT_FILES = (
    "metadata.json",
    "checkpoint.json",
    "public_history.jsonl",
    "public_state_trace.jsonl",
    "action_trace.jsonl",
)
_AUTO_RESUME_REQUEST = "__auto_resume__"

try:
    from .catanatron_adapter import CatanatronEngineAdapter
except (
    RuntimeError
) as exc:  # pragma: no cover - dependency missing in some environments.
    CatanatronEngineAdapter = None
    _catanatron_import_error = exc
else:
    _catanatron_import_error = None


def build_engine(
    game_config: GameConfig,
    players: Sequence[PlayerConfig],
    *,
    game_id: str | None = None,
):
    if game_config.engine != "catanatron":
        raise ValueError(f"Unsupported engine {game_config.engine!r}.")
    if CatanatronEngineAdapter is None:
        raise RuntimeError(str(_catanatron_import_error))

    return CatanatronEngineAdapter(
        player_ids=[player.id for player in players],
        game_id=game_id,
        seed=game_config.seed,
        discard_limit=game_config.discard_limit,
        vps_to_win=game_config.vps_to_win,
    )


def build_players(
    players: Sequence[PlayerConfig], game_config: GameConfig | None = None
):
    built_players = {}
    effective_game_config = game_config or GameConfig()
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
                reasoning_effort=player_config.reasoning_effort,
                prompt_history_limit=effective_game_config.prompt_history_limit,
            )
        else:  # pragma: no cover - validated in config loading.
            raise ValueError(f"Unsupported player type {player_config.type!r}.")

    return built_players


def run_from_config_files(
    *,
    game_config_path: str | Path,
    players_config_path: str | Path,
    run_dir: str | Path | None = None,
    debug: bool = False,
    debug_from_setup: bool = False,
    debug_trade: bool = False,
):
    players_config_path = Path(players_config_path)
    _load_local_env(players_config_path.resolve().parent)
    game_config = load_game_config(game_config_path)
    game_config = load_game_config_overrides(players_config_path, game_config)
    player_configs = load_player_configs(players_config_path)
    players_config_toml = players_config_path.read_text(encoding="utf-8")
    effective_run_dir, resume_run_dir = _resolve_requested_run_dir(
        requested_run_dir=run_dir,
        configured_run_dir=game_config.run_dir,
        players_config_path=players_config_path.resolve(),
        run_label=players_config_path.stem,
        game_seed=game_config.seed,
        run_tags=game_config.run_tags,
    )
    resume_game_id = _load_resume_game_id(resume_run_dir)
    engine = build_engine(game_config, player_configs, game_id=resume_game_id)
    players = build_players(player_configs, game_config)
    orchestrator = GameOrchestrator(
        engine,
        players,
        observation_builder=ObservationBuilder(
            recent_event_window=game_config.history_window,
            game_rules=build_game_rules(game_config.vps_to_win),
        ),
        run_dir=effective_run_dir,
        run_tags=game_config.run_tags,
        run_label=players_config_path.stem,
        game_seed=game_config.seed,
        players_config_path=str(players_config_path.resolve()),
        players_config_toml=players_config_toml,
        resume_run_dir=resume_run_dir,
        public_chat_enabled=game_config.public_chat_enabled,
        public_chat_message_chars=game_config.public_chat_message_chars,
        public_chat_history_limit=game_config.public_chat_history_limit,
        trading_chat_enabled=game_config.trading_chat_enabled,
        trading_chat_max_failed_attempts_per_turn=(
            game_config.trading_chat_max_failed_attempts_per_turn
        ),
        trading_chat_max_rooms_per_turn=game_config.trading_chat_max_rooms_per_turn,
        trading_chat_max_rounds_per_attempt=game_config.trading_chat_max_rounds_per_attempt,
        trading_chat_message_chars=game_config.trading_chat_message_chars,
        trading_chat_history_limit=game_config.trading_chat_history_limit,
        reporter=(
            DebugTerminalReporter(
                skip_setup=debug_from_setup,
                debug_trade=debug_trade,
            )
            if (debug or debug_from_setup or debug_trade)
            else TerminalReporter()
        ),
    )
    return orchestrator.run()


def _load_local_env(start_dir: Path) -> None:
    env_path = _find_dotenv(start_dir)
    if env_path is None:
        return
    load_dotenv(env_path, override=False)


def _resolve_requested_run_dir(
    *,
    requested_run_dir: str | Path | None,
    configured_run_dir: Path | None,
    players_config_path: Path | None = None,
    run_label: str | None = None,
    game_seed: int | None = None,
    run_tags: tuple[str, ...] = (),
) -> tuple[Path | None, Path | None]:
    if requested_run_dir is None:
        return configured_run_dir, None
    if requested_run_dir == _AUTO_RESUME_REQUEST:
        return None, _find_matching_resume_run_dir(
            configured_run_dir=configured_run_dir,
            players_config_path=players_config_path,
            run_label=run_label,
            game_seed=game_seed,
            run_tags=run_tags,
        )
    requested_path = Path(requested_run_dir)
    missing_resume_artifacts = _missing_resume_artifact_files(requested_path)
    if missing_resume_artifacts and len(missing_resume_artifacts) < len(
        _RESUME_ARTIFACT_FILES
    ):
        missing_list = ", ".join(missing_resume_artifacts)
        raise ValueError(
            "Requested run directory contains benchmark artifacts but is missing "
            f"required resume files: {missing_list}."
        )
    if _is_existing_run_directory(requested_path):
        return None, requested_path
    return requested_path, None


def _is_existing_run_directory(path: Path) -> bool:
    return not _missing_resume_artifact_files(path)


def _find_matching_resume_run_dir(
    *,
    configured_run_dir: Path | None,
    players_config_path: Path | None,
    run_label: str | None,
    game_seed: int | None,
    run_tags: tuple[str, ...],
) -> Path:
    if configured_run_dir is None:
        raise ValueError(
            "Cannot auto-resume without a configured `run_dir` in the game config."
        )
    if players_config_path is None:
        raise ValueError(
            "Cannot auto-resume without a concrete players config path to match."
        )
    if not configured_run_dir.exists() or not configured_run_dir.is_dir():
        raise ValueError(
            "Cannot auto-resume because the configured run directory does not exist: "
            f"{configured_run_dir}"
        )

    matching_runs: list[Path] = []
    for child in sorted(Path(configured_run_dir).iterdir()):
        if not child.is_dir() or not _is_existing_run_directory(child):
            continue
        metadata = read_json(child / "metadata.json") or {}
        if _resume_run_matches_config(
            metadata=metadata,
            players_config_path=players_config_path,
            run_label=run_label,
            game_seed=game_seed,
            run_tags=run_tags,
        ):
            matching_runs.append(child)

    if not matching_runs:
        raise ValueError(
            "No matching run directory found under "
            f"{configured_run_dir} for players config {players_config_path}."
        )
    if len(matching_runs) > 1:
        matching_list = ", ".join(str(path) for path in matching_runs)
        raise ValueError(
            "Multiple matching run directories found for this config: "
            f"{matching_list}"
        )
    return matching_runs[0]


def _resume_run_matches_config(
    *,
    metadata: dict,
    players_config_path: Path,
    run_label: str | None,
    game_seed: int | None,
    run_tags: tuple[str, ...],
) -> bool:
    matched = False

    metadata_players_config_path = metadata.get("players_config_path")
    if isinstance(metadata_players_config_path, str):
        if Path(metadata_players_config_path) != players_config_path:
            return False
        matched = True

    metadata_run_label = metadata.get("run_label")
    if isinstance(metadata_run_label, str):
        if run_label is not None and metadata_run_label != run_label:
            return False
        matched = True

    metadata_game_seed = metadata.get("game_seed")
    if metadata_game_seed is not None or game_seed is not None:
        if metadata_game_seed != game_seed:
            return False
        matched = True

    metadata_run_tags = metadata.get("run_tags")
    if isinstance(metadata_run_tags, list):
        normalized_run_tags = tuple(str(tag) for tag in metadata_run_tags)
        if normalized_run_tags != tuple(str(tag) for tag in run_tags):
            return False
        matched = True

    return matched


def _missing_resume_artifact_files(path: Path) -> tuple[str, ...]:
    if not path.exists() or not path.is_dir():
        return _RESUME_ARTIFACT_FILES
    return tuple(
        file_name
        for file_name in _RESUME_ARTIFACT_FILES
        if not (path / file_name).exists()
    )


def _load_resume_game_id(resume_run_dir: str | Path | None) -> str | None:
    if resume_run_dir is None:
        return None
    metadata = read_json(Path(resume_run_dir) / "metadata.json") or {}
    game_id = metadata.get("game_id")
    return str(game_id) if isinstance(game_id, str) else None


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
    parser.add_argument(
        "--run-dir",
        help=(
            "For a new run, use this as the base output directory. "
            "If it points to an existing run directory, resume that run in place."
        ),
    )
    parser.add_argument(
        "--resume-run",
        dest="run_dir",
        nargs="?",
        const=_AUTO_RESUME_REQUEST,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print each player prompt/answer and pause for N before continuing.",
    )
    parser.add_argument(
        "--debug-from-setup",
        action="store_true",
        help="Like --debug but skips pausing during the initial placement phase.",
    )
    parser.add_argument(
        "--debug-trade",
        action="store_true",
        help="Run silently until the first trade chat opens, then pause for every prompt.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run post-game analysis after the game ends and print a summary.",
    )
    args = parser.parse_args(argv)

    result = run_from_config_files(
        game_config_path=args.game,
        players_config_path=args.players,
        run_dir=args.run_dir,
        debug=args.debug,
        debug_from_setup=args.debug_from_setup,
        debug_trade=args.debug_trade,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    if args.analyze:
        run_directory = result.metadata.get("benchmark", {}).get("run_directory")
        if run_directory:
            from .analysis import analyze_game, print_terminal_summary

            analysis = analyze_game(run_directory)
            print_terminal_summary(analysis)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
