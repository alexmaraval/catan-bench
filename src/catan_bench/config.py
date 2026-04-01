from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True, slots=True)
class GameConfig:
    engine: str = "catanatron"
    seed: int | None = None
    discard_limit: int = 7
    vps_to_win: int = 10
    run_dir: Path | None = None
    run_tags: tuple[str, ...] = ()
    history_window: int | None = 40
    prompt_history_limit: int | None = 12
    public_chat_enabled: bool = False
    public_chat_message_chars: int = 500
    public_chat_history_limit: int | None = 40
    trading_chat_enabled: bool = False
    trading_chat_max_failed_attempts_per_turn: int = 5
    trading_chat_max_rooms_per_turn: int = 5
    trading_chat_max_rounds_per_attempt: int = 3
    trading_chat_message_chars: int = 500
    trading_chat_history_limit: int | None = 16


@dataclass(frozen=True, slots=True)
class PlayerConfig:
    id: str
    type: str = "random"
    seed: int | None = None
    model: str | None = None
    api_base: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2
    top_p: float | None = None
    reasoning_enabled: bool | None = None
    reasoning_effort: str | None = None
    timeout_seconds: float = 60.0


def _parse_game_payload(payload: dict, defaults: GameConfig) -> GameConfig:
    engine = str(payload.get("engine", defaults.engine))
    if engine != "catanatron":
        raise ValueError(
            f"Unsupported engine {engine!r}. Only 'catanatron' is supported."
        )

    seed = payload.get("seed", defaults.seed)
    run_dir = payload.get("run_dir", defaults.run_dir)
    run_tags = payload.get("run_tags", defaults.run_tags)
    history_window = payload.get("history_window", defaults.history_window)
    prompt_history_limit = payload.get("prompt_history_limit", defaults.prompt_history_limit)
    public_chat_history_limit = payload.get("public_chat_history_limit", defaults.public_chat_history_limit)
    trading_chat_history_limit = payload.get("trading_chat_history_limit", defaults.trading_chat_history_limit)

    if history_window is not None:
        history_window = int(history_window)
    if prompt_history_limit is not None:
        prompt_history_limit = int(prompt_history_limit)
        if prompt_history_limit < 0:
            raise ValueError(
                "`prompt_history_limit` must be non-negative when provided."
            )
    if public_chat_history_limit is not None:
        public_chat_history_limit = int(public_chat_history_limit)
    if trading_chat_history_limit is not None:
        trading_chat_history_limit = int(trading_chat_history_limit)
    if run_dir is not None:
        run_dir = Path(run_dir)
    if run_tags is None:
        run_tags = ()
    elif isinstance(run_tags, (list, tuple)):
        run_tags = tuple(str(tag).strip() for tag in run_tags if str(tag).strip())
    else:
        raise ValueError("`run_tags` must be a list of strings when provided.")

    return GameConfig(
        engine=engine,
        seed=None if seed is None else int(seed),
        discard_limit=int(payload.get("discard_limit", defaults.discard_limit)),
        vps_to_win=int(payload.get("vps_to_win", defaults.vps_to_win)),
        run_dir=run_dir,
        run_tags=run_tags,
        history_window=history_window,
        prompt_history_limit=prompt_history_limit,
        public_chat_enabled=bool(payload.get("public_chat_enabled", defaults.public_chat_enabled)),
        public_chat_message_chars=int(payload.get("public_chat_message_chars", defaults.public_chat_message_chars)),
        public_chat_history_limit=public_chat_history_limit,
        trading_chat_enabled=bool(payload.get("trading_chat_enabled", defaults.trading_chat_enabled)),
        trading_chat_max_failed_attempts_per_turn=int(payload.get("trading_chat_max_failed_attempts_per_turn", defaults.trading_chat_max_failed_attempts_per_turn)),
        trading_chat_max_rooms_per_turn=int(payload.get("trading_chat_max_rooms_per_turn", defaults.trading_chat_max_rooms_per_turn)),
        trading_chat_max_rounds_per_attempt=int(payload.get("trading_chat_max_rounds_per_attempt", defaults.trading_chat_max_rounds_per_attempt)),
        trading_chat_message_chars=int(payload.get("trading_chat_message_chars", defaults.trading_chat_message_chars)),
        trading_chat_history_limit=trading_chat_history_limit,
    )


def load_game_config(path: str | Path) -> GameConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    return _parse_game_payload(data.get("game", data), GameConfig())


def load_game_config_overrides(
    path: str | Path,
    base_config: GameConfig,
) -> GameConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    payload = data.get("game")
    if not isinstance(payload, dict):
        return base_config
    return _parse_game_payload(payload, base_config)


def load_player_configs(path: str | Path) -> list[PlayerConfig]:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    players_payload = data.get("players")
    if not isinstance(players_payload, list) or not players_payload:
        raise ValueError("Players config must define a non-empty [[players]] list.")

    seen_ids: set[str] = set()
    configs: list[PlayerConfig] = []
    for entry in players_payload:
        if not isinstance(entry, dict):
            raise ValueError("Each [[players]] entry must be a table.")

        player_id = str(entry["id"]).upper()
        if player_id in seen_ids:
            raise ValueError(f"Duplicate player id {player_id!r} in players config.")
        seen_ids.add(player_id)

        player_type = str(entry.get("type", "random"))
        if player_type not in {"random", "first_legal", "llm"}:
            raise ValueError(
                f"Unsupported player type {player_type!r}. "
                "Expected 'random', 'first_legal', or 'llm'."
            )

        seed = entry.get("seed")
        model = str(entry["model"]) if "model" in entry else None
        if player_type == "llm" and not model:
            raise ValueError("LLM players must define a non-empty `model`.")
        api_base = str(entry.get("api_base", "https://api.openai.com/v1"))
        api_key_env = str(entry.get("api_key_env", "OPENAI_API_KEY"))
        temperature = float(entry.get("temperature", 0.2))
        top_p = float(entry["top_p"]) if "top_p" in entry else None
        if "prompt_history_limit" in entry:
            raise ValueError(
                "`prompt_history_limit` now belongs in the game config, not individual [[players]] entries."
            )
        reasoning_enabled = bool(entry["reasoning_enabled"]) if "reasoning_enabled" in entry else None
        reasoning_effort = entry.get("reasoning_effort")
        if reasoning_effort is not None:
            reasoning_effort = str(reasoning_effort).strip().lower()
            if reasoning_effort not in {
                "none",
                "minimal",
                "low",
                "medium",
                "high",
                "xhigh",
            }:
                raise ValueError(
                    "`reasoning_effort` must be one of "
                    "'none', 'minimal', 'low', 'medium', 'high', or 'xhigh'."
                )
        if reasoning_enabled is not None and reasoning_effort is not None:
            raise ValueError(
                "Choose either `reasoning_enabled` or `reasoning_effort` in a player config, not both."
            )
        timeout_seconds = float(entry.get("timeout_seconds", 60.0))
        configs.append(
            PlayerConfig(
                id=player_id,
                type=player_type,
                seed=None if seed is None else int(seed),
                model=model,
                api_base=api_base,
                api_key_env=api_key_env,
                temperature=temperature,
                top_p=top_p,
                reasoning_enabled=reasoning_enabled,
                reasoning_effort=reasoning_effort,
                timeout_seconds=timeout_seconds,
            )
        )

    if not 2 <= len(configs) <= 4:
        raise ValueError("Catan requires between 2 and 4 configured players.")

    return configs
