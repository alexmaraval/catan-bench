from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


VALID_PLAYER_IDS = ("RED", "BLUE", "ORANGE", "WHITE")


@dataclass(frozen=True, slots=True)
class GameConfig:
    engine: str = "catanatron"
    seed: int | None = None
    discard_limit: int = 7
    vps_to_win: int = 10
    run_dir: Path | None = None
    history_window: int | None = 40


@dataclass(frozen=True, slots=True)
class PlayerConfig:
    id: str
    type: str = "random"
    seed: int | None = None
    allow_trade_offers: bool = False
    model: str | None = None
    api_base: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2


def load_game_config(path: str | Path) -> GameConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    payload = data.get("game", data)
    engine = str(payload.get("engine", "catanatron"))
    if engine != "catanatron":
        raise ValueError(f"Unsupported engine {engine!r}. Only 'catanatron' is supported.")

    seed = payload.get("seed")
    discard_limit = int(payload.get("discard_limit", 7))
    vps_to_win = int(payload.get("vps_to_win", 10))
    run_dir = payload.get("run_dir")
    history_window = payload.get("history_window", 40)
    if history_window is not None:
        history_window = int(history_window)
    if run_dir is not None:
        run_dir = Path(run_dir)

    return GameConfig(
        engine=engine,
        seed=None if seed is None else int(seed),
        discard_limit=discard_limit,
        vps_to_win=vps_to_win,
        run_dir=run_dir,
        history_window=history_window,
    )


def load_player_configs(path: str | Path) -> list[PlayerConfig]:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    players_payload = data.get("players")
    if not isinstance(players_payload, list) or not players_payload:
        raise ValueError("players.toml must define a non-empty [[players]] list.")

    seen_ids: set[str] = set()
    configs: list[PlayerConfig] = []
    for entry in players_payload:
        if not isinstance(entry, dict):
            raise ValueError("Each [[players]] entry must be a table.")

        player_id = str(entry["id"]).upper()
        if player_id not in VALID_PLAYER_IDS:
            raise ValueError(
                f"Unsupported player id {player_id!r}. Expected one of {VALID_PLAYER_IDS}."
            )
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
        allow_trade_offers = bool(entry.get("allow_trade_offers", False))
        model = entry.get("model")
        if model is not None:
            model = str(model)
        if player_type == "llm" and not model:
            raise ValueError("LLM players must define a non-empty `model`.")
        api_base = str(entry.get("api_base", "https://api.openai.com/v1"))
        api_key_env = str(entry.get("api_key_env", "OPENAI_API_KEY"))
        temperature = float(entry.get("temperature", 0.2))
        configs.append(
            PlayerConfig(
                id=player_id,
                type=player_type,
                seed=None if seed is None else int(seed),
                allow_trade_offers=allow_trade_offers,
                model=model,
                api_base=api_base,
                api_key_env=api_key_env,
                temperature=temperature,
            )
        )

    if not 2 <= len(configs) <= 4:
        raise ValueError("Catan requires between 2 and 4 configured players.")

    return configs
