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
    run_dir: str | None = None


@dataclass(frozen=True, slots=True)
class PlayerConfig:
    id: str
    type: str = "random"
    seed: int | None = None
    allow_trade_offers: bool = False


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
    if run_dir is not None:
        run_dir = str(run_dir)

    return GameConfig(
        engine=engine,
        seed=None if seed is None else int(seed),
        discard_limit=discard_limit,
        vps_to_win=vps_to_win,
        run_dir=run_dir,
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
        if player_type not in {"random", "first_legal"}:
            raise ValueError(
                f"Unsupported player type {player_type!r}. "
                "Expected 'random' or 'first_legal'."
            )

        seed = entry.get("seed")
        allow_trade_offers = bool(entry.get("allow_trade_offers", False))
        configs.append(
            PlayerConfig(
                id=player_id,
                type=player_type,
                seed=None if seed is None else int(seed),
                allow_trade_offers=allow_trade_offers,
            )
        )

    if not 2 <= len(configs) <= 4:
        raise ValueError("Catan requires between 2 and 4 configured players.")

    return configs
