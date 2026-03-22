from .config import GameConfig, PlayerConfig, load_game_config, load_player_configs
from .engine import EngineAdapter
from .observations import ObservationBuilder
from .orchestrator import GameOrchestrator, InvalidActionError, MissingPlayerError
from .players import FirstLegalPlayer, Player, RandomLegalPlayer, ScriptedPlayer
from .replay import ReplayTimelineItem, build_replay_timeline, export_replay_html
from .runner import build_engine, build_players, run_from_config_files
from .schemas import (
    Action,
    DecisionPoint,
    Event,
    GameResult,
    MemoryEntry,
    Observation,
    PlayerResponse,
    TransitionResult,
)
from .storage import EventLog, MemoryStore

try:
    from .catanatron_adapter import CatanatronEngineAdapter
except RuntimeError:  # pragma: no cover - dependency may be intentionally absent.
    CatanatronEngineAdapter = None

__all__ = [
    "Action",
    "CatanatronEngineAdapter",
    "DecisionPoint",
    "EngineAdapter",
    "Event",
    "EventLog",
    "FirstLegalPlayer",
    "GameConfig",
    "GameOrchestrator",
    "GameResult",
    "InvalidActionError",
    "PlayerConfig",
    "MemoryEntry",
    "MemoryStore",
    "MissingPlayerError",
    "Observation",
    "ObservationBuilder",
    "Player",
    "PlayerResponse",
    "RandomLegalPlayer",
    "ReplayTimelineItem",
    "ScriptedPlayer",
    "TransitionResult",
    "build_engine",
    "build_players",
    "build_replay_timeline",
    "export_replay_html",
    "load_game_config",
    "load_player_configs",
    "run_from_config_files",
]
