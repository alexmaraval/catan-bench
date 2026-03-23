from typing import TYPE_CHECKING

from .config import GameConfig, PlayerConfig, load_game_config, load_player_configs
from .engine import EngineAdapter
from .llm import LLMClient, LLMRequestTooLargeError, OpenAICompatibleChatClient
from .observations import ObservationBuilder
from .orchestrator import GameOrchestrator, InvalidActionError, MissingPlayerError
from .players import FirstLegalPlayer, LLMPlayer, Player, RandomLegalPlayer, ScriptedPlayer
from .prompts import CATAN_RULES_SUMMARY
from .runner import build_engine, build_players, run_from_config_files
from .schemas import (
    Action,
    DecisionPoint,
    Event,
    GameResult,
    MemoryEntry,
    MemoryResponse,
    Observation,
    PlayerResponse,
    PromptTrace,
    PromptTraceAttempt,
    RecallObservation,
    ReflectionObservation,
    TradeChatObservation,
    TradeChatOpenResponse,
    TradeChatQuote,
    TradeChatReplyResponse,
    TradeChatSelectionResponse,
    TransitionResult,
)
from .storage import EventLog, MemoryStore, PromptTraceStore

if TYPE_CHECKING:  # pragma: no cover - import only for static analysis.
    from .replay import (
        ReplayTimelineItem,
        build_player_replay_timeline,
        build_replay_timeline,
        export_player_replay_html,
        export_replay_html,
    )

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
    "LLMClient",
    "LLMRequestTooLargeError",
    "LLMPlayer",
    "PlayerConfig",
    "MemoryEntry",
    "MemoryResponse",
    "MemoryStore",
    "MissingPlayerError",
    "Observation",
    "OpenAICompatibleChatClient",
    "ObservationBuilder",
    "Player",
    "PlayerResponse",
    "PromptTrace",
    "PromptTraceAttempt",
    "PromptTraceStore",
    "RandomLegalPlayer",
    "RecallObservation",
    "ReplayTimelineItem",
    "ReflectionObservation",
    "ScriptedPlayer",
    "TradeChatObservation",
    "TradeChatOpenResponse",
    "TradeChatQuote",
    "TradeChatReplyResponse",
    "TradeChatSelectionResponse",
    "TransitionResult",
    "CATAN_RULES_SUMMARY",
    "build_engine",
    "build_player_replay_timeline",
    "build_players",
    "build_replay_timeline",
    "export_player_replay_html",
    "export_replay_html",
    "load_game_config",
    "load_player_configs",
    "run_from_config_files",
]


def __getattr__(name):
    if name in {
        "ReplayTimelineItem",
        "build_player_replay_timeline",
        "build_replay_timeline",
        "export_player_replay_html",
        "export_replay_html",
    }:
        from . import replay as replay_module

        return getattr(replay_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
