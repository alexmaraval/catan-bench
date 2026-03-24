from .config import GameConfig, PlayerConfig, load_game_config, load_player_configs
from .engine import EngineAdapter
from .llm import LLMClient, LLMRequestTooLargeError, OpenAICompatibleChatClient
from .observations import ObservationBuilder
from .orchestrator import GameOrchestrator, InvalidActionError, MissingPlayerError
from .players import FirstLegalPlayer, LLMPlayer, Player, RandomLegalPlayer, ScriptedPlayer
from .prompting import PromptRenderer
from .prompts import CATAN_RULES_SUMMARY
from .runner import build_engine, build_players, run_from_config_files
from .schemas import (
    Action,
    ActionDecision,
    ActionObservation,
    ActionTraceEntry,
    DecisionPoint,
    Event,
    GameResult,
    MemorySnapshot,
    OpeningStrategyObservation,
    OpeningStrategyResponse,
    PlayerMemory,
    PromptTrace,
    PromptTraceAttempt,
    PublicStateSnapshot,
    ReactiveObservation,
    TradeChatObservation,
    TradeChatOpenResponse,
    TradeChatOwnerDecisionResponse,
    TradeChatProposal,
    TradeChatQuote,
    TradeChatReplyResponse,
    TransitionResult,
    TurnEndObservation,
    TurnEndResponse,
    TurnStartObservation,
    TurnStartResponse,
)
from .storage import ActionTraceStore, EventLog, MemoryStore, PromptTraceStore, PublicStateStore

try:
    from .catanatron_adapter import CatanatronEngineAdapter
except RuntimeError:  # pragma: no cover - dependency may be intentionally absent.
    CatanatronEngineAdapter = None

__all__ = [
    "Action",
    "ActionDecision",
    "ActionObservation",
    "ActionTraceEntry",
    "ActionTraceStore",
    "CATAN_RULES_SUMMARY",
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
    "LLMPlayer",
    "LLMRequestTooLargeError",
    "MemorySnapshot",
    "MemoryStore",
    "MissingPlayerError",
    "ObservationBuilder",
    "OpeningStrategyObservation",
    "OpeningStrategyResponse",
    "OpenAICompatibleChatClient",
    "Player",
    "PlayerConfig",
    "PlayerMemory",
    "PromptRenderer",
    "PromptTrace",
    "PromptTraceAttempt",
    "PromptTraceStore",
    "PublicStateSnapshot",
    "PublicStateStore",
    "RandomLegalPlayer",
    "ReactiveObservation",
    "ScriptedPlayer",
    "TradeChatObservation",
    "TradeChatOpenResponse",
    "TradeChatOwnerDecisionResponse",
    "TradeChatProposal",
    "TradeChatQuote",
    "TradeChatReplyResponse",
    "TransitionResult",
    "TurnEndObservation",
    "TurnEndResponse",
    "TurnStartObservation",
    "TurnStartResponse",
    "build_engine",
    "build_players",
    "load_game_config",
    "load_player_configs",
    "run_from_config_files",
]
