"""Terminal reporter: prints a live human-readable game summary to stderr."""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from .schemas import Action, DecisionPoint, GameResult, PlayerResponse, TransitionResult

# ── ANSI helpers ─────────────────────────────────────────────────────────────

_RESET = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"

_PLAYER_ANSI: dict[str, str] = {
    "RED":    "\033[31m",
    "BLUE":   "\033[34m",
    "ORANGE": "\033[33m",
    "WHITE":  "\033[97m",
    "GREEN":  "\033[32m",
    "TEAL":   "\033[36m",
}


def _c(text: str, code: str, *, on: bool) -> str:
    return f"{code}{text}{_RESET}" if on else text


# ── Reporter ─────────────────────────────────────────────────────────────────

class TerminalReporter:
    """Emits structured, coloured game progress to *file* (default: stderr)."""

    def __init__(self, *, file: TextIO | None = None) -> None:
        self._file = file or sys.stderr
        self._colour = getattr(self._file, "isatty", lambda: False)()
        self._last_turn: int = -1

    # ── hooks ────────────────────────────────────────────────────────────────

    def on_game_start(self, game_id: str, player_ids: list[str]) -> None:
        players = "  ·  ".join(self._player(p) for p in player_ids)
        self._put(f"\nGame  {_c(game_id, _DIM, on=self._colour)}")
        self._put(f"Players  {players}\n")

    def on_step(
        self,
        *,
        decision: DecisionPoint,
        action: Action,
        response: PlayerResponse,
        transition: TransitionResult,
    ) -> None:
        if decision.turn_index != self._last_turn:
            self._last_turn = decision.turn_index
            label = f" Turn {decision.turn_index} "
            bar = _c("─" * 4 + label + "─" * max(0, 42 - len(label)), _DIM, on=self._colour)
            self._put(f"\n{bar}")

        # Collect descriptions from public events; fall back to action description
        descs = [d for e in transition.public_events if (d := _describe(e)) is not None]
        summary = "  ·  ".join(descs) if descs else (action.description or action.action_type)

        pid = decision.acting_player_id
        pad = " " * max(0, 8 - len(pid))
        self._put(f"  {self._player(pid)}{pad}  {summary}")

    def on_game_end(self, result: GameResult) -> None:
        bar = _c("─" * 4 + " Game over " + "─" * 31, _DIM, on=self._colour)
        self._put(f"\n{bar}")
        if result.winner_ids:
            winners = "  ·  ".join(self._player(w) for w in result.winner_ids)
            self._put(f"  Winner     {winners}")
        else:
            self._put("  No winner.")
        self._put(f"  Decisions  {result.total_decisions:,}")
        self._put("")

    # ── internal ─────────────────────────────────────────────────────────────

    def _player(self, player_id: str) -> str:
        code = _PLAYER_ANSI.get(player_id.upper(), _BOLD)
        return _c(player_id, code, on=self._colour)

    def _put(self, text: str) -> None:
        print(text, file=self._file, flush=True)


# ── Event description ─────────────────────────────────────────────────────────

def _describe(event: object) -> str | None:
    """Return a short human-readable string for a public event, or None to skip."""
    kind: str = getattr(event, "kind", "")
    p: dict = getattr(event, "payload", {})

    if kind == "dice_rolled":
        result = p.get("result")
        if isinstance(result, list) and result:
            total = sum(int(v) for v in result)
            breakdown = "+".join(str(v) for v in result)
            return f"rolled {total} ({breakdown})"
        if isinstance(result, dict):
            total = result.get("total")
            values = result.get("values") or result.get("dice") or result.get("rolls")
            if isinstance(total, int) and isinstance(values, list):
                return f"rolled {total} ({'+'.join(str(v) for v in values)})"
            if isinstance(total, int):
                return f"rolled {total}"
        if result is not None:
            return f"rolled {result}"
        return "rolled dice"

    if kind == "settlement_built":
        return f"settlement on node {p.get('node_id')}"
    if kind == "city_built":
        return f"city on node {p.get('node_id')}"
    if kind == "road_built":
        return f"road on {p.get('edge')}"

    if kind == "robber_moved":
        coord  = p.get("coordinate")
        victim = p.get("victim")
        return f"robber → {coord}, stole from {victim}" if victim else f"robber → {coord}"

    if kind == "trade_offered":
        return f"offered {_res(p.get('offer'))} for {_res(p.get('request'))}"
    if kind == "trade_accepted":
        return f"accepted {p.get('offering_player_id') or '?'}'s offer"
    if kind == "trade_rejected":
        return f"rejected {p.get('offering_player_id') or '?'}'s offer"
    if kind == "trade_confirmed":
        a = p.get("offering_player_id") or "?"
        b = p.get("accepting_player_id") or "?"
        return f"{a} ↔ {b}: {_res(p.get('offer'))} for {_res(p.get('request'))}"
    if kind == "trade_cancelled":
        return "trade cancelled"
    if kind == "trade_chat_opened":
        requested = _res(p.get("requested_resources"))
        return f"opened trade chat for {requested}"
    if kind == "trade_chat_message":
        message = p.get("message")
        if isinstance(message, str) and message.strip():
            return f"said: {message.strip()}"
        offer = p.get("offer")
        request = p.get("request")
        if isinstance(offer, dict) and isinstance(request, dict):
            return f"quoted {_res(offer)} for {_res(request)}"
        return "spoke in trade chat"
    if kind == "trade_chat_quote_selected":
        return f"selected {p.get('selected_player_id')}'s quote"
    if kind == "trade_chat_no_deal":
        return "ended trade chat with no deal"
    if kind == "trade_chat_closed":
        return None

    if kind == "development_card_played":
        action_p = p.get("action") if isinstance(p.get("action"), dict) else {}
        desc = action_p.get("description") if action_p else None
        if desc:
            return str(desc)
        at = action_p.get("action_type") if action_p else None
        return f"played {str(at).lower().replace('_', ' ')}" if at else "played dev card"

    if kind == "turn_ended":
        return None  # turn boundary already shown by the header

    return None


def _res(resource_map: object) -> str:
    if not isinstance(resource_map, dict) or not resource_map:
        return str(resource_map or "nothing")
    return ", ".join(f"{qty} {r}" for r, qty in sorted(resource_map.items()))
