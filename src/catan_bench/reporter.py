"""Terminal reporter: prints a live human-readable game summary to stderr."""
from __future__ import annotations

import json
import shutil
import sys
import textwrap
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from .schemas import (
        Action,
        ActionDecision,
        DecisionPoint,
        GameResult,
        PromptTrace,
        TransitionResult,
    )

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
        response: ActionDecision,
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


_SETUP_PHASES = {"build_initial_settlement", "build_initial_road"}
_TRADE_CHAT_STAGES = {"open", "reply", "select"}


class DebugTerminalReporter(TerminalReporter):
    """Interactive terminal reporter that prints prompt traces and pauses after each one."""

    def __init__(
        self,
        *,
        file: TextIO | None = None,
        input_file: TextIO | None = None,
        skip_setup: bool = False,
        debug_trade: bool = False,
    ) -> None:
        super().__init__(file=file)
        self._input_file = input_file or sys.stdin
        self._skip_setup = skip_setup
        self._debug_trade = debug_trade
        self._trade_chat_triggered = False

    def on_prompt_trace(self, trace: PromptTrace) -> None:
        if self._skip_setup and trace.phase in _SETUP_PHASES:
            return
        if self._debug_trade and not self._trade_chat_triggered:
            if trace.stage in _TRADE_CHAT_STAGES:
                self._trade_chat_triggered = True
            else:
                return
        header = (
            f" Debug {trace.player_id} "
            f"[turn={trace.turn_index} phase={trace.phase} decision={trace.decision_index} "
            f"stage={trace.stage}] "
        )
        bar = _c("─" * 3 + header + "─" * max(0, 24 - len(header)), _DIM, on=self._colour)
        self._put(f"\n{bar}")
        self._put(f"  model  {trace.model}")
        self._put(f"  temperature  {trace.temperature}")
        for attempt_index, attempt in enumerate(trace.attempts, start=1):
            self._put(f"  attempt {attempt_index}")
            self._put_split_columns(
                left_title="prompt",
                left_lines=self._render_attempt_messages(attempt.messages),
                right_title="answer",
                right_lines=self._render_attempt_response(attempt.response_text, attempt.response),
            )
            self._wait_for_next()

    def _wait_for_next(self) -> None:
        while True:
            self._put("  Press N then Enter to continue.")
            raw_value = self._input_file.readline()
            if raw_value == "":
                return
            if raw_value.strip().upper() == "N":
                return
            self._put("  Input not recognized. Use N to continue.")

    def _put_split_columns(
        self,
        *,
        left_title: str,
        left_lines: list[str],
        right_title: str,
        right_lines: list[str],
    ) -> None:
        total_width = self._available_width()
        gutter = "   |   "
        column_width = max(24, (total_width - len(gutter) - 2) // 2)
        self._put(
            f"  {left_title.upper():<{column_width}}{gutter}{right_title.upper():<{column_width}}"
        )
        self._put(
            f"  {'-' * column_width}{gutter}{'-' * column_width}"
        )
        wrapped_left = self._wrap_lines(left_lines, width=column_width)
        wrapped_right = self._wrap_lines(right_lines, width=column_width)
        row_count = max(len(wrapped_left), len(wrapped_right))
        for row_index in range(row_count):
            left = wrapped_left[row_index] if row_index < len(wrapped_left) else ""
            right = wrapped_right[row_index] if row_index < len(wrapped_right) else ""
            self._put(f"  {left:<{column_width}}{gutter}{right:<{column_width}}")

    def _available_width(self) -> int:
        if getattr(self._file, "isatty", lambda: False)():
            return max(72, shutil.get_terminal_size(fallback=(120, 24)).columns)
        return 120

    @staticmethod
    def _render_attempt_messages(messages: tuple[dict[str, object], ...]) -> list[str]:
        rendered: list[str] = []
        for message in messages:
            role = str(message.get("role", "unknown"))
            content = message.get("content")
            rendered.append(f"[{role}]")
            if role == "system":
                rendered.append(DebugTerminalReporter._collapsed_system_prompt_summary(content))
            else:
                rendered.extend(DebugTerminalReporter._render_debug_content(content))
            rendered.append("")
        if rendered and rendered[-1] == "":
            rendered.pop()
        return rendered

    @staticmethod
    def _render_attempt_response(
        response_text: str | None, parsed_response: dict[str, object]
    ) -> list[str]:
        if response_text is not None:
            return DebugTerminalReporter._render_debug_content(response_text)
        return DebugTerminalReporter._render_debug_content(parsed_response)

    @staticmethod
    def _render_debug_content(content: object) -> list[str]:
        if isinstance(content, str):
            parsed = DebugTerminalReporter._maybe_parse_json_text(content)
            if parsed is not None:
                return DebugTerminalReporter._render_structured_json(parsed)
            return content.splitlines() or [""]
        return DebugTerminalReporter._render_structured_json(content)

    @staticmethod
    def _collapsed_system_prompt_summary(content: object) -> str:
        lines = DebugTerminalReporter._render_debug_content(content)
        non_empty_lines = [line for line in lines if line.strip()]
        line_count = len(non_empty_lines)
        if line_count <= 0:
            return "(collapsed static system prompt)"
        return f"(collapsed static system prompt: {line_count} lines)"

    @staticmethod
    def _maybe_parse_json_text(content: str) -> object | None:
        stripped = DebugTerminalReporter._strip_markdown_fences(content).strip()
        if not stripped or stripped[0] not in "{[":
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _strip_markdown_fences(content: str) -> str:
        stripped = content.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if not lines:
            return stripped
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _render_structured_json(content: object) -> list[str]:
        if isinstance(content, dict):
            return DebugTerminalReporter._render_json_mapping(content)
        return json.dumps(content, indent=2, sort_keys=True).splitlines()

    @staticmethod
    def _render_json_mapping(mapping: dict[object, object]) -> list[str]:
        rendered: list[str] = []
        for key, value in mapping.items():
            key_label = str(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                rendered.append(f"{key_label}: {json.dumps(value, sort_keys=True)}")
            else:
                rendered.append(f"{key_label}:")
                rendered.extend(json.dumps(value, indent=2, sort_keys=True).splitlines())
            rendered.append("")
        if rendered and rendered[-1] == "":
            rendered.pop()
        return rendered

    @staticmethod
    def _wrap_lines(lines: list[str], *, width: int) -> list[str]:
        wrapped: list[str] = []
        for line in lines:
            if not line:
                wrapped.append("")
                continue
            wrapped.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                or [""]
            )
        return wrapped


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
    if kind == "trade_counter_offered":
        owner = p.get("owner_player_id") or "?"
        return f"countered {owner}: {_res(p.get('offer'))} for {_res(p.get('request'))}"
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
