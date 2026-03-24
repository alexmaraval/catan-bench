from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from .schemas import Event, JsonValue, MemorySnapshot, PromptTrace, PublicStateSnapshot

PLAYER_COLORS: dict[str, tuple[str, str, str]] = {
    "RED": ("#dc2626", "#fee2e2", "#dc2626"),
    "BLUE": ("#2563eb", "#dbeafe", "#2563eb"),
    "ORANGE": ("#ea580c", "#ffedd5", "#ea580c"),
    "WHITE": ("#7c3aed", "#ede9fe", "#7c3aed"),
}
NEUTRAL_COLORS = ("#374151", "#f3f4f6", "#9ca3af")
TRADE_EVENT_KINDS = {
    "trade_offered",
    "trade_accepted",
    "trade_rejected",
    "trade_confirmed",
    "trade_cancelled",
    "trade_chat_opened",
    "trade_chat_message",
    "trade_chat_quote_selected",
    "trade_chat_no_deal",
    "trade_chat_closed",
}
TRADE_CLOSE_KINDS = {"trade_confirmed", "trade_cancelled"}
TURN_DIGEST_EVENT_KINDS = frozenset(
    {
        "dice_rolled",
        "settlement_built",
        "city_built",
        "road_built",
        "robber_moved",
        "trade_chat_opened",
        "trade_confirmed",
        "trade_chat_no_deal",
        "trade_cancelled",
        "development_card_played",
        "action_taken",
        "turn_ended",
    }
)
EVENT_EMOJIS = {
    "dice_rolled": "🎲",
    "settlement_built": "🏠",
    "city_built": "🏙️",
    "road_built": "🛣️",
    "robber_moved": "🥷",
    "trade_offered": "🤝",
    "trade_accepted": "✅",
    "trade_rejected": "❌",
    "trade_confirmed": "🤝",
    "trade_cancelled": "🚫",
    "trade_chat_opened": "💬",
    "trade_chat_message": "🗨️",
    "trade_chat_quote_selected": "📌",
    "trade_chat_no_deal": "🙅",
    "trade_chat_closed": "🔒",
    "turn_ended": "⏹️",
    "development_card_played": "🃏",
    "action_taken": "⚙️",
}
MEMORY_EMOJI = "🧠"
PROMPT_EMOJI = "🧾"
TURN_EMOJI = "🌀"
# Explicit lifecycle order so traces sort correctly when history_index and
# decision_index tie (e.g. turn_start and choose_action can share both).
STAGE_ORDER: dict[str, int] = {
    "opening_strategy": 0,
    "turn_start": 1,
    "choose_action": 2,
    "trade_chat_open": 3,
    "trade_chat_reply": 4,
    "trade_chat_owner_decision": 5,
    "trade_chat_select": 5,
    "trade_chat_no_deal": 6,
    "reactive_action": 7,
    "repair_action": 8,
    "turn_end": 9,
    "turn_cleanup": 10,
}
TRADE_EMOJI = "🤝"
RESOURCE_TILE_COLORS = {
    "WOOD": "#7fb069",
    "BRICK": "#d97757",
    "SHEEP": "#a7d676",
    "WHEAT": "#f1d36b",
    "ORE": "#9ca3af",
}
PLAYER_STROKES = {
    "RED": "#991b1b",
    "BLUE": "#1e40af",
    "ORANGE": "#c2410c",
    "WHITE": "#5b21b6",
}
NODE_DIRECTION_ANGLES = {
    "NORTH": -90,
    "NORTHEAST": -30,
    "SOUTHEAST": 30,
    "SOUTH": 90,
    "SOUTHWEST": 150,
    "NORTHWEST": 210,
}
EDGE_DIRECTION_VERTICES = {
    "NORTHEAST": ("NORTH", "NORTHEAST"),
    "EAST": ("NORTHEAST", "SOUTHEAST"),
    "SOUTHEAST": ("SOUTHEAST", "SOUTH"),
    "SOUTHWEST": ("SOUTH", "SOUTHWEST"),
    "WEST": ("SOUTHWEST", "NORTHWEST"),
    "NORTHWEST": ("NORTHWEST", "NORTH"),
}


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    run_dir: Path
    metadata: dict[str, JsonValue]
    player_ids: tuple[str, ...]
    public_events: tuple[Event, ...]
    public_state_snapshots: tuple[PublicStateSnapshot, ...]
    memory_traces_by_player: dict[str, tuple[MemorySnapshot, ...]]
    prompt_traces_by_player: dict[str, tuple[PromptTrace, ...]]
    result: dict[str, JsonValue] | None

    @property
    def max_history_index(self) -> int:
        if self.public_state_snapshots:
            return max(snapshot.history_index for snapshot in self.public_state_snapshots)
        return 0


@dataclass(frozen=True, slots=True)
class TradeArtifact:
    owner_player_id: str
    turn_index: int
    start_history_index: int
    end_history_index: int
    events: tuple[Event, ...]


@dataclass(frozen=True, slots=True)
class TurnArtifact:
    player_id: str
    turn_index: int
    start_history_index: int
    end_history_index: int
    events: tuple[Event, ...]
    memory_snapshots: tuple[MemorySnapshot, ...]
    prompt_traces: tuple[PromptTrace, ...]
    trade_artifacts: tuple[TradeArtifact, ...]
    phases: tuple[str, ...]


@dataclass(slots=True)
class _TurnBucket:
    events: list[Event] = field(default_factory=list)
    memory_snapshots: list[MemorySnapshot] = field(default_factory=list)
    prompt_traces: list[PromptTrace] = field(default_factory=list)


def load_dashboard_snapshot(run_dir: str | Path) -> DashboardSnapshot:
    run_path = Path(run_dir)
    metadata = _read_json_if_exists(run_path / "metadata.json") or {}
    player_ids = _player_ids_from_run(run_path, metadata)
    public_events = tuple(
        Event.from_dict(entry) for entry in _read_jsonl(run_path / "public_history.jsonl")
    )
    public_state_snapshots = tuple(
        PublicStateSnapshot.from_dict(entry)
        for entry in _read_jsonl(run_path / "public_state_trace.jsonl")
    )
    memory_traces_by_player = {
        player_id: tuple(
            MemorySnapshot.from_dict(entry)
            for entry in _read_jsonl(run_path / "players" / player_id / "memory_trace.jsonl")
        )
        for player_id in player_ids
    }
    prompt_traces_by_player = {
        player_id: tuple(
            PromptTrace.from_dict(entry)
            for entry in _read_jsonl(run_path / "players" / player_id / "prompt_trace.jsonl")
        )
        for player_id in player_ids
    }
    result = _read_json_if_exists(run_path / "result.json")
    return DashboardSnapshot(
        run_dir=run_path,
        metadata=metadata,
        player_ids=player_ids,
        public_events=public_events,
        public_state_snapshots=public_state_snapshots,
        memory_traces_by_player=memory_traces_by_player,
        prompt_traces_by_player=prompt_traces_by_player,
        result=result,
    )


def discover_run_directories(base_run_dir: str | Path) -> tuple[Path, ...]:
    base_path = Path(base_run_dir)
    if _is_run_directory(base_path):
        return (base_path,)
    if not base_path.exists() or not base_path.is_dir():
        return ()
    candidates = [path for path in base_path.iterdir() if path.is_dir() and _is_run_directory(path)]
    return tuple(sorted(candidates, key=_run_sort_key, reverse=True))


def build_player_timelines(
    snapshot: DashboardSnapshot,
    *,
    cursor: int,
) -> dict[str, tuple[TurnArtifact, ...]]:
    timelines: dict[str, tuple[TurnArtifact, ...]] = {}
    for player_id in snapshot.player_ids:
        bucket_by_turn: dict[int, _TurnBucket] = {}
        for event in snapshot.public_events:
            if event.history_index > cursor:
                break
            if not _event_relevant_to_player(event, player_id):
                continue
            bucket_by_turn.setdefault(event.turn_index, _TurnBucket()).events.append(event)
        for snapshot_entry in snapshot.memory_traces_by_player.get(player_id, ()):
            if snapshot_entry.history_index > cursor:
                break
            bucket_by_turn.setdefault(snapshot_entry.turn_index, _TurnBucket()).memory_snapshots.append(
                snapshot_entry
            )
        for trace in snapshot.prompt_traces_by_player.get(player_id, ()):
            if trace.history_index > cursor:
                break
            bucket_by_turn.setdefault(trace.turn_index, _TurnBucket()).prompt_traces.append(trace)

        artifacts: list[TurnArtifact] = []
        for turn_index, bucket in sorted(
            bucket_by_turn.items(),
            key=lambda item: (
                _bucket_start_history_index(item[1]),
                item[0],
            ),
        ):
            if not (bucket.events or bucket.memory_snapshots or bucket.prompt_traces):
                continue
            sorted_events = tuple(sorted(bucket.events, key=lambda item: item.history_index))
            sorted_memories = tuple(
                sorted(
                    bucket.memory_snapshots,
                    key=lambda item: (item.history_index, item.decision_index, STAGE_ORDER.get(item.stage, 99)),
                )
            )
            sorted_traces = tuple(
                sorted(
                    bucket.prompt_traces,
                    key=lambda item: (item.history_index, item.decision_index, STAGE_ORDER.get(item.stage, 99)),
                )
            )
            phases = tuple(
                dict.fromkeys(
                    [event.phase for event in sorted_events]
                    + [memory.phase for memory in sorted_memories]
                    + [trace.phase for trace in sorted_traces]
                )
            )
            history_points = [
                item.history_index for item in (*sorted_events, *sorted_memories, *sorted_traces)
            ]
            trade_artifacts = _build_trade_artifacts(player_id, sorted_events)
            artifacts.append(
                TurnArtifact(
                    player_id=player_id,
                    turn_index=turn_index,
                    start_history_index=min(history_points) if history_points else 0,
                    end_history_index=max(history_points) if history_points else 0,
                    events=sorted_events,
                    memory_snapshots=sorted_memories,
                    prompt_traces=sorted_traces,
                    trade_artifacts=trade_artifacts,
                    phases=phases,
                )
            )
        timelines[player_id] = tuple(artifacts)
    return timelines


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    render_dashboard(default_run_dir=args.run_dir)


def render_dashboard(*, default_run_dir: str | Path) -> None:
    st = _require_streamlit()
    st.set_page_config(
        page_title="catan-bench live dashboard",
        page_icon="🎲",
        layout="wide",
    )
    _inject_styles(st)

    st.title("catan-bench live dashboard")
    st.caption("Turn-focused view: navigate turns to inspect every LLM interaction, memory write, and trade session.")

    base_run_dir = Path(default_run_dir)
    available_runs = discover_run_directories(base_run_dir)
    auto_refresh = st.sidebar.toggle("Auto refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 15, 2)
    st.sidebar.caption(f"Base directory: `{base_run_dir}`")
    st.sidebar.button("Refresh now", use_container_width=True)

    if not base_run_dir.exists():
        st.warning(f"Base run directory does not exist yet: `{base_run_dir}`")
        _maybe_rerun(st, auto_refresh=auto_refresh, refresh_interval=refresh_interval)
        return
    if not available_runs:
        st.info("No run artifacts found yet under the selected base directory.")
        _maybe_rerun(st, auto_refresh=auto_refresh, refresh_interval=refresh_interval)
        return

    selected_run_name = st.sidebar.selectbox(
        "Run",
        options=[path.name for path in available_runs],
        index=0,
    )
    run_path = next(path for path in available_runs if path.name == selected_run_name)

    snapshot = load_dashboard_snapshot(run_path)
    if not snapshot.metadata:
        st.info("Waiting for metadata.json. Start a game and this page will populate live.")
        _maybe_rerun(st, auto_refresh=auto_refresh, refresh_interval=refresh_interval)
        return

    tab_replay, tab_analysis = st.tabs(["Game Replay", "Post-Game Analysis"])
    with tab_replay:
        cursor = _render_cursor_controls(st, snapshot)
        _render_board_summary(st, snapshot, cursor=cursor)
        _render_header(st, snapshot, cursor=cursor)
        _render_current_turn_view(st, snapshot, cursor=cursor)
    with tab_analysis:
        _render_analysis_tab(st, snapshot)
    _maybe_rerun(st, auto_refresh=auto_refresh, refresh_interval=refresh_interval)


def _render_cursor_controls(st, snapshot: DashboardSnapshot) -> int:
    cursor_key = f"history_cursor::{snapshot.run_dir}"
    max_history_index = max(0, snapshot.max_history_index)
    if cursor_key not in st.session_state:
        st.session_state[cursor_key] = max_history_index
    st.session_state[cursor_key] = min(max_history_index, max(0, st.session_state[cursor_key]))

    turn_markers = _turn_markers(snapshot)
    current_cursor = int(st.session_state[cursor_key])
    current_turn = _turn_index_for_cursor(current_cursor, turn_markers)

    st.subheader("Timeline")
    info_col, prev_turn_col, prev_event_col, next_event_col, next_turn_col, live_col = st.columns(
        (2.2, 1, 1, 1, 1, 1)
    )
    with info_col:
        st.caption(
            f"Turn {current_turn} · history {current_cursor}/{max_history_index}"
        )
    with prev_turn_col:
        if st.button("Previous turn", use_container_width=True, key=f"prev-turn::{snapshot.run_dir}"):
            st.session_state[cursor_key] = _jump_history_to_previous_turn(
                current_cursor,
                turn_markers,
            )
    with prev_event_col:
        if st.button("Previous event", use_container_width=True, key=f"prev-event::{snapshot.run_dir}"):
            st.session_state[cursor_key] = max(0, current_cursor - 1)
    with next_event_col:
        if st.button("Next event", use_container_width=True, key=f"next-event::{snapshot.run_dir}"):
            st.session_state[cursor_key] = min(max_history_index, current_cursor + 1)
    with next_turn_col:
        if st.button("Next turn", use_container_width=True, key=f"next-turn::{snapshot.run_dir}"):
            st.session_state[cursor_key] = _jump_history_to_next_turn(
                current_cursor,
                turn_markers,
                max_history_index=max_history_index,
            )
    with live_col:
        if st.button("Jump to latest", use_container_width=True, key=f"jump-live::{snapshot.run_dir}"):
            st.session_state[cursor_key] = max_history_index

    cursor = st.slider(
        "History cursor",
        min_value=0,
        max_value=max_history_index,
        key=cursor_key,
    )
    return int(cursor)


def _render_header(st, snapshot: DashboardSnapshot, *, cursor: int) -> None:
    result = snapshot.result or {}
    current_state = _latest_state_at_or_before(snapshot.public_state_snapshots, cursor)
    current_turn = {}
    if current_state is not None and isinstance(current_state.public_state.get("turn"), dict):
        current_turn = current_state.public_state["turn"]
    current_player = current_turn.get("turn_player_id") or current_turn.get("current_player_id") or "-"

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Run", snapshot.run_dir.name)
    col2.metric("Status", _status_label(snapshot))
    col3.metric("Cursor", cursor)
    col4.metric("Current player", str(current_player))
    col5.metric("Public events", len([event for event in snapshot.public_events if event.history_index <= cursor]))
    if result.get("winner_ids"):
        st.success("Winner: " + ", ".join(str(player_id) for player_id in result["winner_ids"]))


def _render_board_summary(st, snapshot: DashboardSnapshot, *, cursor: int) -> None:
    selected = _latest_state_at_or_before(snapshot.public_state_snapshots, cursor)
    if selected is None:
        st.caption("No public state snapshots yet.")
        return
    public_state = selected.public_state
    turn_state = public_state.get("turn") if isinstance(public_state.get("turn"), dict) else {}
    players = public_state.get("players") if isinstance(public_state.get("players"), dict) else {}
    board = public_state.get("board") if isinstance(public_state.get("board"), dict) else {}
    trade_state = (
        public_state.get("trade_state") if isinstance(public_state.get("trade_state"), dict) else {}
    )

    st.subheader("Board")
    top_col, side_col = st.columns((2.2, 1))
    with top_col:
        st.markdown("**Board visualization**")
        st.markdown(
            build_board_svg(board, height=520),
            unsafe_allow_html=True,
        )
        with st.expander("State details", expanded=False):
            turn_col, trade_col = st.columns(2)
            with turn_col:
                st.markdown("**Turn state**")
                st.json(turn_state, expanded=False)
            with trade_col:
                st.markdown("**Trade state**")
                st.json(trade_state, expanded=False)
        with st.expander("Board details", expanded=False):
            st.json(board, expanded=False)
    with side_col:
        st.markdown("**Players**")
        if players:
            _render_player_summary_table(st, players)
        else:
            st.caption("No public player summary yet.")
        _render_turn_event_digest(st, snapshot, cursor=cursor)


def _render_player_summary_table(st, players: dict) -> None:
    rows = []
    for player_id, summary in players.items():
        if not isinstance(summary, dict):
            continue
        accent, _, _ = _palette_for_player(player_id)
        vp = summary.get("visible_victory_points") or summary.get("vp", 0)
        res = summary.get("resource_card_count") or summary.get("res_cards", 0)
        dev = summary.get("development_card_count") or summary.get("dev_cards", 0)
        roads = summary.get("longest_road_length", "-")
        army = "🏆" if summary.get("has_largest_army") else "—"
        flags = []
        if summary.get("has_longest_road"):
            flags.append("🏆 Road")
        if summary.get("has_largest_army"):
            flags.append("⚔️ Army")
        rows.append(
            f"<tr>"
            f"<td><span style='color:{accent};font-weight:600'>{player_id}</span></td>"
            f"<td style='text-align:center'>{vp}</td>"
            f"<td style='text-align:center'>{res}</td>"
            f"<td style='text-align:center'>{dev}</td>"
            f"<td style='text-align:center'>{roads}</td>"
            f"<td style='text-align:center'>{army}</td>"
            f"<td>{' '.join(flags)}</td>"
            f"</tr>"
        )
    if rows:
        table_html = (
            "<table style='width:100%;border-collapse:collapse;font-size:0.82rem'>"
            "<thead><tr style='border-bottom:1px solid #e5e7eb'>"
            "<th style='text-align:left'>Player</th>"
            "<th>VP</th><th>Res</th><th>Dev</th><th>Road</th><th>Army</th><th></th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
        st.markdown(table_html, unsafe_allow_html=True)


def _render_turn_event_digest(
    st,
    snapshot: DashboardSnapshot,
    *,
    cursor: int,
    max_turns: int = 12,
) -> None:
    st.markdown("**Turn events**")
    sections = _recent_turn_event_sections(snapshot, cursor, max_turns=max_turns)
    if not sections:
        st.caption("No public turn events yet.")
        return

    current_turn = _turn_index_for_cursor(cursor, _turn_markers(snapshot))
    html_parts = ["<div class='turn-events-panel'>"]
    for turn_index, label, rows in sections:
        group_class = "turn-events-group turn-events-group-current" if turn_index == current_turn else "turn-events-group"
        html_parts.append(f"<section class='{group_class}'>")
        html_parts.append(
            "<div class='turn-events-divider'>"
            f"<span>{_escape_html(label)}</span>"
            "</div>"
        )
        for speaker, body in rows:
            accent, _, _ = _palette_for_player(None if speaker == "SYSTEM" else speaker)
            html_parts.append(
                "<div class='turn-event-row'>"
                f"<div class='turn-event-player' style='color:{accent}'>{_escape_html(speaker)}</div>"
                f"<div class='turn-event-text'>{_escape_html(body)}</div>"
                "</div>"
            )
        html_parts.append("</section>")
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def _recent_turn_event_sections(
    snapshot: DashboardSnapshot,
    cursor: int,
    *,
    max_turns: int = 12,
) -> tuple[tuple[int, str, tuple[tuple[str, str], ...]], ...]:
    grouped: dict[int, list[tuple[Event, str]]] = {}
    for event in snapshot.public_events:
        if event.history_index > cursor:
            break
        summary = _turn_event_digest_body(event)
        if summary is None:
            continue
        grouped.setdefault(event.turn_index, []).append((event, summary))

    grouped_items = list(grouped.items())
    if max_turns > 0:
        grouped_items = grouped_items[-max_turns:]

    sections: list[tuple[int, str, tuple[tuple[str, str], ...]]] = []
    for turn_index, rows in grouped_items:
        turn_events = tuple(event for event, _summary in rows)
        sections.append(
            (
                turn_index,
                _turn_event_section_label(turn_index, turn_events),
                tuple((str(event.actor_player_id or "SYSTEM"), summary) for event, summary in rows),
            )
        )
    return tuple(sections)


def _turn_event_section_label(turn_index: int, events: tuple[Event, ...]) -> str:
    first_phase = next((event.phase for event in events if event.phase), "")
    if first_phase.startswith("build_initial"):
        return f"Setup {turn_index}"
    return f"Turn {turn_index}"


def _turn_event_digest_body(event: Event) -> str | None:
    if event.kind not in TURN_DIGEST_EVENT_KINDS:
        return None

    payload = event.payload
    if event.kind == "dice_rolled":
        return _turn_event_dice_summary(payload.get("result"))
    if event.kind == "settlement_built":
        return f"settlement on node {payload.get('node_id')}"
    if event.kind == "city_built":
        return f"city on node {payload.get('node_id')}"
    if event.kind == "road_built":
        return f"road on {payload.get('edge')}"
    if event.kind == "robber_moved":
        coordinate = payload.get("coordinate")
        victim = payload.get("victim")
        if victim:
            return f"robber → {coordinate}, stole from {victim}"
        return f"robber → {coordinate}"
    if event.kind == "trade_chat_opened":
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return f"opened trade chat: {message.strip()}"
        requested_resources = payload.get("requested_resources")
        requested = _resource_map(requested_resources)
        if requested != "nothing":
            return f"opened trade chat for {requested}"
        return "opened trade chat"
    if event.kind == "trade_confirmed":
        counterparty = payload.get("accepting_player_id") or "?"
        return f"↔ {counterparty}: {_resource_map(payload.get('offer'))} for {_resource_map(payload.get('request'))}"
    if event.kind == "trade_chat_no_deal":
        return "trade chat ended with no deal"
    if event.kind == "trade_cancelled":
        return "trade cancelled"
    if event.kind == "development_card_played":
        description = _turn_event_action_description(payload)
        if description is not None:
            return description
        card_type = payload.get("card_type") or payload.get("dev_card_type")
        if isinstance(card_type, str) and card_type:
            return f"played {card_type.lower().replace('_', ' ')}"
        return "played dev card"
    if event.kind == "action_taken":
        description = _turn_event_action_description(payload)
        return description or "took an action"
    if event.kind == "turn_ended":
        return "End the current turn."
    return None


def _turn_event_action_description(payload: dict[str, JsonValue]) -> str | None:
    action = payload.get("action")
    if not isinstance(action, dict):
        return None
    description = action.get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()
    action_type = action.get("action_type")
    if isinstance(action_type, str) and action_type:
        return action_type.lower().replace("_", " ")
    return None


def _turn_event_dice_summary(result: object) -> str:
    if isinstance(result, list) and result:
        try:
            values = [int(value) for value in result]
        except (TypeError, ValueError):
            return f"rolled {result}"
        return f"rolled {sum(values)} ({'+'.join(str(value) for value in values)})"
    if isinstance(result, dict):
        total = result.get("total")
        values = result.get("values") or result.get("dice") or result.get("rolls")
        if isinstance(total, int) and isinstance(values, list) and values:
            return f"rolled {total} ({'+'.join(str(value) for value in values)})"
        if isinstance(total, int):
            return f"rolled {total}"
    if result is not None:
        return f"rolled {result}"
    return "rolled dice"


_TRADE_TRACE_STAGES = frozenset(
    {
        "trade_chat_open",
        "trade_chat_reply",
        "trade_chat_owner_decision",
        "trade_chat_select",
        "trade_chat_no_deal",
    }
)


def _render_current_turn_view(st, snapshot: DashboardSnapshot, *, cursor: int) -> None:
    """Show all interactions for the current turn only."""
    current_state = _latest_state_at_or_before(snapshot.public_state_snapshots, cursor)
    if current_state is None:
        st.caption("No turn data yet.")
        return

    turn_index = current_state.turn_index
    turn_dict = current_state.public_state.get("turn") if isinstance(current_state.public_state.get("turn"), dict) else {}
    acting_player_id = str(turn_dict.get("turn_player_id") or turn_dict.get("current_player_id") or "")

    accent, background, _ = _palette_for_player(acting_player_id or None)
    first_phase = current_state.phase or ""
    label_prefix = "Setup" if first_phase.startswith("build_initial") else "Turn"
    st.markdown(
        "<div class='player-column-header' "
        f"style='border-left:6px solid {accent}; background:{background};'>"
        f"<strong>{TURN_EMOJI} {label_prefix} {turn_index}</strong>&nbsp;"
        + (
            f"<span style='color:{accent};font-weight:700'>{acting_player_id}</span>&nbsp;is acting"
            if acting_player_id
            else ""
        )
        + f"&nbsp;<span style='font-size:0.8rem;color:#6b7280'>{first_phase}</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    # --- collect data for this turn -----------------------------------------
    traces_for_turn: list[PromptTrace] = []
    for player_id in snapshot.player_ids:
        for trace in snapshot.prompt_traces_by_player.get(player_id, ()):
            if trace.turn_index == turn_index and trace.history_index <= cursor:
                traces_for_turn.append(trace)
    traces_for_turn.sort(key=lambda t: (t.history_index, t.decision_index, STAGE_ORDER.get(t.stage, 99)))

    # memories keyed by (player_id, decision_index, stage) → latest snapshot
    memories_for_turn: dict[tuple[str, int, str], MemorySnapshot] = {}
    for player_id in snapshot.player_ids:
        for mem in snapshot.memory_traces_by_player.get(player_id, ()):
            if mem.turn_index == turn_index and mem.history_index <= cursor:
                key = (mem.player_id, mem.decision_index, mem.stage)
                existing = memories_for_turn.get(key)
                if existing is None or mem.history_index >= existing.history_index:
                    memories_for_turn[key] = mem

    turn_events = sorted(
        (e for e in snapshot.public_events if e.turn_index == turn_index and e.history_index <= cursor),
        key=lambda e: e.history_index,
    )

    trade_artifacts = _build_trade_artifacts(acting_player_id, tuple(turn_events)) if acting_player_id else ()
    trade_hi_set = {e.history_index for ta in trade_artifacts for e in ta.events}
    non_trade_events = [e for e in turn_events if e.history_index not in trade_hi_set]

    # assign trade-related traces to their artifact by history range
    ta_ranges = [(ta.start_history_index, ta.end_history_index, ta) for ta in trade_artifacts]
    trade_traces_by_ta: dict[int, list[PromptTrace]] = {id(ta): [] for ta in trade_artifacts}
    regular_traces: list[PromptTrace] = []
    for trace in traces_for_turn:
        if trace.stage in _TRADE_TRACE_STAGES:
            assigned = False
            for start_hi, end_hi, ta in ta_ranges:
                if start_hi <= trace.history_index <= end_hi:
                    trade_traces_by_ta[id(ta)].append(trace)
                    assigned = True
                    break
            if not assigned:
                regular_traces.append(trace)
        else:
            regular_traces.append(trace)

    # --- build chronological feed --------------------------------------------
    feed: list[tuple[int, str, object]] = []
    for trace in regular_traces:
        feed.append((trace.history_index, "trace", trace))
    for event in non_trade_events:
        feed.append((event.history_index, "event", event))
    for ta in trade_artifacts:
        feed.append((ta.start_history_index, "trade", (ta, trade_traces_by_ta[id(ta)])))

    if not feed:
        st.caption("No interactions recorded yet for this turn.")
        return

    # Events and trades sort before traces at the same history_index so that board
    # changes (dice, buildings, turn-end) always appear before the LLM interaction
    # that follows them at the same snapshot point.
    _FEED_ITEM_ORDER = {"event": 0, "trade": 0, "trace": 1}
    feed.sort(key=lambda x: (x[0], _FEED_ITEM_ORDER.get(x[1], 1)))

    for _, item_type, data in feed:
        if item_type == "trace":
            trace = data  # type: ignore[assignment]
            mem_key = (trace.player_id, trace.decision_index, trace.stage)
            _render_llm_interaction_card(st, trace, memory=memories_for_turn.get(mem_key))
        elif item_type == "event":
            _render_event_card(st, data)  # type: ignore[arg-type]
        else:
            ta, ta_traces = data  # type: ignore[misc]
            _render_trade_chat_box(st, ta, traces=ta_traces, memories_for_turn=memories_for_turn)


def _render_json_or_text(st, value: object, *, expanded: bool) -> None:
    """Render a memory value: plain strings as code blocks, everything else as JSON."""
    if value is None:
        st.caption("_(empty)_")
    elif isinstance(value, str):
        st.code(value, language=None)
    else:
        st.json(value, expanded=expanded)


def _system_prompt_message_summary(content: object) -> str:
    if isinstance(content, str):
        lines = [line for line in content.splitlines() if line.strip()]
        if not lines:
            return "Static system prompt"
        return f"Static system prompt ({len(lines)} lines)"
    return "Static system prompt"


def _render_prompt_message(st, role: str, content: object) -> None:
    role_bg = {"user": "#dbeafe", "assistant": "#dcfce7", "system": "#fef3c7"}.get(role, "#f3f4f6")
    st.markdown(
        f"<div style='background:{role_bg};border-radius:0.4rem;"
        f"padding:0.3rem 0.6rem;margin-bottom:0.25rem;"
        f"font-size:0.72rem;font-weight:700;text-transform:uppercase;color:#374151'>"
        f"{_escape_html(role)}</div>",
        unsafe_allow_html=True,
    )
    if role == "system":
        with st.expander(_system_prompt_message_summary(content), expanded=False):
            if isinstance(content, str):
                st.code(content, language=None)
            else:
                st.json(content, expanded=False)
        return
    if isinstance(content, str):
        st.code(content, language=None)
    else:
        st.json(content, expanded=False)


def _render_llm_interaction_card(
    st,
    trace: PromptTrace,
    *,
    memory: MemorySnapshot | None,
) -> None:
    """Render a single LLM call as a query + answer interaction."""
    accent, background, border = _palette_for_player(trace.player_id)
    stage_label = trace.stage.replace("_", " ").title()
    decision_label = f"· decision {trace.decision_index}" if trace.decision_index is not None else ""
    st.markdown(
        "<div class='timeline-card' "
        f"style='border-left:6px solid {accent}; background:{background}; border-color:{border};'>"
        f"<div class='timeline-title'>{PROMPT_EMOJI} {_escape_html(stage_label)}"
        f"<span style='font-weight:400;font-size:0.8rem;color:#6b7280'> {_escape_html(decision_label)}</span></div>"
        f"<div class='timeline-footer'>"
        f"history {trace.history_index} · turn {trace.turn_index} · "
        f"phase {trace.phase} · model {_escape_html(trace.model)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Query (prompt messages)", expanded=False):
        for attempt_idx, attempt in enumerate(trace.attempts, start=1):
            if len(trace.attempts) > 1:
                st.caption(f"Attempt {attempt_idx}")
            for message in attempt.messages:
                role = str(message.get("role", "unknown"))
                content = message.get("content")
                _render_prompt_message(st, role, content)

    for attempt_idx, attempt in enumerate(trace.attempts, start=1):
        answer_label = "Answer" if len(trace.attempts) == 1 else f"Answer (attempt {attempt_idx})"
        with st.expander(answer_label, expanded=True):
            if attempt.response_text is not None:
                st.code(attempt.response_text, language="json")
            else:
                st.json(attempt.response, expanded=True)

    if memory is not None:
        with st.expander(f"{MEMORY_EMOJI} Memory written — {memory.stage}", expanded=False):
            st.caption(f"history {memory.history_index} · decision {memory.decision_index}")
            col_lt, col_st = st.columns(2)
            with col_lt:
                st.markdown("`long_term`")
                _render_json_or_text(st, memory.memory.long_term, expanded=True)
            with col_st:
                st.markdown("`short_term`")
                _render_json_or_text(st, memory.memory.short_term, expanded=True)


def _render_trade_chat_box(
    st,
    artifact: TradeArtifact,
    *,
    traces: list[PromptTrace],
    memories_for_turn: dict[tuple[str, int, str], MemorySnapshot],
) -> None:
    """Render a trade session as an expandable chat room."""
    _render_trade_transcript(
        st,
        artifact,
        traces=traces,
        memories_for_turn=memories_for_turn,
        expanded=True,
    )


def _render_trade_transcript(
    st,
    artifact: TradeArtifact,
    *,
    traces: list[PromptTrace],
    memories_for_turn: dict[tuple[str, int, str], MemorySnapshot],
    expanded: bool,
) -> None:
    counterparties = sorted(
        {cp for e in artifact.events for cp in _trade_counterparties(e, artifact.owner_player_id)}
    )
    participants = [artifact.owner_player_id] + counterparties
    title = f"{TRADE_EMOJI} Trade Room"
    if participants:
        title += " — " + " · ".join(participants)

    with st.expander(title, expanded=expanded):
        st.caption(
            f"history {artifact.start_history_index}–{artifact.end_history_index} "
            f"· turn {artifact.turn_index} · owner {artifact.owner_player_id}"
        )

        feed: list[tuple[int, str, object]] = []
        for event in artifact.events:
            feed.append((event.history_index, "event", event))
        for trace in traces:
            feed.append((trace.history_index, "trace", trace))
        feed.sort(key=lambda item: (item[0], 0 if item[1] == "event" else 1))

        for _, item_type, data in feed:
            if item_type == "event":
                _render_trade_chat_bubble(st, data)  # type: ignore[arg-type]
                continue
            trace = data  # type: ignore[assignment]
            mem_key = (trace.player_id, trace.decision_index, trace.stage)
            _render_llm_interaction_card(st, trace, memory=memories_for_turn.get(mem_key))


def _render_trade_chat_bubble(st, event: Event) -> None:
    """Render a single trade event as a player-colored trade bubble."""
    speaker = event.payload.get("speaker_player_id") or event.actor_player_id or "SYSTEM"
    speaker_str = str(speaker)
    alignment = _trade_bubble_alignment(event, speaker_player_id=speaker_str)
    accent, background, _ = _palette_for_player(
        speaker_str if alignment != "center" and speaker_str != "SYSTEM" else None
    )
    emoji = _emoji_for_event(event.kind)
    kind_label = _trade_bubble_kind_label(event.kind)
    body = _event_body(event)
    metadata_bits = [f"h{event.history_index}"]
    attempt_index = event.payload.get("attempt_index")
    round_index = event.payload.get("round_index")
    proposal_id = event.payload.get("proposal_id") or event.payload.get("selected_proposal_id")
    if isinstance(attempt_index, int):
        metadata_bits.insert(0, f"attempt {attempt_index}")
    if isinstance(round_index, int) and round_index > 0:
        metadata_bits.insert(1 if isinstance(attempt_index, int) else 0, f"round {round_index}")
    if isinstance(proposal_id, str):
        metadata_bits.append(proposal_id)
    justify_content = {
        "left": "flex-start",
        "right": "flex-end",
        "center": "center",
    }[alignment]
    st.markdown(
        "<div class='trade-chat-row' "
        f"style='justify-content:{justify_content};'>"
        "<div class='trade-chat-bubble' "
        f"style='border-color:{accent}; background:{background};'>"
        "<div class='trade-chat-header'>"
        f"<span class='trade-chat-speaker' style='color:{accent}'>{emoji} "
        f"<strong>{_escape_html(speaker_str)}</strong></span>"
        f"<span class='trade-chat-kind'>{_escape_html(kind_label)}</span>"
        "</div>"
        f"<div class='trade-chat-body'>{_escape_html(body)}</div>"
        f"<div class='trade-chat-meta'>{_escape_html(' · '.join(metadata_bits))}</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def _trade_bubble_alignment(event: Event, *, speaker_player_id: str) -> str:
    if event.kind in {"trade_chat_opened", "trade_chat_closed", "trade_chat_no_deal"}:
        return "center"
    owner_player_id = _trade_owner_player_id(event)
    if owner_player_id is not None and speaker_player_id == owner_player_id:
        return "right"
    if speaker_player_id == "SYSTEM":
        return "center"
    return "left"


def _trade_bubble_kind_label(kind: str) -> str:
    return {
        "trade_chat_opened": "opened the room",
        "trade_chat_message": "said",
        "trade_chat_quote_selected": "picked a proposal",
        "trade_chat_no_deal": "called no deal",
        "trade_chat_closed": "closed the room",
        "trade_offered": "posted a direct offer",
        "trade_accepted": "accepted",
        "trade_rejected": "passed",
        "trade_confirmed": "confirmed the trade",
        "trade_cancelled": "cancelled the trade",
    }.get(kind, kind.replace("_", " "))


def _render_player_columns(
    st,
    snapshot: DashboardSnapshot,
    *,
    timelines: dict[str, tuple[TurnArtifact, ...]],
    cursor: int,
) -> None:
    if not snapshot.player_ids:
        st.caption("No players discovered yet.")
        return
    selected_state = _latest_state_at_or_before(snapshot.public_state_snapshots, cursor)
    current_turn_player = None
    if selected_state is not None:
        turn_state = selected_state.public_state.get("turn")
        if isinstance(turn_state, dict):
            value = turn_state.get("turn_player_id") or turn_state.get("current_player_id")
            if isinstance(value, str):
                current_turn_player = value

    columns = st.columns(len(snapshot.player_ids))
    for column, player_id in zip(columns, snapshot.player_ids):
        with column:
            _render_player_column(
                st,
                snapshot,
                player_id=player_id,
                timeline=timelines.get(player_id, ()),
                cursor=cursor,
                is_current_player=(player_id == current_turn_player),
            )


def _render_player_column(
    st,
    snapshot: DashboardSnapshot,
    *,
    player_id: str,
    timeline: tuple[TurnArtifact, ...],
    cursor: int,
    is_current_player: bool,
) -> None:
    _render_column_header(st, snapshot, player_id=player_id, cursor=cursor, is_current_player=is_current_player)
    if not timeline:
        st.caption("No timeline entries yet.")
        return
    for artifact in timeline:
        _render_turn_artifact(
            st,
            artifact,
            snapshot=snapshot,
            cursor=cursor,
            expanded=is_current_player and artifact.end_history_index >= cursor >= artifact.start_history_index,
        )


def _render_column_header(
    st,
    snapshot: DashboardSnapshot,
    *,
    player_id: str,
    cursor: int,
    is_current_player: bool,
) -> None:
    accent, background, border = _palette_for_player(player_id)
    st.markdown(
        (
            "<div class='player-column-header' "
            f"style='border-left: 6px solid {accent}; background: {background};'>"
            f"<strong>{player_id}</strong>"
            + (" <span class='player-column-badge'>active</span>" if is_current_player else "")
            + "</div>"
        ),
        unsafe_allow_html=True,
    )
    latest_memory = _latest_at_or_before(
        snapshot.memory_traces_by_player.get(player_id, ()),
        cursor,
        key=lambda item: item.history_index,
    )
    latest_trace = _latest_at_or_before(
        snapshot.prompt_traces_by_player.get(player_id, ()),
        cursor,
        key=lambda item: item.history_index,
    )
    with st.expander("Player Memory", expanded=False):
        if latest_memory is None:
            st.caption("No memory snapshot yet.")
        else:
            st.caption(
                f"memory stage `{latest_memory.stage}` at history {latest_memory.history_index}"
            )
            st.markdown("`long_term`")
            _render_json_or_text(st, latest_memory.memory.long_term, expanded=False)
            st.markdown("`short_term`")
            _render_json_or_text(st, latest_memory.memory.short_term, expanded=False)
        if latest_trace is not None:
            st.caption(
                f"latest prompt `{latest_trace.stage}` at history {latest_trace.history_index}"
            )


def _render_turn_artifact(
    st,
    artifact: TurnArtifact,
    *,
    snapshot: DashboardSnapshot,
    cursor: int,
    expanded: bool,
) -> None:
    label = _turn_artifact_label(artifact)
    with st.expander(label, expanded=expanded):
        st.caption(
            f"history {artifact.start_history_index} to {artifact.end_history_index} · "
            f"turn {artifact.turn_index} · phases {', '.join(artifact.phases) or '-'}"
        )

        trade_by_start = {trade.start_history_index: trade for trade in artifact.trade_artifacts}
        consumed_trade_histories = {
            event.history_index for trade in artifact.trade_artifacts for event in trade.events
        }
        events_by_history: dict[int, list[Event]] = {}
        for event in artifact.events:
            events_by_history.setdefault(event.history_index, []).append(event)
        memories_by_history: dict[int, list[MemorySnapshot]] = {}
        for memory in artifact.memory_snapshots:
            memories_by_history.setdefault(memory.history_index, []).append(memory)
        traces_by_history: dict[int, list[PromptTrace]] = {}
        for trace in artifact.prompt_traces:
            traces_by_history.setdefault(trace.history_index, []).append(trace)

        history_points = sorted(
            set(events_by_history) | set(memories_by_history) | set(traces_by_history)
        )
        for history_index in history_points:
            trade_artifact = trade_by_start.get(history_index)
            if trade_artifact is not None:
                _render_trade_artifact(st, trade_artifact)
            for event in events_by_history.get(history_index, ()):
                if event.history_index in consumed_trade_histories:
                    continue
                _render_event_card(st, event)
            for memory in memories_by_history.get(history_index, ()):
                _render_memory_card(st, memory, player_id=artifact.player_id)
            for trace in traces_by_history.get(history_index, ()):
                _render_prompt_trace_card(st, trace, player_id=artifact.player_id)

        final_state = _latest_state_at_or_before(snapshot.public_state_snapshots, min(cursor, artifact.end_history_index))
        if final_state is not None and final_state.turn_index == artifact.turn_index:
            with st.expander("Turn-end public snapshot", expanded=False):
                board = final_state.public_state.get("board")
                if isinstance(board, dict):
                    st.markdown(
                        build_board_svg(board, height=420),
                        unsafe_allow_html=True,
                    )
                st.json(final_state.public_state, expanded=False)


def _render_trade_artifact(st, artifact: TradeArtifact) -> None:
    _render_trade_transcript(
        st,
        artifact,
        traces=[],
        memories_for_turn={},
        expanded=False,
    )


def _render_event_card(st, event: Event) -> None:
    accent, background, border = _palette_for_event(event)
    st.markdown(
        _card_html(
            emoji=_emoji_for_event(event.kind),
            title=_event_title(event),
            body=_event_body(event),
            footer=(
                f"history {event.history_index} · turn {event.turn_index} · "
                f"phase {event.phase} · decision {event.decision_index if event.decision_index is not None else '-'}"
            ),
            accent=accent,
            background=background,
            border=border,
        ),
        unsafe_allow_html=True,
    )


def _render_memory_card(st, memory: MemorySnapshot, *, player_id: str) -> None:
    accent, background, border = _palette_for_player(player_id)
    st.markdown(
        _card_html(
            emoji=MEMORY_EMOJI,
            title=f"{memory.stage} memory write",
            body="Long-term and short-term memory snapshot.",
            footer=(
                f"history {memory.history_index} · turn {memory.turn_index} · "
                f"phase {memory.phase} · decision {memory.decision_index}"
            ),
            accent=accent,
            background=background,
            border=border,
        ),
        unsafe_allow_html=True,
    )
    st.markdown("`long_term`")
    _render_json_or_text(st, memory.memory.long_term, expanded=False)
    st.markdown("`short_term`")
    _render_json_or_text(st, memory.memory.short_term, expanded=False)


def _render_prompt_trace_card(st, trace: PromptTrace, *, player_id: str) -> None:
    accent, background, border = _palette_for_player(player_id)
    with st.expander(
        f"{PROMPT_EMOJI} {trace.stage} prompt · history {trace.history_index}",
        expanded=False,
    ):
        st.markdown(
            _card_html(
                emoji=PROMPT_EMOJI,
                title=f"{trace.stage} prompt trace",
                body=f"{len(trace.attempts)} attempt(s) · model `{trace.model}`.",
                footer=(
                    f"history {trace.history_index} · turn {trace.turn_index} · "
                    f"phase {trace.phase} · decision {trace.decision_index}"
                ),
                accent=accent,
                background=background,
                border=border,
            ),
            unsafe_allow_html=True,
        )
        for attempt_index, attempt in enumerate(trace.attempts, start=1):
            with st.expander(f"Attempt {attempt_index}", expanded=False):
                if attempt.response_text is not None:
                    st.markdown("**Response text**")
                    st.code(attempt.response_text, language="json")
                st.markdown("**Parsed response**")
                st.json(attempt.response, expanded=False)
                st.markdown("**Messages**")
                for message in attempt.messages:
                    role = str(message.get("role", "unknown"))
                    content = message.get("content")
                    _render_prompt_message(st, role, content)


def _turn_artifact_label(artifact: TurnArtifact) -> str:
    prefix = TURN_EMOJI
    first_phase = artifact.phases[0] if artifact.phases else "unknown"
    if first_phase.startswith("build_initial"):
        return f"{prefix} Setup {artifact.turn_index}"
    return f"{prefix} Turn {artifact.turn_index}"


def _build_trade_artifacts(
    player_id: str,
    events: tuple[Event, ...],
) -> tuple[TradeArtifact, ...]:
    artifacts: list[TradeArtifact] = []
    current: list[Event] = []
    for event in events:
        if event.kind not in TRADE_EVENT_KINDS or _trade_owner_player_id(event) != player_id:
            continue
        if current and _starts_new_trade_artifact(current, event):
            artifacts.append(_trade_artifact_from_events(player_id, tuple(current)))
            current = []
        current.append(event)
        if _ends_trade_artifact(event):
            artifacts.append(_trade_artifact_from_events(player_id, tuple(current)))
            current = []
    if current:
        artifacts.append(_trade_artifact_from_events(player_id, tuple(current)))
    return tuple(artifacts)


def _starts_new_trade_artifact(current: list[Event], event: Event) -> bool:
    previous = current[-1]
    previous_attempt = previous.payload.get("attempt_index")
    current_attempt = event.payload.get("attempt_index")
    if (
        previous_attempt is not None
        and current_attempt is not None
        and previous_attempt != current_attempt
    ):
        return True
    if event.kind in {"trade_chat_opened", "trade_offered"} and previous.kind not in {
        "trade_chat_opened",
        "trade_offered",
    }:
        return True
    return False


def _ends_trade_artifact(event: Event) -> bool:
    if event.kind in TRADE_CLOSE_KINDS:
        return True
    if event.kind == "trade_chat_closed":
        return event.payload.get("outcome") != "selected"
    return False


def _trade_artifact_from_events(player_id: str, events: tuple[Event, ...]) -> TradeArtifact:
    return TradeArtifact(
        owner_player_id=player_id,
        turn_index=events[0].turn_index,
        start_history_index=events[0].history_index,
        end_history_index=events[-1].history_index,
        events=events,
    )


def _event_relevant_to_player(event: Event, player_id: str) -> bool:
    if event.actor_player_id == player_id:
        return True
    for key in (
        "owner_player_id",
        "offering_player_id",
        "accepting_player_id",
        "selected_player_id",
        "speaker_player_id",
        "player_id",
        "target_player_id",
        "victim_player_id",
    ):
        if event.payload.get(key) == player_id:
            return True
    return False


def _trade_owner_player_id(event: Event) -> str | None:
    for key in ("owner_player_id", "offering_player_id"):
        value = event.payload.get(key)
        if isinstance(value, str):
            return value
    return event.actor_player_id


def _trade_counterparties(event: Event, owner_player_id: str) -> tuple[str, ...]:
    counterparties = []
    for key in ("speaker_player_id", "accepting_player_id", "selected_player_id"):
        value = event.payload.get(key)
        if isinstance(value, str) and value != owner_player_id:
            counterparties.append(value)
    return tuple(dict.fromkeys(counterparties))


def _event_title(event: Event) -> str:
    actor = event.actor_player_id or "SYSTEM"
    title = {
        "dice_rolled": "Dice rolled",
        "settlement_built": "Settlement built",
        "city_built": "City built",
        "road_built": "Road built",
        "robber_moved": "Robber moved",
        "trade_offered": "Trade offered",
        "trade_accepted": "Trade accepted",
        "trade_rejected": "Trade rejected",
        "trade_confirmed": "Trade confirmed",
        "trade_cancelled": "Trade cancelled",
        "trade_chat_opened": "Trade chat opened",
        "trade_chat_message": "Trade chat message",
        "trade_chat_quote_selected": "Trade proposal selected",
        "trade_chat_no_deal": "Trade chat ended with no deal",
        "trade_chat_closed": "Trade chat closed",
        "turn_ended": "Turn ended",
        "development_card_played": "Development card played",
        "action_taken": "Action",
    }.get(event.kind, event.kind.replace("_", " ").title())
    return f"{actor} · {title}"


def _event_body(event: Event) -> str:
    payload = event.payload
    if event.kind == "trade_offered":
        return (
            f"{payload.get('offering_player_id') or event.actor_player_id} offered "
            f"{_resource_map(payload.get('offer'))} for {_resource_map(payload.get('request'))}."
        )
    if event.kind == "trade_accepted":
        return f"{event.actor_player_id} accepted the offer."
    if event.kind == "trade_rejected":
        return f"{event.actor_player_id} rejected the offer."
    if event.kind == "trade_confirmed":
        return (
            f"{payload.get('offering_player_id')} traded {_resource_map(payload.get('offer'))} "
            f"with {payload.get('accepting_player_id')} for {_resource_map(payload.get('request'))}."
        )
    if event.kind == "trade_cancelled":
        return "The trade was cancelled."
    if event.kind == "trade_chat_opened":
        requested = _resource_map(payload.get("requested_resources"))
        message = payload.get("message")
        if isinstance(message, str) and message:
            return f"Requested {requested}. Message: {message}"
        return f"Requested {requested}."
    if event.kind == "trade_chat_message":
        message = payload.get("message")
        quote_offer = _resource_map(payload.get("offer"))
        quote_request = _resource_map(payload.get("request"))
        if payload.get("offer") and payload.get("request"):
            quote_summary = f" Offer: {quote_offer} for {quote_request}."
        else:
            quote_summary = ""
        if isinstance(message, str) and message:
            return f"{message}{quote_summary}"
        return quote_summary.strip() or "Trade chat update."
    if event.kind == "trade_chat_quote_selected":
        return (
            f"Selected {payload.get('selected_player_id')} for "
            f"{_resource_map(payload.get('offer'))} against {_resource_map(payload.get('request'))}."
        )
    if event.kind == "trade_chat_no_deal":
        message = payload.get("message")
        return str(message) if isinstance(message, str) and message else "No trade selected."
    if event.kind == "trade_chat_closed":
        outcome = payload.get("outcome")
        selected = payload.get("selected_player_id")
        if selected:
            return f"Closed with outcome `{outcome}` and selected `{selected}`."
        return f"Closed with outcome `{outcome}`."
    if event.kind == "dice_rolled":
        result = payload.get("result")
        return f"Dice result: {result}."
    if event.kind == "road_built":
        return f"Built road on {payload.get('edge')}."
    if event.kind in {"settlement_built", "city_built"}:
        return f"Built on node {payload.get('node_id')}."
    if event.kind == "robber_moved":
        return f"Moved robber to {payload.get('coordinate')}."
    return json.dumps(payload, sort_keys=True)


def _emoji_for_event(kind: str) -> str:
    return EVENT_EMOJIS.get(kind, "🧩")


def _resource_map(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "nothing"
    return ", ".join(f"{amount} {resource}" for resource, amount in sorted(value.items()))


def _palette_for_event(event: Event) -> tuple[str, str, str]:
    if event.actor_player_id is None:
        return NEUTRAL_COLORS
    return _palette_for_player(event.actor_player_id)


def _palette_for_player(player_id: str | None) -> tuple[str, str, str]:
    if player_id is None:
        return NEUTRAL_COLORS
    return PLAYER_COLORS.get(player_id, NEUTRAL_COLORS)


def _card_html(
    *,
    emoji: str,
    title: str,
    body: str,
    footer: str,
    accent: str,
    background: str,
    border: str,
) -> str:
    safe_body = _escape_html(body)
    safe_title = _escape_html(title)
    safe_footer = _escape_html(footer)
    return (
        "<div class='timeline-card' "
        f"style='border-left: 6px solid {accent}; background: {background}; border-color: {border};'>"
        f"<div class='timeline-title'>{emoji} {safe_title}</div>"
        f"<div class='timeline-body'>{safe_body}</div>"
        f"<div class='timeline-footer'>{safe_footer}</div>"
        "</div>"
    )


def build_board_svg(board: Mapping[str, JsonValue], *, height: int = 520) -> str:
    tiles = board.get("tiles")
    nodes = board.get("nodes")
    edges = board.get("edges")
    robber_coordinate = board.get("robber_coordinate")
    if not isinstance(tiles, list) or not isinstance(nodes, dict) or not isinstance(edges, list):
        return (
            "<div class='board-fallback'>Board visualization unavailable for this snapshot.</div>"
        )

    hex_size = 42.0
    tile_centers = {
        tuple(_as_int_tuple(tile_entry.get("coordinate"), size=3)): _cube_to_point(
            _as_int_tuple(tile_entry.get("coordinate"), size=3),
            hex_size,
        )
        for tile_entry in tiles
        if isinstance(tile_entry, dict) and _as_int_tuple(tile_entry.get("coordinate"), size=3) is not None
    }
    node_positions = _board_node_positions(nodes, tile_centers, hex_size)
    all_points = list(tile_centers.values()) + list(node_positions.values())
    if not all_points:
        return (
            "<div class='board-fallback'>Board visualization unavailable for this snapshot.</div>"
        )

    min_x = min(point[0] for point in all_points) - hex_size * 2.2
    max_x = max(point[0] for point in all_points) + hex_size * 2.2
    min_y = min(point[1] for point in all_points) - hex_size * 2.2
    max_y = max(point[1] for point in all_points) + hex_size * 2.2
    width = max_x - min_x
    view_box = f"{min_x:.1f} {min_y:.1f} {width:.1f} {(max_y - min_y):.1f}"

    edge_fragments = [
        _edge_svg_fragment(edge, node_positions=node_positions, tile_centers=tile_centers, hex_size=hex_size)
        for edge in edges
        if isinstance(edge, dict)
    ]
    tile_fragments = [
        _tile_svg_fragment(
            tile_entry,
            center=tile_centers.get(_as_int_tuple(tile_entry.get("coordinate"), size=3)),
            hex_size=hex_size,
            robber_coordinate=robber_coordinate,
        )
        for tile_entry in tiles
        if isinstance(tile_entry, dict)
    ]
    node_fragments = [
        _node_svg_fragment(node, node_positions)
        for node in nodes.values()
        if isinstance(node, dict)
    ]

    return (
        "<div class='board-shell'>"
        f"<svg class='board-svg' viewBox='{view_box}' style='height:{height}px'>"
        + "".join(tile_fragments)
        + "".join(edge_fragments)
        + "".join(node_fragments)
        + "</svg></div>"
    )


def _board_node_positions(
    nodes: Mapping[str, JsonValue],
    tile_centers: dict[tuple[int, int, int], tuple[float, float]],
    hex_size: float,
) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    for value in nodes.values():
        if not isinstance(value, dict):
            continue
        node_id = value.get("id")
        if not isinstance(node_id, int):
            continue
        tile_coordinate = _as_int_tuple(value.get("tile_coordinate"), size=3)
        if tile_coordinate is None:
            continue
        center = tile_centers.get(tile_coordinate)
        if center is None:
            continue
        direction = value.get("direction")
        if not isinstance(direction, str):
            continue
        positions[node_id] = _point_for_node_direction(center, direction, hex_size)
    return positions


def _cube_to_point(cube: tuple[int, int, int] | None, hex_size: float) -> tuple[float, float]:
    if cube is None:
        return (0.0, 0.0)
    q, _, r = cube
    x = hex_size * math.sqrt(3) * (q + r / 2)
    y = hex_size * 1.5 * r
    return (x, y)


def _hex_points(center: tuple[float, float], hex_size: float) -> str:
    cx, cy = center
    points = []
    for angle in (-90, -30, 30, 90, 150, 210):
        radians = math.radians(angle)
        points.append(
            f"{cx + hex_size * math.cos(radians):.1f},{cy + hex_size * math.sin(radians):.1f}"
        )
    return " ".join(points)


def _point_for_node_direction(
    center: tuple[float, float],
    direction: str,
    hex_size: float,
) -> tuple[float, float]:
    angle = NODE_DIRECTION_ANGLES.get(direction, -90)
    radians = math.radians(angle)
    radius = hex_size * 0.98
    return (
        center[0] + radius * math.cos(radians),
        center[1] + radius * math.sin(radians),
    )


def _tile_svg_fragment(
    tile_entry: Mapping[str, JsonValue],
    *,
    center: tuple[float, float] | None,
    hex_size: float,
    robber_coordinate: object,
) -> str:
    if center is None:
        return ""
    tile = tile_entry.get("tile")
    coordinate = _as_int_tuple(tile_entry.get("coordinate"), size=3)
    if not isinstance(tile, dict):
        return ""
    tile_type = str(tile.get("type", "UNKNOWN"))
    fill = _tile_fill(tile)
    stroke = "#94a3b8" if tile_type in {"WATER", "PORT"} else "#475569"
    label = _tile_label(tile)
    robber = " 🥷" if coordinate is not None and list(coordinate) == robber_coordinate else ""
    number = tile.get("number")
    cx, cy = center
    label_y = cy - 6 if number is not None else cy + 4
    number_markup = ""
    if number is not None:
        text_fill = "#991b1b" if int(number) in {6, 8} else "#0f172a"
        number_markup = (
            f"<text x='{cx:.1f}' y='{cy + 10:.1f}' text-anchor='middle' "
            f"class='board-number' fill='{text_fill}'>{int(number)}</text>"
        )
    return (
        f"<polygon points='{_hex_points(center, hex_size)}' fill='{fill}' stroke='{stroke}' "
        "stroke-width='2'/>"
        f"<text x='{cx:.1f}' y='{label_y:.1f}' text-anchor='middle' class='board-label'>"
        f"{_escape_html(label + robber)}</text>"
        + number_markup
    )


def _tile_fill(tile: Mapping[str, JsonValue]) -> str:
    tile_type = str(tile.get("type", "UNKNOWN"))
    if tile_type == "RESOURCE_TILE":
        return RESOURCE_TILE_COLORS.get(str(tile.get("resource")), "#e5e7eb")
    if tile_type == "DESERT":
        return "#e9d8a6"
    if tile_type == "PORT":
        return "#bfdbfe"
    if tile_type == "WATER":
        return "#dbeafe"
    return "#e5e7eb"


def _tile_label(tile: Mapping[str, JsonValue]) -> str:
    tile_type = str(tile.get("type", "UNKNOWN"))
    resource = tile.get("resource")
    if tile_type == "RESOURCE_TILE":
        return str(resource or "").title()[:5]
    if tile_type == "DESERT":
        return "Desert"
    if tile_type == "PORT":
        return "3:1" if resource is None else f"{str(resource).title()} port"
    if tile_type == "WATER":
        return ""
    return tile_type.title()


def _edge_svg_fragment(
    edge: Mapping[str, JsonValue],
    *,
    node_positions: dict[int, tuple[float, float]],
    tile_centers: dict[tuple[int, int, int], tuple[float, float]],
    hex_size: float,
) -> str:
    edge_id = edge.get("id")
    points: tuple[tuple[float, float], tuple[float, float]] | None = None
    if isinstance(edge_id, list) and len(edge_id) == 2 and all(isinstance(item, int) for item in edge_id):
        first = node_positions.get(edge_id[0])
        second = node_positions.get(edge_id[1])
        if first is not None and second is not None:
            points = (first, second)
    if points is None:
        tile_coordinate = _as_int_tuple(edge.get("tile_coordinate"), size=3)
        direction = edge.get("direction")
        if tile_coordinate is None or not isinstance(direction, str):
            return ""
        center = tile_centers.get(tile_coordinate)
        if center is None:
            return ""
        node_pair = EDGE_DIRECTION_VERTICES.get(direction)
        if node_pair is None:
            return ""
        points = (
            _point_for_node_direction(center, node_pair[0], hex_size),
            _point_for_node_direction(center, node_pair[1], hex_size),
        )

    color = edge.get("color")
    if not isinstance(color, str) or color not in PLAYER_COLORS:
        return (
            f"<line x1='{points[0][0]:.1f}' y1='{points[0][1]:.1f}' "
            f"x2='{points[1][0]:.1f}' y2='{points[1][1]:.1f}' "
            "stroke='#cbd5e1' stroke-width='4' stroke-linecap='round' />"
        )
    accent, _, _ = _palette_for_player(color)
    stroke = PLAYER_STROKES.get(color, accent)
    return (
        f"<line x1='{points[0][0]:.1f}' y1='{points[0][1]:.1f}' "
        f"x2='{points[1][0]:.1f}' y2='{points[1][1]:.1f}' "
        f"stroke='{accent}' stroke-width='8' stroke-linecap='round' />"
        f"<line x1='{points[0][0]:.1f}' y1='{points[0][1]:.1f}' "
        f"x2='{points[1][0]:.1f}' y2='{points[1][1]:.1f}' "
        f"stroke='{stroke}' stroke-width='3' stroke-linecap='round' />"
    )


def _node_svg_fragment(
    node: Mapping[str, JsonValue],
    node_positions: dict[int, tuple[float, float]],
) -> str:
    node_id = node.get("id")
    if not isinstance(node_id, int):
        return ""
    point = node_positions.get(node_id)
    if point is None:
        return ""
    building = node.get("building")
    color = node.get("color")
    cx, cy = point
    if not isinstance(building, str) or not isinstance(color, str) or color not in PLAYER_COLORS:
        return (
            f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='3.2' fill='#94a3b8' opacity='0.55' />"
        )
    accent, _, _ = _palette_for_player(color)
    stroke = PLAYER_STROKES.get(color, accent)
    if building == "CITY":
        size = 8.5
        points = [
            (cx - size, cy + size * 0.65),
            (cx - size, cy - size * 0.15),
            (cx, cy - size),
            (cx + size, cy - size * 0.15),
            (cx + size, cy + size * 0.65),
        ]
        point_string = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        return (
            f"<polygon points='{point_string}' fill='{accent}' stroke='{stroke}' "
            "stroke-width='2' />"
        )
    return (
        f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='7.2' fill='{accent}' "
        f"stroke='{stroke}' stroke-width='2' />"
    )


def _as_int_tuple(value: object, *, size: int) -> tuple[int, ...] | None:
    if not isinstance(value, list) or len(value) != size or not all(isinstance(item, int) for item in value):
        return None
    return tuple(value)


def _escape_html(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _bucket_start_history_index(bucket: _TurnBucket) -> int:
    history_points = [
        item.history_index for item in (*bucket.events, *bucket.memory_snapshots, *bucket.prompt_traces)
    ]
    return min(history_points) if history_points else 0


def _latest_state_at_or_before(
    snapshots: tuple[PublicStateSnapshot, ...],
    cursor: int,
) -> PublicStateSnapshot | None:
    return _latest_at_or_before(snapshots, cursor, key=lambda item: item.history_index)


def _latest_at_or_before(items, cursor: int, *, key):
    latest = None
    for item in items:
        if key(item) <= cursor:
            latest = item
        else:
            break
    return latest


def _status_label(snapshot: DashboardSnapshot) -> str:
    if snapshot.result is not None:
        metadata = snapshot.result.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("status"), str):
            return metadata["status"]
        return "finished"
    return "in_progress"


def _player_ids_from_run(
    run_path: Path,
    metadata: dict[str, JsonValue],
) -> tuple[str, ...]:
    metadata_player_ids = metadata.get("player_ids")
    if isinstance(metadata_player_ids, list):
        return tuple(str(player_id) for player_id in metadata_player_ids)
    players_dir = run_path / "players"
    if not players_dir.exists():
        return ()
    return tuple(sorted(path.name for path in players_dir.iterdir() if path.is_dir()))


def _read_json_if_exists(path: Path) -> dict[str, JsonValue] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, JsonValue]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _is_run_directory(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(
        (path / file_name).exists()
        for file_name in ("metadata.json", "public_history.jsonl", "public_state_trace.jsonl")
    )


def _run_sort_key(path: Path) -> float:
    return max(
        [
            path.stat().st_mtime,
            *((path / name).stat().st_mtime for name in ("result.json", "public_history.jsonl", "metadata.json") if (path / name).exists()),
        ]
    )


def _turn_markers(snapshot: DashboardSnapshot) -> tuple[tuple[int, int], ...]:
    max_history_by_turn: dict[int, int] = {0: 0}
    for item in snapshot.public_state_snapshots:
        max_history_by_turn[item.turn_index] = max(
            max_history_by_turn.get(item.turn_index, 0),
            item.history_index,
        )
    for event in snapshot.public_events:
        max_history_by_turn[event.turn_index] = max(
            max_history_by_turn.get(event.turn_index, 0),
            event.history_index,
        )
    return tuple(sorted(max_history_by_turn.items()))


def _turn_index_for_cursor(cursor: int, turn_markers: tuple[tuple[int, int], ...]) -> int:
    current_turn = 0
    for turn_index, history_index in turn_markers:
        if history_index <= cursor:
            current_turn = turn_index
        else:
            break
    return current_turn


def _jump_history_to_previous_turn(
    cursor: int,
    turn_markers: tuple[tuple[int, int], ...],
) -> int:
    current_turn = _turn_index_for_cursor(cursor, turn_markers)
    previous_history = 0
    for turn_index, history_index in turn_markers:
        if turn_index >= current_turn:
            break
        previous_history = history_index
    return previous_history


def _jump_history_to_next_turn(
    cursor: int,
    turn_markers: tuple[tuple[int, int], ...],
    *,
    max_history_index: int,
) -> int:
    current_turn = _turn_index_for_cursor(cursor, turn_markers)
    for turn_index, history_index in turn_markers:
        if turn_index > current_turn:
            return history_index
    return max_history_index


def _render_analysis_tab(st, snapshot: DashboardSnapshot) -> None:
    """Render the post-game analysis tab."""
    analysis_path = snapshot.run_dir / "analysis.json"
    analysis_data = _read_json_if_exists(analysis_path)

    if analysis_data is None:
        if snapshot.result is None:
            st.info("Game is still in progress. Analysis will be available once the game ends.")
            return
        if st.button("Generate Analysis", type="primary"):
            from .analysis import analyze_game
            analysis_data = analyze_game(snapshot.run_dir)
            st.rerun()
        else:
            st.info("No analysis.json found. Click the button above to generate it.")
            return

    gs = analysis_data.get("game_summary", {})
    players = analysis_data.get("players", {})

    # ── Game summary metrics ──
    st.subheader("Game Summary")
    cols = st.columns(5)
    cols[0].metric("Winner", ", ".join(gs.get("winner_ids", [])) or "—")
    cols[1].metric("Turns", gs.get("num_turns", "?"))
    cols[2].metric("Total Decisions", gs.get("total_decisions", "?"))
    cols[3].metric("Trade Activity", f"{gs.get('trade_activity_rate', 0):.1%}")
    cols[4].metric("Trade Efficiency", f"{gs.get('trade_efficiency', 0):.1%}")

    # ── VP Progression Chart ──
    st.subheader("Victory Point Progression")
    vp_chart_data: dict[str, dict[int, int]] = {}
    max_turn = 0
    for pid, pdata in players.items():
        vp_prog = pdata.get("vp_progression", [])
        vp_chart_data[pid] = {entry["turn_index"]: entry["vp"] for entry in vp_prog}
        if vp_prog:
            max_turn = max(max_turn, max(e["turn_index"] for e in vp_prog))

    if vp_chart_data and max_turn > 0:
        chart_rows = []
        for turn in range(max_turn + 1):
            row: dict[str, int | float] = {"Turn": turn}
            for pid in players:
                series = vp_chart_data.get(pid, {})
                # Forward fill: use last known VP
                vp = 0
                for t in range(turn + 1):
                    if t in series:
                        vp = series[t]
                row[pid] = vp
            chart_rows.append(row)
        st.line_chart(chart_rows, x="Turn", y=list(players.keys()))

    # ── Resource Production ──
    st.subheader("Estimated Resource Production")
    res_rows = []
    for pid, pdata in players.items():
        production = pdata.get("resource_production", {}).get("total", {})
        row: dict[str, Any] = {"Player": pid}
        for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"):
            row[r] = production.get(r, 0)
        row["Total"] = sum(row[r] for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"))
        res_rows.append(row)
    if res_rows:
        st.bar_chart(res_rows, x="Player", y=["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"])

    # ── Trade Activity ──
    st.subheader("Trade Activity")
    trade_rows = []
    for pid, pdata in players.items():
        trade = pdata.get("trade", {})
        trade_rows.append({
            "Player": pid,
            "Offers Made": trade.get("offers_made", 0),
            "Acceptances": trade.get("acceptances", 0),
            "Rejections": trade.get("rejections", 0),
            "Completed": trade.get("confirmations_as_offerer", 0) + trade.get("confirmations_as_acceptee", 0),
        })
    if trade_rows:
        st.bar_chart(trade_rows, x="Player", y=["Offers Made", "Acceptances", "Rejections", "Completed"])

    # ── Building Timeline ──
    st.subheader("Building Timeline")
    building_rows = []
    for pid, pdata in players.items():
        buildings = pdata.get("buildings", {})
        for s in buildings.get("settlements", []):
            building_rows.append({"Player": pid, "Turn": s.get("turn_index"), "Type": "Settlement", "Location": str(s.get("node_id"))})
        for c in buildings.get("cities", []):
            building_rows.append({"Player": pid, "Turn": c.get("turn_index"), "Type": "City", "Location": str(c.get("node_id"))})
        for r in buildings.get("roads", []):
            building_rows.append({"Player": pid, "Turn": r.get("turn_index"), "Type": "Road", "Location": str(r.get("edge"))})
    if building_rows:
        building_rows.sort(key=lambda r: (r["Turn"], r["Player"]))
        st.dataframe(building_rows, use_container_width=True)

    # ── Per-Player Summary Cards ──
    st.subheader("Player Summary")
    player_cols = st.columns(len(players) or 1)
    for idx, (pid, pdata) in enumerate(players.items()):
        with player_cols[idx]:
            color = PLAYER_COLORS.get(pid, NEUTRAL_COLORS)
            st.markdown(f"**:{pid.lower()}[{pid}]** {'🏆' if pdata.get('is_winner') else ''}")
            st.metric("Final VP", pdata.get("final_vp", "?"))

            counts = pdata.get("buildings", {}).get("counts", {})
            st.caption(f"Buildings: {counts.get('settlements', 0)}S / {counts.get('cities', 0)}C / {counts.get('roads', 0)}R")

            achievements = pdata.get("achievements", {})
            if achievements.get("has_longest_road"):
                st.caption("Longest Road")
            if achievements.get("has_largest_army"):
                st.caption("Largest Army")

            robber = pdata.get("robber", {})
            st.caption(f"Robber: moved {robber.get('times_moved_robber', 0)}x, targeted {robber.get('times_targeted', 0)}x")

            dev = pdata.get("dev_cards", {})
            if dev.get("cards_played", 0) or dev.get("cards_held_at_end", 0):
                st.caption(f"Dev cards: {dev.get('cards_played', 0)} played, {dev.get('cards_held_at_end', 0)} held")

            dq = pdata.get("decision_quality", {})
            if dq.get("total_prompts", 0):
                st.caption(f"Retries: {dq.get('retries', 0)}/{dq.get('total_prompts', 0)} ({dq.get('retry_rate', 0):.1%})")

            phase = pdata.get("phase_analysis", {}).get("opening", {})
            if phase.get("pip_count", 0):
                st.caption(f"Opening: {phase.get('resource_diversity', 0)} types, {phase.get('pip_count', 0)} pips")


def _maybe_rerun(st, *, auto_refresh: bool, refresh_interval: int) -> None:
    if not auto_refresh:
        return
    time.sleep(refresh_interval)
    st.rerun()


def _inject_styles(st) -> None:
    st.markdown(
        """
        <style>
        .timeline-card {
          padding: 0.7rem 0.85rem;
          border-radius: 0.75rem;
          border: 1px solid;
          margin-bottom: 0.7rem;
        }
        .timeline-title {
          font-weight: 700;
          margin-bottom: 0.2rem;
        }
        .timeline-body {
          font-size: 0.95rem;
          line-height: 1.35;
          margin-bottom: 0.35rem;
          white-space: pre-wrap;
        }
        .timeline-footer {
          font-size: 0.75rem;
          color: #6b7280;
        }
        .player-column-header {
          border-radius: 0.75rem;
          padding: 0.7rem 0.8rem;
          margin-bottom: 0.8rem;
        }
        .player-column-badge {
          font-size: 0.72rem;
          text-transform: uppercase;
          color: #475569;
          background: rgba(255,255,255,0.75);
          border-radius: 999px;
          padding: 0.18rem 0.5rem;
          margin-left: 0.4rem;
        }
        .board-shell {
          width: 100%;
          overflow-x: auto;
          background: linear-gradient(180deg, #fffefb 0%, #f8fafc 100%);
          border: 1px solid #e5e7eb;
          border-radius: 1rem;
          padding: 0.6rem;
          margin-bottom: 1rem;
        }
        .board-svg {
          width: 100%;
          min-width: 620px;
        }
        .board-label {
          font-size: 7px;
          fill: #0f172a;
          font-weight: 700;
          letter-spacing: 0.02em;
        }
        .board-number {
          font-size: 13px;
          font-weight: 800;
        }
        .board-fallback {
          border: 1px dashed #cbd5e1;
          border-radius: 0.75rem;
          padding: 0.9rem;
          color: #64748b;
          background: #f8fafc;
        }
        .trade-chat-row {
          display: flex;
          width: 100%;
          margin-bottom: 0.55rem;
        }
        .trade-chat-bubble {
          width: min(100%, 48rem);
          max-width: 86%;
          border-radius: 1rem;
          border: 1px solid #e5e7eb;
          padding: 0.65rem 0.85rem;
          box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
        }
        .trade-chat-header {
          display: flex;
          flex-wrap: wrap;
          align-items: baseline;
          gap: 0.45rem;
          margin-bottom: 0.2rem;
        }
        .trade-chat-speaker {
          font-size: 0.84rem;
          font-weight: 700;
        }
        .trade-chat-kind {
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          color: #64748b;
        }
        .trade-chat-body {
          font-size: 0.93rem;
          line-height: 1.4;
          white-space: pre-wrap;
          word-break: break-word;
        }
        .trade-chat-meta {
          margin-top: 0.35rem;
          font-size: 0.72rem;
          color: #94a3b8;
        }
        .turn-events-panel {
          margin-top: 0.75rem;
          max-height: 36rem;
          overflow-y: auto;
          padding-right: 0.2rem;
        }
        .turn-events-group {
          margin-bottom: 0.85rem;
        }
        .turn-events-group-current {
          background: rgba(148, 163, 184, 0.08);
          border-radius: 0.9rem;
          padding: 0.5rem 0.55rem 0.35rem;
        }
        .turn-events-divider {
          display: flex;
          align-items: center;
          gap: 0.65rem;
          margin: 0.05rem 0 0.55rem;
          color: #64748b;
          font-size: 0.74rem;
          font-weight: 700;
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }
        .turn-events-divider::before,
        .turn-events-divider::after {
          content: "";
          flex: 1;
          height: 1px;
          background: rgba(148, 163, 184, 0.65);
        }
        .turn-event-row {
          display: grid;
          grid-template-columns: max-content 1fr;
          gap: 0.7rem;
          align-items: baseline;
          margin-bottom: 0.35rem;
          font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
          font-size: 0.9rem;
          line-height: 1.35;
        }
        .turn-event-player {
          font-weight: 700;
          min-width: 4.6rem;
        }
        .turn-event-text {
          color: #0f172a;
          word-break: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _require_streamlit():
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised manually.
        raise RuntimeError(
            "Streamlit is not installed. Install it with `uv sync --group dashboard`."
        ) from exc
    return st


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the Streamlit dashboard for a catan-bench run directory."
    )
    parser.add_argument(
        "--run-dir",
        default="runs/0.3.0/dev",
        help="Run directory to monitor live.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
