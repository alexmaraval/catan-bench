from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import altair as alt
import pandas as pd
import plotly.graph_objects as go

from .schemas import Event, JsonValue, MemorySnapshot, PromptTrace, PublicStateSnapshot

PLAYER_COLORS: dict[str, tuple[str, str, str]] = {
    "RED": ("#dc2626", "#fee2e2", "#dc2626"),
    "BLUE": ("#2563eb", "#dbeafe", "#2563eb"),
    "ORANGE": ("#ea580c", "#ffedd5", "#ea580c"),
    "WHITE": ("#ffffff", "#cbd5e1", "#ffffff"),
}
PLAYER_NAME_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(player_id) for player_id in PLAYER_COLORS) + r")\b"
)
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
    "resources_discarded": "🧺",
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
            return max(
                snapshot.history_index for snapshot in self.public_state_snapshots
            )
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
        Event.from_dict(entry)
        for entry in _read_jsonl(run_path / "public_history.jsonl")
    )
    public_state_snapshots = tuple(
        PublicStateSnapshot.from_dict(entry)
        for entry in _read_jsonl(run_path / "public_state_trace.jsonl")
    )
    memory_traces_by_player = {
        player_id: tuple(
            MemorySnapshot.from_dict(entry)
            for entry in _read_jsonl(
                run_path / "players" / player_id / "memory_trace.jsonl"
            )
        )
        for player_id in player_ids
    }
    prompt_traces_by_player = {
        player_id: tuple(
            PromptTrace.from_dict(entry)
            for entry in _read_jsonl(
                run_path / "players" / player_id / "prompt_trace.jsonl"
            )
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
    candidates = [
        path
        for path in base_path.iterdir()
        if path.is_dir() and _is_run_directory(path)
    ]
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
            bucket_by_turn.setdefault(event.turn_index, _TurnBucket()).events.append(
                event
            )
        for snapshot_entry in snapshot.memory_traces_by_player.get(player_id, ()):
            if snapshot_entry.history_index > cursor:
                break
            bucket_by_turn.setdefault(
                snapshot_entry.turn_index, _TurnBucket()
            ).memory_snapshots.append(snapshot_entry)
        for trace in snapshot.prompt_traces_by_player.get(player_id, ()):
            if trace.history_index > cursor:
                break
            bucket_by_turn.setdefault(
                trace.turn_index, _TurnBucket()
            ).prompt_traces.append(trace)

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
            sorted_events = tuple(
                sorted(bucket.events, key=lambda item: item.history_index)
            )
            sorted_memories = tuple(
                sorted(
                    bucket.memory_snapshots,
                    key=lambda item: (
                        item.history_index,
                        item.decision_index,
                        STAGE_ORDER.get(item.stage, 99),
                    ),
                )
            )
            sorted_traces = tuple(
                sorted(
                    bucket.prompt_traces,
                    key=lambda item: (
                        item.history_index,
                        item.decision_index,
                        STAGE_ORDER.get(item.stage, 99),
                    ),
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
                item.history_index
                for item in (*sorted_events, *sorted_memories, *sorted_traces)
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
    st.caption(
        "Turn-focused view: navigate turns to inspect every LLM interaction, memory write, and trade session."
    )

    base_run_dir = Path(default_run_dir)
    available_runs = discover_run_directories(base_run_dir)
    auto_refresh = st.sidebar.toggle("Auto refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 15, 2)
    st.sidebar.caption(f"Base directory: `{base_run_dir}`")
    st.sidebar.button("Refresh now", width="stretch")

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
        st.info(
            "Waiting for metadata.json. Start a game and this page will populate live."
        )
        _maybe_rerun(st, auto_refresh=auto_refresh, refresh_interval=refresh_interval)
        return

    tab_replay, tab_analysis, tab_benchmark = st.tabs(
        ["Game Replay", "Post-Game Analysis", "Benchmark"]
    )
    with tab_replay:
        cursor = _render_cursor_controls(st, snapshot)
        _render_board_summary(st, snapshot, cursor=cursor)
        _render_header(st, snapshot, cursor=cursor)
        _render_current_turn_view(st, snapshot, cursor=cursor)
    with tab_analysis:
        _render_analysis_tab(st, snapshot)
    with tab_benchmark:
        from .benchmark_dashboard import render_benchmark_tab

        render_benchmark_tab(st, base_run_dir=base_run_dir)
    _maybe_rerun(st, auto_refresh=auto_refresh, refresh_interval=refresh_interval)


def _render_cursor_controls(st, snapshot: DashboardSnapshot) -> int:
    cursor_key = f"history_cursor::{snapshot.run_dir}"
    max_history_index = max(0, snapshot.max_history_index)
    if cursor_key not in st.session_state:
        st.session_state[cursor_key] = max_history_index
    st.session_state[cursor_key] = min(
        max_history_index, max(0, st.session_state[cursor_key])
    )

    turn_markers = _turn_markers(snapshot)
    current_cursor = int(st.session_state[cursor_key])
    current_turn = _turn_index_for_cursor(current_cursor, turn_markers)

    st.subheader("Timeline")
    info_col, prev_turn_col, prev_event_col, next_event_col, next_turn_col, live_col = (
        st.columns((2.2, 1, 1, 1, 1, 1))
    )
    with info_col:
        st.caption(
            f"Turn {current_turn} · history {current_cursor}/{max_history_index}"
        )
    with prev_turn_col:
        if st.button(
            "Previous turn", width="stretch", key=f"prev-turn::{snapshot.run_dir}"
        ):
            st.session_state[cursor_key] = _jump_history_to_previous_turn(
                current_cursor,
                turn_markers,
            )
    with prev_event_col:
        if st.button(
            "Previous event", width="stretch", key=f"prev-event::{snapshot.run_dir}"
        ):
            st.session_state[cursor_key] = max(0, current_cursor - 1)
    with next_event_col:
        if st.button(
            "Next event", width="stretch", key=f"next-event::{snapshot.run_dir}"
        ):
            st.session_state[cursor_key] = min(max_history_index, current_cursor + 1)
    with next_turn_col:
        if st.button(
            "Next turn", width="stretch", key=f"next-turn::{snapshot.run_dir}"
        ):
            st.session_state[cursor_key] = _jump_history_to_next_turn(
                current_cursor,
                turn_markers,
                max_history_index=max_history_index,
            )
    with live_col:
        if st.button(
            "Jump to latest", width="stretch", key=f"jump-live::{snapshot.run_dir}"
        ):
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
    if current_state is not None and isinstance(
        current_state.public_state.get("turn"), dict
    ):
        current_turn = current_state.public_state["turn"]
    current_player = (
        current_turn.get("turn_player_id")
        or current_turn.get("current_player_id")
        or "-"
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Run", snapshot.run_dir.name)
    col2.metric("Status", _status_label(snapshot))
    col3.metric("Cursor", cursor)
    col4.metric("Current player", str(current_player))
    col5.metric(
        "Public events",
        len(
            [event for event in snapshot.public_events if event.history_index <= cursor]
        ),
    )
    if result.get("winner_ids"):
        st.success(
            "Winner: " + ", ".join(str(player_id) for player_id in result["winner_ids"])
        )


def _render_board_summary(st, snapshot: DashboardSnapshot, *, cursor: int) -> None:
    selected = _latest_state_at_or_before(snapshot.public_state_snapshots, cursor)
    if selected is None:
        st.caption("No public state snapshots yet.")
        return
    public_state = selected.public_state
    turn_state = (
        public_state.get("turn") if isinstance(public_state.get("turn"), dict) else {}
    )
    players = (
        public_state.get("players")
        if isinstance(public_state.get("players"), dict)
        else {}
    )
    board = (
        public_state.get("board") if isinstance(public_state.get("board"), dict) else {}
    )
    trade_state = (
        public_state.get("trade_state")
        if isinstance(public_state.get("trade_state"), dict)
        else {}
    )

    st.subheader("Board")
    top_col, side_col = st.columns((2.2, 1))
    with top_col:
        st.markdown("**Board visualization**")
        st.markdown(
            build_board_svg(board, height=520, players=players),
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
            winner_ids = (
                snapshot.result.get("winner_ids", [])
                if isinstance(snapshot.result, dict)
                else []
            )
            _render_player_summary_table(st, players, winner_ids=winner_ids)
        else:
            st.caption("No public player summary yet.")
        _render_turn_event_digest(st, snapshot, cursor=cursor)


def _render_player_summary_table(
    st, players: dict, *, winner_ids: list[str] | tuple[str, ...] | None = None
) -> None:
    rows = []
    winner_set = {str(player_id) for player_id in (winner_ids or [])}
    for player_id, summary in players.items():
        if not isinstance(summary, dict):
            continue
        accent, _, _ = _palette_for_player(player_id)
        visible_vp = int(summary.get("visible_victory_points") or summary.get("vp", 0))
        dev_vp = summary.get("dev_victory_points", 0)
        if not isinstance(dev_vp, int):
            dev_vp = 0
        has_road = bool(summary.get("has_longest_road"))
        has_army = bool(summary.get("has_largest_army"))
        # Board VP = visible VP minus road/army bonuses (each worth 2)
        board_vp = visible_vp - (2 if has_road else 0) - (2 if has_army else 0)
        total_vp = visible_vp + dev_vp
        road_len = summary.get("longest_road_length", "-")
        road_cell = f"{road_len} 🛤️" if has_road else str(road_len)
        army_count = summary.get("played_knights", 0)
        if not isinstance(army_count, int):
            army_count = 0
        army_cell = f"{army_count} ⚔️" if has_army else str(army_count)
        rows.append(
            (player_id, accent, board_vp, dev_vp, road_cell, army_cell, total_vp)
        )

    if rows:
        html_rows = []
        for pid, accent, board_vp, dev_vp, road_cell, army_cell, total_vp in rows:
            trophy = " 🏆" if pid in winner_set else ""
            html_rows.append(
                f"<tr>"
                f"<td><span style='color:{accent};font-weight:600'>{pid}</span></td>"
                f"<td style='text-align:center'>{board_vp}</td>"
                f"<td style='text-align:center'>{dev_vp}</td>"
                f"<td style='text-align:center'>{road_cell}</td>"
                f"<td style='text-align:center'>{army_cell}</td>"
                f"<td style='text-align:center;font-weight:700'>{total_vp}{trophy}</td>"
                f"</tr>"
            )
        table_html = (
            "<table style='width:100%;border-collapse:collapse;font-size:0.82rem'>"
            "<thead><tr style='border-bottom:1px solid #e5e7eb'>"
            "<th style='text-align:left'>Player</th>"
            "<th>Board VP</th><th>Dev VP</th><th>L. Road</th><th>L. Army</th><th>Total VP</th>"
            "</tr></thead><tbody>" + "".join(html_rows) + "</tbody></table>"
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
        group_class = (
            "turn-events-group turn-events-group-current"
            if turn_index == current_turn
            else "turn-events-group"
        )
        html_parts.append(f"<section class='{group_class}'>")
        html_parts.append(
            f"<div class='turn-events-divider'><span>{_escape_html(label)}</span></div>"
        )
        for speaker, body in rows:
            accent, _, _ = _palette_for_player(None if speaker == "SYSTEM" else speaker)
            html_parts.append(
                "<div class='turn-event-row'>"
                f"<div class='turn-event-player' style='color:{accent}'>{_escape_html(speaker)}</div>"
                f"<div class='turn-event-text'>{_colorize_player_mentions_html(body)}</div>"
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
                tuple(
                    (str(event.actor_player_id or "SYSTEM"), summary)
                    for event, summary in rows
                ),
            )
        )
    return tuple(sections)


def _turn_event_section_label(turn_index: int, events: tuple[Event, ...]) -> str:
    first_phase = next((event.phase for event in events if event.phase), "")
    if first_phase.startswith("build_initial"):
        return f"Setup {turn_index}"
    return f"Turn {turn_index}"


def _infer_turn_owner_player_id(
    snapshot: DashboardSnapshot,
    *,
    turn_index: int,
    cursor: int,
    current_state: PublicStateSnapshot | None,
    turn_events: tuple[Event, ...] = (),
) -> str:
    for event in turn_events:
        if isinstance(event.actor_player_id, str) and event.actor_player_id:
            return event.actor_player_id

    candidate_traces: list[PromptTrace] = []
    for player_id in snapshot.player_ids:
        for trace in snapshot.prompt_traces_by_player.get(player_id, ()):
            if trace.turn_index == turn_index and trace.history_index <= cursor:
                candidate_traces.append(trace)
    candidate_traces.sort(
        key=lambda trace: (
            trace.history_index,
            trace.decision_index,
            STAGE_ORDER.get(trace.stage, 99),
        )
    )
    for trace in candidate_traces:
        if trace.player_id:
            return trace.player_id

    if current_state is None:
        return ""
    turn_dict = (
        current_state.public_state.get("turn")
        if isinstance(current_state.public_state.get("turn"), dict)
        else {}
    )
    owner = turn_dict.get("turn_player_id") or turn_dict.get("current_player_id")
    return str(owner) if isinstance(owner, str) else ""


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
    turn_events = sorted(
        (
            e
            for e in snapshot.public_events
            if e.turn_index == turn_index and e.history_index <= cursor
        ),
        key=lambda e: e.history_index,
    )
    turn_owner_id = _infer_turn_owner_player_id(
        snapshot,
        turn_index=turn_index,
        cursor=cursor,
        current_state=current_state,
        turn_events=tuple(turn_events),
    )

    accent, background, _ = _palette_for_player(turn_owner_id or None)
    first_phase = current_state.phase or ""
    label_prefix = "Setup" if first_phase.startswith("build_initial") else "Turn"
    st.markdown(
        "<div class='player-column-header' "
        f"style='border-left:6px solid {accent}; background:{background};'>"
        f"<strong>{TURN_EMOJI} {label_prefix} {turn_index}</strong>&nbsp;"
        + (
            f"<span style='color:{accent};font-weight:700'>{turn_owner_id}</span>&nbsp;is acting"
            if turn_owner_id
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
    traces_for_turn.sort(
        key=lambda t: (t.history_index, t.decision_index, STAGE_ORDER.get(t.stage, 99))
    )

    # memories keyed by (player_id, decision_index, stage) → latest snapshot
    memories_for_turn: dict[tuple[str, int, str], MemorySnapshot] = {}
    for player_id in snapshot.player_ids:
        for mem in snapshot.memory_traces_by_player.get(player_id, ()):
            if mem.turn_index == turn_index and mem.history_index <= cursor:
                key = (mem.player_id, mem.decision_index, mem.stage)
                existing = memories_for_turn.get(key)
                if existing is None or mem.history_index >= existing.history_index:
                    memories_for_turn[key] = mem

    trade_artifacts = (
        _build_trade_artifacts(turn_owner_id, tuple(turn_events))
        if turn_owner_id
        else ()
    )
    trade_hi_set = {e.history_index for ta in trade_artifacts for e in ta.events}
    non_trade_events = [e for e in turn_events if e.history_index not in trade_hi_set]

    # assign trade-related traces to their artifact by history range
    ta_ranges = [
        (ta.start_history_index, ta.end_history_index, ta) for ta in trade_artifacts
    ]
    trade_traces_by_ta: dict[int, list[PromptTrace]] = {
        id(ta): [] for ta in trade_artifacts
    }
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
            _render_llm_interaction_card(
                st, trace, memory=memories_for_turn.get(mem_key)
            )
        elif item_type == "event":
            _render_event_card(st, data)  # type: ignore[arg-type]
        else:
            ta, ta_traces = data  # type: ignore[misc]
            _render_trade_chat_box(
                st, ta, traces=ta_traces, memories_for_turn=memories_for_turn
            )


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
    role_bg = {"user": "#dbeafe", "assistant": "#dcfce7", "system": "#fef3c7"}.get(
        role, "#f3f4f6"
    )
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
    decision_label = (
        f"· decision {trace.decision_index}" if trace.decision_index is not None else ""
    )
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
        answer_label = (
            "Answer" if len(trace.attempts) == 1 else f"Answer (attempt {attempt_idx})"
        )
        with st.expander(answer_label, expanded=True):
            if attempt.response_text is not None:
                st.code(attempt.response_text, language="json")
            else:
                st.json(attempt.response, expanded=True)

    if memory is not None:
        with st.expander(
            f"{MEMORY_EMOJI} Memory written — {memory.stage}", expanded=False
        ):
            st.caption(
                f"history {memory.history_index} · decision {memory.decision_index}"
            )
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
        {
            cp
            for e in artifact.events
            for cp in _trade_counterparties(e, artifact.owner_player_id)
        }
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
            _render_llm_interaction_card(
                st, trace, memory=memories_for_turn.get(mem_key)
            )


def _render_trade_chat_bubble(st, event: Event) -> None:
    """Render a single trade event as a player-colored trade bubble."""
    speaker = (
        event.payload.get("speaker_player_id") or event.actor_player_id or "SYSTEM"
    )
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
    proposal_id = event.payload.get("proposal_id") or event.payload.get(
        "selected_proposal_id"
    )
    if isinstance(attempt_index, int):
        metadata_bits.insert(0, f"attempt {attempt_index}")
    if isinstance(round_index, int) and round_index > 0:
        metadata_bits.insert(
            1 if isinstance(attempt_index, int) else 0, f"round {round_index}"
        )
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
            value = turn_state.get("turn_player_id") or turn_state.get(
                "current_player_id"
            )
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
    _render_column_header(
        st,
        snapshot,
        player_id=player_id,
        cursor=cursor,
        is_current_player=is_current_player,
    )
    if not timeline:
        st.caption("No timeline entries yet.")
        return
    for artifact in timeline:
        _render_turn_artifact(
            st,
            artifact,
            snapshot=snapshot,
            cursor=cursor,
            expanded=is_current_player
            and artifact.end_history_index >= cursor >= artifact.start_history_index,
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
            + (
                " <span class='player-column-badge'>active</span>"
                if is_current_player
                else ""
            )
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

        trade_by_start = {
            trade.start_history_index: trade for trade in artifact.trade_artifacts
        }
        consumed_trade_histories = {
            event.history_index
            for trade in artifact.trade_artifacts
            for event in trade.events
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

        final_state = _latest_state_at_or_before(
            snapshot.public_state_snapshots, min(cursor, artifact.end_history_index)
        )
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
        if (
            event.kind not in TRADE_EVENT_KINDS
            or _trade_owner_player_id(event) != player_id
        ):
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


def _trade_artifact_from_events(
    player_id: str, events: tuple[Event, ...]
) -> TradeArtifact:
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
        "resources_discarded": "Resources discarded",
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
        return (
            str(message)
            if isinstance(message, str) and message
            else "No trade selected."
        )
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
    if event.kind == "resources_discarded":
        count = payload.get("discarded_count", "?")
        noun = "resource" if count == 1 else "resources"
        return f"Discarded {count} {noun} for the robber."
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
    return ", ".join(
        f"{amount} {resource}" for resource, amount in sorted(value.items())
    )


def _palette_for_event(event: Event) -> tuple[str, str, str]:
    if event.actor_player_id is None:
        return NEUTRAL_COLORS
    return _palette_for_player(event.actor_player_id)


def _palette_for_player(player_id: str | None) -> tuple[str, str, str]:
    if player_id is None:
        return NEUTRAL_COLORS
    return PLAYER_COLORS.get(player_id, NEUTRAL_COLORS)


def _colorize_player_mentions_html(text: str) -> str:
    html_parts: list[str] = []
    cursor = 0
    for match in PLAYER_NAME_PATTERN.finditer(text):
        start, end = match.span()
        if start > cursor:
            html_parts.append(_escape_html(text[cursor:start]))
        player_id = match.group(0)
        accent, _, _ = _palette_for_player(player_id)
        html_parts.append(
            f"<span style='color:{accent}'>{_escape_html(player_id)}</span>"
        )
        cursor = end
    if cursor < len(text):
        html_parts.append(_escape_html(text[cursor:]))
    return "".join(html_parts)


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


def _player_panel_svg(
    player_id: str,
    summary: dict,
    roads_built: int,
    settlements_built: int,
    cities_built: int,
    px: float,
    py: float,
) -> str:
    """Render a player hand/piece panel as an SVG <g> element at (px, py)."""
    # Resource colors matching the tile palette
    _RES_COLORS: dict[str, tuple[str, str]] = {
        "WOOD": ("#7fb069", "#4a7a40"),
        "BRICK": ("#d97757", "#a0472e"),
        "SHEEP": ("#a7d676", "#6a9e40"),
        "WHEAT": ("#f1d36b", "#c4a020"),
        "ORE": ("#9ca3af", "#5a6270"),
    }
    _RES_ORDER = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")

    W, H = 140, 96
    accent, _, _ = _palette_for_player(player_id)
    resource_hand: dict[str, int] = summary.get("resource_hand") or {}  # type: ignore[assignment]
    res = int(summary.get("resource_card_count") or sum(resource_hand.values()))
    dev = int(summary.get("development_card_count") or 0)
    vp = int(summary.get("visible_victory_points") or 0)
    roads_left = max(0, 15 - roads_built)
    sets_left = max(0, 5 - settlements_built)
    cits_left = max(0, 4 - cities_built)
    has_road = bool(summary.get("has_longest_road"))
    has_army = bool(summary.get("has_largest_army"))

    # Grid constants — label sits on the LEFT, cards start after it
    CW, CH, CG = 8, 11, 2  # card width / height / gap
    LBL_W = 28  # pixels reserved for "RES" / "DEV" text
    CARD_X0 = px + 8 + LBL_W  # first card x
    MAX_CARDS = int((W - 8 - LBL_W - 8) // (CW + CG))  # ≈ 9

    # Row y: top of card rects (label baseline = card bottom)
    RES_TOP = py + 24
    DEV_TOP = py + 42

    def _resource_cards(card_top: float) -> str:
        """Draw one colored rect per resource card, grouped by type."""
        frags: list[str] = []
        idx = 0
        for res_name in _RES_ORDER:
            count = resource_hand.get(res_name, 0)
            fill, stroke = _RES_COLORS[res_name]
            for _ in range(count):
                if idx >= MAX_CARDS:
                    break
                x = CARD_X0 + idx * (CW + CG)
                frags.append(
                    f"<rect x='{x:.1f}' y='{card_top:.1f}' "
                    f"width='{CW}' height='{CH}' rx='1.5' "
                    f"fill='{fill}' stroke='{stroke}' stroke-width='0.7'/>"
                )
                idx += 1
            if idx >= MAX_CARDS:
                break
        overflow = res - idx
        if overflow > 0:
            ox = CARD_X0 + MAX_CARDS * (CW + CG) + 2
            frags.append(
                f"<text x='{ox:.1f}' y='{card_top + CH:.1f}' font-size='8' "
                f"fill='#e5e7eb' font-family='monospace'>+{overflow}</text>"
            )
        if res == 0:
            frags.append(
                f"<text x='{CARD_X0:.1f}' y='{card_top + CH:.1f}' font-size='8' "
                f"fill='#475569' font-family='monospace'>—</text>"
            )
        return "".join(frags)

    def _dev_cards(n: int, card_top: float) -> str:
        shown = min(n, MAX_CARDS)
        frags = []
        for i in range(shown):
            frags.append(
                f"<rect x='{CARD_X0 + i * (CW + CG):.1f}' y='{card_top:.1f}' "
                f"width='{CW}' height='{CH}' rx='1.5' "
                f"fill='#1e3a5f' stroke='#60a5fa' stroke-width='0.7'/>"
            )
        if n > MAX_CARDS:
            ox = CARD_X0 + MAX_CARDS * (CW + CG) + 2
            frags.append(
                f"<text x='{ox:.1f}' y='{card_top + CH:.1f}' font-size='8' "
                f"fill='#60a5fa' font-family='monospace'>+{n - MAX_CARDS}</text>"
            )
        if n == 0:
            frags.append(
                f"<text x='{CARD_X0:.1f}' y='{card_top + CH:.1f}' font-size='8' "
                f"fill='#475569' font-family='monospace'>—</text>"
            )
        return "".join(frags)

    # Badges inline after VP (no separate rects that collide)
    badge_suffix = (" ★R" if has_road else "") + (" ⚔A" if has_army else "")

    # ── Piece row ────────────────────────────────────────────────────────────
    PIECE_Y = py + H - 12
    road_icon = (
        f"<rect x='{px + 8:.1f}' y='{PIECE_Y - 4:.1f}' width='13' height='4.5' "
        f"rx='2' fill='{accent}'/>"
    )
    stx = px + 53
    set_icon = (
        f"<polygon points='{stx:.1f},{PIECE_Y:.1f} {stx - 6:.1f},{PIECE_Y:.1f} "
        f"{stx - 3:.1f},{PIECE_Y - 9:.1f}' fill='{accent}'/>"
    )
    ctx = px + 98
    cit_icon = (
        f"<polygon points='"
        f"{ctx - 6:.1f},{PIECE_Y:.1f} "
        f"{ctx - 6:.1f},{PIECE_Y - 5:.1f} "
        f"{ctx:.1f},{PIECE_Y - 10:.1f} "
        f"{ctx + 6:.1f},{PIECE_Y - 5:.1f} "
        f"{ctx + 6:.1f},{PIECE_Y:.1f}' fill='{accent}'/>"
    )

    return (
        f"<rect x='{px:.1f}' y='{py:.1f}' width='{W}' height='{H}' rx='6' "
        f"fill='rgba(5,10,20,0.90)' stroke='{accent}' stroke-width='1.8'/>"
        # Row 1: name + VP
        + f"<text x='{px + 8:.1f}' y='{py + 15:.1f}' font-size='12' font-weight='bold' "
        f"fill='{accent}' font-family='monospace'>{player_id}</text>"
        + f"<text x='{px + W - 8:.1f}' y='{py + 15:.1f}' font-size='11' fill='{accent}' "
        f"font-family='monospace' text-anchor='end'>{vp} VP{badge_suffix}</text>"
        # Divider 1
        + f"<line x1='{px + 5:.1f}' y1='{py + 20:.1f}' x2='{px + W - 5:.1f}' y2='{py + 20:.1f}' "
        f"stroke='{accent}' stroke-opacity='0.35' stroke-width='0.7'/>"
        # Row 2: RES label (left) + per-resource colored cards (right)
         + f"<text x='{px + 8:.1f}' y='{RES_TOP + CH:.1f}' font-size='8' "
        f"fill='#64748b' font-family='monospace'>RES</text>"
        + _resource_cards(RES_TOP)
        # Row 3: DEV label (left) + dev cards (right)
        + f"<text x='{px + 8:.1f}' y='{DEV_TOP + CH:.1f}' font-size='8' "
        f"fill='#64748b' font-family='monospace'>DEV</text>"
        + _dev_cards(dev, DEV_TOP)
        # Divider 2
        + f"<line x1='{px + 5:.1f}' y1='{py + H - 20:.1f}' x2='{px + W - 5:.1f}' y2='{py + H - 20:.1f}' "
        f"stroke='{accent}' stroke-opacity='0.25' stroke-width='0.7'/>"
        # Row 4: remaining pieces
         + road_icon + f"<text x='{px + 23:.1f}' y='{PIECE_Y + 1:.1f}' font-size='10' "
        f"fill='#94a3b8' font-family='monospace'>{roads_left}</text>"
        + set_icon
        + f"<text x='{stx + 4:.1f}' y='{PIECE_Y + 1:.1f}' font-size='10' "
        f"fill='#94a3b8' font-family='monospace'>{sets_left}</text>"
        + cit_icon
        + f"<text x='{ctx + 8:.1f}' y='{PIECE_Y + 1:.1f}' font-size='10' "
        f"fill='#94a3b8' font-family='monospace'>{cits_left}</text>"
    )


# seat_index → (corner_h, corner_v): "left"/"right", "top"/"bottom"
_SEAT_CORNER: dict[int, tuple[str, str]] = {
    0: ("left", "top"),
    1: ("right", "top"),
    2: ("right", "bottom"),
    3: ("left", "bottom"),
}


def build_board_svg(
    board: Mapping[str, JsonValue],
    *,
    height: int = 520,
    players: dict | None = None,
) -> str:
    tiles = board.get("tiles")
    nodes = board.get("nodes")
    edges = board.get("edges")
    robber_coordinate = board.get("robber_coordinate")
    if (
        not isinstance(tiles, list)
        or not isinstance(nodes, dict)
        or not isinstance(edges, list)
    ):
        return "<div class='board-fallback'>Board visualization unavailable for this snapshot.</div>"

    hex_size = 42.0
    tile_centers = {
        tuple(_as_int_tuple(tile_entry.get("coordinate"), size=3)): _cube_to_point(
            _as_int_tuple(tile_entry.get("coordinate"), size=3),
            hex_size,
        )
        for tile_entry in tiles
        if isinstance(tile_entry, dict)
        and _as_int_tuple(tile_entry.get("coordinate"), size=3) is not None
    }
    node_positions = _board_node_positions(nodes, tile_centers, hex_size)
    all_points = list(tile_centers.values()) + list(node_positions.values())
    if not all_points:
        return "<div class='board-fallback'>Board visualization unavailable for this snapshot.</div>"

    min_x = min(point[0] for point in all_points) - hex_size * 2.2
    max_x = max(point[0] for point in all_points) + hex_size * 2.2
    min_y = min(point[1] for point in all_points) - hex_size * 2.2
    max_y = max(point[1] for point in all_points) + hex_size * 2.2
    width = max_x - min_x
    view_box = f"{min_x:.1f} {min_y:.1f} {width:.1f} {(max_y - min_y):.1f}"

    edge_fragments = [
        _edge_svg_fragment(
            edge,
            node_positions=node_positions,
            tile_centers=tile_centers,
            hex_size=hex_size,
        )
        for edge in edges
        if isinstance(edge, dict)
    ]
    tile_fragments = [
        _tile_svg_fragment(
            tile_entry,
            center=tile_centers.get(
                _as_int_tuple(tile_entry.get("coordinate"), size=3)
            ),
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

    bg_rect = f"<rect x='{min_x:.1f}' y='{min_y:.1f}' width='{width:.1f}' height='{(max_y - min_y):.1f}' fill='#000000'/>"

    # ── Corner player panels ────────────────────────────────────────────────
    panel_fragments: list[str] = []
    if players:
        # Count built pieces per player from board data
        roads_by_player: dict[str, int] = {}
        sets_by_player: dict[str, int] = {}
        cits_by_player: dict[str, int] = {}
        for edge in edges or []:
            if isinstance(edge, dict):
                ec = edge.get("color")
                if isinstance(ec, str):
                    roads_by_player[ec] = roads_by_player.get(ec, 0) + 1
        for node in nodes.values() if isinstance(nodes, dict) else []:
            if isinstance(node, dict):
                nc = node.get("color")
                bldg = node.get("building")
                if isinstance(nc, str) and isinstance(bldg, str):
                    if bldg == "SETTLEMENT":
                        sets_by_player[nc] = sets_by_player.get(nc, 0) + 1
                    elif bldg == "CITY":
                        cits_by_player[nc] = cits_by_player.get(nc, 0) + 1

        PW, PH, MARGIN = 140, 96, 6
        for pid, summary in players.items():
            if not isinstance(summary, dict):
                continue
            seat = int(summary.get("seat_index", 0)) % 4
            h_side, v_side = _SEAT_CORNER.get(seat, ("left", "top"))
            px = (min_x + MARGIN) if h_side == "left" else (max_x - MARGIN - PW)
            py = (min_y + MARGIN) if v_side == "top" else (max_y - MARGIN - PH)
            panel_fragments.append(
                _player_panel_svg(
                    pid,
                    summary,
                    roads_by_player.get(pid, 0),
                    sets_by_player.get(pid, 0),
                    cits_by_player.get(pid, 0),
                    px,
                    py,
                )
            )

    return (
        "<div class='board-shell'>"
        f"<svg class='board-svg' viewBox='{view_box}' style='height:{height}px'>"
        + bg_rect
        + "".join(tile_fragments)
        + "".join(edge_fragments)
        + "".join(node_fragments)
        + "".join(panel_fragments)
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


def _cube_to_point(
    cube: tuple[int, int, int] | None, hex_size: float
) -> tuple[float, float]:
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
    robber = (
        " 🥷"
        if coordinate is not None and list(coordinate) == robber_coordinate
        else ""
    )
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
        f"{_escape_html(label + robber)}</text>" + number_markup
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
    if (
        isinstance(edge_id, list)
        and len(edge_id) == 2
        and all(isinstance(item, int) for item in edge_id)
    ):
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
    if (
        not isinstance(building, str)
        or not isinstance(color, str)
        or color not in PLAYER_COLORS
    ):
        return f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='3.2' fill='#94a3b8' opacity='0.55' />"
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
    if (
        not isinstance(value, list)
        or len(value) != size
        or not all(isinstance(item, int) for item in value)
    ):
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
        item.history_index
        for item in (*bucket.events, *bucket.memory_snapshots, *bucket.prompt_traces)
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
        for file_name in (
            "metadata.json",
            "public_history.jsonl",
            "public_state_trace.jsonl",
        )
    )


def _run_sort_key(path: Path) -> float:
    return max(
        [
            path.stat().st_mtime,
            *(
                (path / name).stat().st_mtime
                for name in ("result.json", "public_history.jsonl", "metadata.json")
                if (path / name).exists()
            ),
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


def _turn_index_for_cursor(
    cursor: int, turn_markers: tuple[tuple[int, int], ...]
) -> int:
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
            st.info(
                "Game is still in progress. Analysis will be available once the game ends."
            )
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
    _chat_rooms_total = sum(
        p.get("trade_chat", {}).get("rooms_opened", 0) for p in players.values()
    )
    if _chat_rooms_total and gs.get("trade_efficiency", 0) == 0.0:
        _selected = sum(
            p.get("trade_chat", {}).get("rooms_closed_selected", 0)
            for p in players.values()
        )
        _chat_rate = _selected / _chat_rooms_total
        cols[4].metric("Chat Success Rate", f"{_chat_rate:.1%}")
    else:
        cols[4].metric("Trade Efficiency", f"{gs.get('trade_efficiency', 0):.1%}")

    # ── VP Progression Chart ──
    st.subheader("Victory Point Progression")
    num_turns = gs.get("num_turns", 0)
    vp_chart_data: dict[str, dict[int, int]] = {}
    max_turn = 0
    for pid, pdata in players.items():
        vp_prog = pdata.get("vp_progression", [])
        vp_chart_data[pid] = {entry["turn_index"]: entry["vp"] for entry in vp_prog}
        if vp_prog:
            max_turn = max(max_turn, max(e["turn_index"] for e in vp_prog))
        # Extend to the final turn with actual VP so the plot reaches the game end
        final_vp = pdata.get("final_vp", 0)
        if final_vp and num_turns > 0:
            vp_chart_data[pid][num_turns] = final_vp
            max_turn = max(max_turn, num_turns)

    if vp_chart_data and max_turn > 0:
        chart_rows = []
        for turn in range(max_turn + 1):
            for pid in players:
                series = vp_chart_data.get(pid, {})
                vp = 0
                for t in range(turn + 1):
                    if t in series:
                        vp = series[t]
                chart_rows.append({"Turn": turn, "Player": pid, "VP": vp})
        pids = list(players.keys())
        colors = [PLAYER_COLORS.get(p, NEUTRAL_COLORS)[0] for p in pids]
        vp_chart = (
            alt.Chart(pd.DataFrame(chart_rows))
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("Turn:Q", axis=alt.Axis(title=None, labelFontSize=10)),
                y=alt.Y("VP:Q", axis=alt.Axis(title=None, labelFontSize=10)),
                color=alt.Color(
                    "Player:N",
                    scale=alt.Scale(domain=pids, range=colors),
                    legend=alt.Legend(title=None),
                ),
                tooltip=["Turn:Q", "Player:N", "VP:Q"],
            )
            .properties(height=250)
        )
        st.altair_chart(vp_chart, width="stretch")

    # ── Resource Production ──
    st.subheader("Estimated Resource Production")
    _RES_ORDER = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
    _RES_COLOR = {
        "WOOD": "#16a34a",
        "BRICK": "#c2410c",
        "SHEEP": "#86efac",
        "WHEAT": "#fbbf24",
        "ORE": "#94a3b8",
    }
    res_rows = []
    for pid, pdata in players.items():
        production = pdata.get("resource_production", {}).get("total", {})
        for r in _RES_ORDER:
            res_rows.append(
                {"Player": pid, "Resource": r, "Count": production.get(r, 0)}
            )
    if res_rows:
        df_res = pd.DataFrame(res_rows)
        pids = list(players.keys())
        res_color_list = [_RES_COLOR[r] for r in _RES_ORDER]
        _emoji_expr = "{'WOOD':'🪵','BRICK':'🧱','SHEEP':'🐑','WHEAT':'🌾','ORE':'🪨'}[datum.value]"
        player_charts = []
        for pid in pids:
            accent, _, _ = PLAYER_COLORS.get(pid, NEUTRAL_COLORS)
            c = (
                alt.Chart(df_res[df_res["Player"] == pid])
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=32)
                .encode(
                    x=alt.X(
                        "Resource:N",
                        sort=_RES_ORDER,
                        axis=alt.Axis(
                            title=None,
                            labelExpr=_emoji_expr,
                            labelFontSize=18,
                            ticks=False,
                            domain=False,
                            labelAngle=0,
                        ),
                    ),
                    y=alt.Y(
                        "Count:Q",
                        axis=alt.Axis(title=None, labelFontSize=9, tickCount=4),
                    ),
                    color=alt.Color(
                        "Resource:N",
                        scale=alt.Scale(domain=_RES_ORDER, range=res_color_list),
                        legend=None,
                    ),
                    tooltip=["Resource:N", "Count:Q"],
                )
                .properties(
                    title=alt.TitleParams(
                        pid,
                        color=accent,
                        fontSize=13,
                        fontWeight="bold",
                        anchor="middle",
                    ),
                    height=160,
                    width=180,
                )
            )
            player_charts.append(c)
        st.altair_chart(
            alt.hconcat(*player_charts, spacing=40).configure_view(strokeWidth=0),
            width="stretch",
        )

    # ── Trade Activity ──
    st.subheader("Trade Activity")
    has_classic = any(
        pdata.get("trade", {}).get("offers_made", 0) > 0 for pdata in players.values()
    )
    has_chat = any(
        pdata.get("trade_chat", {}).get("rooms_opened", 0) > 0
        for pdata in players.values()
    )

    def _cumulative_by_turn(by_player: dict[str, dict], max_turn: int) -> list[dict]:
        cum = {pid: 0 for pid in by_player}
        rows = []
        for t in range(max_turn + 1):
            row: dict = {"Turn": t}
            for pid, by_turn in by_player.items():
                cum[pid] += by_turn.get(str(t), 0) + by_turn.get(t, 0)
                row[pid] = cum[pid]
            rows.append(row)
        return rows

    def _altair_line(
        by_player: dict[str, dict], max_turn: int, title: str
    ) -> alt.Chart:
        rows = _cumulative_by_turn(by_player, max_turn)
        df = pd.DataFrame(rows).melt(
            id_vars=["Turn"], var_name="Player", value_name="Count"
        )
        pids = list(by_player.keys())
        colors = [PLAYER_COLORS.get(p, NEUTRAL_COLORS)[0] for p in pids]
        return (
            alt.Chart(df)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("Turn:Q", axis=alt.Axis(title=None, labelFontSize=10)),
                y=alt.Y("Count:Q", axis=alt.Axis(title=None, labelFontSize=10)),
                color=alt.Color(
                    "Player:N",
                    scale=alt.Scale(domain=pids, range=colors),
                    legend=None,
                ),
                tooltip=["Turn:Q", "Player:N", "Count:Q"],
            )
            .properties(title=title, height=180)
        )

    _max_turn = gs.get("num_turns", 0)
    _player_ids = list(players.keys())

    def _render_chart_row(specs: list[tuple[str, dict[str, dict]]]) -> int:
        """Render a row of altair line charts in columns. Returns number rendered."""
        active = [(lbl, bp) for lbl, bp in specs if any(bp.values())]
        if not active:
            return 0
        cols = st.columns(len(active))
        for col, (lbl, bp) in zip(cols, active):
            col.altair_chart(_altair_line(bp, _max_turn, lbl), width="stretch")
        return len(active)

    rendered = 0
    if has_classic:
        rendered += _render_chart_row(
            [
                (
                    "Offers Made",
                    {
                        pid: players[pid]
                        .get("trade", {})
                        .get("offers_made_by_turn", {})
                        for pid in _player_ids
                    },
                ),
                (
                    "Completed",
                    {
                        pid: players[pid].get("trade", {}).get("completed_by_turn", {})
                        for pid in _player_ids
                    },
                ),
            ]
        )
    if has_chat:
        if not has_classic:
            st.caption("All trades negotiated via chat rooms")
        rendered += _render_chart_row(
            [
                (
                    "Rooms Opened",
                    {
                        pid: players[pid]
                        .get("trade_chat", {})
                        .get("rooms_opened_by_turn", {})
                        for pid in _player_ids
                    },
                ),
                (
                    "Rooms Joined",
                    {
                        pid: players[pid]
                        .get("trade_chat", {})
                        .get("rooms_participated_by_turn", {})
                        for pid in _player_ids
                    },
                ),
                (
                    "Proposals Made",
                    {
                        pid: players[pid]
                        .get("trade_chat", {})
                        .get("proposals_made_by_turn", {})
                        for pid in _player_ids
                    },
                ),
                (
                    "Proposals Accepted",
                    {
                        pid: players[pid]
                        .get("trade_chat", {})
                        .get("proposals_accepted_by_turn", {})
                        for pid in _player_ids
                    },
                ),
                (
                    "Completed",
                    {
                        pid: players[pid].get("trade", {}).get("completed_by_turn", {})
                        for pid in _player_ids
                    },
                ),
            ]
        )
    if rendered == 0 and (has_classic or has_chat):
        st.info(
            "Per-turn data not available — regenerate analysis.json to see trade activity charts."
        )

    # ── Building Timeline + Longest Road + Largest Army ──
    st.subheader("Building & Achievement Progression")
    _build_events: list[dict] = []
    for pid, pdata in players.items():
        buildings = pdata.get("buildings", {})
        for s in buildings.get("settlements", []):
            if s.get("turn_index") is not None:
                _build_events.append(
                    {"player": pid, "turn": s["turn_index"], "type": "Settlement"}
                )
        for c in buildings.get("cities", []):
            if c.get("turn_index") is not None:
                _build_events.append(
                    {"player": pid, "turn": c["turn_index"], "type": "City"}
                )
        for r in buildings.get("roads", []):
            if r.get("turn_index") is not None:
                _build_events.append(
                    {"player": pid, "turn": r["turn_index"], "type": "Road"}
                )

    _game_max_turn = gs.get("num_turns", 0)
    _plotly_layout_base = dict(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Turn",
            gridcolor="#1e293b",
            color="#94a3b8",
            range=[0, _game_max_turn] if _game_max_turn else None,
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=10)),
        margin=dict(l=40, r=10, t=30, b=40),
        hovermode="x unified",
    )

    _build_col, _road_col, _army_col = st.columns(3)

    # (a) Building timeline
    with _build_col:
        st.markdown("**🏘️ Buildings**")
        if _build_events:
            _max_turn = max(e["turn"] for e in _build_events)
            _turns = list(range(0, _max_turn + 2))
            _dash = {"Road": "longdash", "Settlement": "solid", "City": "solid"}
            _width = {"Road": 1.5, "Settlement": 2, "City": 4}
            _btimeline_fig = go.Figure()

            for pid in sorted(players.keys()):
                _pcolor = PLAYER_COLORS.get(pid, NEUTRAL_COLORS)[0]
                for btype in ("Road", "Settlement", "City"):
                    _event_turns = sorted(
                        e["turn"]
                        for e in _build_events
                        if e["player"] == pid and e["type"] == btype
                    )
                    if not _event_turns:
                        continue
                    _cum, _count = [], 0
                    for t in _turns:
                        _count += _event_turns.count(t)
                        _cum.append(_count)
                    _btimeline_fig.add_trace(
                        go.Scatter(
                            x=_turns,
                            y=_cum,
                            mode="lines",
                            name=f"{pid} – {btype}",
                            line=dict(
                                color=_pcolor, dash=_dash[btype], width=_width[btype]
                            ),
                            showlegend=False,
                            hovertemplate=f"<b>{pid}</b> {btype}<br>Turn %{{x}}: %{{y}}<extra></extra>",
                        )
                    )
                    if btype == "Settlement":
                        _mx = list(_event_turns)
                        _my = [_cum[_turns.index(t)] for t in _mx]
                        _btimeline_fig.add_trace(
                            go.Scatter(
                                x=_mx,
                                y=_my,
                                mode="markers",
                                marker=dict(
                                    symbol="circle",
                                    size=8,
                                    color=_pcolor,
                                    line=dict(width=1.5, color="#0f172a"),
                                ),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )
                    elif btype == "City":
                        _mx = list(_event_turns)
                        _my = [_cum[_turns.index(t)] for t in _mx]
                        _btimeline_fig.add_trace(
                            go.Scatter(
                                x=_mx,
                                y=_my,
                                mode="text",
                                text=["⌂"] * len(_mx),
                                textfont=dict(size=20, color=_pcolor),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )

            # Static legend as annotations at the top
            _btimeline_fig.update_layout(
                **{**_plotly_layout_base, "margin": dict(l=40, r=10, t=50, b=40)},
                showlegend=False,
                yaxis=dict(
                    title="Cumulative count", gridcolor="#1e293b", color="#94a3b8"
                ),
                annotations=[
                    dict(
                        text="- - Road &nbsp;&nbsp; ── ● Settlement &nbsp;&nbsp; ━━ ⌂ City",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=1.12,
                        showarrow=False,
                        font=dict(color="#94a3b8", size=10),
                        xanchor="center",
                    ),
                ],
            )
            st.plotly_chart(_btimeline_fig, width="stretch")
        else:
            st.info("No building data available.")

    # (b) Longest road progression
    with _road_col:
        st.markdown("**🛤️ Longest Road**")
        _road_fig = go.Figure()
        _has_road_data = False
        for pid in sorted(players.keys()):
            _pcolor = PLAYER_COLORS.get(pid, NEUTRAL_COLORS)[0]
            road_prog = players[pid].get("road_progression", [])
            if not road_prog:
                continue
            _has_road_data = True
            turns = [e["turn_index"] for e in road_prog]
            lengths = [e["road_length"] for e in road_prog]
            _road_fig.add_trace(
                go.Scatter(
                    x=turns,
                    y=lengths,
                    mode="lines",
                    name=pid,
                    line=dict(color=_pcolor, width=2),
                    showlegend=False,
                    hovertemplate=f"<b>{pid}</b><br>Turn %{{x}}: %{{y}} segments<extra></extra>",
                )
            )
        # Add threshold line at 5 (minimum for longest road award)
        if _has_road_data:
            _road_fig.add_hline(
                y=5,
                line_dash="dot",
                line_color="#94a3b8",
                annotation_text="Min for 🏆",
                annotation_position="top left",
                annotation_font_color="#94a3b8",
                annotation_font_size=10,
            )
            _road_fig.update_layout(
                **_plotly_layout_base,
                yaxis=dict(title="Road length", gridcolor="#1e293b", color="#94a3b8"),
            )
            st.plotly_chart(_road_fig, width="stretch")
        else:
            st.info("No road progression data — regenerate analysis.json.")

    # (c) Largest army progression
    with _army_col:
        st.markdown("**⚔️ Largest Army**")
        _army_fig = go.Figure()
        _has_army_data = False
        for pid in sorted(players.keys()):
            _pcolor = PLAYER_COLORS.get(pid, NEUTRAL_COLORS)[0]
            army_prog = players[pid].get("army_progression", [])
            if not army_prog:
                continue
            _has_army_data = True
            turns = [e["turn_index"] for e in army_prog]
            knights = [e["knights"] for e in army_prog]
            _army_fig.add_trace(
                go.Scatter(
                    x=turns,
                    y=knights,
                    mode="lines",
                    name=pid,
                    line=dict(color=_pcolor, width=2),
                    showlegend=False,
                    hovertemplate=f"<b>{pid}</b><br>Turn %{{x}}: %{{y}} knights<extra></extra>",
                )
            )
        if _has_army_data:
            _army_fig.add_hline(
                y=3,
                line_dash="dot",
                line_color="#94a3b8",
                annotation_text="Min for 🏆",
                annotation_position="top left",
                annotation_font_color="#94a3b8",
                annotation_font_size=10,
            )
            _army_fig.update_layout(
                **_plotly_layout_base,
                yaxis=dict(
                    title="Knights played", gridcolor="#1e293b", color="#94a3b8"
                ),
            )
            st.plotly_chart(_army_fig, width="stretch")
        else:
            st.info("No army progression data — regenerate analysis.json.")

    # ── Per-Player Summary Cards ──
    st.subheader("Player Summary")
    player_cols = st.columns(len(players) or 1)
    for idx, (pid, pdata) in enumerate(players.items()):
        with player_cols[idx]:
            accent, _, _ = PLAYER_COLORS.get(pid, NEUTRAL_COLORS)
            st.markdown(
                f"<span style='color:{accent};font-weight:700'>{pid}</span> {'🏆' if pdata.get('is_winner') else ''}",
                unsafe_allow_html=True,
            )
            st.metric("Final VP", pdata.get("final_vp", "?"))

            counts = pdata.get("buildings", {}).get("counts", {})
            st.caption(
                f"Buildings: {counts.get('settlements', 0)}S / {counts.get('cities', 0)}C / {counts.get('roads', 0)}R"
            )

            achievements = pdata.get("achievements", {})
            if achievements.get("has_longest_road"):
                st.caption("Longest Road")
            if achievements.get("has_largest_army"):
                st.caption("Largest Army")

            robber = pdata.get("robber", {})
            st.caption(
                f"Robber: moved {robber.get('times_moved_robber', 0)}x, targeted {robber.get('times_targeted', 0)}x"
            )

            dev = pdata.get("dev_cards", {})
            if dev.get("cards_played", 0) or dev.get("cards_held_at_end", 0):
                st.caption(
                    f"Dev cards: {dev.get('cards_played', 0)} played, {dev.get('cards_held_at_end', 0)} held"
                )

            dq = pdata.get("decision_quality", {})
            if dq.get("total_prompts", 0):
                st.caption(
                    f"Retries: {dq.get('retries', 0)}/{dq.get('total_prompts', 0)} ({dq.get('retry_rate', 0):.1%})"
                )

            phase = pdata.get("phase_analysis", {}).get("opening", {})
            if phase.get("pip_count", 0):
                st.caption(
                    f"Opening: {phase.get('resource_diversity', 0)} types, {phase.get('pip_count', 0)} pips"
                )
            market_profile = pdata.get("market_profile", {})
            if market_profile:
                st.caption(
                    "Market: "
                    f"{market_profile.get('market_role', '—')} · "
                    f"init {market_profile.get('market_initiation_rate', 0):.0%} · "
                    f"offerer {market_profile.get('offerer_share', 0):.0%}"
                )
            strat = pdata.get("strategy", {})
            if strat.get("strategy_stability") is not None:
                st.caption(
                    f"Strategy stability: {strat.get('strategy_stability', 0):.0%}"
                )

    # ── Trade Negotiation Breakdown ──
    any_chat = any(
        pdata.get("trade_chat", {}).get("rooms_opened", 0)
        or pdata.get("trade_chat", {}).get("proposals_made", 0)
        for pdata in players.values()
    )
    if any_chat:
        num_turns = gs.get("num_turns", 0)

        st.subheader("Trade Chat Negotiations")
        chat_rows = []
        for pid, pdata in players.items():
            tc = pdata.get("trade_chat", {})
            opened = tc.get("rooms_opened", 0)
            chat_rows.append(
                {
                    "Player": pid,
                    "Rooms Opened": opened,
                    "Successful Initiated Trades": f"{tc.get('negotiation_success_rate', 0):.0%}",
                    "Proposals Made": tc.get("proposals_made", 0),
                    "Proposals Accepted": tc.get("proposals_accepted", 0),
                    "Counter-Offers Made": tc.get("counter_offers_made", 0),
                    "Others' Rooms Participated": tc.get("rooms_participated_in", 0),
                    "Rooms Opened / Turn": (
                        f"{opened / num_turns:.2f}" if num_turns > 0 else "—"
                    ),
                }
            )
        st.dataframe(chat_rows, width="stretch")

        # ── Negotiation Strategy Analysis ──
        st.subheader("🎯 Negotiation Strategy")
        st.caption("Owner perspective: whether final deals match the original ask.")
        neg_rows = []
        for pid, pdata in players.items():
            tc = pdata.get("trade_chat", {})
            opened = tc.get("rooms_opened", 0)
            matched = tc.get("deals_matched_original", 0)
            negotiated = tc.get("deals_negotiated_away", 0)
            total_deals = matched + negotiated
            neg_rows.append(
                {
                    "Player": pid,
                    "🏠 Rooms Opened": opened,
                    "✅ Deals at Original Terms": matched,
                    "🔄 Deals Negotiated Away": negotiated,
                    "📊 Original Terms Rate": (
                        f"{matched / total_deals:.0%}" if total_deals > 0 else "—"
                    ),
                    "🔁 Counter-Offers Received": tc.get("counter_offers_received", 0),
                    "🤝 Others' Rooms Participated": tc.get("rooms_participated_in", 0),
                }
            )
        if any(r["🏠 Rooms Opened"] > 0 for r in neg_rows):
            st.dataframe(neg_rows, width="stretch")
        else:
            st.info("No trade room data available for negotiation analysis.")

        # ── Bipartite trade network (aggregated across all resources) ──
        _RESOURCE_EMOJIS = {
            "WOOD": "🪵",
            "BRICK": "🧱",
            "SHEEP": "🐑",
            "WHEAT": "🌾",
            "ORE": "🪨",
        }
        _ENTITY_COLORS: dict[str, str] = {"BANK": "#6b7280", "PORT": "#8b5cf6"}
        # Build total trading volume between each pair of actors
        pair_volume: dict[tuple[str, str], int] = {}
        all_trade_nodes: set[str] = set()
        # Player-to-player edges from chat trades
        for pid, pdata in players.items():
            flow = pdata.get("trade_chat", {}).get("counterparty_resource_flow", {})
            for partner, res_counts in flow.items():
                all_trade_nodes.update([pid, partner])
                for cnt in res_counts.values():
                    if cnt > 0:
                        key: tuple[str, str] = tuple(sorted([pid, partner]))  # type: ignore[assignment]
                        pair_volume[key] = pair_volume.get(key, 0) + cnt

        # Add BANK/PORT edges from market actor data
        market = analysis_data.get("market", {})
        market_actors = market.get("actors", {}) if isinstance(market, dict) else {}
        for passive_entity in ("BANK", "PORT"):
            entity_stats = market_actors.get(passive_entity, {})
            if not isinstance(entity_stats, dict):
                continue
            taker_vol = entity_stats.get("taker_volume_by_resource", {})
            for res, vol in taker_vol.items():
                if vol <= 0:
                    continue
                for pid in players:
                    p_stats = market_actors.get(pid, {})
                    if not isinstance(p_stats, dict):
                        continue
                    if p_stats.get("maker_volume_by_resource", {}).get(res, 0) > 0:
                        all_trade_nodes.update([pid, passive_entity])
                        key = tuple(sorted([pid, passive_entity]))  # type: ignore[assignment]
                        pair_volume[key] = pair_volume.get(key, 0) + vol

        if pair_volume and all_trade_nodes:
            st.caption(
                "Total trade volume between each pair of actors (incl. 🏦 BANK / ⚓ PORT maritime)"
            )

            _NODE_ORDER = ("BLUE", "ORANGE", "RED", "WHITE", "BANK", "PORT")
            _NODE_EMOJIS: dict[str, str] = {
                "BLUE": "🔵",
                "RED": "🔴",
                "WHITE": "⚪",
                "ORANGE": "🟠",
                "BANK": "🏦",
                "PORT": "⚓",
            }
            hex_nodes = [n for n in _NODE_ORDER if n in all_trade_nodes]

            # Regular hexagon: all 6 nodes at equal radius, one per vertex
            # Vertex order clockwise from top: BLUE, ORANGE, BANK, WHITE, RED, PORT
            _HEX_ORDER = ("BLUE", "ORANGE", "BANK", "WHITE", "RED", "PORT")
            all_hex = [n for n in _HEX_ORDER if n in hex_nodes]
            # Pad with any nodes not in the fixed order
            all_hex += [n for n in hex_nodes if n not in _HEX_ORDER]
            _R = 1.0
            pos: dict[str, tuple[float, float]] = {}
            n_total = len(all_hex)
            for i, node in enumerate(all_hex):
                angle = math.pi / 2 - 2 * math.pi * i / n_total  # clockwise from top
                pos[node] = (_R * math.cos(angle), _R * math.sin(angle))

            # Faint circle guide
            circle_xs = [_R * math.cos(2 * math.pi * i / 60) for i in range(61)]
            circle_ys = [_R * math.sin(2 * math.pi * i / 60) for i in range(61)]

            fig = go.Figure()
            max_vol = max(pair_volume.values()) if pair_volume else 1

            # Faint circle
            fig.add_trace(
                go.Scatter(
                    x=circle_xs,
                    y=circle_ys,
                    mode="lines",
                    line=dict(width=1, color="rgba(148,163,184,0.1)"),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

            # Edges
            for (a, b), vol in pair_volume.items():
                if a not in pos or b not in pos:
                    continue
                x0, y0 = pos[a]
                x1, y1 = pos[b]
                lw = 1.5 + 8.5 * (vol / max_vol)
                # Slight quadratic bezier curving outward
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                dist = math.hypot(mx, my)
                if dist > 0.05:
                    cx = mx + 0.25 * mx / dist
                    cy = my + 0.25 * my / dist
                else:
                    cx, cy = mx, my
                curve_xs = [
                    (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1
                    for t in [i / 40 for i in range(41)]
                ]
                curve_ys = [
                    (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1
                    for t in [i / 40 for i in range(41)]
                ]
                fig.add_trace(
                    go.Scatter(
                        x=curve_xs + [None],
                        y=curve_ys + [None],
                        mode="lines",
                        line=dict(width=lw, color="rgba(200,200,200,0.4)"),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )

            # Nodes
            for node in all_hex:
                x, y = pos[node]
                node_color = _ENTITY_COLORS.get(
                    node, PLAYER_COLORS.get(node, NEUTRAL_COLORS)[0]
                )
                node_size = 34
                label = f"{_NODE_EMOJIS.get(node, '')} {node}"
                # Position label away from center
                if abs(y) > 0.5:
                    text_pos = "top center" if y > 0 else "bottom center"
                elif x < 0:
                    text_pos = "middle left"
                else:
                    text_pos = "middle right"
                fig.add_trace(
                    go.Scatter(
                        x=[x],
                        y=[y],
                        mode="markers+text",
                        marker=dict(
                            size=node_size,
                            color=node_color,
                            symbol="hexagon",
                            line=dict(width=2, color="#1f2937"),
                        ),
                        text=label,
                        textposition=text_pos,
                        textfont=dict(size=11, color="#cbd5e1"),
                        hovertext=f"{node}: {sum(v for (a, b), v in pair_volume.items() if node in (a, b))} total vol",
                        hoverinfo="text",
                        showlegend=False,
                    )
                )

            fig.update_layout(
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    range=[-1.6, 1.6],
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    range=[-1.6, 1.6],
                    scaleanchor="x",
                    scaleratio=1,
                ),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, width="stretch")

    # ── Market Structure ──
    market = analysis_data.get("market", {})
    market_actors = market.get("actors", {}) if isinstance(market, dict) else {}
    if market_actors:
        st.subheader("Market Structure")

        _TABLE_EMOJIS: dict[str, str] = {
            "BLUE": "🔵",
            "RED": "🔴",
            "WHITE": "⚪",
            "ORANGE": "🟠",
            "BANK": "🏦",
            "PORT": "⚓",
        }

        # ── Maker / Taker bar chart ──
        _MT_ACTOR_EMOJIS: dict[str, str] = {
            "BLUE": "🔵",
            "RED": "🔴",
            "WHITE": "⚪",
            "ORANGE": "🟠",
            "BANK": "🏦",
            "PORT": "⚓",
        }
        maker_taker_rows = []
        # Players first, then BANK/PORT grouped at the end
        mt_actor_order_raw = [a for a in market_actors if a in players] + [
            a for a in ("BANK", "PORT") if a in market_actors
        ]
        for actor in mt_actor_order_raw:
            stats = market_actors[actor]
            if not isinstance(stats, dict):
                continue
            label = f"{_MT_ACTOR_EMOJIS.get(actor, '')} {actor}"
            maker_taker_rows.append(
                {
                    "Actor": label,
                    "Deals": stats.get("maker_deals", 0),
                    "Type": "📈 Maker",
                }
            )
            maker_taker_rows.append(
                {
                    "Actor": label,
                    "Deals": stats.get("taker_deals", 0),
                    "Type": "📉 Taker",
                }
            )
        if maker_taker_rows:
            mt_df = pd.DataFrame(maker_taker_rows)
            actor_order = [
                f"{_MT_ACTOR_EMOJIS.get(a, '')} {a}" for a in mt_actor_order_raw
            ]
            mt_chart = (
                alt.Chart(mt_df)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X(
                        "Actor:N",
                        sort=actor_order,
                        axis=alt.Axis(
                            labelColor="#e2e8f0", titleColor="#e2e8f0", labelAngle=0
                        ),
                    ),
                    y=alt.Y(
                        "Deals:Q",
                        axis=alt.Axis(labelColor="#94a3b8", titleColor="#e2e8f0"),
                    ),
                    color=alt.Color(
                        "Type:N",
                        scale=alt.Scale(
                            domain=["📈 Maker", "📉 Taker"],
                            range=["#22c55e", "#ef4444"],
                        ),
                        legend=alt.Legend(
                            title="Type", labelColor="#e2e8f0", titleColor="#e2e8f0"
                        ),
                    ),
                    xOffset="Type:N",
                    tooltip=["Actor", "Type", "Deals"],
                )
                .properties(
                    height=360, title=alt.Title("Maker vs Taker Deals", color="#e2e8f0")
                )
                .configure_view(strokeWidth=0)
                .configure_axis(gridColor="rgba(148,163,184,0.15)")
            )
            st.altair_chart(mt_chart, width="stretch")

        # ── Market Share Histogram ──
        resource_market_share = market.get("resource_market_share", {})
        if resource_market_share:
            st.subheader("Market Share by Resource")
            st.caption(
                "Each player's share of total trade volume per resource (including BANK/PORT)."
            )

            _RESOURCE_LABELS = {
                "WOOD": "🪵 WOOD",
                "BRICK": "🧱 BRICK",
                "SHEEP": "🐑 SHEEP",
                "WHEAT": "🌾 WHEAT",
                "ORE": "🪨 ORE",
            }
            _ACTOR_LABELS = {
                "BLUE": "🔵 BLUE",
                "RED": "🔴 RED",
                "WHITE": "⚪ WHITE",
                "ORANGE": "🟠 ORANGE",
                "BANK": "🏦 BANK",
                "PORT": "⚓ PORT",
            }
            share_chart_rows = []
            all_share_actors: list[str] = []
            for resource in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"):
                shares = resource_market_share.get(resource, {})
                for actor, share in shares.items():
                    if actor not in all_share_actors:
                        all_share_actors.append(actor)
                    share_chart_rows.append(
                        {
                            "Resource": _RESOURCE_LABELS.get(resource, resource),
                            "Actor": _ACTOR_LABELS.get(actor, actor),
                            "Share": share,
                        }
                    )

            if share_chart_rows:
                share_df = pd.DataFrame(share_chart_rows)
                # Stack order (bottom→top): BANK, PORT, BLUE, ORANGE, RED, WHITE
                # Legend reads top→bottom matching the visual top→bottom of bars
                _STACK_ORDER = ("BANK", "PORT", "BLUE", "ORANGE", "RED", "WHITE")
                ordered_actors_raw_bottom_up = [
                    a for a in _STACK_ORDER if a in all_share_actors
                ]
                # Legend order is reversed (top→bottom = top of stack first)
                legend_order_raw = list(reversed(ordered_actors_raw_bottom_up))
                legend_actors = [_ACTOR_LABELS.get(a, a) for a in legend_order_raw]
                legend_colors = [
                    _ENTITY_COLORS.get(a, PLAYER_COLORS.get(a, NEUTRAL_COLORS)[0])
                    for a in legend_order_raw
                ]
                # Add stack sort index (lower = bottom of stack)
                stack_index_map = {
                    _ACTOR_LABELS.get(a, a): i
                    for i, a in enumerate(ordered_actors_raw_bottom_up)
                }
                share_df["_stack_order"] = share_df["Actor"].map(stack_index_map)
                resource_order = [
                    _RESOURCE_LABELS[r]
                    for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
                ]

                share_chart = (
                    alt.Chart(share_df)
                    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(
                        x=alt.X(
                            "Resource:N",
                            sort=resource_order,
                            axis=alt.Axis(labelColor="#e2e8f0", titleColor="#e2e8f0"),
                        ),
                        y=alt.Y(
                            "Share:Q",
                            stack="normalize",
                            axis=alt.Axis(
                                format="%", labelColor="#94a3b8", titleColor="#e2e8f0"
                            ),
                            title="Market Share",
                        ),
                        color=alt.Color(
                            "Actor:N",
                            scale=alt.Scale(domain=legend_actors, range=legend_colors),
                            legend=alt.Legend(
                                title="Actor",
                                labelColor="#e2e8f0",
                                titleColor="#e2e8f0",
                            ),
                        ),
                        order=alt.Order("_stack_order:Q"),
                        tooltip=[
                            alt.Tooltip("Resource:N"),
                            alt.Tooltip("Actor:N"),
                            alt.Tooltip("Share:Q", format=".1%"),
                        ],
                    )
                    .properties(
                        height=380,
                        title=alt.Title("Market Share per Resource", color="#e2e8f0"),
                    )
                    .configure_view(strokeWidth=0)
                    .configure_axis(gridColor="rgba(148,163,184,0.15)")
                )
                st.altair_chart(share_chart, width="stretch")

    # ── Strategy Evolution ──
    any_strategy = any(
        pdata.get("strategy", {}).get("opening_strategy") for pdata in players.values()
    )
    if any_strategy:
        st.subheader("Strategy Evolution")
        for pid, pdata in players.items():
            strat = pdata.get("strategy", {})
            if not strat.get("opening_strategy") and not strat.get("strategy_updates"):
                continue
            with st.expander(
                f"{pid} — {strat.get('strategy_update_count', 0)} strategy updates"
            ):
                st.caption(
                    f"Strategy stability: {strat.get('strategy_stability', 0):.0%}"
                )
                if strat.get("opening_strategy"):
                    st.markdown(f"**Opening strategy:** {strat['opening_strategy']}")
                updates = strat.get("strategy_updates", [])
                for i, upd in enumerate(updates):
                    if i == 0 and upd.get("stage") == "opening_strategy":
                        continue  # Already shown above
                    lt = upd.get("long_term", "")
                    st.markdown(
                        f"**Turn {upd.get('turn_index', '?')}** ({upd.get('stage', '')}): {lt}"
                    )
                if strat.get("final_strategy") and (
                    not updates
                    or updates[-1].get("long_term") != strat["final_strategy"]
                ):
                    st.markdown(f"**Final strategy:** {strat['final_strategy']}")


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
          color: #0f172a;
        }
        .timeline-body {
          font-size: 0.95rem;
          line-height: 1.35;
          margin-bottom: 0.35rem;
          white-space: pre-wrap;
          color: #0f172a;
        }
        .timeline-footer {
          font-size: 0.75rem;
          color: #6b7280;
        }
        .player-column-header {
          border-radius: 0.75rem;
          padding: 0.7rem 0.8rem;
          margin-bottom: 0.8rem;
          color: #0f172a;
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
          background: #000000;
          border: 1px solid #374151;
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
          color: #0f172a;
        }
        .trade-chat-header {
          display: flex;
          flex-wrap: wrap;
          align-items: baseline;
          gap: 0.45rem;
          margin-bottom: 0.2rem;
          color: #0f172a;
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
          color: #0f172a;
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
          color: #cbd5e1;
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
          color: #e2e8f0;
        }
        .turn-event-player {
          font-weight: 700;
          min-width: 4.6rem;
        }
        .turn-event-text {
          color: #e2e8f0;
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
        default="runs/0.4.0/dev",
        help="Run directory to monitor live.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
