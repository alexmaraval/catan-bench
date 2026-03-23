from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from .replay import ReplayTimelineItem, build_player_replay_timeline, build_replay_timeline
from .schemas import JsonValue


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    run_dir: Path
    metadata: dict[str, JsonValue]
    player_ids: tuple[str, ...]
    public_timeline: list[ReplayTimelineItem]
    player_timelines: dict[str, list[ReplayTimelineItem]]
    memories_by_player: dict[str, JsonValue | None]
    prompt_traces_by_player: dict[str, tuple[dict[str, JsonValue], ...]]
    result: dict[str, JsonValue] | None


def load_dashboard_snapshot(run_dir: str | Path) -> DashboardSnapshot:
    run_path = Path(run_dir)
    metadata = _read_json_if_exists(run_path / "metadata.json") or {}
    player_ids = _player_ids_from_run(run_path, metadata)

    public_timeline = build_replay_timeline(run_path) if metadata else []
    player_timelines = {
        player_id: build_player_replay_timeline(run_path, player_id)
        for player_id in player_ids
        if metadata
    }
    memories_by_player = {
        player_id: _memory_content(run_path / "players" / player_id / "memory.json")
        for player_id in player_ids
    }
    prompt_traces_by_player = {
        player_id: tuple(
            _read_jsonl(run_path / "players" / player_id / "prompt_trace.jsonl")
        )
        for player_id in player_ids
    }
    result = _read_json_if_exists(run_path / "result.json")

    return DashboardSnapshot(
        run_dir=run_path,
        metadata=metadata,
        player_ids=player_ids,
        public_timeline=public_timeline,
        player_timelines=player_timelines,
        memories_by_player=memories_by_player,
        prompt_traces_by_player=prompt_traces_by_player,
        result=result,
    )


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

    st.title("catan-bench live dashboard")
    st.caption("Live view over run artifacts written by the benchmark harness.")

    run_dir_input = st.sidebar.text_input("Run directory", value=str(default_run_dir))
    auto_refresh = st.sidebar.toggle("Auto refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 15, 2)
    timeline_limit = st.sidebar.slider("Timeline items", 5, 100, 20)
    selected_player = st.sidebar.selectbox(
        "Player detail",
        options=["Public"] + sorted(_player_ids_for_sidebar(run_dir_input)),
    )
    st.sidebar.button("Refresh now", use_container_width=True)

    run_path = Path(run_dir_input)
    if not run_path.exists():
        st.warning(f"Run directory does not exist yet: `{run_path}`")
        _maybe_rerun(st, auto_refresh=auto_refresh, refresh_interval=refresh_interval)
        return

    snapshot = load_dashboard_snapshot(run_path)
    if not snapshot.metadata:
        st.info("Waiting for metadata.json. Start a game and this page will populate live.")
        _maybe_rerun(st, auto_refresh=auto_refresh, refresh_interval=refresh_interval)
        return

    _render_header(st, snapshot)
    overview_tab, public_tab, player_tab, prompt_tab = st.tabs(
        ["Overview", "Public Timeline", "Player View", "Prompt Traces"]
    )

    with overview_tab:
        _render_overview(st, snapshot, timeline_limit=timeline_limit)
    with public_tab:
        _render_timeline(
            st,
            title="Recent public events",
            items=snapshot.public_timeline,
            limit=timeline_limit,
        )
    with player_tab:
        if selected_player == "Public":
            st.info("Select a player in the sidebar to inspect private history and memory.")
        else:
            _render_player_view(
                st,
                snapshot,
                player_id=selected_player,
                timeline_limit=timeline_limit,
            )
    with prompt_tab:
        _render_prompt_traces(st, snapshot, timeline_limit=timeline_limit)

    _maybe_rerun(st, auto_refresh=auto_refresh, refresh_interval=refresh_interval)


def _render_header(st, snapshot: DashboardSnapshot) -> None:
    metadata = snapshot.metadata
    result = snapshot.result or {}
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Game", str(metadata.get("game_id", snapshot.run_dir.name)))
    col2.metric("Status", _status_label(snapshot))
    col3.metric("Players", len(snapshot.player_ids))
    col4.metric("Public events", len(snapshot.public_timeline))
    col5.metric("Decisions", _decision_count(snapshot))

    if result.get("winner_ids"):
        st.success("Winner: " + ", ".join(str(player_id) for player_id in result["winner_ids"]))


def _render_overview(st, snapshot: DashboardSnapshot, *, timeline_limit: int) -> None:
    left, right = st.columns((2, 1))
    with left:
        st.subheader("Latest public events")
        _render_timeline_items(st, snapshot.public_timeline[-timeline_limit:])
    with right:
        st.subheader("Current memories")
        for player_id in snapshot.player_ids:
            with st.container(border=True):
                st.markdown(f"**{player_id}**")
                memory = snapshot.memories_by_player.get(player_id)
                if memory is None:
                    st.caption("No memory stored yet.")
                else:
                    st.json(memory, expanded=False)


def _render_player_view(
    st,
    snapshot: DashboardSnapshot,
    *,
    player_id: str,
    timeline_limit: int,
) -> None:
    st.subheader(f"{player_id} private view")
    memory = snapshot.memories_by_player.get(player_id)
    memory_col, trace_col = st.columns((1, 1))
    with memory_col:
        st.markdown("**Current memory**")
        if memory is None:
            st.caption("No memory stored yet.")
        else:
            st.json(memory, expanded=False)
    with trace_col:
        st.markdown("**Latest prompt trace**")
        traces = snapshot.prompt_traces_by_player.get(player_id, ())
        if not traces:
            st.caption("No prompt traces yet.")
        else:
            latest_trace = traces[-1]
            st.json(
                {
                    "turn_index": latest_trace.get("turn_index"),
                    "phase": latest_trace.get("phase"),
                    "stage": latest_trace.get("stage"),
                    "model": latest_trace.get("model"),
                    "attempt_count": len(latest_trace.get("attempts", [])),
                },
                expanded=False,
            )

    st.markdown("**Recent combined timeline**")
    _render_timeline_items(
        st,
        snapshot.player_timelines.get(player_id, [])[-timeline_limit:],
    )


def _render_prompt_traces(st, snapshot: DashboardSnapshot, *, timeline_limit: int) -> None:
    for player_id in snapshot.player_ids:
        traces = snapshot.prompt_traces_by_player.get(player_id, ())
        with st.expander(f"{player_id} prompt traces ({len(traces)})", expanded=False):
            if not traces:
                st.caption("No prompt traces yet.")
                continue
            for trace in reversed(traces[-timeline_limit:]):
                label = (
                    f"turn={trace.get('turn_index')} "
                    f"phase={trace.get('phase')} "
                    f"stage={trace.get('stage')}"
                )
                with st.container(border=True):
                    st.markdown(f"**{label}**")
                    st.json(trace, expanded=False)


def _render_timeline(
    st,
    *,
    title: str,
    items: list[ReplayTimelineItem],
    limit: int,
) -> None:
    st.subheader(title)
    _render_timeline_items(st, items[-limit:])


def _render_timeline_items(st, items: list[ReplayTimelineItem]) -> None:
    if not items:
        st.caption("No events yet.")
        return

    for item in items:
        with st.container(border=True):
            st.markdown(f"**{item.title}**")
            st.write(item.body)
            st.caption(
                f"turn {item.turn_index} · phase `{item.phase}` · "
                f"decision {item.decision_index if item.decision_index is not None else '-'} · "
                f"stream `{item.stream}`"
            )
            with st.expander("Raw payload", expanded=False):
                st.json(item.raw_payload, expanded=False)


def _status_label(snapshot: DashboardSnapshot) -> str:
    if snapshot.result is not None:
        return str(_as_dict(snapshot.result.get("metadata")).get("status", "finished"))
    return "in_progress"


def _decision_count(snapshot: DashboardSnapshot) -> int:
    if snapshot.result is not None:
        total = snapshot.result.get("total_decisions")
        if isinstance(total, int):
            return total
    decisions = [
        item.decision_index
        for item in snapshot.public_timeline
        if item.decision_index is not None
    ]
    if not decisions:
        return 0
    return max(decisions) + 1


def _memory_content(path: Path) -> JsonValue | None:
    payload = _read_json_if_exists(path)
    if payload is None:
        return None
    memory = payload.get("memory")
    if memory is not None:
        return memory
    return payload.get("content")


def _player_ids_for_sidebar(run_dir: str | Path) -> tuple[str, ...]:
    run_path = Path(run_dir)
    metadata = _read_json_if_exists(run_path / "metadata.json") or {}
    return _player_ids_from_run(run_path, metadata)


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


def _as_dict(value: JsonValue | None) -> dict[str, JsonValue]:
    if isinstance(value, dict):
        return value
    return {}


def _maybe_rerun(st, *, auto_refresh: bool, refresh_interval: int) -> None:
    if not auto_refresh:
        return
    time.sleep(refresh_interval)
    st.rerun()


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
        default="runs/quickstart",
        help="Run directory to monitor live.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
