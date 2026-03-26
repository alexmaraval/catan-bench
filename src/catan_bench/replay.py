from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .schemas import Event, JsonValue


@dataclass(frozen=True, slots=True)
class ReplayTimelineItem:
    history_index: int
    title: str
    body: str
    turn_index: int
    phase: str
    decision_index: int | None
    event_kind: str
    raw_payload: dict[str, JsonValue]
    actor_player_id: str | None = None


def build_replay_timeline(run_dir: str | Path) -> list[ReplayTimelineItem]:
    run_path = Path(run_dir)
    return [
        _timeline_item_from_event(Event.from_dict(entry))
        for entry in _read_jsonl(run_path / "public_history.jsonl")
    ]


def build_player_replay_timeline(
    run_dir: str | Path, player_id: str
) -> list[ReplayTimelineItem]:
    _ = player_id
    return build_replay_timeline(run_dir)


def export_replay_html(
    run_dir: str | Path, output_path: str | Path | None = None
) -> Path:
    run_path = Path(run_dir)
    timeline = build_replay_timeline(run_path)
    output = Path(output_path) if output_path is not None else run_path / "replay.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    html_items = []
    for item in timeline:
        html_items.append(
            "<article>"
            f"<h3>{item.history_index}. {item.title}</h3>"
            f"<p>{item.body}</p>"
            f"<small>turn {item.turn_index} · phase {item.phase}</small>"
            "</article>"
        )
    output.write_text(
        (
            "<html><head><meta charset='utf-8'><title>catan-bench replay</title></head><body>"
            "<h1>catan-bench replay</h1>" + "".join(html_items) + "</body></html>"
        ),
        encoding="utf-8",
    )
    return output


def export_player_replay_html(
    run_dir: str | Path,
    player_id: str,
    output_path: str | Path | None = None,
) -> Path:
    run_path = Path(run_dir)
    default_output = run_path / "players" / player_id / "replay.html"
    return export_replay_html(run_path, output_path=output_path or default_output)


def _timeline_item_from_event(event: Event) -> ReplayTimelineItem:
    actor = event.actor_player_id or "SYSTEM"
    title = {
        "trade_offered": f"{actor} · Trade offered",
        "trade_confirmed": f"{actor} · Trade confirmed",
        "trade_chat_message": f"{actor} · Trade chat message",
        "dice_rolled": f"{actor} · Dice rolled",
        "road_built": f"{actor} · Road built",
        "settlement_built": f"{actor} · Settlement built",
        "city_built": f"{actor} · City built",
    }.get(event.kind, f"{actor} · {event.kind.replace('_', ' ')}")
    body = json.dumps(event.payload, sort_keys=True)
    return ReplayTimelineItem(
        history_index=event.history_index,
        title=title,
        body=body,
        turn_index=event.turn_index,
        phase=event.phase,
        decision_index=event.decision_index,
        event_kind=event.kind,
        raw_payload=event.payload,
        actor_player_id=event.actor_player_id,
    )


def _read_jsonl(path: Path) -> list[dict[str, JsonValue]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export a simple public replay HTML page."
    )
    parser.add_argument("run_dir", help="Run directory to export.")
    parser.add_argument("--output", help="Optional output file path.")
    args = parser.parse_args(argv)
    export_replay_html(args.run_dir, output_path=args.output)


if __name__ == "__main__":
    main()
