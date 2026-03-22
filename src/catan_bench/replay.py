from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Iterable

from .schemas import JsonValue

PLAYER_PALETTE = ("red", "blue", "orange", "white", "green", "teal")
EVENT_KIND_TITLES = {
    "dice_rolled": "Dice Rolled",
    "settlement_built": "Settlement Built",
    "city_built": "City Built",
    "road_built": "Road Built",
    "robber_moved": "Robber Moved",
    "trade_offered": "Trade Offered",
    "trade_accepted": "Trade Accepted",
    "trade_rejected": "Trade Rejected",
    "trade_confirmed": "Trade Confirmed",
    "trade_cancelled": "Trade Cancelled",
    "turn_ended": "Turn Ended",
    "development_card_played": "Development Card Played",
    "action_taken": "Action",
}
SYSTEM_EVENT_KINDS = {"trade_confirmed"}
SPECIAL_EVENT_KINDS = {
    "dice_rolled",
    "robber_moved",
    "trade_offered",
    "trade_accepted",
    "trade_rejected",
    "trade_confirmed",
    "trade_cancelled",
    "development_card_played",
}


@dataclass(frozen=True, slots=True)
class ReplayTimelineItem:
    speaker_type: str
    speaker_id: str
    variant: str
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
    metadata = _read_json(run_path / "metadata.json")
    events = _read_jsonl(run_path / "public_history.jsonl")
    player_ids = [str(player_id) for player_id in metadata.get("player_ids", [])]
    color_by_player = _player_colors(player_ids)
    return [
        _timeline_item_from_event(event, color_by_player=color_by_player)
        for event in events
    ]


def export_replay_html(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    run_path = Path(run_dir)
    metadata = _read_json(run_path / "metadata.json")
    result = _read_json(run_path / "result.json")
    timeline = build_replay_timeline(run_path)
    output = Path(output_path) if output_path is not None else run_path / "replay.html"
    output.write_text(
        _render_html(metadata=metadata, result=result, timeline=timeline),
        encoding="utf-8",
    )
    return output


def _read_json(path: Path) -> dict[str, JsonValue]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, JsonValue]]:
    events: list[dict[str, JsonValue]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _player_colors(player_ids: Iterable[str]) -> dict[str, str]:
    return {
        player_id: PLAYER_PALETTE[index % len(PLAYER_PALETTE)]
        for index, player_id in enumerate(player_ids)
    }


def _timeline_item_from_event(
    event: dict[str, JsonValue], *, color_by_player: dict[str, str]
) -> ReplayTimelineItem:
    payload = _as_dict(event.get("payload"))
    kind = str(event["kind"])
    actor_player_id = _as_optional_str(event.get("actor_player_id"))

    speaker_type = "event" if kind in SYSTEM_EVENT_KINDS or actor_player_id is None else "player"
    if speaker_type == "player" and actor_player_id is not None:
        speaker_id = actor_player_id
        variant = color_by_player.get(actor_player_id, "neutral")
        title = actor_player_id
    else:
        speaker_id = kind
        variant = "special" if kind in SPECIAL_EVENT_KINDS else "system"
        title = EVENT_KIND_TITLES.get(kind, _titleize(kind))

    if speaker_type == "player" and kind in SPECIAL_EVENT_KINDS:
        title = f"{actor_player_id} · {EVENT_KIND_TITLES.get(kind, _titleize(kind))}"

    return ReplayTimelineItem(
        speaker_type=speaker_type,
        speaker_id=speaker_id,
        variant=variant,
        title=title,
        body=_format_event_body(kind=kind, payload=payload, actor_player_id=actor_player_id),
        turn_index=int(event.get("turn_index", 0)),
        phase=str(event.get("phase", "unknown")),
        decision_index=_as_optional_int(event.get("decision_index")),
        event_kind=kind,
        raw_payload=payload,
        actor_player_id=actor_player_id,
    )


def _format_event_body(
    *, kind: str, payload: dict[str, JsonValue], actor_player_id: str | None
) -> str:
    action_payload = _as_dict(payload.get("action"))
    action_type = _as_optional_str(action_payload.get("action_type"))
    action_detail = _as_dict(action_payload.get("payload"))
    description = _as_optional_str(action_payload.get("description"))

    if kind == "trade_offered":
        offering_player = _as_optional_str(payload.get("offering_player_id")) or actor_player_id
        offer = _format_resource_map(_as_dict(payload.get("offer")))
        request = _format_resource_map(_as_dict(payload.get("request")))
        return f"{offering_player} offered {offer} for {request}."
    if kind == "trade_accepted":
        offering_player = _as_optional_str(payload.get("offering_player_id"))
        return f"Accepted {offering_player}'s trade offer."
    if kind == "trade_rejected":
        offering_player = _as_optional_str(payload.get("offering_player_id"))
        return f"Rejected {offering_player}'s trade offer."
    if kind == "trade_confirmed":
        offering_player = _as_optional_str(payload.get("offering_player_id"))
        accepting_player = _as_optional_str(payload.get("accepting_player_id"))
        offer = _format_resource_map(_as_dict(payload.get("offer")))
        request = _format_resource_map(_as_dict(payload.get("request")))
        return (
            f"{offering_player} traded {offer} with {accepting_player} for {request}."
        )
    if kind == "trade_cancelled":
        offering_player = _as_optional_str(payload.get("offering_player_id")) or actor_player_id
        return f"{offering_player} cancelled the current trade offer."
    if kind == "dice_rolled":
        result = payload.get("result")
        return f"Rolled the dice: {json.dumps(result, sort_keys=True)}."
    if kind == "settlement_built":
        return f"Built a settlement on node {payload.get('node_id')}."
    if kind == "city_built":
        return f"Upgraded node {payload.get('node_id')} to a city."
    if kind == "road_built":
        return f"Built a road on edge {payload.get('edge')}."
    if kind == "robber_moved":
        coordinate = payload.get("coordinate")
        victim = payload.get("victim")
        if victim is None:
            return f"Moved the robber to {coordinate}."
        return f"Moved the robber to {coordinate} and targeted {victim}."
    if kind == "development_card_played":
        if description is not None:
            return description
        if action_type is not None:
            return f"Played {action_type.lower().replace('_', ' ')}."
    if kind == "turn_ended":
        return "Ended the turn."

    if description is not None:
        return description
    if action_type is not None:
        if action_detail:
            return f"{action_type}: {json.dumps(action_detail, sort_keys=True)}"
        return action_type
    return json.dumps(payload, sort_keys=True)


def _format_resource_map(resource_map: dict[str, JsonValue]) -> str:
    if not resource_map:
        return "nothing"
    parts = []
    for resource, amount in sorted(resource_map.items()):
        parts.append(f"{amount} {str(resource).lower()}")
    return ", ".join(parts)


def _render_html(
    *,
    metadata: dict[str, JsonValue],
    result: dict[str, JsonValue],
    timeline: list[ReplayTimelineItem],
) -> str:
    game_id = escape(str(metadata.get("game_id", "unknown-game")))
    player_ids = [str(player_id) for player_id in metadata.get("player_ids", [])]
    winner_ids = ", ".join(str(player_id) for player_id in result.get("winner_ids", [])) or "None"
    total_decisions = escape(str(result.get("total_decisions", "unknown")))
    num_turns = escape(str(result.get("metadata", {}).get("num_turns", result.get("num_turns", "unknown"))))

    bubbles = "\n".join(_render_item(item) for item in timeline)
    player_legend = "\n".join(
        f'<li class="legend__item"><span class="legend__swatch legend__swatch--{escape(color)}"></span>{escape(player_id)}</li>'
        for player_id, color in _player_colors(player_ids).items()
    )
    timeline_json = escape(json.dumps([asdict(item) for item in timeline], sort_keys=True))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{game_id} Replay</title>
  <style>
    :root {{
      --bg: #f5efe4;
      --panel: #fffaf2;
      --ink: #221b16;
      --muted: #776a61;
      --border: #d9cdbd;
      --shadow: 0 18px 40px rgba(76, 50, 31, 0.12);
      --red-bg: #f8d8d2;
      --blue-bg: #d8e8f8;
      --orange-bg: #fde1c4;
      --white-bg: #f2f2ef;
      --green-bg: #dcefd8;
      --teal-bg: #d7efed;
      --special-bg: #f9efc8;
      --system-bg: #ece7e1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top, rgba(196, 155, 86, 0.18), transparent 35%),
        linear-gradient(180deg, #f8f2e8 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    main {{
      width: min(1080px, calc(100vw - 32px));
      margin: 24px auto 48px;
    }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 24px;
      box-shadow: var(--shadow);
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 3vw, 3rem);
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
    }}
    .stats, .legend, .filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      padding: 0;
      margin: 18px 0 0;
      list-style: none;
    }}
    .stats li, .filters button {{
      background: rgba(255,255,255,0.78);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 10px 14px;
    }}
    .filters button {{
      cursor: pointer;
      font: inherit;
    }}
    .legend__item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      background: rgba(255,255,255,0.72);
      border-radius: 999px;
      border: 1px solid var(--border);
    }}
    .legend__swatch {{
      width: 14px;
      height: 14px;
      border-radius: 50%;
      border: 1px solid rgba(0,0,0,0.15);
    }}
    .legend__swatch--red {{ background: var(--red-bg); }}
    .legend__swatch--blue {{ background: var(--blue-bg); }}
    .legend__swatch--orange {{ background: var(--orange-bg); }}
    .legend__swatch--white {{ background: var(--white-bg); }}
    .legend__swatch--green {{ background: var(--green-bg); }}
    .legend__swatch--teal {{ background: var(--teal-bg); }}
    .timeline {{
      margin-top: 24px;
      display: grid;
      gap: 16px;
    }}
    .item {{
      display: grid;
      gap: 6px;
      animation: rise 240ms ease-out;
    }}
    .item[data-hidden="true"] {{ display: none; }}
    .meta {{
      font-size: 0.9rem;
      color: var(--muted);
      padding: 0 10px;
    }}
    .bubble {{
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 14px 16px;
      box-shadow: 0 10px 20px rgba(76, 50, 31, 0.08);
    }}
    .bubble--red {{ background: var(--red-bg); }}
    .bubble--blue {{ background: var(--blue-bg); }}
    .bubble--orange {{ background: var(--orange-bg); }}
    .bubble--white {{ background: var(--white-bg); }}
    .bubble--green {{ background: var(--green-bg); }}
    .bubble--teal {{ background: var(--teal-bg); }}
    .bubble--special {{ background: var(--special-bg); }}
    .bubble--system {{ background: var(--system-bg); }}
    .title {{
      font-weight: 700;
      margin-bottom: 4px;
    }}
    .body {{
      line-height: 1.45;
    }}
    details {{
      margin-top: 10px;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin: 10px 0 0;
      font-size: 0.84rem;
    }}
    @keyframes rise {{
      from {{ opacity: 0; transform: translateY(8px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{game_id}</h1>
      <p>Replay transcript for a completed catan-bench run.</p>
      <ul class="stats">
        <li><strong>Winners:</strong> {escape(winner_ids)}</li>
        <li><strong>Turns:</strong> {num_turns}</li>
        <li><strong>Decisions:</strong> {total_decisions}</li>
        <li><strong>Messages:</strong> {len(timeline)}</li>
      </ul>
      <ul class="legend">{player_legend}</ul>
      <div class="filters">
        <button type="button" data-filter="all">Show All</button>
        <button type="button" data-filter="player">Players Only</button>
        <button type="button" data-filter="event">Events Only</button>
      </div>
    </section>
    <section class="timeline" id="timeline">
      {bubbles}
    </section>
  </main>
  <script id="timeline-data" type="application/json">{timeline_json}</script>
  <script>
    const buttons = document.querySelectorAll('[data-filter]');
    const items = document.querySelectorAll('.item');
    for (const button of buttons) {{
      button.addEventListener('click', () => {{
        const filter = button.dataset.filter;
        for (const item of items) {{
          const type = item.dataset.speakerType;
          item.dataset.hidden = filter !== 'all' && filter !== type ? 'true' : 'false';
        }}
      }});
    }}
  </script>
</body>
</html>
"""


def _render_item(item: ReplayTimelineItem) -> str:
    meta_bits = [
        f"Turn {item.turn_index}",
        item.phase.replace("_", " "),
    ]
    if item.decision_index is not None:
        meta_bits.append(f"Decision {item.decision_index}")
    meta = " · ".join(meta_bits)
    payload = escape(json.dumps(item.raw_payload, indent=2, sort_keys=True))
    return f"""
    <article class="item" data-speaker-type="{escape(item.speaker_type)}" data-event-kind="{escape(item.event_kind)}" data-hidden="false">
      <div class="meta">{escape(meta)}</div>
      <div class="bubble bubble--{escape(item.variant)}">
        <div class="title">{escape(item.title)}</div>
        <div class="body">{escape(item.body)}</div>
        <details>
          <summary>Raw event</summary>
          <pre>{payload}</pre>
        </details>
      </div>
    </article>"""


def _as_dict(value: JsonValue | None) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def _as_optional_str(value: JsonValue | None) -> str | None:
    return value if isinstance(value, str) else None


def _as_optional_int(value: JsonValue | None) -> int | None:
    return value if isinstance(value, int) else None


def _titleize(value: str) -> str:
    return value.replace("_", " ").title()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a replay HTML artifact for a run.")
    parser.add_argument("run_dir", help="Run directory containing metadata.json and public_history.jsonl.")
    parser.add_argument(
        "--output",
        help="Optional output file path. Defaults to <run_dir>/replay.html.",
    )
    args = parser.parse_args(argv)
    output = export_replay_html(args.run_dir, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
