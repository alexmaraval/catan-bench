from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
    "trade_chat_opened": "Trade Chat Opened",
    "trade_chat_message": "Trade Chat",
    "trade_chat_quote_selected": "Trade Quote Selected",
    "trade_chat_no_deal": "Trade Chat No Deal",
    "trade_chat_closed": "Trade Chat Closed",
    "turn_ended": "Turn Ended",
    "development_card_played": "Development Card Played",
    "action_taken": "Action",
    "player_decision": "Player Decision",
    "trade_offer_received": "Private Trade Alert",
    "resource_delta": "Resource Change",
    "private_state_changed": "Private State Changed",
    "memory_note": "Memory Note",
}
SYSTEM_EVENT_KINDS = {"trade_confirmed", "trade_chat_opened", "trade_chat_no_deal", "trade_chat_closed"}
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
    stream: str = "public"


def build_replay_timeline(run_dir: str | Path) -> list[ReplayTimelineItem]:
    run_path = Path(run_dir)
    metadata = _read_json(run_path / "metadata.json")
    events = _read_jsonl(run_path / "public_history.jsonl")
    player_ids = _player_ids_from_metadata(metadata)
    color_by_player = _player_colors(player_ids)
    return [
        _timeline_item_from_event(
            event,
            color_by_player=color_by_player,
            stream="public",
        )
        for event in events
    ]


def build_player_replay_timeline(
    run_dir: str | Path, player_id: str
) -> list[ReplayTimelineItem]:
    run_path = Path(run_dir)
    metadata = _read_json(run_path / "metadata.json")
    player_ids = _player_ids_from_metadata(metadata)
    if player_id not in player_ids:
        raise ValueError(f"Unknown player_id {player_id!r} for run {run_path}.")

    color_by_player = _player_colors(player_ids)
    combined: list[tuple[tuple[int, int, int, int], ReplayTimelineItem]] = []

    for index, event in enumerate(_read_jsonl(run_path / "public_history.jsonl")):
        combined.append(
            (
                _timeline_sort_key(
                    turn_index=int(event.get("turn_index", 0)),
                    decision_index=_as_optional_int(event.get("decision_index")),
                    stream_rank=0,
                    index=index,
                ),
                _timeline_item_from_event(
                    event,
                    color_by_player=color_by_player,
                    stream="public",
                ),
            )
        )

    for index, event in enumerate(
        _read_jsonl(run_path / "players" / player_id / "private_history.jsonl")
    ):
        combined.append(
            (
                _timeline_sort_key(
                    turn_index=int(event.get("turn_index", 0)),
                    decision_index=_as_optional_int(event.get("decision_index")),
                    stream_rank=1,
                    index=index,
                ),
                _timeline_item_from_private_event(
                    event,
                    player_id=player_id,
                ),
            )
        )

    for index, entry in enumerate(_read_memory_entries(run_path, player_id)):
        combined.append(
            (
                _timeline_sort_key(
                    turn_index=int(entry.get("turn_index", 0)),
                    decision_index=_as_optional_int(entry.get("decision_index")),
                    stream_rank=2,
                    index=index,
                ),
                _timeline_item_from_memory_entry(entry, player_id=player_id),
            )
        )

    combined.sort(key=lambda pair: pair[0])
    return [item for _, item in combined]


def export_replay_html(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    run_path = Path(run_dir)
    metadata = _read_json(run_path / "metadata.json")
    timeline = build_replay_timeline(run_path)
    result = _load_result(run_path, timeline=timeline)
    player_ids = _player_ids_from_metadata(metadata)
    output = Path(output_path) if output_path is not None else run_path / "replay.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _render_html(
            metadata=metadata,
            result=result,
            timeline=timeline,
            page_title=f"{metadata.get('game_id', 'unknown-game')} Replay",
            subtitle=_subtitle_for_result(
                result,
                default_finished="Public replay transcript for the full game.",
                default_in_progress="Public replay transcript for an in-progress game.",
            ),
            nav_links=_nav_links(player_ids=player_ids, active_player_id=None),
            filter_buttons=(
                ("Show All", "all"),
                ("Player Bubbles", "player"),
                ("Event Bubbles", "event"),
            ),
            filter_attr="speaker-type",
        ),
        encoding="utf-8",
    )

    for player_id in player_ids:
        export_player_replay_html(run_path, player_id)

    return output


def export_player_replay_html(
    run_dir: str | Path,
    player_id: str,
    output_path: str | Path | None = None,
) -> Path:
    run_path = Path(run_dir)
    metadata = _read_json(run_path / "metadata.json")
    timeline = build_player_replay_timeline(run_path, player_id)
    result = _load_result(run_path, timeline=timeline)
    player_ids = _player_ids_from_metadata(metadata)

    output = (
        Path(output_path)
        if output_path is not None
        else run_path / "players" / player_id / "replay.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _render_html(
            metadata=metadata,
            result=result,
            timeline=timeline,
            page_title=f"{metadata.get('game_id', 'unknown-game')} · {player_id} Personal Replay",
            subtitle=_subtitle_for_result(
                result,
                default_finished=(
                    f"Combined public history, private history, and memory log for {player_id}."
                ),
                default_in_progress=(
                    f"Combined public history, private history, and memory log for {player_id} "
                    "while the game is still in progress."
                ),
            ),
            nav_links=_nav_links(player_ids=player_ids, active_player_id=player_id),
            filter_buttons=(
                ("Show All", "all"),
                ("Public", "public"),
                ("Private", "private"),
                ("Memory", "memory"),
            ),
            filter_attr="stream",
        ),
        encoding="utf-8",
    )
    return output


def _read_json(path: Path) -> dict[str, JsonValue]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_result(run_path: Path, *, timeline: list[ReplayTimelineItem]) -> dict[str, JsonValue]:
    result_path = run_path / "result.json"
    if result_path.exists():
        result = _read_json(result_path)
        metadata = _as_dict(result.get("metadata"))
        metadata.setdefault("status", "finished")
        result["metadata"] = metadata
        return result

    max_turn = max((item.turn_index for item in timeline), default=0)
    max_decision = max(
        (
            item.decision_index
            for item in timeline
            if item.decision_index is not None
        ),
        default=-1,
    )
    return {
        "game_id": str(run_path.name),
        "winner_ids": [],
        "total_decisions": max_decision + 1 if max_decision >= 0 else 0,
        "metadata": {
            "num_turns": max_turn,
            "status": "in_progress",
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, JsonValue]]:
    entries: list[dict[str, JsonValue]] = []
    if not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def _read_memory_entries(run_path: Path, player_id: str) -> list[dict[str, JsonValue]]:
    trace_path = run_path / "players" / player_id / "memory_trace.jsonl"
    if trace_path.exists():
        return _read_jsonl(trace_path)

    snapshot_path = run_path / "players" / player_id / "memory.json"
    if snapshot_path.exists():
        snapshot = _read_json(snapshot_path)
        if "content" in snapshot:
            return [snapshot]
    return []


def _player_ids_from_metadata(metadata: dict[str, JsonValue]) -> list[str]:
    return [str(player_id) for player_id in metadata.get("player_ids", [])]


def _player_colors(player_ids: Iterable[str]) -> dict[str, str]:
    return {
        player_id: PLAYER_PALETTE[index % len(PLAYER_PALETTE)]
        for index, player_id in enumerate(player_ids)
    }


def _subtitle_for_result(
    result: dict[str, JsonValue], *, default_finished: str, default_in_progress: str
) -> str:
    metadata = _as_dict(result.get("metadata"))
    if metadata.get("status") == "in_progress":
        return default_in_progress
    return default_finished


def _timeline_sort_key(
    *, turn_index: int, decision_index: int | None, stream_rank: int, index: int
) -> tuple[int, int, int, int]:
    return (turn_index, decision_index if decision_index is not None else 10**9, stream_rank, index)


def _timeline_item_from_event(
    event: dict[str, JsonValue],
    *,
    color_by_player: dict[str, str],
    stream: str,
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
        body=_format_public_event_body(kind=kind, payload=payload, actor_player_id=actor_player_id),
        turn_index=int(event.get("turn_index", 0)),
        phase=str(event.get("phase", "unknown")),
        decision_index=_as_optional_int(event.get("decision_index")),
        event_kind=kind,
        raw_payload=payload,
        actor_player_id=actor_player_id,
        stream=stream,
    )


def _timeline_item_from_private_event(
    event: dict[str, JsonValue], *, player_id: str
) -> ReplayTimelineItem:
    payload = _as_dict(event.get("payload"))
    kind = str(event["kind"])
    return ReplayTimelineItem(
        speaker_type="private",
        speaker_id=player_id,
        variant="private",
        title=EVENT_KIND_TITLES.get(kind, _titleize(kind)),
        body=_format_private_event_body(kind=kind, payload=payload),
        turn_index=int(event.get("turn_index", 0)),
        phase=str(event.get("phase", "unknown")),
        decision_index=_as_optional_int(event.get("decision_index")),
        event_kind=kind,
        raw_payload=payload,
        actor_player_id=_as_optional_str(event.get("actor_player_id")),
        stream="private",
    )


def _timeline_item_from_memory_entry(
    entry: dict[str, JsonValue], *, player_id: str
) -> ReplayTimelineItem:
    content = entry.get("content")
    update_kind = _as_optional_str(entry.get("update_kind")) or "memory"
    payload = {
        "content": content,
        "tags": entry.get("tags", []),
        "update_kind": update_kind,
    }
    return ReplayTimelineItem(
        speaker_type="memory",
        speaker_id=player_id,
        variant="memory",
        title=f"{player_id} Memory ({update_kind})",
        body=_format_memory_content(content),
        turn_index=int(entry.get("turn_index", 0)),
        phase=str(entry.get("phase", "unknown")),
        decision_index=_as_optional_int(entry.get("decision_index")),
        event_kind="memory_note",
        raw_payload=payload,
        actor_player_id=player_id,
        stream="memory",
    )


def _format_public_event_body(
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
        return f"{offering_player} traded {offer} with {accepting_player} for {request}."
    if kind == "trade_cancelled":
        offering_player = _as_optional_str(payload.get("offering_player_id")) or actor_player_id
        return f"{offering_player} cancelled the current trade offer."
    if kind == "trade_chat_opened":
        owner_player_id = _as_optional_str(payload.get("owner_player_id")) or actor_player_id
        requested = _format_resource_map(_as_dict(payload.get("requested_resources")))
        message = _as_optional_str(payload.get("message"))
        if message:
            return f"{owner_player_id} opened a trade chat for {requested}: {message}"
        return f"{owner_player_id} opened a trade chat for {requested}."
    if kind == "trade_chat_message":
        message = _as_optional_str(payload.get("message"))
        offer = _as_dict(payload.get("offer"))
        request = _as_dict(payload.get("request"))
        if offer and request and message:
            return f"{message} Offer: {_format_resource_map(offer)} for {_format_resource_map(request)}."
        if offer and request:
            return (
                f"Quoted {_format_resource_map(offer)} for {_format_resource_map(request)}."
            )
        if message:
            return message
        return "Spoke in trade chat."
    if kind == "trade_chat_quote_selected":
        owner_player_id = _as_optional_str(payload.get("owner_player_id")) or actor_player_id
        selected_player_id = _as_optional_str(payload.get("selected_player_id"))
        offer = _format_resource_map(_as_dict(payload.get("offer")))
        request = _format_resource_map(_as_dict(payload.get("request")))
        return f"{owner_player_id} selected {selected_player_id}'s quote: {offer} for {request}."
    if kind == "trade_chat_no_deal":
        owner_player_id = _as_optional_str(payload.get("owner_player_id")) or actor_player_id
        message = _as_optional_str(payload.get("message"))
        if message:
            return f"{owner_player_id} passed on all quotes: {message}"
        return f"{owner_player_id} passed on all quotes."
    if kind == "trade_chat_closed":
        owner_player_id = _as_optional_str(payload.get("owner_player_id")) or actor_player_id
        return f"{owner_player_id}'s trade chat closed."
    if kind == "dice_rolled":
        result = payload.get("result")
        if isinstance(result, (int, float)):
            return f"Rolled {result}."
        if isinstance(result, list) and all(isinstance(v, (int, float)) for v in result):
            total = sum(int(v) for v in result)
            breakdown = " + ".join(str(v) for v in result)
            return f"Rolled {total} ({breakdown})."
        if isinstance(result, dict):
            total = result.get("total")
            values = result.get("values") or result.get("dice") or result.get("rolls")
            if isinstance(total, int) and isinstance(values, list):
                breakdown = " + ".join(str(v) for v in values)
                return f"Rolled {total} ({breakdown})."
            if isinstance(total, int):
                return f"Rolled {total}."
        return f"Rolled {json.dumps(result, sort_keys=True)}."
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


def _format_private_event_body(*, kind: str, payload: dict[str, JsonValue]) -> str:
    if kind == "player_decision":
        action = _as_dict(payload.get("action"))
        action_type = _as_optional_str(action.get("action_type")) or "UNKNOWN_ACTION"
        description = _as_optional_str(action.get("description"))
        if description:
            return f"Chose {action_type}: {description}"
        return f"Chose {action_type}."
    if kind == "trade_offer_received":
        return f"Received a trade offer from {payload.get('from')}."
    if kind == "resource_delta":
        parts = []
        for resource, delta in sorted(payload.items()):
            if isinstance(delta, int):
                prefix = "+" if delta > 0 else ""
                parts.append(f"{resource}: {prefix}{delta}")
        return "Resources changed: " + ", ".join(parts) + "." if parts else "Resources changed."
    if kind == "private_state_changed":
        diffs: list[str] = []
        _collect_state_diffs(payload, [], diffs)
        return "State: " + ", ".join(diffs) + "." if diffs else "Private state changed."
    return json.dumps(payload, sort_keys=True)


def _collect_state_diffs(obj: JsonValue, path: list[str], out: list[str]) -> None:
    if isinstance(obj, dict):
        if "before" in obj and "after" in obj:
            label = ".".join(path) if path else "value"
            out.append(f"{label}: {obj['before']} → {obj['after']}")
        else:
            for key, value in sorted(obj.items()):
                _collect_state_diffs(value, path + [key], out)


def _format_memory_content(content: JsonValue) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return json.dumps(content, sort_keys=True)
    if isinstance(content, dict):
        if len(content) == 1:
            key, value = next(iter(content.items()))
            return f"{key}: {json.dumps(value, sort_keys=True)}"
        return json.dumps(content, sort_keys=True)
    return json.dumps(content, sort_keys=True)


def _format_resource_map(resource_map: dict[str, JsonValue]) -> str:
    if not resource_map:
        return "nothing"
    parts = []
    for resource, amount in sorted(resource_map.items()):
        name = str(resource).lower()
        if isinstance(amount, int) and amount != 1:
            name = name + "s"
        parts.append(f"{amount} {name}")
    return ", ".join(parts)


def _nav_links(*, player_ids: list[str], active_player_id: str | None) -> list[tuple[str, str, bool]]:
    links: list[tuple[str, str, bool]] = []
    if active_player_id is None:
        links.append(("Public Replay", "replay.html", True))
        for player_id in player_ids:
            links.append((f"{player_id} Personal View", f"players/{player_id}/replay.html", False))
        return links

    links.append(("Public Replay", "../../replay.html", False))
    for player_id in player_ids:
        links.append(
            (
                f"{player_id} Personal View",
                f"../{player_id}/replay.html",
                player_id == active_player_id,
            )
        )
    return links


def _render_html(
    *,
    metadata: dict[str, JsonValue],
    result: dict[str, JsonValue],
    timeline: list[ReplayTimelineItem],
    page_title: str,
    subtitle: str,
    nav_links: list[tuple[str, str, bool]],
    filter_buttons: tuple[tuple[str, str], ...],
    filter_attr: str,
) -> str:
    game_id = escape(str(metadata.get("game_id", "unknown-game")))
    player_ids = _player_ids_from_metadata(metadata)
    winner_ids = ", ".join(str(player_id) for player_id in result.get("winner_ids", [])) or "None"
    total_decisions = escape(str(result.get("total_decisions", "unknown")))
    num_turns = escape(
        str(result.get("metadata", {}).get("num_turns", result.get("num_turns", "unknown")))
    )

    bubbles = "\n".join(_render_item(item) for item in timeline)
    player_legend = "\n".join(
        f'<li class="legend__item" data-player-id="{escape(player_id)}"><span class="legend__swatch legend__swatch--{escape(color)}"></span>{escape(player_id)}</li>'
        for player_id, color in _player_colors(player_ids).items()
    )
    nav = "\n".join(
        f'<a class="nav__link{" nav__link--active" if active else ""}" href="{escape(href)}">{escape(label)}</a>'
        for label, href, active in nav_links
    )
    filters = "\n".join(
        f'<button type="button" data-filter-attr="{escape(filter_attr)}" data-filter-value="{escape(value)}">{escape(label)}</button>'
        for label, value in filter_buttons
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(page_title)}</title>
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
      --private-bg: #e2dcf5;
      --memory-bg: #d9f0d2;
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
    .stats, .legend, .filters, .nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      padding: 0;
      margin: 18px 0 0;
      list-style: none;
    }}
    .stats li, .filters button, .nav__link {{
      background: rgba(255,255,255,0.78);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 10px 14px;
    }}
    .filters button {{
      cursor: pointer;
      font: inherit;
      transition: background 120ms, color 120ms, border-color 120ms;
    }}
    .filters button.filter--active {{
      background: var(--ink);
      color: var(--panel);
      border-color: var(--ink);
    }}
    .nav__link {{
      color: inherit;
      text-decoration: none;
    }}
    .nav__link--active {{
      background: #e6ddcf;
      font-weight: 700;
    }}
    .legend__item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      background: rgba(255,255,255,0.72);
      border-radius: 999px;
      border: 1px solid var(--border);
      cursor: pointer;
      user-select: none;
      transition: background 120ms, color 120ms, border-color 120ms;
    }}
    .legend__item--active {{
      background: var(--ink);
      color: var(--panel);
      border-color: var(--ink);
    }}
    .legend__item--active .legend__swatch {{
      border-color: rgba(255,255,255,0.4);
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
    .bubble--private {{ background: var(--private-bg); }}
    .bubble--memory {{ background: var(--memory-bg); }}
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
    details summary {{
      cursor: pointer;
      font-size: 0.88rem;
      font-weight: 600;
      color: var(--muted);
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin: 8px 0 0;
      font-size: 0.84rem;
    }}
    .memory-note {{
      margin-top: 10px;
      padding-top: 8px;
      border-top: 1px solid var(--border);
      font-size: 0.88rem;
      font-style: italic;
      color: var(--muted);
    }}
    .memory-note__label {{
      font-style: normal;
      font-weight: 700;
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
      <h1>{escape(page_title)}</h1>
      <p>{escape(subtitle)}</p>
      <div class="nav">{nav}</div>
      <ul class="stats">
        <li><strong>Game:</strong> {game_id}</li>
        <li><strong>Winners:</strong> {escape(winner_ids)}</li>
        <li><strong>Turns:</strong> {num_turns}</li>
        <li><strong>Decisions:</strong> {total_decisions}</li>
        <li><strong>Messages:</strong> {len(timeline)}</li>
      </ul>
      <ul class="legend">{player_legend}</ul>
      <div class="filters">{filters}</div>
    </section>
    <section class="timeline" id="timeline">
      {bubbles}
    </section>
  </main>
  <script>
    const buttons = document.querySelectorAll('[data-filter-value]');
    const items = document.querySelectorAll('.item');
    const legendItems = document.querySelectorAll('.legend__item[data-player-id]');

    function applyFilter(attr, value) {{
      for (const item of items) {{
        if (value === 'all') {{
          item.dataset.hidden = 'false';
        }} else {{
          const itemValue = item.getAttribute('data-' + attr);
          item.dataset.hidden = itemValue === value ? 'false' : 'true';
        }}
      }}
    }}

    function resetToShowAll() {{
      for (const b of buttons) b.classList.remove('filter--active');
      for (const l of legendItems) l.classList.remove('legend__item--active');
      if (buttons.length > 0) buttons[0].classList.add('filter--active');
      applyFilter('all', 'all');
    }}

    if (buttons.length > 0) buttons[0].classList.add('filter--active');

    for (const button of buttons) {{
      button.addEventListener('click', () => {{
        for (const b of buttons) b.classList.remove('filter--active');
        for (const l of legendItems) l.classList.remove('legend__item--active');
        button.classList.add('filter--active');
        applyFilter(button.dataset.filterAttr, button.dataset.filterValue);
      }});
    }}

    for (const li of legendItems) {{
      li.addEventListener('click', () => {{
        const isActive = li.classList.contains('legend__item--active');
        if (isActive) {{
          resetToShowAll();
        }} else {{
          for (const b of buttons) b.classList.remove('filter--active');
          for (const l of legendItems) l.classList.remove('legend__item--active');
          li.classList.add('legend__item--active');
          applyFilter('actor-player-id', li.dataset.playerId);
        }}
      }});
    }}
  </script>
</body>
</html>
"""


def _render_item(item: ReplayTimelineItem) -> str:
    meta_bits = [f"Turn {item.turn_index}", item.phase.replace("_", " ").title()]
    if item.decision_index is not None:
        meta_bits.append(f"Decision {item.decision_index}")
    meta = " · ".join(meta_bits)
    payload_json = escape(json.dumps(item.raw_payload, indent=2, sort_keys=True))

    extra = ""
    if item.event_kind == "player_decision":
        memory_write = item.raw_payload.get("memory_write")
        reasoning = item.raw_payload.get("reasoning")
        decision_prompt = item.raw_payload.get("decision_prompt")
        if memory_write is not None:
            extra += f'\n        <div class="memory-note"><span class="memory-note__label">Memory note:</span> {escape(str(memory_write))}</div>'
        if reasoning is not None:
            extra += f'\n        <details>\n          <summary>Reasoning</summary>\n          <pre>{escape(str(reasoning))}</pre>\n        </details>'
        if decision_prompt is not None:
            extra += f'\n        <details>\n          <summary>Decision prompt</summary>\n          <pre>{escape(str(decision_prompt))}</pre>\n        </details>'

    return f"""
    <article class="item" data-speaker-type="{escape(item.speaker_type)}" data-stream="{escape(item.stream)}" data-actor-player-id="{escape(item.actor_player_id or '')}" data-hidden="false">
      <div class="meta">{escape(meta)}</div>
      <div class="bubble bubble--{escape(item.variant)}">
        <div class="title">{escape(item.title)}</div>
        <div class="body">{escape(item.body)}</div>{extra}
        <details>
          <summary>Raw entry</summary>
          <pre>{payload_json}</pre>
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
    parser = argparse.ArgumentParser(description="Export replay HTML artifacts for a run.")
    parser.add_argument("run_dir", help="Run directory containing metadata.json and history logs.")
    parser.add_argument(
        "--output",
        help="Optional output file path for the public replay. Defaults to <run_dir>/replay.html.",
    )
    args = parser.parse_args(argv)
    output = export_replay_html(args.run_dir, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
