from __future__ import annotations

import json
from pathlib import Path
import re

try:
    from jinja2 import Environment, FileSystemLoader
except ModuleNotFoundError:  # pragma: no cover - exercised in lean test envs.
    Environment = None
    FileSystemLoader = None


_RESOURCE_ORDER = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
_DEV_CARD_ORDER = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT")
# Keys that are harness metadata or already in the system prompt — excluded from payload_json
_NOISE_KEYS = frozenset({"game_rules", "game_id", "history_index", "phase", "decision_index"})


def fmt_resources(resources: object) -> str:
    """Compact human-readable resource summary, e.g. '2×WOOD, 1×ORE'."""
    if not isinstance(resources, dict):
        return "none"
    parts = [f"{resources[r]}×{r}" for r in _RESOURCE_ORDER if resources.get(r)]
    return ", ".join(parts) or "none"


def fmt_dev_cards(dev_cards: object) -> str:
    """Compact non-zero dev card summary, e.g. '2×KNIGHT, 1×MONOPOLY'."""
    if not isinstance(dev_cards, dict):
        return "none"
    parts = [f"{dev_cards[c]}×{c}" for c in _DEV_CARD_ORDER if dev_cards.get(c)]
    return ", ".join(parts) or "none"


def fmt_pieces(private_state: object) -> str:
    """'10R / 3S / 4C' from private_state dict, gracefully handling missing keys."""
    if not isinstance(private_state, dict):
        return ""
    pieces = private_state.get("pieces") or {}
    roads = pieces.get("roads", pieces.get("roads_available", "?"))
    settlements = pieces.get("settlements", pieces.get("settlements_available", "?"))
    cities = pieces.get("cities", pieces.get("cities_available", "?"))
    return f"{roads}R / {settlements}S / {cities}C"


def fmt_vp(private_state: object) -> str:
    """'5 (4 visible)' from private_state dict."""
    if not isinstance(private_state, dict):
        return "?"
    vp = private_state.get("victory_points") or {}
    actual = vp.get("actual", "?")
    visible = vp.get("visible", "?")
    return f"{actual} ({visible} visible)"


def fmt_event(event: object) -> str:
    """Single-line prose description of a public event dict."""
    if not isinstance(event, dict):
        return str(event)
    kind = str(event.get("kind", "unknown"))
    actor = str(event.get("actor_player_id") or "SYSTEM")
    p = event.get("payload") or {}
    if not isinstance(p, dict):
        p = {}

    if kind == "dice_rolled":
        return f"{actor} rolled {p.get('result', '?')}"
    if kind == "settlement_built":
        return f"{actor} built settlement at node {p.get('node_id', '?')}"
    if kind == "city_built":
        return f"{actor} upgraded to city at node {p.get('node_id', '?')}"
    if kind == "road_built":
        return f"{actor} built road on edge {p.get('edge', '?')}"
    if kind == "robber_moved":
        coord = p.get("coordinate", "?")
        victim = p.get("victim")
        if victim:
            return f"{actor} moved robber to {coord}, stole from {victim}"
        return f"{actor} moved robber to {coord}"
    if kind == "turn_ended":
        return f"{actor} ended turn"
    if kind == "development_card_played":
        card = p.get("card_type") or p.get("dev_card_type") or "dev card"
        return f"{actor} played {card}"
    if kind == "action_taken":
        return f"{actor}: {p.get('action_type', kind)}"
    if kind == "trade_offered":
        offer = fmt_resources(p.get("offer", {}))
        req = fmt_resources(p.get("request", {}))
        return f"{actor} offered {offer} for {req}"
    if kind == "trade_accepted":
        return f"{actor} accepted trade"
    if kind == "trade_rejected":
        return f"{actor} rejected trade"
    if kind == "trade_confirmed":
        offer = fmt_resources(p.get("offer", {}))
        req = fmt_resources(p.get("request", {}))
        accepting = p.get("accepting_player_id", "?")
        return f"{actor} confirmed trade with {accepting}: gave {offer}, received {req}"
    if kind == "trade_cancelled":
        return f"{actor} cancelled trade"
    if kind == "trade_chat_opened":
        req = fmt_resources(p.get("requested_resources", {}))
        return f"{actor} opened trade chat requesting {req}"
    if kind == "trade_chat_message":
        msg = p.get("message", "")
        return f"{actor}: {msg}" if msg else f"{actor} sent trade quote"
    if kind == "trade_chat_closed":
        outcome = p.get("outcome", "?")
        return f"{actor} closed trade chat (outcome: {outcome})"
    return f"{actor}: {kind.replace('_', ' ')}"


class PromptRenderer:
    """Renders prompt stages from package-local Jinja templates."""

    def __init__(self, templates_dir: str | Path | None = None) -> None:
        if templates_dir is None:
            templates_dir = Path(__file__).with_name("templates")
        self._templates_dir = Path(templates_dir)
        self._env = None
        if Environment is not None and FileSystemLoader is not None:
            env = Environment(
                loader=FileSystemLoader(str(self._templates_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
            )
            env.globals["fmt_resources"] = fmt_resources
            env.globals["fmt_dev_cards"] = fmt_dev_cards
            env.globals["fmt_pieces"] = fmt_pieces
            env.globals["fmt_vp"] = fmt_vp
            env.globals["fmt_event"] = fmt_event
            self._env = env

    def render_messages(
        self,
        *,
        system_template: str,
        user_template: str,
        payload: dict[str, object],
        **context: object,
    ) -> list[dict[str, object]]:
        clean_payload = {k: v for k, v in payload.items() if k not in _NOISE_KEYS}
        template_context = {
            **context,
            "payload": payload,
            "payload_json": json.dumps(clean_payload, sort_keys=True, indent=2),
        }
        system_content = self.render(system_template, **template_context)
        user_content = self.render(user_template, **template_context)
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def render(self, template_name: str, **context: object) -> str:
        if self._env is not None:
            return self._env.get_template(template_name).render(**context).strip()
        return self._render_without_jinja(template_name, context).strip()

    def _render_without_jinja(self, template_name: str, context: dict[str, object]) -> str:
        template_path = self._templates_dir / template_name
        template = template_path.read_text(encoding="utf-8")
        template = re.sub(
            r'{%\s*include\s+"([^"]+)"\s*%}',
            lambda match: self._render_without_jinja(match.group(1), context),
            template,
        )
        template = re.sub(
            r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}",
            lambda match: str(context.get(match.group(1), "")),
            template,
        )
        return template
