"""Tests for the post-game analysis module."""

from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from catan_bench.analysis import (
    analyze_game,
    compute_market_analysis,
    compute_building_timeline,
    compute_decision_quality,
    compute_dev_card_analysis,
    compute_discard_analysis,
    compute_game_summary,
    compute_phase_analysis,
    compute_resource_production,
    compute_robber_analysis,
    compute_strategy_evolution,
    compute_trade_analysis,
    compute_trade_chat_analysis,
    compute_turn_progress_metrics,
    compute_vp_progression,
    discover_completed_run_directories,
    main as analysis_main,
    PIPS,
)
from catan_bench.schemas import (
    Event,
    MemorySnapshot,
    PlayerMemory,
    PromptTrace,
    PromptTraceAttempt,
    PublicStateSnapshot,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _event(
    kind: str,
    *,
    actor: str | None = None,
    turn: int = 0,
    history: int = 1,
    phase: str = "play_turn",
    **payload,
) -> Event:
    return Event(
        kind=kind,
        actor_player_id=actor,
        turn_index=turn,
        history_index=history,
        phase=phase,
        payload=dict(payload),
    )


def _snapshot(
    turn: int,
    history: int,
    *,
    phase: str = "play_turn",
    players: dict | None = None,
    board: dict | None = None,
) -> PublicStateSnapshot:
    public_state: dict = {}
    if players:
        public_state["players"] = players
    if board:
        public_state["board"] = board
    return PublicStateSnapshot(
        history_index=history,
        turn_index=turn,
        phase=phase,
        decision_index=None,
        public_state=public_state,
    )


def _prompt_trace(player_id: str, *, attempts: int = 1, turn: int = 0) -> PromptTrace:
    dummy_attempt = PromptTraceAttempt(messages=(), response_text="ok", response={})
    return PromptTrace(
        player_id=player_id,
        history_index=0,
        turn_index=turn,
        phase="play_turn",
        decision_index=0,
        stage="choose_action",
        model="test-model",
        temperature=0.5,
        attempts=tuple(dummy_attempt for _ in range(attempts)),
    )


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


# ── Game summary tests ────────────────────────────────────────────────────────


class TestComputeGameSummary(unittest.TestCase):
    def test_basic_counts(self) -> None:
        events = [
            _event("dice_rolled"),
            _event("settlement_built"),
            _event("trade_offered"),
            _event("trade_confirmed"),
            _event("turn_ended"),
        ]
        result = {
            "winner_ids": ["RED"],
            "metadata": {"num_turns": 10},
        }
        summary = compute_game_summary(
            result=result,
            events=events,
            num_turns=10,
            total_decisions=20,
        )
        self.assertEqual(summary["num_turns"], 10)
        self.assertEqual(summary["total_events"], 5)
        self.assertEqual(summary["total_decisions"], 20)
        self.assertAlmostEqual(summary["events_per_turn"], 0.5)
        self.assertAlmostEqual(summary["decisions_per_turn"], 2.0)

    def test_trade_activity_rate(self) -> None:
        events = [
            _event("trade_offered"),
            _event("trade_accepted"),
            _event("dice_rolled"),
            _event("dice_rolled"),
        ]
        summary = compute_game_summary(
            result={"winner_ids": []},
            events=events,
            num_turns=5,
            total_decisions=10,
        )
        self.assertAlmostEqual(summary["trade_activity_rate"], 0.5)

    def test_trade_efficiency(self) -> None:
        events = [
            _event("trade_offered"),
            _event("trade_offered"),
            _event("trade_offered"),
            _event("trade_confirmed"),
        ]
        summary = compute_game_summary(
            result={"winner_ids": []},
            events=events,
            num_turns=1,
            total_decisions=5,
        )
        # 1 confirmed out of 3 offered — result is rounded to 4 decimal places
        self.assertAlmostEqual(summary["trade_efficiency"], 1 / 3, places=3)

    def test_no_trades_yields_zero_efficiency(self) -> None:
        summary = compute_game_summary(
            result={"winner_ids": []},
            events=[_event("dice_rolled")],
            num_turns=1,
            total_decisions=1,
        )
        self.assertEqual(summary["trade_efficiency"], 0.0)

    def test_trade_chat_no_deal_rate(self) -> None:
        events = [
            _event("trade_chat_opened"),
            _event("trade_chat_closed", outcome="selected"),
            _event("trade_chat_opened"),
            _event("trade_chat_closed", outcome="no_deal"),
        ]
        summary = compute_game_summary(
            result={"winner_ids": []},
            events=events,
            num_turns=4,
            total_decisions=8,
        )
        self.assertEqual(summary["trade_chat_rooms"], 2)
        self.assertEqual(summary["trade_chat_no_deals"], 1)
        self.assertAlmostEqual(summary["trade_chat_no_deal_rate"], 0.5)


# ── Building timeline tests ───────────────────────────────────────────────────


class TestComputeBuildingTimeline(unittest.TestCase):
    def test_filters_by_player(self) -> None:
        events = [
            _event("settlement_built", actor="RED", turn=0, node_id=5),
            _event("settlement_built", actor="BLUE", turn=1, node_id=10),
            _event("road_built", actor="RED", turn=0, edge=[5, 6]),
            _event("city_built", actor="RED", turn=5, node_id=5),
        ]
        result = compute_building_timeline("RED", events)
        self.assertEqual(result["counts"]["settlements"], 1)
        self.assertEqual(result["counts"]["cities"], 1)
        self.assertEqual(result["counts"]["roads"], 1)
        self.assertEqual(result["settlements"][0]["node_id"], 5)
        self.assertEqual(result["cities"][0]["node_id"], 5)

    def test_empty_events(self) -> None:
        result = compute_building_timeline("RED", [])
        self.assertEqual(result["counts"]["settlements"], 0)
        self.assertEqual(result["counts"]["cities"], 0)
        self.assertEqual(result["counts"]["roads"], 0)


# ── Robber analysis tests ─────────────────────────────────────────────────────


class TestComputeRobberAnalysis(unittest.TestCase):
    def test_times_moved_and_targeted(self) -> None:
        events = [
            _event("robber_moved", actor="RED", coordinate=[0, 0, 0], victim="BLUE"),
            _event("robber_moved", actor="BLUE", coordinate=[1, 0, 0], victim="RED"),
            _event("robber_moved", actor="RED", coordinate=[0, 1, 0], victim=None),
        ]
        result = compute_robber_analysis("RED", events)
        self.assertEqual(result["times_moved_robber"], 2)
        self.assertEqual(result["times_targeted"], 1)

    def test_no_robber_events(self) -> None:
        result = compute_robber_analysis("RED", [_event("dice_rolled")])
        self.assertEqual(result["times_moved_robber"], 0)
        self.assertEqual(result["times_targeted"], 0)


# ── Discard analysis tests ────────────────────────────────────────────────────


class TestComputeDiscardAnalysis(unittest.TestCase):
    def test_counts_discards_for_player(self) -> None:
        events = [
            _event("resources_discarded", actor="RED", discarded_count=4),
            _event("resources_discarded", actor="BLUE", discarded_count=5),
            _event("resources_discarded", actor="RED", discarded_count=3),
        ]
        result = compute_discard_analysis("RED", events)
        self.assertEqual(result["times_discarded"], 2)
        self.assertEqual(result["total_cards_discarded"], 7)

    def test_no_discard_events(self) -> None:
        result = compute_discard_analysis("RED", [_event("dice_rolled")])
        self.assertEqual(result["times_discarded"], 0)
        self.assertEqual(result["total_cards_discarded"], 0)

    def test_missing_discarded_count(self) -> None:
        events = [_event("resources_discarded", actor="RED")]
        result = compute_discard_analysis("RED", events)
        self.assertEqual(result["times_discarded"], 1)
        self.assertEqual(result["total_cards_discarded"], 0)


# ── Dev card analysis tests ───────────────────────────────────────────────────


class TestComputeDevCardAnalysis(unittest.TestCase):
    def test_groups_cards_by_type(self) -> None:
        events = [
            _event(
                "development_card_played",
                actor="RED",
                action={"action_type": "PLAY_KNIGHT_CARD", "payload": {}},
            ),
            _event(
                "development_card_played",
                actor="RED",
                action={"action_type": "PLAY_KNIGHT_CARD", "payload": {}},
            ),
            _event(
                "development_card_played",
                actor="RED",
                action={"action_type": "PLAY_MONOPOLY", "payload": {}},
            ),
            _event(
                "development_card_played",
                actor="BLUE",
                action={"action_type": "PLAY_KNIGHT_CARD", "payload": {}},
            ),
        ]
        result = compute_dev_card_analysis("RED", events, None)
        self.assertEqual(result["cards_played"], 3)
        self.assertEqual(result["cards_played_by_type"]["PLAY_KNIGHT_CARD"], 2)
        self.assertEqual(result["cards_played_by_type"]["PLAY_MONOPOLY"], 1)
        # BLUE's card not counted
        self.assertNotIn("BLUE", result["cards_played_by_type"].values())

    def test_cards_held_from_final_snapshot(self) -> None:
        final = _snapshot(
            10,
            100,
            players={"RED": {"development_card_count": 2, "visible_victory_points": 5}},
        )
        result = compute_dev_card_analysis("RED", [], final)
        self.assertEqual(result["cards_held_at_end"], 2)

    def test_no_events_no_snapshot(self) -> None:
        result = compute_dev_card_analysis("RED", [], None)
        self.assertEqual(result["cards_played"], 0)
        self.assertEqual(result["cards_held_at_end"], 0)


# ── Trade analysis tests ──────────────────────────────────────────────────────


class TestComputeTradeAnalysis(unittest.TestCase):
    def test_offers_made_and_received(self) -> None:
        events = [
            _event(
                "trade_offered",
                actor="RED",
                offering_player_id="RED",
                offer={"WOOD": 1},
                request={"BRICK": 1},
            ),
            _event(
                "trade_offered",
                actor="BLUE",
                offering_player_id="BLUE",
                offer={"ORE": 1},
                request={"WHEAT": 1},
            ),
        ]
        result = compute_trade_analysis("RED", events)
        self.assertEqual(result["offers_made"], 1)
        self.assertEqual(result["offers_received"], 1)

    def test_accepted_and_rejected_count(self) -> None:
        events = [
            _event(
                "trade_accepted",
                responding_player_id="RED",
                offering_player_id="BLUE",
                offer={},
                request={},
            ),
            _event(
                "trade_rejected",
                responding_player_id="RED",
                offering_player_id="BLUE",
                offer={},
                request={},
            ),
            _event(
                "trade_rejected",
                responding_player_id="RED",
                offering_player_id="BLUE",
                offer={},
                request={},
            ),
        ]
        result = compute_trade_analysis("RED", events)
        self.assertEqual(result["acceptances"], 1)
        self.assertEqual(result["rejections"], 2)
        self.assertAlmostEqual(result["acceptance_rate"], 1 / 3, places=3)

    def test_confirmed_trade_resources_as_offerer(self) -> None:
        events = [
            _event(
                "trade_confirmed",
                offering_player_id="RED",
                accepting_player_id="BLUE",
                offer={"WOOD": 2},
                request={"ORE": 1},
            ),
        ]
        result = compute_trade_analysis("RED", events)
        self.assertEqual(result["confirmations_as_offerer"], 1)
        self.assertEqual(result["resources_given"].get("WOOD"), 2)
        self.assertEqual(result["resources_received"].get("ORE"), 1)

    def test_confirmed_trade_resources_as_acceptee(self) -> None:
        events = [
            _event(
                "trade_confirmed",
                offering_player_id="BLUE",
                accepting_player_id="RED",
                offer={"WOOD": 1},
                request={"BRICK": 2},
            ),
        ]
        result = compute_trade_analysis("RED", events)
        self.assertEqual(result["confirmations_as_acceptee"], 1)
        # Acceptee gives what offerer requested, receives what offerer offered
        self.assertEqual(result["resources_given"].get("BRICK"), 2)
        self.assertEqual(result["resources_received"].get("WOOD"), 1)

    def test_net_trade_balance(self) -> None:
        events = [
            _event(
                "trade_confirmed",
                offering_player_id="RED",
                accepting_player_id="BLUE",
                offer={"WOOD": 3},
                request={"ORE": 2},
            ),
        ]
        result = compute_trade_analysis("RED", events)
        self.assertEqual(result["net_trade_balance"].get("ORE"), 2)
        self.assertEqual(result["net_trade_balance"].get("WOOD"), -3)

    def test_no_trade_events(self) -> None:
        result = compute_trade_analysis("RED", [_event("dice_rolled")])
        self.assertEqual(result["offers_made"], 0)
        self.assertEqual(result["acceptance_rate"], 0.0)


# ── Resource production tests ─────────────────────────────────────────────────


def _board_with_tile(
    node_id: int = 1,
    resource: str = "WOOD",
    number: int = 6,
    robber: list | None = None,
) -> dict:
    """Minimal board state with one tile and one node adjacent to it."""
    tile_coord = [0, 0, 0]
    return {
        "tiles": {
            "0": {
                "type": "RESOURCE_TILE",
                "resource": resource,
                "number": number,
                "coordinate": tile_coord,
            }
        },
        "nodes": {
            str(node_id): {
                "id": node_id,
                "color": "RED",
                "building": "SETTLEMENT",
            }
        },
        "edges": [],
        "adjacent_tiles": {
            str(node_id): [
                {
                    "type": "RESOURCE_TILE",
                    "resource": resource,
                    "number": number,
                    "coordinate": tile_coord,
                }
            ]
        },
        "robber_coordinate": robber or [9, 9, 9],  # Far away from tile
    }


class TestComputeResourceProduction(unittest.TestCase):
    def test_settlement_produces_one(self) -> None:
        board = _board_with_tile(node_id=1, resource="WOOD", number=6)
        snap = _snapshot(
            0, 0, board=board, players={"RED": {"visible_victory_points": 1}}
        )
        events = [_event("dice_rolled", turn=1, history=2, dice=[3, 3])]
        result = compute_resource_production("RED", events, [snap])
        self.assertEqual(result["total"].get("WOOD"), 1)

    def test_city_produces_two(self) -> None:
        board = _board_with_tile(node_id=1, resource="ORE", number=6)
        board["nodes"]["1"]["building"] = "CITY"
        snap = _snapshot(
            0, 0, board=board, players={"RED": {"visible_victory_points": 2}}
        )
        events = [_event("dice_rolled", turn=1, history=2, dice=[3, 3])]
        result = compute_resource_production("RED", events, [snap])
        self.assertEqual(result["total"].get("ORE"), 2)

    def test_skips_roll_of_seven(self) -> None:
        board = _board_with_tile(node_id=1, resource="WHEAT", number=7)
        snap = _snapshot(0, 0, board=board)
        events = [_event("dice_rolled", turn=1, history=2, dice=[3, 4])]
        result = compute_resource_production("RED", events, [snap])
        self.assertEqual(result["total"].get("WHEAT", 0), 0)

    def test_robber_blocks_production(self) -> None:
        # Robber sitting on the tile coordinate [0, 0, 0]
        board = _board_with_tile(
            node_id=1, resource="BRICK", number=6, robber=[0, 0, 0]
        )
        snap = _snapshot(
            0, 0, board=board, players={"RED": {"visible_victory_points": 1}}
        )
        events = [_event("dice_rolled", turn=1, history=2, dice=[3, 3])]
        result = compute_resource_production("RED", events, [snap])
        self.assertEqual(result["total"].get("BRICK", 0), 0)

    def test_number_mismatch_no_production(self) -> None:
        board = _board_with_tile(node_id=1, resource="SHEEP", number=8)
        snap = _snapshot(
            0, 0, board=board, players={"RED": {"visible_victory_points": 1}}
        )
        events = [_event("dice_rolled", turn=1, history=2, dice=[3, 3])]  # roll 6 != 8
        result = compute_resource_production("RED", events, [snap])
        self.assertEqual(result["total"].get("SHEEP", 0), 0)

    def test_empty_snapshots_returns_empty(self) -> None:
        events = [_event("dice_rolled", dice=[3, 3])]
        result = compute_resource_production("RED", events, [])
        self.assertEqual(result["total"], {})
        self.assertEqual(result["by_turn"], [])


# ── VP progression tests ──────────────────────────────────────────────────────


class TestComputeVpProgression(unittest.TestCase):
    def test_deduplicates_to_max_vp_per_turn(self) -> None:
        snapshots = [
            _snapshot(1, 1, players={"RED": {"visible_victory_points": 2}}),
            _snapshot(1, 2, players={"RED": {"visible_victory_points": 3}}),
            _snapshot(2, 3, players={"RED": {"visible_victory_points": 3}}),
        ]
        result = compute_vp_progression("RED", snapshots)
        turns = {entry["turn_index"]: entry["vp"] for entry in result}
        self.assertEqual(turns[1], 3)
        self.assertEqual(turns[2], 3)
        self.assertEqual(len(result), 2)

    def test_includes_hidden_dev_vp_in_total(self) -> None:
        snapshots = [
            _snapshot(
                4,
                10,
                players={
                    "RED": {
                        "visible_victory_points": 5,
                        "dev_victory_points": 2,
                    }
                },
            ),
        ]
        result = compute_vp_progression("RED", snapshots)
        self.assertEqual(result, [{"turn_index": 4, "vp": 7}])

    def test_sorted_by_turn(self) -> None:
        snapshots = [
            _snapshot(5, 10, players={"RED": {"visible_victory_points": 8}}),
            _snapshot(2, 5, players={"RED": {"visible_victory_points": 5}}),
        ]
        result = compute_vp_progression("RED", snapshots)
        self.assertEqual(result[0]["turn_index"], 2)
        self.assertEqual(result[1]["turn_index"], 5)

    def test_player_not_in_snapshot_skipped(self) -> None:
        snapshots = [
            _snapshot(1, 1, players={"BLUE": {"visible_victory_points": 2}}),
        ]
        result = compute_vp_progression("RED", snapshots)
        self.assertEqual(result, [])


# ── Phase analysis tests ──────────────────────────────────────────────────────


class TestComputePhaseAnalysis(unittest.TestCase):
    def test_opening_pip_count(self) -> None:
        board = {
            "tiles": {},
            "nodes": {},
            "edges": [],
            "adjacent_tiles": {
                "5": [
                    {"type": "RESOURCE_TILE", "resource": "WOOD", "number": 6},
                    {"type": "RESOURCE_TILE", "resource": "ORE", "number": 9},
                ],
            },
            "robber_coordinate": [9, 9, 9],
        }
        snaps = [_snapshot(0, 0, board=board)]
        events = [
            _event(
                "settlement_built",
                actor="RED",
                phase="build_initial_settlement",
                turn=0,
                node_id=5,
            ),
        ]
        vp_prog = [{"turn_index": 0, "vp": 1}]
        result = compute_phase_analysis("RED", events, snaps, vp_prog)
        # WOOD@6 = 5 pips, ORE@9 = 4 pips
        self.assertEqual(result["opening"]["pip_count"], PIPS[6] + PIPS[9])
        self.assertEqual(result["opening"]["resource_diversity"], 2)
        self.assertIn("WOOD", result["opening"]["resource_types"])
        self.assertIn("ORE", result["opening"]["resource_types"])

    def test_vp_milestones_detected(self) -> None:
        vp_prog = [
            {"turn_index": 0, "vp": 2},
            {"turn_index": 3, "vp": 3},
            {"turn_index": 5, "vp": 5},
            {"turn_index": 10, "vp": 7},
        ]
        result = compute_phase_analysis("RED", [], [], vp_prog)
        milestones = result["vp_milestones"]
        self.assertEqual(milestones["3vp_turn"], 3)
        self.assertEqual(milestones["5vp_turn"], 5)
        self.assertEqual(milestones["7vp_turn"], 10)
        self.assertIsNone(milestones["10vp_turn"])

    def test_non_initial_settlements_not_counted(self) -> None:
        events = [
            _event("settlement_built", actor="RED", phase="play_turn", node_id=5),
        ]
        result = compute_phase_analysis("RED", events, [], [])
        self.assertEqual(result["opening"]["initial_settlement_nodes"], [])


# ── Decision quality tests ────────────────────────────────────────────────────


class TestComputeDecisionQuality(unittest.TestCase):
    def test_retry_rate(self) -> None:
        traces = [
            _prompt_trace("RED", attempts=1),
            _prompt_trace("RED", attempts=3),
            _prompt_trace("RED", attempts=1),
            _prompt_trace("RED", attempts=2),
        ]
        result = compute_decision_quality(traces)
        self.assertEqual(result["total_prompts"], 4)
        self.assertEqual(result["retries"], 2)
        self.assertAlmostEqual(result["retry_rate"], 0.5)
        self.assertEqual(result["max_attempts_on_single_decision"], 3)

    def test_no_retries(self) -> None:
        traces = [_prompt_trace("RED", attempts=1)] * 5
        result = compute_decision_quality(traces)
        self.assertEqual(result["retries"], 0)
        self.assertAlmostEqual(result["retry_rate"], 0.0)

    def test_empty_traces(self) -> None:
        result = compute_decision_quality([])
        self.assertEqual(result["total_prompts"], 0)
        self.assertEqual(result["retry_rate"], 0.0)
        self.assertEqual(result["max_attempts_on_single_decision"], 0)


class TestComputeTurnProgressMetrics(unittest.TestCase):
    def test_dead_turns_and_progress_milestones(self) -> None:
        events = [
            _event(
                "turn_ended",
                actor="RED",
                turn=1,
                turn_player_id_before="RED",
            ),
            _event(
                "road_built",
                actor="RED",
                turn=5,
                turn_player_id_before="RED",
                edge=[1, 2],
            ),
            _event(
                "turn_ended",
                actor="RED",
                turn=5,
                turn_player_id_before="RED",
            ),
            _event(
                "action_taken",
                actor="RED",
                turn=9,
                turn_player_id_before="RED",
                action={"action_type": "BUY_DEVELOPMENT_CARD", "payload": {}},
            ),
            _event(
                "turn_ended",
                actor="RED",
                turn=9,
                turn_player_id_before="RED",
            ),
            _event(
                "turn_ended",
                actor="RED",
                turn=13,
                turn_player_id_before="RED",
            ),
            _event(
                "turn_ended",
                actor="RED",
                turn=17,
                turn_player_id_before="RED",
            ),
        ]
        vp_prog = [
            {"turn_index": 0, "vp": 2},
            {"turn_index": 5, "vp": 3},
            {"turn_index": 9, "vp": 5},
            {"turn_index": 13, "vp": 7},
        ]

        result = compute_turn_progress_metrics(
            "RED",
            events,
            vp_prog,
            final_vp=10,
            num_turns=17,
            vps_to_win=10,
            is_winner=True,
        )

        self.assertEqual(result["active_turns"], 5)
        self.assertEqual(result["dead_turns"], 2)
        self.assertAlmostEqual(result["dead_turn_rate"], 0.4)
        self.assertEqual(result["turns_to_first_5_vp"], 9)
        self.assertEqual(result["first_7_vp_turn"], 13)
        self.assertEqual(result["win_turn"], 17)
        self.assertEqual(result["turns_from_7_vp_to_win"], 4)


# ── Trade chat analysis tests ─────────────────────────────────────────────────


class TestComputeTradeChatAnalysis(unittest.TestCase):
    def _chat_event(
        self, kind: str, owner: str = "RED", turn: int = 1, attempt: int = 0, **extra
    ) -> Event:
        payload: dict = {"owner_player_id": owner, "attempt_index": attempt, **extra}
        return Event(
            kind=kind,
            payload=payload,
            turn_index=turn,
            history_index=0,
            phase="play_turn",
        )

    def test_rooms_opened_and_success_rate(self) -> None:
        events = [
            self._chat_event("trade_chat_opened", owner="RED", attempt=0),
            self._chat_event(
                "trade_chat_closed",
                owner="RED",
                attempt=0,
                outcome="selected",
                selected_player_id="BLUE",
            ),
            self._chat_event("trade_chat_opened", owner="RED", attempt=1),
            self._chat_event(
                "trade_chat_closed", owner="RED", attempt=1, outcome="no_deal"
            ),
        ]
        result = compute_trade_chat_analysis("RED", events)
        self.assertEqual(result["rooms_opened"], 2)
        self.assertEqual(result["rooms_closed_selected"], 1)
        self.assertEqual(result["rooms_closed_no_deal"], 1)
        self.assertAlmostEqual(result["negotiation_success_rate"], 0.5)

    def test_proposals_made_and_accepted(self) -> None:
        events = [
            self._chat_event("trade_chat_opened", owner="RED"),
            self._chat_event(
                "trade_chat_message",
                owner="RED",
                speaker_player_id="BLUE",
                proposal_id="A0.0.1",
                offer={"WOOD": 1},
                request={"ORE": 1},
                round_index=0,
            ),
            self._chat_event(
                "trade_chat_message",
                owner="RED",
                speaker_player_id="ORANGE",
                proposal_id="A0.0.2",
                offer={"BRICK": 1},
                request={"ORE": 1},
                round_index=0,
            ),
            self._chat_event(
                "trade_chat_quote_selected",
                owner="RED",
                selected_player_id="BLUE",
                offer={"WOOD": 1},
                request={"ORE": 1},
            ),
            self._chat_event(
                "trade_chat_closed",
                owner="RED",
                outcome="selected",
                selected_player_id="BLUE",
            ),
        ]
        blue_result = compute_trade_chat_analysis("BLUE", events)
        self.assertEqual(blue_result["proposals_made"], 1)
        self.assertEqual(blue_result["proposals_accepted"], 1)

        orange_result = compute_trade_chat_analysis("ORANGE", events)
        self.assertEqual(orange_result["proposals_made"], 1)
        self.assertEqual(orange_result["proposals_accepted"], 0)

    def test_rooms_participated_in(self) -> None:
        events = [
            self._chat_event("trade_chat_opened", owner="RED"),
            self._chat_event(
                "trade_chat_message",
                owner="RED",
                speaker_player_id="BLUE",
                round_index=0,
            ),
            self._chat_event("trade_chat_closed", owner="RED", outcome="no_deal"),
        ]
        blue_result = compute_trade_chat_analysis("BLUE", events)
        self.assertEqual(blue_result["rooms_participated_in"], 1)
        self.assertEqual(blue_result["rooms_opened"], 0)

    def test_counterparty_frequency(self) -> None:
        events = [
            self._chat_event("trade_chat_opened", owner="RED", attempt=0),
            self._chat_event(
                "trade_chat_quote_selected",
                owner="RED",
                attempt=0,
                selected_player_id="BLUE",
                offer={"WOOD": 1},
                request={"ORE": 1},
            ),
            self._chat_event(
                "trade_chat_closed",
                owner="RED",
                attempt=0,
                outcome="selected",
                selected_player_id="BLUE",
            ),
            self._chat_event("trade_chat_opened", owner="RED", attempt=1),
            self._chat_event(
                "trade_chat_quote_selected",
                owner="RED",
                attempt=1,
                selected_player_id="BLUE",
                offer={"BRICK": 1},
                request={"SHEEP": 1},
            ),
            self._chat_event(
                "trade_chat_closed",
                owner="RED",
                attempt=1,
                outcome="selected",
                selected_player_id="BLUE",
            ),
        ]
        result = compute_trade_chat_analysis("RED", events)
        self.assertEqual(result["counterparty_frequency"].get("BLUE"), 2)

    def test_resource_flow_as_owner(self) -> None:
        events = [
            self._chat_event("trade_chat_opened", owner="RED"),
            self._chat_event(
                "trade_chat_quote_selected",
                owner="RED",
                selected_player_id="BLUE",
                offer={"WOOD": 2},
                request={"ORE": 3},
            ),
            self._chat_event(
                "trade_chat_closed",
                owner="RED",
                outcome="selected",
                selected_player_id="BLUE",
            ),
        ]
        result = compute_trade_chat_analysis("RED", events)
        self.assertEqual(result["resources_given_via_chat"].get("WOOD"), 2)
        self.assertEqual(result["resources_gained_via_chat"].get("ORE"), 3)

    def test_resource_flow_as_acceptee(self) -> None:
        events = [
            self._chat_event("trade_chat_opened", owner="RED"),
            self._chat_event(
                "trade_chat_message",
                owner="RED",
                speaker_player_id="BLUE",
                proposal_id="A0.0.1",
                offer={"WOOD": 1},
                request={"ORE": 1},
                round_index=0,
            ),
            self._chat_event(
                "trade_chat_quote_selected",
                owner="RED",
                selected_player_id="BLUE",
                offer={"WOOD": 1},
                request={"ORE": 1},
            ),
            self._chat_event(
                "trade_chat_closed",
                owner="RED",
                outcome="selected",
                selected_player_id="BLUE",
            ),
        ]
        # BLUE is acceptee: gains what owner offered (WOOD), gives what owner requested (ORE)
        result = compute_trade_chat_analysis("BLUE", events)
        self.assertEqual(result["resources_gained_via_chat"].get("WOOD"), 1)
        self.assertEqual(result["resources_given_via_chat"].get("ORE"), 1)

    def test_avg_rounds(self) -> None:
        events = [
            self._chat_event("trade_chat_opened", owner="RED"),
            self._chat_event(
                "trade_chat_message",
                owner="RED",
                speaker_player_id="BLUE",
                round_index=0,
            ),
            self._chat_event(
                "trade_chat_message",
                owner="RED",
                speaker_player_id="BLUE",
                round_index=1,
            ),
            self._chat_event("trade_chat_closed", owner="RED", outcome="no_deal"),
        ]
        result = compute_trade_chat_analysis("RED", events)
        self.assertAlmostEqual(result["avg_rounds_per_room"], 2.0)

    def test_no_chat_events(self) -> None:
        result = compute_trade_chat_analysis("RED", [_event("dice_rolled")])
        self.assertEqual(result["rooms_opened"], 0)
        self.assertEqual(result["proposals_made"], 0)
        self.assertEqual(result["negotiation_success_rate"], 0.0)


# ── Strategy evolution tests ──────────────────────────────────────────────────


def _mem_snapshot(
    player_id: str,
    turn: int,
    stage: str,
    long_term: str | None = None,
    short_term: str | None = None,
) -> MemorySnapshot:
    return MemorySnapshot(
        player_id=player_id,
        history_index=0,
        turn_index=turn,
        phase="play_turn",
        decision_index=0,
        stage=stage,
        memory=PlayerMemory(long_term=long_term, short_term=short_term),
    )


class TestComputeStrategyEvolution(unittest.TestCase):
    def test_opening_strategy_extracted(self) -> None:
        snapshots = [
            _mem_snapshot(
                "RED", 1, "opening_strategy", long_term="Focus on ore and wheat"
            ),
        ]
        result = compute_strategy_evolution("RED", snapshots)
        self.assertEqual(result["opening_strategy"], "Focus on ore and wheat")
        self.assertEqual(result["strategy_update_count"], 1)

    def test_deduplicates_unchanged_rewrites(self) -> None:
        snapshots = [
            _mem_snapshot("RED", 1, "opening_strategy", long_term="Plan A"),
            _mem_snapshot(
                "RED", 3, "turn_end", long_term="Plan A"
            ),  # Same — should be skipped
            _mem_snapshot("RED", 5, "turn_end", long_term="Plan B"),  # Changed
        ]
        result = compute_strategy_evolution("RED", snapshots)
        self.assertEqual(result["strategy_update_count"], 2)
        self.assertEqual(result["strategy_updates"][0]["long_term"], "Plan A")
        self.assertEqual(result["strategy_updates"][1]["long_term"], "Plan B")

    def test_final_strategy_captured(self) -> None:
        snapshots = [
            _mem_snapshot("RED", 1, "opening_strategy", long_term="Start"),
            _mem_snapshot("RED", 10, "turn_end", long_term="Endgame pivot"),
        ]
        result = compute_strategy_evolution("RED", snapshots)
        self.assertEqual(result["final_strategy"], "Endgame pivot")

    def test_short_term_note_count(self) -> None:
        snapshots = [
            _mem_snapshot("RED", 1, "turn_start", short_term="note 1"),
            _mem_snapshot("RED", 1, "choose_action", short_term="note 2"),
            _mem_snapshot("RED", 2, "turn_start", short_term="note 3"),
            _mem_snapshot("RED", 2, "turn_end", long_term="strategy"),  # Not counted
        ]
        result = compute_strategy_evolution("RED", snapshots)
        self.assertEqual(result["short_term_note_count"], 3)

    def test_empty_memory_traces(self) -> None:
        result = compute_strategy_evolution("RED", [])
        self.assertIsNone(result["opening_strategy"])
        self.assertEqual(result["strategy_update_count"], 0)
        self.assertIsNone(result["final_strategy"])
        self.assertEqual(result["short_term_note_count"], 0)

    def test_filters_by_player_id(self) -> None:
        snapshots = [
            _mem_snapshot("RED", 1, "opening_strategy", long_term="Red plan"),
            _mem_snapshot("BLUE", 1, "opening_strategy", long_term="Blue plan"),
        ]
        red_result = compute_strategy_evolution("RED", snapshots)
        self.assertEqual(red_result["opening_strategy"], "Red plan")
        self.assertEqual(red_result["strategy_update_count"], 1)

    def test_strategy_stability_reflects_rewrites(self) -> None:
        snapshots = [
            _mem_snapshot("RED", 1, "opening_strategy", long_term="ore city plan"),
            _mem_snapshot("RED", 2, "turn_end", long_term="sheep knight strategy"),
        ]
        result = compute_strategy_evolution("RED", snapshots)
        self.assertEqual(result["strategy_stability"], 0.0)


class TestComputeMarketAnalysis(unittest.TestCase):
    def test_market_roles_and_bank_taker_are_tracked(self) -> None:
        events = [
            _event(
                "trade_confirmed",
                offering_player_id="RED",
                accepting_player_id="BLUE",
                offer={"WOOD": 1},
                request={"BRICK": 1},
            ),
            _event(
                "action_taken",
                actor="WHITE",
                action={
                    "action_type": "MARITIME_TRADE",
                    "payload": {
                        "give": ["ORE", "ORE", "ORE", "ORE"],
                        "receive": "WOOD",
                    },
                },
            ),
        ]

        result = compute_market_analysis(events, ["RED", "BLUE", "WHITE"])

        self.assertEqual(result["actors"]["RED"]["market_role"], "Market maker")
        self.assertEqual(result["actors"]["BLUE"]["market_role"], "Market taker")
        self.assertEqual(result["actors"]["WHITE"]["market_role"], "Market maker")
        self.assertEqual(result["actors"]["BANK"]["market_role"], "Market taker")
        self.assertEqual(result["actors"]["BANK"]["taker_deals"], 1)
        self.assertAlmostEqual(result["actors"]["RED"]["market_initiation_rate"], 1.0)
        self.assertGreater(
            result["actors"]["WHITE"]["resource_market_share"]["WOOD"], 0.0
        )
        self.assertGreater(result["resource_involvement_totals"]["WOOD"], 0)


# ── PIPS constant test ────────────────────────────────────────────────────────


class TestPipsConstant(unittest.TestCase):
    def test_six_and_eight_have_most_pips(self) -> None:
        self.assertEqual(PIPS[6], 5)
        self.assertEqual(PIPS[8], 5)

    def test_two_and_twelve_have_one_pip(self) -> None:
        self.assertEqual(PIPS[2], 1)
        self.assertEqual(PIPS[12], 1)

    def test_seven_not_in_pips(self) -> None:
        self.assertNotIn(7, PIPS)


# ── Integration test ──────────────────────────────────────────────────────────


class TestAnalyzeGameIntegration(unittest.TestCase):
    def _make_minimal_run(self, run_dir: Path) -> None:
        _write_json(
            run_dir / "metadata.json",
            {
                "player_ids": ["RED", "BLUE"],
                "player_adapter_types": {
                    "RED": "LLMPlayer",
                    "BLUE": "RandomLegalPlayer",
                },
            },
        )
        _write_json(
            run_dir / "result.json",
            {
                "game_id": "test-game",
                "winner_ids": ["RED"],
                "total_decisions": 10,
                "public_event_count": 8,
                "memory_writes": 2,
                "metadata": {
                    "num_turns": 5,
                    "players": {
                        "RED": {
                            "actual_victory_points": 5,
                            "visible_victory_points": 5,
                        },
                        "BLUE": {
                            "actual_victory_points": 2,
                            "visible_victory_points": 2,
                        },
                    },
                },
            },
        )
        _write_jsonl(
            run_dir / "public_history.jsonl",
            [
                {
                    "kind": "dice_rolled",
                    "payload": {"dice": [3, 3]},
                    "turn_index": 1,
                    "history_index": 1,
                    "phase": "play_turn",
                    "actor_player_id": "RED",
                },
                {
                    "kind": "settlement_built",
                    "payload": {"node_id": 1},
                    "turn_index": 0,
                    "history_index": 2,
                    "phase": "build_initial_settlement",
                    "actor_player_id": "RED",
                },
                {
                    "kind": "turn_ended",
                    "payload": {},
                    "turn_index": 1,
                    "history_index": 3,
                    "phase": "play_turn",
                    "actor_player_id": "RED",
                },
            ],
        )
        _write_jsonl(
            run_dir / "public_state_trace.jsonl",
            [
                {
                    "history_index": 0,
                    "turn_index": 0,
                    "phase": "initial",
                    "decision_index": None,
                    "public_state": {
                        "players": {
                            "RED": {
                                "visible_victory_points": 1,
                                "longest_road_length": 2,
                                "has_longest_road": False,
                                "has_largest_army": False,
                                "resource_card_count": 0,
                                "development_card_count": 0,
                            },
                            "BLUE": {
                                "visible_victory_points": 0,
                                "longest_road_length": 0,
                                "has_longest_road": False,
                                "has_largest_army": False,
                                "resource_card_count": 0,
                                "development_card_count": 0,
                            },
                        },
                        "board": {
                            "tiles": {},
                            "nodes": {},
                            "edges": [],
                            "adjacent_tiles": {},
                            "robber_coordinate": [0, 0, 0],
                        },
                    },
                },
            ],
        )
        _write_jsonl(
            run_dir / "players" / "RED" / "prompt_trace.jsonl",
            [
                {
                    "player_id": "RED",
                    "history_index": 1,
                    "turn_index": 1,
                    "phase": "play_turn",
                    "decision_index": 0,
                    "stage": "choose_action",
                    "model": "gpt-4o",
                    "temperature": 0.5,
                    "attempts": [
                        {"messages": [], "response_text": "ok", "response": {}},
                        {"messages": [], "response_text": "ok2", "response": {}},
                    ],
                }
            ],
        )
        _write_jsonl(run_dir / "players" / "BLUE" / "prompt_trace.jsonl", [])
        _write_jsonl(
            run_dir / "players" / "RED" / "memory_trace.jsonl",
            [
                {
                    "player_id": "RED",
                    "history_index": 0,
                    "turn_index": 1,
                    "phase": "play_turn",
                    "decision_index": 0,
                    "stage": "opening_strategy",
                    "memory": {
                        "long_term": "Expand toward ore ports",
                        "short_term": None,
                    },
                },
                {
                    "player_id": "RED",
                    "history_index": 2,
                    "turn_index": 3,
                    "phase": "play_turn",
                    "decision_index": 5,
                    "stage": "turn_end",
                    "memory": {
                        "long_term": "Pivot to city strategy",
                        "short_term": None,
                    },
                },
            ],
        )
        _write_jsonl(run_dir / "players" / "BLUE" / "memory_trace.jsonl", [])

    def test_analyze_game_returns_valid_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            self._make_minimal_run(run_dir)

            analysis = analyze_game(run_dir, write=True)

            self.assertEqual(analysis["game_id"], "test-game")
            self.assertEqual(analysis["version"], "3")

            gs = analysis["game_summary"]
            self.assertEqual(gs["winner_ids"], ["RED"])
            self.assertEqual(gs["num_turns"], 5)
            self.assertEqual(gs["total_decisions"], 10)
            self.assertIn("trade_chat_rooms", gs)
            self.assertIn("trade_chat_success_rate", gs)

            players = analysis["players"]
            self.assertIn("RED", players)
            self.assertIn("BLUE", players)

            self.assertTrue(players["RED"]["is_winner"])
            self.assertFalse(players["BLUE"]["is_winner"])
            self.assertEqual(players["RED"]["final_vp"], 5)

            # Discard key present
            self.assertIn("discard", players["RED"])

            # New keys present
            self.assertIn("trade_chat", players["RED"])
            self.assertIn("strategy", players["RED"])
            self.assertIn("market_profile", players["RED"])
            self.assertEqual(
                players["RED"]["strategy"]["opening_strategy"],
                "Expand toward ore ports",
            )
            self.assertEqual(players["RED"]["strategy"]["strategy_update_count"], 2)
            self.assertEqual(
                players["RED"]["strategy"]["final_strategy"], "Pivot to city strategy"
            )
            self.assertIn("strategy_stability", players["RED"]["strategy"])
            self.assertIn("market", analysis)

            # analysis.json was written
            self.assertTrue((run_dir / "analysis.json").exists())

    def test_analyze_game_no_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            self._make_minimal_run(run_dir)

            analyze_game(run_dir, write=False)

            self.assertFalse((run_dir / "analysis.json").exists())

    def test_decision_quality_captured_from_prompt_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            self._make_minimal_run(run_dir)

            analysis = analyze_game(run_dir, write=False)

            dq = analysis["players"]["RED"]["decision_quality"]
            self.assertEqual(dq["total_prompts"], 1)
            self.assertEqual(dq["retries"], 1)
            self.assertAlmostEqual(dq["retry_rate"], 1.0)

    def test_missing_result_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                analyze_game(Path(tmpdir), write=False)

    def test_discover_completed_run_directories_from_base_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            completed = base_dir / "0.4.0-dev-game-a"
            other_completed = base_dir / "0.4.0-dev-game-b"
            incomplete = base_dir / "0.4.0-dev-game-c"
            self._make_minimal_run(completed)
            self._make_minimal_run(other_completed)
            _write_json(
                incomplete / "metadata.json",
                {"player_ids": ["RED", "BLUE"]},
            )
            _write_jsonl(incomplete / "public_history.jsonl", [])
            _write_jsonl(incomplete / "public_state_trace.jsonl", [])

            discovered = discover_completed_run_directories(base_dir)

            self.assertEqual({path.name for path in discovered}, {completed.name, other_completed.name})

    def test_analysis_main_accepts_base_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            run_a = base_dir / "0.4.0-dev-game-a"
            run_b = base_dir / "0.4.0-dev-game-b"
            self._make_minimal_run(run_a)
            self._make_minimal_run(run_b)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = analysis_main([str(base_dir), "--json-only", "--no-write"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(len(payload), 2)
            self.assertEqual({Path(item["run_dir"]).name for item in payload}, {run_a.name, run_b.name})


if __name__ == "__main__":
    unittest.main()
