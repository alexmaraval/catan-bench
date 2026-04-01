from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

try:
    from catan_bench.catanatron_adapter import CatanatronEngineAdapter
except RuntimeError:  # pragma: no cover - dependency missing in some environments.
    CatanatronEngineAdapter = None

from catan_bench import Action


@unittest.skipIf(
    CatanatronEngineAdapter is None, "catanatron dependency is not installed"
)
class CatanatronAdapterTests(unittest.TestCase):
    @staticmethod
    def _affordable_trade_payload(
        adapter: CatanatronEngineAdapter,
    ) -> dict[str, dict[str, int]]:
        decision = adapter.current_decision()
        private_state = adapter.private_state(decision.acting_player_id)
        resources = private_state.get("resources", {})
        if not isinstance(resources, dict):
            raise AssertionError("Expected resource map in private state.")

        offered_resource = next(
            (
                resource
                for resource, amount in resources.items()
                if isinstance(amount, int) and amount > 0
            ),
            None,
        )
        if offered_resource is None:
            raise AssertionError("Expected at least one resource to offer for trade.")
        requested_resource = next(
            resource
            for resource in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
            if resource != offered_resource
        )
        return {
            "offer": {str(offered_resource): 1},
            "request": {requested_resource: 1},
        }

    def test_resolve_action_canonicalizes_unique_move_robber_coordinate(self) -> None:
        adapter = object.__new__(CatanatronEngineAdapter)
        legal_actions = (
            Action(
                "MOVE_ROBBER",
                payload={"coordinate": [1, -1, 0], "victim": "BLUE"},
            ),
            Action(
                "BUILD_ROAD",
                payload={"edge": [0, 1]},
            ),
        )

        resolved = adapter.resolve_action(
            proposed_action=Action(
                "MOVE_ROBBER",
                payload={"coordinate": [1, -1, 0], "victim": "RED"},
            ),
            legal_actions=legal_actions,
        )

        self.assertEqual(resolved, legal_actions[0])

    def test_resolve_action_rejects_ambiguous_move_robber_coordinate(self) -> None:
        adapter = object.__new__(CatanatronEngineAdapter)
        legal_actions = (
            Action(
                "MOVE_ROBBER",
                payload={"coordinate": [1, -1, 0], "victim": "BLUE"},
            ),
            Action(
                "MOVE_ROBBER",
                payload={"coordinate": [1, -1, 0], "victim": "ORANGE"},
            ),
        )

        with self.assertRaises(ValueError):
            adapter.resolve_action(
                proposed_action=Action(
                    "MOVE_ROBBER",
                    payload={"coordinate": [1, -1, 0], "victim": "RED"},
                ),
                legal_actions=legal_actions,
            )

    def test_resolve_action_canonicalizes_trade_response_by_action_type(self) -> None:
        adapter = object.__new__(CatanatronEngineAdapter)
        legal_actions = (
            Action(
                "ACCEPT_TRADE",
                payload={
                    "offer": {"WOOD": 1},
                    "request": {"BRICK": 1},
                    "offering_player_id": "RED",
                },
            ),
            Action(
                "REJECT_TRADE",
                payload={
                    "offer": {"WOOD": 1},
                    "request": {"BRICK": 1},
                    "offering_player_id": "RED",
                },
            ),
        )

        resolved = adapter.resolve_action(
            proposed_action=Action("ACCEPT_TRADE", payload={}),
            legal_actions=legal_actions,
        )

        self.assertEqual(resolved, legal_actions[0])

    def test_resolve_action_canonicalizes_confirm_trade_by_accepting_player(
        self,
    ) -> None:
        adapter = object.__new__(CatanatronEngineAdapter)
        legal_actions = (
            Action(
                "CONFIRM_TRADE",
                payload={
                    "offer": {"WOOD": 1},
                    "request": {"BRICK": 1},
                    "accepting_player_id": "BLUE",
                },
            ),
            Action(
                "CONFIRM_TRADE",
                payload={
                    "offer": {"WOOD": 1},
                    "request": {"BRICK": 1},
                    "accepting_player_id": "ORANGE",
                },
            ),
        )

        resolved = adapter.resolve_action(
            proposed_action=Action(
                "CONFIRM_TRADE",
                payload={"with_player_id": "ORANGE"},
            ),
            legal_actions=legal_actions,
        )

        self.assertEqual(resolved, legal_actions[1])

    def test_description_for_payload_heavy_actions_is_informative(self) -> None:
        self.assertEqual(
            CatanatronEngineAdapter._description_for_action(
                "MOVE_ROBBER",
                {"coordinate": [1, -1, 0], "victim": "BLUE"},
            ),
            "Move the robber to [1, -1, 0] and steal from BLUE.",
        )
        self.assertEqual(
            CatanatronEngineAdapter._description_for_action(
                "MARITIME_TRADE",
                {"give": ["WOOD", "WOOD", "WOOD", "WOOD"], "receive": "ORE"},
            ),
            "Trade ['WOOD', 'WOOD', 'WOOD', 'WOOD'] to the bank for ORE.",
        )
        self.assertEqual(
            CatanatronEngineAdapter._description_for_action(
                "DISCARD",
                {"resources": {"WOOD": 1, "ORE": 2}},
            ),
            "Discard 1×WOOD, 2×ORE for the robber event.",
        )

    def test_discard_payload_round_trips_as_resource_counts(self) -> None:
        adapter = object.__new__(CatanatronEngineAdapter)
        native_value = (2, 1, 0, 0, 0)

        payload = adapter._native_value_to_payload("DISCARD", native_value)

        self.assertEqual(payload, {"resources": {"WOOD": 2, "BRICK": 1}})
        self.assertEqual(
            adapter._payload_to_native_value("DISCARD", payload),
            native_value,
        )

    def test_public_event_payload_redacts_exact_discard_resources(self) -> None:
        adapter = object.__new__(CatanatronEngineAdapter)
        red = SimpleNamespace(value="RED")
        prompt = SimpleNamespace(value="discard")
        state_before = SimpleNamespace(
            colors=[red],
            current_turn_index=0,
            current_prompt=prompt,
            is_resolving_trade=False,
            acceptees=[],
            current_trade=(),
        )
        state_after = SimpleNamespace(
            colors=[red],
            current_turn_index=0,
            current_prompt=prompt,
            is_resolving_trade=False,
            acceptees=[],
            current_trade=(),
        )

        payload = adapter._public_event_payload(
            action=Action("DISCARD", payload={"resources": {"WOOD": 1, "ORE": 2}}),
            action_result=None,
            actor_player_id="RED",
            state_before=state_before,
            state_after=state_after,
        )

        self.assertEqual(payload["action"], {"action_type": "DISCARD", "payload": {}})
        self.assertEqual(payload["discarded_count"], 3)
        self.assertEqual(
            CatanatronEngineAdapter._event_kind("DISCARD"), "resources_discarded"
        )

    def test_ordered_legal_actions_moves_end_turn_to_the_end(self) -> None:
        ordered = CatanatronEngineAdapter._ordered_legal_actions(
            (
                Action("END_TURN"),
                Action("OFFER_TRADE", payload={"offer": {}, "request": {}}),
                Action("BUY_DEVELOPMENT_CARD"),
            )
        )

        self.assertEqual(
            [action.action_type for action in ordered],
            ["OFFER_TRADE", "BUY_DEVELOPMENT_CARD", "END_TURN"],
        )

    def test_initial_decision_exposes_player_scoped_state(self) -> None:
        adapter = CatanatronEngineAdapter(seed=7)

        decision = adapter.current_decision()
        self.assertEqual(decision.phase, "build_initial_settlement")
        self.assertGreater(len(decision.legal_actions), 0)
        self.assertTrue(
            all(
                action.action_type == "BUILD_SETTLEMENT"
                for action in decision.legal_actions
            )
        )

        private_state = adapter.private_state(decision.acting_player_id)
        self.assertIn("resources", private_state)
        self.assertIn("development_cards", private_state)
        self.assertNotIn("players", private_state)

        public_state = adapter.public_state()
        self.assertIn("players", public_state)
        self.assertIn("board", public_state)
        self.assertNotIn("player_state", public_state)

    def test_decision_scoped_state_condenses_summaries_and_board(self) -> None:
        adapter = CatanatronEngineAdapter(seed=7)
        decision = adapter.current_decision()

        public_state = adapter.public_state_for_decision(
            player_id=decision.acting_player_id,
            phase=decision.phase,
            legal_actions=decision.legal_actions,
        )
        private_state = adapter.private_state_for_decision(
            player_id=decision.acting_player_id,
            phase=decision.phase,
            legal_actions=decision.legal_actions,
        )

        current_player = public_state["players"][decision.acting_player_id]
        self.assertIn("vp", current_player)
        self.assertIn("roads", current_player)
        self.assertIn("resource_card_count", current_player)
        self.assertIn("development_card_count", current_player)
        self.assertIn("vps_to_win", public_state["turn"])
        self.assertIn("roads_left", current_player)
        self.assertIn("settlements_left", current_player)
        self.assertIn("cities_left", current_player)
        self.assertNotIn("seat_index", current_player)
        self.assertNotIn("development_cards_remaining", public_state["bank"])
        self.assertIn("your_network", public_state["board"])
        self.assertIn("settlement_candidates", public_state["board"])
        self.assertNotIn("topology", public_state["board"])
        self.assertEqual(
            len(public_state["board"]["settlement_candidates"]),
            len(decision.legal_actions),
        )
        first_candidate = public_state["board"]["settlement_candidates"][0]
        self.assertIn("action_index", first_candidate)
        self.assertIsInstance(first_candidate["adjacent_tiles"][0], str)
        self.assertLess(
            len(json.dumps(public_state["board"], sort_keys=True)),
            len(json.dumps(adapter.public_state()["board"], sort_keys=True)),
        )
        self.assertNotIn("status_flags", private_state)
        self.assertNotIn("ports", private_state)

    def test_initial_road_prompt_view_uses_road_candidates(self) -> None:
        adapter = CatanatronEngineAdapter(seed=1)
        adapter.apply_action(adapter.current_decision().legal_actions[0])

        decision = adapter.current_decision()
        self.assertEqual(decision.phase, "build_initial_road")

        public_state = adapter.public_state_for_decision(
            player_id=decision.acting_player_id,
            phase=decision.phase,
            legal_actions=decision.legal_actions,
        )

        self.assertIn("road_candidates", public_state["board"])
        self.assertNotIn("topology", public_state["board"])
        self.assertEqual(
            len(public_state["board"]["road_candidates"]),
            len(decision.legal_actions),
        )
        self.assertTrue(
            all(
                "action_index" in candidate and "edge" in candidate
                for candidate in public_state["board"]["road_candidates"]
            )
        )
        self.assertFalse(
            any(
                "endpoints" in candidate
                for candidate in public_state["board"]["road_candidates"]
            )
        )

    def test_decision_scoped_state_omits_topology_when_not_building(self) -> None:
        adapter = CatanatronEngineAdapter(seed=1)

        while adapter.current_decision().phase != "play_turn":
            adapter.apply_action(adapter.current_decision().legal_actions[0])

        roll_transition = adapter.apply_action(Action("ROLL"))
        self.assertTrue(
            any(event.kind == "dice_rolled" for event in roll_transition.public_events)
        )

        while adapter.current_decision().phase != "play_turn":
            adapter.apply_action(adapter.current_decision().legal_actions[0])
        trade_payload = self._affordable_trade_payload(adapter)
        adapter.apply_action(
            adapter.resolve_action(
                proposed_action=Action(
                    "OFFER_TRADE",
                    payload=trade_payload,
                ),
                legal_actions=adapter.current_decision().legal_actions,
            )
        )

        decision = adapter.current_decision()
        self.assertEqual(decision.phase, "decide_trade")
        public_state = adapter.public_state_for_decision(
            player_id=decision.acting_player_id,
            phase=decision.phase,
            legal_actions=decision.legal_actions,
        )
        private_state = adapter.private_state_for_decision(
            player_id=decision.acting_player_id,
            phase=decision.phase,
            legal_actions=decision.legal_actions,
        )

        self.assertIn("your_network", public_state["board"])
        self.assertNotIn("topology", public_state["board"])
        self.assertNotIn("ports", private_state)

    def test_can_advance_to_turn_play_and_offer_trade(self) -> None:
        adapter = CatanatronEngineAdapter(seed=1)

        initial_event_kinds = set()
        while adapter.current_decision().phase != "play_turn":
            transition = adapter.apply_action(
                adapter.current_decision().legal_actions[0]
            )
            initial_event_kinds.update(event.kind for event in transition.public_events)

        self.assertTrue({"settlement_built", "road_built"} & initial_event_kinds)

        roll_transition = adapter.apply_action(Action("ROLL"))
        self.assertTrue(
            any(event.kind == "dice_rolled" for event in roll_transition.public_events)
        )

        while adapter.current_decision().phase != "play_turn":
            adapter.apply_action(adapter.current_decision().legal_actions[0])

        decision = adapter.current_decision()
        trade_actions = [
            action
            for action in decision.legal_actions
            if action.action_type == "OFFER_TRADE"
        ]
        self.assertEqual(len(trade_actions), 1)
        trade_payload = self._affordable_trade_payload(adapter)

        canonical_offer = adapter.resolve_action(
            proposed_action=Action(
                "OFFER_TRADE",
                payload=trade_payload,
            ),
            legal_actions=decision.legal_actions,
        )
        self.assertEqual(canonical_offer.action_type, "OFFER_TRADE")

        transition = adapter.apply_action(canonical_offer)
        self.assertTrue(
            any(event.kind == "trade_offered" for event in transition.public_events)
        )
        offered_event = transition.public_events[0]
        self.assertEqual(offered_event.actor_player_id, decision.acting_player_id)
        self.assertEqual(offered_event.payload["offer"], trade_payload["offer"])
        self.assertEqual(offered_event.payload["request"], trade_payload["request"])
        self.assertEqual(adapter.current_decision().phase, "decide_trade")
        self.assertTrue(
            all(
                action.action_type in {"ACCEPT_TRADE", "REJECT_TRADE", "COUNTER_OFFER"}
                for action in adapter.current_decision().legal_actions
            )
        )

    def test_resolve_action_rejects_trade_offer_when_player_lacks_resources(
        self,
    ) -> None:
        adapter = CatanatronEngineAdapter(seed=1)

        while adapter.current_decision().phase != "play_turn":
            adapter.apply_action(adapter.current_decision().legal_actions[0])

        adapter.apply_action(Action("ROLL"))
        while adapter.current_decision().phase != "play_turn":
            adapter.apply_action(adapter.current_decision().legal_actions[0])

        decision = adapter.current_decision()
        with self.assertRaises(ValueError):
            adapter.resolve_action(
                proposed_action=Action(
                    "OFFER_TRADE",
                    payload={"offer": {"WOOD": 99}, "request": {"BRICK": 1}},
                ),
                legal_actions=decision.legal_actions,
            )

    def test_decide_trade_exposes_counter_offer_template(self) -> None:
        adapter = CatanatronEngineAdapter(seed=1)

        while adapter.current_decision().phase != "play_turn":
            adapter.apply_action(adapter.current_decision().legal_actions[0])
        adapter.apply_action(Action("ROLL"))
        while adapter.current_decision().phase != "play_turn":
            adapter.apply_action(adapter.current_decision().legal_actions[0])

        decision = adapter.current_decision()
        trade_payload = self._affordable_trade_payload(adapter)
        adapter.apply_action(
            adapter.resolve_action(
                proposed_action=Action(
                    "OFFER_TRADE",
                    payload=trade_payload,
                ),
                legal_actions=decision.legal_actions,
            )
        )

        decide_trade = adapter.current_decision()
        self.assertEqual(decide_trade.phase, "decide_trade")
        self.assertIn(
            "COUNTER_OFFER",
            [action.action_type for action in decide_trade.legal_actions],
        )

    def test_public_event_for_self_trade_response_is_suppressed(self) -> None:
        adapter = object.__new__(CatanatronEngineAdapter)
        adapter._native_action_to_action = lambda native_action: Action(  # type: ignore[method-assign]
            "REJECT_TRADE",
            payload={
                "offer": {"SHEEP": 1},
                "request": {"BRICK": 1},
                "offering_player_id": "BLUE",
            },
        )
        adapter._public_event_payload = lambda **kwargs: {  # type: ignore[method-assign]
            "offering_player_id": "BLUE",
            "responding_player_id": "BLUE",
        }
        adapter._phase_name = lambda prompt: "decide_trade"  # type: ignore[method-assign]

        event = adapter._public_event_for_action(
            native_action=SimpleNamespace(color=SimpleNamespace(value="BLUE")),
            action_result=None,
            state_before=SimpleNamespace(
                num_turns=1,
                current_prompt=SimpleNamespace(value="DECIDE_TRADE"),
                action_records=[],
            ),
            state_after=SimpleNamespace(),
        )

        self.assertIsNone(event)

    def test_current_decision_auto_skips_self_trade_response_state(self) -> None:
        from catan_bench.catanatron_adapter import (
            ActionPrompt,
            ActionType,
            Color,
            NativeAction,
        )

        class _FakeState:
            def __init__(self) -> None:
                self.current_prompt = ActionPrompt.DECIDE_TRADE
                self.is_resolving_trade = True
                self.colors = (Color.BLUE, Color.WHITE)
                self.current_trade = (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0)
                self.num_turns = 1
                self.action_records = []
                self._current_color = Color.BLUE

            def current_color(self):
                return self._current_color

        class _FakeGame:
            def __init__(self) -> None:
                self.state = _FakeState()
                self.playable_actions = [
                    NativeAction(
                        Color.BLUE, ActionType.REJECT_TRADE, self.state.current_trade
                    )
                ]

            def execute(self, native_action, validate_action=True):
                self.state.action_records.append(native_action)
                self.state.current_prompt = ActionPrompt.PLAY_TURN
                self.state._current_color = Color.WHITE
                self.playable_actions = [
                    NativeAction(Color.WHITE, ActionType.END_TURN, None)
                ]
                return None

        adapter = object.__new__(CatanatronEngineAdapter)
        adapter.game = _FakeGame()
        adapter._native_action_to_action = lambda native_action: Action(
            native_action.action_type.value
        )  # type: ignore[method-assign]
        adapter._can_offer_trade = lambda: False  # type: ignore[method-assign]
        adapter._can_counter_offer = lambda legal_actions: False  # type: ignore[method-assign]
        adapter._ordered_legal_actions = lambda legal_actions: legal_actions  # type: ignore[method-assign]

        decision = adapter.current_decision()

        self.assertEqual(decision.phase, "play_turn")
        self.assertEqual(decision.acting_player_id, "WHITE")
        self.assertEqual(len(adapter.game.state.action_records), 1)
        self.assertEqual(
            adapter.game.state.action_records[0].action_type, ActionType.REJECT_TRADE
        )

    def test_result_exposes_victory_point_audit_fields(self) -> None:
        adapter = CatanatronEngineAdapter(seed=7)

        result = adapter.result()

        players = result["players"]
        self.assertTrue(players)
        first_player = next(iter(players.values()))
        self.assertIn("visible_victory_points", first_player)
        self.assertIn("actual_victory_points", first_player)
        self.assertIn("dev_victory_points", first_player)
        self.assertIn("played_knights", first_player)
        self.assertIn("has_longest_road", first_player)
        self.assertIn("has_largest_army", first_player)

    def test_recompute_longest_road_counts_segment_ending_at_enemy_settlement(
        self,
    ) -> None:
        from catan_bench.catanatron_adapter import Color

        orange = Color.ORANGE
        blue = Color.BLUE
        red = Color.RED
        state = SimpleNamespace(
            colors=(orange, blue, red),
            color_to_index={orange: 0, blue: 1, red: 2},
            player_state={
                "P0_HAS_ROAD": False,
                "P0_VICTORY_POINTS": 3,
                "P0_ACTUAL_VICTORY_POINTS": 3,
                "P0_LONGEST_ROAD_LENGTH": 0,
                "P1_HAS_ROAD": False,
                "P1_VICTORY_POINTS": 2,
                "P1_ACTUAL_VICTORY_POINTS": 2,
                "P1_LONGEST_ROAD_LENGTH": 0,
                "P2_HAS_ROAD": False,
                "P2_VICTORY_POINTS": 2,
                "P2_ACTUAL_VICTORY_POINTS": 2,
                "P2_LONGEST_ROAD_LENGTH": 0,
            },
            board=SimpleNamespace(
                roads={
                    (0, 5): orange,
                    (5, 0): orange,
                    (0, 20): orange,
                    (20, 0): orange,
                    (19, 20): orange,
                    (20, 19): orange,
                    (20, 22): orange,
                    (22, 20): orange,
                    (22, 49): orange,
                    (49, 22): orange,
                },
                buildings={
                    5: (blue, "SETTLEMENT"),
                    19: (orange, "SETTLEMENT"),
                    49: (orange, "SETTLEMENT"),
                },
                road_lengths={},
                road_color=None,
                road_length=0,
            ),
        )
        adapter = object.__new__(CatanatronEngineAdapter)
        adapter.game = SimpleNamespace(state=state)

        adapter._recompute_longest_road_state()

        self.assertEqual(state.player_state["P0_LONGEST_ROAD_LENGTH"], 4)
        self.assertFalse(state.player_state["P0_HAS_ROAD"])

    def test_recompute_longest_road_keeps_current_holder_on_tie(self) -> None:
        from catan_bench.catanatron_adapter import Color

        blue = Color.BLUE
        orange = Color.ORANGE
        state = SimpleNamespace(
            colors=(blue, orange),
            color_to_index={blue: 0, orange: 1},
            player_state={
                "P0_HAS_ROAD": True,
                "P0_VICTORY_POINTS": 5,
                "P0_ACTUAL_VICTORY_POINTS": 5,
                "P0_LONGEST_ROAD_LENGTH": 5,
                "P1_HAS_ROAD": False,
                "P1_VICTORY_POINTS": 3,
                "P1_ACTUAL_VICTORY_POINTS": 3,
                "P1_LONGEST_ROAD_LENGTH": 4,
            },
            board=SimpleNamespace(
                roads={},
                buildings={},
                road_lengths={},
                road_color=blue,
                road_length=5,
            ),
        )
        adapter = object.__new__(CatanatronEngineAdapter)
        adapter.game = SimpleNamespace(state=state)
        adapter._longest_road_length_for_color = lambda color: 5  # type: ignore[method-assign]

        adapter._recompute_longest_road_state()

        self.assertTrue(state.player_state["P0_HAS_ROAD"])
        self.assertFalse(state.player_state["P1_HAS_ROAD"])
        self.assertEqual(state.player_state["P0_VICTORY_POINTS"], 5)
        self.assertEqual(state.player_state["P1_VICTORY_POINTS"], 3)
        self.assertEqual(state.player_state["P0_LONGEST_ROAD_LENGTH"], 5)
        self.assertEqual(state.player_state["P1_LONGEST_ROAD_LENGTH"], 5)

    def test_recompute_largest_army_uses_played_knights_and_awards_vp(self) -> None:
        from catan_bench.catanatron_adapter import Color

        blue = Color.BLUE
        orange = Color.ORANGE
        state = SimpleNamespace(
            colors=(blue, orange),
            color_to_index={blue: 0, orange: 1},
            player_state={
                "P0_HAS_ARMY": False,
                "P0_PLAYED_KNIGHT": 2,
                "P0_VICTORY_POINTS": 2,
                "P0_ACTUAL_VICTORY_POINTS": 2,
                "P1_HAS_ARMY": False,
                "P1_PLAYED_KNIGHT": 3,
                "P1_VICTORY_POINTS": 3,
                "P1_ACTUAL_VICTORY_POINTS": 3,
            },
        )
        adapter = object.__new__(CatanatronEngineAdapter)
        adapter.game = SimpleNamespace(state=state)

        adapter._recompute_largest_army_state()

        self.assertFalse(state.player_state["P0_HAS_ARMY"])
        self.assertTrue(state.player_state["P1_HAS_ARMY"])
        self.assertEqual(state.player_state["P1_VICTORY_POINTS"], 5)
        self.assertEqual(state.player_state["P1_ACTUAL_VICTORY_POINTS"], 5)

    def test_recompute_largest_army_keeps_current_holder_on_tie(self) -> None:
        from catan_bench.catanatron_adapter import Color

        blue = Color.BLUE
        orange = Color.ORANGE
        state = SimpleNamespace(
            colors=(blue, orange),
            color_to_index={blue: 0, orange: 1},
            player_state={
                "P0_HAS_ARMY": True,
                "P0_PLAYED_KNIGHT": 3,
                "P0_VICTORY_POINTS": 5,
                "P0_ACTUAL_VICTORY_POINTS": 5,
                "P1_HAS_ARMY": False,
                "P1_PLAYED_KNIGHT": 3,
                "P1_VICTORY_POINTS": 3,
                "P1_ACTUAL_VICTORY_POINTS": 3,
            },
        )
        adapter = object.__new__(CatanatronEngineAdapter)
        adapter.game = SimpleNamespace(state=state)

        adapter._recompute_largest_army_state()

        self.assertTrue(state.player_state["P0_HAS_ARMY"])
        self.assertFalse(state.player_state["P1_HAS_ARMY"])
        self.assertEqual(state.player_state["P0_VICTORY_POINTS"], 5)
        self.assertEqual(state.player_state["P1_VICTORY_POINTS"], 3)


if __name__ == "__main__":
    unittest.main()
