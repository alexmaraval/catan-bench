from __future__ import annotations

import json
import unittest

try:
    from catan_bench.catanatron_adapter import CatanatronEngineAdapter
except RuntimeError:  # pragma: no cover - dependency missing in some environments.
    CatanatronEngineAdapter = None

from catan_bench import Action


@unittest.skipIf(CatanatronEngineAdapter is None, "catanatron dependency is not installed")
class CatanatronAdapterTests(unittest.TestCase):
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

    def test_resolve_action_canonicalizes_confirm_trade_by_accepting_player(self) -> None:
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
            all(action.action_type == "BUILD_SETTLEMENT" for action in decision.legal_actions)
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
        self.assertTrue(any(event.kind == "dice_rolled" for event in roll_transition.public_events))

        while adapter.current_decision().phase != "play_turn":
            adapter.apply_action(adapter.current_decision().legal_actions[0])
        adapter.apply_action(
            adapter.resolve_action(
                proposed_action=Action(
                    "OFFER_TRADE",
                    payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
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
            transition = adapter.apply_action(adapter.current_decision().legal_actions[0])
            initial_event_kinds.update(event.kind for event in transition.public_events)

        self.assertTrue(
            {"settlement_built", "road_built"} & initial_event_kinds
        )

        roll_transition = adapter.apply_action(Action("ROLL"))
        self.assertTrue(any(event.kind == "dice_rolled" for event in roll_transition.public_events))

        while adapter.current_decision().phase != "play_turn":
            adapter.apply_action(adapter.current_decision().legal_actions[0])

        decision = adapter.current_decision()
        trade_actions = [action for action in decision.legal_actions if action.action_type == "OFFER_TRADE"]
        self.assertEqual(len(trade_actions), 1)

        canonical_offer = adapter.resolve_action(
            proposed_action=Action(
                "OFFER_TRADE",
                payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
            ),
            legal_actions=decision.legal_actions,
        )
        self.assertEqual(canonical_offer.action_type, "OFFER_TRADE")

        transition = adapter.apply_action(canonical_offer)
        self.assertTrue(any(event.kind == "trade_offered" for event in transition.public_events))
        offered_event = transition.public_events[0]
        self.assertEqual(offered_event.actor_player_id, decision.acting_player_id)
        self.assertEqual(offered_event.payload["offer"], {"WOOD": 1})
        self.assertEqual(offered_event.payload["request"], {"BRICK": 1})
        self.assertEqual(adapter.current_decision().phase, "decide_trade")
        self.assertTrue(
            all(
                action.action_type in {"ACCEPT_TRADE", "REJECT_TRADE", "COUNTER_OFFER"}
                for action in adapter.current_decision().legal_actions
            )
        )

    def test_decide_trade_exposes_counter_offer_template(self) -> None:
        adapter = CatanatronEngineAdapter(seed=1)

        while adapter.current_decision().phase != "play_turn":
            adapter.apply_action(adapter.current_decision().legal_actions[0])
        adapter.apply_action(Action("ROLL"))
        while adapter.current_decision().phase != "play_turn":
            adapter.apply_action(adapter.current_decision().legal_actions[0])

        decision = adapter.current_decision()
        adapter.apply_action(
            adapter.resolve_action(
                proposed_action=Action(
                    "OFFER_TRADE",
                    payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
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


if __name__ == "__main__":
    unittest.main()
