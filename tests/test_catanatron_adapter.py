from __future__ import annotations

import unittest

try:
    from catan_bench.catanatron_adapter import CatanatronEngineAdapter
except RuntimeError:  # pragma: no cover - dependency missing in some environments.
    CatanatronEngineAdapter = None

from catan_bench import Action


@unittest.skipIf(CatanatronEngineAdapter is None, "catanatron dependency is not installed")
class CatanatronAdapterTests(unittest.TestCase):
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
                action.action_type in {"ACCEPT_TRADE", "REJECT_TRADE"}
                for action in adapter.current_decision().legal_actions
            )
        )


if __name__ == "__main__":
    unittest.main()
