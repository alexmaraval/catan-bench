from __future__ import annotations

import random
from collections import deque
from typing import Iterable, Protocol

from .schemas import Action, Observation, PlayerResponse


class Player(Protocol):
    def respond(self, observation: Observation) -> PlayerResponse:
        ...


class ScriptedPlayer:
    """Small utility adapter for tests and scripted demos."""

    def __init__(self, responses: Iterable[PlayerResponse | Action]) -> None:
        self._responses = deque(responses)
        self.observations: list[Observation] = []

    def respond(self, observation: Observation) -> PlayerResponse:
        self.observations.append(observation)
        if not self._responses:
            raise RuntimeError("ScriptedPlayer ran out of scripted responses.")

        next_response = self._responses.popleft()
        if isinstance(next_response, PlayerResponse):
            return next_response
        return PlayerResponse(action=next_response)


def _is_trade_template(action: Action) -> bool:
    return action.action_type == "OFFER_TRADE" and action.payload == {
        "offer": {},
        "request": {},
    }


def _materialize_default_trade_offer() -> Action:
    return Action(
        "OFFER_TRADE",
        payload={"offer": {"WOOD": 1}, "request": {"BRICK": 1}},
    )


class FirstLegalPlayer:
    """Simple deterministic baseline that always picks the first legal action."""

    def __init__(
        self, *, allow_trade_offers: bool = False, record_observations: bool = False
    ) -> None:
        self.allow_trade_offers = allow_trade_offers
        self.record_observations = record_observations
        self.observations: list[Observation] = []

    def respond(self, observation: Observation) -> PlayerResponse:
        if self.record_observations:
            self.observations.append(observation)
        for action in observation.legal_actions:
            if not _is_trade_template(action):
                return PlayerResponse(action=action)

        if self.allow_trade_offers:
            return PlayerResponse(action=_materialize_default_trade_offer())

        raise RuntimeError("FirstLegalPlayer found no concrete legal actions to take.")


class RandomLegalPlayer:
    """Random baseline that samples from the current legal concrete actions."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        allow_trade_offers: bool = False,
        record_observations: bool = False,
    ) -> None:
        self._rng = random.Random(seed)
        self.allow_trade_offers = allow_trade_offers
        self.record_observations = record_observations
        self.observations: list[Observation] = []

    def respond(self, observation: Observation) -> PlayerResponse:
        if self.record_observations:
            self.observations.append(observation)
        concrete_actions = [
            action for action in observation.legal_actions if not _is_trade_template(action)
        ]
        if concrete_actions:
            return PlayerResponse(action=self._rng.choice(concrete_actions))

        if self.allow_trade_offers:
            return PlayerResponse(action=_materialize_default_trade_offer())

        raise RuntimeError("RandomLegalPlayer found no concrete legal actions to sample.")
