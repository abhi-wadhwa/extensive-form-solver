"""Tests for normal-form conversion."""

from __future__ import annotations

import random

import pytest
import numpy as np

from src.core.game_tree import GameNode, GameTree, TerminalNode, ChanceNode
from src.core.normal_form import to_normal_form, _evaluate
from src.games.centipede import build_centipede
from src.games.entry_deterrence import build_entry_deterrence
from src.games.kuhn_poker import build_kuhn_poker


class TestEntryDeterrence:
    """Entry Deterrence has 2 strategies for P0 and 2 for P1."""

    def test_strategy_counts(self):
        game = build_entry_deterrence()
        nf = to_normal_form(game)
        assert len(nf.strategy_labels[0]) == 2  # Enter, Stay Out
        assert len(nf.strategy_labels[1]) == 2  # Fight, Accommodate

    def test_payoff_matrix(self):
        game = build_entry_deterrence()
        nf = to_normal_form(game)
        # Strategies for P0: sorted by infoset -> action.
        # P0 info set "Entrant": [Enter, Stay Out]
        # P1 info set "Incumbent": [Accommodate, Fight]
        # We need to find which index is which.
        p0_strats = nf.strategy_labels[0]
        p1_strats = nf.strategy_labels[1]

        # Find indices.
        enter_idx = next(
            i for i, s in enumerate(p0_strats) if s.get("Entrant") == "Enter"
        )
        stay_idx = next(
            i for i, s in enumerate(p0_strats) if s.get("Entrant") == "Stay Out"
        )
        accom_idx = next(
            i for i, s in enumerate(p1_strats) if s.get("Incumbent") == "Accommodate"
        )
        fight_idx = next(
            i for i, s in enumerate(p1_strats) if s.get("Incumbent") == "Fight"
        )

        m0 = nf.payoff_matrices[0]
        # Enter + Accommodate => (1, 1)
        assert m0[enter_idx, accom_idx] == pytest.approx(1.0)
        # Enter + Fight => (-1, -1)
        assert m0[enter_idx, fight_idx] == pytest.approx(-1.0)
        # Stay Out + anything => (0, 2)
        assert m0[stay_idx, accom_idx] == pytest.approx(0.0)
        assert m0[stay_idx, fight_idx] == pytest.approx(0.0)


class TestCentipede:
    """Normal form of the 4-round centipede game."""

    def test_strategy_enumeration(self):
        game = build_centipede(4)
        nf = to_normal_form(game)
        # P0 has 2 info sets (r0, r2), 2 actions each => 4 strategies.
        assert len(nf.strategy_labels[0]) == 4
        # P1 has 2 info sets (r1, r3), 2 actions each => 4 strategies.
        assert len(nf.strategy_labels[1]) == 4

    def test_payoff_matrix_shape(self):
        game = build_centipede(4)
        nf = to_normal_form(game)
        assert nf.payoff_matrices[0].shape == (4, 4)


class TestPayoffEquivalenceBySampling:
    """Verify that the normal-form payoffs match the extensive form
    by sampling random strategy profiles and comparing."""

    def test_entry_deterrence_sampling(self):
        game = build_entry_deterrence()
        nf = to_normal_form(game)
        random.seed(42)

        for _ in range(20):
            idx0 = random.randrange(len(nf.strategy_labels[0]))
            idx1 = random.randrange(len(nf.strategy_labels[1]))
            nf_pay = nf.payoff((idx0, idx1))

            profile = {
                0: nf.strategy_labels[0][idx0],
                1: nf.strategy_labels[1][idx1],
            }
            ef_pay = _evaluate(game.root, profile)

            for p in game.players:
                assert nf_pay[p] == pytest.approx(ef_pay[p], abs=1e-9)

    def test_centipede_sampling(self):
        game = build_centipede(4)
        nf = to_normal_form(game)
        random.seed(123)

        for _ in range(30):
            idx0 = random.randrange(len(nf.strategy_labels[0]))
            idx1 = random.randrange(len(nf.strategy_labels[1]))
            nf_pay = nf.payoff((idx0, idx1))

            profile = {
                0: nf.strategy_labels[0][idx0],
                1: nf.strategy_labels[1][idx1],
            }
            ef_pay = _evaluate(game.root, profile)

            for p in game.players:
                assert nf_pay[p] == pytest.approx(ef_pay[p], abs=1e-9)

    def test_kuhn_poker_sampling(self):
        """Sample strategy profiles in Kuhn Poker and verify payoff equivalence."""
        game = build_kuhn_poker()
        nf = to_normal_form(game)
        random.seed(999)

        for _ in range(50):
            idx0 = random.randrange(len(nf.strategy_labels[0]))
            idx1 = random.randrange(len(nf.strategy_labels[1]))
            nf_pay = nf.payoff((idx0, idx1))

            profile = {
                0: nf.strategy_labels[0][idx0],
                1: nf.strategy_labels[1][idx1],
            }
            ef_pay = _evaluate(game.root, profile)

            for p in game.players:
                assert nf_pay[p] == pytest.approx(ef_pay[p], abs=1e-9)
