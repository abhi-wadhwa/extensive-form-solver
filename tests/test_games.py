"""Tests for preset game construction and properties."""

from __future__ import annotations

import pytest

from src.core.game_tree import ChanceNode, GameNode, TerminalNode
from src.core.kuhn_theorem import (
    check_perfect_recall,
    behavioral_to_mixed,
    mixed_to_behavioral,
)
from src.games.centipede import build_centipede
from src.games.entry_deterrence import build_entry_deterrence
from src.games.kuhn_poker import build_kuhn_poker


class TestCentipedeConstruction:
    """Verify the structure of centipede games."""

    def test_4_round_node_count(self):
        game = build_centipede(4)
        # 4 decision nodes + 5 terminals.
        assert len(game.nodes()) == 9
        assert len(game.terminal_nodes()) == 5

    def test_6_round_node_count(self):
        game = build_centipede(6)
        assert len(game.decision_nodes()) == 6
        assert len(game.terminal_nodes()) == 7

    def test_single_round(self):
        game = build_centipede(1)
        assert len(game.decision_nodes()) == 1
        assert len(game.terminal_nodes()) == 2

    def test_invalid_rounds(self):
        with pytest.raises(ValueError):
            build_centipede(0)

    def test_perfect_recall(self):
        game = build_centipede(4)
        ok, msg = check_perfect_recall(game)
        assert ok


class TestEntryDeterrenceConstruction:

    def test_node_count(self):
        game = build_entry_deterrence()
        assert len(game.nodes()) == 5  # 2 decision + 3 terminal
        assert len(game.terminal_nodes()) == 3

    def test_perfect_recall(self):
        game = build_entry_deterrence()
        ok, _ = check_perfect_recall(game)
        assert ok


class TestKuhnPokerConstruction:

    def test_chance_root(self):
        game = build_kuhn_poker()
        assert isinstance(game.root, ChanceNode)
        assert len(game.root.children) == 6  # 3P2 = 6 deals

    def test_probabilities(self):
        game = build_kuhn_poker()
        for p in game.root.distribution.values():
            assert p == pytest.approx(1.0 / 6.0)

    def test_zero_sum(self):
        game = build_kuhn_poker()
        for t in game.terminal_nodes():
            assert sum(t.payoffs.values()) == pytest.approx(0.0)

    def test_perfect_recall(self):
        game = build_kuhn_poker()
        ok, msg = check_perfect_recall(game)
        assert ok, msg


class TestKuhnTheorem:
    """Test behavioral <-> mixed strategy conversion."""

    def test_behavioral_to_mixed_entry_deterrence(self):
        game = build_entry_deterrence()
        # Behavioral: P0 enters with prob 0.7, stays out 0.3.
        behavioral = {"Entrant": {"Enter": 0.7, "Stay Out": 0.3}}
        mixed = behavioral_to_mixed(game, 0, behavioral)
        # P0 has 2 pure strategies: {Entrant: Enter}, {Entrant: Stay Out}.
        assert sum(mixed.values()) == pytest.approx(1.0)
        # The mixed strategy should match the behavioral probabilities.
        probs = sorted(mixed.values())
        assert probs[0] == pytest.approx(0.3)
        assert probs[1] == pytest.approx(0.7)

    def test_mixed_to_behavioral_entry_deterrence(self):
        game = build_entry_deterrence()
        # Mixed: P1 plays strategy 0 (Fight or Accommodate) with some probs.
        # P1 strategies: sorted info sets => "Incumbent": [Accommodate, Fight]
        # (sorted alphabetically).
        isets = game.information_sets(1)
        actions = isets["Incumbent"][0].actions
        # strategies: [{Incumbent: Fight}, {Incumbent: Accommodate}]
        # or [{Incumbent: Accommodate}, {Incumbent: Fight}] depending on sort.
        from src.core.kuhn_theorem import _enumerate_pure_strategies
        strats = _enumerate_pure_strategies(game, 1)
        # Give 60% to first strategy, 40% to second.
        mixed = {0: 0.6, 1: 0.4}
        behavioral = mixed_to_behavioral(game, 1, mixed)
        # Behavioral should have probabilities summing to 1 at "Incumbent".
        total = sum(behavioral["Incumbent"].values())
        assert total == pytest.approx(1.0)

    def test_round_trip_centipede(self):
        """behavioral -> mixed -> behavioral should round-trip."""
        game = build_centipede(4)

        # P0 has info sets P0_r0 and P0_r2.
        behavioral_orig = {
            "P0_r0": {"Take": 0.4, "Pass": 0.6},
            "P0_r2": {"Take": 0.8, "Pass": 0.2},
        }
        mixed = behavioral_to_mixed(game, 0, behavioral_orig)
        assert sum(mixed.values()) == pytest.approx(1.0)

        behavioral_back = mixed_to_behavioral(game, 0, mixed)

        for iset_id in behavioral_orig:
            for action in behavioral_orig[iset_id]:
                assert behavioral_back[iset_id][action] == pytest.approx(
                    behavioral_orig[iset_id][action], abs=1e-9
                )

    def test_round_trip_kuhn_poker(self):
        """Round-trip for Kuhn Poker player 0."""
        game = build_kuhn_poker()

        isets = game.information_sets(0)
        # Build a uniform behavioral strategy.
        behavioral_orig = {}
        for iset_id, nodes in isets.items():
            actions = nodes[0].actions
            behavioral_orig[iset_id] = {a: 1.0 / len(actions) for a in actions}

        mixed = behavioral_to_mixed(game, 0, behavioral_orig)
        assert sum(mixed.values()) == pytest.approx(1.0)

        behavioral_back = mixed_to_behavioral(game, 0, mixed)

        for iset_id in behavioral_orig:
            for action in behavioral_orig[iset_id]:
                assert behavioral_back[iset_id][action] == pytest.approx(
                    behavioral_orig[iset_id][action], abs=1e-6
                )
