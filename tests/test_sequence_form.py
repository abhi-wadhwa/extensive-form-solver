"""Tests for the sequence-form LP solver."""

from __future__ import annotations

import pytest

from src.core.sequence_form import sequence_form_solve
from src.core.game_tree import (
    ChanceNode,
    GameNode,
    GameTree,
    TerminalNode,
)
from src.games.kuhn_poker import build_kuhn_poker


class TestMatchingPennies:
    """Simple simultaneous zero-sum game encoded as extensive form.

    Matching Pennies:
        P0 chooses H/T, then P1 (not knowing P0's choice) chooses H/T.
        If same => P0 wins +1, P1 wins -1.  Otherwise reversed.
    Value = 0 (symmetric).
    """

    def _build(self) -> GameTree:
        t_hh = TerminalNode(payoffs={0: 1.0, 1: -1.0}, name="HH")
        t_ht = TerminalNode(payoffs={0: -1.0, 1: 1.0}, name="HT")
        t_th = TerminalNode(payoffs={0: -1.0, 1: 1.0}, name="TH")
        t_tt = TerminalNode(payoffs={0: 1.0, 1: -1.0}, name="TT")

        # P1 has a single info set (doesn't see P0's choice).
        p1_after_h = GameNode(
            player=1, actions=["H", "T"],
            children={"H": t_hh, "T": t_ht},
            infoset_id="P1",
        )
        p1_after_t = GameNode(
            player=1, actions=["H", "T"],
            children={"H": t_th, "T": t_tt},
            infoset_id="P1",
        )

        root = GameNode(
            player=0, actions=["H", "T"],
            children={"H": p1_after_h, "T": p1_after_t},
            infoset_id="P0",
        )
        return GameTree(root=root, players=[0, 1], title="Matching Pennies")

    def test_game_value_zero(self):
        game = self._build()
        plans, gv = sequence_form_solve(game)
        assert gv == pytest.approx(0.0, abs=1e-6)

    def test_realization_plans_sum(self):
        game = self._build()
        plans, gv = sequence_form_solve(game)
        # Each player's non-empty sequences should sum to 1.
        p0_sum = sum(v for k, v in plans["P0"].items() if k != "(empty)")
        p1_sum = sum(v for k, v in plans["P1"].items() if k != "(empty)")
        assert p0_sum == pytest.approx(1.0, abs=1e-6)
        assert p1_sum == pytest.approx(1.0, abs=1e-6)

    def test_equilibrium_is_half_half(self):
        game = self._build()
        plans, gv = sequence_form_solve(game)
        # Equilibrium of matching pennies: each player mixes 50-50.
        # Sequence keys are now "infoset:action", e.g. "P0:H", "P1:T".
        h_prob_p0 = plans["P0"].get("P0:H", 0.0)
        t_prob_p0 = plans["P0"].get("P0:T", 0.0)
        assert h_prob_p0 == pytest.approx(0.5, abs=1e-4)
        assert t_prob_p0 == pytest.approx(0.5, abs=1e-4)

        h_prob_p1 = plans["P1"].get("P1:H", 0.0)
        t_prob_p1 = plans["P1"].get("P1:T", 0.0)
        assert h_prob_p1 == pytest.approx(0.5, abs=1e-4)
        assert t_prob_p1 == pytest.approx(0.5, abs=1e-4)


class TestKuhnPokerValue:
    """The known value of Kuhn Poker to player 0 is -1/18."""

    def test_game_value(self):
        game = build_kuhn_poker()
        plans, gv = sequence_form_solve(game)
        assert gv == pytest.approx(-1.0 / 18.0, abs=1e-4)

    def test_realization_plans_valid(self):
        game = build_kuhn_poker()
        plans, gv = sequence_form_solve(game)
        # Empty sequence has probability 1.
        assert plans["P0"]["(empty)"] == pytest.approx(1.0, abs=1e-6)
        assert plans["P1"]["(empty)"] == pytest.approx(1.0, abs=1e-6)


class TestRockPaperScissors:
    """RPS as a zero-sum extensive-form game (simultaneous)."""

    def _build(self) -> GameTree:
        actions = ["R", "P", "S"]
        # Payoff table for P0: R>S, S>P, P>R.
        payoff_map = {
            ("R", "R"): 0, ("R", "P"): -1, ("R", "S"): 1,
            ("P", "R"): 1, ("P", "P"): 0,  ("P", "S"): -1,
            ("S", "R"): -1, ("S", "P"): 1, ("S", "S"): 0,
        }

        p1_nodes = {}
        for a0 in actions:
            children = {}
            for a1 in actions:
                v = payoff_map[(a0, a1)]
                children[a1] = TerminalNode(
                    payoffs={0: float(v), 1: float(-v)},
                    name=f"{a0}{a1}",
                )
            p1_nodes[a0] = GameNode(
                player=1, actions=actions,
                children=children,
                infoset_id="P1",  # P1 doesn't see P0's choice
            )

        root = GameNode(
            player=0, actions=actions,
            children={a: p1_nodes[a] for a in actions},
            infoset_id="P0",
        )
        return GameTree(root=root, players=[0, 1], title="RPS")

    def test_value_zero(self):
        game = self._build()
        _, gv = sequence_form_solve(game)
        assert gv == pytest.approx(0.0, abs=1e-6)

    def test_uniform_mixing(self):
        game = self._build()
        plans, _ = sequence_form_solve(game)
        # Sequence keys: "P0:R", "P0:P", "P0:S" etc.
        for a in ["R", "P", "S"]:
            assert plans["P0"].get(f"P0:{a}", 0.0) == pytest.approx(1 / 3, abs=1e-4)
            assert plans["P1"].get(f"P1:{a}", 0.0) == pytest.approx(1 / 3, abs=1e-4)


class TestNonZeroSumRejection:
    """Sequence-form LP should reject non-zero-sum games."""

    def test_raises_non_zero_sum(self):
        t = TerminalNode(payoffs={0: 1.0, 1: 1.0}, name="t")
        root = GameNode(
            player=0, actions=["A"],
            children={"A": t},
            infoset_id="P0",
        )
        game = GameTree(root=root, players=[0, 1], title="NonZeroSum")
        with pytest.raises(ValueError, match="not zero-sum"):
            sequence_form_solve(game)
