"""Tests for backward induction solver."""

from __future__ import annotations

import pytest

from src.core.backward_induction import backward_induction
from src.core.game_tree import GameNode, GameTree, TerminalNode
from src.games.centipede import build_centipede
from src.games.entry_deterrence import build_entry_deterrence


class TestEntryDeterrence:
    """The classic entry-deterrence game."""

    def test_spe_enter_accommodate(self):
        game = build_entry_deterrence()
        eq, payoffs = backward_induction(game)

        # Incumbent accommodates (Fight gives -1, Accommodate gives 1).
        assert eq[1]["Incumbent"] == "Accommodate"
        # Entrant enters (Enter yields 1, Stay Out yields 0).
        assert eq[0]["Entrant"] == "Enter"
        # SPE payoffs.
        assert payoffs[0] == pytest.approx(1.0)
        assert payoffs[1] == pytest.approx(1.0)


class TestCentipede:
    """Backward induction in the centipede game predicts immediate Take."""

    def test_immediate_take_4_rounds(self):
        game = build_centipede(4)
        eq, payoffs = backward_induction(game)

        # Player 0 takes at round 0.
        assert eq[0]["P0_r0"] == "Take"
        # Payoffs: taker (P0) gets 2+0=2, other (P1) gets 0.
        assert payoffs[0] == pytest.approx(2.0)
        assert payoffs[1] == pytest.approx(0.0)

    def test_immediate_take_6_rounds(self):
        game = build_centipede(6)
        eq, payoffs = backward_induction(game)
        assert eq[0]["P0_r0"] == "Take"
        assert payoffs[0] == pytest.approx(2.0)

    def test_single_round(self):
        game = build_centipede(1)
        eq, payoffs = backward_induction(game)
        # Only one decision node, player 0: Take gives (2,0), Pass gives (1,3).
        assert eq[0]["P0_r0"] == "Take"
        assert payoffs[0] == pytest.approx(2.0)


class TestCustomGame:
    """Small hand-built game for sanity checks."""

    def test_simple_sequential(self):
        """P0 chooses L/R, then P1 chooses U/D (only on R branch).

             P0
            /  \\
           L    R
          (3,1) P1
                / \\
               U   D
             (0,0) (2,4)
        """
        t_l = TerminalNode(payoffs={0: 3.0, 1: 1.0}, name="L")
        t_u = TerminalNode(payoffs={0: 0.0, 1: 0.0}, name="U")
        t_d = TerminalNode(payoffs={0: 2.0, 1: 4.0}, name="D")

        p1_node = GameNode(
            player=1,
            actions=["U", "D"],
            children={"U": t_u, "D": t_d},
            infoset_id="P1_right",
            name="P1_right",
        )

        root = GameNode(
            player=0,
            actions=["L", "R"],
            children={"L": t_l, "R": p1_node},
            infoset_id="P0_root",
            name="P0_root",
        )

        game = GameTree(root=root, players=[0, 1], title="Simple Sequential")
        eq, payoffs = backward_induction(game)

        # P1 chooses D (4 > 0).
        assert eq[1]["P1_right"] == "D"
        # P0 chooses L (3 > 2).
        assert eq[0]["P0_root"] == "L"
        assert payoffs == {0: 3.0, 1: 1.0}


class TestImperfectInfoRejection:
    """Backward induction must raise on imperfect-information games."""

    def test_raises_on_shared_infoset(self):
        """Two nodes share the same info set for player 1."""
        t1 = TerminalNode(payoffs={0: 1.0, 1: 0.0}, name="t1")
        t2 = TerminalNode(payoffs={0: 0.0, 1: 1.0}, name="t2")
        t3 = TerminalNode(payoffs={0: 2.0, 1: 0.0}, name="t3")
        t4 = TerminalNode(payoffs={0: 0.0, 1: 2.0}, name="t4")

        # Two P1 nodes share infoset "P1_shared".
        p1_left = GameNode(
            player=1, actions=["A", "B"],
            children={"A": t1, "B": t2},
            infoset_id="P1_shared",
        )
        p1_right = GameNode(
            player=1, actions=["A", "B"],
            children={"A": t3, "B": t4},
            infoset_id="P1_shared",
        )

        root = GameNode(
            player=0, actions=["L", "R"],
            children={"L": p1_left, "R": p1_right},
            infoset_id="P0_root",
        )

        game = GameTree(root=root, players=[0, 1], title="Imperfect")
        with pytest.raises(ValueError, match="perfect information"):
            backward_induction(game)
