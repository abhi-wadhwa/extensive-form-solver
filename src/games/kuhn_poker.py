"""
Kuhn Poker -- the simplest non-trivial poker game.

* Deck: {J, Q, K}  (one card each)
* Two players each ante 1 chip then receive one card.
* Player 0 acts first: Check or Bet (add 1 chip).
  - If Check, Player 1: Check (showdown) or Bet.
    - If Player 1 Bets, Player 0: Fold or Call.
  - If Bet, Player 1: Fold or Call.
* Showdown: higher card wins the pot.

This is a 2-player zero-sum game with imperfect information
(each player sees only their own card).

There are 6 chance outcomes (permutations of dealing 2 out of 3 cards).
"""

from __future__ import annotations

from itertools import permutations
from typing import Dict

from src.core.game_tree import (
    ChanceNode,
    GameNode,
    GameTree,
    Node,
    TerminalNode,
)

_CARDS = ["J", "Q", "K"]
_RANK = {"J": 0, "Q": 1, "K": 2}


def _showdown_payoff(card0: str, card1: str, pot: int) -> Dict[int, float]:
    """Player with higher card wins the pot."""
    if _RANK[card0] > _RANK[card1]:
        return {0: float(pot // 2), 1: float(-(pot // 2))}
    else:
        return {0: float(-(pot // 2)), 1: float(pot // 2)}


def _build_deal(card0: str, card1: str) -> Node:
    """Build the subtree for a specific deal (card0 to P0, card1 to P1).

    Pot starts at 2 (each player anted 1).
    """

    # ---- Player 0 acts: Check / Bet ----------------------------- #
    # -- Branch: P0 Checks -> P1 acts: Check / Bet
    # ---- P1 Checks => showdown (pot=2)
    p1_check_after_check = TerminalNode(
        payoffs=_showdown_payoff(card0, card1, 2),
        name=f"SD({card0}v{card1},chk-chk)",
    )

    # ---- P1 Bets => P0 acts: Fold / Call
    p0_fold_after_cb = TerminalNode(
        payoffs={0: -1.0, 1: 1.0},
        name=f"P0fold({card0}v{card1},chk-bet)",
    )
    p0_call_after_cb = TerminalNode(
        payoffs=_showdown_payoff(card0, card1, 4),
        name=f"SD({card0}v{card1},chk-bet-call)",
    )
    p0_after_cb = GameNode(
        player=0,
        actions=["Fold", "Call"],
        children={"Fold": p0_fold_after_cb, "Call": p0_call_after_cb},
        infoset_id=f"P0_{card0}_cb",
        name=f"P0({card0})_after_chk-bet",
    )

    p1_after_check = GameNode(
        player=1,
        actions=["Check", "Bet"],
        children={"Check": p1_check_after_check, "Bet": p0_after_cb},
        infoset_id=f"P1_{card1}_chk",
        name=f"P1({card1})_after_chk",
    )

    # -- Branch: P0 Bets -> P1 acts: Fold / Call
    p1_fold_after_bet = TerminalNode(
        payoffs={0: 1.0, 1: -1.0},
        name=f"P1fold({card0}v{card1},bet)",
    )
    p1_call_after_bet = TerminalNode(
        payoffs=_showdown_payoff(card0, card1, 4),
        name=f"SD({card0}v{card1},bet-call)",
    )
    p1_after_bet = GameNode(
        player=1,
        actions=["Fold", "Call"],
        children={"Fold": p1_fold_after_bet, "Call": p1_call_after_bet},
        infoset_id=f"P1_{card1}_bet",
        name=f"P1({card1})_after_bet",
    )

    # Player 0's first decision.
    p0_root = GameNode(
        player=0,
        actions=["Check", "Bet"],
        children={"Check": p1_after_check, "Bet": p1_after_bet},
        infoset_id=f"P0_{card0}",
        name=f"P0({card0})_open",
    )

    return p0_root


def build_kuhn_poker() -> GameTree:
    """Build the full Kuhn Poker game tree with chance root."""

    deals = list(permutations(_CARDS, 2))  # 6 deals
    prob = 1.0 / len(deals)
    distribution = {f"{c0}{c1}": prob for c0, c1 in deals}

    children = {}
    for c0, c1 in deals:
        children[f"{c0}{c1}"] = _build_deal(c0, c1)

    root = ChanceNode(
        distribution=distribution,
        children=children,
        name="Deal",
    )

    return GameTree(root=root, players=[0, 1], title="Kuhn Poker")
