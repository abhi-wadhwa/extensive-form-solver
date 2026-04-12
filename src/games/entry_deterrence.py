"""
Entry Deterrence game.

Player 0 (Entrant) chooses to Enter or Stay Out.
If Enter, Player 1 (Incumbent) chooses to Fight or Accommodate.

Payoffs:
    Stay Out  => (0, 2)
    Enter + Accommodate => (1, 1)
    Enter + Fight => (-1, -1)

Backward induction: Incumbent accommodates, so Entrant enters.
"""

from __future__ import annotations

from src.core.game_tree import GameNode, GameTree, TerminalNode


def build_entry_deterrence() -> GameTree:
    """Build the classic Entry Deterrence game."""

    # Terminal nodes.
    stay_out = TerminalNode(payoffs={0: 0.0, 1: 2.0}, name="Stay Out")
    fight = TerminalNode(payoffs={0: -1.0, 1: -1.0}, name="Fight")
    accommodate = TerminalNode(payoffs={0: 1.0, 1: 1.0}, name="Accommodate")

    # Incumbent's decision (only reached if Entrant enters).
    incumbent = GameNode(
        player=1,
        actions=["Fight", "Accommodate"],
        children={"Fight": fight, "Accommodate": accommodate},
        infoset_id="Incumbent",
        name="Incumbent",
    )

    # Entrant's decision.
    entrant = GameNode(
        player=0,
        actions=["Enter", "Stay Out"],
        children={"Enter": incumbent, "Stay Out": stay_out},
        infoset_id="Entrant",
        name="Entrant",
    )

    return GameTree(root=entrant, players=[0, 1], title="Entry Deterrence")
