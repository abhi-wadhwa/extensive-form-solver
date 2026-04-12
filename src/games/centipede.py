"""
Centipede game.

Two players alternate between "Take" (end the game) and "Pass"
(continue).  The pot grows each round, but the player who takes
gets a larger share.  Backward induction predicts immediate taking.

With *n* rounds the payoffs at round *k* (0-indexed) are:
    Take by player k%2 at round k  =>  taker gets 2+k, other gets k
"""

from __future__ import annotations

from src.core.game_tree import GameNode, GameTree, TerminalNode


def build_centipede(rounds: int = 4) -> GameTree:
    """Build a centipede game with *rounds* decision points.

    Parameters
    ----------
    rounds : int
        Number of decision nodes (alternating between players 0 and 1).

    Returns
    -------
    GameTree
    """
    if rounds < 1:
        raise ValueError("Need at least 1 round.")

    def _build(k: int) -> GameNode:
        player = k % 2
        other = 1 - player

        # Payoffs if this player Takes at round k.
        take_payoffs = {player: 2.0 + k, other: float(k)}
        take_node = TerminalNode(payoffs=take_payoffs, name=f"Take_r{k}")

        if k == rounds - 1:
            # Last round: if Pass, the game ends with a "generous" outcome.
            pass_payoffs = {player: float(k + 1), other: float(k + 3)}
            pass_node = TerminalNode(payoffs=pass_payoffs, name=f"Pass_r{k}")
        else:
            pass_node = _build(k + 1)

        return GameNode(
            player=player,
            actions=["Take", "Pass"],
            children={"Take": take_node, "Pass": pass_node},
            infoset_id=f"P{player}_r{k}",
            name=f"P{player}_r{k}",
        )

    root = _build(0)
    return GameTree(root=root, players=[0, 1], title=f"Centipede({rounds})")
