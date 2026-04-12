"""
Backward induction for perfect-information extensive-form games.

Returns the Subgame Perfect Equilibrium (SPE) strategy profile and
the expected payoff vector at the root.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.core.game_tree import ChanceNode, GameNode, GameTree, Node, TerminalNode


def _backward_induction_rec(
    node: Node,
    strategy: Dict[int, Dict[str, str]],
) -> Dict[int, float]:
    """Recursively compute the SPE value and fill *strategy*.

    Parameters
    ----------
    node : Node
        Current node being solved.
    strategy : dict
        Mutated in-place.  ``strategy[player][infoset_id] = action``.

    Returns
    -------
    dict[int, float]
        Expected payoff vector (one entry per player) at *node*.
    """
    # --- terminal ------------------------------------------------- #
    if isinstance(node, TerminalNode):
        return dict(node.payoffs)

    # --- chance --------------------------------------------------- #
    if isinstance(node, ChanceNode):
        expected: Dict[int, float] = {}
        for action, child in node.children.items():
            child_payoffs = _backward_induction_rec(child, strategy)
            prob = node.distribution[action]
            for pid, val in child_payoffs.items():
                expected[pid] = expected.get(pid, 0.0) + prob * val
        return expected

    # --- decision (GameNode) -------------------------------------- #
    assert isinstance(node, GameNode)
    best_action: Optional[str] = None
    best_value: Optional[float] = None
    best_payoffs: Dict[int, float] = {}

    for action in node.actions:
        child = node.children[action]
        child_payoffs = _backward_induction_rec(child, strategy)
        val = child_payoffs.get(node.player, 0.0)
        if best_value is None or val > best_value:
            best_value = val
            best_action = action
            best_payoffs = child_payoffs

    # Record the optimal action for this information set.
    strategy.setdefault(node.player, {})[node.infoset_id] = best_action  # type: ignore[arg-type]
    return best_payoffs


def backward_induction(
    game: GameTree,
) -> Tuple[Dict[int, Dict[str, str]], Dict[int, float]]:
    """Solve a perfect-information game by backward induction.

    Parameters
    ----------
    game : GameTree
        Must be a perfect-information game (each information set has
        exactly one node, and no chance nodes share info sets with
        decision nodes).

    Returns
    -------
    strategy : dict[int, dict[str, str]]
        ``strategy[player][infoset_id]`` is the action chosen by *player*
        at information set *infoset_id* under the SPE.
    root_payoffs : dict[int, float]
        Expected payoff for each player at the root under the SPE.

    Raises
    ------
    ValueError
        If the game has an information set with more than one node
        (imperfect information).
    """
    # Validate perfect information.
    for player in game.players:
        for iset_id, nodes in game.information_sets(player).items():
            if len(nodes) > 1:
                raise ValueError(
                    f"Backward induction requires perfect information, but "
                    f"player {player}'s information set '{iset_id}' contains "
                    f"{len(nodes)} nodes."
                )

    strategy: Dict[int, Dict[str, str]] = {}
    root_payoffs = _backward_induction_rec(game.root, strategy)
    return strategy, root_payoffs
