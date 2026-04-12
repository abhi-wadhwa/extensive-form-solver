"""
Convert an extensive-form game to its normal (strategic) form.

A *pure strategy* for a player is a complete contingency plan: one
action for every information set of that player.  The normal form
enumerates all combinations of pure strategies and tabulates the
expected payoffs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Tuple

import numpy as np

from src.core.game_tree import ChanceNode, GameNode, GameTree, Node, TerminalNode


@dataclass
class NormalFormGame:
    """Strategic-form representation of a game.

    Attributes
    ----------
    players : list[int]
        Player identifiers.
    strategy_labels : dict[int, list[dict[str, str]]]
        ``strategy_labels[p]`` is a list of pure strategies for player *p*.
        Each pure strategy is ``{infoset_id: action, ...}``.
    payoff_matrices : dict[int, np.ndarray]
        ``payoff_matrices[p]`` is an n-dimensional array whose shape is
        ``(|S_0|, |S_1|, ..., |S_{n-1}|)`` and whose entry at index
        ``(i_0, i_1, ...)`` is the payoff to player *p* when each player *k*
        plays their *i_k*-th pure strategy.
    """

    players: List[int]
    strategy_labels: Dict[int, List[Dict[str, str]]]
    payoff_matrices: Dict[int, np.ndarray] = field(repr=False)

    def __repr__(self) -> str:
        sizes = {p: len(self.strategy_labels[p]) for p in self.players}
        return f"NormalFormGame(players={self.players}, strategies={sizes})"

    def payoff(self, strategy_indices: Tuple[int, ...]) -> Dict[int, float]:
        """Return payoff dict for a given combination of strategy indices."""
        return {
            p: float(self.payoff_matrices[p][strategy_indices])
            for p in self.players
        }


def _enumerate_strategies(
    game: GameTree,
    player: int,
) -> List[Dict[str, str]]:
    """Enumerate all pure strategies for *player*.

    A pure strategy assigns one action to every information set.
    """
    infosets = game.information_sets(player)
    if not infosets:
        return [{}]

    iset_ids = sorted(infosets.keys())
    action_lists = []
    for iset_id in iset_ids:
        representative = infosets[iset_id][0]
        action_lists.append(representative.actions)

    strategies: List[Dict[str, str]] = []
    for combo in product(*action_lists):
        strategies.append(dict(zip(iset_ids, combo)))
    return strategies


def _evaluate(
    node: Node,
    strategy_profile: Dict[int, Dict[str, str]],
) -> Dict[int, float]:
    """Compute the expected payoff vector at *node* given *strategy_profile*.

    Parameters
    ----------
    node : Node
    strategy_profile : dict[int, dict[str, str]]
        ``strategy_profile[player][infoset_id]`` is the action chosen.

    Returns
    -------
    dict[int, float]
    """
    if isinstance(node, TerminalNode):
        return dict(node.payoffs)

    if isinstance(node, ChanceNode):
        result: Dict[int, float] = {}
        for action, child in node.children.items():
            child_pay = _evaluate(child, strategy_profile)
            prob = node.distribution[action]
            for pid, val in child_pay.items():
                result[pid] = result.get(pid, 0.0) + prob * val
        return result

    assert isinstance(node, GameNode)
    action = strategy_profile[node.player].get(node.infoset_id)
    if action is None:
        raise ValueError(
            f"Strategy profile for player {node.player} has no action for "
            f"infoset '{node.infoset_id}'."
        )
    child = node.children[action]
    return _evaluate(child, strategy_profile)


def to_normal_form(game: GameTree) -> NormalFormGame:
    """Convert *game* to its normal-form representation.

    Parameters
    ----------
    game : GameTree

    Returns
    -------
    NormalFormGame
    """
    strats: Dict[int, List[Dict[str, str]]] = {}
    for p in game.players:
        strats[p] = _enumerate_strategies(game, p)

    # Determine the shape of the payoff tensor.
    shape = tuple(len(strats[p]) for p in game.players)

    payoffs: Dict[int, np.ndarray] = {
        p: np.zeros(shape) for p in game.players
    }

    # Iterate over all combinations of pure strategies.
    for idx_combo in product(*(range(s) for s in shape)):
        profile: Dict[int, Dict[str, str]] = {}
        for k, p in enumerate(game.players):
            profile[p] = strats[p][idx_combo[k]]

        result = _evaluate(game.root, profile)
        for p in game.players:
            payoffs[p][idx_combo] = result.get(p, 0.0)

    return NormalFormGame(
        players=list(game.players),
        strategy_labels=strats,
        payoff_matrices=payoffs,
    )
