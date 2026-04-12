"""
Kuhn's theorem: behavioral-strategy / mixed-strategy equivalence.

Under *perfect recall* every mixed strategy induces a unique behavioral
strategy (and vice versa) with the same distribution over terminal
histories.  This module provides:

* ``check_perfect_recall`` -- verify that a game satisfies perfect recall.
* ``behavioral_to_mixed``  -- convert a behavioral strategy to a mixed
  strategy (probability over pure strategies).
* ``mixed_to_behavioral``  -- convert a mixed strategy to a behavioral
  strategy (action probabilities at each information set).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.core.game_tree import ChanceNode, GameNode, GameTree, Node, TerminalNode


# ------------------------------------------------------------------ #
#  Perfect-recall check                                               #
# ------------------------------------------------------------------ #

def _player_sequence_at(
    game: GameTree,
    target: GameNode,
    player: int,
) -> List[Tuple[str, str]]:
    """Return the sequence of (infoset_id, action) pairs for *player*
    on the path from the root to *target* (exclusive of *target*'s own
    decision).
    """

    def dfs(
        node: Node,
        seq: List[Tuple[str, str]],
    ) -> Optional[List[Tuple[str, str]]]:
        if node is target:
            return list(seq)
        if node.is_terminal:
            return None
        for action, child in node.children.items():
            if isinstance(node, GameNode) and node.player == player:
                seq.append((node.infoset_id, action))
                result = dfs(child, seq)
                seq.pop()
            else:
                result = dfs(child, seq)
            if result is not None:
                return result
        return None

    result = dfs(game.root, [])
    return result if result is not None else []


def check_perfect_recall(game: GameTree) -> Tuple[bool, str]:
    """Check whether the game satisfies perfect recall.

    Perfect recall requires that for every player *p* and every
    information set *h* of *p*, all nodes in *h* share the same
    sequence of earlier information sets and actions by *p*.

    Returns
    -------
    (is_valid, message)
        ``is_valid`` is True when the game has perfect recall.
    """
    for player in game.players:
        for iset_id, nodes in game.information_sets(player).items():
            if len(nodes) <= 1:
                continue
            reference_seq = _player_sequence_at(game, nodes[0], player)
            for node in nodes[1:]:
                seq = _player_sequence_at(game, node, player)
                if seq != reference_seq:
                    return (
                        False,
                        f"Player {player}, info set '{iset_id}': nodes have "
                        f"different player-sequences {reference_seq} vs {seq}.",
                    )
    return True, "Perfect recall holds."


# ------------------------------------------------------------------ #
#  Behavioral -> Mixed                                                #
# ------------------------------------------------------------------ #

def _enumerate_pure_strategies(
    game: GameTree,
    player: int,
) -> List[Dict[str, str]]:
    """Return list of pure strategies (one action per info set)."""
    from itertools import product as _product

    infosets = game.information_sets(player)
    if not infosets:
        return [{}]
    iset_ids = sorted(infosets.keys())
    action_lists = [infosets[iid][0].actions for iid in iset_ids]
    return [dict(zip(iset_ids, combo)) for combo in _product(*action_lists)]


def behavioral_to_mixed(
    game: GameTree,
    player: int,
    behavioral: Dict[str, Dict[str, float]],
) -> Dict[int, float]:
    """Convert a behavioral strategy to a mixed strategy.

    Parameters
    ----------
    game : GameTree
    player : int
    behavioral : dict[str, dict[str, float]]
        ``behavioral[infoset_id][action]`` is the probability of *action*
        at information set *infoset_id*.

    Returns
    -------
    dict[int, float]
        ``mixed[i]`` is the probability of the *i*-th pure strategy
        (as ordered by ``_enumerate_pure_strategies``).
    """
    ok, msg = check_perfect_recall(game)
    if not ok:
        raise ValueError(f"Kuhn's theorem requires perfect recall: {msg}")

    pure_strategies = _enumerate_pure_strategies(game, player)
    mixed: Dict[int, float] = {}
    for idx, pure in enumerate(pure_strategies):
        prob = 1.0
        for iset_id, action in pure.items():
            prob *= behavioral.get(iset_id, {}).get(action, 0.0)
        mixed[idx] = prob
    return mixed


# ------------------------------------------------------------------ #
#  Mixed -> Behavioral                                                #
# ------------------------------------------------------------------ #

def mixed_to_behavioral(
    game: GameTree,
    player: int,
    mixed: Dict[int, float],
) -> Dict[str, Dict[str, float]]:
    """Convert a mixed strategy to a behavioral strategy.

    Parameters
    ----------
    game : GameTree
    player : int
    mixed : dict[int, float]
        ``mixed[i]`` is the probability of the *i*-th pure strategy.

    Returns
    -------
    dict[str, dict[str, float]]
        ``behavioral[infoset_id][action]`` is the probability of *action*.
    """
    ok, msg = check_perfect_recall(game)
    if not ok:
        raise ValueError(f"Kuhn's theorem requires perfect recall: {msg}")

    pure_strategies = _enumerate_pure_strategies(game, player)
    infosets = game.information_sets(player)

    behavioral: Dict[str, Dict[str, float]] = {}

    for iset_id in sorted(infosets.keys()):
        actions = infosets[iset_id][0].actions
        behavioral[iset_id] = {a: 0.0 for a in actions}

        # Find all ancestors of this info set for the player.
        representative = infosets[iset_id][0]
        ancestor_seq = _player_sequence_at(game, representative, player)
        ancestor_isets = {h_id for h_id, _ in ancestor_seq}

        # Sum mixed[i] over pure strategies that are consistent with
        # reaching this info set.
        for idx, pure in enumerate(pure_strategies):
            # Check that pure strategy is consistent with ancestor sequence.
            consistent = True
            for h_id, a in ancestor_seq:
                if pure.get(h_id) != a:
                    consistent = False
                    break
            if not consistent:
                continue

            action_chosen = pure[iset_id]
            behavioral[iset_id][action_chosen] += mixed.get(idx, 0.0)

        # Normalize.
        total = sum(behavioral[iset_id].values())
        if total > 1e-12:
            for a in actions:
                behavioral[iset_id][a] /= total
        else:
            # Info set unreachable under this strategy; use uniform.
            for a in actions:
                behavioral[iset_id][a] = 1.0 / len(actions)

    return behavioral
