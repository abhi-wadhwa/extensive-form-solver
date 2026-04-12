"""
Sequence-form linear program for 2-player zero-sum extensive-form games.

The sequence form avoids the exponential blowup of the normal form by
using *realization plans* (one probability variable per player-sequence)
subject to linear constraints that enforce behavioral consistency.

A *sequence* for a player is a tuple of ``(infoset_id, action)`` pairs
representing the ordered choices made by that player along some path
from the root.  The empty tuple ``()`` is always the first sequence.

We solve the LP with ``scipy.optimize.linprog``.

Reference: B. von Stengel, "Efficient Computation of Behavior
Strategies", Games and Economic Behavior, 1996.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

from src.core.game_tree import ChanceNode, GameNode, GameTree, Node, TerminalNode

# A sequence element is (infoset_id, action).
SeqElem = Tuple[str, str]
Sequence = Tuple[SeqElem, ...]


# ------------------------------------------------------------------ #
#  Internal helpers                                                   #
# ------------------------------------------------------------------ #

def _collect_sequences(
    game: GameTree,
    player: int,
) -> Tuple[List[Sequence], Dict[str, List[str]]]:
    """Return the list of sequences and the information-set action map.

    Returns
    -------
    sequences : list[Sequence]
        Ordered list of distinct sequences for *player*.
    infoset_actions : dict[str, list[str]]
        ``infoset_actions[h]`` gives the actions available at info set *h*.
    """
    seq_set: set = {()}
    infoset_actions: Dict[str, List[str]] = {}

    def dfs(node: Node, seq: Sequence) -> None:
        if node.is_terminal:
            return
        if isinstance(node, ChanceNode):
            for action, child in node.children.items():
                dfs(child, seq)
            return
        assert isinstance(node, GameNode)
        if node.player == player:
            infoset_actions.setdefault(node.infoset_id, list(node.actions))
            for action in node.actions:
                new_seq = seq + ((node.infoset_id, action),)
                seq_set.add(new_seq)
                dfs(node.children[action], new_seq)
        else:
            for action in node.actions:
                dfs(node.children[action], seq)

    dfs(game.root, ())
    sequences = sorted(seq_set, key=lambda s: (len(s), s))
    return sequences, infoset_actions


def _terminal_reach(
    node: Node,
    seq0: Sequence,
    seq1: Sequence,
    chance_prob: float,
) -> List[Tuple[Sequence, Sequence, float, Dict[int, float]]]:
    """Walk the tree, collecting (seq_p0, seq_p1, chance_prob, payoffs)
    at every terminal node.
    """
    results: List[Tuple[Sequence, Sequence, float, Dict[int, float]]] = []

    if isinstance(node, TerminalNode):
        results.append((seq0, seq1, chance_prob, node.payoffs))
        return results

    if isinstance(node, ChanceNode):
        for action, child in node.children.items():
            results.extend(
                _terminal_reach(
                    child, seq0, seq1,
                    chance_prob * node.distribution[action],
                )
            )
        return results

    assert isinstance(node, GameNode)
    for action in node.actions:
        child = node.children[action]
        elem: SeqElem = (node.infoset_id, action)
        if node.player == 0:
            results.extend(
                _terminal_reach(child, seq0 + (elem,), seq1, chance_prob)
            )
        else:
            results.extend(
                _terminal_reach(child, seq0, seq1 + (elem,), chance_prob)
            )
    return results


def _parent_sequence(
    game: GameTree,
    player: int,
    infoset_id: str,
) -> Sequence:
    """Find the sequence of (infoset_id, action) pairs for *player* on
    the path from the root to (any node in) information set *infoset_id*,
    *excluding* the action at *infoset_id* itself.
    """
    target_nodes = game.information_sets(player).get(infoset_id, [])
    if not target_nodes:
        return ()

    target = target_nodes[0]

    def dfs(node: Node, seq: Sequence) -> Optional[Sequence]:
        if node is target:
            return seq
        if node.is_terminal:
            return None
        for action, child in node.children.items():
            if isinstance(node, GameNode) and node.player == player:
                result = dfs(child, seq + ((node.infoset_id, action),))
            else:
                result = dfs(child, seq)
            if result is not None:
                return result
        return None

    result = dfs(game.root, ())
    return result if result is not None else ()


def _build_constraint_matrix(
    game: GameTree,
    player: int,
    sequences: List[Sequence],
    seq_idx: Dict[Sequence, int],
    infoset_actions: Dict[str, List[str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the sequence-form constraint matrix E and vector e.

    The constraint is  E * x = e  where x is the realization plan.

    There is one row for the empty sequence (x[()] = 1) and one row
    for each information set h:
        x[parent_seq(h)] - sum_{a in actions(h)} x[parent_seq(h) + (h,a)] = 0

    Returns
    -------
    E : ndarray of shape (1 + num_infosets, num_sequences)
    e : ndarray of shape (1 + num_infosets,)
    """
    infosets = game.information_sets(player)
    n_seq = len(sequences)
    n_rows = 1 + len(infosets)

    E = np.zeros((n_rows, n_seq))
    e = np.zeros(n_rows)

    # Row 0: x[()] = 1
    E[0, seq_idx[()]] = 1.0
    e[0] = 1.0

    row = 1
    for h_id in infosets:
        parent_seq = _parent_sequence(game, player, h_id)
        parent_idx = seq_idx.get(parent_seq)
        if parent_idx is not None:
            E[row, parent_idx] = 1.0
        actions = infoset_actions[h_id]
        for a in actions:
            child_seq = parent_seq + ((h_id, a),)
            child_idx = seq_idx.get(child_seq)
            if child_idx is not None:
                E[row, child_idx] = -1.0
        row += 1

    return E, e


# ------------------------------------------------------------------ #
#  Public API                                                         #
# ------------------------------------------------------------------ #

def _format_sequence(seq: Sequence) -> str:
    """Human-readable representation of a sequence."""
    if not seq:
        return "(empty)"
    return ",".join(f"{h}:{a}" for h, a in seq)


def sequence_form_solve(
    game: GameTree,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """Solve a 2-player zero-sum extensive-form game via the sequence form LP.

    Uses the standard LP formulation from von Stengel (1996):

    **Player 1's (minimizer's) LP:**
        max  e_1^T * p
        s.t. E_2^T * p  <=  A^T * x_1
             E_1 * x_1   =  e_1
             x_1         >= 0

    where x_1 is player 1's realization plan, p is the dual variable
    vector (one per sequence constraint of player 0), and A is the
    payoff matrix.  The optimal value is the game value to player 0.

    We reformulate as a single LP for each player.

    Parameters
    ----------
    game : GameTree
        Must have ``game.players == [0, 1]`` and payoffs that are
        zero-sum.

    Returns
    -------
    realization_plans : dict[str, dict[str, float]]
        Realization plans for P0 and P1.
    game_value : float
        The value of the game to player 0.
    """
    if sorted(game.players) != [0, 1]:
        raise ValueError("Sequence-form LP requires exactly 2 players [0,1].")

    # Verify zero-sum.
    for t in game.terminal_nodes():
        total = sum(t.payoffs.values())
        if abs(total) > 1e-9:
            raise ValueError(
                f"Game is not zero-sum at terminal '{t.name}': "
                f"payoffs sum to {total}."
            )

    # ---- enumerate sequences ------------------------------------ #
    seqs0, ia0 = _collect_sequences(game, 0)
    seqs1, ia1 = _collect_sequences(game, 1)

    s0_idx = {s: i for i, s in enumerate(seqs0)}
    s1_idx = {s: i for i, s in enumerate(seqs1)}

    n0 = len(seqs0)
    n1 = len(seqs1)

    # ---- payoff matrix ------------------------------------------ #
    # A[i, j] = payoff to player 0 when P0 plays sequence i, P1 plays
    #           sequence j, summed over all terminals weighted by chance.
    A = np.zeros((n0, n1))
    entries = _terminal_reach(game.root, (), (), 1.0)
    for sq0, sq1, cp, pays in entries:
        i = s0_idx.get(sq0)
        j = s1_idx.get(sq1)
        if i is not None and j is not None:
            A[i, j] += cp * pays[0]

    # ---- constraint matrices ------------------------------------ #
    E0, e0 = _build_constraint_matrix(game, 0, seqs0, s0_idx, ia0)
    E1, e1 = _build_constraint_matrix(game, 1, seqs1, s1_idx, ia1)

    # Number of constraints for each player.
    m0 = E0.shape[0]  # 1 + |infosets of P0|
    m1 = E1.shape[0]  # 1 + |infosets of P1|

    # ---- LP for Player 1's realization plan (and game value) ---- #
    #
    # The primal-dual pair from von Stengel:
    #
    # Player 0 (maximizer):
    #   max  e_0^T * q
    #   s.t. E_1^T * q  <=  A * x_0      (dual constraints)
    #        E_0 * x_0   =  e_0           (sequence constraints)
    #        x_0         >= 0
    #
    # This is not a standard LP because it mixes primal (x_0) and
    # dual (q) variables.  The game value = e_0^T * q* = e_1^T * p*.
    #
    # Equivalently, we solve two separate LPs (one per player):
    #
    # LP-1 (find player 1's strategy, dual gives game value):
    #   min  e_1^T * x_1
    #   s.t. A * x_1  +  E_0^T * p  >= 0     (one ineq per P0 sequence)
    #        E_1 * x_1               =  e_1
    #        x_1 >= 0,  p free
    #
    # But we want min, and linprog does min, so:
    # Variables: [x_1 (n1 vars, >= 0), p (m0 vars, free)]
    # Objective: min e_1^T * x_1
    # Constraints:
    #   A * x_1 + E_0^T * p >= 0   =>  -A * x_1 - E_0^T * p <= 0
    #   E_1 * x_1 = e_1

    # Objective: min e_1^T x_1  (only x_1 part matters)
    c1 = np.zeros(n1 + m0)
    c1[:n1] = e1 @ E1  # Actually we want e_1^T x_1.
    # Wait, e_1^T x_1 = sum of e_1[row] * (E_1 x_1)[row]?  No.
    # e_1 is a vector of length m1.  x_1 is length n1.
    # We want: min e_1^T x_1 ... but e_1 is not the same dimension as x_1.
    #
    # Let me re-derive from the standard formulation.
    #
    # The standard sequence-form LP (von Stengel, Koller et al.):
    #
    # VALUE TO PLAYER 0 = optimal value of:
    #
    #   max_{p}    e_0^T p
    #   s.t.       A^T p <= E_1^T q   for some q
    #              ... this isn't right either.
    #
    # Let me use the cleaner formulation:
    #
    # max_x min_y  x^T A y
    # s.t. E0 x = e0, x >= 0
    #      E1 y = e1, y >= 0
    #
    # By LP duality, this equals:
    #
    # LP-P1: min_{y, p}  e0^T p
    #        s.t. E0^T p >= A y        (n0 constraints)
    #             E1 y    = e1         (m1 constraints)
    #             y       >= 0
    #             p       free
    #
    # equivalently: min e0^T p
    #   s.t. -E0^T p + A y <= 0
    #        E1 y = e1
    #        y >= 0, p free

    # Reformulate for linprog:
    # Variables: [y (n1, >= 0), p (m0, free)]
    # Objective: min 0^T y + e0^T p = min e0^T p
    c_lp = np.zeros(n1 + m0)
    c_lp[n1:] = e0  # coefficients for p

    # Inequality: -E0^T p + A y <= 0
    # That is: A[:, :] y - E0.T[:, :] p <= 0
    # In matrix form: [A | -E0^T] [y; p]^T <= 0
    A_ub_lp = np.zeros((n0, n1 + m0))
    A_ub_lp[:, :n1] = A           # A y
    A_ub_lp[:, n1:] = -E0.T       # -E0^T p
    b_ub_lp = np.zeros(n0)

    # Equality: E1 y = e1
    A_eq_lp = np.zeros((m1, n1 + m0))
    A_eq_lp[:, :n1] = E1
    b_eq_lp = e1

    # Bounds: y >= 0, p free
    bounds_lp = [(0.0, None)] * n1 + [(None, None)] * m0

    result1 = linprog(
        c_lp,
        A_ub=A_ub_lp,
        b_ub=b_ub_lp,
        A_eq=A_eq_lp,
        b_eq=b_eq_lp,
        bounds=bounds_lp,
        method="highs",
    )
    if not result1.success:
        raise ValueError(f"LP for player 1 failed: {result1.message}")

    y_star = result1.x[:n1]
    game_value = result1.fun  # = e0^T p*

    # ---- LP for Player 0's realization plan ---------------------- #
    # LP-P0: max_{x, q}  e1^T q
    #        s.t. E1^T q <= A^T x      (n1 constraints)
    #             E0 x    = e0
    #             x       >= 0
    #             q       free
    #
    # min -e1^T q
    # s.t. -A^T x + E1^T q <= 0
    #      E0 x = e0
    #      x >= 0, q free

    c_lp0 = np.zeros(n0 + m1)
    c_lp0[n0:] = -e1  # min -e1^T q

    A_ub_lp0 = np.zeros((n1, n0 + m1))
    A_ub_lp0[:, :n0] = -A.T        # -A^T x
    A_ub_lp0[:, n0:] = E1.T        # E1^T q
    b_ub_lp0 = np.zeros(n1)

    A_eq_lp0 = np.zeros((m0, n0 + m1))
    A_eq_lp0[:, :n0] = E0
    b_eq_lp0 = e0

    bounds_lp0 = [(0.0, None)] * n0 + [(None, None)] * m1

    result0 = linprog(
        c_lp0,
        A_ub=A_ub_lp0,
        b_ub=b_ub_lp0,
        A_eq=A_eq_lp0,
        b_eq=b_eq_lp0,
        bounds=bounds_lp0,
        method="highs",
    )
    if not result0.success:
        raise ValueError(f"LP for player 0 failed: {result0.message}")

    x_star = result0.x[:n0]

    # ---- format output ------------------------------------------ #
    plan0 = {
        _format_sequence(seqs0[i]): float(x_star[i]) for i in range(n0)
    }
    plan1 = {
        _format_sequence(seqs1[j]): float(y_star[j]) for j in range(n1)
    }

    return {"P0": plan0, "P1": plan1}, float(game_value)
