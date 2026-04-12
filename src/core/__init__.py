"""Core game tree data structures and solvers."""

from src.core.game_tree import GameNode, ChanceNode, TerminalNode, GameTree
from src.core.backward_induction import backward_induction
from src.core.sequence_form import sequence_form_solve
from src.core.normal_form import to_normal_form, NormalFormGame
from src.core.kuhn_theorem import (
    check_perfect_recall,
    behavioral_to_mixed,
    mixed_to_behavioral,
)

__all__ = [
    "GameNode",
    "ChanceNode",
    "TerminalNode",
    "GameTree",
    "backward_induction",
    "sequence_form_solve",
    "to_normal_form",
    "NormalFormGame",
    "check_perfect_recall",
    "behavioral_to_mixed",
    "mixed_to_behavioral",
]
