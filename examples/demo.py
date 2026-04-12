"""
Demonstration of the extensive-form game solver library.

Run from project root:
    python examples/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable.
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.core.backward_induction import backward_induction
from src.core.sequence_form import sequence_form_solve
from src.core.normal_form import to_normal_form
from src.core.kuhn_theorem import (
    check_perfect_recall,
    behavioral_to_mixed,
    mixed_to_behavioral,
)
from src.games.centipede import build_centipede
from src.games.entry_deterrence import build_entry_deterrence
from src.games.kuhn_poker import build_kuhn_poker


def divider(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def demo_entry_deterrence() -> None:
    divider("Entry Deterrence -- Backward Induction")
    game = build_entry_deterrence()
    print(game)

    eq, payoffs = backward_induction(game)
    print(f"SPE strategy:  {eq}")
    print(f"Root payoffs:  {payoffs}")

    nf = to_normal_form(game)
    print(f"\nNormal form:   {nf}")
    print(f"P0 payoff matrix:\n{nf.payoff_matrices[0]}")


def demo_centipede() -> None:
    divider("Centipede (4 rounds) -- Backward Induction")
    game = build_centipede(4)
    print(game)

    eq, payoffs = backward_induction(game)
    print(f"SPE strategy:  {eq}")
    print(f"Root payoffs:  {payoffs}")
    print("(Backward induction predicts immediate Take at round 0)")


def demo_kuhn_poker() -> None:
    divider("Kuhn Poker -- Sequence-Form LP")
    game = build_kuhn_poker()
    print(game)

    ok, msg = check_perfect_recall(game)
    print(f"Perfect recall: {msg}")

    plans, gv = sequence_form_solve(game)
    print(f"Game value to P0: {gv:.6f}  (theoretical: {-1/18:.6f})")

    print("\nP0 realization plan (non-zero):")
    for seq, prob in sorted(plans["P0"].items()):
        if prob > 1e-9:
            print(f"  {seq:30s}  {prob:.4f}")

    print("\nP1 realization plan (non-zero):")
    for seq, prob in sorted(plans["P1"].items()):
        if prob > 1e-9:
            print(f"  {seq:30s}  {prob:.4f}")


def demo_kuhn_theorem() -> None:
    divider("Kuhn's Theorem -- Behavioral <-> Mixed Conversion")
    game = build_centipede(4)
    print(game)

    behavioral = {
        "P0_r0": {"Take": 0.4, "Pass": 0.6},
        "P0_r2": {"Take": 0.8, "Pass": 0.2},
    }
    print(f"Behavioral strategy for P0: {behavioral}")

    mixed = behavioral_to_mixed(game, 0, behavioral)
    print(f"Equivalent mixed strategy:  {mixed}")

    behavioral_back = mixed_to_behavioral(game, 0, mixed)
    print(f"Round-trip behavioral:      {behavioral_back}")


if __name__ == "__main__":
    demo_entry_deterrence()
    demo_centipede()
    demo_kuhn_poker()
    demo_kuhn_theorem()
    print("\nDone.")
