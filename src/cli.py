"""
Command-line interface for the extensive-form game solver.

Usage:
    python -m src.cli --game centipede --solver backward
    python -m src.cli --game kuhn_poker --solver sequence
    python -m src.cli --game entry_deterrence --solver normal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.core.backward_induction import backward_induction
from src.core.sequence_form import sequence_form_solve
from src.core.normal_form import to_normal_form
from src.core.kuhn_theorem import check_perfect_recall
from src.games.centipede import build_centipede
from src.games.entry_deterrence import build_entry_deterrence
from src.games.kuhn_poker import build_kuhn_poker


_GAMES = {
    "centipede": lambda: build_centipede(4),
    "centipede6": lambda: build_centipede(6),
    "entry_deterrence": build_entry_deterrence,
    "kuhn_poker": build_kuhn_poker,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extensive-form game solver CLI."
    )
    parser.add_argument(
        "--game",
        choices=list(_GAMES.keys()),
        default="entry_deterrence",
        help="Preset game to load.",
    )
    parser.add_argument(
        "--solver",
        choices=["backward", "sequence", "normal", "info"],
        default="info",
        help="Solving algorithm.",
    )
    args = parser.parse_args()

    game = _GAMES[args.game]()
    print(f"Game: {game}")
    print(f"Nodes: {len(game.nodes())}  Terminals: {len(game.terminal_nodes())}")

    pr_ok, pr_msg = check_perfect_recall(game)
    print(f"Perfect recall: {pr_msg}")

    if args.solver == "info":
        for p in game.players:
            isets = game.information_sets(p)
            print(f"Player {p} information sets: {list(isets.keys())}")
        return

    if args.solver == "backward":
        try:
            eq, payoffs = backward_induction(game)
            print("\n--- Backward Induction ---")
            for p in game.players:
                print(f"  Player {p}: {eq.get(p, {})}")
            print(f"  Root payoffs: {payoffs}")
        except ValueError as exc:
            print(f"Error: {exc}")

    elif args.solver == "sequence":
        try:
            plans, gv = sequence_form_solve(game)
            print("\n--- Sequence-Form LP ---")
            print(f"  Game value (P0): {gv:.6f}")
            for label, plan in plans.items():
                print(f"  {label} realization plan:")
                for seq, prob in plan.items():
                    if prob > 1e-9:
                        print(f"    {seq}: {prob:.4f}")
        except ValueError as exc:
            print(f"Error: {exc}")

    elif args.solver == "normal":
        nf = to_normal_form(game)
        print(f"\n--- Normal Form ---")
        print(f"  {nf}")
        for p in game.players:
            print(f"  Player {p} strategies:")
            for idx, strat in enumerate(nf.strategy_labels[p]):
                print(f"    {idx}: {strat}")


if __name__ == "__main__":
    main()
