"""
Streamlit application for interactive game tree exploration and solving.

Run with:
    streamlit run src/viz/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import streamlit as st
import graphviz

# Ensure the project root is importable.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.core.game_tree import ChanceNode, GameNode, GameTree, Node, TerminalNode
from src.core.backward_induction import backward_induction
from src.core.sequence_form import sequence_form_solve
from src.core.normal_form import to_normal_form
from src.core.kuhn_theorem import check_perfect_recall
from src.games.centipede import build_centipede
from src.games.entry_deterrence import build_entry_deterrence
from src.games.kuhn_poker import build_kuhn_poker


# ------------------------------------------------------------------ #
#  Colour palette per player                                          #
# ------------------------------------------------------------------ #

_PLAYER_COLOURS = {
    0: "#3498db",   # blue
    1: "#e74c3c",   # red
    2: "#2ecc71",   # green
    3: "#f39c12",   # orange
}
_CHANCE_COLOUR = "#9b59b6"   # purple
_TERMINAL_COLOUR = "#7f8c8d"  # grey


# ------------------------------------------------------------------ #
#  Graphviz rendering                                                 #
# ------------------------------------------------------------------ #

def _node_label(node: Node) -> str:
    if isinstance(node, TerminalNode):
        pays = ", ".join(f"P{k}:{v:g}" for k, v in sorted(node.payoffs.items()))
        label = node.name or "T"
        return f"{label}\\n[{pays}]"
    if isinstance(node, ChanceNode):
        return node.name or "Chance"
    assert isinstance(node, GameNode)
    label = node.name or node.infoset_id or f"P{node.player}"
    return label


def render_game_tree(
    game: GameTree,
    equilibrium: Optional[Dict[int, Dict[str, str]]] = None,
    realization_plans: Optional[Dict[str, Dict[str, float]]] = None,
) -> graphviz.Digraph:
    """Build a Graphviz Digraph of the game tree.

    Parameters
    ----------
    game : GameTree
    equilibrium : dict, optional
        ``equilibrium[player][infoset_id] = action`` for backward induction.
    realization_plans : dict, optional
        Sequence-form realization plans (edge probability overlay).

    Returns
    -------
    graphviz.Digraph
    """
    dot = graphviz.Digraph(
        name=game.title,
        format="svg",
        graph_attr={"rankdir": "TB", "bgcolor": "transparent", "dpi": "72"},
        node_attr={"fontname": "Helvetica", "fontsize": "10"},
        edge_attr={"fontname": "Helvetica", "fontsize": "9"},
    )

    # Track information sets so we can draw dashed grouping lines.
    infoset_members: Dict[str, list] = {}

    def add_node(node: Node) -> str:
        nid = str(node._id)
        if isinstance(node, TerminalNode):
            dot.node(
                nid,
                label=_node_label(node),
                shape="box",
                style="filled",
                fillcolor=_TERMINAL_COLOUR,
                fontcolor="white",
            )
        elif isinstance(node, ChanceNode):
            dot.node(
                nid,
                label=_node_label(node),
                shape="diamond",
                style="filled",
                fillcolor=_CHANCE_COLOUR,
                fontcolor="white",
            )
        else:
            assert isinstance(node, GameNode)
            colour = _PLAYER_COLOURS.get(node.player, "#555555")
            dot.node(
                nid,
                label=_node_label(node),
                shape="ellipse",
                style="filled",
                fillcolor=colour,
                fontcolor="white",
            )
            if node.infoset_id:
                infoset_members.setdefault(node.infoset_id, []).append(nid)
        return nid

    def walk(node: Node) -> None:
        parent_id = add_node(node)
        if node.is_terminal:
            return
        for action, child in node.children.items():
            child_id = add_node(child)
            # Edge label.
            elabel = action
            # Highlight equilibrium edges.
            attrs: Dict[str, str] = {}
            if isinstance(node, GameNode) and equilibrium:
                eq_action = equilibrium.get(node.player, {}).get(node.infoset_id)
                if eq_action == action:
                    attrs["penwidth"] = "2.5"
                    attrs["color"] = _PLAYER_COLOURS.get(node.player, "black")
                else:
                    attrs["style"] = "dashed"
                    attrs["color"] = "gray"
            if isinstance(node, ChanceNode):
                prob = node.distribution.get(action, 0)
                elabel = f"{action} ({prob:.2f})"

            dot.edge(parent_id, child_id, label=elabel, **attrs)
            if not child.is_terminal:
                walk(child)

    walk(game.root)

    # Draw dashed edges between nodes in the same information set.
    for iset_id, members in infoset_members.items():
        if len(members) > 1:
            for i in range(len(members) - 1):
                dot.edge(
                    members[i],
                    members[i + 1],
                    style="dashed",
                    color="orange",
                    constraint="false",
                    label=f"I:{iset_id}",
                    fontcolor="orange",
                )

    return dot


# ------------------------------------------------------------------ #
#  Streamlit UI                                                       #
# ------------------------------------------------------------------ #

def main() -> None:
    st.set_page_config(page_title="Extensive-Form Game Solver", layout="wide")
    st.title("Extensive-Form Game Solver")
    st.markdown(
        "Explore and solve extensive-form games with backward induction, "
        "sequence-form LP, and normal-form conversion."
    )

    # ---- sidebar: game selection -------------------------------- #
    st.sidebar.header("Game Selection")
    game_choice = st.sidebar.selectbox(
        "Choose a preset game",
        ["Entry Deterrence", "Centipede (4 rounds)", "Centipede (6 rounds)", "Kuhn Poker"],
    )

    if game_choice == "Entry Deterrence":
        game = build_entry_deterrence()
    elif game_choice == "Centipede (4 rounds)":
        game = build_centipede(4)
    elif game_choice == "Centipede (6 rounds)":
        game = build_centipede(6)
    else:
        game = build_kuhn_poker()

    st.sidebar.markdown("---")
    st.sidebar.header("Game Info")
    st.sidebar.write(f"**Title:** {game.title}")
    st.sidebar.write(f"**Players:** {game.players}")
    st.sidebar.write(f"**Nodes:** {len(game.nodes())}")
    st.sidebar.write(f"**Terminal nodes:** {len(game.terminal_nodes())}")

    # Perfect recall check.
    pr_ok, pr_msg = check_perfect_recall(game)
    st.sidebar.write(f"**Perfect recall:** {'Yes' if pr_ok else 'No'}")

    # ---- solver selection --------------------------------------- #
    st.sidebar.markdown("---")
    st.sidebar.header("Solver")
    solver = st.sidebar.selectbox(
        "Algorithm",
        ["None (just show tree)", "Backward Induction", "Sequence-Form LP", "Normal-Form Conversion"],
    )

    equilibrium = None
    realization_plans = None

    if solver == "Backward Induction":
        is_perfect_info = all(
            len(nodes) == 1
            for p in game.players
            for nodes in game.information_sets(p).values()
        )
        has_chance_root = isinstance(game.root, ChanceNode)

        if not is_perfect_info or has_chance_root:
            st.warning(
                "Backward induction requires perfect information and no "
                "chance root.  Try a different solver or game."
            )
        else:
            eq, payoffs = backward_induction(game)
            equilibrium = eq
            st.subheader("Backward Induction Result")
            for p in game.players:
                st.write(f"**Player {p} strategy:** {eq.get(p, {})}")
            st.write(f"**Root payoffs:** {payoffs}")

    elif solver == "Sequence-Form LP":
        if sorted(game.players) != [0, 1]:
            st.warning("Sequence-form LP requires exactly 2 players.")
        else:
            try:
                plans, gv = sequence_form_solve(game)
                realization_plans = plans
                st.subheader("Sequence-Form LP Result")
                st.write(f"**Game value (to Player 0):** {gv:.6f}")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Player 0 realization plan:**")
                    for seq, prob in plans["P0"].items():
                        if prob > 1e-9:
                            st.write(f"  {seq}: {prob:.4f}")
                with col2:
                    st.write("**Player 1 realization plan:**")
                    for seq, prob in plans["P1"].items():
                        if prob > 1e-9:
                            st.write(f"  {seq}: {prob:.4f}")
            except ValueError as exc:
                st.error(str(exc))

    elif solver == "Normal-Form Conversion":
        nf = to_normal_form(game)
        st.subheader("Normal-Form Representation")
        st.write(f"**Strategy counts:** {nf}")
        for p in game.players:
            with st.expander(f"Player {p} pure strategies"):
                for idx, strat in enumerate(nf.strategy_labels[p]):
                    st.write(f"  Strategy {idx}: {strat}")
        if len(game.players) == 2:
            st.write("**Payoff matrix (Player 0):**")
            st.dataframe(
                data=nf.payoff_matrices[0],
                use_container_width=True,
            )

    # ---- render the tree ---------------------------------------- #
    st.subheader("Game Tree")
    dot = render_game_tree(game, equilibrium=equilibrium, realization_plans=realization_plans)
    st.graphviz_chart(dot.source)


if __name__ == "__main__":
    main()
