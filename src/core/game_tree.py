"""
Game tree data structures for extensive-form games.

Classes
-------
GameNode     -- A decision node owned by a player.
ChanceNode   -- A chance (nature) node with a probability distribution.
TerminalNode -- A leaf node carrying a payoff vector.
GameTree     -- Container that holds the root and provides queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


# ------------------------------------------------------------------ #
#  Nodes                                                              #
# ------------------------------------------------------------------ #

@dataclass
class TerminalNode:
    """Leaf node with payoffs for each player.

    Parameters
    ----------
    payoffs : dict[int, float]
        Mapping from player id to payoff at this terminal history.
    name : str, optional
        Human-readable label for display.
    """

    payoffs: Dict[int, float]
    name: str = ""
    _id: int = field(default=-1, repr=False)

    # Convenience helpers expected by the rest of the code base.
    @property
    def is_terminal(self) -> bool:
        return True

    @property
    def is_chance(self) -> bool:
        return False


@dataclass
class ChanceNode:
    """Nature node that samples an action from a fixed distribution.

    Parameters
    ----------
    distribution : dict[str, float]
        Mapping from action label to probability.  Must sum to 1.
    children : dict[str, Node]
        Mapping from action label to successor node.
    name : str, optional
        Human-readable label.
    """

    distribution: Dict[str, float]
    children: Dict[str, "Node"] = field(default_factory=dict)
    name: str = ""
    _id: int = field(default=-1, repr=False)

    def __post_init__(self) -> None:
        total = sum(self.distribution.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Chance-node probabilities must sum to 1 (got {total})."
            )

    @property
    def actions(self) -> List[str]:
        return list(self.distribution.keys())

    @property
    def is_terminal(self) -> bool:
        return False

    @property
    def is_chance(self) -> bool:
        return True


@dataclass
class GameNode:
    """Decision node belonging to a specific player.

    Parameters
    ----------
    player : int
        The player who moves at this node (0-indexed).
    actions : list[str]
        Available actions.
    children : dict[str, Node]
        Mapping from action label to successor node.
    infoset_id : str | None
        Information-set identifier.  Nodes with the same *infoset_id*
        for the same *player* are indistinguishable to that player.
    name : str, optional
        Human-readable label.
    """

    player: int
    actions: List[str]
    children: Dict[str, "Node"] = field(default_factory=dict)
    infoset_id: Optional[str] = None
    name: str = ""
    _id: int = field(default=-1, repr=False)

    def __post_init__(self) -> None:
        if self.infoset_id is None:
            self.infoset_id = self.name or f"P{self.player}_{id(self)}"

    @property
    def is_terminal(self) -> bool:
        return False

    @property
    def is_chance(self) -> bool:
        return False


# Union type for any node.
Node = Union[GameNode, ChanceNode, TerminalNode]


# ------------------------------------------------------------------ #
#  GameTree                                                           #
# ------------------------------------------------------------------ #

class GameTree:
    """Container for an extensive-form game.

    Parameters
    ----------
    root : Node
        Root of the game tree.
    players : list[int]
        Player identifiers (e.g. [0, 1]).
    title : str
        A human-readable name for the game.
    """

    def __init__(
        self,
        root: Node,
        players: List[int],
        title: str = "Untitled Game",
    ) -> None:
        self.root = root
        self.players = players
        self.title = title
        self._assign_ids()

    # -- traversal helpers ----------------------------------------- #

    def _assign_ids(self) -> None:
        """Give every node a unique integer id (BFS order)."""
        counter = 0
        queue: List[Node] = [self.root]
        while queue:
            node = queue.pop(0)
            node._id = counter
            counter += 1
            if not node.is_terminal:
                for child in node.children.values():
                    queue.append(child)

    def nodes(self) -> List[Node]:
        """Return all nodes in BFS order."""
        result: List[Node] = []
        queue: List[Node] = [self.root]
        while queue:
            node = queue.pop(0)
            result.append(node)
            if not node.is_terminal:
                for child in node.children.values():
                    queue.append(child)
        return result

    def decision_nodes(self, player: Optional[int] = None) -> List[GameNode]:
        """Return decision nodes, optionally filtered by player."""
        return [
            n
            for n in self.nodes()
            if isinstance(n, GameNode) and (player is None or n.player == player)
        ]

    def terminal_nodes(self) -> List[TerminalNode]:
        return [n for n in self.nodes() if isinstance(n, TerminalNode)]

    def chance_nodes(self) -> List[ChanceNode]:
        return [n for n in self.nodes() if isinstance(n, ChanceNode)]

    def information_sets(self, player: int) -> Dict[str, List[GameNode]]:
        """Return {infoset_id: [nodes]} for *player*."""
        info: Dict[str, List[GameNode]] = {}
        for n in self.decision_nodes(player):
            info.setdefault(n.infoset_id, []).append(n)
        return info

    # -- path helpers ---------------------------------------------- #

    def _paths_to_node(
        self,
        target: Node,
    ) -> List[List[Tuple[Node, Optional[str]]]]:
        """Return all root-to-*target* paths as [(node, action_taken), ...]."""
        results: List[List[Tuple[Node, Optional[str]]]] = []

        def dfs(
            current: Node,
            path: List[Tuple[Node, Optional[str]]],
        ) -> None:
            if current is target:
                results.append(list(path))
                return
            if current.is_terminal:
                return
            for action, child in current.children.items():
                path.append((current, action))
                dfs(child, path)
                path.pop()

        dfs(self.root, [(self.root, None)])
        # The very first tuple always has action=None (the root itself).
        # We actually want (root, None) only at position 0:
        cleaned: List[List[Tuple[Node, Optional[str]]]] = []
        for p in results:
            # p already starts with (root, None) from the initial call
            cleaned.append(p)
        return cleaned

    def __repr__(self) -> str:
        n_nodes = len(self.nodes())
        n_term = len(self.terminal_nodes())
        return (
            f"GameTree(title={self.title!r}, players={self.players}, "
            f"nodes={n_nodes}, terminals={n_term})"
        )
