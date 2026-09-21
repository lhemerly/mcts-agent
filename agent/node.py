"""
node.py — MCTS Node data class.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Optional

_node_id_counter = itertools.count()


@dataclass
class Node:
    """A single node in the MCTS search tree."""

    state: str
    parent: Optional["Node"] = field(default=None, repr=False)
    action_taken: Optional[str] = None
    children: list["Node"] = field(default_factory=list, repr=False)
    depth: int = 0  # depth from the current root (updated on reroot)

    # ── MCTS statistics ──────────────────────────────────────────────────────
    visits: int = 0
    value_sum: float = 0.0
    prior_probability: float = 0.0

    # ── Unique ID (auto-assigned) ────────────────────────────────────────────
    node_id: str = field(
        default_factory=lambda: f"n{next(_node_id_counter)}", init=False
    )

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def average_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def puct_score(self, exploration_constant: float = 1.4) -> float:
        """
        PUCT score for child selection.

        Q is normalized to [0, 1] by dividing by the max possible Score value
        (10.0) so the exploration term isn't crushed by the value scale.

            PUCT = Q/10 + c · P · √N / (1 + n)
        """
        if self.parent is None:
            return float("inf")

        parent_visits = self.parent.visits

        # First-Play Urgency (FPU): If unvisited, inherit parent's value estimate.
        # This prevents unvisited siblings from being permanently starved when one child gets a high score.
        if self.visits == 0:
            q_normalized = self.parent.average_value / 10.0
        else:
            q_normalized = self.average_value / 10.0

        exploration = (
            exploration_constant
            * self.prior_probability
            * math.sqrt(parent_visits)
            / (1 + self.visits)
        )
        return q_normalized + exploration

    def subtree_leaves(self) -> list["Node"]:
        """Return all leaf nodes in this subtree (DFS)."""
        if not self.children:
            return [self]
        leaves: list["Node"] = []
        for child in self.children:
            leaves.extend(child.subtree_leaves())
        return leaves

    def collect_actions_by_depth(self) -> dict[int, list[str]]:
        """
        BFS to collect all action_taken values grouped by their depth field.
        Skips the root node itself (which has no action_taken).

        Returns
        -------
        dict[depth, list[action_text]]
        """
        result: dict[int, list[str]] = {}
        stack = [self]
        while stack:
            node = stack.pop()
            if node.action_taken:
                result.setdefault(node.depth, []).append(node.action_taken)
            stack.extend(node.children)
        return result

    def all_actions_flat(self) -> list[str]:
        """Return every action_taken in this subtree as a flat list (for exclusion sets)."""
        actions: list[str] = []
        stack = [self]
        while stack:
            node = stack.pop()
            if node.action_taken:
                actions.append(node.action_taken)
            stack.extend(node.children)
        return actions

    def __repr__(self) -> str:
        snippet = (self.state[:60] + "…") if len(self.state) > 63 else self.state
        return (
            f"Node({self.node_id}, depth={self.depth}, visits={self.visits}, "
            f"avg={self.average_value:.3f}, prior={self.prior_probability:.3f}, "
            f'action="{self.action_taken}", state="{snippet}")'
        )
