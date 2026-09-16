"""
logger.py — Structured JSON event logger for the MCTS run.

Every key moment in the search is emitted as a typed event dict and
flushed to a .json file that the visualizer can replay frame-by-frame.

Event types (in order of emission per iteration):
  init            – root node created
  iteration_start – new iteration begins
  select          – PUCT traversal path to leaf
  candidates      – Gemini-proposed actions
  noul            – Noul gate result per action (batched call, one event each)
  new_node        – child node added to tree
  eval            – which node is being scored
  score           – Score primitive result
  backprop_start  – path that will be updated
  node_update     – per-node stat update during backprop
  iteration_end   – iteration complete
  complete        – final best action
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MCTSLogger:
    def __init__(
        self,
        goal: str,
        initial_state: str,
        iterations: int,
        log_dir: str = "logs",
        step: int | None = None,
        run_id: str | None = None,
    ) -> None:
        self._events: list[dict[str, Any]] = []
        self._meta: dict[str, Any] = {
            "goal": goal,
            "initial_state": initial_state,
            "iterations": iterations,
            "step": step,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        ts = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_step{step}" if step is not None else ""
        self.log_path = log_path / f"mcts_{ts}{suffix}.json"

    # ── Emitters ─────────────────────────────────────────────────────────────

    def emit_init(self, root_node: Any) -> None:
        self._emit("init", node_id=root_node.node_id, state=root_node.state)

    def emit_iteration_start(self, i: int) -> None:
        self._emit("iteration_start", i=i)

    def emit_select(self, path: list[Any]) -> None:
        self._emit("select", path=[n.node_id for n in path])

    def emit_candidates(self, candidates: list[str]) -> None:
        self._emit("candidates", candidates=candidates)

    def emit_noul(self, action: str, valid: bool, confidence: float) -> None:
        self._emit("noul", action=action, valid=valid, confidence=round(confidence, 4))

    def emit_new_node(self, node: Any) -> None:
        self._emit(
            "new_node",
            node_id=node.node_id,
            parent_id=node.parent.node_id if node.parent else None,
            action=node.action_taken,
            prior=round(node.prior_probability, 4),
            state=node.state,
        )

    def emit_eval(self, node: Any) -> None:
        self._emit("eval", node_id=node.node_id)

    def emit_score(self, node: Any, value: float) -> None:
        self._emit("score", node_id=node.node_id, value=round(value, 4))

    def emit_backprop_start(self, path: list[Any], value: float) -> None:
        self._emit(
            "backprop_start",
            path=[n.node_id for n in path],
            value=round(value, 4),
        )

    def emit_node_update(self, node: Any) -> None:
        self._emit(
            "node_update",
            node_id=node.node_id,
            visits=node.visits,
            value_sum=round(node.value_sum, 4),
        )

    def emit_iteration_end(self, i: int) -> None:
        self._emit("iteration_end", i=i)

    def emit_task_completed(self, confidence: float, iteration: int) -> None:
        self._emit("task_completed", confidence=round(confidence, 4), iteration=iteration)

    def emit_complete(self, best_node: Any) -> None:
        self._emit(
            "complete",
            best_action=best_node.action_taken,
            best_node_id=best_node.node_id,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> Path:
        payload = {"meta": self._meta, "events": self._events}
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return self.log_path

    def _emit(self, event_type: str, **kwargs: Any) -> None:
        self._events.append({"type": event_type, **kwargs})


def save_agent_summary(
    run_id: str,
    summary: dict[str, Any],
    log_dir: str = "logs",
) -> Path:
    """Save an overall summary of the closed-loop agent run across all steps."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    summary_file = log_path / f"run_{run_id}_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary_file
