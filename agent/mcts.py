"""
mcts.py — Core MCTS loop with structured event logging.

Architecture
------------
The search operates in two distinct phases per closed-loop step:

Phase A — Deep Materialized Expansion
  Starting from the root, `_deep_expand` recursively builds a tree of real nodes
  down to `expansion_depth` levels.  With width W and depth D the tree contains
  sum(W^k for k in 1..D) nodes and W^D leaf paths, all scored at construction time.
  PUCT-guided selection then walks existing paths rather than re-expanding.

Phase B — Tree Reuse + Action Vocabulary Recombination (between closed-loop steps)
  After executing the best first action and observing the ground-truth result:
    1. Re-root the tree at the chosen child node; update its state with the observation.
    2. Harvest the action vocabulary (all action_taken values by depth) from the
       *entire* original tree before discarding sibling branches.
    3. Keep the chosen node's original subtree (W^(D-1) leaf paths) as survivors.
    4. Ground surviving states in the execution observation and rescore them.
    5. If a survivor meets the reuse threshold, choose from that tree directly.
       Otherwise assemble unique cross-branch paths from the vocabulary and score
       them. Rebuild only if no actionable path reaches the threshold.

Every event is recorded by an MCTSLogger and saved as a JSON log
that the MCTS visualizer can replay.
"""

from __future__ import annotations

import os
import random
import subprocess
import textwrap
from datetime import datetime
from typing import Any, Optional

from agent.config import AgentConfig, load_config
from agent.logger import MCTSLogger, save_agent_summary
from agent.node import Node
from agent.primitives import (
    NOUL_COMPLETION_THRESHOLD,
    check_action_execution,
    check_task_completion,
    discriminative_choose_best_action,
    evaluate_state,
    get_action_priors,
    select_action_count,
)
from agent.providers import get_executor_provider, get_planner_provider


def _is_mock_llm() -> bool:
    return os.getenv("USE_MOCK_PRIMITIVES", "false").lower() in ("1", "true", "yes")


_AGY_MODEL = os.getenv("AGY_MODEL", "gemini-3.8-flash-medium")
_DEFAULT_PROPOSAL_MODEL = "gemini-3.6-flash-low"
_AGY_PROPOSAL_MODEL = os.getenv("AGY_PROPOSAL_MODEL", _DEFAULT_PROPOSAL_MODEL)

_MOCK_ACTION_POOL: list[str] = [
    "Search for relevant documentation online",
    "Decompose the problem into smaller sub-tasks",
    "Verify assumptions by checking existing data",
    "Draft an initial outline and review it",
    "Identify potential blockers and mitigate them",
    "Run a quick prototype to test the hypothesis",
    "Consult domain-specific knowledge base",
    "Summarise findings and propose next step",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _path_to_root(node: Node) -> list[Node]:
    """Walk parent pointers and return [root, …, node]."""
    path: list[Node] = []
    current: Optional[Node] = node
    while current is not None:
        path.append(current)
        current = current.parent
    return list(reversed(path))


# ── Action proposal via pluggable planner provider ──────────────────────────────

def _propose_actions(
    state: str,
    goal: str,
    n: int = 3,
    *,
    explored_actions: Optional[list[str]] = None,
    config: AgentConfig | None = None,
) -> list[str]:
    """
    Propose `n` distinct next actions using the configured PlannerProvider.

    Parameters
    ----------
    explored_actions : list[str] | None
        All action_taken values already present anywhere in the current search tree.
        Passed through to the provider so the LLM avoids generating duplicates.
    """
    cfg = config or load_config()
    planner = get_planner_provider(cfg)
    return planner.propose_actions(state, goal, n=n, explored_actions=explored_actions)


# ── Phase A: Deep Materialized Expansion ──────────────────────────────────────

def _deep_expand(
    node: Node,
    goal: str,
    width: int,
    remaining_depth: int,
    logger: MCTSLogger,
    *,
    explored_actions: Optional[list[str]] = None,
    config: AgentConfig | None = None,
) -> None:
    """
    Recursively expand `node` down to `remaining_depth` more levels.

    At each level:
      1. Propose `width` candidate actions (with full explored_actions context).
      2. Assign Choice priors to every proposed action.
      3. Create child nodes and recurse.
      4. At the leaf level (remaining_depth == 0), score the node and store the value.

    This replaces both the old flat `_expand` and the phantom `_simulate` rollout.
    All nodes in the tree are real, scored, and backprop-eligible.
    """
    if explored_actions is None:
        explored_actions = []

    if remaining_depth == 0:
        # Leaf: score and record
        val = evaluate_state(goal, node.state, system_one_provider=config.system_one_provider if config else None)
        node.visits = 1
        node.value_sum = val
        logger.emit_score(node, val)
        return

    # Propose candidates, avoiding anything already in the tree
    candidates = _propose_actions(
        node.state, goal, n=width,
        explored_actions=explored_actions,
        config=config,
    )
    logger.emit_candidates(candidates)

    if not candidates:
        print("  [expand] Planner returned no actions; scoring the current path.")
        val = evaluate_state(goal, node.state, system_one_provider=config.system_one_provider if config else None)
        node.visits = 1
        node.value_sum = val
        logger.emit_score(node, val)
        return

    priors = get_action_priors(
        node.state, candidates, system_one_provider=config.system_one_provider if config else None
    )

    # Share the action vocabulary across sibling branches in this expansion.
    # Descendants should not ask the planner for actions already proposed elsewhere.
    explored_actions.extend(candidates)

    for action in candidates:
        child = Node(
            state=f"{node.state}\n[Action taken]: {action}",
            parent=node,
            action_taken=action,
            depth=node.depth + 1,
            prior_probability=priors.get(action, 1.0 / len(candidates)),
        )
        node.children.append(child)
        logger.emit_new_node(child)

        _deep_expand(
            child,
            goal,
            width,
            remaining_depth - 1,
            logger,
            explored_actions=explored_actions,
            config=config,
        )


# ── Phase A: PUCT Selection across materialised tree ──────────────────────────

def _select(root: Node, c: float, logger: MCTSLogger) -> Node:
    """Walk the tree using PUCT until a leaf (unexpanded or scored-only node) is reached."""
    node = root
    while not node.is_leaf:
        node = max(node.children, key=lambda ch: ch.puct_score(c))
    logger.emit_select(_path_to_root(node))
    return node


# ── Phase A: Backpropagation ────────────────────────────────────────────────────

def _backpropagate(node: Node, value: float, logger: MCTSLogger) -> None:
    path = _path_to_root(node)
    logger.emit_backprop_start(path, value)
    current: Optional[Node] = node
    while current is not None:
        current.visits += 1
        current.value_sum += value
        logger.emit_node_update(current)
        current = current.parent


# ── Phase B: Tree Reuse Helpers ───────────────────────────────────────────────

def _reroot(chosen: Node, observation: str, grounded_state: str | None = None) -> Node:
    """
    Update the chosen node's state with the ground-truth observation from execution,
    then detach it from its parent (making it the new root).

    The node's depth is reset to 0; all descendant depths are shifted accordingly.
    """
    chosen.state = grounded_state if grounded_state is not None else (
        f"{chosen.state}\n[Ground-Truth Observation]: {observation}"
    )
    chosen.parent = None
    # Shift depths: chosen is now depth=0, children become depth=1, etc.
    _shift_depths(chosen, target_depth=0)
    _rebase_descendant_states(chosen)
    return chosen


def _rebase_descendant_states(parent: Node) -> None:
    """Rebuild hypothetical states from the latest observed root state."""
    for child in parent.children:
        child.state = f"{parent.state}\n[Action taken]: {child.action_taken}"
        _rebase_descendant_states(child)


def _shift_depths(node: Node, target_depth: int) -> None:
    """Recursively set node.depth = target_depth and adjust all descendants."""
    node.depth = target_depth
    for child in node.children:
        _shift_depths(child, target_depth + 1)


def _prune_low_value_children(node: Node, threshold: float) -> None:
    """
    Recursively remove children whose average_value falls below `threshold`.
    Unvisited children (average_value == 0.0) are retained to avoid
    discarding newly recombined nodes that haven't been scored yet.
    """
    node.children = [
        c for c in node.children
        if c.visits == 0 or c.average_value >= threshold
    ]
    for child in node.children:
        _prune_low_value_children(child, threshold)


def _recombine_paths(
    new_root: Node,
    vocab_by_depth: dict[int, list[str]],
    n_paths: int,
    path_depth: int,
) -> int:
    """
    Randomly assemble new paths from the action vocabulary and attach them to new_root.

    The vocabulary is depth-shifted: actions that were at original depth k become
    candidates for new depth k-1 (since we consumed one level by executing the action).
    This means sibling/cousin branch actions can now appear as children of the new root,
    enabling cross-branch "thinking out of the box" without new LLM proposal calls.

    Returns the number of new leaf nodes added.
    """
    # Remap: old_depth → new_depth = old_depth - 1 (skip depth-0 which was the old root)
    shifted: dict[int, list[str]] = {}
    for old_depth, actions in vocab_by_depth.items():
        new_d = old_depth - 1
        if new_d >= 1 and actions:
            shifted.setdefault(new_d, []).extend(actions)

    if not shifted:
        return 0

    leaves_added = 0
    existing_paths: set[tuple[str, ...]] = set()
    stack: list[tuple[Node, tuple[str, ...]]] = [(new_root, ())]
    while stack:
        node, path = stack.pop()
        if path:
            existing_paths.add(path)
        stack.extend((child, path + (child.action_taken or "",)) for child in node.children)
    attempts = 0
    while leaves_added < n_paths and attempts < n_paths * 5:
        attempts += 1
        actions: list[str] = []
        for d in range(1, path_depth + 1):
            pool = shifted.get(d, [])
            if not pool:
                break
            actions.append(random.choice(pool))
        if not actions or tuple(actions) in existing_paths:
            continue
        parent = new_root
        for d, action in enumerate(actions, start=1):
            existing = next((child for child in parent.children if child.action_taken == action), None)
            if existing is not None:
                parent = existing
                continue
            state = f"{parent.state}\n[Action taken]: {action}"
            child = Node(
                state=state,
                parent=parent,
                action_taken=action,
                depth=d,
            )
            parent.children.append(child)
            parent = child
        existing_paths.add(tuple(actions))
        leaves_added += 1

    return leaves_added


def _rescore_leaves(
    root: Node, goal: str, logger: MCTSLogger, config: AgentConfig | None = None
) -> dict[str, float]:
    """
    Re-evaluate all unvisited leaf nodes (newly recombined) and all existing leaves
    under the new root.  Visited leaves keep their existing scores unless they are
    direct survivors from a pre-observation state — those are also rescored so their
    values reflect the updated context.

    Returns a mapping {node_id: score} for all rescored leaves.
    """
    scores: dict[str, float] = {}
    for leaf in root.subtree_leaves():
        val = evaluate_state(goal, leaf.state, system_one_provider=config.system_one_provider if config else None)
        leaf.value_sum = val
        leaf.visits = 1
        logger.emit_score(leaf, val)
        scores[leaf.node_id] = val
        print(f"  [rescore] '{leaf.action_taken}' @ depth {leaf.depth} → {val:.3f}/10")
    _refresh_tree_statistics(root)
    return scores


def _refresh_tree_statistics(node: Node) -> None:
    """Discard old search counts after grounding and aggregate current leaf scores."""
    if not node.children:
        return
    for child in node.children:
        _refresh_tree_statistics(child)
    node.visits = sum(child.visits for child in node.children)
    node.value_sum = sum(child.value_sum for child in node.children)


# ── Public interface ───────────────────────────────────────────────────────────

def run_mcts(
    root: Node,
    goal: str,
    *,
    iterations: int | None = None,
    actions_per_node: int | None = None,
    simulation_depth: int | None = None,
    expansion_depth: int | None = None,
    early_stop_noul: bool = True,
    step: int | None = None,
    run_id: str | None = None,
    execute: bool = False,
    exploration_constant: float = 1.4,
    log_dir: str = "logs",
    config: AgentConfig | None = None,
    reuse_tree: bool = False,
) -> tuple[Node, str]:
    """
    Run MCTS from `root` to evaluate candidates and select the single best immediate action.

    Strategy
    --------
    1. Deep Expansion  — `_deep_expand` builds a fully materialised tree of real nodes
       down to `expansion_depth` levels (default from config).  All leaves are scored
       at construction time via the Score primitive.
    2. PUCT Iterations — additional PUCT-guided iterations refine fresh trees.
       Reused trees are already scored, so they proceed directly to choice.
    3. Discriminative Choice — the best immediate child of root is selected via
       TypeSafe Choice synthesising MCTS statistics with goal alignment.

    Parameters
    ----------
    actions_per_node : int | None
        Override the branching width for this call (default: config.expansion_width).
    simulation_depth : int | None
        Legacy alias for expansion_depth.  If both are given, expansion_depth wins.
    expansion_depth : int | None
        Override tree depth for this call (default: config.expansion_depth).
    iterations : int | None
        Number of additional PUCT iterations *after* the initial deep expand.
        Reused trees always skip these iterations because they add no new evidence.
    reuse_tree : bool
        Use an already rescored tree without proposing additional actions.
    """
    cfg = config or load_config()

    # Resolve width and depth with override priority
    width = actions_per_node if actions_per_node is not None else cfg.expansion_width
    depth = (
        expansion_depth if expansion_depth is not None
        else simulation_depth if simulation_depth is not None
        else cfg.expansion_depth
    )
    # Default PUCT iterations: one per leaf in the initial tree so every leaf gets
    # a backprop pass.  Callers can override with iterations=0 to skip.
    effective_iterations = 0 if reuse_tree else (
        iterations if iterations is not None else (width ** depth)
    )

    logger = MCTSLogger(
        goal=goal,
        initial_state=root.state,
        iterations=effective_iterations,
        log_dir=log_dir,
        step=step,
        run_id=run_id,
    )
    logger.emit_init(root)

    step_info = f"Step {step} | " if step is not None else ""
    print(f"\n{'='*60}")
    print(
        f"Starting MCTS  |  {step_info}"
        f"width={width}, depth={depth}, puct_iters={effective_iterations}  |  "
        f"goal='{goal[:80]}'"
    )
    print(f"{'='*60}\n")

    # ── Phase A: Deep materialised expansion ───────────────────────────────────
    if reuse_tree:
        print(f"[mcts] Reusing {len(root.subtree_leaves())} rescored leaf paths.\n")
    else:
        print(f"[mcts] Phase A: deep expanding tree (width={width}, depth={depth})...")
        explored: list[str] = root.all_actions_flat()
        _deep_expand(root, goal, width, depth, logger, explored_actions=explored, config=cfg)
        print(f"[mcts] Phase A complete: {len(root.subtree_leaves())} leaf paths materialised.\n")

    if not root.children:
        print("[mcts] Planner produced no immediate action.")
        logger.emit_complete(root)
        return root, str(logger.save())

    # ── Phase B: PUCT refinement iterations ───────────────────────────────────
    completed_early = False
    for i in range(1, effective_iterations + 1):
        iter_label = f"{i}/{effective_iterations}"
        print(f"── PUCT Iteration {iter_label} ──────────────────────────────────")
        logger.emit_iteration_start(i)

        leaf = _select(root, exploration_constant, logger)
        print(f"  [select] Leaf: visits={leaf.visits}, action='{leaf.action_taken}'")

        # Safeguard for an unexpectedly unscored leaf.
        if not reuse_tree and leaf.is_leaf and leaf.visits == 0:
            explored = root.all_actions_flat()
            _deep_expand(
                leaf, goal, width, depth - leaf.depth, logger,
                explored_actions=explored, config=cfg,
            )

        # Backpropagate the leaf's current value
        _backpropagate(leaf, leaf.average_value, logger)
        print(f"  [backprop] Value {leaf.average_value:.3f} propagated up the tree.\n")

        # Early stopping via Noul
        if early_stop_noul and i >= 2 and root.children:
            best_so_far = max(root.children, key=lambda c: (c.visits, c.average_value))
            is_done, conf = check_task_completion(
                goal, best_so_far.state, system_one_provider=cfg.system_one_provider
            )
            if is_done:
                print(
                    f"  [noul/jev] 🎯 JEV determined task completed / sufficiently refined "
                    f"(confidence={conf:.3f}). Search converged at iteration {i}.\n"
                )
                logger.emit_task_completed(confidence=conf, iteration=i)
                logger.emit_iteration_end(i)
                completed_early = True
                break

        logger.emit_iteration_end(i)

    if not root.children:
        print("[mcts] Warning: root has no children after search.")
        logger.emit_complete(root)
        log_path = logger.save()
        return root, str(log_path)

    # ── Final discriminative selection ─────────────────────────────────────────
    best = discriminative_choose_best_action(
        goal, root.state, root.children, system_one_provider=cfg.system_one_provider
    )
    logger.emit_complete(best)

    print(f"\n{'='*60}")
    print(f"MCTS step search complete. Best immediate action: '{best.action_taken}'")
    print(f"  avg_value={best.average_value:.3f}/10  visits={best.visits}")
    print(f"{'='*60}\n")

    log_path = logger.save()
    print(f"[mcts] Step log saved → {log_path}")

    return best, str(log_path)


def get_best_trajectory(root: Node) -> list[Node]:
    """Walk from root through highest-scoring/most-visited children to leaf."""
    trajectory = [root]
    current = root
    while current.children:
        best_child = max(current.children, key=lambda c: (c.visits, c.average_value))
        trajectory.append(best_child)
        current = best_child
    return trajectory


# ── Closed-Loop Execute, Review, Adapt Primitives ──────────────────────────────

def execute_single_action(
    action: str,
    goal: str,
    current_state: str = "",
    workspace_dir: str = ".",
    config: AgentConfig | None = None,
) -> dict[str, Any]:
    """
    Execute ONLY this single immediate action in the workspace using the configured ExecutorProvider.
    """
    cfg = config or load_config()
    executor = get_executor_provider(cfg)
    before = _workspace_file_snapshot(workspace_dir)
    result = executor.execute_action(action, goal, workspace_dir)
    after = _workspace_file_snapshot(workspace_dir)
    result["changed_files"] = sorted(
        path for path in before.keys() | after.keys() if before.get(path) != after.get(path)
    )
    return result


def _workspace_file_snapshot(workspace_dir: str) -> dict[str, tuple[int, int]]:
    """Capture file metadata to identify artifacts changed by one execution."""
    snapshot: dict[str, tuple[int, int]] = {}
    ignored = {".git", ".venv", "node_modules", "__pycache__"}
    for directory, dirs, files in os.walk(workspace_dir):
        dirs[:] = [name for name in dirs if name not in ignored]
        for name in files:
            path = os.path.join(directory, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            snapshot[os.path.relpath(path, workspace_dir)] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def _workspace_inventory(workspace_dir: str) -> str:
    """Give the first planning step a compact view of existing workspace entries."""
    try:
        with os.scandir(workspace_dir) as listing:
            entries = sorted(
                (entry.name + ("/" if entry.is_dir() else ""))
                for entry in listing
                if entry.name not in {".git", ".venv", "node_modules", ".env"}
            )
    except OSError as exc:
        return f"[Workspace inventory unavailable]: {exc}"
    shown = entries[:60]
    remainder = f" (and {len(entries) - 60} more)" if len(entries) > 60 else ""
    return f"[Existing workspace entries]: {', '.join(shown) or '(empty)'}{remainder}"


def review_action(
    action: str,
    execution_result: dict[str, Any],
    workspace_dir: str = ".",
    config: AgentConfig | None = None,
) -> str:
    """
    Observe the environment after execution: capture git status/diff and execution output.
    """
    git_status = ""
    git_diff = ""
    try:
        git_res = subprocess.run(
            ["git", "status", "--short"],
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if git_res.returncode == 0 and git_res.stdout.strip():
            git_status = git_res.stdout.strip()
        diff_res = subprocess.run(
            ["git", "diff", "--stat"], cwd=workspace_dir,
            capture_output=True, text=True, timeout=10,
        )
        if diff_res.returncode == 0:
            git_diff = diff_res.stdout.strip()
    except Exception:
        pass

    changed_files = execution_result.get("changed_files", [])
    changed_sample = changed_files[:8] + changed_files[-5:] if len(changed_files) > 13 else changed_files
    command = str(execution_result.get("command", ""))
    obs_lines: list[str] = []
    if execution_result.get("skipped"):
        execution_result["verified"] = False
        execution_result["verification_confidence"] = 0.0
        obs_lines.append(f"Action '{action}' was skipped and is not verified.")
    elif execution_result.get("success"):
        evidence = (
            f"Command:\n{command[-4000:]}\n"
            f"Output:\n{str(execution_result.get('stdout', ''))[-4000:]}\n"
            f"Files changed by this command ({len(changed_files)} total):\n{changed_sample!s}\n"
            f"Git status:\n{git_status[-2000:]}\nDiff summary:\n{git_diff[-2000:]}"
        )
        verified, confidence = check_action_execution(
            action, evidence, system_one_provider=config.system_one_provider if config else None
        )
        execution_result["verified"] = verified
        execution_result["verification_confidence"] = confidence
        status = "verified" if verified else "not verified"
        obs_lines.append(
            f"Command for action '{action}' returned 0; action {status} "
            f"(confidence={confidence:.3f})."
        )
    else:
        execution_result["verified"] = False
        err = execution_result.get("stderr") or "Non-zero return code"
        obs_lines.append(f"Action '{action}' encountered errors: {err}")

    if changed_files:
        obs_lines.append(f"Files changed: {len(changed_files)}; sample: {changed_sample}")
    if command:
        obs_lines.append(f"Command preview: {command[:300].replace(chr(10), ' ')}")

    stdout = execution_result.get("stdout", "")
    if stdout:
        lines = stdout.splitlines()
        tail = "\n".join(lines[-8:]) if len(lines) > 8 else stdout
        obs_lines.append(f"Output summary:\n{tail}")

    observation = "\n".join(obs_lines)
    print("\n── Step Review / Observation ───────────────────────────────────────")
    print(observation)
    print("────────────────────────────────────────────────────────────────────\n")
    return observation


def adapt_state(
    current_state: str,
    step: int,
    action: str,
    observation: str,
) -> str:
    """
    Synthesize an updated, grounded state incorporating the executed action and observation.
    """
    step_summary = (
        f"\n\n[Step {step} Attempted Action]: {action}\n"
        f"[Step {step} Ground-Truth Observation]:\n{observation}"
    )
    return current_state + step_summary


# ── Top-Level Closed-Loop Orchestrator ─────────────────────────────────────────

def run_closed_loop_agent(
    goal: str,
    initial_state: str,
    *,
    max_steps: int | None = None,
    iterations_per_step: int | None = None,
    actions_per_node: int | None = None,
    simulation_depth: int | None = None,
    expansion_depth: int | None = None,
    early_stop_noul: bool = True,
    execute: bool = True,
    exploration_constant: float = 1.4,
    log_dir: str = "logs",
    workspace_dir: str | None = None,
    config: AgentConfig | None = None,
) -> dict[str, Any]:
    """
    Run the closed-loop MCTS agent with tree reuse and action vocabulary recombination.

    At each step:
      1. Plan & Choose   — MCTS deep-expands candidates and selects the best action.
      2. Execute         — Execute ONLY that action in the workspace.
      3. Review          — Observe git diff, execution logs, and environment changes.
      4. Adapt & Reuse   — Harvest vocabulary, reroot and ground survivor states.
                           Recombine only when survivors miss the threshold.
      5. Assess          — Check goal completion via Noul.
    """
    if max_steps is None and not early_stop_noul:
        raise ValueError(
            "Cannot disable early_stop_noul when max_steps is set to None "
            "(dynamic mode requires early stopping)."
        )

    cfg = config or load_config()
    width = actions_per_node if actions_per_node is not None else cfg.expansion_width
    depth = (
        expansion_depth if expansion_depth is not None
        else simulation_depth if simulation_depth is not None
        else cfg.expansion_depth
    )

    max_dynamic_steps = int(os.getenv("MCTS_DYNAMIC_MAX_STEPS", "50"))
    effective_max_steps = max_steps if max_steps is not None else max_dynamic_steps

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        current_cwd = os.getcwd()
    except Exception:
        current_cwd = "."
    target_workspace = workspace_dir or os.getenv("MCTS_WORKSPACE_DIR") or current_cwd
    resolved_log_dir = log_dir if os.path.isabs(log_dir) else os.path.join(agent_root, log_dir)
    current_state = f"{initial_state}\n{_workspace_inventory(target_workspace)}"
    steps_history: list[dict[str, Any]] = []
    goal_completed = False
    stop_reason = "max_steps"

    # Persistent tree across steps (None on first step)
    _carried_tree: Optional[Node] = None
    _carried_vocab: dict[int, list[str]] = {}

    steps_desc = f"Max Steps: {max_steps}" if max_steps is not None else f"Steps: Dynamic (JEV/Noul, cap={effective_max_steps})"
    iter_desc = f"Iterations/Step: {iterations_per_step}" if iterations_per_step is not None else "Iterations/Step: Dynamic (depth-driven)"

    print(f"\n{'#'*70}")
    print(f"CLOSED-LOOP MCTS AGENT (Plan -> Choose -> Execute -> Review -> Assess)")
    print(f"Run ID: {run_id} | {steps_desc} | {iter_desc}")
    print(f"Planner Provider: {cfg.planner_provider} (model={cfg.planner_model})")
    print(f"Executor Provider: {cfg.executor_provider} (model={cfg.executor_model})")
    print(f"Tree: width={width}, depth={depth}, reuse={cfg.tree_reuse_enabled}, "
          f"threshold={cfg.reuse_score_threshold}, recombined={cfg.n_recombined_paths}")
    print(f"Goal: {goal}")
    print(f"Workspace: {target_workspace}")
    print(f"Logs Dir: {resolved_log_dir}")
    print(f"{'#'*70}\n")

    step = 0
    while True:
        step += 1
        if step > effective_max_steps:
            if max_steps is None:
                print(
                    f"\n⚠️ [closed-loop] Dynamic agent reached safety ceiling of "
                    f"{effective_max_steps} steps without completion. Halting execution.\n"
                )
            break

        step_label = f"{step}/{max_steps}" if max_steps is not None else f"{step} (dynamic)"
        print(f"\n{'*' * 60}")
        print(f" STEP {step_label} (JEV completion assessment enabled)")
        print(f"{'*' * 60}")

        # ── 1. PLAN & CHOOSE ──────────────────────────────────────────────────
        print(f"\n[PLAN & CHOOSE] Evaluating candidates for immediate action...")

        scramble_triggered = False
        reused_tree = False
        if _carried_tree is not None and cfg.tree_reuse_enabled:
            print(f"[tree-reuse] Rescoring {len(_carried_tree.subtree_leaves())} surviving paths...")
            # Build a minimal logger for the rescore phase
            rescore_logger = MCTSLogger(
                goal=goal,
                initial_state=current_state,
                iterations=0,
                log_dir=resolved_log_dir,
                step=step,
                run_id=run_id,
            )
            rescore_logger.emit_init(_carried_tree)
            scores = _rescore_leaves(_carried_tree, goal, rescore_logger, cfg)

            best_survivor_score = max(scores.values()) if scores else 0.0
            if _carried_tree.children and best_survivor_score >= cfg.reuse_score_threshold:
                print(
                    f"[tree-reuse] ✓ Best surviving path score={best_survivor_score:.3f} "
                    f"≥ threshold={cfg.reuse_score_threshold}. Reusing tree."
                )
                root = _carried_tree
                reused_tree = True
                rescore_logger.save()
            else:
                print(f"[tree-reuse] No surviving path reached {cfg.reuse_score_threshold}; trying recombination.")
                n_added = _recombine_paths(
                    _carried_tree, _carried_vocab,
                    n_paths=cfg.n_recombined_paths,
                    path_depth=max(1, depth - 1),
                )
                print(f"[tree-reuse] Recombined {n_added} unique paths.")
                if n_added:
                    scores = _rescore_leaves(_carried_tree, goal, rescore_logger, cfg)
                best_recombined_score = max(scores.values()) if scores else 0.0
                if _carried_tree.children and best_recombined_score >= cfg.reuse_score_threshold:
                    print(f"[tree-reuse] ✓ Recombined tree score={best_recombined_score:.3f}. Reusing tree.")
                    root = _carried_tree
                    reused_tree = True
                else:
                    print(f"[tree-reuse] ✗ No actionable path scored ≥ {cfg.reuse_score_threshold}. Scrambling.")
                    scramble_triggered = True
                rescore_logger.save()

        if _carried_tree is None or not cfg.tree_reuse_enabled or scramble_triggered:
            root = Node(state=current_state)

        best_node, log_path = run_mcts(
            root,
            goal=goal,
            iterations=iterations_per_step,
            actions_per_node=width,
            expansion_depth=depth,
            early_stop_noul=early_stop_noul,
            step=step,
            run_id=run_id,
            exploration_constant=exploration_constant,
            log_dir=resolved_log_dir,
            config=cfg,
            reuse_tree=reused_tree,
        )

        chosen_action = best_node.action_taken
        if not chosen_action:
            print(f"[closed-loop] Warning: No action selected at step {step}.")
            stop_reason = "no_action_generated"
            break

        print(f"\n🎯 [CHOSEN ACTION]: '{chosen_action}' (score={best_node.average_value:.2f}/10)")

        # ── 2. EXECUTE ────────────────────────────────────────────────────────
        print(f"\n[EXECUTE] Executing chosen action in workspace...")
        if execute:
            exec_result = execute_single_action(
                action=chosen_action,
                goal=goal,
                current_state=current_state,
                workspace_dir=target_workspace,
                config=cfg,
            )
        else:
            print("[closed-loop] Execution skipped (--no-execute mode).")
            exec_result = {
                "success": True,
                "skipped": True,
                "stdout": "Dry run (execution skipped).",
                "stderr": "",
                "returncode": 0,
            }

        # ── 3. REVIEW ─────────────────────────────────────────────────────────
        print(f"\n[REVIEW] Inspecting execution outcome & environment changes...")
        observation = review_action(
            action=chosen_action,
            execution_result=exec_result,
            workspace_dir=target_workspace,
            config=cfg,
        )

        # ── 4. ADAPT & TREE REUSE ─────────────────────────────────────────────
        print(f"\n[ADAPT] Grounding real state with observation...")
        current_state = adapt_state(
            current_state=current_state,
            step=step,
            action=chosen_action,
            observation=observation,
        )

        if cfg.tree_reuse_enabled and best_node.action_taken and exec_result.get("verified"):
            print(f"\n[TREE-REUSE] Harvesting action vocabulary and preparing next tree...")

            # Harvest full vocabulary from entire tree BEFORE rerooting
            vocab_by_depth = root.collect_actions_by_depth()
            total_vocab = sum(len(v) for v in vocab_by_depth.values())
            print(f"  Vocabulary: {total_vocab} actions across {len(vocab_by_depth)} depth levels")

            # Reroot at best_node with ground-truth observation
            new_root = _reroot(best_node, observation, grounded_state=current_state)

            surviving_leaves = len(new_root.subtree_leaves())
            print(f"  Preserved {surviving_leaves} leaf paths from chosen subtree.")

            _carried_tree = new_root
            _carried_vocab = vocab_by_depth
        else:
            if not exec_result.get("verified"):
                print("[tree-reuse] Action not verified; discarding hypothetical subtree.")
            _carried_tree = None
            _carried_vocab = {}

        steps_history.append({
            "step": step,
            "action": chosen_action,
            "score": best_node.average_value,
            "visits": best_node.visits,
            "mcts_log": log_path,
            "execution": exec_result,
            "observation": observation,
            "scramble_triggered": scramble_triggered,
        })

        # ── 5. ASSESS ─────────────────────────────────────────────────────────
        print(f"\n[ASSESS] Checking goal completion via Noul...")
        if early_stop_noul and exec_result.get("verified"):
            is_done, conf = check_task_completion(
                goal, current_state, system_one_provider=cfg.system_one_provider
            )
            if is_done:
                print(
                    f"\n🎯 [ASSESS] Goal verified COMPLETED by Noul "
                    f"(confidence={conf:.3f}) after step {step}!"
                )
                goal_completed = True
                stop_reason = "completed"
                break
            else:
                print(
                    f"  [ASSESS] Goal not yet complete (confidence={conf:.3f} "
                    f"< {NOUL_COMPLETION_THRESHOLD}). Proceeding to next step.\n"
                )
        elif early_stop_noul:
            print("  [ASSESS] Skipped: the chosen action was not verified.\n")

    summary = {
        "run_id": run_id,
        "goal": goal,
        "initial_state": initial_state,
        "final_state": current_state,
        "completed": goal_completed,
        "stop_reason": stop_reason,
        "total_steps_run": len(steps_history),
        "steps": steps_history,
    }

    summary_file = save_agent_summary(run_id, summary, log_dir=resolved_log_dir)
    print(f"\n{'='*70}")
    print(f"Closed-Loop Agent Finished! Total steps: {len(steps_history)}")
    print(f"Run Summary saved → {summary_file}")
    print(f"{'='*70}\n")

    return summary
