"""
mcts.py — Core MCTS loop with structured event logging.

Implements the four classic stages:
  1. Selection   — PUCT traversal (Q normalized to [0,1])
  2. Expansion   — Gemini (via agy) proposes actions → Noul prune → Choice priors
  3. Simulation  — Score evaluates the simulated state
  4. Backpropagation — propagate value back to root

Every event is recorded by an MCTSLogger and saved as a JSON log
that the MCTS visualizer can replay.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from datetime import datetime
from typing import Any, Optional

from agent.logger import MCTSLogger, save_agent_summary
from agent.node import Node
from agent.primitives import (
    NOUL_COMPLETION_THRESHOLD,
    batch_check_validity,
    check_task_completion,
    evaluate_state,
    get_action_priors,
    select_action_count,
)

_USE_MOCK_LLM = os.getenv("USE_MOCK_PRIMITIVES", "false").lower() in ("1", "true", "yes")
_AGY_MODEL = os.getenv("AGY_MODEL", "gemini-3.8-flash-medium")

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


# ── Action proposal via antigravity CLI ────────────────────────────────────────

def _propose_actions(state: str, goal: str, n: int = 3) -> list[str]:
    """
    Ask Gemini 3.8 Flash (via `agy --print`) to propose `n` distinct next
    actions. agy is the authenticated Gemini harness — no API key needed.
    """
    if _USE_MOCK_LLM:
        import random
        return random.sample(_MOCK_ACTION_POOL, min(n, len(_MOCK_ACTION_POOL)))

    prompt = textwrap.dedent(f"""\
        You are a planning assistant. Given the current reasoning state and the
        overall goal, propose exactly {n} distinct, concrete next actions.

        Goal:
        {goal}

        Current state / context:
        {state}

        Output ONLY a numbered list of {n} short action sentences (one per line),
        with no extra commentary or preamble. Example format:
        1. <action one>
        2. <action two>
        3. <action three>
    """)

    try:
        result = subprocess.run(
            ["agy", "--model", _AGY_MODEL, "--print", prompt],
            capture_output=True, text=True, timeout=60,
        )
        raw = result.stdout.strip()
        actions = [
            line.lstrip("0123456789.-) ").strip()
            for line in raw.splitlines()
            if line.lstrip("0123456789.-) ").strip()
        ]
        if not actions:
            raise ValueError(f"agy returned no parseable actions. stdout: {raw!r}")
        return actions[:n]
    except Exception as exc:
        print(f"[mcts] agy action proposal failed: {exc}. Using mock pool.")
        import random
        return random.sample(_MOCK_ACTION_POOL, min(n, len(_MOCK_ACTION_POOL)))


# ── MCTS stages ────────────────────────────────────────────────────────────────

def _select(root: Node, c: float, logger: MCTSLogger) -> Node:
    node = root
    while not node.is_leaf:
        node = max(node.children, key=lambda ch: ch.puct_score(c))
    logger.emit_select(_path_to_root(node))
    return node


def _expand(
    leaf: Node,
    goal: str,
    actions_per_node: int | None,
    logger: MCTSLogger,
) -> list[Node]:
    if actions_per_node is not None:
        n_actions = actions_per_node
    else:
        # Dynamically select branching factor from primes up to 13 (2, 3, 5, 7, 11, 13) via Choice
        n_actions = select_action_count(leaf.state, goal)

    candidates = _propose_actions(leaf.state, goal, n=n_actions)
    print(f"  [expand] Proposed {len(candidates)} action(s) (target {n_actions}): {candidates}")
    logger.emit_candidates(candidates)

    # Batched Noul gate — ONE system_one() call
    validity = batch_check_validity(leaf.state, candidates)
    valid_actions: list[str] = []
    for action in candidates:
        is_valid, conf = validity.get(action, (False, 0.0))
        symbol = "✓" if is_valid else "✗"
        print(f"  [noul] {symbol} '{action[:60]}' (confidence={conf:.3f})")
        logger.emit_noul(action, is_valid, conf)
        if is_valid:
            valid_actions.append(action)

    if not valid_actions:
        print("  [expand] All actions pruned — keeping candidates as fallback.")
        valid_actions = candidates

    priors = get_action_priors(leaf.state, valid_actions)

    new_children: list[Node] = []
    for action in valid_actions:
        simulated = f"{leaf.state}\n[Action taken]: {action}"
        child = Node(
            state=simulated,
            parent=leaf,
            action_taken=action,
            prior_probability=priors.get(action, 1.0 / len(valid_actions)),
        )
        val = evaluate_state(goal, child.state)
        child.visits = 1
        child.value_sum = val
        print(f"  [score] Value={val:.3f}/10 for candidate='{action[:60]}'")
        logger.emit_eval(child)
        logger.emit_score(child, val)
        new_children.append(child)
        logger.emit_new_node(child)

    leaf.children.extend(new_children)
    return new_children


def _simulate(node: Node, goal: str, logger: MCTSLogger) -> float:
    logger.emit_eval(node)
    value = evaluate_state(goal, node.state)
    print(f"  [score] Value={value:.3f}/10 for action='{node.action_taken}'")
    logger.emit_score(node, value)
    return value


def _backpropagate(node: Node, value: float, logger: MCTSLogger) -> None:
    path = _path_to_root(node)
    logger.emit_backprop_start(path, value)
    current: Optional[Node] = node
    while current is not None:
        current.visits += 1
        current.value_sum += value
        logger.emit_node_update(current)
        current = current.parent


# ── Public interface ───────────────────────────────────────────────────────────

# ── Public interface ───────────────────────────────────────────────────────────

def run_mcts(
    root: Node,
    goal: str,
    *,
    iterations: int = 15,
    actions_per_node: int | None = None,
    early_stop_noul: bool = True,
    step: int | None = None,
    run_id: str | None = None,
    execute: bool = False,
    exploration_constant: float = 1.4,
    log_dir: str = "logs",
) -> tuple[Node, str]:
    """
    Run MCTS from `root` to evaluate candidates and select the single best immediate action.

    Parameters
    ----------
    iterations : int
        Maximum number of iterations.
    actions_per_node : int | None
        Fixed number of candidate actions to propose, or None to use TypeSafe Choice
        to dynamically choose branching variance among primes <= 13 (2, 3, 5, 7, 11, 13).
    early_stop_noul : bool
        If True, queries Noul at each iteration to detect if the task has been
        completed or refined enough, stopping search early when confidence is high.
    step : int | None
        Step number in a multi-step closed-loop run.
    run_id : str | None
        Unique run ID to group multi-step logs.
    exploration_constant : float
        PUCT exploration parameter.

    Returns
    -------
    (best_child, log_path) — best immediate action node + path to the JSON log.
    """
    logger = MCTSLogger(
        goal=goal,
        initial_state=root.state,
        iterations=iterations,
        log_dir=log_dir,
        step=step,
        run_id=run_id,
    )
    logger.emit_init(root)

    step_info = f"Step {step} | " if step is not None else ""
    mode_info = f"max_iter={iterations} | actions={'dynamic (primes <= 13)' if actions_per_node is None else actions_per_node}"
    print(f"\n{'='*60}")
    print(f"Starting MCTS  |  {step_info}{mode_info}  |  goal='{goal[:80]}'")
    print(f"{'='*60}\n")

    completed_early = False
    for i in range(1, iterations + 1):
        print(f"── Iteration {i}/{iterations} ──────────────────────────────────")
        logger.emit_iteration_start(i)

        # 1. Selection
        leaf = _select(root, exploration_constant, logger)
        print(f"  [select] Leaf: visits={leaf.visits}, action='{leaf.action_taken}'")

        # 2. Simulation
        value = _simulate(leaf, goal, logger)

        # 3. Expansion
        if leaf.is_leaf:
            _expand(leaf, goal, actions_per_node, logger)

        # 4. Backpropagation
        _backpropagate(leaf, value, logger)
        print(f"  [backprop] Value {value:.3f} propagated up the tree.\n")

        # 5. Check early completion via Noul
        if early_stop_noul and i >= 2 and root.children:
            best_so_far = max(root.children, key=lambda c: (c.visits, c.average_value))
            is_done, conf = check_task_completion(goal, best_so_far.state)
            if is_done:
                print(f"  [noul] 🎯 Task completed / sufficiently refined (confidence={conf:.3f}). Early stopping at iteration {i}.\n")
                logger.emit_task_completed(confidence=conf, iteration=i)
                logger.emit_iteration_end(i)
                completed_early = True
                break

        logger.emit_iteration_end(i)

    # Best action = highest visits among root's direct children (standard robust MCTS)
    # Tiebreaker: average value
    if not root.children:
        print("[mcts] Warning: root has no children after search.")
        logger.emit_complete(root)
        log_path = logger.save()
        return root, str(log_path)

    best = max(root.children, key=lambda c: (c.average_value, c.visits, c.prior_probability))
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
    current_state: str,
    workspace_dir: str = ".",
) -> dict[str, Any]:
    """
    Execute ONLY this single immediate action in the real workspace using agy.
    """
    if _USE_MOCK_LLM:
        return {
            "success": True,
            "stdout": f"[mock] Successfully executed action: {action}",
            "stderr": "",
            "returncode": 0,
        }

    prompt = textwrap.dedent(f"""\
        You are the execution agent in a closed-loop reasoning system.
        
        Goal:
        {goal}
        
        Current context / state:
        {current_state}
        
        Task:
        Execute ONLY this specific immediate action now in this workspace:
        >>> {action} <<<
        
        Apply the necessary edits, write the code, or run the commands required for this action.
        Do NOT attempt to execute future hypothetical steps beyond this immediate action.
    """)

    print(f"\n{'='*60}")
    print(f"[executor] Executing action with agy: '{action}'")
    print(f"{'='*60}\n")

    cmd = [
        "agy",
        "--model", _AGY_MODEL,
        "--mode", "accept-edits",
        "--dangerously-skip-permissions",
        "--print-timeout", "10m0s",
        "--print",
        prompt,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )
        return {
            "success": proc.returncode == 0,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "returncode": proc.returncode,
        }
    except Exception as exc:
        print(f"[executor] agy execution failed: {exc}")
        return {
            "success": False,
            "stdout": "",
            "stderr": str(exc),
            "returncode": -1,
        }


def review_action(
    action: str,
    execution_result: dict[str, Any],
    workspace_dir: str = ".",
) -> str:
    """
    Observe the environment after execution: capture git status/diff and execution output.
    """
    git_status = ""
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
    except Exception:
        pass

    obs_lines: list[str] = []
    if execution_result.get("success"):
        obs_lines.append(f"Action '{action}' executed successfully (returncode 0).")
    else:
        err = execution_result.get("stderr") or "Non-zero return code"
        obs_lines.append(f"Action '{action}' encountered errors: {err}")

    if git_status:
        obs_lines.append(f"Modified workspace files:\n{git_status}")

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
        f"\n\n[Step {step} Completed Action]: {action}\n"
        f"[Step {step} Ground-Truth Observation]:\n{observation}"
    )
    return current_state + step_summary


# ── Top-Level Closed-Loop Orchestrator ─────────────────────────────────────────

def run_closed_loop_agent(
    goal: str,
    initial_state: str,
    *,
    max_steps: int = 5,
    iterations_per_step: int = 1,
    actions_per_node: int | None = None,
    early_stop_noul: bool = True,
    execute: bool = True,
    exploration_constant: float = 1.4,
    log_dir: str = "logs",
    workspace_dir: str | None = None,
) -> dict[str, Any]:
    """
    Run the closed-loop MCTS agent:
      At each step:
        1. Plan & Choose: MCTS evaluates candidate actions and selects best immediate action.
        2. Execute: Execute ONLY that action in the workspace.
        3. Review: Observe git diff, execution logs, and environment changes.
        4. Adapt: Update real state.
        5. Assess: Check goal completion via Noul.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_workspace = workspace_dir or os.getenv("MCTS_WORKSPACE_DIR") or agent_root
    resolved_log_dir = log_dir if os.path.isabs(log_dir) else os.path.join(agent_root, log_dir)
    current_state = initial_state
    steps_history: list[dict[str, Any]] = []
    goal_completed = False

    print(f"\n{'#'*70}")
    print(f"CLOSED-LOOP MCTS AGENT (Plan -> Choose -> Execute -> Review -> Assess)")
    print(f"Run ID: {run_id} | Max Steps: {max_steps} | Search Iterations/Step: {iterations_per_step}")
    print(f"Goal: {goal}")
    print(f"Workspace: {target_workspace}")
    print(f"Logs Dir: {resolved_log_dir}")
    print(f"{'#'*70}\n")

    for step in range(1, max_steps + 1):
        print(f"\n{'*' * 60}")
        print(f" STEP {step} (Max ceiling: {max_steps} | Dynamic early-stop enabled)")
        print(f"{'*' * 60}")

        # 1. PLAN & CHOOSE: evaluate candidate actions from current real state
        print(f"\n[PLAN & CHOOSE] Evaluating candidates for immediate action...")
        root = Node(state=current_state)
        best_node, log_path = run_mcts(
            root,
            goal=goal,
            iterations=iterations_per_step,
            actions_per_node=actions_per_node,
            early_stop_noul=early_stop_noul,
            step=step,
            run_id=run_id,
            exploration_constant=exploration_constant,
            log_dir=resolved_log_dir,
        )

        chosen_action = best_node.action_taken
        if not chosen_action:
            print(f"[closed-loop] Warning: No action selected at step {step}.")
            break

        print(f"\n🎯 [CHOSEN ACTION]: '{chosen_action}' (score={best_node.average_value:.2f}/10)")

        # 2. EXECUTE
        print(f"\n[EXECUTE] Executing chosen action in workspace...")
        if execute:
            exec_result = execute_single_action(
                action=chosen_action,
                goal=goal,
                current_state=current_state,
                workspace_dir=target_workspace,
            )
        else:
            print("[closed-loop] Execution skipped (--no-execute mode).")
            exec_result = {
                "success": True,
                "stdout": "Dry run (execution skipped).",
                "stderr": "",
                "returncode": 0,
            }

        # 3. REVIEW
        print(f"\n[REVIEW] Inspecting execution outcome & environment changes...")
        observation = review_action(
            action=chosen_action,
            execution_result=exec_result,
            workspace_dir=target_workspace,
        )

        # 4. ADAPT
        print(f"\n[ADAPT] Grounding real state with observation...")
        current_state = adapt_state(
            current_state=current_state,
            step=step,
            action=chosen_action,
            observation=observation,
        )

        steps_history.append({
            "step": step,
            "action": chosen_action,
            "score": best_node.average_value,
            "visits": best_node.visits,
            "mcts_log": log_path,
            "execution": exec_result,
            "observation": observation,
        })

        # 5. ASSESS
        print(f"\n[ASSESS] Checking goal completion via Noul...")
        if early_stop_noul:
            is_done, conf = check_task_completion(goal, current_state)
            if is_done:
                print(f"\n🎯 [ASSESS] Goal verified COMPLETED by Noul (confidence={conf:.3f}) after step {step}!")
                goal_completed = True
                break
            else:
                print(f"  [ASSESS] Goal not yet complete (confidence={conf:.3f} < {NOUL_COMPLETION_THRESHOLD}). Proceeding to next step.\n")

    summary = {
        "run_id": run_id,
        "goal": goal,
        "initial_state": initial_state,
        "final_state": current_state,
        "completed": goal_completed,
        "total_steps_run": len(steps_history),
        "steps": steps_history,
    }

    summary_file = save_agent_summary(run_id, summary, log_dir=resolved_log_dir)
    print(f"\n{'='*70}")
    print(f"Closed-Loop Agent Finished! Total steps: {len(steps_history)}")
    print(f"Run Summary saved → {summary_file}")
    print(f"{'='*70}\n")

    return summary
