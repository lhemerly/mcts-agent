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
    PRIME_SIMULATION_DEPTHS,
    batch_check_validity,
    check_task_completion,
    discriminative_choose_best_action,
    evaluate_state,
    get_action_priors,
    select_action_count,
    select_simulation_depth,
)

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


# ── Action proposal via antigravity CLI ────────────────────────────────────────

def _propose_actions(state: str, goal: str, n: int = 3) -> list[str]:
    """
    Ask Gemini (via `agy --print`) to propose `n` distinct next actions,
    requesting them one by one in separate requests. Each request provides the
    goal, state, and already generated actions so far in the current expansion,
    prompting the model to explore distinct, novel angles and strategies.
    """
    if _is_mock_llm():
        import random
        return random.sample(_MOCK_ACTION_POOL, min(n, len(_MOCK_ACTION_POOL)))

    actions: list[str] = []
    proposal_model = os.getenv("AGY_PROPOSAL_MODEL", _AGY_PROPOSAL_MODEL)

    for idx in range(n):
        existing_actions_str = (
            "\n".join(f"- {act}" for act in actions)
            if actions
            else "(No actions proposed yet for this expansion)"
        )

        prompt = textwrap.dedent(f"""\
            You are a creative planning assistant. Given the overall goal, the current
            reasoning state, and candidate actions already proposed so far, propose
            ONE distinct, novel next action exploring a different angle or strategy.

            Goal:
            {goal}

            Current state / context:
            {state}

            Actions already generated so far in this expansion:
            {existing_actions_str}

            Instructions:
            - Be creative and propose a distinct, novel next action exploring a different angle, methodology, or strategy.
            - Do NOT duplicate, overlap, or rephrase the already generated actions.
            - Propose exactly ONE single, concrete action.
            - Output ONLY the single action sentence, with no commentary, numbering, bullets, or preamble.
        """)

        try:
            result = subprocess.run(
                ["agy", "--model", proposal_model, "--print", prompt],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(f"agy exited with code {result.returncode}: {result.stderr.strip()}")
            raw = result.stdout.strip()
            lines = [
                line.lstrip("0123456789.-*#) ").strip().strip('"\'')
                for line in raw.splitlines()
                if line.lstrip("0123456789.-*#) ").strip()
            ]
            if not lines:
                raise ValueError(f"agy returned no parseable action. stdout: {raw!r}")

            chosen_action = None
            for candidate in lines:
                if candidate and candidate not in actions:
                    chosen_action = candidate
                    break
            if not chosen_action:
                chosen_action = lines[0]

            actions.append(chosen_action)
        except Exception as exc:
            print(f"[mcts] agy action proposal failed for candidate {idx + 1}: {exc}. Using fallback from mock pool.")
            import random
            unused_mock = [a for a in _MOCK_ACTION_POOL if a not in actions]
            if unused_mock:
                actions.append(random.choice(unused_mock))
            elif _MOCK_ACTION_POOL:
                actions.append(random.choice(_MOCK_ACTION_POOL))
            else:
                actions.append(f"Alternative strategic exploration step {idx + 1}")

    return actions


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


def _simulate(
    node: Node,
    goal: str,
    logger: MCTSLogger,
    *,
    rollout_depth: int = 1,
) -> float:
    """
    Multi-step rollout simulation:
    Starting from node.state, simulate lookahead steps up to `rollout_depth`.
    At each simulated step:
      - Propose candidate actions from current simulated state
      - Prune invalid actions via Noul
      - Select the most promising simulated step via Choice priors
      - Advance the simulated state
    Finally, evaluate the projected horizon state using Score.
    """
    logger.emit_eval(node)

    current_sim_state = node.state
    if rollout_depth > 1:
        print(f"  [simulate] Projecting {rollout_depth} steps ahead into the future...")
        for step_idx in range(1, rollout_depth):
            cands = _propose_actions(current_sim_state, goal, n=2)
            validity = batch_check_validity(current_sim_state, cands)
            valid_cands = [a for a in cands if validity.get(a, (False, 0.0))[0]]
            if not valid_cands:
                valid_cands = cands

            priors = get_action_priors(current_sim_state, valid_cands)
            sim_action = max(valid_cands, key=lambda a: priors.get(a, 0.0))
            current_sim_state += f"\n[Simulated Lookahead Step {step_idx + 1}]: {sim_action}"
            print(f"    [rollout step {step_idx + 1}/{rollout_depth}]: '{sim_action[:60]}'")

    value = evaluate_state(goal, current_sim_state)
    depth_note = f" (sim depth={rollout_depth})" if rollout_depth > 1 else ""
    print(f"  [score] Value={value:.3f}/10 for action='{node.action_taken}'{depth_note}")
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
    simulation_depth: int | None = None,
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
    simulation_depth : int | None
        Fixed number of lookahead rollout steps to project into the future, or None
        to use TypeSafe Choice to dynamically choose prime depth from (2, 3, 5).
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

    # Dynamic lookahead depth from primes (2, 3, 5) if not fixed
    if simulation_depth is not None:
        rollout_depth = simulation_depth
    else:
        rollout_depth = select_simulation_depth(root.state, goal)

    step_info = f"Step {step} | " if step is not None else ""
    mode_info = (
        f"max_iter={iterations} | "
        f"actions={'dynamic (primes <= 13)' if actions_per_node is None else actions_per_node} | "
        f"sim_depth={rollout_depth}"
    )
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

        # 2. Simulation (multi-step lookahead rollout)
        value = _simulate(leaf, goal, logger, rollout_depth=rollout_depth)

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

    if not root.children:
        print("[mcts] Warning: root has no children after search.")
        logger.emit_complete(root)
        log_path = logger.save()
        return root, str(log_path)

    # Use TypeSafe Choice to make the final discriminative decision on which branch to commit to
    best = discriminative_choose_best_action(goal, root.state, root.children)
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
) -> dict[str, Any]:
    """
    Execute ONLY this single immediate action in the real workspace using agy.
    Uses a lean execution context omitting accumulated history/outputs to keep
    the execution prompt focused on the immediate task and essential goal.
    """
    if _is_mock_llm():
        return {
            "success": True,
            "stdout": f"[mock] Successfully executed action: {action}",
            "stderr": "",
            "returncode": 0,
        }

    abs_workspace = os.path.abspath(workspace_dir)
    prompt = textwrap.dedent(f"""\
        You are the execution agent in a closed-loop reasoning system.

        Goal:
        {goal.strip()}

        Target Workspace Directory:
        {abs_workspace}

        Task:
        Execute ONLY this specific immediate action now in this workspace:
        >>> {action.strip()} <<<

        All files created or modified MUST be written inside {abs_workspace}.
        Apply the necessary edits, write the code, or run the commands required for this action.
        Always execute commands synchronously to full completion in the foreground; do not leave background tasks running.
        Do NOT attempt to execute future hypothetical steps beyond this immediate action.
    """)

    print(f"\n{'='*60}")
    print(f"[executor] Executing action with agy in {abs_workspace}: '{action}'")
    print(f"{'='*60}\n")

    exec_timeout = int(os.getenv("MCTS_EXEC_TIMEOUT", "1800"))
    timeout_str = f"{max(1, exec_timeout // 60)}m0s"
    cmd = [
        "agy",
        "--model", _AGY_MODEL,
        "--mode", "accept-edits",
        "--add-dir", abs_workspace,
        "--dangerously-skip-permissions",
        "--print-timeout", timeout_str,
        "--print",
        prompt,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            timeout=exec_timeout,
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
    simulation_depth: int | None = None,
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
            simulation_depth=simulation_depth,
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
