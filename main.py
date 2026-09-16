"""
main.py — Entry point for the Discriminative MCTS Agent.

Usage
-----
  python main.py                   # live demo (requires TYPESAFE_API_KEY)
  python main.py --mock            # mock mode, no API calls
  python main.py --interactive     # supply your own goal
  python main.py --iterations 15   # more search iterations

After each run a JSON log is saved to logs/ and the path is printed.
Open visualizer.html in a browser and load the log to replay the search.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(_ENV_PATH)

if not os.getenv("TYPESAFE_API_KEY") and os.getenv("API_KEY"):
    os.environ["TYPESAFE_API_KEY"] = os.environ["API_KEY"]

from agent.mcts import (
    adapt_state,
    execute_single_action,
    review_action,
    run_closed_loop_agent,
    run_mcts,
)
from agent.node import Node

_DEMO_GOAL = (
    "Design a minimal REST API in Python (FastAPI) for a task management app: "
    "endpoints for creating, listing, updating, and deleting tasks, with in-memory storage."
)
_DEMO_INITIAL_STATE = (
    "We are starting from scratch. No code exists yet. "
    "The API must support: POST /tasks (create), GET /tasks (list all), "
    "PATCH /tasks/{id} (update), DELETE /tasks/{id} (delete). "
    "Storage is in-memory (a Python dict). No auth required."
)


def print_tree(node: Node, prefix: str = "", is_last: bool = True) -> None:
    connector = "└── " if is_last else "├── "
    label = f'"{node.action_taken}"' if node.action_taken else "[root]"
    print(
        f"{prefix}{connector}{label}  "
        f"(visits={node.visits}, avg={node.average_value:.2f}/10, "
        f"prior={node.prior_probability:.2f})"
    )
    child_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(node.children):
        print_tree(child, child_prefix, i == len(node.children) - 1)


def run_demo(
    iterations: int,
    max_steps: int = 5,
    actions_per_node: int | None = None,
    goal: str | None = None,
    initial_state: str | None = None,
    early_stop_noul: bool = True,
    execute: bool = True,
    workspace_dir: str | None = None,
) -> None:
    mock = os.getenv("USE_MOCK_PRIMITIVES", "false").lower() in ("1", "true", "yes")
    active_goal = goal or _DEMO_GOAL
    active_state = initial_state or _DEMO_INITIAL_STATE
    print(f"\n[main] Running Closed-Loop MCTS in {'MOCK' if mock else 'LIVE'} mode.")
    print(f"[main] Goal: {active_goal}")
    print(f"[main] Max Steps: {max_steps} | Iterations per step: {iterations}\n")

    summary = run_closed_loop_agent(
        goal=active_goal,
        initial_state=active_state,
        max_steps=max_steps,
        iterations_per_step=iterations,
        actions_per_node=actions_per_node,
        early_stop_noul=early_stop_noul,
        execute=execute,
        workspace_dir=workspace_dir,
    )

    print("\n── Summary of Executed Steps ───────────────────────────────────────")
    for s in summary.get("steps", []):
        print(f"Step {s['step']}: {s['action']} (avg score: {s['score']:.2f}/10, visits: {s['visits']})")
        print(f"  Log: {s['mcts_log']}")

    print(f"\nGoal Status: {'COMPLETED' if summary.get('completed') else 'MAX STEPS REACHED'}")
    print(f"Open visualizer.html in your browser to load any individual step's search tree log.")


def run_interactive(
    iterations: int,
    max_steps: int = 5,
    actions_per_node: int | None = None,
    early_stop_noul: bool = True,
    execute: bool = True,
) -> None:
    print("\n[main] Interactive Closed-Loop MCTS Agent  (type 'quit' to exit)\n")
    goal = input("Enter your goal: ").strip()
    if not goal or goal.lower() == "quit":
        return
    initial_state = input("Enter the initial context/state (Enter for default): ").strip()
    if not initial_state:
        initial_state = f"Starting state for goal: {goal}"

    current_state = initial_state
    workspace_dir = os.path.dirname(os.path.abspath(__file__))

    for step in range(1, max_steps + 1):
        print(f"\n═══ Turn {step} / {max_steps} ═══════════════════════════════════════════════════")
        root = Node(state=current_state)
        best, log_path = run_mcts(
            root,
            goal=goal,
            iterations=iterations,
            actions_per_node=actions_per_node,
            early_stop_noul=early_stop_noul,
            step=step,
        )

        chosen_action = best.action_taken
        if not chosen_action:
            print("[main] No action proposed. Ending run.")
            break

        print(f"\nRecommended action: {chosen_action}")
        print(f"Step MCTS Log: {os.path.abspath(log_path)}")

        prompt_msg = "\n[E]xecute action, [S]kip execution, [N]ew state manually, or [Q]uit? [E/s/n/q]: "
        user_choice = input(prompt_msg).strip().lower()

        if user_choice == "q" or user_choice == "quit":
            break

        if user_choice == "n":
            custom_state = input("Enter new state description: ").strip()
            if custom_state:
                current_state = custom_state
            continue

        if user_choice == "s" or not execute:
            print("[main] Skipping execution of this step.")
            exec_res = {"success": True, "stdout": "Execution skipped by user.", "stderr": "", "returncode": 0}
        else:
            exec_res = execute_single_action(chosen_action, goal, current_state, workspace_dir=workspace_dir)

        observation = review_action(chosen_action, exec_res, workspace_dir=workspace_dir)
        current_state = adapt_state(current_state, step=step, action=chosen_action, observation=observation)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discriminative MCTS Agent — Closed-Loop MVP")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("MCTS_MAX_STEPS", "5")), help="Max steps in the execute-review-adapt loop (default: 5)")
    parser.add_argument("--iterations", "-n", type=int, default=int(os.getenv("MCTS_ITERATIONS", "10")), help="MCTS iterations per step (default: 10)")
    parser.add_argument("--actions", "-a", type=int, default=None, help="Fixed actions per node (default: dynamic prime selection up to 13)")
    parser.add_argument("--goal", "-g", type=str, default=None, help="Custom goal for MCTS agent")
    parser.add_argument("--state", "-s", type=str, default=None, help="Initial state / context for search")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable Noul-based completion early stopping")
    parser.add_argument("--no-execute", action="store_true", help="Skip executing actions in the workspace")
    parser.add_argument("--workspace", "-w", type=str, default=None, help="Working directory for agent execution")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock:
        os.environ["USE_MOCK_PRIMITIVES"] = "true"

    mock_mode = os.getenv("USE_MOCK_PRIMITIVES", "false").lower() in ("1", "true", "yes")
    if not mock_mode and not os.getenv("TYPESAFE_API_KEY"):
        print("[main] WARNING: TYPESAFE_API_KEY not set. Pass --mock to skip live calls.")

    early_stop = not args.no_early_stop
    execute_plan = not args.no_execute
    if args.interactive:
        run_interactive(
            args.iterations,
            max_steps=args.max_steps,
            actions_per_node=args.actions,
            early_stop_noul=early_stop,
            execute=execute_plan,
        )
    else:
        run_demo(
            args.iterations,
            max_steps=args.max_steps,
            actions_per_node=args.actions,
            goal=args.goal,
            initial_state=args.state,
            early_stop_noul=early_stop,
            execute=execute_plan,
            workspace_dir=args.workspace,
        )


if __name__ == "__main__":
    main()
