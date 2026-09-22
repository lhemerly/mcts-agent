"""
agent/providers.py — Pluggable Planner and Executor providers for mcts-agent.

Architecture: harness-of-harnesses
  mcts-agent never connects to models directly.  Every provider delegates to an
  *agent harness* (AGY, Pi, Codex, …) which owns its own tool loop, file
  read/write capabilities, and model backend.  mcts-agent only cares about the
  result: did the harness succeed and what changed in the workspace?

Planner harnesses  (propose candidate action strings, no workspace access):
  - AGYPlannerProvider   — Antigravity CLI  (`agy --print`)
  - PiPlannerProvider    — Pi coding agent  (`pi --print`)
  - MockPlannerProvider  — Deterministic mock for unit tests

Executor harnesses  (carry out a single action in the workspace):
  - AGYExecutorProvider  — Antigravity CLI  (`agy --mode accept-edits --print`)
  - PiExecutorProvider   — Pi coding agent  (`pi --print`)
  - MockExecutorProvider — No-op mock for unit tests
"""

from __future__ import annotations

import os
import random
import re
import subprocess
import textwrap
from abc import ABC, abstractmethod
from typing import Any, Optional

from agent.config import AgentConfig


def _safe_abspath(path: str) -> str:
    """Return an absolute path for *path*, falling back gracefully when the
    current working directory no longer exists (e.g. it was deleted while the
    process was running).  ``os.path.abspath`` internally calls ``os.getcwd()``
    for relative paths, which raises ``FileNotFoundError`` in that situation."""
    try:
        return os.path.abspath(path)
    except OSError:
        # cwd is gone – anchor relative paths to the user's home directory
        # so the executor still has a valid, writable location to work in.
        if os.path.isabs(path):
            return path
        return os.path.join(os.path.expanduser("~"), path)


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

_ATOMIC_ACTION_RULES = (
    "Propose ONE small, atomic next action with ONE observable result. "
    "Limit it to one file edit or one short workspace operation. "
    "Do not combine steps with 'and', 'then', or a sequence of tasks. "
    "For example, 'Clone the repository' and 'Install dependencies' are two "
    "separate actions. Do not propose the entire feature or a multi-file build."
)


def _action_key(action: str) -> str:
    """Normalize superficial punctuation and spacing in generated actions."""
    return re.sub(r"[^\w]+", " ", action.casefold()).strip()


# ── Base Interfaces ────────────────────────────────────────────────────────────

class BasePlannerProvider(ABC):
    @abstractmethod
    def propose_actions(
        self,
        state: str,
        goal: str,
        n: int = 3,
        explored_actions: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Propose n distinct candidate actions given the current state and goal.

        Parameters
        ----------
        state : str
            Current node state context.
        goal : str
            Overall goal description.
        n : int
            Number of distinct actions to propose.
        explored_actions : list[str] | None
            All actions already present anywhere in the search tree.
            Planners should inject these into their prompts to avoid duplicates.
        """


class BaseExecutorProvider(ABC):
    @abstractmethod
    def execute_action(
        self, action: str, goal: str, workspace_dir: str
    ) -> dict[str, Any]:
        """Execute a single action in the workspace and return execution results."""


# ── Planner Implementations ────────────────────────────────────────────────────

class AGYPlannerProvider(BasePlannerProvider):
    def __init__(self, model: str = "gemini-3.6-flash-low"):
        self.model = model

    def propose_actions(
        self,
        state: str,
        goal: str,
        n: int = 3,
        explored_actions: Optional[list[str]] = None,
    ) -> list[str]:
        if os.getenv("USE_MOCK_PRIMITIVES", "false").lower() in ("1", "true", "yes"):
            return random.sample(_MOCK_ACTION_POOL, min(n, len(_MOCK_ACTION_POOL)))

        explored_str = (
            "\n".join(f"- {a}" for a in (explored_actions or []))
            or "(None yet)"
        )

        actions: list[str] = []
        repeated_suggestions: dict[str, int] = {}
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

                Actions ALREADY EXPLORED anywhere in the search tree (DO NOT reproduce these):
                {explored_str}

                Actions already proposed in this current expansion batch:
                {existing_actions_str}

                Instructions:
                - {_ATOMIC_ACTION_RULES}
                - Choose a concrete step that can be completed and checked before the next step is planned.
                - {random.choice(_CREATIVE_STRATEGIES)}
                - If the obvious answer repeats an action above, brainstorm alternatives privately and output the second or third best distinct action.
                - Do NOT duplicate, overlap, or rephrase any action listed above (explored or batch).
                - Output ONLY the single action sentence, with no commentary, numbering, bullets, or preamble.
            """)

            try:
                result = subprocess.run(
                    ["agy", "--model", self.model, "--print", prompt],
                    capture_output=True,
                    text=True,
                    timeout=60,
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
                all_explored = {_action_key(action) for action in (explored_actions or []) + actions}
                for candidate in lines:
                    if candidate and _action_key(candidate) not in all_explored:
                        chosen_action = candidate
                        break
                if not chosen_action:
                    key = _action_key(lines[0])
                    repeated_suggestions[key] = repeated_suggestions.get(key, 0) + 1
                    print(f"[planner/agy] Candidate {idx + 1}/{n} repeated: {lines[0]!r}")
                    if repeated_suggestions[key] >= 4:
                        print("[planner/agy] Same action suggested four times; stopping proposal batch.")
                        break
                    continue

                key = _action_key(chosen_action)
                repeated_suggestions[key] = repeated_suggestions.get(key, 0) + 1
                actions.append(chosen_action)
                print(f"[planner/agy] Generated candidate {idx + 1}/{n}: {chosen_action!r}")
            except Exception as exc:
                print(f"[planner/agy] Candidate {idx + 1}/{n} failed: {exc}. Skipping slot.")

        return actions


_CREATIVE_STRATEGIES: list[str] = [
    "Propose a direct, practical next action to make immediate progress toward the goal.",
    "Brainstorm three atomic next actions privately; output the SECOND best action, not the obvious first choice.",
    "Brainstorm four atomic next actions privately; output the THIRD best action, not the obvious first choice.",
    "Go crazy and think outside the box: find an unconventional but executable tiny step.",
    "Switch perspective to a tester or maintainer and choose a different concrete next step.",
]


class PiPlannerProvider(BasePlannerProvider):
    """Planner using the Pi agent harness (`pi --print`).

    Pi owns its own model backend configuration (Ollama, OpenAI, Anthropic, etc.)
    via ~/.pi/agent/models.json.  mcts-agent passes only the planning prompt and
    reads back the proposed action string.  The model flag is forwarded to Pi as
    `--model <model>` so you can override the default from the mcts-agent config
    without changing Pi's global settings.
    """

    def __init__(self, model: str = "", timeout: int = 60):
        self.model = model
        self.timeout = timeout

    def propose_actions(
        self,
        state: str,
        goal: str,
        n: int = 3,
        explored_actions: Optional[list[str]] = None,
    ) -> list[str]:
        actions: list[str] = []
        repeated_suggestions: dict[str, int] = {}
        explored_str = (
            "\n".join(f"- {a}" for a in (explored_actions or []))
            or "(None yet)"
        )

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

                Actions ALREADY EXPLORED anywhere in the search tree (DO NOT reproduce these):
                {explored_str}

                Actions already proposed in this current expansion batch:
                {existing_actions_str}

                Instructions:
                - {_ATOMIC_ACTION_RULES}
                - Choose a concrete step that can be completed and checked before the next step is planned.
                - {random.choice(_CREATIVE_STRATEGIES)}
                - If the obvious answer repeats an action above, brainstorm alternatives privately and output the second or third best distinct action.
                - Do NOT duplicate, overlap, or rephrase any action listed above (explored or batch).
                - Output ONLY the single action sentence, with no commentary, numbering, bullets, or preamble.
            """)

            cmd = ["pi", "--no-session", "--print", prompt]
            if self.model:
                cmd = ["pi", "--no-session", "--model", self.model, "--print", prompt]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"pi exited with code {result.returncode}: {result.stderr.strip()}")
                raw = result.stdout.strip()
                lines = [
                    line.lstrip("0123456789.-*#) ").strip().strip('"\'')
                    for line in raw.splitlines()
                    if line.lstrip("0123456789.-*#) ").strip()
                ]
                if not lines:
                    raise ValueError(f"pi returned no parseable action. stdout: {raw!r}")

                chosen_action = None
                all_explored = {_action_key(a) for a in (explored_actions or []) + actions}
                for candidate in lines:
                    if candidate and _action_key(candidate) not in all_explored:
                        chosen_action = candidate
                        break
                if not chosen_action:
                    key = _action_key(lines[0])
                    repeated_suggestions[key] = repeated_suggestions.get(key, 0) + 1
                    print(f"[planner/pi] Candidate {idx + 1}/{n} repeated: {lines[0]!r}")
                    if repeated_suggestions[key] >= 4:
                        print("[planner/pi] Same action suggested four times; stopping proposal batch.")
                        break
                    continue

                key = _action_key(chosen_action)
                repeated_suggestions[key] = repeated_suggestions.get(key, 0) + 1
                actions.append(chosen_action)
                print(f"[planner/pi] Generated candidate {idx + 1}/{n}: {chosen_action!r}")
            except Exception as exc:
                print(f"[planner/pi] Candidate {idx + 1}/{n} failed: {exc}. Skipping slot.")

        return actions



class MockPlannerProvider(BasePlannerProvider):
    def propose_actions(
        self,
        state: str,
        goal: str,
        n: int = 3,
        explored_actions: Optional[list[str]] = None,
    ) -> list[str]:
        all_explored = set(explored_actions or [])
        available = [a for a in _MOCK_ACTION_POOL if a not in all_explored]
        # Fall back to full pool if all explored (avoids empty sample in long tests)
        if len(available) < n:
            available = _MOCK_ACTION_POOL
        return random.sample(available, min(n, len(available)))


# ── Executor Implementations ───────────────────────────────────────────────────

class AGYExecutorProvider(BaseExecutorProvider):
    def __init__(self, model: str = "gemini-3.8-flash-medium", timeout: int = 1800):
        self.model = model
        self.timeout = timeout

    def execute_action(
        self, action: str, goal: str, workspace_dir: str
    ) -> dict[str, Any]:
        abs_workspace = _safe_abspath(workspace_dir)
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
        print(f"[executor/agy] Executing action with agy in {abs_workspace}: '{action}'")
        print(f"{'='*60}\n")

        timeout_str = f"{max(1, self.timeout // 60)}m0s"
        cmd = [
            "agy",
            "--model", self.model,
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
                timeout=self.timeout,
            )
            return {
                "success": proc.returncode == 0,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "returncode": proc.returncode,
            }
        except Exception as exc:
            print(f"[executor/agy] agy execution failed: {exc}")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(exc),
                "returncode": -1,
            }


class PiExecutorProvider(BaseExecutorProvider):
    """Executor using the Pi agent harness (`pi --print`).

    Pi owns its own model backend and tool loop (read, write, edit, bash).
    mcts-agent passes only the action prompt; Pi decides how to implement it —
    reading workspace files first if needed, retrying, etc.  Pi exits 0 on
    success and writes files directly into workspace_dir.

    Model is forwarded as `--model <model>` when non-empty, so you can override
    Pi's configured default from the mcts-agent config without touching
    ~/.pi/agent/models.json.
    """

    def __init__(self, model: str = "", timeout: int = 1800):
        self.model = model
        self.timeout = timeout

    def execute_action(
        self, action: str, goal: str, workspace_dir: str
    ) -> dict[str, Any]:
        abs_workspace = _safe_abspath(workspace_dir)
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
        print(f"[executor/pi] Executing action with pi in {abs_workspace}: '{action}'")
        print(f"{'='*60}\n")

        cmd = ["pi", "--no-session", "--print", prompt]
        if self.model:
            cmd = ["pi", "--no-session", "--model", self.model, "--print", prompt]

        try:
            proc = subprocess.run(
                cmd,
                cwd=workspace_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return {
                "success": proc.returncode == 0,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "returncode": proc.returncode,
            }
        except Exception as exc:
            print(f"[executor/pi] pi execution failed: {exc}")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(exc),
                "returncode": -1,
            }


class MockExecutorProvider(BaseExecutorProvider):
    def execute_action(
        self, action: str, goal: str, workspace_dir: str
    ) -> dict[str, Any]:
        return {
            "success": True,
            "stdout": f"[mock] Successfully executed action: {action}",
            "stderr": "",
            "returncode": 0,
        }


# ── Factories ─────────────────────────────────────────────────────────────────

_KNOWN_PLANNER_HARNESSES = ("agy", "pi", "mock")
_KNOWN_EXECUTOR_HARNESSES = ("agy", "pi", "mock")


def get_planner_provider(config: AgentConfig) -> BasePlannerProvider:
    """Return the configured planner harness.

    mcts-agent never connects to models directly; all planning is delegated to
    an agent harness.  Raises ValueError for unknown harness names so
    misconfigurations are caught early.
    """
    provider = config.planner_provider.lower()
    match provider:
        case "agy":
            return AGYPlannerProvider(model=config.planner_model)
        case "pi":
            return PiPlannerProvider(model=config.planner_model)
        case "mock":
            return MockPlannerProvider()
        case _:
            raise ValueError(
                f"Unknown planner harness '{provider}'. "
                f"Valid options: {_KNOWN_PLANNER_HARNESSES}"
            )


def get_executor_provider(config: AgentConfig) -> BaseExecutorProvider:
    """Return the configured executor harness.

    mcts-agent never connects to models directly; all execution is delegated to
    an agent harness that has its own tool loop.  Raises ValueError for unknown
    harness names so misconfigurations are caught early.
    """
    provider = config.executor_provider.lower()
    match provider:
        case "agy":
            return AGYExecutorProvider(model=config.executor_model, timeout=config.executor_timeout)
        case "pi":
            return PiExecutorProvider(model=config.executor_model, timeout=config.executor_timeout)
        case "mock":
            return MockExecutorProvider()
        case _:
            raise ValueError(
                f"Unknown executor harness '{provider}'. "
                f"Valid options: {_KNOWN_EXECUTOR_HARNESSES}"
            )
