"""
agent/providers.py — Pluggable Planner and Executor providers for mcts-agent.

Architecture: harness-of-harnesses
  mcts-agent never connects to models directly.  Every provider delegates to an
  *agent harness* (AGY, Pi, Codex, …) which owns its own tool loop, file
  read/write capabilities, and model backend.  mcts-agent only cares about the
  result: did the harness succeed and what changed in the workspace?

Planner harnesses  (propose candidate action strings, no workspace access):
  - AGYPlannerProvider      — Antigravity CLI  (`agy --print`)
  - PiPlannerProvider       — Pi coding agent  (`pi --print`)
  - OpenCodePlannerProvider — OpenCode CLI     (`opencode run --standalone`)
  - MockPlannerProvider     — Deterministic mock for unit tests

Executor harnesses  (carry out a single action in the workspace):
  - AGYExecutorProvider      — Antigravity CLI  (`agy --mode accept-edits --print`)
  - PiExecutorProvider       — Pi coding agent  (`pi --print`)
  - OpenCodeExecutorProvider — OpenCode CLI     (`opencode run --standalone --auto`)
  - CodexExecutorProvider    — Codex CLI        (`codex exec`)
  - MockExecutorProvider     — No-op mock for unit tests
"""

from __future__ import annotations

import os
import random
import re
import subprocess
import sys
import textwrap
import time
from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from typing import Any, Callable, Optional

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

_ACTION_SCOPE_RULES = (
    "Propose ONE distinct, executable next action or implementation plan. "
    "It may include multiple coordinated operations or file changes when they "
    "are needed to make meaningful progress. Keep it within a clear, bounded "
    "scope that an executor harness can complete in one run."
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
    def __init__(self, model: str = "gemini-3.6-flash-low", timeout: int = 0):
        self.model = model
        self.timeout = timeout

    def propose_actions(
        self,
        state: str,
        goal: str,
        n: int = 3,
        explored_actions: Optional[list[str]] = None,
    ) -> list[str]:
        explored_str = (
            "\n".join(f"- {a}" for a in (explored_actions or []))
            or "(None yet)"
        )

        actions: list[str] = []
        repeated_suggestions: dict[str, int] = {}
        timeout_val = self.timeout if self.timeout and self.timeout > 0 else None

        for idx in range(n):
            existing_actions_str = (
                "\n".join(f"- {act}" for act in actions)
                if actions
                else "(No actions proposed yet for this expansion)"
            )

            prompt = textwrap.dedent(f"""\
                You are a planning assistant in a Monte Carlo Tree Search (MCTS) reasoning system.
                The question is always: given the current state, what is the next course of action to achieve the goal?

                Goal:
                {goal}

                Current state / context:
                {state}

                Actions ALREADY EXPLORED anywhere in the search tree (DO NOT reproduce these):
                {explored_str}

                Actions already proposed in this current expansion batch:
                {existing_actions_str}

                Instructions:
                - {_ACTION_SCOPE_RULES}
                - {random.choice(_CREATIVE_STRATEGIES)}
                - Do NOT duplicate, overlap, or rephrase any action listed above (explored or batch).
                - Output ONLY the single action sentence, with no commentary, numbering, bullets, or preamble.
            """)

            max_retries = 3
            chosen_action = None
            for attempt in range(1, max_retries + 1):
                try:
                    result = subprocess.run(
                        ["agy", "--model", self.model, "--print", prompt],
                        capture_output=True,
                        text=True,
                        timeout=timeout_val,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(f"agy exited with code {result.returncode}: {result.stderr.strip() or result.stdout.strip()}")
                    raw = result.stdout.strip()
                    lines = [
                        line.lstrip("0123456789.-*#) ").strip().strip('"\'')
                        for line in raw.splitlines()
                        if line.lstrip("0123456789.-*#) ").strip()
                    ]
                    if not lines:
                        raise ValueError(f"agy returned no parseable action. stdout: {raw!r}")

                    all_explored = {_action_key(action) for action in (explored_actions or []) + actions}
                    for candidate in lines:
                        if candidate and _action_key(candidate) not in all_explored:
                            chosen_action = candidate
                            break
                    if chosen_action:
                        key = _action_key(chosen_action)
                        repeated_suggestions[key] = repeated_suggestions.get(key, 0) + 1
                        actions.append(chosen_action)
                        sys.stderr.write(f"[planner/agy] Generated candidate {idx + 1}/{n}: {chosen_action!r}\n")
                        sys.stderr.flush()
                        break
                    else:
                        key = _action_key(lines[0])
                        repeated_suggestions[key] = repeated_suggestions.get(key, 0) + 1
                        sys.stderr.write(f"[planner/agy] Candidate {idx + 1}/{n} repeated: {lines[0]!r}\n")
                        sys.stderr.flush()
                        if repeated_suggestions[key] >= 4:
                            sys.stderr.write("[planner/agy] Same action suggested four times; stopping proposal batch.\n")
                            sys.stderr.flush()
                            break
                except Exception as exc:
                    sys.stderr.write(f"[planner/agy] Candidate {idx + 1}/{n} attempt {attempt}/{max_retries} failed: {exc}\n")
                    sys.stderr.flush()
                    if attempt < max_retries:
                        time.sleep(3)

        if not actions:
            fallback = "Inspect current state and workspace to determine the immediate next action toward the goal."
            sys.stderr.write(f"[planner/agy] Warning: no candidate generated, using fallback action: {fallback!r}\n")
            sys.stderr.flush()
            actions.append(fallback)

        return actions


_CREATIVE_STRATEGIES: list[str] = [
    "Focus on the most direct and decisive next action that moves the current state toward the goal.",
    "Focus on executing a concrete action that fulfills a key requirement or prerequisite of the goal.",
    "Focus on inspecting, testing, or diagnosing the current state to uncover necessary information.",
    "Focus on verifying the results of prior actions and confirming progress toward the goal.",
    "Focus on the simplest, most reliable action that makes meaningful progress from the current state.",
]


class PiPlannerProvider(BasePlannerProvider):
    """Planner using the Pi agent harness (`pi --print`).

    Pi owns its own model backend configuration (Ollama, OpenAI, Anthropic, etc.)
    via ~/.pi/agent/models.json.  mcts-agent passes only the planning prompt and
    reads back the proposed action string.  The model flag is forwarded to Pi as
    `--model <model>` so you can override the default from the mcts-agent config
    without changing Pi's global settings.
    """

    def __init__(self, model: str = "", timeout: int = 0):
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
        timeout_val = self.timeout if self.timeout and self.timeout > 0 else None

        for idx in range(n):
            existing_actions_str = (
                "\n".join(f"- {act}" for act in actions)
                if actions
                else "(No actions proposed yet for this expansion)"
            )

            prompt = textwrap.dedent(f"""\
                You are a planning assistant in a Monte Carlo Tree Search (MCTS) reasoning system.
                The question is always: given the current state, what is the next course of action to achieve the goal?

                Goal:
                {goal}

                Current state / context:
                {state}

                Actions ALREADY EXPLORED anywhere in the search tree (DO NOT reproduce these):
                {explored_str}

                Actions already proposed in this current expansion batch:
                {existing_actions_str}

                Instructions:
                - {_ACTION_SCOPE_RULES}
                - {random.choice(_CREATIVE_STRATEGIES)}
                - Do NOT duplicate, overlap, or rephrase any action listed above (explored or batch).
                - Output ONLY the single action sentence, with no commentary, numbering, bullets, or preamble.
            """)

            cmd = ["pi", "--no-session", "--no-tools", "--print", prompt]
            if self.model:
                cmd = ["pi", "--no-session", "--no-tools", "--model", self.model, "--print", prompt]

            max_retries = 3
            chosen_action = None
            for attempt in range(1, max_retries + 1):
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=timeout_val,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(f"pi exited with code {result.returncode}: {result.stderr.strip() or result.stdout.strip()}")
                    raw = result.stdout.strip()
                    lines = [
                        line.lstrip("0123456789.-*#) ").strip().strip('"\'')
                        for line in raw.splitlines()
                        if line.lstrip("0123456789.-*#) ").strip()
                    ]
                    if not lines:
                        raise ValueError(f"pi returned no parseable action. stdout: {raw!r}")

                    all_explored = {_action_key(a) for a in (explored_actions or []) + actions}
                    for candidate in lines:
                        if candidate and _action_key(candidate) not in all_explored:
                            chosen_action = candidate
                            break
                    if chosen_action:
                        key = _action_key(chosen_action)
                        repeated_suggestions[key] = repeated_suggestions.get(key, 0) + 1
                        actions.append(chosen_action)
                        sys.stderr.write(f"[planner/pi] Generated candidate {idx + 1}/{n}: {chosen_action!r}\n")
                        sys.stderr.flush()
                        break
                    else:
                        key = _action_key(lines[0])
                        repeated_suggestions[key] = repeated_suggestions.get(key, 0) + 1
                        sys.stderr.write(f"[planner/pi] Candidate {idx + 1}/{n} repeated: {lines[0]!r}\n")
                        sys.stderr.flush()
                        if repeated_suggestions[key] >= 4:
                            sys.stderr.write("[planner/pi] Same action suggested four times; stopping proposal batch.\n")
                            sys.stderr.flush()
                            break
                except Exception as exc:
                    sys.stderr.write(f"[planner/pi] Candidate {idx + 1}/{n} attempt {attempt}/{max_retries} failed: {exc}\n")
                    sys.stderr.flush()
                    if attempt < max_retries:
                        time.sleep(3)

        if not actions:
            fallback = "Inspect current state and workspace to determine the immediate next action toward the goal."
            sys.stderr.write(f"[planner/pi] Warning: no candidate generated, using fallback action: {fallback!r}\n")
            sys.stderr.flush()
            actions.append(fallback)

        return actions


class OpenCodePlannerProvider(BasePlannerProvider):
    """Planner using OpenCode (`opencode run --standalone`)."""

    def __init__(self, model: str = "", timeout: int = 0):
        self.model = model
        self.timeout = timeout

    def propose_actions(
        self,
        state: str,
        goal: str,
        n: int = 3,
        explored_actions: Optional[list[str]] = None,
    ) -> list[str]:
        explored_str = (
            "\n".join(f"- {a}" for a in (explored_actions or []))
            or "(None yet)"
        )

        actions: list[str] = []
        timeout_val = self.timeout if self.timeout and self.timeout > 0 else None
        count_str = f"{n} distinct, novel next candidate actions" if n > 1 else "ONE distinct, novel next action"
        num_instruction = f"Output each distinct candidate action on its own line (numbered 1 to {n})." if n > 1 else "Output ONLY the single action sentence, with no commentary, numbering, bullets, or preamble."

        prompt = textwrap.dedent(f"""\
            You are a planning assistant in a Monte Carlo Tree Search (MCTS) reasoning system.
            The question is always: given the current state, what is the next course of action to achieve the goal?

            Goal:
            {goal}

            Current state / context:
            {state}

            Actions ALREADY EXPLORED anywhere in the search tree (DO NOT reproduce these):
            {explored_str}

            Instructions:
            - You are solely a planning assistant. Do NOT invoke any tools, do NOT run commands, and do NOT inspect or modify files. Answer purely in text directly.
            - {_ACTION_SCOPE_RULES}
            - {random.choice(_CREATIVE_STRATEGIES)}
            - Propose a focused immediate next action to take from the current state toward the goal, rather than bundling the entire end-to-end task into one action.
            - Do NOT duplicate, overlap, or rephrase any action listed above.
            - {num_instruction}
        """)

        cmd = ["opencode", "run", "--standalone", "--auto"]
        if self.model:
            cmd.extend(["-m", self.model])
        cmd.append(prompt)

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                proc = subprocess.run(
                    cmd,
                    cwd="/tmp",
                    capture_output=True,
                    text=True,
                    timeout=timeout_val,
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"opencode exited with code {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}")
                raw = proc.stdout.strip()
                lines = [
                    line.lstrip("0123456789.-*#)> ").strip().strip('"\'')
                    for line in raw.splitlines()
                    if line.strip() and not line.strip().startswith(">") and not line.strip().startswith("$") and len(line.strip()) > 8
                ]
                all_explored = {_action_key(action) for action in (explored_actions or [])}
                for candidate in lines:
                    ckey = _action_key(candidate)
                    if candidate and ckey not in all_explored and ckey not in {_action_key(a) for a in actions}:
                        actions.append(candidate)
                        sys.stderr.write(f"[planner/opencode] Candidate {len(actions)}/{n}: {candidate!r}\n")
                        sys.stderr.flush()
                        if len(actions) >= n:
                            break
                if actions:
                    break
            except Exception as exc:
                sys.stderr.write(f"[planner/opencode] Attempt {attempt}/{max_retries} failed: {exc}\n")
                sys.stderr.flush()
                if attempt < max_retries:
                    time.sleep(2)

        if not actions:
            fallback = "Inspect current state and workspace to determine the immediate next action toward the goal."
            sys.stderr.write(f"[planner/opencode] Warning: no candidate generated, using fallback action: {fallback!r}\n")
            sys.stderr.flush()
            actions.append(fallback)

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
        workspace_dir = _safe_abspath(workspace_dir)
        abs_workspace = workspace_dir
        prompt = textwrap.dedent(f"""\
            You are a focused execution tool in a multi-step planning system.

            Target Workspace Directory:
            {abs_workspace}

            Immediate Task to execute:
            >>> {action.strip()} <<<

            All files created or modified MUST be written inside {abs_workspace}.
            Execute ONLY this specific task. Do not attempt to solve future steps, do not write full end-to-end solutions unless explicitly asked by this task, and do not perform work outside this immediate task.
            Always execute commands synchronously to full completion in the foreground; do not leave background tasks running.
        """)

        print(f"\n{'='*60}")
        print(f"[executor/agy] Executing action with agy in {abs_workspace}: '{action}'")
        print(f"{'='*60}\n")

        timeout_val = self.timeout if self.timeout and self.timeout > 0 else None
        timeout_str = f"{max(1, (self.timeout // 60) if self.timeout and self.timeout > 0 else 600)}m0s"
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
                timeout=timeout_val,
            )
            return {
                "success": proc.returncode == 0,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "returncode": proc.returncode,
            }
        except Exception as exc:
            sys.stderr.write(f"[executor/agy] agy execution failed: {exc}\n")
            sys.stderr.flush()
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

    def __init__(self, model: str = "", timeout: int = 0):
        self.model = model
        self.timeout = timeout

    def execute_action(
        self, action: str, goal: str, workspace_dir: str
    ) -> dict[str, Any]:
        abs_workspace = _safe_abspath(workspace_dir)
        prompt = textwrap.dedent(f"""\
            You are a focused execution tool in a multi-step planning system.

            Target Workspace Directory:
            {abs_workspace}

            Immediate Task to execute:
            >>> {action.strip()} <<<

            All files created or modified MUST be written inside {abs_workspace}.
            Execute ONLY this specific task. Do not attempt to solve future steps, do not write full end-to-end solutions unless explicitly asked by this task, and do not perform work outside this immediate task.
            Always execute commands synchronously to full completion in the foreground; do not leave background tasks running.
        """)

        print(f"\n{'='*60}")
        print(f"[executor/pi] Executing action with pi in {abs_workspace}: '{action}'")
        print(f"{'='*60}\n")

        cmd = ["pi", "--no-session", "--print", prompt]
        if self.model:
            cmd = ["pi", "--no-session", "--model", self.model, "--print", prompt]

        timeout_val = self.timeout if self.timeout and self.timeout > 0 else None
        try:
            proc = subprocess.run(
                cmd,
                cwd=workspace_dir,
                capture_output=True,
                text=True,
                timeout=timeout_val,
            )
            return {
                "success": proc.returncode == 0,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "returncode": proc.returncode,
            }
        except Exception as exc:
            sys.stderr.write(f"[executor/pi] pi execution failed: {exc}\n")
            sys.stderr.flush()
            return {"success": False, "stdout": "", "stderr": str(exc), "returncode": -1}


class CodexExecutorProvider(BaseExecutorProvider):
    """Executor using the Codex CLI's structured ``codex exec`` interface.

    Codex owns its model, tools, sandbox, and approvals. The connector supplies
    the task and workspace boundary; it never disables Codex permissions.
    """

    def __init__(self, model: str = "", timeout: int = 0, command: str = "codex"):
        self.model = model
        self.timeout = timeout
        self.command = command

    def execute_action(self, action: str, goal: str, workspace_dir: str) -> dict[str, Any]:
        workspace = _safe_abspath(workspace_dir)
        prompt = textwrap.dedent(f"""\
            You are a focused execution tool in a multi-step planning system.

            Target Workspace Directory:
            {workspace}

            Bounded task to execute:
            >>> {action.strip()} <<<

            Work only in the supplied workspace. Complete ONLY this specific task and report actual
            observations, conclusions, artifacts, and workspace changes separately.
            Do not claim an experiment or test ran unless it actually ran.
            Do not perform work outside this immediate task.
        """)
        cmd = [self.command, "exec", "--json", "--sandbox", "workspace-write", "--cd", workspace,
               "--skip-git-repo-check"]
        if self.model:
            cmd += ["--model", self.model]
        cmd.append(prompt)
        timeout_val = self.timeout if self.timeout and self.timeout > 0 else None
        try:
            proc = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True, timeout=timeout_val)
            return {"success": proc.returncode == 0, "stdout": proc.stdout.strip(),
                    "stderr": proc.stderr.strip(), "returncode": proc.returncode}
        except subprocess.TimeoutExpired as exc:
            return {"success": False, "stdout": (exc.stdout or ""), "stderr": "Codex execution timed out",
                    "returncode": -1, "cancelled": True}
        except Exception as exc:
            sys.stderr.write(f"[executor/codex] codex execution failed: {exc}\n")
            sys.stderr.flush()
            return {"success": False, "stdout": "", "stderr": str(exc), "returncode": -1}


class OpenCodeExecutorProvider(BaseExecutorProvider):
    """Executor using OpenCode (`opencode run --standalone --auto`)."""

    def __init__(self, model: str = "", timeout: int = 0):
        self.model = model
        self.timeout = timeout

    def execute_action(
        self, action: str, goal: str, workspace_dir: str
    ) -> dict[str, Any]:
        abs_workspace = _safe_abspath(workspace_dir)
        prompt = textwrap.dedent(f"""\
            You are a focused execution tool in a multi-step planning system.

            Target Workspace Directory:
            {abs_workspace}

            Immediate Task to execute:
            >>> {action.strip()} <<<

            All files created or modified MUST be written inside {abs_workspace}.
            Execute ONLY this specific task. Do not attempt to solve future steps, do not write full end-to-end solutions unless explicitly asked by this task, and do not perform work outside this immediate task.
            Always execute commands synchronously to full completion in the foreground; do not leave background tasks running.
        """)

        print(f"\n{'='*60}")
        print(f"[executor/opencode] Executing action with opencode in {abs_workspace}: '{action}'")
        print(f"{'='*60}\n")

        cmd = ["opencode", "run", "--standalone", "--auto"]
        if self.model:
            cmd.extend(["-m", self.model])
        cmd.append(prompt)

        timeout_val = self.timeout if self.timeout and self.timeout > 0 else None
        try:
            proc = subprocess.run(
                cmd,
                cwd=workspace_dir,
                capture_output=True,
                text=True,
                timeout=timeout_val,
            )
            return {
                "success": proc.returncode == 0,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "success": False,
                "stdout": (exc.stdout or ""),
                "stderr": "OpenCode execution timed out",
                "returncode": -1,
                "cancelled": True,
            }
        except Exception as exc:
            sys.stderr.write(f"[executor/opencode] opencode execution failed: {exc}\n")
            sys.stderr.flush()
            return {"success": False, "stdout": "", "stderr": str(exc), "returncode": -1}


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


# ── Harness registry ──────────────────────────────────────────────────────────

PlannerFactory = Callable[[AgentConfig], BasePlannerProvider]
ExecutorFactory = Callable[[AgentConfig], BaseExecutorProvider]
_HARNESS_REGISTRY: dict[str, tuple[PlannerFactory | None, ExecutorFactory | None]] = {}
_LOADED_HARNESS_ENTRY_POINTS = False


def register_harness(
    name: str,
    *,
    planner_factory: PlannerFactory | None = None,
    executor_factory: ExecutorFactory | None = None,
    replace: bool = False,
) -> None:
    """Register a harness connector.

    External packages can register at import time or publish an entry point in
    the ``mcts_agent.harnesses`` group whose callable performs this registration.
    """
    key = name.strip().lower()
    if not key or (planner_factory is None and executor_factory is None):
        raise ValueError("A harness name and at least one factory are required")
    if key in _HARNESS_REGISTRY and not replace:
        raise ValueError(f"Harness '{key}' is already registered")
    _HARNESS_REGISTRY[key] = (planner_factory, executor_factory)


def _load_harness_entry_points() -> None:
    global _LOADED_HARNESS_ENTRY_POINTS
    if _LOADED_HARNESS_ENTRY_POINTS:
        return
    _LOADED_HARNESS_ENTRY_POINTS = True
    try:
        discovered = entry_points(group="mcts_agent.harnesses")
    except TypeError:  # Python 3.10 compatibility
        discovered = entry_points().get("mcts_agent.harnesses", [])
    for entry_point in discovered:
        entry_point.load()()


def available_harnesses() -> tuple[str, ...]:
    _load_harness_entry_points()
    return tuple(sorted(_HARNESS_REGISTRY))


def get_planner_provider(config: AgentConfig) -> BasePlannerProvider:
    """Return the configured planner harness.

    mcts-agent never connects to models directly; all planning is delegated to
    an agent harness.  Raises ValueError for unknown harness names so
    misconfigurations are caught early.
    """
    _load_harness_entry_points()
    provider = config.planner_provider.lower()
    factory = _HARNESS_REGISTRY.get(provider, (None, None))[0]
    if factory is None:
        raise ValueError(f"Unknown planner harness '{provider}'. Available: {available_harnesses()}")
    return factory(config)


def get_executor_provider(config: AgentConfig) -> BaseExecutorProvider:
    """Return the configured executor harness.

    mcts-agent never connects to models directly; all execution is delegated to
    an agent harness that has its own tool loop.  Raises ValueError for unknown
    harness names so misconfigurations are caught early.
    """
    _load_harness_entry_points()
    provider = config.executor_provider.lower()
    factory = _HARNESS_REGISTRY.get(provider, (None, None))[1]
    if factory is None:
        raise ValueError(f"Unknown executor harness '{provider}'. Available: {available_harnesses()}")
    return factory(config)


register_harness(
    "agy",
    planner_factory=lambda cfg: AGYPlannerProvider(
        cfg.planner_model or "gemini-3.6-flash-low",
        cfg.planner_timeout,
    ),
    executor_factory=lambda cfg: AGYExecutorProvider(
        cfg.executor_model or "gemini-3.8-flash-medium",
        cfg.executor_timeout,
    ),
)
register_harness(
    "pi",
    planner_factory=lambda cfg: PiPlannerProvider(
        cfg.planner_model,
        cfg.planner_timeout,
    ),
    executor_factory=lambda cfg: PiExecutorProvider(
        cfg.executor_model,
        cfg.executor_timeout,
    ),
)
register_harness(
    "codex",
    executor_factory=lambda cfg: CodexExecutorProvider(
        cfg.executor_model,
        cfg.executor_timeout,
    ),
)
register_harness(
    "opencode",
    planner_factory=lambda cfg: OpenCodePlannerProvider(
        cfg.planner_model,
        cfg.planner_timeout,
    ),
    executor_factory=lambda cfg: OpenCodeExecutorProvider(
        cfg.executor_model,
        cfg.executor_timeout,
    ),
)
register_harness(
    "mock",
    planner_factory=lambda cfg: MockPlannerProvider(),
    executor_factory=lambda cfg: MockExecutorProvider(),
)
