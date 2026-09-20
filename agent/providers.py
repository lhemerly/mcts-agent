"""
agent/providers.py — Pluggable Planner and Executor providers for mcts-agent.

Provides backend implementations for:
  - Planning (Action Proposal): AGY, llama.cpp / OpenAI HTTP API, Mock
  - Execution (Workspace changes): AGY, llama.cpp / Local Shell, Mock
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import textwrap
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Optional

from agent.config import AgentConfig

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


# ── Base Interfaces ────────────────────────────────────────────────────────────

class BasePlannerProvider(ABC):
    @abstractmethod
    def propose_actions(self, state: str, goal: str, n: int = 3) -> list[str]:
        """Propose n distinct candidate actions given the current state and goal."""
        pass


class BaseExecutorProvider(ABC):
    @abstractmethod
    def execute_action(
        self, action: str, goal: str, workspace_dir: str
    ) -> dict[str, Any]:
        """Execute a single action in the workspace and return execution results."""
        pass


# ── Planner Implementations ────────────────────────────────────────────────────

class AGYPlannerProvider(BasePlannerProvider):
    def __init__(self, model: str = "gemini-3.6-flash-low"):
        self.model = model

    def propose_actions(self, state: str, goal: str, n: int = 3) -> list[str]:
        if os.getenv("USE_MOCK_PRIMITIVES", "false").lower() in ("1", "true", "yes"):
            return random.sample(_MOCK_ACTION_POOL, min(n, len(_MOCK_ACTION_POOL)))

        actions: list[str] = []
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
                for candidate in lines:
                    if candidate and candidate not in actions:
                        chosen_action = candidate
                        break
                if not chosen_action:
                    chosen_action = lines[0]

                actions.append(chosen_action)
                print(f"[planner/agy] Generated candidate {idx + 1}/{n}: '{chosen_action[:60]}'")
            except Exception as exc:
                print(f"[planner/agy] Action proposal failed for candidate {idx + 1}: {exc}. Using fallback.")
                unused_mock = [a for a in _MOCK_ACTION_POOL if a not in actions]
                if unused_mock:
                    actions.append(random.choice(unused_mock))
                else:
                    actions.append(f"Strategic step {idx + 1}")

        return actions


_CREATIVE_STRATEGIES: list[str] = [
    "Propose a direct, practical next action to make immediate progress toward the goal.",
    "Think outside the box! Explore an unconventional, highly creative, or novel strategic angle.",
    "Be analytical! Focus on decomposing the problem, verifying assumptions, or mitigating risks.",
    "Explore a high-leverage diagnostic or alternative technical direction completely distinct from previous steps.",
]


class OpenAIHTTPPlannerProvider(BasePlannerProvider):
    """Planner using any OpenAI-compatible HTTP endpoint (llama.cpp server, vLLM, Ollama, OpenAI)."""

    def __init__(
        self,
        endpoint: str = "http://localhost:8080/v1/chat/completions",
        model: str = "local-model",
        api_key: str = "",
    ):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key

    def propose_actions(self, state: str, goal: str, n: int = 3) -> list[str]:
        actions: list[str] = []

        for idx in range(n):
            existing_actions_str = (
                "\n".join(f"- {act}" for act in actions)
                if actions
                else "(No actions proposed yet for this step)"
            )

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            chosen_action: str | None = None

            for attempt in range(4):
                strategy_hint = _CREATIVE_STRATEGIES[attempt % len(_CREATIVE_STRATEGIES)]
                temp = min(1.0, 0.7 + attempt * 0.1)
                presence_pen = min(1.0, 0.3 + attempt * 0.25)
                frequency_pen = min(1.0, 0.2 + attempt * 0.25)

                prompt = textwrap.dedent(f"""\
                    You are a creative planning assistant.
                    Overall Goal: {goal}
                    Current Reasoning State: {state}

                    Actions already proposed for this expansion:
                    {existing_actions_str}

                    Instructions:
                    - {strategy_hint}
                    - Do NOT repeat, rephrase, or overlap with any previously proposed actions.
                    - Output ONLY the single action sentence itself with no commentary, bullets, or numbers.
                """)

                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temp,
                    "presence_penalty": presence_pen,
                    "frequency_penalty": frequency_pen,
                    "max_tokens": 150,
                }
                req_data = json.dumps(payload).encode("utf-8")

                try:
                    req = urllib.request.Request(self.endpoint, data=req_data, headers=headers)
                    with urllib.request.urlopen(req, timeout=45) as resp:
                        resp_json = json.loads(resp.read().decode("utf-8"))
                        raw_content = resp_json["choices"][0]["message"]["content"].strip()

                    lines = [
                        line.lstrip("0123456789.-*#) ").strip().strip('"\'')
                        for line in raw_content.splitlines()
                        if line.lstrip("0123456789.-*#) ").strip()
                    ]

                    for cand in lines:
                        if cand and cand not in actions:
                            chosen_action = cand
                            break

                    if chosen_action:
                        break
                except Exception as exc:
                    print(f"[planner/http] Candidate {idx + 1} attempt {attempt + 1} failed: {exc}")

            if chosen_action:
                actions.append(chosen_action)
                print(f"[planner/http] Generated candidate {idx + 1}/{n}: '{chosen_action[:60]}'")
            else:
                print(f"[planner/http] Candidate {idx + 1}/{n} failed after retries. Using fallback.")
                unused_mock = [a for a in _MOCK_ACTION_POOL if a not in actions]
                fallback = random.choice(unused_mock) if unused_mock else f"Strategic step {idx + 1}"
                actions.append(fallback)

        return actions


class MockPlannerProvider(BasePlannerProvider):
    def propose_actions(self, state: str, goal: str, n: int = 3) -> list[str]:
        return random.sample(_MOCK_ACTION_POOL, min(n, len(_MOCK_ACTION_POOL)))


# ── Executor Implementations ───────────────────────────────────────────────────

class AGYExecutorProvider(BaseExecutorProvider):
    def __init__(self, model: str = "gemini-3.8-flash-medium", timeout: int = 1800):
        self.model = model
        self.timeout = timeout

    def execute_action(
        self, action: str, goal: str, workspace_dir: str
    ) -> dict[str, Any]:
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


class OpenAIHTTPExecutorProvider(BaseExecutorProvider):
    """Executor using an OpenAI-compatible HTTP endpoint to generate shell commands and run locally."""

    def __init__(
        self,
        endpoint: str = "http://localhost:8080/v1/chat/completions",
        model: str = "local-model",
        api_key: str = "",
        timeout: int = 600,
    ):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def execute_action(
        self, action: str, goal: str, workspace_dir: str
    ) -> dict[str, Any]:
        abs_workspace = os.path.abspath(workspace_dir)
        prompt = textwrap.dedent(f"""\
            You are an automated local workspace executor.
            Target Workspace: {abs_workspace}
            Goal: {goal}
            Action to execute: {action}

            Instructions:
            Write a single bash command block (wrapped in ```bash ... ```) to execute this action in the workspace.
            Only output bash commands, no markdown explanations.
        """)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        req_data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        print(f"\n[executor/http] Requesting shell commands from {self.endpoint} for: '{action}'")

        try:
            req = urllib.request.Request(self.endpoint, data=req_data, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
                raw_text = resp_json["choices"][0]["message"]["content"]

            if "```bash" in raw_text:
                cmd_str = raw_text.split("```bash")[1].split("```")[0].strip()
            elif "```" in raw_text:
                cmd_str = raw_text.split("```")[1].split("```")[0].strip()
            else:
                cmd_str = raw_text.strip()

            print(f"[executor/http] Running shell command:\n{cmd_str}\n")
            proc = subprocess.run(
                ["bash", "-c", cmd_str],
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
            print(f"[executor/http] Local shell execution failed: {exc}")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(exc),
                "returncode": -1,
            }


class LocalCmdExecutorProvider(BaseExecutorProvider):
    """Executes action string directly as a local shell command."""

    def __init__(self, timeout: int = 600):
        self.timeout = timeout

    def execute_action(
        self, action: str, goal: str, workspace_dir: str
    ) -> dict[str, Any]:
        print(f"[executor/cmd] Executing shell command: '{action}'")
        try:
            proc = subprocess.run(
                action,
                shell=True,
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

def get_planner_provider(config: AgentConfig) -> BasePlannerProvider:
    provider = config.planner_provider.lower()
    if provider == "agy":
        return AGYPlannerProvider(model=config.planner_model)
    elif provider in ("llama_cpp", "llama", "openai", "http"):
        return OpenAIHTTPPlannerProvider(
            endpoint=config.planner_endpoint,
            model=config.planner_model,
            api_key=config.planner_api_key,
        )
    elif provider == "mock":
        return MockPlannerProvider()
    else:
        print(f"[providers] Unknown planner provider '{provider}'. Falling back to AGY.")
        return AGYPlannerProvider(model=config.planner_model)


def get_executor_provider(config: AgentConfig) -> BaseExecutorProvider:
    provider = config.executor_provider.lower()
    if provider == "agy":
        return AGYExecutorProvider(model=config.executor_model, timeout=config.executor_timeout)
    elif provider in ("llama_cpp", "llama", "openai", "http"):
        return OpenAIHTTPExecutorProvider(
            endpoint=config.executor_endpoint,
            model=config.executor_model,
            api_key=config.executor_api_key,
            timeout=config.executor_timeout,
        )
    elif provider in ("local_cmd", "cmd", "shell"):
        return LocalCmdExecutorProvider(timeout=config.executor_timeout)
    elif provider == "mock":
        return MockExecutorProvider()
    else:
        print(f"[providers] Unknown executor provider '{provider}'. Falling back to AGY.")
        return AGYExecutorProvider(model=config.executor_model, timeout=config.executor_timeout)
