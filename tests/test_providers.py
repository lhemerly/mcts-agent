"""Tests for AgentConfig resolution and pluggable Planner/Executor providers."""

import os
import unittest

os.environ["USE_MOCK_PRIMITIVES"] = "true"

import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent.config import AgentConfig, load_config
from agent.providers import (
    AGYExecutorProvider,
    AGYPlannerProvider,
    LocalCmdExecutorProvider,
    MockExecutorProvider,
    MockPlannerProvider,
    OpenAIHTTPExecutorProvider,
    OpenAIHTTPPlannerProvider,
    get_executor_provider,
    get_planner_provider,
)
from agent.mcts import run_closed_loop_agent


class TestConfigAndProviders(unittest.TestCase):
    def test_load_config_defaults(self):
        config = load_config()
        self.assertIsNotNone(config.planner_provider)
        self.assertIsNotNone(config.executor_provider)

    def test_cli_overrides(self):
        config = load_config(cli_overrides={
            "planner_provider": "llama_cpp",
            "executor_provider": "local_cmd",
            "planner_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
        })
        self.assertEqual(config.planner_provider, "llama_cpp")
        self.assertEqual(config.executor_provider, "local_cmd")
        self.assertEqual(config.planner_endpoint, "http://127.0.0.1:8080/v1/chat/completions")

    def test_planner_factory(self):
        config_mock = AgentConfig(planner_provider="mock")
        planner_mock = get_planner_provider(config_mock)
        self.assertIsInstance(planner_mock, MockPlannerProvider)

        config_http = AgentConfig(planner_provider="llama_cpp", planner_endpoint="http://localhost:8080/v1/chat/completions")
        planner_http = get_planner_provider(config_http)
        self.assertIsInstance(planner_http, OpenAIHTTPPlannerProvider)

        config_agy = AgentConfig(planner_provider="agy")
        planner_agy = get_planner_provider(config_agy)
        self.assertIsInstance(planner_agy, AGYPlannerProvider)

    def test_executor_factory(self):
        config_mock = AgentConfig(executor_provider="mock")
        executor_mock = get_executor_provider(config_mock)
        self.assertIsInstance(executor_mock, MockExecutorProvider)

        config_cmd = AgentConfig(executor_provider="local_cmd")
        executor_cmd = get_executor_provider(config_cmd)
        self.assertIsInstance(executor_cmd, LocalCmdExecutorProvider)

        config_http = AgentConfig(executor_provider="llama_cpp")
        executor_http = get_executor_provider(config_http)
        self.assertIsInstance(executor_http, OpenAIHTTPExecutorProvider)

        config_agy = AgentConfig(executor_provider="agy")
        executor_agy = get_executor_provider(config_agy)
        self.assertIsInstance(executor_agy, AGYExecutorProvider)

    def test_propose_and_execute_mock(self):
        config = AgentConfig(planner_provider="mock", executor_provider="mock")
        planner = get_planner_provider(config)
        actions = planner.propose_actions(state="Start", goal="Test goal", n=3)
        self.assertEqual(len(actions), 3)

        executor = get_executor_provider(config)
        res = executor.execute_action(action=actions[0], goal="Test goal", workspace_dir=".")
        self.assertTrue(res["success"])

    def test_run_closed_loop_agent_with_config(self):
        config = AgentConfig(planner_provider="mock", executor_provider="mock")
        summary = run_closed_loop_agent(
            goal="Test goal with custom config",
            initial_state="Initial context",
            max_steps=2,
            iterations_per_step=3,
            config=config,
        )
        self.assertIn("steps", summary)
        self.assertEqual(summary["goal"], "Test goal with custom config")


if __name__ == "__main__":
    unittest.main()
