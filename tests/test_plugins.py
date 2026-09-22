"""Tests for external harness and System One plugin registration."""

import os
import sys
import unittest
from unittest.mock import patch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent.config import AgentConfig
from agent.primitives import evaluate_state
from agent.providers import (
    BaseExecutorProvider,
    BasePlannerProvider,
    get_executor_provider,
    get_planner_provider,
    register_harness,
)
from agent.system_one import (
    BaseSystemOneProvider,
    ChoiceResult,
    register_system_one_provider,
)


class _PluginPlanner(BasePlannerProvider):
    def propose_actions(self, state, goal, n=3, explored_actions=None):
        return ["Plugin-generated plan"]


class _PluginExecutor(BaseExecutorProvider):
    def execute_action(self, action, goal, workspace_dir):
        return {"success": True, "stdout": "plugin", "stderr": "", "returncode": 0}


class _PluginSystemOne(BaseSystemOneProvider):
    def choose(self, state, instructions, criteria):
        first = next(iter(criteria))
        return ChoiceResult({key: float(key == first) for key in criteria}, first)

    def score(self, state, instructions, criteria):
        return 7.0

    def noul(self, state, instructions):
        return 0.9


class TestPluginRegistries(unittest.TestCase):
    def test_external_harness_can_be_registered_without_factory_changes(self):
        register_harness(
            "test-plugin-harness",
            planner_factory=lambda config: _PluginPlanner(),
            executor_factory=lambda config: _PluginExecutor(),
        )
        config = AgentConfig(
            planner_provider="test-plugin-harness",
            executor_provider="test-plugin-harness",
        )
        self.assertIsInstance(get_planner_provider(config), _PluginPlanner)
        self.assertIsInstance(get_executor_provider(config), _PluginExecutor)

    def test_external_system_one_provider_is_selected_from_environment(self):
        register_system_one_provider("test-plugin-system-one", _PluginSystemOne)
        with patch.dict(os.environ, {
            "USE_MOCK_PRIMITIVES": "false",
            "MCTS_SYSTEM_ONE_PROVIDER": "test-plugin-system-one",
        }, clear=False):
            self.assertEqual(evaluate_state("goal", "state", mock=False), 8.0)


if __name__ == "__main__":
    unittest.main()
