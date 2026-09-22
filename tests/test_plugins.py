"""Tests for external harness and System One plugin registration."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent.config import AgentConfig
from agent.mcts import run_mcts
from agent.node import Node
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


class _OutOfRangeSystemOne(_PluginSystemOne):
    def score(self, state, instructions, criteria):
        return 11.0


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
            self.assertEqual(evaluate_state("goal", "state", mock=False), 7.0)

    def test_system_one_scores_are_bounded_to_the_public_scale(self):
        register_system_one_provider("test-out-of-range-system-one", _OutOfRangeSystemOne)
        with patch.dict(os.environ, {
            "USE_MOCK_PRIMITIVES": "false",
            "MCTS_SYSTEM_ONE_PROVIDER": "test-out-of-range-system-one",
        }, clear=False):
            self.assertEqual(evaluate_state("goal", "state", mock=False), 10.0)

    def test_run_mcts_uses_its_explicit_system_one_configuration(self):
        register_system_one_provider("test-run-config-system-one", _PluginSystemOne)
        config = AgentConfig(
            planner_provider="mock",
            executor_provider="mock",
            system_one_provider="test-run-config-system-one",
            expansion_width=1,
            expansion_depth=1,
        )
        with patch.dict(os.environ, {"USE_MOCK_PRIMITIVES": "false"}, clear=False):
            best, _ = run_mcts(Node(state="initial"), "goal", iterations=0, config=config)
        self.assertEqual(best.average_value, 7.0)

    def test_harness_entry_point_is_loaded_lazily(self):
        import agent.providers as providers

        def register_plugin():
            register_harness(
                "entry-point-harness",
                planner_factory=lambda config: _PluginPlanner(),
                executor_factory=lambda config: _PluginExecutor(),
            )

        entry_point = MagicMock()
        entry_point.load.return_value = register_plugin
        with patch.object(providers, "_LOADED_HARNESS_ENTRY_POINTS", False), patch(
            "agent.providers.entry_points", return_value=[entry_point]
        ):
            provider = get_planner_provider(AgentConfig(planner_provider="entry-point-harness"))
        self.assertIsInstance(provider, _PluginPlanner)
        entry_point.load.assert_called_once_with()

    def test_system_one_entry_point_is_loaded_lazily(self):
        import agent.system_one as system_one

        def register_plugin():
            register_system_one_provider("entry-point-system-one", _PluginSystemOne)

        entry_point = MagicMock()
        entry_point.load.return_value = register_plugin
        with patch.object(system_one, "_LOADED_ENTRY_POINTS", False), patch(
            "agent.system_one.entry_points", return_value=[entry_point]
        ):
            provider = system_one.get_system_one_provider("entry-point-system-one")
        self.assertIsInstance(provider, _PluginSystemOne)
        entry_point.load.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
