"""Tests for harness-only planner and executor providers."""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

os.environ["USE_MOCK_PRIMITIVES"] = "true"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent.config import AgentConfig, load_config
from agent.mcts import run_closed_loop_agent
from agent.providers import (
    AGYExecutorProvider,
    AGYPlannerProvider,
    MockExecutorProvider,
    MockPlannerProvider,
    PiExecutorProvider,
    PiPlannerProvider,
    _MOCK_ACTION_POOL,
    _action_key,
    get_executor_provider,
    get_planner_provider,
)


class TestConfigAndProviders(unittest.TestCase):
    def test_load_config_tree_defaults(self):
        config = load_config()
        self.assertEqual(config.expansion_width, 3)
        self.assertEqual(config.expansion_depth, 3)
        self.assertTrue(config.tree_reuse_enabled)
        self.assertAlmostEqual(config.reuse_score_threshold, 1.5)

    def test_cli_overrides_harnesses(self):
        config = load_config(cli_overrides={
            "planner_provider": "pi", "executor_provider": "agy",
            "planner_model": "qwen", "executor_model": "claude",
        })
        self.assertEqual(config.planner_provider, "pi")
        self.assertEqual(config.executor_provider, "agy")
        self.assertEqual(config.planner_model, "qwen")
        self.assertEqual(config.executor_model, "claude")

    def test_pi_empty_toml_models_preserve_pi_defaults(self):
        config_toml = '''
[providers]
planner = "pi"
executor = "pi"

[planner]
model = ""

[executor]
model = ""
'''
        with tempfile.NamedTemporaryFile("w", suffix=".toml") as config_file, patch.dict(
            os.environ, {"USE_MOCK_PRIMITIVES": "false"}, clear=False
        ):
            config_file.write(config_toml)
            config_file.flush()
            config = load_config(config_file.name)

        self.assertEqual(config.planner_model, "")
        self.assertEqual(config.executor_model, "")
        self.assertEqual(get_planner_provider(config).model, "")
        self.assertEqual(get_executor_provider(config).model, "")

    def test_agy_without_model_override_uses_gemini_defaults(self):
        config_toml = '''
[providers]
planner = "agy"
executor = "agy"
'''
        with tempfile.NamedTemporaryFile("w", suffix=".toml") as config_file, patch.dict(
            os.environ,
            {"USE_MOCK_PRIMITIVES": "false", "AGY_PROPOSAL_MODEL": "", "AGY_MODEL": ""},
            clear=False,
        ):
            config_file.write(config_toml)
            config_file.flush()
            config = load_config(config_file.name)

        self.assertEqual(config.planner_model, "gemini-3.6-flash-low")
        self.assertEqual(config.executor_model, "gemini-3.8-flash-medium")

    def test_factories_support_only_harnesses(self):
        self.assertIsInstance(get_planner_provider(AgentConfig(planner_provider="mock")), MockPlannerProvider)
        self.assertIsInstance(get_planner_provider(AgentConfig(planner_provider="agy")), AGYPlannerProvider)
        self.assertIsInstance(get_planner_provider(AgentConfig(planner_provider="pi")), PiPlannerProvider)
        self.assertIsInstance(get_executor_provider(AgentConfig(executor_provider="mock")), MockExecutorProvider)
        self.assertIsInstance(get_executor_provider(AgentConfig(executor_provider="agy")), AGYExecutorProvider)
        self.assertIsInstance(get_executor_provider(AgentConfig(executor_provider="pi")), PiExecutorProvider)

    def test_unknown_harness_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown planner harness"):
            get_planner_provider(AgentConfig(planner_provider="llama_cpp"))
        with self.assertRaisesRegex(ValueError, "Unknown executor harness"):
            get_executor_provider(AgentConfig(executor_provider="local_cmd"))

    def test_action_key_normalizes_equivalent_suggestions(self):
        self.assertEqual(_action_key("Create   the visualizer!"), _action_key("CREATE THE VISUALIZER"))

    def test_pi_planner_invokes_harness_with_optional_model(self):
        with patch.dict(os.environ, {"USE_MOCK_PRIMITIVES": "false"}), patch(
            "agent.providers.subprocess.run",
            return_value=MagicMock(returncode=0, stdout="Inspect the repository\n", stderr=""),
        ) as run:
            actions = PiPlannerProvider(model="qwen").propose_actions("State", "Goal", n=1)
        self.assertEqual(actions, ["Inspect the repository"])
        command = run.call_args.args[0]
        self.assertEqual(command[:2], ["pi", "--no-session"])
        model_index = command.index("--model")
        self.assertEqual(command[model_index + 1], "qwen")
        self.assertIn("multiple coordinated operations", command[-1])

    def test_pi_planner_omits_model_flag_without_override(self):
        with patch.dict(os.environ, {"USE_MOCK_PRIMITIVES": "false"}), patch(
            "agent.providers.subprocess.run",
            return_value=MagicMock(returncode=0, stdout="Inspect the repository\n", stderr=""),
        ) as run:
            PiPlannerProvider().propose_actions("State", "Goal", n=1)
        self.assertNotIn("--model", run.call_args.args[0])

    def test_pi_executor_invokes_harness_in_workspace(self):
        with tempfile.TemporaryDirectory() as workspace, patch(
            "agent.providers.subprocess.run",
            return_value=MagicMock(returncode=0, stdout="done", stderr=""),
        ) as run:
            result = PiExecutorProvider(model="qwen").execute_action("Edit one file", "Goal", workspace)
        self.assertTrue(result["success"])
        self.assertEqual(run.call_args.kwargs["cwd"], workspace)
        self.assertIn("Target Workspace Directory", run.call_args.args[0][-1])

    def test_mock_providers_remain_hermetic(self):
        config = AgentConfig(planner_provider="mock", executor_provider="mock")
        actions = get_planner_provider(config).propose_actions("Start", "Goal", n=3)
        self.assertEqual(len(actions), 3)
        self.assertTrue(get_executor_provider(config).execute_action(actions[0], "Goal", ".")["success"])

    def test_mock_planner_avoids_known_actions_when_possible(self):
        actions = MockPlannerProvider().propose_actions("state", "goal", n=1, explored_actions=_MOCK_ACTION_POOL[:-1])
        self.assertEqual(actions, [_MOCK_ACTION_POOL[-1]])

    def test_closed_loop_runs_with_mock_harnesses(self):
        with tempfile.TemporaryDirectory() as workspace:
            summary = run_closed_loop_agent(
                goal="Test goal", initial_state="Initial", max_steps=1,
                iterations_per_step=0, workspace_dir=workspace,
                config=AgentConfig(planner_provider="mock", executor_provider="mock", expansion_depth=1),
            )
        self.assertEqual(summary["total_steps_run"], 1)


if __name__ == "__main__":
    unittest.main()
