"""Tests for MCTS Agent CLI interface (Typer & Rich)."""
import json
import io
import os
import re
import sys
import unittest
from typer.testing import CliRunner

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent.cli import app, _render_summary
from rich.console import Console
from main import app as main_app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from text."""
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


class TestCLI(unittest.TestCase):
    def setUp(self):
        os.environ["USE_MOCK_PRIMITIVES"] = "true"
        os.environ["NO_COLOR"] = "1"

    def test_app_exposed_in_main(self):
        self.assertIs(app, main_app)

    def test_no_action_summary_does_not_claim_max_steps(self):
        output = io.StringIO()
        _render_summary(Console(file=output, width=120, no_color=True), {
            "goal": "Goal", "completed": False, "stop_reason": "no_action_generated",
            "total_steps_run": 0, "final_state": "Initial", "steps": [],
        })
        self.assertIn("NO ACTION GENERATED", output.getvalue())
        self.assertNotIn("MAX STEPS REACHED", output.getvalue())

    def test_help_commands(self):
        result = runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        output = strip_ansi(result.output)
        self.assertIn("run", output)
        self.assertIn("demo", output)
        self.assertIn("interactive", output)
        self.assertIn("visualize", output)

    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        self.assertEqual(result.exit_code, 0)
        output = strip_ansi(result.output)
        self.assertIn("--goal", output)
        self.assertIn("--state", output)
        self.assertIn("--workspace", output)
        self.assertIn("--max-steps", output)
        self.assertIn("--iterations", output)
        self.assertIn("--sim-depth", output)
        self.assertIn("--actions", output)
        self.assertIn("--proposal-model", output)
        self.assertIn("--mock", output)
        self.assertIn("--no-early-stop", output)
        self.assertIn("--no-execute", output)
        self.assertIn("--json", output)

    def test_visualize_help(self):
        result = runner.invoke(app, ["visualize", "--help"])
        self.assertEqual(result.exit_code, 0)
        output = strip_ansi(result.output)
        self.assertIn("--port", output)
        self.assertIn("--no-browser", output)

    def test_run_mock_json_mode(self):
        result = runner.invoke(
            app,
            [
                "run",
                "--mock",
                "--goal", "Test Goal",
                "--state", "Initial test state",
                "--max-steps", "1",
                "--iterations", "2",
                "--no-execute",
                "--json",
            ],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output.strip())
        self.assertEqual(data.get("goal"), "Test Goal")
        self.assertIn("completed", data)
        self.assertIn("steps", data)
        self.assertIn("total_steps_run", data)

    def test_global_json_flag(self):
        result = runner.invoke(
            app,
            [
                "--json",
                "run",
                "--mock",
                "--goal", "Global flag goal",
                "--max-steps", "1",
                "--iterations", "2",
                "--no-execute",
            ],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output.strip())
        self.assertEqual(data.get("goal"), "Global flag goal")

    def test_demo_mock_json(self):
        result = runner.invoke(
            app,
            [
                "demo",
                "--mock",
                "--max-steps", "1",
                "--iterations", "2",
                "--no-execute",
                "--json",
            ],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output.strip())
        self.assertIn("steps", data)

    def test_proposal_model_cli_flag(self):
        result = runner.invoke(
            app,
            [
                "run",
                "--mock",
                "--goal", "Custom Model Goal",
                "--proposal-model", "test-custom-model",
                "--max-steps", "1",
                "--iterations", "1",
                "--no-execute",
                "--json",
            ],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(os.environ.get("AGY_PROPOSAL_MODEL"), "test-custom-model")

    def test_proposal_model_envvar(self):
        old_env = os.environ.get("AGY_PROPOSAL_MODEL")
        try:
            os.environ["AGY_PROPOSAL_MODEL"] = "env-specified-model"
            result = runner.invoke(
                app,
                [
                    "run",
                    "--mock",
                    "--goal", "Env Model Goal",
                    "--max-steps", "1",
                    "--iterations", "1",
                    "--no-execute",
                    "--json",
                ],
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertEqual(os.environ.get("AGY_PROPOSAL_MODEL"), "env-specified-model")
        finally:
            if old_env is not None:
                os.environ["AGY_PROPOSAL_MODEL"] = old_env
            else:
                os.environ.pop("AGY_PROPOSAL_MODEL", None)
