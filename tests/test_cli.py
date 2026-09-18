"""Tests for MCTS Agent CLI interface (Typer & Rich)."""
import json
import os
import sys
import unittest
from typer.testing import CliRunner

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent.cli import app
from main import app as main_app

runner = CliRunner()


class TestCLI(unittest.TestCase):
    def setUp(self):
        os.environ["USE_MOCK_PRIMITIVES"] = "true"

    def test_app_exposed_in_main(self):
        self.assertIs(app, main_app)

    def test_help_commands(self):
        result = runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("run", result.output)
        self.assertIn("demo", result.output)
        self.assertIn("interactive", result.output)
        self.assertIn("visualize", result.output)

    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--goal", result.output)
        self.assertIn("--state", result.output)
        self.assertIn("--workspace", result.output)
        self.assertIn("--max-steps", result.output)
        self.assertIn("--iterations", result.output)
        self.assertIn("--sim-depth", result.output)
        self.assertIn("--actions", result.output)
        self.assertIn("--mock", result.output)
        self.assertIn("--no-early-stop", result.output)
        self.assertIn("--no-execute", result.output)
        self.assertIn("--json", result.output)

    def test_visualize_help(self):
        result = runner.invoke(app, ["visualize", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--port", result.output)
        self.assertIn("--no-browser", result.output)

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
