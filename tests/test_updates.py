"""Tests for version checks and update-command behavior."""

import io
import json
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from urllib.error import HTTPError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from agent import updates
from agent.cli import app


class TestUpdates(unittest.TestCase):
    def test_semantic_version_comparison(self):
        self.assertTrue(updates.update_available("0.5.3", "0.6.1"))
        self.assertFalse(updates.update_available("0.6.1", "0.6.1"))
        self.assertFalse(updates.update_available("0.6.1", "0.5.3"))

    def test_installer_commands_pin_the_detected_release(self):
        with patch.object(updates.sys, "prefix", "/home/user/.local/pipx/venvs/mcts-agent"):
            self.assertEqual(
                updates._upgrade_command("0.6.1"),
                ["pipx", "install", "--force", "mcts-agent==0.6.1"],
            )
        with patch.object(updates.sys, "prefix", "/home/user/.local/share/uv/tools/mcts-agent"):
            self.assertEqual(
                updates._upgrade_command("0.6.1"),
                ["uv", "tool", "upgrade", "mcts-agent==0.6.1"],
            )

    def test_cache_uses_iso_timestamp_and_expires_after_one_day(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "update-check.json"
            now = 1_800_000_000.0
            updates._write_cache(cache, "0.8.0", now)
            contents = json.loads(cache.read_text(encoding="utf-8"))
            self.assertTrue(contents["checked_at"].endswith("Z"))
            self.assertEqual(updates._read_cache(cache, now + 60), (True, "0.8.0"))
            self.assertIsNone(
                updates._read_cache(cache, now + updates.CHECK_INTERVAL_SECONDS + 1)
            )

    def test_latest_version_uses_cache_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "update-check.json"
            updates._write_cache(cache, "0.9.0", time.time())
            with patch.object(updates, "_cache_path", return_value=cache), patch.object(
                updates, "_fetch_json", side_effect=AssertionError("network called")
            ):
                self.assertEqual(updates.latest_version(), "0.9.0")

    def test_missing_pypi_project_is_reported_as_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "update-check.json"
            missing = HTTPError(
                updates.PYPI_JSON_URL, 404, "Not Found", headers=None, fp=None
            )
            with patch.object(updates, "_cache_path", return_value=cache), patch.object(
                updates, "_fetch_json", side_effect=missing
            ):
                with self.assertRaisesRegex(
                    updates.PackageNotPublishedError, "not published on PyPI yet"
                ):
                    updates.latest_version(force=True)
            self.assertFalse(cache.exists())

    def test_update_command_explains_package_is_not_published(self):
        runner = CliRunner()
        with patch.object(updates, "installed_version", return_value="0.1.dev59"), patch.object(
            updates,
            "latest_version",
            side_effect=updates.PackageNotPublishedError(
                "mcts-agent is not published on PyPI yet; publish a release before using update"
            ),
        ):
            result = runner.invoke(app, ["--no-update-check", "update"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("not published on PyPI yet", result.stderr)

    def test_failed_check_is_cached_for_a_day(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "update-check.json"
            with patch.object(updates, "_cache_path", return_value=cache), patch.object(
                updates, "_fetch_json", side_effect=OSError("offline")
            ) as fetch:
                with self.assertRaises(updates.UpdateCheckUnavailable):
                    updates.latest_version(force=True)
                with self.assertRaises(updates.UpdateCheckUnavailable):
                    updates.latest_version()
            fetch.assert_called_once()

    def test_version_command_reports_current_and_latest(self):
        runner = CliRunner()
        with patch.object(updates, "installed_version", return_value="0.5.3"), patch.object(
            updates, "latest_version", return_value="0.6.1"
        ):
            result = runner.invoke(app, ["--no-update-check", "version"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("mcts-agent 0.5.3", result.output)
        self.assertIn("Latest: 0.6.1", result.output)
        self.assertIn("mcts-agent update", result.output)

    def test_update_command_does_not_install_when_current(self):
        runner = CliRunner()
        with patch.object(updates, "installed_version", return_value="0.6.1"), patch.object(
            updates, "latest_version", return_value="0.6.1"
        ), patch.object(updates, "perform_update") as perform_update:
            result = runner.invoke(app, ["--no-update-check", "update"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Already up to date", result.output)
        perform_update.assert_not_called()

    def test_available_update_is_reported_without_silent_install(self):
        output = io.StringIO()
        with patch.object(updates, "installed_version", return_value="0.5.3"), patch.object(
            updates, "check_latest_quietly", return_value="0.6.1"
        ), redirect_stderr(output):
            updates.show_update_notice(disabled=False, quiet=False)
        self.assertIn("Run `mcts-agent update` to upgrade.", output.getvalue())

    def test_disable_update_check_environment_variable_skips_network(self):
        with patch.object(
            updates, "check_latest_quietly", side_effect=AssertionError("network check called")
        ):
            updates.show_update_notice(disabled=True, quiet=False)

    def test_global_and_command_local_quiet_suppress_update_notice(self):
        runner = CliRunner()
        base_args = [
            "run", "--mock", "--goal", "quiet test", "--max-steps", "1",
            "--iterations", "1", "--no-execute",
        ]
        summary = {
            "run_id": "quiet-test", "completed": True, "stop_reason": "completed",
            "total_steps_run": 0, "final_state": "done", "steps": [],
        }
        with patch.dict("os.environ", {"MCTS_AGENT_DISABLE_UPDATE_CHECK": "0"}), patch.object(
            updates, "check_latest_quietly", side_effect=AssertionError("network check called")
        ), patch("agent.cli.load_config", return_value=SimpleNamespace(planner_model="test")), patch(
            "agent.cli.run_closed_loop_agent", return_value=summary
        ):
            for args in (base_args + ["--quiet"], ["--quiet"] + base_args):
                result = runner.invoke(app, args)
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
