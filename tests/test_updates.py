"""Tests for version checks and update-command behavior."""

import json
import tempfile
import time
import unittest
from pathlib import Path
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
        runner = CliRunner()
        with patch.dict("os.environ", {"MCTS_AGENT_DISABLE_UPDATE_CHECK": "0"}), patch.object(
            updates, "installed_version", return_value="0.5.3"
        ), patch.object(updates, "check_latest_quietly", return_value="0.6.1"), patch.object(
            updates, "perform_update"
        ) as perform_update:
            result = runner.invoke(app, ["--no-color", "research", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Run `mcts-agent update` to upgrade.", result.stderr)
        perform_update.assert_not_called()

    def test_disable_update_check_environment_variable_skips_network(self):
        runner = CliRunner()
        with patch.dict("os.environ", {"MCTS_AGENT_DISABLE_UPDATE_CHECK": "1"}), patch.object(
            updates, "check_latest_quietly", side_effect=AssertionError("network check called")
        ):
            result = runner.invoke(app, ["research", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)


if __name__ == "__main__":
    unittest.main()
