"""
agent/config.py — Configuration management for mcts-agent providers.

Supports loading and resolving configuration options from:
  1. CLI argument overrides
  2. Environment variables (.env)
  3. antigravity.toml / config.toml file
  4. Defaults
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


@dataclass
class AgentConfig:
    # Providers
    planner_provider: str = "agy"      # agy | llama_cpp | openai | mock
    executor_provider: str = "agy"     # agy | llama_cpp | local_cmd | mock

    # Planner settings
    planner_model: str = "gemini-3.6-flash-low"
    planner_endpoint: str = "http://localhost:8080/v1/chat/completions"
    planner_api_key: str = ""

    # Executor settings
    executor_model: str = "gemini-3.8-flash-medium"
    executor_endpoint: str = "http://localhost:8080/v1/chat/completions"
    executor_api_key: str = ""
    executor_timeout: int = 1800


def load_config(
    config_path: str | Path | None = None,
    cli_overrides: Optional[dict[str, Any]] = None,
) -> AgentConfig:
    """
    Load and resolve AgentConfig across config files, environment variables, and CLI overrides.
    """
    cli_overrides = cli_overrides or {}
    config_data: dict[str, Any] = {}

    # 1. Try reading TOML config file
    target_path = Path(config_path) if config_path else None
    if not target_path:
        package_root = Path(__file__).resolve().parent.parent
        candidates = (
            Path.cwd() / "antigravity.toml",
            Path.cwd() / "config.toml",
            package_root / "antigravity.toml",
            package_root / "config.toml",
        )
        target_path = next((candidate for candidate in candidates if candidate.exists()), None)

    if target_path and target_path.exists() and tomllib is not None:
        try:
            with open(target_path, "rb") as f:
                config_data = tomllib.load(f)
        except Exception as exc:
            print(f"[config] Warning: Failed to parse TOML config {target_path}: {exc}")

    providers_sec = config_data.get("providers", {})
    planner_sec = config_data.get("planner", {})
    executor_sec = config_data.get("executor", {})

    # 2. Resolve values with priority: CLI > ENV > TOML > Defaults
    planner_provider = (
        cli_overrides.get("planner_provider")
        or os.getenv("MCTS_PLANNER_PROVIDER")
        or providers_sec.get("planner")
        or "agy"
    )

    executor_provider = (
        cli_overrides.get("executor_provider")
        or os.getenv("MCTS_EXECUTOR_PROVIDER")
        or providers_sec.get("executor")
        or "agy"
    )

    planner_model = (
        cli_overrides.get("planner_model")
        or os.getenv("AGY_PROPOSAL_MODEL")
        or os.getenv("MCTS_PLANNER_MODEL")
        or planner_sec.get("model")
        or "gemini-3.6-flash-low"
    )

    planner_endpoint = (
        cli_overrides.get("planner_endpoint")
        or os.getenv("MCTS_PLANNER_ENDPOINT")
        or planner_sec.get("endpoint")
        or "http://localhost:8080/v1/chat/completions"
    )

    planner_api_key = (
        cli_overrides.get("planner_api_key")
        or os.getenv("MCTS_PLANNER_API_KEY")
        or os.getenv("OPENAI_API_KEY", "")
        or planner_sec.get("api_key", "")
    )

    executor_model = (
        cli_overrides.get("executor_model")
        or os.getenv("AGY_MODEL")
        or os.getenv("MCTS_EXECUTOR_MODEL")
        or executor_sec.get("model")
        or "gemini-3.8-flash-medium"
    )

    executor_endpoint = (
        cli_overrides.get("executor_endpoint")
        or os.getenv("MCTS_EXECUTOR_ENDPOINT")
        or executor_sec.get("endpoint")
        or "http://localhost:8080/v1/chat/completions"
    )

    executor_api_key = (
        cli_overrides.get("executor_api_key")
        or os.getenv("MCTS_EXECUTOR_API_KEY")
        or os.getenv("OPENAI_API_KEY", "")
        or executor_sec.get("api_key", "")
    )

    timeout_raw = (
        cli_overrides.get("executor_timeout")
        or os.getenv("MCTS_EXEC_TIMEOUT")
        or executor_sec.get("timeout")
        or 1800
    )
    try:
        executor_timeout = int(timeout_raw)
    except (ValueError, TypeError):
        executor_timeout = 1800

    return AgentConfig(
        planner_provider=str(planner_provider).lower(),
        executor_provider=str(executor_provider).lower(),
        planner_model=str(planner_model),
        planner_endpoint=str(planner_endpoint),
        planner_api_key=str(planner_api_key),
        executor_model=str(executor_model),
        executor_endpoint=str(executor_endpoint),
        executor_api_key=str(executor_api_key),
        executor_timeout=executor_timeout,
    )
