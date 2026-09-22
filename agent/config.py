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
    planner_provider: str = "agy"      # agy | pi | mock
    executor_provider: str = "agy"     # agy | pi | mock

    # Planner settings
    planner_model: str = "gemini-3.6-flash-low"

    # Executor settings
    executor_model: str = "gemini-3.8-flash-medium"
    executor_timeout: int = 1800

    # ── Tree expansion settings ──────────────────────────────────────────────
    # expansion_width: branching factor at each tree level (replaces actions_per_node
    # for the deep-expand strategy; actions_per_node still overrides per-call).
    expansion_width: int = 3

    # expansion_depth: how many levels deep to materialise the tree via _deep_expand.
    # With width=3 and depth=3, the tree has 3+9+27=39 nodes and 27 full leaf paths.
    expansion_depth: int = 3

    # ── Tree reuse settings ──────────────────────────────────────────────────
    # tree_reuse_enabled: carry the subtree of the executed action across closed-loop steps.
    tree_reuse_enabled: bool = True

    # reuse_score_threshold: minimum average_value a surviving leaf must have after
    # rescoring to avoid triggering a scramble (full tree rebuild).
    reuse_score_threshold: float = 1.5

    # n_recombined_paths: maximum unique vocabulary paths to try when grounded
    # surviving paths fall below the reuse threshold.
    n_recombined_paths: int = 18


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
        try:
            cwd_path = Path.cwd()
        except Exception:
            cwd_path = Path(".")
        candidates = (
            cwd_path / "antigravity.toml",
            cwd_path / "config.toml",
            Path.home() / ".config" / "mcts-agent" / "config.toml",
            Path.home() / ".antigravity.toml",
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
    tree_sec = config_data.get("tree", {})

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

    executor_model = (
        cli_overrides.get("executor_model")
        or os.getenv("AGY_MODEL")
        or os.getenv("MCTS_EXECUTOR_MODEL")
        or executor_sec.get("model")
        or "gemini-3.8-flash-medium"
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

    # ── Tree expansion / reuse config ────────────────────────────────────────
    def _int_cfg(key: str, env: str, section: dict, default: int) -> int:
        raw = cli_overrides.get(key) or os.getenv(env) or section.get(key)
        try:
            return int(raw) if raw is not None else default
        except (ValueError, TypeError):
            return default

    def _float_cfg(key: str, env: str, section: dict, default: float) -> float:
        raw = cli_overrides.get(key) or os.getenv(env) or section.get(key)
        try:
            return float(raw) if raw is not None else default
        except (ValueError, TypeError):
            return default

    def _bool_cfg(key: str, env: str, section: dict, default: bool) -> bool:
        # Use key-presence check so that an explicit False CLI override is respected.
        if key in cli_overrides:
            val = cli_overrides[key]
            if isinstance(val, bool):
                return val
            return str(val).lower() in ("1", "true", "yes")
        env_val = os.getenv(env)
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes")
        toml_val = section.get(key)
        if toml_val is not None:
            return bool(toml_val)
        return default

    expansion_width = _int_cfg("expansion_width", "MCTS_EXPANSION_WIDTH", tree_sec, 3)
    expansion_depth = _int_cfg("expansion_depth", "MCTS_EXPANSION_DEPTH", tree_sec, 3)
    tree_reuse_enabled = _bool_cfg("tree_reuse_enabled", "MCTS_TREE_REUSE", tree_sec, True)
    reuse_score_threshold = _float_cfg(
        "reuse_score_threshold", "MCTS_REUSE_THRESHOLD", tree_sec, 1.5
    )
    n_recombined_paths = _int_cfg(
        "n_recombined_paths", "MCTS_N_RECOMBINED", tree_sec, 18
    )

    # Default to mock providers when USE_MOCK_PRIMITIVES is active (e.g. --mock or unit tests)
    if os.getenv("USE_MOCK_PRIMITIVES", "false").lower() in ("1", "true", "yes"):
        if not cli_overrides.get("planner_provider") and not os.getenv("MCTS_PLANNER_PROVIDER"):
            planner_provider = "mock"
        if not cli_overrides.get("executor_provider") and not os.getenv("MCTS_EXECUTOR_PROVIDER"):
            executor_provider = "mock"

    return AgentConfig(
        planner_provider=str(planner_provider).lower(),
        executor_provider=str(executor_provider).lower(),
        planner_model=str(planner_model),
        executor_model=str(executor_model),
        executor_timeout=executor_timeout,
        expansion_width=expansion_width,
        expansion_depth=expansion_depth,
        tree_reuse_enabled=tree_reuse_enabled,
        reuse_score_threshold=reuse_score_threshold,
        n_recombined_paths=n_recombined_paths,
    )
