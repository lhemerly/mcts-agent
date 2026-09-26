"""Version checks and explicit package updates for the mcts-agent CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from http.client import HTTPException
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PACKAGE_NAME = "mcts-agent"
PYPI_JSON_URL = "https://pypi.org/pypi/mcts-agent/json"
GITHUB_RELEASE_URL = "https://api.github.com/repos/lhemerly/mcts-agent/releases/tags/v{version}"
CHECK_INTERVAL_SECONDS = 24 * 60 * 60
REQUEST_TIMEOUT_SECONDS = 2


class UpdateCheckUnavailable(RuntimeError):
    """The most recent version lookup failed or returned invalid data."""


class PackageNotPublishedError(UpdateCheckUnavailable):
    """The package is not available on PyPI yet."""


def installed_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "0+unknown"


def source_commit() -> str | None:
    configured = os.getenv("MCTS_AGENT_GIT_COMMIT")
    if configured:
        return configured
    try:
        from agent._version import __commit_id__

        if isinstance(__commit_id__, str) and __commit_id__:
            return __commit_id__.removeprefix("g")
    except ImportError:
        pass
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            capture_output=True,
            check=False,
            timeout=1,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _cache_path() -> Path:
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData/Local"))
        return base / "mcts-agent" / "update-check.json"
    base = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "mcts-agent" / "update-check.json"


def _fetch_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "mcts-agent-update-check"})
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise TypeError("Expected a JSON object")
    return result


def _read_cache(path: Path, now: float) -> tuple[bool, str | None] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        checked_value = payload["checked_at"]
        if isinstance(checked_value, str):
            checked_at = datetime.fromisoformat(
                checked_value.replace("Z", "+00:00")
            ).timestamp()
        else:
            checked_at = float(checked_value)
        latest = payload["latest"]
        age = now - checked_at
        if (latest is None or isinstance(latest, str)) and 0 <= age < CHECK_INTERVAL_SECONDS:
            return True, latest
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def _write_cache(path: Path, latest: str | None, now: float) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "checked_at": datetime.fromtimestamp(now, timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "latest": latest,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        # A cache failure must never interfere with normal CLI use.
        pass


def latest_version(*, force: bool = False) -> str:
    """Return the latest PyPI version, using a 24-hour cache unless forced."""
    now = time.time()
    cache = _cache_path()
    if not force:
        cached = _read_cache(cache, now)
        if cached:
            if cached[1] is None:
                raise UpdateCheckUnavailable("the last PyPI check failed; retry in 24 hours")
            return cached[1]
    try:
        payload = _fetch_json(PYPI_JSON_URL)
        info = payload.get("info")
        latest = info.get("version") if isinstance(info, dict) else None
        if not isinstance(latest, str) or not latest:
            raise ValueError("PyPI response did not contain a package version")
    except HTTPError as exc:
        if exc.code == 404:
            raise PackageNotPublishedError(
                "mcts-agent is not published on PyPI yet; publish a release before using update"
            ) from exc
        _write_cache(cache, None, now)
        raise UpdateCheckUnavailable(f"PyPI request failed with HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError, HTTPException, ValueError, TypeError) as exc:
        _write_cache(cache, None, now)
        raise UpdateCheckUnavailable("PyPI could not be reached") from exc
    _write_cache(cache, latest, now)
    return latest


def update_available(installed: str, latest: str) -> bool:
    """Compare versions with PEP 440 semantics."""
    from packaging.version import InvalidVersion, Version

    try:
        return Version(latest) > Version(installed)
    except InvalidVersion:
        return False


def check_latest_quietly() -> str | None:
    """Perform the periodic check; network and cache errors are non-fatal."""
    try:
        return latest_version()
    except (
        HTTPError,
        URLError,
        TimeoutError,
        OSError,
        HTTPException,
        UpdateCheckUnavailable,
        ValueError,
        TypeError,
    ):
        return None


def show_update_notice(*, disabled: bool, quiet: bool) -> None:
    """Print a non-blocking availability notice unless explicitly suppressed."""
    if disabled or quiet:
        return
    latest = check_latest_quietly()
    installed = installed_version()
    if latest and update_available(installed, latest):
        sys.stderr.write(
            f"mcts-agent {latest} available (installed {installed}). "
            "Run `mcts-agent update` to upgrade.\n"
        )


def release_notes(release_version: str) -> str | None:
    try:
        data = _fetch_json(GITHUB_RELEASE_URL.format(version=release_version))
    except (HTTPError, URLError, TimeoutError, OSError, HTTPException, ValueError, TypeError):
        return None
    body = data.get("body")
    return body.strip() if isinstance(body, str) and body.strip() else None


def _upgrade_command(target_version: str) -> list[str]:
    prefix = str(Path(sys.prefix)).lower()
    package_spec = f"{PACKAGE_NAME}=={target_version}"
    if "pipx" in prefix and "venvs" in prefix:
        return ["pipx", "install", "--force", package_spec]
    if "uv" in prefix and "tools" in prefix:
        return ["uv", "tool", "upgrade", package_spec]
    return [sys.executable, "-m", "pip", "install", "--upgrade", package_spec]


def perform_update(target_version: str) -> subprocess.CompletedProcess[str]:
    """Run the appropriate explicit installer upgrade command."""
    command = _upgrade_command(target_version)
    return subprocess.run(command, text=True, check=False)
