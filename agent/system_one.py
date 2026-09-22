"""Pluggable System One judgment providers.

Third-party packages can register a provider at import time or expose an entry
point in the ``mcts_agent.system_one`` group.  An entry point receives no
arguments and should call :func:`register_system_one_provider`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Callable, Mapping


class SystemOneProviderError(RuntimeError):
    """A System One provider could not complete a judgment."""


@dataclass(frozen=True)
class ChoiceResult:
    probabilities: Mapping[str, float]
    choice: str | None = None


class BaseSystemOneProvider(ABC):
    """Provider-neutral interface for discriminative judgments."""

    @abstractmethod
    def choose(self, state: Mapping[str, Any], instructions: str,
               criteria: Mapping[str, Any]) -> ChoiceResult: ...

    @abstractmethod
    def score(self, state: Mapping[str, Any], instructions: str,
              criteria: list[str]) -> float:
        """Return a score on the public 1–10 scale used by MCTS."""
        raise NotImplementedError

    @abstractmethod
    def noul(self, state: Mapping[str, Any], instructions: str) -> float: ...

    def batch_noul(self, state: Mapping[str, Any],
                   instructions: Mapping[str, str]) -> dict[str, float]:
        return {key: self.noul(state, prompt) for key, prompt in instructions.items()}


class TypeSafeSystemOneProvider(BaseSystemOneProvider):
    """Adapter for TypeSafe's Choice, Score, and Noul primitives."""

    def __init__(self) -> None:
        try:
            from typesafe_sdk import Choice, Noul, Score, TypeSafeAPIError, TypeSafeClient
        except ImportError as exc:
            raise SystemOneProviderError("typesafe-sdk is not installed") from exc
        self._Choice, self._Noul, self._Score = Choice, Noul, Score
        self._api_error, self._client_type, self._client = TypeSafeAPIError, TypeSafeClient, None

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._client_type()
        return self._client

    def _call(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Any:
        try:
            return self.client.system_one(state=dict(state), questions=dict(questions))
        except self._api_error as exc:
            raise SystemOneProviderError(str(exc)) from exc

    def choose(self, state: Mapping[str, Any], instructions: str,
               criteria: Mapping[str, Any]) -> ChoiceResult:
        result = self._call(state, {"choice": self._Choice(instructions=instructions, criteria=dict(criteria))})
        choice = result.choices["choice"]
        return ChoiceResult(choice.probabilities, getattr(choice, "choice", None))

    def score(self, state: Mapping[str, Any], instructions: str, criteria: list[str]) -> float:
        result = self._call(state, {"score": self._Score(instructions=instructions, criteria=criteria)})
        # TypeSafe scores are zero-indexed across the supplied rubric levels.
        return float(result.scores["score"].score) + 1.0

    def noul(self, state: Mapping[str, Any], instructions: str) -> float:
        return self.batch_noul(state, {"noul": instructions})["noul"]

    def batch_noul(self, state: Mapping[str, Any], instructions: Mapping[str, str]) -> dict[str, float]:
        result = self._call(state, {key: self._Noul(instructions=value) for key, value in instructions.items()})
        return {key: float(result.nouls[key].noul) for key in instructions}


SystemOneFactory = Callable[[], BaseSystemOneProvider]
_REGISTRY: dict[str, SystemOneFactory] = {}
_LOADED_ENTRY_POINTS = False


def register_system_one_provider(name: str, factory: SystemOneFactory, *, replace: bool = False) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("System One provider name cannot be empty")
    if key in _REGISTRY and not replace:
        raise ValueError(f"System One provider '{key}' is already registered")
    _REGISTRY[key] = factory


def _load_entry_points() -> None:
    global _LOADED_ENTRY_POINTS
    if _LOADED_ENTRY_POINTS:
        return
    _LOADED_ENTRY_POINTS = True
    try:
        discovered = entry_points(group="mcts_agent.system_one")
    except TypeError:  # Python 3.10 compatibility
        discovered = entry_points().get("mcts_agent.system_one", [])
    for entry_point in discovered:
        entry_point.load()()


def available_system_one_providers() -> tuple[str, ...]:
    _load_entry_points()
    return tuple(sorted(_REGISTRY))


def get_system_one_provider(name: str = "typesafe") -> BaseSystemOneProvider:
    _load_entry_points()
    try:
        return _REGISTRY[name.lower()]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown System One provider '{name}'. Available: {', '.join(available_system_one_providers())}"
        ) from exc


register_system_one_provider("typesafe", TypeSafeSystemOneProvider)
