"""
primitives.py — Wrappers around the TypeSafe SDK's discriminative primitives.

The search uses Choice for priors and Score for hypothetical states. Noul is
used for execution checks and task completion, not for pruning proposed paths.

  • get_action_priors(state, actions)   → Choice  (Prioritise / Step 2)
  • evaluate_state(goal, state)         → Score   (Simulation / Step 3)
      Uses a 1-10 rubric; returns a float in [1, 10].

Mock mode
---------
Set USE_MOCK_PRIMITIVES=true in the environment to skip real API calls.
"""

from __future__ import annotations

import os
import random
from agent.config import load_config
from agent.system_one import SystemOneProviderError, get_system_one_provider

# ── Configuration ──────────────────────────────────────────────────────────────
_USE_MOCK = os.getenv("USE_MOCK_PRIMITIVES", "false").lower() in ("1", "true", "yes")


def _is_mock_enabled() -> bool:
    return os.getenv("USE_MOCK_PRIMITIVES", "false").lower() in ("1", "true", "yes")

# Pruning threshold: discard actions whose Noul probability is below this.
NOUL_VALIDITY_THRESHOLD: float = 0.8

# Completion threshold: trigger early stopping when Noul probability meets or exceeds this.
NOUL_COMPLETION_THRESHOLD: float = 0.85

# Candidate primes for dynamic branching factor
PRIME_ACTION_COUNTS: list[int] = [2, 3, 5, 7, 11, 13]

# Candidate primes for dynamic simulation lookahead depth
PRIME_SIMULATION_DEPTHS: list[int] = [2, 3, 5]

# Score rubric — 10 ordered levels mapping to a 1-10 scale.
# TypeSafe Score returns a 0-indexed float (0..9); we add 1 to report as 1-10.
_SCORE_RUBRIC: list[str] = [
    "1/10 — Completely off-track, no relevant progress whatsoever",
    "2/10 — Barely started, minimal alignment with the goal",
    "3/10 — Early steps taken, some relevance but far from done",
    "4/10 — Partial progress, notable steps completed yet still incomplete",
    "5/10 — Moderate progress, roughly halfway to the goal",
    "6/10 — Good progress, past halfway with significant work done",
    "7/10 — Substantial progress, most sub-tasks resolved",
    "8/10 — Strong progress, only minor gaps remain",
    "9/10 — Near completion, finishing touches needed",
    "10/10 — Goal fully and completely achieved",
]


def _get_provider(name: str | None = None):
    """Resolve the configured System One plugin for non-mock judgments."""
    return get_system_one_provider(name or load_config().system_one_provider)


# ── Mock helpers ───────────────────────────────────────────────────────────────

def _mock_batch_check_validity(actions: list[str]) -> dict[str, tuple[bool, float]]:
    """Keep mock search deterministic; rejection behavior is tested explicitly."""
    return {action: (True, 1.0) for action in actions}


def _mock_get_action_priors(actions: list[str]) -> dict[str, float]:
    """Return random (but normalised) priors for each action."""
    weights = [random.random() for _ in actions]
    total = sum(weights)
    return {a: w / total for a, w in zip(actions, weights)}


def _mock_evaluate_state() -> float:
    """Return a random score in [1, 10]."""
    return random.uniform(1.0, 10.0)


# ── Public primitives ──────────────────────────────────────────────────────────

def batch_check_validity(
    state: str, actions: list[str], *, mock: bool | None = None, system_one_provider: str | None = None
) -> dict[str, tuple[bool, float]]:
    """
    Noul primitive — batched pruning gate.

    ALL candidate actions are evaluated in a SINGLE system_one() call.
    Each action becomes its own Noul question keyed as 'action_0', 'action_1', …
    The action text is embedded directly in the question instructions so the
    model knows exactly what to evaluate.

    Returns
    -------
    dict mapping action text → (is_valid, confidence)
        is_valid   — True if the Noul probability ≥ NOUL_VALIDITY_THRESHOLD
        confidence — raw Noul probability (0–1, where 1 = definitely yes)
    """
    if mock is None:
        mock = _is_mock_enabled()
    if not actions:
        return {}

    if mock:
        return _mock_batch_check_validity(actions)

    # Build one Noul question per action in a single questions dict.
    # Key format: "action_<index>" (keys are not sent to the model).
    # The action text lives inside `instructions` so the model sees it.
    index_to_action: dict[str, str] = {}
    questions: dict[str, str] = {}
    for i, action in enumerate(actions):
        key = f"action_{i}"
        index_to_action[key] = action
        questions[key] = (
                f"Given the current state context, is this a feasible, clearly "
                f"scoped action plan that is relevant to making progress? "
                f"Multi-step or multi-file work is acceptable when it belongs to "
                f"the same requested outcome. Answer no only if it is unrelated, "
                f"internally contradictory, or impractically broad.\n"
                f"Proposed action: {action}"
        )

    try:
        probabilities = _get_provider(system_one_provider).batch_noul({"current_state": state}, questions)
        results: dict[str, tuple[bool, float]] = {}
        for key, action in index_to_action.items():
            prob = probabilities[key]
            results[action] = (prob >= NOUL_VALIDITY_THRESHOLD, prob)
        return results
    except (SystemOneProviderError, RuntimeError, ValueError) as exc:
        print(f"[primitives] Noul batch API error: {exc}. Defaulting all to invalid.")
        return {a: (False, 0.0) for a in actions}


def get_action_priors(
    state: str, actions: list[str], *, mock: bool | None = None, system_one_provider: str | None = None
) -> dict[str, float]:
    """
    Choice primitive — policy (prior probability) assignment.

    Given the current state and a list of candidate actions, Choice returns a
    probability distribution indicating how promising each action is.

    Returns
    -------
    dict mapping action text → prior probability (values sum to ~1.0)
    """
    if mock is None:
        mock = _is_mock_enabled()

    if mock:
        return _mock_get_action_priors(actions)

    if not actions:
        return {}

    criteria: dict[str, None] = {a: None for a in actions}

    try:
        response = _get_provider(system_one_provider).choose(
            {"current_state": state},
            "Given the current state, which of the following actions is most likely to make meaningful progress toward the goal?",
            criteria,
        )
        probs = response.probabilities
        n = len(actions)
        return {a: probs.get(a, 1.0 / n) for a in actions}
    except (SystemOneProviderError, RuntimeError, ValueError) as exc:
        print(f"[primitives] Choice API error: {exc}. Falling back to uniform priors.")
        n = len(actions)
        return {a: 1.0 / n for a in actions}


def evaluate_state(goal: str, simulated_state: str, *, mock: bool | None = None, system_one_provider: str | None = None) -> float:
    """
    Score primitive — value function / simulation replacement.

    Rates the simulated state against the goal on a 1-10 rubric.
    Providers return the public 1–10 scale; values are bounded defensively to
    preserve the MCTS score invariant.

    Returns
    -------
    float in [1, 10] — higher is better.
    """
    if mock is None:
        mock = _is_mock_enabled()

    if mock:
        return _mock_evaluate_state()

    try:
        score = _get_provider(system_one_provider).score(
            {"goal": goal, "simulated_state": simulated_state},
            "Rate how much progress has been made toward the goal based on the simulated state. "
            "Use the 1-10 rubric strictly, where 1 means no progress and 10 means the goal is fully achieved.",
            _SCORE_RUBRIC,
        )
        return min(10.0, max(1.0, score))
    except (SystemOneProviderError, RuntimeError, ValueError) as exc:
        print(f"[primitives] Score API error: {exc}. Defaulting to 1.")
        return 1.0


def select_action_count(
    state: str,
    goal: str,
    *,
    candidate_counts: list[int] | None = None,
    mock: bool | None = None,
    system_one_provider: str | None = None,
) -> int:
    """
    Use TypeSafe Choice to dynamically select the number of candidate actions
    to propose for expansion from primes up to 13 (2, 3, 5, 7, 11, 13).

    Provides path variance by exploring more branches when appropriate.
    """
    if mock is None:
        mock = _is_mock_enabled()
    counts = candidate_counts or PRIME_ACTION_COUNTS
    if mock:
        return random.choice(counts[:4])  # bias mock to 2, 3, 5, 7

    str_counts = [str(c) for c in counts]
    criteria = {s: None for s in str_counts}

    try:
        response = _get_provider(system_one_provider).choose(
            {"goal": goal, "current_state": state},
            "Given the current problem state and overall goal, choose how many distinct next "
            "action candidates should be generated. Options are prime numbers: 2, 3, 5, 7, 11, 13.",
            criteria,
        )
        probs = response.probabilities
        # Pick the prime with the highest probability
        chosen_str = max(probs, key=lambda k: probs.get(k, 0.0))
        chosen_count = int(chosen_str)
        print(f"[primitives] Dynamic branching factor chosen via Choice: {chosen_count} (probs: {probs})")
        return chosen_count
    except (SystemOneProviderError, RuntimeError, ValueError) as exc:
        print(f"[primitives] Choice action count error: {exc}. Defaulting to 3.")
        return 3


def check_task_completion(
    goal: str,
    state: str,
    *,
    mock: bool | None = None,
    system_one_provider: str | None = None,
) -> tuple[bool, float]:
    """
    Use TypeSafe Noul to determine if the task has been completed or the plan
    has been refined enough such that further MCTS search iterations are unnecessary.

    Returns
    -------
    (is_completed, confidence)
    """
    if mock is None:
        mock = _is_mock_enabled()
    if mock:
        # Mock mode: never complete early by default
        return False, 0.0

    try:
        prob = _get_provider(system_one_provider).noul(
            {"goal": goal, "current_state_or_plan": state},
            "Given the overall goal and the current state / accumulated plan, is the task fully "
            "completed or ready for execution without further search?",
        )
        is_done = prob >= NOUL_COMPLETION_THRESHOLD
        return is_done, prob
    except (SystemOneProviderError, RuntimeError, ValueError) as exc:
        print(f"[primitives] Noul completion check error: {exc}.")
        return False, 0.0


def check_action_execution(
    action: str,
    evidence: str,
    *,
    mock: bool | None = None,
    system_one_provider: str | None = None,
) -> tuple[bool, float]:
    """Check whether execution evidence supports completion of this exact action."""
    if mock is None:
        mock = _is_mock_enabled()
    if mock:
        return True, 1.0
    try:
        confidence = _get_provider(system_one_provider).noul(
            {"requested_action": action, "execution_evidence": evidence},
            "Does the execution evidence show that the exact requested action was completed? "
            "A zero exit code only shows that commands ran. Reject unrelated edits, placeholder results, and missing requested work.",
        )
        return confidence >= NOUL_VALIDITY_THRESHOLD, confidence
    except (SystemOneProviderError, RuntimeError, ValueError) as exc:
        print(f"[primitives] Action verification error: {exc}.")
        return False, 0.0


def check_command_alignment(
    action: str,
    command: str,
    *,
    mock: bool | None = None,
    system_one_provider: str | None = None,
) -> tuple[bool, float]:
    """Reject generated commands that do not directly implement the chosen action."""
    if mock is None:
        mock = _is_mock_enabled()
    if mock:
        return True, 1.0
    try:
        confidence = _get_provider(system_one_provider).noul(
            {"requested_action": action, "proposed_shell_command": command[:12000]},
            "Would these commands directly and completely carry out the requested action? "
            "Reject unrelated work, placeholder data, incomplete scripts, or effects that cannot be determined.",
        )
        return confidence >= NOUL_VALIDITY_THRESHOLD, confidence
    except (SystemOneProviderError, RuntimeError, ValueError) as exc:
        print(f"[primitives] Command alignment check error: {exc}.")
        return False, 0.0

def select_simulation_depth(
    state: str,
    goal: str,
    *,
    candidate_depths: list[int] | None = None,
    mock: bool | None = None,
    system_one_provider: str | None = None,
) -> int:
    """
    Use TypeSafe Choice to dynamically select the number of simulated lookahead
    steps (depth) to project into the future during MCTS simulation from primes: 2, 3, 5.
    """
    if mock is None:
        mock = _is_mock_enabled()
    depths = candidate_depths or PRIME_SIMULATION_DEPTHS
    if mock:
        return random.choice(depths)

    str_depths = [str(d) for d in depths]
    criteria = {s: None for s in str_depths}

    try:
        response = _get_provider(system_one_provider).choose(
            {"goal": goal, "current_state": state},
            "Given the current problem state and overall goal, choose the simulated lookahead "
            "depth for MCTS. Options are prime numbers: 2, 3, 5.",
            criteria,
        )
        probs = response.probabilities
        chosen_str = max(probs, key=lambda k: probs.get(k, 0.0))
        chosen_depth = int(chosen_str)
        print(f"[primitives] Dynamic simulation depth chosen via Choice: {chosen_depth} (probs: {probs})")
        return chosen_depth
    except (SystemOneProviderError, RuntimeError, ValueError) as exc:
        print(f"[primitives] Choice simulation depth error: {exc}. Defaulting to 2.")
        return 2


def discriminative_choose_best_action(
    goal: str,
    state: str,
    candidates: list[Any],
    *,
    mock: bool | None = None,
    system_one_provider: str | None = None,
) -> Any:
    """
    Use TypeSafe Choice to make the final discriminative decision on which candidate branch
    to commit to and execute, synthesizing MCTS statistics (visits, value score, prior) with
    semantic goal alignment.
    """
    if mock is None:
        mock = _is_mock_enabled()
    if not candidates:
        raise ValueError("No candidate nodes to choose from.")
    if len(candidates) == 1:
        return candidates[0]

    # Filter out unvisited nodes if any visited nodes exist
    visited = [c for c in candidates if c.visits > 0]
    eval_candidates = visited if visited else candidates

    if mock:
        return max(eval_candidates, key=lambda c: (c.visits, c.average_value))

    action_to_node: dict[str, Any] = {}
    criteria: dict[str, str | None] = {}
    for i, node in enumerate(eval_candidates):
        key = node.action_taken or f"action_{i}"
        action_to_node[key] = node
        criteria[key] = (
            f"MCTS visits: {node.visits}, "
            f"Average score: {node.average_value:.2f}/10, "
            f"Prior: {node.prior_probability:.2f}"
        )

    try:
        response = _get_provider(system_one_provider).choose(
            {"goal": goal, "current_state": state},
            "Given the goal, current state context, and MCTS search tree statistics for each "
            "candidate branch, select the best action to execute next.",
            criteria,
        )
        choice_key = response.choice
        best_node = action_to_node.get(choice_key)
        if best_node is None:
            probs = response.probabilities
            if probs:
                top_key = max(probs, key=lambda k: probs.get(k, 0.0))
                best_node = action_to_node.get(top_key)

        if best_node is not None:
            print(f"[primitives] Discriminative Choice selected immediate action: '{best_node.action_taken}'")
            return best_node

        return max(eval_candidates, key=lambda c: (c.visits, c.average_value))
    except (SystemOneProviderError, RuntimeError, ValueError) as exc:
        print(f"[primitives] Choice decision error: {exc}. Falling back to MCTS robust child.")
        return max(eval_candidates, key=lambda c: (c.visits, c.average_value))
