"""Research orchestration: plan, investigate, validate evidence, checkpoint, repeat."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping
from uuid import uuid4

from agent.config import AgentConfig, load_config
from agent.mcts import run_mcts
from agent.node import Node
from agent.providers import get_executor_provider
from agent.system_one import get_system_one_provider
from .harness import HarnessResearchAdapter, MockResearchHarness, ResearchHarness
from .models import (PendingOperation, ResearchBrief, ResearchSettings, ResearchState,
                     ResearchStep, StepReport, ValidationResult)
from .storage import (atomic_write, capture_evidence, confined_path, load_state,
                      read_object, run_lock, save_state)
from .validation import MockValidator, ResearchScorer, ResearchValidator, SystemOneValidator

Selector = Callable[[ResearchState], tuple[str | None, str | None]]


def _validate_report(state: ResearchState, report: StepReport, success: bool,
                     run_dir: Path, validators: Mapping[str, ResearchValidator]) -> list[ValidationResult]:
    criteria = {c.id: c for c in state.brief.criteria}
    if any(f.criterion_id not in criteria for f in report.findings):
        raise ValueError("Report refers to a criterion outside the fixed research brief")
    results = []
    for finding in report.findings:
        criterion = criteria[finding.criterion_id]
        validator = validators.get(criterion.validator)
        captured, errors = [], []
        known_evidence = {e.id: e for e in state.evidence}
        for evidence_id in dict.fromkeys(finding.evidence_ids):
            if evidence_id in known_evidence:
                captured.append(known_evidence[evidence_id])
            else:
                errors.append(f"Unknown evidence ID: {evidence_id}")
        for path in dict.fromkeys(finding.evidence_paths):
            try:
                item = capture_evidence(Path(state.workspace), run_dir, path)
                if item.id not in {e.id for e in captured}:
                    captured.append(item)
                if item.id not in {e.id for e in state.evidence}:
                    state.evidence.append(item)
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
        result = ValidationResult(
            criterion_id=criterion.id, claim=finding.claim, status="untested",
            validator=criterion.validator, kind=validator.kind if validator else "heuristic",
            evidence_ids=[e.id for e in captured], reason="No evidence supplied",
        )
        if not success or errors or validator is None:
            result.status = "inconclusive"
            result.reason = "; ".join(errors) or (
                "Execution failed or was skipped" if not success else "Requested validator is unavailable"
            )
        elif captured:
            try:
                result = validator.validate(criterion, finding, captured, answer=report.answer)
                valid_ids = {e.id for e in captured}
                if (result.criterion_id != criterion.id or result.claim != finding.claim
                        or not set(result.evidence_ids).issubset(valid_ids)
                        or (result.status in ("supported", "contradicted") and not result.evidence_ids)):
                    raise ValueError("Validator returned mismatched claim or evidence references")
                # Classification comes from installed adapter code, never model output.
                result.validator, result.kind = criterion.validator, validator.kind
            except Exception as exc:
                result = ValidationResult(
                    criterion_id=criterion.id, claim=finding.claim, status="inconclusive",
                    validator=criterion.validator, kind=validator.kind,
                    evidence_ids=[e.id for e in captured], reason=f"Validation error: {exc}",
                )
        results.append(result)
    return results


def _consume_pending(state: ResearchState, run_dir: Path,
                     validators: Mapping[str, ResearchValidator]) -> bool:
    pending = state.pending
    if pending.execution_success is None:
        state.status = "blocked"
        state.error = "Interrupted harness call has unknown effects; reconcile it before restarting. It will not be replayed."
        save_state(run_dir, state)
        return False
    path = confined_path(run_dir, pending.report_path)
    if pending.kind == "brief":
        if not pending.execution_success:
            raise ValueError("Research brief generation failed")
        brief = ResearchBrief.model_validate(read_object(path))
        if any(c.validator != "system_one" or not c.required for c in brief.criteria):
            raise ValueError("Generated criteria must use system_one and be required; supply a reviewed brief for custom validators")
        state.brief = brief
    else:
        report = StepReport.model_validate(read_object(path))
        validations = _validate_report(state, report, pending.execution_success, run_dir, validators)
        state.steps.append(ResearchStep(
            number=len(state.steps) + 1, action=pending.action, search_log=pending.search_log,
            execution_success=pending.execution_success, report=report, validations=validations,
        ))
        if pending.execution_success and report.answer and report.answer.strip():
            state.answer = report.answer
            # Recheck every required criterion against this synthesis. Old support
            # for a different answer must not certify a new, unrelated answer.
            current = {v.criterion_id: v for v in validations}
            if all(c.id in current and current[c.id].status == "supported"
                   for c in state.brief.criteria if c.required):
                state.status = "candidate_ready"
    state.pending = None
    save_state(run_dir, state)
    return True


def _perform(state: ResearchState, run_dir: Path, harness: ResearchHarness,
             action: str, kind: str, search_log: str | None = None) -> None:
    relative = f"operations/{uuid4().hex}.json"
    output = run_dir / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    state.pending = PendingOperation(kind=kind, action=action, report_path=relative, search_log=search_log)
    save_state(run_dir, state)  # Commit intent before any external side effects.
    result = harness.perform(state, action, output, kind)
    atomic_write(output.with_suffix(".execution.json"), json.dumps(result, default=str, indent=2))
    state.pending.execution_success = (
        result.get("success") is True and result.get("returncode", 0) == 0 and not result.get("skipped", False)
    )
    save_state(run_dir, state)  # Resume consumes the report, never executes it again.


def run_research(
    query: str | None = None, *, workspace: str | None = None, resume: str | None = None,
    config: AgentConfig | None = None, settings: ResearchSettings | None = None,
    max_steps: int | None = None, brief: ResearchBrief | None = None, mock: bool = False,
    harness: ResearchHarness | None = None, validators: Mapping[str, ResearchValidator] | None = None,
    selector: Selector | None = None,
) -> tuple[ResearchState, Path]:
    """Run up to max_steps additional investigations, or resume a saved notebook.

    Adapter injection supports domain-specific validation without changing the
    orchestrator. Validators should be side-effect-free because validation can
    be repeated after interruption; harness operations are never auto-replayed.
    """
    if resume:
        if query is not None or workspace is not None or brief is not None or config is not None:
            raise ValueError("Resume uses the saved query, workspace, brief, and provider configuration")
        run_dir = Path(resume).resolve()
        if run_dir.is_file():
            run_dir = run_dir.parent
    else:
        if query is None or not query.strip():
            raise ValueError("A non-empty research query is required")
        work = Path(workspace or ".").resolve()
        if not work.is_dir():
            raise ValueError("Workspace must be an existing directory")
        run_dir = work / ".mcts-research" / uuid4().hex
        run_dir.mkdir(parents=True)
    with run_lock(run_dir):
        if resume:
            state = load_state(run_dir)
            if mock and not state.mock:
                raise ValueError("Cannot resume a live run in mock mode")
            mock = state.mock
            cfg = AgentConfig(**state.agent_config)
            limits = settings or ResearchSettings(**state.settings)
        else:
            cfg = config or load_config()
            limits = settings or ResearchSettings()
            state = ResearchState(run_id=run_dir.name, query=query, workspace=str(work),
                                  mock=mock, brief=brief, agent_config=asdict(cfg))
        if max_steps is not None:
            limits = ResearchSettings(**{**limits.model_dump(), "max_steps": max_steps})
        state.settings = limits.model_dump()
        state.run_directory = str(run_dir)
        if not Path(state.workspace).is_dir():
            raise ValueError("Saved workspace no longer exists")
        if state.status == "candidate_ready":
            return state, run_dir
        state.status, state.error = "running", None
        save_state(run_dir, state)
        try:
            provider = None
            if not mock and (validators is None or selector is None):
                provider = get_system_one_provider(cfg.system_one_provider)
            active_validators = validators if validators is not None else {
                "system_one": MockValidator() if mock else SystemOneValidator(provider)
            }
            active_harness = harness or (MockResearchHarness() if mock else
                                        HarnessResearchAdapter(get_executor_provider(cfg)))

            def select(current: ResearchState) -> tuple[str | None, str | None]:
                if mock:
                    return "Record a synthetic observation and identify remaining validation gaps", None
                goal = (
                    f"Investigate the original query: {current.query}\n"
                    "Select an evidence-producing action or scoped synthesis addressing the brief's "
                    "unresolved criteria. Falsification and reduced uncertainty count as progress. "
                    "All proposed continuations are hypothetical. Do not infer completion from a plan."
                )
                best, log = run_mcts(
                    Node(state=current.context()), goal, iterations=0,
                    actions_per_node=limits.width, expansion_depth=limits.depth,
                    early_stop_noul=False, log_dir=str(run_dir / "search"), config=cfg,
                    state_evaluator=ResearchScorer(provider), step=len(current.steps) + 1,
                    run_id=current.run_id,
                )
                return best.action_taken, log

            # Count a pending investigation against this invocation's budget.
            starting_steps = len(state.steps)
            if state.pending and not _consume_pending(state, run_dir, active_validators):
                return state, run_dir
            if state.brief is None:
                _perform(state, run_dir, active_harness, "Formulate the research brief", "brief")
                _consume_pending(state, run_dir, active_validators)
            while len(state.steps) - starting_steps < limits.max_steps and state.status == "running":
                action, search_log = (selector or select)(state)
                if not action or not action.strip():
                    state.status, state.error = "blocked", "Planner produced no actionable investigation"
                    break
                _perform(state, run_dir, active_harness, action, "step", search_log)
                if not _consume_pending(state, run_dir, active_validators):
                    break
            if state.status == "running":
                state.status = "budget_exhausted"
        except Exception as exc:
            state.status, state.error = "failed", str(exc)
        save_state(run_dir, state)
        return state, run_dir
