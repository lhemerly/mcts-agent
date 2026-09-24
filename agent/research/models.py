"""Versioned research notebook; model-written findings are distinct from evidence."""

import json
from typing import Any, Literal

from agent.execution import ExecutionTraceRef

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Criterion(Record):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    validator: str = "system_one"
    required: bool = True

    @field_validator("id", "description", "validator")
    @classmethod
    def nonblank_trimmed(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be blank")
        return value


class ResearchBrief(Record):
    scope: str = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list, max_length=32)
    questions: list[str] = Field(default_factory=list, max_length=32)
    criteria: list[Criterion] = Field(min_length=1, max_length=32)

    @field_validator("scope")
    @classmethod
    def trim_scope(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Scope cannot be blank")
        return value

    @model_validator(mode="after")
    def check_criteria(self):
        ids = [c.id for c in self.criteria]
        if len(ids) != len(set(ids)) or not any(c.required for c in self.criteria):
            raise ValueError("Criteria need unique IDs and at least one required criterion")
        return self


class Finding(Record):
    criterion_id: str
    claim: str = Field(min_length=1)
    evidence_paths: list[str] = Field(default_factory=list, max_length=16)
    evidence_ids: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("claim")
    @classmethod
    def trim_claim(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Claim cannot be blank")
        return value


class StepReport(Record):
    summary: str = Field(min_length=1, max_length=12000)
    findings: list[Finding] = Field(default_factory=list, max_length=32)
    open_questions: list[str] = Field(default_factory=list, max_length=32)
    answer: str | None = Field(default=None, max_length=32000)

    @field_validator("summary")
    @classmethod
    def trim_summary(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Summary cannot be blank")
        return value

    @model_validator(mode="after")
    def unique_findings(self):
        ids = [f.criterion_id for f in self.findings]
        if len(ids) != len(set(ids)):
            raise ValueError("Use one combined finding per criterion per step")
        return self


class Evidence(Record):
    id: str
    source_path: str
    snapshot_path: str
    sha256: str
    text: str
    truncated: bool = False
    origin: Literal["harness_artifact"] = "harness_artifact"


class ValidationResult(Record):
    criterion_id: str
    claim: str
    status: Literal["supported", "contradicted", "inconclusive", "untested"]
    validator: str
    kind: Literal["heuristic", "deterministic", "mock"]
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    reason: str


class ResearchStep(Record):
    number: int
    action: str
    search_log: str | None = None
    execution_success: bool
    execution_trace: ExecutionTraceRef | None = None
    report: StepReport
    validations: list[ValidationResult] = Field(default_factory=list)


class PendingOperation(Record):
    kind: Literal["brief", "step"]
    action: str
    report_path: str
    search_log: str | None = None
    # Written only after a harness call returns. Unknown means never replay it.
    execution_success: bool | None = None
    execution_trace: ExecutionTraceRef | None = None


class ResearchState(Record):
    schema_version: Literal[1] = 1
    run_id: str
    query: str = Field(min_length=1, max_length=32000)
    workspace: str
    run_directory: str | None = None
    mock: bool = False
    agent_config: dict[str, Any] = Field(default_factory=dict)
    harness_state: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)
    brief: ResearchBrief | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    steps: list[ResearchStep] = Field(default_factory=list)
    pending: PendingOperation | None = None
    status: Literal["running", "candidate_ready", "budget_exhausted", "blocked", "failed"] = "running"
    answer: str | None = None
    error: str | None = None

    @field_validator("query")
    @classmethod
    def trim_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query cannot be blank")
        return value

    def latest_validations(self) -> dict[str, ValidationResult]:
        return {v.criterion_id: v for step in self.steps for v in step.validations}

    def context(self) -> str:
        """Compact prompt view; full history and evidence remain in the notebook."""
        context = {
            "query": self.query,
            "full_notebook_directory": self.run_directory,
            "brief": self.brief.model_dump() if self.brief else None,
            "latest_validations": {k: v.model_dump() for k, v in self.latest_validations().items()},
            "recent_steps": [s.model_dump() for s in self.steps[-3:]],
            "evidence": [{"id": e.id, "snapshot_path": e.snapshot_path,
                          "source_path": e.source_path, "sha256": e.sha256} for e in self.evidence[-12:]],
            "total_evidence_items": len(self.evidence),
            "answer": self.answer,
        }
        rendered = json.dumps(context, ensure_ascii=False)
        if len(rendered) <= 48_000:
            return rendered
        # Keep a strict prompt budget even when legacy checkpoints contain large
        # strings or many accumulated records.
        compact_brief = None
        if self.brief:
            compact_brief = {
                "scope": self.brief.scope[:4000],
                "assumptions": [x[:500] for x in self.brief.assumptions[:8]],
                "questions": [x[:500] for x in self.brief.questions[:8]],
                "criteria": [
                    {**c.model_dump(), "description": c.description[:500]}
                    for c in self.brief.criteria[:16]
                ],
            }
        compact = {
            "query": self.query[:8000], "full_notebook_directory": self.run_directory,
            "brief": compact_brief,
            "latest_validations": {
                k: {**v.model_dump(), "claim": v.claim[:500], "reason": v.reason[:500]}
                for k, v in list(self.latest_validations().items())[-16:]
            },
            "recent_steps": [
                {"number": s.number, "action": s.action[:500], "execution_success": s.execution_success,
                 "report": {"summary": s.report.summary[:1000], "answer": (s.report.answer or "")[:4000]},
                 "validations": [{**v.model_dump(), "claim": v.claim[:500], "reason": v.reason[:500]}
                                 for v in s.validations[:16]]}
                for s in self.steps[-1:]
            ],
            "evidence": [{"id": e.id, "snapshot_path": e.snapshot_path,
                          "source_path": e.source_path[:500], "sha256": e.sha256}
                         for e in self.evidence[-12:]],
            "total_evidence_items": len(self.evidence), "answer": (self.answer or "")[:8000],
        }
        rendered = json.dumps(compact, ensure_ascii=False)
        if len(rendered) <= 48_000:
            return rendered
        return json.dumps({
            "query": self.query[:8000],
            "full_notebook_directory": (self.run_directory or "")[:1000],
            "brief": {"scope": self.brief.scope[:2000]} if self.brief else None,
            "recent_steps": [], "latest_validations": {}, "evidence": [],
            "total_evidence_items": len(self.evidence), "answer": (self.answer or "")[:8000],
        }, ensure_ascii=False)


class ResearchSettings(Record):
    max_steps: int = Field(default=5, ge=1)
    width: int = Field(default=2, ge=1, le=8)
    depth: int = Field(default=2, ge=1, le=5)
    max_nodes: int = Field(default=64, ge=1, le=10000)

    @model_validator(mode="after")
    def bounded_search(self):
        if sum(self.width ** d for d in range(1, self.depth + 1)) > self.max_nodes:
            raise ValueError("Research tree exceeds max_nodes; reduce width or depth")
        return self

