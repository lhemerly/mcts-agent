"""Versioned research notebook; model-written findings are distinct from evidence."""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Criterion(Record):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    validator: str = "system_one"
    required: bool = True


class ResearchBrief(Record):
    scope: str = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    criteria: list[Criterion] = Field(min_length=1)

    @model_validator(mode="after")
    def check_criteria(self):
        ids = [c.id for c in self.criteria]
        if len(ids) != len(set(ids)) or not any(c.required for c in self.criteria):
            raise ValueError("Criteria need unique IDs and at least one required criterion")
        return self


class Finding(Record):
    criterion_id: str
    claim: str = Field(min_length=1)
    evidence_paths: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class StepReport(Record):
    summary: str = Field(min_length=1)
    findings: list[Finding] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    answer: str | None = None

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
    report: StepReport
    validations: list[ValidationResult] = Field(default_factory=list)


class PendingOperation(Record):
    kind: Literal["brief", "step"]
    action: str
    report_path: str
    search_log: str | None = None
    # Written only after a harness call returns. Unknown means never replay it.
    execution_success: bool | None = None


class ResearchState(Record):
    schema_version: Literal[1] = 1
    run_id: str
    query: str = Field(min_length=1)
    workspace: str
    run_directory: str | None = None
    mock: bool = False
    agent_config: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)
    brief: ResearchBrief | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    steps: list[ResearchStep] = Field(default_factory=list)
    pending: PendingOperation | None = None
    status: Literal["running", "candidate_ready", "budget_exhausted", "blocked", "failed"] = "running"
    answer: str | None = None
    error: str | None = None

    def latest_validations(self) -> dict[str, ValidationResult]:
        return {v.criterion_id: v for step in self.steps for v in step.validations}

    def context(self) -> str:
        """Compact prompt view; full history and evidence remain in the notebook."""
        return json.dumps({
            "query": self.query,
            "full_notebook_directory": self.run_directory,
            "brief": self.brief.model_dump() if self.brief else None,
            "latest_validations": {k: v.model_dump() for k, v in self.latest_validations().items()},
            "recent_steps": [s.model_dump() for s in self.steps[-3:]],
            "evidence": [{"id": e.id, "snapshot_path": e.snapshot_path,
                          "source_path": e.source_path, "sha256": e.sha256} for e in self.evidence[-12:]],
            "total_evidence_items": len(self.evidence),
            "answer": self.answer,
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
