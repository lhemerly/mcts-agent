"""Atomic checkpoints, a per-run lock, and content-addressed artifact snapshots."""

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .models import Evidence, ResearchState


def atomic_write(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def run_lock(run_dir: Path):
    """Fail closed after a crash: inspect stale lock/pending work before removal."""
    path = run_dir / "run.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"Run is locked: {path}. Check for an active process before removing a stale lock.") from exc
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def save_state(run_dir: Path, state: ResearchState) -> None:
    ensure_run_contained(Path(state.workspace), run_dir)
    # The checkpoint is the authoritative event ledger; completed steps are immutable.
    atomic_write(run_dir / "checkpoint.json", state.model_dump_json(indent=2))
    brief = state.brief
    rows = [f"# Research run {state.run_id}", "", f"Status: {state.status}", "",
            state.query, "", "## Scope and assumptions", "",
            brief.scope if brief else "Brief pending", ""]
    if brief:
        rows += [f"- {a}" for a in brief.assumptions]
    rows += ["", "## Current candidate answer", "", state.answer or "No candidate answer yet.",
             "", "## Validation record", ""]
    latest = state.latest_validations()
    for criterion in brief.criteria if brief else []:
        result = latest.get(criterion.id)
        rows += [f"- {criterion.id}: {criterion.description} (required={criterion.required})"]
        if result:
            rows += [f"  Status: {result.status} ({result.kind}, {result.validator}). "
                     f"{result.reason} Evidence: {', '.join(result.evidence_ids) or 'none'}"]
        else:
            rows += ["  Status: untested; no finding has been evaluated."]
    rows += ["", "## Open questions", ""]
    questions = state.steps[-1].report.open_questions if state.steps else (brief.questions if brief else [])
    rows += [f"- {question}" for question in questions]
    rows += ["", "## Evidence snapshots", ""]
    rows += [f"- [{e.id}]({e.snapshot_path}) from {e.source_path}" for e in state.evidence]
    if state.error:
        rows += ["", f"Unresolved error: {state.error}"]
    rows += ["", "Candidate readiness is scoped to the recorded criteria. Heuristic support is not proof.", ""]
    atomic_write(run_dir / "report.md", "\n".join(rows))


def load_state(run_dir: Path) -> ResearchState:
    state = ResearchState.model_validate_json((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    ensure_run_contained(Path(state.workspace), run_dir)
    # Prior claims cannot silently retain support if their captured evidence changed.
    for evidence in state.evidence:
        snapshot = confined_path(run_dir, evidence.snapshot_path)
        verify_evidence(evidence, run_dir)
    return state


def verify_evidence(evidence: Evidence, run_dir: Path) -> Evidence:
    snapshot = confined_path(run_dir, evidence.snapshot_path)
    content = snapshot.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    try:
        snapshot_text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Evidence snapshot is not UTF-8: {evidence.id}") from exc
    if digest != evidence.sha256 or evidence.id != f"e-{digest}":
        raise ValueError(f"Evidence snapshot changed: {evidence.id}")
    if evidence.text != snapshot_text[:16000] or evidence.truncated != (len(snapshot_text) > 16000):
        raise ValueError(f"Cached evidence does not match its authenticated snapshot: {evidence.id}")
    return evidence


def confined_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("Evidence and report paths must be relative")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes the workspace: {relative}")
    return resolved


def ensure_run_contained(workspace: Path, run_dir: Path) -> None:
    """Reject a symlinked control root or run directory before any runner write."""
    work = workspace.resolve(strict=True)
    control = work / ".mcts-research"
    if control.is_symlink():
        raise ValueError("Research control directory must not be a symlink")
    if run_dir.parent.absolute() != control.absolute():
        raise ValueError("Research run directory must be a direct child of workspace/.mcts-research")
    if run_dir.is_symlink():
        raise ValueError("Research run directory must not be a symlink")
    if control.exists() and control.resolve(strict=True) != control:
        raise ValueError("Research control directory resolves outside the workspace")
    if run_dir.exists() and run_dir.resolve(strict=True) != run_dir:
        raise ValueError("Research run directory resolves outside the workspace")
    if run_dir.resolve(strict=False).parent != control:
        raise ValueError("Research run directory escapes the workspace")


def create_run_dir(workspace: Path, run_id: str) -> Path:
    """Create a private run directory under a non-symlinked workspace control root."""
    work = workspace.resolve(strict=True)
    control = work / ".mcts-research"
    if control.is_symlink():
        raise ValueError("Research control directory must not be a symlink")
    control.mkdir(mode=0o700, exist_ok=True)
    if control.is_symlink() or not control.is_dir() or control.resolve(strict=True) != control:
        raise ValueError("Research control directory must resolve inside the workspace")
    run_dir = control / run_id
    run_dir.mkdir(mode=0o700)
    ensure_run_contained(work, run_dir)
    return run_dir


def capture_evidence(workspace: Path, run_dir: Path, relative: str) -> Evidence:
    path = confined_path(workspace, relative)
    if not path.is_file() or path.is_relative_to(run_dir.resolve()):
        raise ValueError("Evidence must be a workspace artifact outside the research control directory")
    # Bound bytes before decoding; arbitrary binary files are not text evidence.
    with path.open("rb") as handle:
        content = handle.read(1_048_577)
    if len(content) > 1_048_576:
        raise ValueError("Evidence exceeds 1 MiB; produce a compact result artifact")
    text = content.decode("utf-8")
    if not text.strip():
        raise ValueError("Empty artifact is not evidence")
    digest = hashlib.sha256(content).hexdigest()
    snapshot = f"evidence/{digest}.txt"
    ensure_run_contained(workspace, run_dir)
    atomic_write_bytes(run_dir / snapshot, content)
    return Evidence(id=f"e-{digest}", source_path=relative, snapshot_path=snapshot,
                    sha256=digest, text=text[:16000], truncated=len(text) > 16000)


def read_object(path: Path) -> dict:
    with path.open("rb") as handle:
        content = handle.read(1_048_577)
    if len(content) > 1_048_576:
        raise ValueError("Structured harness report exceeds 1 MiB")
    return json.loads(content)
