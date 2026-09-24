"""Typed execution provenance derived only from captured harness event streams.

The checkpoint's reference is the trust anchor. Hashes detect changes; they are
not signatures against an attacker who can replace both the trace and checkpoint.
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ExecutionEvent(BaseModel):
    """One completed command; sequence is its zero-based physical JSONL line."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    kind: Literal["command"] = "command"
    provider_item_id: str
    command: str
    exit_code: int | None
    sequence: int = Field(ge=0)
    provider_event_type: Literal["item.completed"] = "item.completed"
    # Hash includes the original line terminator, if any.
    raw_sha256: str


class ExecutionTraceRef(BaseModel):
    """Compact checkpoint reference to an immutable manifest and its raw stream."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    trace_id: str = Field(pattern=r"^x-[0-9a-f]{32}$")
    harness: str
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExecutionTrace(BaseModel):
    """Generic execution envelope; events never come from the agent's answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    harness: str
    task_id: str
    workspace: str
    thread_id: str | None = None
    events: tuple[ExecutionEvent, ...] = ()


def _object(pairs: list[tuple[str, object]]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key: {key}")
        obj[key] = value
    return obj


def _invalid_constant(value: str):
    raise ValueError(f"Invalid JSON constant: {value}")


def parse_codex_jsonl(raw: bytes, *, task_id: str, workspace: str) -> ExecutionTrace:
    """Parse exact UTF-8 JSONL bytes, rejecting ambiguity without partial receipts.

    Unknown event/item types are retained in raw storage but yield no commands.
    Started/updated commands are checked for identity consistency, not receipts.
    Missing/null exit codes stay unknown; neither process success nor prose fills
    them in. Empty streams and incomplete final JSON are rejected.
    """
    events = []
    thread_id = None
    items: dict[str, tuple[str, str | None, bool]] = {}
    saw_event = False
    for sequence, line in enumerate(raw.splitlines(keepends=True)):
        if not line.strip():
            raise ValueError(f"Blank JSONL line {sequence}")
        try:
            event = json.loads(line.decode("utf-8"), object_pairs_hook=_object,
                               parse_constant=_invalid_constant)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"Invalid JSONL line {sequence}: {exc}") from exc
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise ValueError(f"Invalid event at line {sequence}")
        saw_event = True
        event_type = event["type"]
        if event_type == "thread.started":
            if thread_id is not None or not isinstance(event.get("thread_id"), str) or not event["thread_id"]:
                raise ValueError("Invalid or duplicate thread.started")
            thread_id = event["thread_id"]
        if event_type not in {"item.started", "item.updated", "item.completed"}:
            continue
        item = event.get("item")
        if not isinstance(item, dict) or not isinstance(item.get("type"), str):
            raise ValueError(f"Invalid item at line {sequence}")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"Missing item ID at line {sequence}")
        command = item.get("command") if item["type"] == "command_execution" else None
        if item["type"] == "command_execution":
            if not isinstance(command, str) or not command.strip():
                raise ValueError(f"Invalid command at line {sequence}")
            code = item.get("exit_code")
            if code is not None and type(code) is not int:
                raise ValueError(f"Invalid exit code at line {sequence}")
        previous = items.get(item_id)
        if previous is not None:
            if previous[:2] != (item["type"], command) or previous[2] or event_type == "item.started":
                raise ValueError(f"Duplicate or conflicting item: {item_id}")
        items[item_id] = (item["type"], command, event_type == "item.completed")
        if event_type == "item.completed" and item["type"] == "command_execution":
            if item.get("status") not in (None, "completed", "failed"):
                raise ValueError(f"Unfinished command marked completed: {item_id}")
            events.append(ExecutionEvent(provider_item_id=item_id, command=command,
                exit_code=item.get("exit_code"), sequence=sequence,
                raw_sha256=hashlib.sha256(line).hexdigest()))
    if not saw_event:
        raise ValueError("Empty execution trace")
    return ExecutionTrace(harness="codex", task_id=task_id, workspace=workspace,
                          thread_id=thread_id, events=tuple(events))


def _path(root: Path, relative: str) -> Path:
    """Confine artifacts, rejecting symlinks even when their target is in-run."""
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("Execution trace path must be relative and confined to the run")
    root = root.resolve(strict=True)
    path = root
    for part in rel.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Execution trace paths must not contain symlinks")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Execution trace path escapes run directory")
    return path


def _write_once(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)


def capture_codex_trace(raw: bytes, run_dir: Path, *, task_id: str,
                        workspace: str) -> ExecutionTraceRef:
    """Persist before parsing. Never overwrite an earlier attempt's artifacts.

    Malformed streams remain on disk for diagnosis but produce no reference.
    The caller must protect the control directory from the execution sandbox.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
        raise ValueError("Unsafe execution task ID")
    digest = hashlib.sha256(raw).hexdigest()
    trace_id = f"x-{uuid4().hex}"
    directory = _path(run_dir, f"executions/{task_id}/{trace_id}")
    directory.mkdir(parents=True, exist_ok=False)
    raw_path = directory / "codex.jsonl"
    _write_once(raw_path, raw)
    # Read the authoritative saved bytes, never an agent-authored manifest.
    trace = parse_codex_jsonl(raw_path.read_bytes(), task_id=task_id, workspace=workspace)
    manifest = trace.model_dump_json(indent=2).encode("utf-8")
    _write_once(directory / "trace.json", manifest)
    return ExecutionTraceRef(trace_id=trace_id, harness="codex",
        path=raw_path.relative_to(run_dir.resolve()).as_posix(), sha256=digest,
        manifest_sha256=hashlib.sha256(manifest).hexdigest())


def verify_execution_trace(ref: ExecutionTraceRef, run_dir: Path) -> ExecutionTrace:
    """Authenticate raw bytes and metadata, then reparse; never trust cached events.

    Raises ValueError/OSError on malformed, missing, unsupported or changed data.
    Verification authenticates recorded behavior, not the truth of a conclusion.
    """
    if ref.harness != "codex":
        raise ValueError(f"Unsupported execution harness: {ref.harness}")
    raw_path = _path(run_dir, ref.path)
    if raw_path.name != "codex.jsonl" or raw_path.parent.name != ref.trace_id:
        raise ValueError("Execution trace identity/path mismatch")
    raw = raw_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref.sha256:
        raise ValueError("Execution trace changed")
    manifest = _path(run_dir, str(Path(ref.path).with_name("trace.json"))).read_bytes()
    if hashlib.sha256(manifest).hexdigest() != ref.manifest_sha256:
        raise ValueError("Execution trace manifest changed")
    stored = ExecutionTrace.model_validate_json(manifest)
    trace = parse_codex_jsonl(raw, task_id=stored.task_id, workspace=stored.workspace)
    if trace != stored:
        raise ValueError("Execution trace manifest disagrees with raw events")
    return trace
