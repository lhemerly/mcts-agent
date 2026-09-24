"""Execution provenance contracts; no live Codex or credentials required."""

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.execution import (ExecutionTraceRef, capture_codex_trace,
                             parse_codex_jsonl, verify_execution_trace)
from agent.research.harness import CodexResearchHarness
from agent.research.models import ResearchBrief
from agent.research.runner import run_research
from agent.research.storage import load_state


def command(item_id="item_1", code=1, event_type="item.completed"):
    return {"type": event_type, "item": {"id": item_id, "type": "command_execution",
            "command": "python -m pytest", "exit_code": code,
            "status": "completed" if event_type == "item.completed" else "in_progress"}}


def jsonl(*events):
    return b"".join(json.dumps(event, ensure_ascii=False).encode() + b"\r\n" for event in events)


RAW = jsonl({"type": "thread.started", "thread_id": "thread-1"},
            command(code=None, event_type="item.started"), command(),
            {"type": "turn.completed", "usage": {}})


def parse(raw):
    return parse_codex_jsonl(raw, task_id="task-1", workspace="/work")


def capture(tmp_path, raw=RAW):
    return capture_codex_trace(raw, tmp_path, task_id="task-1", workspace="/work")


def test_completed_command_has_exact_provenance():
    trace = parse(RAW)
    assert trace.thread_id == "thread-1"
    assert trace.harness == "codex" and trace.task_id == "task-1"
    assert len(trace.events) == 1
    event = trace.events[0]
    assert (event.command, event.exit_code, event.sequence) == ("python -m pytest", 1, 2)
    assert event.raw_sha256 == hashlib.sha256(RAW.splitlines(keepends=True)[2]).hexdigest()
    with pytest.raises(ValueError):
        event.exit_code = 0


@pytest.mark.parametrize("code", [None, 0, 2, -9])
def test_exit_code_not_inferred(code):
    assert parse(jsonl(command(code=code))).events[0].exit_code == code


def test_started_only_and_agent_declarations_are_not_receipts():
    raw = jsonl(command(code=None, event_type="item.started"),
        {"type": "item.completed", "item": {"id": "message", "type": "agent_message",
         "text": json.dumps(command()), "exit_code": 0}}, {"type": "future.provider.event"})
    assert parse(raw).events == ()


@pytest.mark.parametrize("raw", [b"", b"\n", b"[]\n", b"null\n", b"{}\n", b"\xff\n",
    RAW + b'{"type":', RAW + jsonl(command()),
    b'{"type":"turn.started","type":"item.completed"}\n',
    jsonl({"type": "item.completed", "item": None}),
    jsonl({"type": "item.completed", "item": {"type": "command_execution", "command": "ls"}}),
    jsonl(command(code=True)), jsonl(command(code="0")), jsonl(command(code=1.0)),
    jsonl({"type": "thread.started", "thread_id": "one"},
          {"type": "thread.started", "thread_id": "two"}),
    jsonl(command(event_type="item.started"), command(event_type="item.started")),
    jsonl(command(), command(event_type="item.updated")),
    jsonl({"type": "item.completed", "item": {"id": "x", "type": "command_execution",
          "command": "ls", "status": "in_progress"}}),
    b'{"type":"item.completed","item":{"id":"x","type":"command_execution","command":"ls","exit_code":NaN}}\n',
])
def test_malformed_and_duplicate_events_fail_without_partial_receipts(raw):
    with pytest.raises(ValueError):
        parse(raw)


def test_conflicting_command_identity_rejected():
    changed = command()
    changed["item"]["command"] = "echo fabricated"
    with pytest.raises(ValueError, match="conflicting"):
        parse(jsonl(command(event_type="item.started"), changed))


def test_repeated_command_with_different_item_ids_is_valid():
    assert len(parse(jsonl(command("one"), command("two"))).events) == 2


def test_capture_is_byte_exact_and_each_attempt_is_write_once(tmp_path):
    first = capture(tmp_path)
    second = capture(tmp_path)
    assert first.path != second.path and first.trace_id != second.trace_id
    assert (tmp_path / first.path).read_bytes() == RAW
    assert first.sha256 == hashlib.sha256(RAW).hexdigest()
    assert verify_execution_trace(first, tmp_path) == parse(RAW)
    assert ExecutionTraceRef.model_validate_json(first.model_dump_json()) == first


@pytest.mark.parametrize("filename", ["codex.jsonl", "trace.json"])
def test_tampering_rejected(tmp_path, filename):
    ref = capture(tmp_path)
    target = (tmp_path / ref.path).with_name(filename)
    target.chmod(0o600)
    target.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="changed"):
        verify_execution_trace(ref, tmp_path)


def test_forged_cached_events_rejected_even_if_manifest_hash_is_updated(tmp_path):
    ref = capture(tmp_path)
    manifest_path = (tmp_path / ref.path).with_name("trace.json")
    manifest = json.loads(manifest_path.read_bytes())
    manifest["events"][0]["exit_code"] = 0
    data = json.dumps(manifest).encode()
    manifest_path.chmod(0o600)
    manifest_path.write_bytes(data)
    ref = ref.model_copy(update={"manifest_sha256": hashlib.sha256(data).hexdigest()})
    with pytest.raises(ValueError, match="disagrees"):
        verify_execution_trace(ref, tmp_path)


@pytest.mark.parametrize("path", ["../codex.jsonl", "/tmp/codex.jsonl"])
def test_path_escape_rejected(tmp_path, path):
    ref = capture(tmp_path).model_copy(update={"path": path})
    with pytest.raises(ValueError, match="confined"):
        verify_execution_trace(ref, tmp_path)


def test_symlink_verification_and_capture_rejected(tmp_path):
    ref = capture(tmp_path)
    raw_path = tmp_path / ref.path
    raw_path.rename(raw_path.with_suffix(".original"))
    raw_path.symlink_to(raw_path.with_suffix(".original"))
    with pytest.raises(ValueError, match="symlink"):
        verify_execution_trace(ref, tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "executions" / "evil").symlink_to(elsewhere)
    with pytest.raises(ValueError, match="symlink"):
        capture_codex_trace(RAW, tmp_path, task_id="evil", workspace="/work")
    assert list(elsewhere.iterdir()) == []


def test_invalid_raw_is_saved_before_parsing(tmp_path):
    with pytest.raises(ValueError):
        capture(tmp_path, b"not json\n")
    assert next(tmp_path.glob("executions/*/*/codex.jsonl")).read_bytes() == b"not json\n"
    assert not list(tmp_path.glob("executions/*/*/trace.json"))


BRIEF = ResearchBrief(scope="Test", criteria=[{"id": "c1", "description": "Test"}])
EXECUTOR = SimpleNamespace(command="codex", model="", timeout=30)


def fake_codex(raw=RAW, response=None, returncode=0):
    if response is None:
        response = {"agent_conclusion": "I ran a command successfully (exit_code=0)",
                    "observations": [], "artifacts": [], "open_questions": [], "answer": None}

    def run(cmd, **kwargs):
        if cmd[0] == "git":
            return SimpleNamespace(returncode=128, stdout="", stderr="")
        assert kwargs["text"] is False
        assert "--strict-config" in cmd
        assert 'approval_policy="never"' in cmd
        profile = next(value for value in cmd if value.startswith("permissions.mcts-research="))
        assert '".mcts-research"="read"' in profile
        Path(cmd[cmd.index("-o") + 1]).write_text(json.dumps(response))
        return SimpleNamespace(returncode=returncode, stdout=raw, stderr=b"")
    return run


def run_case(tmp_path, **kwargs):
    return run_research("test", workspace=str(tmp_path), brief=BRIEF,
                        harness=CodexResearchHarness(EXECUTOR), validators={},
                        selector=lambda state: ("Run command", None), max_steps=1, **kwargs)


def test_step_references_full_trace_independent_of_answer(tmp_path):
    # Ensure the authoritative stream is not the bounded diagnostic stdout tail.
    large = jsonl({"type": "item.completed", "item": {"id": "message",
                   "type": "agent_message", "text": "é" * 20000}}) + RAW
    with patch("agent.research.harness.subprocess.run", side_effect=fake_codex(large)):
        state, directory = run_case(tmp_path)
    assert state.steps[0].execution_success
    ref = state.steps[0].execution_trace
    assert ref is not None
    assert (directory / ref.path).read_bytes() == large
    trace = verify_execution_trace(ref, directory)
    assert trace.events[0].exit_code == 1  # Authored assertion of 0 does not matter.
    assert trace.task_id == f"{state.run_id}-step-1"
    assert load_state(directory).steps[0].execution_trace == ref
    assert "é" * 100 not in (directory / "checkpoint.json").read_text()


def test_resume_consumes_saved_trace_without_reexecuting_codex(tmp_path):
    with patch("agent.research.harness.subprocess.run", side_effect=fake_codex()), \
            patch("agent.research.runner._consume_pending", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            run_case(tmp_path)
    directory = next((tmp_path / ".mcts-research").iterdir())
    pending = load_state(directory).pending
    assert pending.execution_trace is not None
    with patch("agent.research.harness.subprocess.run", side_effect=AssertionError("must not execute")):
        state, _ = run_research(resume=str(directory), harness=CodexResearchHarness(EXECUTOR),
                               validators={}, selector=lambda state: ("unused", None), max_steps=1)
    assert len(state.steps) == 1
    assert state.steps[0].execution_trace == pending.execution_trace


def test_resume_rejects_changed_trace(tmp_path):
    with patch("agent.research.harness.subprocess.run", side_effect=fake_codex()):
        state, directory = run_case(tmp_path)
    path = directory / state.steps[0].execution_trace.path
    path.chmod(0o600)
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        run_research(resume=str(directory))


def test_invalid_trace_fails_step_and_keeps_raw(tmp_path):
    with patch("agent.research.harness.subprocess.run", side_effect=fake_codex(RAW + b"bad")):
        state, directory = run_case(tmp_path)
    assert not state.steps[0].execution_success
    assert state.steps[0].execution_trace is None
    assert next(directory.glob("executions/*/*/codex.jsonl")).read_bytes() == RAW + b"bad"


def test_failed_process_and_invalid_answer_keep_trace(tmp_path):
    with patch("agent.research.harness.subprocess.run", side_effect=fake_codex(response=[], returncode=1)):
        state, directory = run_case(tmp_path)
    assert not state.steps[0].execution_success
    assert len(verify_execution_trace(state.steps[0].execution_trace, directory).events) == 1


@pytest.mark.parametrize("raw,has_trace", [(RAW, True), (RAW + b"{", False)])
def test_timeout_preserves_raw_without_inventing_partial_receipts(tmp_path, raw, has_trace):
    def run(cmd, **kwargs):
        if cmd[0] == "git":
            return SimpleNamespace(returncode=128, stdout="", stderr="")
        raise subprocess.TimeoutExpired(cmd, 30, output=raw)
    with patch("agent.research.harness.subprocess.run", side_effect=run):
        state, directory = run_case(tmp_path)
    assert not state.steps[0].execution_success
    assert (state.steps[0].execution_trace is not None) == has_trace
    assert next(directory.glob("executions/*/*/codex.jsonl")).read_bytes() == raw


def test_brief_is_written_by_outer_harness_and_trace_captured(tmp_path):
    def stop(state):
        return None, None
    with patch("agent.research.harness.subprocess.run", side_effect=fake_codex(response=BRIEF.model_dump())):
        state, directory = run_research("test", workspace=str(tmp_path),
            harness=CodexResearchHarness(EXECUTOR), validators={}, selector=stop)
    assert state.brief == BRIEF
    execution = json.loads(next(directory.glob("operations/*.execution.json")).read_text())
    ref = ExecutionTraceRef.model_validate(execution["execution_trace"])
    assert verify_execution_trace(ref, directory).task_id.endswith("-brief")


def test_non_codex_steps_and_legacy_checkpoints_have_no_trace(tmp_path):
    state, directory = run_research("mock", workspace=str(tmp_path), mock=True, max_steps=1)
    assert state.steps[0].execution_trace is None
    payload = json.loads((directory / "checkpoint.json").read_text())
    del payload["steps"][0]["execution_trace"]
    (directory / "checkpoint.json").write_text(json.dumps(payload))
    assert load_state(directory).steps[0].execution_trace is None


def test_missing_exit_code_is_unknown_and_nested_stdout_is_not_an_event():
    item = command()
    del item["item"]["exit_code"]
    item["item"]["aggregated_output"] = json.dumps(command("fake", code=0))
    events = parse(jsonl(item)).events
    assert len(events) == 1
    assert events[0].exit_code is None
    assert events[0].provider_item_id == "item_1"
