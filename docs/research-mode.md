# Research mode

Research mode lets a harness investigate an open question over multiple steps.
It maintains a structured notebook of assumptions, criteria, claims, evidence,
and validation results. System One guides the search and evaluates supplied
evidence; it is not treated as an oracle for an unknown answer.

This initial version is available through the CLI and Python API. The existing
`run`, `demo`, `interactive`, and web UI execution workflows keep their behavior.

## Run and resume

Install with `pip install -e .`. The offline demo needs no keys or harness binaries:

```bash
mkdir research-workspace
mcts-agent research --query "Investigate an open question" \
  --workspace ./research-workspace --mock --max-steps 2 --json
```

The JSON output identifies the checkpoint and Markdown report. Resume using
the returned path, without supplying a new query or workspace:

```bash
mcts-agent research --resume ./research-workspace/.mcts-research/RUN_ID/checkpoint.json \
  --max-steps 3 --json
```

`--max-steps` counts **additional investigations in this invocation**, including
a returned-but-not-yet-consumed pending investigation. Brief generation is a
separate harness call. Mock mode is saved in the checkpoint; resumed mock runs
remain offline and never report scientific validation. Mock mode demonstrates
the notebook/validation/resume plumbing with a deterministic action, not model
quality or live MCTS performance.

For live operation, configure the existing harness and System One providers:

```bash
mcts-agent research --query "Develop and test a numerical model under explicit assumptions" \
  --workspace ./research-workspace --planner pi --executor pi \
  --system-one typesafe --width 2 --depth 2 --max-steps 5
```

Codex CLI can be selected as the research executor:

```bash
mcts-agent research --query "Investigate a bounded question" \
  --workspace ./research-workspace --executor codex --max-steps 5
```

Install and authenticate the Codex CLI separately. The connector invokes
`codex exec` with `--json`, `--output-schema`, and Codex's `workspace-write`
sandbox. It does not bypass the configured approval policy. Each step is a
bounded structured task; the CLI timeout comes from `executor_timeout` (default
30 minutes). A timed-out subprocess is terminated and recorded as cancelled.
Codex's thread ID is saved in the run checkpoint and resumed for later steps.

The structured Codex response preserves its `agent_conclusion` separately from
actual `observations`, `artifacts`, and `workspace_changes` in each operation's
`*.execution.json` record. Only observations that identify a criterion and an
existing workspace artifact become findings in the research ledger. Those
artifact paths are captured as immutable evidence snapshots and passed to the
configured validator; Codex's conclusion by itself cannot support a criterion.
Workspace change paths are currently reported for Git workspaces and represent
the workspace's changed-path view at task completion.

Provider keys and harness installation follow the existing setup. The process
uses the configured executor's tools and permissions; Research mode adds no
sandbox. Run it in a workspace and environment suitable for the intended work.
The standalone `main.py` remains the legacy argparse entrypoint; use
`mcts-agent research` or `python -m agent.cli research` for this command.

## One investigation cycle

1. The executor harness writes a schema-validated **research brief**: scope,
   assumptions, questions, and criteria. A reviewed brief can instead be supplied
   with `--brief path/to/brief.json`. The original query is preserved separately.
2. The existing planner proposes investigative actions using the notebook context.
   MCTS expands a bounded hypothetical tree. Its injected research scorer values
   expected relevance, falsification, and uncertainty reduction, rather than
   pretending an imagined result is progress already achieved.
3. The chosen action runs through the existing executor. It writes a structured
   `StepReport` and actual workspace artifacts. It can produce negative findings,
   revised hypotheses, or a synthesis; a positive result is not required.
4. The runner snapshots referenced artifacts, applies the criterion's validator,
   updates the notebook, and saves a checkpoint and readable report.
5. The next search sees the updated validations and open questions. A candidate
   answer is ready only when **every required criterion is supported in the same
   report that supplies that answer**. Earlier support for a different answer
   cannot certify an unchecked synthesis.

Research searches start fresh from observations each cycle. The optional MCTS
`state_evaluator` hook aggregates leaf values before selection; Research mode
uses zero extra PUCT iterations because replaying fixed leaf scores adds no new
evidence. Existing MCTS behavior is retained for callers without this hook.
Reusing hypothetical branches across real observations is intentionally deferred
until its behavior is evaluated (see issue #11).

## Evidence and validation

`Finding` contains a criterion ID, a claim, and workspace-relative
`evidence_paths` and/or previously captured `evidence_ids`. A `StepReport` also
contains a summary, open questions, and an optional candidate answer.

Evidence artifacts must be nonempty UTF-8 files, at most 1 MiB, inside the
workspace and outside the research control directory. Path traversal and
symlinks escaping the workspace are rejected. Captured files are copied into
content-addressed snapshots; repeated evidence can be cited by ID after its
original file changes. Snapshot hashes are checked on resume.

Validators receive up to 16,000 characters of each artifact, with a `truncated`
flag, along with the criterion, claim, and candidate answer. Produce compact
measurement/test reports when full data exceed this bound. A file's existence
does **not** establish its claim. Artifacts are labeled `harness_artifact`:
their provenance and methodology may themselves need checking. Even a plausible
artifact can be fabricated; System One cannot eliminate this limitation.

Results are `untested`, `supported`, `contradicted`, or `inconclusive`, with
adapter identity, evidence IDs, rationale, and optional confidence. The built-in
System One adapter separately asks about support and contradiction. Conflicting
judgments, unavailable providers, missing evidence, invalid paths, failed
execution, and invalid validator references cannot promote a claim. Its output
is explicitly marked **heuristic**; confidence is not a calibrated probability
of scientific correctness.

## Plug in a domain validator

The orchestrator accepts a `validators` mapping. Implement the
`ResearchValidator` protocol in `agent/research/validation.py`:

```python
class DomainValidator:
    name = "domain"
    kind = "deterministic"  # Only when the adapter really performs such checks.

    def validate(self, criterion, finding, evidence, *, answer=None):
        # Run the domain check on captured evidence and, if present, the answer.
        # Return ValidationResult with the same criterion/claim and real evidence IDs.
        ...
```

Then pass a reviewed `ResearchBrief` whose criteria use `validator="domain"`,
and `validators={"domain": DomainValidator()}` to `run_research(...)`.
Custom validators and a custom `ResearchHarness` or action `selector` can be
injected without modifying provider factories or the search engine. Validation
must be side-effect-free because it can be repeated after a crash. Constructors
for custom adapters must be supplied again when resuming via the Python API;
executable Python objects are not serialized into the checkpoint. The CLI uses
the general System One validator; unknown custom names remain inconclusive.

The integration tests provide a working deterministic example: propose candidate
2, independently reject it because its square is not 9, then test candidate 3.
The next prompt sees the contradiction and the final synthesis must agree with
the checked candidate. This tests a discovery loop without claiming to benchmark
genuinely unsolved science.

## Persistence and stop conditions

Each run creates `.mcts-research/RUN_ID/` containing `checkpoint.json`,
`report.md`, operation reports and executor transcripts, evidence snapshots,
and compatible MCTS search logs. The checkpoint is the authoritative ledger of
completed steps, not just an accumulated text prompt. Prompts include the latest
criterion verdicts, the last three steps and twelve evidence references, and a
pointer to the full notebook for retrieval. This is bounded history exposure,
not a semantic retrieval system.

Checkpoint replacement is atomic and a per-run lock prevents concurrent runners.
Execution intent is saved **before** calling the harness. If a call returned and
its result was checkpointed, resume consumes its report without executing again.
If a crash leaves the call's effects unknown, resume reports `blocked`; it does
not guess that replay is safe. Inspect the operation transcript and workspace
before reconciling the pending operation. A hard process termination may leave
`run.lock`; remove it only after establishing that no runner/harness is active.
Exactly-once external execution is not guaranteed by a local JSON checkpoint.

Statuses:

| Status | Meaning |
| --- | --- |
| `candidate_ready` | The current answer has support under every required criterion; review evidence and validator kinds. |
| `budget_exhausted` | This invocation's step limit was reached; the notebook can be resumed. |
| `blocked` | No action was proposed or an interrupted execution needs reconciliation. |
| `failed` | A provider, structured report, or runtime operation failed; inspect the saved error. |

The default search is width 2, depth 2. A proposed configuration is rejected if
`sum(width**d for d in 1..depth)` exceeds `--max-nodes` (default 64). Providers
cannot expand more candidates than the requested width in custom-scored search.
These are per-search node and per-invocation step limits, not a global token,
cost, or wall-clock budget. Existing executor timeouts still apply. Resume uses
saved providers and search settings; Python callers can explicitly change the
search settings for a later invocation.

Generated criteria are fixed for the run and use the general System One adapter.
This first version does not autonomously weaken criteria, alter the agreed scope,
repair malformed reports, supervise background jobs, or certify novelty. Those
are separate extension points. Open questions and hypotheses can evolve in step
reports while the original query, assumptions, evidence, and required checks
remain visible. A long-running run can pause and resume; it cannot guarantee a
solution merely by being allowed more time.

## Verification

```bash
python -m unittest tests.test_research -v
python -m unittest discover tests -v
```

Tests cover falsification and recovery, answer/criterion alignment, unsupported
claims, failed execution, contradictory System One judgments, evidence capture
and reuse, snapshot integrity, schema enforcement, MCTS scoring integration,
safe resume, and offline CLI behavior.


## Execution provenance

Codex research operations capture the **exact bytes** from `codex exec --json`
stdout after the process exits, before interpreting its authored answer. Each
attempt gets a fresh, write-once directory:

```text
.mcts-research/RUN_ID/executions/TASK_ID/TRACE_ID/codex.jsonl
.mcts-research/RUN_ID/executions/TASK_ID/TRACE_ID/trace.json
```

`codex.jsonl` is authoritative. `trace.json` binds the harness/task/workspace
metadata to typed events. Both files are hashed, created exclusively, flushed to
disk, and made read-only. A `ResearchStep.execution_trace` holds only an
`ExecutionTraceRef` (IDs, relative path, raw SHA-256, manifest SHA-256). The pending
operation holds the same reference before report validation, so a checkpointed
execution resumes without rerunning Codex. Loading either completed or pending
work verifies its trace. Older checkpoints and other harnesses default to `None`.
Brief traces are also captured and referenced in their operation execution record.

Downstream consumers use the public API; they do not parse Codex JSONL:

```python
from agent.execution import verify_execution_trace

if step.execution_trace is not None:
    trace = verify_execution_trace(step.execution_trace, run_dir)
    for event in trace.events:
        print(trace.task_id, event.provider_item_id, event.command, event.exit_code)
```

Verification confines paths to the supplied run directory, rejects symlink paths,
authenticates raw and manifest bytes, reparses the raw stream, and compares the
result with the saved manifest. It raises `ValueError` or `OSError` on failure.
`ExecutionTrace.events` is an immutable tuple of frozen `ExecutionEvent` records.
`sequence` is the zero-based physical line number; `raw_sha256` hashes that line's
exact bytes **including its terminator**, when present. `workspace` is the
harness launch directory, not an assertion about the shell's eventual cwd.

Only `item.completed` items of type `command_execution` create events. Started
and updated items do not establish completion. Repeated commands with different
provider item IDs remain separate events. Duplicate completion IDs, conflicting
item identities, duplicate JSON keys, invalid types, invalid UTF-8, and truncated
JSON fail closed: no partial events are returned. Unknown event/item types remain
in the raw trace and produce no command receipts. Absent or null `exit_code`
remains `None`; process success and authored conclusions never fill it in.

Nonzero process exits and timeouts can retain completed command events. A timeout
with a truncated last event retains the raw file for diagnosis but yields no
verified reference. Malformed streams likewise remain on disk without a manifest.
A process interruption before the harness checkpoints the result still has unknown
effects and is never replayed automatically; existing reconciliation rules apply.

### Trust and compatibility

The outer harness creates provenance; Codex-authored JSON, artifact contents,
agent messages, and command output cannot supply execution events. A recorded
command/exit code describes execution, not whether a scientific claim is true.
Downstream validation still decides what a command establishes.

Codex research now requires a CLI supporting `--strict-config` and named
permissions profiles. The harness selects an explicit profile that permits
workspace edits, makes `.mcts-research` read-only, disables command network access,
and sets `approval_policy="never"`. It applies the same policy on resumed threads
and generates briefs through structured final output instead of allowing Codex
to write control files. Unsupported versions fail without an insecure fallback.
This protection relies on Codex's sandbox enforcement and trusted local
configuration/tools; external MCP services or other processes with independent
filesystem access must not be granted writes to the control directory.

Hashes are integrity checks relative to the trusted checkpoint, **not digital
signatures**. They cannot defend against an actor replacing both checkpoint
references and artifacts. Read-only file modes alone are not a security boundary
against the owning OS user. For hostile hosts or independent external writers,
protect the checkpoint and run directory using separate OS identities or a
read-only mount. The offline test suite checks parser, storage, recovery, and CLI
policy construction; it does not substitute for a platform-specific live sandbox
test.

Schema and configuration references:
[Codex non-interactive events](https://learn.chatgpt.com/docs/non-interactive-mode),
[filesystem permissions](https://learn.chatgpt.com/docs/permissions), and
[CLI configuration flags](https://learn.chatgpt.com/docs/cli/reference).
