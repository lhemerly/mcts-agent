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
