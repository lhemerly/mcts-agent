"""Research-mode integration tests with scripted harnesses and independent checks."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from agent.cli import app
from agent.config import AgentConfig
from agent.mcts import run_mcts
from agent.node import Node
from agent.research import ResearchBrief, ResearchSettings, run_research
from agent.research.harness import CodexResearchHarness
from agent.research.models import (Criterion, Finding, PendingOperation, ResearchState,
                                   ValidationResult)
from agent.research.storage import atomic_write, capture_evidence, load_state, save_state
from agent.research.validation import ResearchScorer, SystemOneValidator


def brief():
    return ResearchBrief(scope="Find a positive integer whose square is nine",
                         assumptions=["Restrict the domain to positive integers"],
                         criteria=[Criterion(id="square", description="Candidate squared equals nine",
                                             validator="square")])


class SquareValidator:
    """Independent deterministic check; does not trust the harness's claim."""
    name, kind = "square", "deterministic"

    def validate(self, criterion, finding, evidence, *, answer=None):
        value = json.loads(evidence[0].text)["candidate"]
        valid = type(value) is int and value > 0 and value * value == 9 and (answer is None or answer == str(value))
        return ValidationResult(criterion_id=criterion.id, claim=finding.claim,
                                status="supported" if valid else "contradicted",
                                validator=self.name, kind=self.kind,
                                evidence_ids=[evidence[0].id], reason="Checked n > 0 and n*n == 9")


class ScriptedHarness:
    def __init__(self, candidates=(2, 3), success=True, evidence=True):
        self.candidates, self.success, self.evidence, self.calls = candidates, success, evidence, 0

    def perform(self, state, action, output, kind):
        self.calls += 1
        if kind == "brief":
            raise AssertionError("Test supplies a reviewed brief")
        candidate = self.candidates[min(len(state.steps), len(self.candidates) - 1)]
        atomic_write(Path(state.workspace) / "result.json", json.dumps({"candidate": candidate}))
        atomic_write(output, json.dumps({
            "summary": f"Tested candidate {candidate}",
            "findings": [{"criterion_id": "square", "claim": f"{candidate} is a solution",
                          "evidence_paths": ["result.json"] if self.evidence else []}],
            "answer": str(candidate), "open_questions": [],
        }))
        return {"success": self.success, "returncode": 0 if self.success else 1}


class ResearchTests(unittest.TestCase):
    def run_case(self, workspace, **kwargs):
        return run_research("Find n", workspace=workspace, brief=brief(), config=AgentConfig(),
                            validators={"square": SquareValidator()},
                            selector=lambda state: ("Test the next candidate", None), **kwargs)

    def test_falsification_guides_next_step_and_only_evidence_supports_answer(self):
        contexts = []
        def select(state):
            contexts.append(state.context())
            return "Test next candidate", None
        with tempfile.TemporaryDirectory() as tmp:
            state, directory = run_research(
                "Find n", workspace=tmp, brief=brief(), harness=ScriptedHarness(),
                validators={"square": SquareValidator()}, selector=select,
            )
            self.assertEqual(state.status, "candidate_ready")
            self.assertEqual(len(state.steps), 2)
            self.assertIn("contradicted", contexts[1])
            self.assertEqual(state.answer, "3")
            self.assertEqual(len(state.evidence), 2)
            self.assertEqual(json.loads(state.evidence[0].text)["candidate"], 2)
            self.assertEqual(load_state(directory), state)
            self.assertIn("deterministic", (directory / "report.md").read_text())

    def test_zero_exit_or_confident_answer_without_evidence_does_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, _ = self.run_case(tmp, harness=ScriptedHarness((3,), evidence=False), max_steps=1)
            self.assertEqual(state.status, "budget_exhausted")
            self.assertEqual(state.steps[0].validations[0].status, "untested")

    def test_failed_execution_does_not_promote_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, _ = self.run_case(tmp, harness=ScriptedHarness((3,), success=False), max_steps=1)
            self.assertEqual(state.status, "budget_exhausted")
            self.assertIsNone(state.answer)
            self.assertEqual(state.steps[0].validations[0].status, "inconclusive")
            self.assertEqual(state.evidence, [])

    def test_resume_continues_without_replaying_completed_steps(self):
        harness = ScriptedHarness()
        with tempfile.TemporaryDirectory() as tmp:
            first, directory = self.run_case(tmp, harness=harness, max_steps=1)
            self.assertEqual(first.status, "budget_exhausted")
            state, _ = run_research(resume=str(directory), harness=harness,
                                    validators={"square": SquareValidator()},
                                    selector=lambda s: ("Try another candidate", None), max_steps=1)
            self.assertEqual(state.status, "candidate_ready")
            self.assertEqual(harness.calls, 2)
            self.assertEqual(len(state.steps), 2)

    def test_interrupted_operation_is_not_reexecuted(self):
        harness = ScriptedHarness()
        with tempfile.TemporaryDirectory() as tmp:
            state, directory = self.run_case(tmp, harness=harness, max_steps=1)
            state.pending = PendingOperation(kind="step", action="unknown effects", report_path="operations/missing.json")
            save_state(directory, state)
            result, _ = run_research(resume=str(directory), harness=harness,
                                     validators={"square": SquareValidator()}, selector=lambda s: ("never", None))
            self.assertEqual(result.status, "blocked")
            self.assertEqual(harness.calls, 1)

    def test_resume_consumes_returned_report_without_executing_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, directory = self.run_case(tmp, harness=ScriptedHarness(), max_steps=1)
            output = directory / "operations" / "pending.json"
            harness = ScriptedHarness()
            harness.perform(state, "candidate", output, "step")
            state.pending = PendingOperation(kind="step", action="candidate", report_path="operations/pending.json",
                                             execution_success=True)
            save_state(directory, state)
            result, _ = run_research(resume=str(directory), harness=harness,
                                     validators={"square": SquareValidator()}, selector=lambda s: ("never", None))
            self.assertEqual(result.status, "candidate_ready")
            self.assertEqual(harness.calls, 1)

    def test_codex_result_separates_conclusion_from_observations_and_artifacts(self):
        class FakeCodex:
            command, model, timeout = "codex", "", 30

        with tempfile.TemporaryDirectory() as tmp:
            state = ResearchState(run_id="run", query="Check a claim", workspace=tmp,
                                  brief=brief(), agent_config={})
            output = Path(tmp) / ".mcts-research" / "run" / "operations" / "report.json"
            output.parent.mkdir(parents=True)
            response = {"agent_conclusion": "Candidate appears valid", "observations": [
                {"type": "test_result", "artifact": "evidence/check.txt", "criterion_id": "square",
                 "claim": "The recorded candidate squares to nine"},
                {"type": "source_check", "artifact": "evidence/source.txt", "criterion_id": "square",
                 "claim": "The implementation checks the square correctly"}],
                "artifacts": ["evidence/check.txt"], "open_questions": [], "answer": "3"}
            Path(tmp, "evidence").mkdir()
            Path(tmp, "evidence", "check.txt").write_text('{"candidate": 3}', encoding="utf-8")
            Path(tmp, "evidence", "source.txt").write_text("assert candidate ** 2 == 9", encoding="utf-8")

            commands = []
            def run(command, **kwargs):
                if command[0] == "git":
                    return Mock(returncode=128, stdout="", stderr="")
                commands.append(command)
                final = Path(command[command.index("-o") + 1])
                final.write_text(json.dumps(response), encoding="utf-8")
                return Mock(returncode=0, stdout='{"type":"thread.started","thread_id":"thread-1"}\n', stderr="")

            with patch("agent.research.harness.subprocess.run", side_effect=run):
                result = CodexResearchHarness(FakeCodex()).perform(state, "Run a check", output, "step")
            self.assertEqual(result["agent_conclusion"], "Candidate appears valid")
            self.assertEqual(result["observations"][0]["type"], "test_result")
            self.assertEqual(state.harness_state["codex_thread_id"], "thread-1")
            self.assertNotIn("codex_thread_id", state.agent_config)
            restored = ResearchState.model_validate_json(state.model_dump_json())
            self.assertEqual(restored.harness_state["codex_thread_id"], "thread-1")
            AgentConfig(**restored.agent_config)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(report["findings"]), 1)
            self.assertCountEqual(report["findings"][0]["evidence_paths"],
                                  ["evidence/check.txt", "evidence/source.txt"])
            self.assertEqual(report["answer"], "3")
            self.assertIn("--skip-git-repo-check", commands[0])
            self.assertLess(commands[0].index("--cd"), commands[0].index("--skip-git-repo-check"))

            # A later turn resumes the saved session only after exec-level flags.
            state.pending = None
            with patch("agent.research.harness.subprocess.run", side_effect=run):
                CodexResearchHarness(FakeCodex()).perform(state, "Continue", output, "step")
            resume_command = commands[-1]
            self.assertEqual(resume_command[1], "exec")
            self.assertLess(resume_command.index("--sandbox"), resume_command.index("resume"))
            self.assertIn("--skip-git-repo-check", resume_command)

    def test_codex_adapter_selection_normalizes_provider_case(self):
        class FakeCodex:
            command, model, timeout = "codex", "", 30

        with tempfile.TemporaryDirectory() as tmp, \
                patch("agent.research.runner.get_executor_provider", return_value=FakeCodex()), \
                patch("agent.research.runner.CodexResearchHarness") as codex_adapter, \
                patch("agent.research.runner.HarnessResearchAdapter") as generic_adapter:
            state, _ = run_research(
                "query", workspace=tmp, brief=brief(), config=AgentConfig(executor_provider="Codex"),
                validators={"square": SquareValidator()}, selector=lambda current: (None, None),
            )
        self.assertEqual(state.status, "blocked")
        codex_adapter.assert_called_once()
        generic_adapter.assert_not_called()

    def test_evidence_paths_and_snapshot_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = root / ".mcts-research" / "run"
            control.mkdir(parents=True)
            for path in ("../escape.txt", str(root / "absolute.txt")):
                with self.assertRaises(ValueError):
                    capture_evidence(root, control, path)
            atomic_write(root / "a.txt", "observation")
            (root / "outside").symlink_to(root.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                capture_evidence(root, control, "outside/a.txt")
            item = capture_evidence(root, control, "a.txt")
            state = ResearchState(run_id="test", query="Q", workspace=tmp, evidence=[item])
            save_state(control, state)
            atomic_write(control / item.snapshot_path, "tampered")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_state(control)

    def test_checkpoint_cannot_change_cached_evidence_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = root / ".mcts-research" / "run"
            control.mkdir(parents=True)
            atomic_write(root / "a.txt", "  exact bytes\r\n")
            item = capture_evidence(root, control, "a.txt")
            self.assertEqual((control / item.snapshot_path).read_bytes(), b"  exact bytes\r\n")
            state = ResearchState(run_id="run", query="Q", workspace=tmp, evidence=[item])
            save_state(control, state)
            data = json.loads((control / "checkpoint.json").read_text())
            data["evidence"][0]["text"] = "forged cache"
            atomic_write(control / "checkpoint.json", json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Cached evidence"):
                load_state(control)

    def test_research_rejects_symlinked_control_root(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            Path(tmp, ".mcts-research").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                self.run_case(tmp, harness=ScriptedHarness(), max_steps=1)

    def test_invalid_report_stays_failed_without_false_completion(self):
        harness = Mock()
        harness.perform.return_value = {"success": True, "returncode": 0}
        with tempfile.TemporaryDirectory() as tmp:
            state, directory = self.run_case(tmp, harness=harness, max_steps=1)
            self.assertEqual(state.status, "failed")
            self.assertIsNotNone(state.pending)
            self.assertEqual(state.steps, [])
            self.assertTrue((directory / "checkpoint.json").exists())

    def test_candidate_requires_all_required_criteria(self):
        rb = brief()
        rb.criteria.append(Criterion(id="second", description="Additional required check", validator="square"))
        with tempfile.TemporaryDirectory() as tmp:
            state, _ = run_research("Find n", workspace=tmp, brief=rb, harness=ScriptedHarness((3,)),
                                    validators={"square": SquareValidator()},
                                    selector=lambda s: ("Check", None), max_steps=1)
            self.assertEqual(state.status, "budget_exhausted")

    def test_old_validation_cannot_certify_a_new_unchecked_answer(self):
        class UncheckedSynthesis(ScriptedHarness):
            def perform(self, state, action, output, kind):
                result = super().perform(state, action, output, kind)
                data = json.loads(output.read_text())
                if not state.steps:
                    data["answer"] = None
                else:
                    data["findings"] = []
                    data["answer"] = "An arbitrary unsupported answer"
                atomic_write(output, json.dumps(data))
                return result
        with tempfile.TemporaryDirectory() as tmp:
            state, _ = self.run_case(tmp, harness=UncheckedSynthesis((3,)), max_steps=2)
            self.assertEqual(state.steps[0].validations[0].status, "supported")
            self.assertEqual(state.status, "budget_exhausted")

    def test_validator_receives_the_synthesis_as_well_as_claim(self):
        class BadAnswer(ScriptedHarness):
            def perform(self, state, action, output, kind):
                result = super().perform(state, action, output, kind)
                data = json.loads(output.read_text())
                data["answer"] = "4"
                atomic_write(output, json.dumps(data))
                return result
        with tempfile.TemporaryDirectory() as tmp:
            state, _ = self.run_case(tmp, harness=BadAnswer((3,)), max_steps=1)
            self.assertEqual(state.status, "budget_exhausted")
            self.assertEqual(state.steps[0].validations[0].status, "contradicted")

    def test_prior_evidence_can_be_reused_by_id_after_source_changes(self):
        class ReuseEvidence(ScriptedHarness):
            def perform(self, state, action, output, kind):
                result = super().perform(state, action, output, kind)
                data = json.loads(output.read_text())
                if not state.steps:
                    data["answer"] = None
                else:
                    data["answer"] = "3"
                    data["findings"] = [{"criterion_id": "square", "claim": "3 is a solution",
                                         "evidence_ids": [state.evidence[0].id]}]
                atomic_write(output, json.dumps(data))
                return result
        with tempfile.TemporaryDirectory() as tmp:
            state, _ = self.run_case(tmp, harness=ReuseEvidence((3, 4)), max_steps=2)
            self.assertEqual(state.status, "candidate_ready")
            self.assertEqual(len(state.evidence), 1)

    def test_real_interrupt_leaves_a_nonreplayable_checkpoint(self):
        harness = Mock()
        harness.perform.side_effect = KeyboardInterrupt
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(KeyboardInterrupt):
                self.run_case(tmp, harness=harness)
            checkpoint = next(Path(tmp).glob(".mcts-research/*/checkpoint.json"))
            state, _ = run_research(resume=str(checkpoint), harness=harness,
                                    validators={"square": SquareValidator()}, selector=lambda s: ("unused", None))
            self.assertEqual(state.status, "blocked")
            self.assertEqual(harness.perform.call_count, 1)

    def test_generated_brief_cannot_select_an_unreviewed_validator(self):
        class BadBrief:
            def perform(self, state, action, output, kind):
                atomic_write(output, brief().model_dump_json())
                return {"success": True, "returncode": 0}
        with tempfile.TemporaryDirectory() as tmp:
            state, _ = run_research("Question", workspace=tmp, harness=BadBrief(),
                                    validators={}, selector=lambda s: ("unused", None))
            self.assertEqual(state.status, "failed")
            self.assertIsNone(state.brief)

    def test_unknown_validator_and_bad_reference_are_inconclusive(self):
        validator = SquareValidator()
        with tempfile.TemporaryDirectory() as tmp:
            state, _ = run_research("Find n", workspace=tmp, brief=brief(), harness=ScriptedHarness((3,)),
                                    validators={}, selector=lambda s: ("Check", None), max_steps=1)
            self.assertEqual(state.steps[0].validations[0].status, "inconclusive")
            with patch.object(validator, "validate", return_value=ValidationResult(
                criterion_id="square", claim="3 is a solution", status="supported", validator="square",
                kind="deterministic", evidence_ids=["invented"], reason="Trust me",
            )):
                state, _ = run_research("Find n", workspace=tmp, brief=brief(), harness=ScriptedHarness((3,)),
                                        validators={"square": validator}, selector=lambda s: ("Check", None), max_steps=1)
            self.assertEqual(state.steps[0].validations[0].status, "inconclusive")

    def test_system_one_uncertainty_and_errors_are_not_support(self):
        provider = Mock()
        validator = SystemOneValidator(provider)
        c, f = Criterion(id="c", description="Criterion"), Finding(criterion_id="c", claim="Claim")
        for values in ({"support": .95, "contradiction": .95}, {"support": float("nan"), "contradiction": 0}):
            provider.batch_noul.return_value = values
            self.assertEqual(validator.validate(c, f, []).status, "inconclusive")
        provider.batch_noul.side_effect = RuntimeError("offline")
        self.assertEqual(validator.validate(c, f, []).status, "inconclusive")

    def test_mcts_custom_evaluator_sees_proposals_and_aggregates_values(self):
        seen = []
        def score(goal, state):
            seen.append(state)
            return 9.0 if "experiment" in state else 2.0
        with tempfile.TemporaryDirectory() as tmp, patch("agent.mcts._propose_actions", return_value=["experiment", "guess"]), \
                patch("agent.mcts.get_action_priors", return_value={"experiment": .5, "guess": .5}), \
                patch("agent.mcts.discriminative_choose_best_action", side_effect=lambda g, s, cs, **kw: max(cs, key=lambda c: c.average_value)), \
                patch("agent.mcts.check_task_completion", side_effect=AssertionError("Completion must not run")):
            root = Node(state="Observed notebook")
            best, _ = run_mcts(root, "Investigate", iterations=0, expansion_depth=2, actions_per_node=2,
                               early_stop_noul=False, state_evaluator=score, log_dir=tmp)
            self.assertEqual(best.action_taken, "experiment")
            self.assertGreater(root.visits, 0)
            self.assertTrue(all("[Proposed action]" in s for s in seen))

    def test_research_scoring_does_not_reward_imagined_completion(self):
        provider = Mock()
        provider.score.return_value = 12
        self.assertEqual(ResearchScorer(provider)("Q", "Proposed experiment"), 10)
        self.assertIn("UNEXECUTED", provider.score.call_args.args[1])

    def test_budget_and_schema_constraints(self):
        with self.assertRaises(ValueError):
            ResearchSettings(width=8, depth=5)
        with self.assertRaises(ValueError):
            ResearchState(schema_version=2, run_id="x", query="Q", workspace=".")
        with self.assertRaises(ValueError):
            ResearchBrief(scope="Q", criteria=[Criterion(id="c", description="Q", required=False)])

    def test_cli_mock_and_resume_are_offline_and_json_clean(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp, patch("agent.research.runner.get_system_one_provider", side_effect=AssertionError("network")), \
                patch("agent.research.runner.get_executor_provider", side_effect=AssertionError("execution")):
            result = runner.invoke(app, ["research", "--query", "Open question", "--workspace", tmp,
                                         "--max-steps", "1", "--mock", "--json"])
            self.assertEqual(result.exit_code, 0, result.output)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "budget_exhausted")
            resumed = runner.invoke(app, ["research", "--resume", payload["checkpoint"], "--max-steps", "1", "--json"])
            self.assertEqual(resumed.exit_code, 0, resumed.output)
            self.assertEqual(json.loads(resumed.stdout)["steps"], 2)

    def test_cli_rejects_missing_query_and_oversized_tree(self):
        runner = CliRunner()
        for args in (["research", "--json"], ["research", "--query", "Q", "--width", "8", "--depth", "5", "--json"]):
            result = runner.invoke(app, args)
            self.assertNotEqual(result.exit_code, 0)
            self.assertEqual(json.loads(result.stdout)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
