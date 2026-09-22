"""Tests for MCTS Agent primitives, node, logger, and search."""
import os
import tempfile
import unittest

# Ensure mock mode is active for all unit tests
os.environ["USE_MOCK_PRIMITIVES"] = "true"

import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent.node import Node
from agent.primitives import (
    batch_check_validity,
    get_action_priors,
    evaluate_state,
    select_action_count,
    discriminative_choose_best_action,
    check_task_completion,
    NOUL_VALIDITY_THRESHOLD,
    PRIME_ACTION_COUNTS,
)
from agent.logger import MCTSLogger
from agent.mcts import (
    run_mcts,
    run_closed_loop_agent,
    _deep_expand,
    _reroot,
    _recombine_paths,
    _rescore_leaves,
    _prune_low_value_children,
    _propose_actions,
)


# ── Node Tests ─────────────────────────────────────────────────────────────────

class TestNode(unittest.TestCase):
    def test_node_creation_and_properties(self):
        root = Node(state="Initial state")
        self.assertEqual(root.state, "Initial state")
        self.assertIsNone(root.parent)
        self.assertTrue(root.is_leaf)
        self.assertEqual(root.visits, 0)
        self.assertEqual(root.average_value, 0.0)
        self.assertEqual(root.depth, 0)

        child = Node(
            state="Child state",
            parent=root,
            action_taken="Do something",
            prior_probability=0.5,
            depth=1,
        )
        root.children.append(child)
        self.assertFalse(root.is_leaf)
        self.assertEqual(child.parent, root)
        self.assertEqual(child.action_taken, "Do something")
        self.assertEqual(child.depth, 1)

    def test_puct_score(self):
        root = Node(state="Root")
        root.visits = 10
        root.value_sum = 50.0  # avg = 5.0

        child = Node(state="Child", parent=root, prior_probability=0.4)
        # Unvisited child should use FPU (First-Play Urgency: parent's average value)
        score_unvisited = child.puct_score(exploration_constant=1.4)
        self.assertGreater(score_unvisited, 0.0)

        child.visits = 2
        child.value_sum = 16.0  # avg = 8.0
        score_visited = child.puct_score(exploration_constant=1.4)
        self.assertGreater(score_visited, 0.0)

    def test_subtree_leaves_single_node(self):
        root = Node(state="Root")
        leaves = root.subtree_leaves()
        self.assertEqual(len(leaves), 1)
        self.assertIs(leaves[0], root)

    def test_subtree_leaves_tree(self):
        root = Node(state="Root")
        c1 = Node(state="C1", parent=root, action_taken="A1", depth=1)
        c2 = Node(state="C2", parent=root, action_taken="A2", depth=1)
        gc1 = Node(state="GC1", parent=c1, action_taken="A1a", depth=2)
        gc2 = Node(state="GC2", parent=c1, action_taken="A1b", depth=2)
        root.children = [c1, c2]
        c1.children = [gc1, gc2]

        leaves = root.subtree_leaves()
        self.assertEqual(len(leaves), 3)
        self.assertIn(gc1, leaves)
        self.assertIn(gc2, leaves)
        self.assertIn(c2, leaves)

    def test_collect_actions_by_depth(self):
        root = Node(state="Root", depth=0)
        c1 = Node(state="C1", parent=root, action_taken="Step A", depth=1)
        c2 = Node(state="C2", parent=root, action_taken="Step B", depth=1)
        gc1 = Node(state="GC1", parent=c1, action_taken="Step C", depth=2)
        root.children = [c1, c2]
        c1.children = [gc1]

        vocab = root.collect_actions_by_depth()
        self.assertIn(1, vocab)
        self.assertIn(2, vocab)
        self.assertIn("Step A", vocab[1])
        self.assertIn("Step B", vocab[1])
        self.assertIn("Step C", vocab[2])

    def test_all_actions_flat(self):
        root = Node(state="Root", depth=0)
        c1 = Node(state="C1", parent=root, action_taken="Action X", depth=1)
        c2 = Node(state="C2", parent=root, action_taken="Action Y", depth=1)
        root.children = [c1, c2]

        actions = root.all_actions_flat()
        self.assertIn("Action X", actions)
        self.assertIn("Action Y", actions)
        self.assertEqual(len(actions), 2)


# ── Primitives Tests ───────────────────────────────────────────────────────────

class TestPrimitivesMock(unittest.TestCase):
    def test_batch_check_validity(self):
        actions = ["Action A", "Action B", "Action C"]
        results = batch_check_validity("state context", actions, mock=True)
        self.assertEqual(len(results), 3)
        for a in actions:
            self.assertIn(a, results)
            is_valid, conf = results[a]
            self.assertIsInstance(is_valid, bool)
            self.assertIsInstance(conf, float)
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)

    def test_get_action_priors(self):
        actions = ["Action 1", "Action 2", "Action 3"]
        priors = get_action_priors("state", actions, mock=True)
        self.assertEqual(len(priors), 3)
        self.assertAlmostEqual(sum(priors.values()), 1.0, places=4)

    def test_evaluate_state(self):
        score = evaluate_state("Goal description", "Simulated state", mock=True)
        self.assertGreaterEqual(score, 1.0)
        self.assertLessEqual(score, 10.0)

    def test_select_action_count(self):
        count = select_action_count("state", "goal", mock=True)
        self.assertIn(count, PRIME_ACTION_COUNTS)

    def test_discriminative_choose_best_action(self):
        root = Node(state="Root state")
        child1 = Node(state="Child 1", parent=root, action_taken="Action 1")
        child1.visits = 3
        child1.value_sum = 24.0  # avg 8.0
        child2 = Node(state="Child 2", parent=root, action_taken="Action 2")
        child2.visits = 5
        child2.value_sum = 45.0  # avg 9.0
        chosen = discriminative_choose_best_action("Goal", "State", [child1, child2], mock=True)
        self.assertEqual(chosen.action_taken, "Action 2")

    def test_check_task_completion(self):
        is_done, conf = check_task_completion("goal", "state", mock=True)
        self.assertIsInstance(is_done, bool)
        self.assertIsInstance(conf, float)


# ── Logger Tests ───────────────────────────────────────────────────────────────

class TestLogger(unittest.TestCase):
    def test_logger_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MCTSLogger(
                goal="Test Goal",
                initial_state="Init",
                iterations=2,
                log_dir=tmpdir,
            )
            root = Node(state="Init")
            logger.emit_init(root)
            logger.emit_iteration_start(1)
            logger.emit_select([root])
            logger.emit_candidates(["action 1"])
            logger.emit_noul("action 1", True, 0.95)
            logger.emit_iteration_end(1)
            path = logger.save()

            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Test Goal", content)
            self.assertIn("action 1", content)


# ── Deep Expand Tests ──────────────────────────────────────────────────────────

class TestDeepExpand(unittest.TestCase):
    def _make_logger(self, tmpdir):
        return MCTSLogger(
            goal="test goal",
            initial_state="state",
            iterations=0,
            log_dir=tmpdir,
        )

    def test_deep_expand_width2_depth1(self):
        """Width=2, depth=1 → at most 2 children, all are leaves with scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = self._make_logger(tmpdir)
            root = Node(state="Root state", depth=0)
            logger.emit_init(root)
            _deep_expand(root, "Goal", width=2, remaining_depth=1, logger=logger)

            # Every proposed action is kept for search.
            self.assertGreater(len(root.children), 0)
            self.assertLessEqual(len(root.children), 2)
            for child in root.children:
                self.assertTrue(child.is_leaf)
                self.assertGreater(child.visits, 0)
                self.assertGreater(child.average_value, 0.0)
                self.assertEqual(child.depth, 1)

    def test_deep_expand_width2_depth2(self):
        """Width=2, depth=2 → at least 1 child, at least 1 leaf."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = self._make_logger(tmpdir)
            root = Node(state="Root", depth=0)
            logger.emit_init(root)
            _deep_expand(root, "Goal", width=2, remaining_depth=2, logger=logger)

            # Every proposed action is kept for search.
            self.assertGreater(len(root.children), 0)
            leaves = root.subtree_leaves()
            # At most 4 leaves (2^2), at least 1
            self.assertGreaterEqual(len(leaves), 1)
            self.assertLessEqual(len(leaves), 4)
            for leaf in leaves:
                self.assertEqual(leaf.depth, 2)
                self.assertGreater(leaf.visits, 0)

    def test_low_action_confidence_does_not_prune_initial_tree(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = self._make_logger(tmpdir)
            root = Node(state="Current state")
            with patch("agent.mcts._propose_actions", return_value=["First small step"]), patch(
                "agent.primitives.batch_check_validity", side_effect=AssertionError("unexpected gate")
            ):
                _deep_expand(root, "Goal", width=1, remaining_depth=1, logger=logger)
        self.assertEqual([child.action_taken for child in root.children], ["First small step"])

    def test_deep_expand_depth3_leaf_count(self):
        """Width=3, depth=3 materializes proposed paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = self._make_logger(tmpdir)
            root = Node(state="Root", depth=0)
            logger.emit_init(root)
            _deep_expand(root, "Goal", width=3, remaining_depth=3, logger=logger)

            leaves = root.subtree_leaves()
            self.assertGreater(len(leaves), 0)
            self.assertLessEqual(len(leaves), 27)

    def test_deep_expand_explored_actions_passed(self):
        """explored_actions are forwarded; mock planner avoids them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = self._make_logger(tmpdir)
            root = Node(state="Root", depth=0)
            logger.emit_init(root)
            explored = ["Search for relevant documentation online"]
            _deep_expand(root, "Goal", width=2, remaining_depth=1, logger=logger,
                         explored_actions=explored)
            # All proposed actions should be different from the explored one
            for child in root.children:
                self.assertNotEqual(child.action_taken, explored[0])

    def test_deep_expand_shares_explored_actions_across_siblings(self):
        from unittest.mock import patch

        seen = []

        def propose(state, goal, n, *, explored_actions, config):
            seen.append((state, list(explored_actions)))
            if state == "Root":
                return ["Step A", "Step B"]
            if state.endswith("Step A"):
                return ["Step C"]
            self.assertIn("Step C", explored_actions)
            return ["Step D"]

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agent.mcts._propose_actions", side_effect=propose
        ):
            root = Node(state="Root")
            _deep_expand(root, "Goal", width=2, remaining_depth=2, logger=self._make_logger(tmpdir))
        self.assertEqual([child.action_taken for child in root.children], ["Step A", "Step B"])
        self.assertIn("Step C", seen[-1][1])


# ── Tree Reuse Tests ───────────────────────────────────────────────────────────

class TestTreeReuse(unittest.TestCase):
    def _build_simple_tree(self) -> Node:
        """Build a simple 2-wide 2-deep tree with scores for testing."""
        root = Node(state="Root", depth=0)
        c1 = Node(state="Root\n[Action taken]: Step A", parent=root, action_taken="Step A", depth=1)
        c2 = Node(state="Root\n[Action taken]: Step B", parent=root, action_taken="Step B", depth=1)
        gc1 = Node(state="...\n[Action taken]: Step A1", parent=c1, action_taken="Step A1", depth=2)
        gc2 = Node(state="...\n[Action taken]: Step A2", parent=c1, action_taken="Step A2", depth=2)
        gc3 = Node(state="...\n[Action taken]: Step B1", parent=c2, action_taken="Step B1", depth=2)
        gc4 = Node(state="...\n[Action taken]: Step B2", parent=c2, action_taken="Step B2", depth=2)

        for node in [c1, c2, gc1, gc2, gc3, gc4]:
            node.visits = 1
            node.value_sum = 7.0

        root.children = [c1, c2]
        c1.children = [gc1, gc2]
        c2.children = [gc3, gc4]
        return root

    def test_reroot_detaches_from_parent(self):
        root = self._build_simple_tree()
        chosen = root.children[0]  # Step A
        new_root = _reroot(chosen, "Observation: file created.")
        self.assertIsNone(new_root.parent)
        self.assertEqual(new_root.depth, 0)
        self.assertIn("Observation: file created.", new_root.state)

    def test_reroot_shifts_child_depths(self):
        root = self._build_simple_tree()
        chosen = root.children[0]  # Step A, depth 1
        new_root = _reroot(chosen, "obs")
        self.assertEqual(new_root.depth, 0)
        for child in new_root.children:
            self.assertEqual(child.depth, 1)

    def test_reroot_rebases_surviving_states_on_observation(self):
        root = self._build_simple_tree()
        new_root = _reroot(root.children[0], "created file", grounded_state="Observed workspace")
        self.assertEqual(new_root.state, "Observed workspace")
        for child in new_root.children:
            self.assertEqual(child.state, f"Observed workspace\n[Action taken]: {child.action_taken}")

    def test_collect_vocab_by_depth(self):
        root = self._build_simple_tree()
        vocab = root.collect_actions_by_depth()
        self.assertIn(1, vocab)
        self.assertIn(2, vocab)
        self.assertIn("Step A", vocab[1])
        self.assertIn("Step B", vocab[1])
        self.assertIn("Step B1", vocab[2])

    def test_recombine_paths_adds_nodes(self):
        root = self._build_simple_tree()
        vocab = root.collect_actions_by_depth()

        # Reroot at Step A
        new_root = _reroot(root.children[0], "obs")
        initial_leaves = len(new_root.subtree_leaves())

        n_added = _recombine_paths(new_root, vocab, n_paths=4, path_depth=1)
        self.assertGreater(n_added, 0)
        self.assertGreater(len(new_root.subtree_leaves()), initial_leaves)

    def test_recombine_paths_does_not_duplicate_existing_path(self):
        from unittest.mock import patch
        root = self._build_simple_tree()
        new_root = _reroot(root.children[0], "obs")
        with patch("agent.mcts.random.choice", return_value="Step A1"):
            added = _recombine_paths(new_root, {2: ["Step A1"]}, n_paths=4, path_depth=1)
        self.assertEqual(added, 0)
        self.assertEqual(len(new_root.children), 2)

    def test_rescore_leaves_updates_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MCTSLogger(
                goal="g", initial_state="s", iterations=0, log_dir=tmpdir
            )
            root = self._build_simple_tree()
            logger.emit_init(root)

            # Forcefully zero out leaf scores to confirm rescore updates them
            for leaf in root.subtree_leaves():
                leaf.value_sum = 0.0
                leaf.visits = 0

            scores = _rescore_leaves(root, "goal", logger)
            self.assertGreater(len(scores), 0)
            for score in scores.values():
                self.assertGreaterEqual(score, 1.0)
                self.assertLessEqual(score, 10.0)
            # All leaf visits should be 1 after rescore
            for leaf in root.subtree_leaves():
                self.assertGreaterEqual(leaf.visits, 1)

    def test_prune_low_value_children(self):
        root = self._build_simple_tree()
        # Lower one branch's score below threshold
        root.children[1].value_sum = 1.0  # avg = 1.0 < 6.0 threshold
        root.children[0].value_sum = 8.0  # avg = 8.0 ≥ 6.0

        _prune_low_value_children(root, threshold=6.0)
        self.assertEqual(len(root.children), 1)
        self.assertEqual(root.children[0].action_taken, "Step A")

    def test_prune_keeps_unvisited(self):
        """Unvisited nodes (average_value=0.0) must not be pruned — they may be new recombinations."""
        root = Node(state="Root", depth=0)
        visited_bad = Node(state="...", parent=root, action_taken="Bad", depth=1)
        visited_bad.visits = 1
        visited_bad.value_sum = 2.0  # below threshold
        unvisited = Node(state="...", parent=root, action_taken="New", depth=1)
        # unvisited.visits == 0 by default
        root.children = [visited_bad, unvisited]

        _prune_low_value_children(root, threshold=6.0)
        self.assertEqual(len(root.children), 1)
        self.assertEqual(root.children[0].action_taken, "New")


# ── Full MCTS Loop Tests ───────────────────────────────────────────────────────

class TestMCTSLoop(unittest.TestCase):
    def test_reused_tree_selects_without_proposing_actions(self):
        from unittest.mock import patch

        root = Node(state="Observed state")
        child = Node(
            state="Observed state\n[Action taken]: Existing action",
            parent=root,
            action_taken="Existing action",
            depth=1,
            visits=1,
            value_sum=8.0,
        )
        root.children.append(child)
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agent.mcts._propose_actions", side_effect=AssertionError("unexpected proposal")
        ):
            best, _ = run_mcts(
                root, "Goal", iterations=2, reuse_tree=True,
                early_stop_noul=False, log_dir=tmpdir,
            )
        self.assertIs(best, child)
        self.assertEqual(len(root.children), 1)
        self.assertEqual(child.visits, 1)

    def test_search_stops_when_planner_returns_no_actions(self):
        from unittest.mock import patch

        root = Node(state="Current state")
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agent.mcts._propose_actions", return_value=[]
        ):
            best, _ = run_mcts(root, "Goal", iterations=10, log_dir=tmpdir)
        self.assertIs(best, root)
        self.assertEqual(root.visits, 1)

    def test_no_planned_action_has_distinct_stop_reason(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agent.mcts._propose_actions", return_value=[]
        ):
            summary = run_closed_loop_agent(
                "Goal", "Initial", max_steps=2, iterations_per_step=10,
                execute=False, log_dir=tmpdir, workspace_dir=tmpdir,
            )
        self.assertEqual(summary["stop_reason"], "no_action_generated")
        self.assertEqual(summary["total_steps_run"], 0)

    def test_initial_state_includes_workspace_inventory(self):
        from agent.mcts import _workspace_inventory

        with tempfile.TemporaryDirectory() as tmpdir:
            os.mkdir(os.path.join(tmpdir, "mcts-agent"))
            with open(os.path.join(tmpdir, "visualizer.html"), "w", encoding="utf-8") as handle:
                handle.write("<html></html>")
            self.assertIn("mcts-agent/", _workspace_inventory(tmpdir))
            self.assertIn("visualizer.html", _workspace_inventory(tmpdir))

    def test_run_mcts_mock_width2_depth1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Node(state="Test search state")
            best_node, log_path = run_mcts(
                root=root,
                goal="Achieve test goal",
                iterations=0,
                actions_per_node=2,
                expansion_depth=1,
                early_stop_noul=False,
                log_dir=tmpdir,
            )
            self.assertIsNotNone(best_node)
            self.assertTrue(os.path.exists(log_path))
            # Mock planner supplies immediate actions.
            self.assertGreater(len(root.children), 0)
            self.assertLessEqual(len(root.children), 2)

    def test_first_step_is_chosen_even_when_every_path_scores_low(self):
        from unittest.mock import patch

        root = Node(state="Starting state")
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agent.mcts._propose_actions",
            return_value=["Inspect existing HTML", "Inspect tree log", "Create a sketch"],
        ), patch("agent.mcts.evaluate_state", return_value=1.1):
            best, _ = run_mcts(
                root, "Long horizon goal", iterations=0,
                actions_per_node=3, expansion_depth=1,
                early_stop_noul=False, log_dir=tmpdir,
            )
        self.assertIn(best.action_taken, [child.action_taken for child in root.children])
        self.assertEqual(len(root.children), 3)

    def test_run_mcts_deep_tree_leaf_count(self):
        """width=2, depth=2 → up to 4 leaf paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Node(state="Test")
            run_mcts(
                root=root,
                goal="Goal",
                iterations=0,
                actions_per_node=2,
                expansion_depth=2,
                early_stop_noul=False,
                log_dir=tmpdir,
            )
            leaves = root.subtree_leaves()
            self.assertGreater(len(leaves), 0)
            self.assertLessEqual(len(leaves), 4)

    def test_run_mcts_with_puct_iterations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Node(state="Test multi-step state")
            best_node, log_path = run_mcts(
                root=root,
                goal="Achieve multi-step goal",
                iterations=3,
                actions_per_node=2,
                expansion_depth=2,
                early_stop_noul=False,
                log_dir=tmpdir,
            )
            self.assertIsNotNone(best_node)
            self.assertTrue(os.path.exists(log_path))
            self.assertGreater(root.visits, 0)
            self.assertGreater(len(root.children), 0)

    def test_run_mcts_backward_compat_simulation_depth(self):
        """simulation_depth is still accepted as alias for expansion_depth."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Node(state="State")
            best_node, log_path = run_mcts(
                root=root,
                goal="Goal",
                iterations=0,
                actions_per_node=2,
                simulation_depth=2,
                early_stop_noul=False,
                log_dir=tmpdir,
            )
            self.assertIsNotNone(best_node)
            # 2-wide, 2-deep = at most 4 leaves.
            leaves = root.subtree_leaves()
            self.assertGreater(len(leaves), 0)
            self.assertLessEqual(len(leaves), 4)

    def test_closed_loop_agent_mock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_closed_loop_agent(
                goal="Minimal closed-loop goal",
                initial_state="Initial state",
                max_steps=1,
                iterations_per_step=0,
                actions_per_node=2,
                expansion_depth=1,
                early_stop_noul=False,
                execute=False,
                log_dir=tmpdir,
            )
            self.assertEqual(summary["total_steps_run"], 1)
            self.assertIn("steps", summary)
            self.assertEqual(len(summary["steps"]), 1)

    def test_closed_loop_agent_two_steps_with_tree_reuse(self):
        """Two steps: second step should attempt tree reuse."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from agent.config import AgentConfig
            cfg = AgentConfig(
                planner_provider="mock",
                executor_provider="mock",
                tree_reuse_enabled=True,
                reuse_score_threshold=0.0,  # always reuse for this test
                n_recombined_paths=4,
                expansion_width=2,
                expansion_depth=2,
            )
            from unittest.mock import patch
            with patch("agent.mcts._recombine_paths", side_effect=AssertionError("unneeded recombination")):
                summary = run_closed_loop_agent(
                    goal="Two-step goal",
                    initial_state="Initial",
                    max_steps=2,
                    iterations_per_step=0,
                    actions_per_node=2,
                    expansion_depth=2,
                    early_stop_noul=False,
                    execute=False,
                    log_dir=tmpdir,
                    config=cfg,
                )
            self.assertEqual(summary["total_steps_run"], 2)
            # Second step should not have triggered a scramble (threshold=0.0)
            self.assertFalse(summary["steps"][1].get("scramble_triggered", True))

    def test_review_does_not_equate_exit_zero_with_action_completion(self):
        from unittest.mock import patch
        from agent.mcts import review_action

        result = {"success": True, "returncode": 0, "stdout": "", "command": "npm install react"}
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agent.mcts.check_action_execution", return_value=(False, 0.1)
        ):
            observation = review_action("Implement SVG animation", result, tmpdir)
        self.assertFalse(result["verified"])
        self.assertIn("action not verified", observation)
        self.assertIn("npm install react", observation)

    def test_execution_reports_files_changed_by_command(self):
        from unittest.mock import patch
        from agent.mcts import execute_single_action

        class WritingHarness:
            def execute_action(self, action, goal, workspace_dir):
                with open(os.path.join(workspace_dir, "artifact.txt"), "w", encoding="utf-8") as handle:
                    handle.write("content")
                return {"success": True, "stdout": "done", "stderr": "", "returncode": 0}

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("agent.mcts.get_executor_provider", return_value=WritingHarness()):
                result = execute_single_action(
                    action="Create artifact", goal="Create artifact", workspace_dir=tmpdir,
                )
        self.assertTrue(result["success"])
        self.assertEqual(result["changed_files"], ["artifact.txt"])

    def test_unverified_execution_does_not_reuse_hypothetical_tree(self):
        from unittest.mock import patch
        from agent.config import AgentConfig

        cfg = AgentConfig(planner_provider="mock", executor_provider="mock", expansion_depth=2)
        failed = {"success": False, "returncode": 1, "stderr": "command failed", "stdout": ""}
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agent.mcts.execute_single_action", return_value=failed
        ), patch("agent.mcts._reroot", side_effect=AssertionError("failed action rerooted")):
            summary = run_closed_loop_agent(
                goal="Goal", initial_state="Initial", max_steps=2,
                iterations_per_step=0, actions_per_node=2,
                early_stop_noul=False, log_dir=tmpdir, config=cfg,
            )
        self.assertEqual(summary["total_steps_run"], 2)
        self.assertFalse(summary["steps"][0]["execution"]["verified"])

    def test_closed_loop_agent_scramble_when_all_low_score(self):
        """With threshold=10.0 (impossible to meet), scramble is always triggered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from agent.config import AgentConfig
            cfg = AgentConfig(
                planner_provider="mock",
                executor_provider="mock",
                tree_reuse_enabled=True,
                reuse_score_threshold=10.1,  # impossible threshold → always scramble
                n_recombined_paths=4,
                expansion_width=2,
                expansion_depth=1,
            )
            summary = run_closed_loop_agent(
                goal="Scramble test goal",
                initial_state="Initial",
                max_steps=2,
                iterations_per_step=0,
                actions_per_node=2,
                expansion_depth=1,
                early_stop_noul=False,
                execute=False,
                log_dir=tmpdir,
                config=cfg,
            )
            self.assertEqual(summary["total_steps_run"], 2)
            self.assertTrue(summary["steps"][1].get("scramble_triggered", False))


# ── Propose Actions Tests ──────────────────────────────────────────────────────

class TestProposeActions(unittest.TestCase):
    def test_propose_actions_mock_mode(self):
        from unittest.mock import patch
        with patch("agent.mcts._is_mock_llm", return_value=True):
            actions = _propose_actions("Current state", "My Goal", n=3)
            self.assertEqual(len(actions), 3)

    def test_propose_actions_respects_explored_actions(self):
        """In mock mode, explored_actions are excluded from proposals."""
        from agent.providers import MockPlannerProvider
        provider = MockPlannerProvider()
        # Exclude 7 of 8 pool entries
        from agent.providers import _MOCK_ACTION_POOL
        explored = _MOCK_ACTION_POOL[:7]
        actions = provider.propose_actions("state", "goal", n=1, explored_actions=explored)
        self.assertEqual(len(actions), 1)
        self.assertNotIn(actions[0], explored)

    def test_propose_actions_iterative_calls_and_model(self):
        from unittest.mock import patch, MagicMock
        with patch.dict("os.environ", {
            "MCTS_PLANNER_PROVIDER": "agy",
            "USE_MOCK_PRIMITIVES": "false",
            "AGY_PROPOSAL_MODEL": "custom-proposal-model",
        }), patch("agent.providers.subprocess.run") as mock_run:

            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="1. First creative strategy\n", stderr=""),
                MagicMock(returncode=0, stdout="Second novel angle\n", stderr=""),
                MagicMock(returncode=0, stdout="Third alternative method\n", stderr=""),
            ]

            actions = _propose_actions("Sample state", "Sample goal", n=3)

            self.assertEqual(len(actions), 3)
            self.assertEqual(actions, [
                "First creative strategy",
                "Second novel angle",
                "Third alternative method",
            ])
            self.assertEqual(mock_run.call_count, 3)

            first_cmd = mock_run.call_args_list[0][0][0]
            self.assertEqual(first_cmd[0], "agy")
            self.assertEqual(first_cmd[1], "--model")
            self.assertEqual(first_cmd[2], "custom-proposal-model")
            first_prompt = first_cmd[4]
            self.assertIn("(No actions proposed yet for this expansion)", first_prompt)
            self.assertIn("ONE small, atomic next action", first_prompt)
            self.assertIn("two separate actions", first_prompt)

            second_cmd = mock_run.call_args_list[1][0][0]
            second_prompt = second_cmd[4]
            self.assertIn("First creative strategy", second_prompt)

            third_cmd = mock_run.call_args_list[2][0][0]
            third_prompt = third_cmd[4]
            self.assertIn("First creative strategy", third_prompt)
            self.assertIn("Second novel angle", third_prompt)

    def test_propose_actions_includes_explored_in_agy_prompt(self):
        """explored_actions appear in the AGY prompt as 'DO NOT reproduce' section."""
        from unittest.mock import patch, MagicMock
        with patch.dict("os.environ", {
            "MCTS_PLANNER_PROVIDER": "agy",
            "USE_MOCK_PRIMITIVES": "false",
        }), patch("agent.providers.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="A brand new action\n", stderr=""
            )
            _propose_actions(
                "state", "goal", n=1,
                explored_actions=["Already tried this", "And this"],
            )
            prompt_used = mock_run.call_args_list[0][0][0][4]
            self.assertIn("Already tried this", prompt_used)
            self.assertIn("And this", prompt_used)
            self.assertIn("DO NOT reproduce", prompt_used)


if __name__ == "__main__":
    unittest.main()
