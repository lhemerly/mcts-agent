"""Tests for MCTS Agent primitives, node, logger, and search."""
import os
import tempfile
import unittest

# Ensure mock mode is active for all unit tests
os.environ["USE_MOCK_PRIMITIVES"] = "true"

from agent.node import Node
from agent.primitives import (
    batch_check_validity,
    get_action_priors,
    evaluate_state,
    select_action_count,
    check_task_completion,
    NOUL_VALIDITY_THRESHOLD,
    PRIME_ACTION_COUNTS,
)
from agent.logger import MCTSLogger
from agent.mcts import run_mcts, run_closed_loop_agent


class TestNode(unittest.TestCase):
    def test_node_creation_and_properties(self):
        root = Node(state="Initial state")
        self.assertEqual(root.state, "Initial state")
        self.assertIsNone(root.parent)
        self.assertTrue(root.is_leaf)
        self.assertEqual(root.visits, 0)
        self.assertEqual(root.average_value, 0.0)

        child = Node(
            state="Child state",
            parent=root,
            action_taken="Do something",
            prior_probability=0.5,
        )
        root.children.append(child)
        self.assertFalse(root.is_leaf)
        self.assertEqual(child.parent, root)
        self.assertEqual(child.action_taken, "Do something")

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

    def test_check_task_completion(self):
        is_done, conf = check_task_completion("goal", "state", mock=True)
        self.assertIsInstance(is_done, bool)
        self.assertIsInstance(conf, float)


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


class TestMCTSLoop(unittest.TestCase):
    def test_run_mcts_mock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Node(state="Test search state")
            best_node, log_path = run_mcts(
                root=root,
                goal="Achieve test goal",
                iterations=2,
                actions_per_node=2,
                early_stop_noul=False,
                log_dir=tmpdir,
            )
            self.assertIsNotNone(best_node)
            self.assertTrue(os.path.exists(log_path))
            self.assertGreater(root.visits, 0)
            self.assertGreater(len(root.children), 0)

    def test_closed_loop_agent_mock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_closed_loop_agent(
                goal="Minimal closed-loop goal",
                initial_state="Initial state",
                max_steps=1,
                iterations_per_step=1,
                actions_per_node=2,
                early_stop_noul=False,
                execute=False,
                log_dir=tmpdir,
            )
            self.assertEqual(summary["total_steps_run"], 1)
            self.assertIn("steps", summary)
            self.assertEqual(len(summary["steps"]), 1)


if __name__ == "__main__":
    unittest.main()
