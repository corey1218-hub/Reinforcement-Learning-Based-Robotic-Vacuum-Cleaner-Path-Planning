"""Behavior checks for environment accounting, terminal updates, and replay."""
import unittest
import numpy as np
from vacuum_rl.environment import Layout, VacuumEnv, bedroom_layout
from vacuum_rl.evaluation import evaluate, nearest_target_route
from vacuum_rl.learning import TrainConfig, td_target, train


class EnvironmentTests(unittest.TestCase):
    def test_bedroom_geometry(self):
        layout = bedroom_layout()
        self.assertEqual((len(layout.obstacles), len(layout.free_cells)), (72, 72))
        self.assertEqual(len(VacuumEnv(layout).targets), 71)
        self.assertEqual(layout.reachable_cells(), set(layout.free_cells))

    def test_wall_and_obstacle_collisions_preserve_state(self):
        env = VacuumEnv()
        start = env.state
        for action, kind in [(1, "wall"), (2, "obstacle")]:
            state, reward, done, info = env.step(action)
            self.assertEqual((state, reward, done, info["collision"]), (start, -1., False, kind))

    def test_first_visit_and_revisit_rewards(self):
        env = VacuumEnv()
        _, reward, _, info = env.step(0)
        self.assertEqual((reward, info["cleaned"], env.remaining), (2., 1, 70))
        env.step(1)
        self.assertEqual(env.step(0)[1], -1.)

    def test_home_is_required_and_terminal_is_absorbing(self):
        env = VacuumEnv(Layout(1, 2, (0, 0), frozenset()))
        self.assertEqual(env.step(0)[1:3], (2., False))
        self.assertEqual(env.step(1)[1:3], (99., True))
        state = env.state
        self.assertEqual(env.step(0)[:3], (state, 0., True))

    def test_same_position_different_cleaning_history_is_different_state(self):
        env = VacuumEnv()
        initial = env.state
        env.step(0)
        env.step(1)
        self.assertEqual(initial[0], env.state[0])
        self.assertNotEqual(initial, env.state)

    def test_reset_restores_all_targets(self):
        env = VacuumEnv()
        initial = env.state
        env.step(0)
        self.assertEqual(env.reset(), initial)

    def test_disconnected_map_and_invalid_action_rejected(self):
        with self.assertRaises(ValueError):
            Layout(1, 3, (0, 0), frozenset({(0, 1)}))
        with self.assertRaises(ValueError):
            VacuumEnv().step(4)


class LearningTests(unittest.TestCase):
    def test_terminal_target_ignores_nonzero_next_value(self):
        self.assertEqual(td_target(10., 1000., .9, True), 10.)
        self.assertEqual(td_target(10., 1000., .9, False), 910.)

    def test_evaluation_does_not_insert_unknown_states(self):
        q = {}
        result, _, _ = evaluate(VacuumEnv(), q, max_steps=8)
        self.assertEqual(q, {})
        self.assertGreater(result["unseen_state_actions"], 0)
        self.assertFalse(result["success"])

    def test_step_cap_is_not_success(self):
        env = VacuumEnv(Layout(1, 2, (0, 0), frozenset()))
        result, _, _ = evaluate(env, max_steps=1, actions=[0])
        self.assertEqual(result["coverage"], 1.)
        self.assertFalse(result["success"])
        self.assertEqual(result["stop_reason"], "step_cap")

    def test_seed_reproduces_both_algorithms(self):
        layout = Layout(2, 2, (0, 0), frozenset())
        for algorithm in ["qlearning", "sarsa"]:
            config = TrainConfig(algorithm=algorithm, episodes=60, max_steps=20, seed=123)
            q1, h1 = train(VacuumEnv(layout), config)
            q2, h2 = train(VacuumEnv(layout), config)
            self.assertEqual(h1, h2)
            self.assertEqual(set(q1), set(q2))
            for state in q1:
                np.testing.assert_array_equal(q1[state], q2[state])

    def test_heuristic_route_covers_room_and_reward_balances(self):
        env = VacuumEnv()
        actions = nearest_target_route(env.layout)
        result, _, trace = evaluate(env, actions=actions, max_steps=1000)
        self.assertTrue(result["success"])
        self.assertEqual(result["cleaned_cells"], 71)
        self.assertEqual(result["wall_collisions"]+result["obstacle_collisions"], 0)
        self.assertEqual(result["revisit_moves"], result["steps"]-71)
        self.assertEqual(result["reward"], -result["steps"] + 3*71 + 100)
        for row in trace:
            self.assertEqual(row["reward"], row["move_reward"]+row["clean_reward"]+row["finish_reward"])

    def test_invalid_training_settings_rejected(self):
        for args in [{"episodes": 0}, {"alpha": 2.}, {"gamma": -1.}, {"max_steps": 0}]:
            with self.assertRaises(ValueError):
                TrainConfig(**args)


if __name__ == "__main__":
    unittest.main()
