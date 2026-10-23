"""Check the saved reference trajectories against the current environment."""
import json
from pathlib import Path
import unittest
from vacuum_rl.environment import VacuumEnv
from vacuum_rl.evaluation import evaluate


class ReferenceTests(unittest.TestCase):
    def test_uploaded_reference_trajectories(self):
        refs = json.loads((Path(__file__).parent/"fixtures/reference_runs.json").read_text())
        for ref in refs:
            with self.subTest(algorithm=ref["algorithm"]):
                summary, _, _ = evaluate(VacuumEnv(), actions=ref["source_actions"], max_steps=500)
                self.assertTrue(summary["success"])
                self.assertEqual(summary["steps"], ref["steps"])
                self.assertEqual(summary["reward"], ref["reward"])


if __name__ == "__main__":
    unittest.main()
