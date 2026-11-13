from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from pilot.harness import PilotHarness, TASK_ID


ROOT = Path(__file__).resolve().parents[1]


class PilotScorerTests(unittest.TestCase):
    def test_real_candidate_passes_correctness_in_subprocess(self):
        with TemporaryDirectory() as temp:
            node_dir = Path(temp)
            shutil.copyfile(ROOT / "pilot/candidate.py", node_dir / "solution.py")
            scorer = PilotHarness(size=40, repetitions=3, measurements_path=node_dir / "measurements.jsonl")
            result = scorer.score(TASK_ID, node_dir, "dev", seed=123)
            self.assertTrue(result["ok"])
            self.assertTrue(result["correct"])
            self.assertGreater(result["raw"], 0)

    def test_wrong_output_is_rejected(self):
        with TemporaryDirectory() as temp:
            node_dir = Path(temp)
            (node_dir / "solution.py").write_text("def distance_matrix(points):\n    return []\n", encoding="utf-8")
            scorer = PilotHarness(size=40, repetitions=3, measurements_path=node_dir / "measurements.jsonl")
            result = scorer.score(TASK_ID, node_dir, "dev", seed=123)
            self.assertTrue(result["ok"])
            self.assertFalse(result["correct"])
            self.assertIn("behavior_change", result["cheat_flags"])


if __name__ == "__main__":
    unittest.main()
