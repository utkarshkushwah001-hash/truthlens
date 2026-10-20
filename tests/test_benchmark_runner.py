import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.run_detector_benchmark import score_images, write_scores


class FakeDetector:
    def predict_path(self, path):
        name = Path(path).name
        score = 0.9 if "synthetic" in name else 0.1
        return SimpleNamespace(as_dict=lambda: {"probability_ai": score})


class BadDetector:
    def predict_path(self, path):
        return {"score": 0.9}


class BenchmarkRunnerTests(unittest.TestCase):
    def test_scores_labeled_folder_and_preserves_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real" / "camera" / "clean"
            ai = root / "ai" / "generator-a" / "jpeg"
            real.mkdir(parents=True)
            ai.mkdir(parents=True)
            (real / "photo.jpg").write_bytes(b"test bytes")
            (ai / "synthetic.jpg").write_bytes(b"test bytes")
            rows = score_images(root, FakeDetector())
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["label"], "real")
            self.assertEqual(rows[0]["generator"], "camera")
            self.assertEqual(rows[0]["transformation"], "clean")
            self.assertEqual(rows[1]["label"], "ai")
            self.assertEqual(rows[1]["generator"], "generator-a")
            self.assertEqual(rows[1]["transformation"], "jpeg")
            self.assertEqual(rows[1]["score"], 0.9)

    def test_rejects_unknown_score_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, sub in (("real", "camera"), ("ai", "generator")):
                directory = root / label / sub
                directory.mkdir(parents=True)
                (directory / "sample.jpg").write_bytes(b"test bytes")
            with self.assertRaisesRegex(ValueError, "probability_ai"):
                score_images(root, BadDetector())

    def test_writes_evaluator_compatible_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "scores.csv"
            write_scores([
                {"label": "real", "score": 0.1, "generator": "camera", "transformation": "clean"},
                {"label": "ai", "score": 0.9, "generator": "gen-a", "transformation": "jpeg"},
            ], output)
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["label"] for row in rows], ["real", "ai"])
            self.assertEqual(float(rows[1]["score"]), 0.9)


if __name__ == "__main__":
    unittest.main()
