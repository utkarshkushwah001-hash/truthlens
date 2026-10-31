import csv
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_detector import evaluate, metrics, parse_rows, roc_auc


class DetectorEvaluationTests(unittest.TestCase):
    def test_metrics_report_perfect_ordering_and_group_breakdowns(self):
        rows = [
            {"label": "real", "score": 0.1, "generator": "none", "transformation": "clean"},
            {"label": "real", "score": 0.2, "generator": "none", "transformation": "jpeg"},
            {"label": "ai", "score": 0.8, "generator": "model-a", "transformation": "clean"},
            {"label": "ai", "score": 0.9, "generator": "model-b", "transformation": "jpeg"},
        ]
        report = evaluate(rows, 0.5)
        self.assertEqual(report["overall"]["balanced_accuracy"], 1.0)
        self.assertEqual(report["overall"]["roc_auc"], 1.0)
        self.assertEqual(report["overall"]["false_positive"], 0)
        self.assertEqual(report["overall"]["accuracy"], 1.0)
        self.assertEqual(report["overall"]["f1_ai"], 1.0)
        self.assertEqual(report["overall"]["confusion_matrix"]["actual_ai"]["predicted_ai"], 2)
        self.assertAlmostEqual(report["score_distributions"]["ai"]["median"], 0.85)
        self.assertIn("model-a", report["by_generator"])
        self.assertIn("jpeg", report["by_transformation"])

    def test_auc_counts_ties_as_half(self):
        rows = [
            {"label": "real", "score": 0.5},
            {"label": "ai", "score": 0.5},
        ]
        self.assertEqual(roc_auc(rows), 0.5)

    def test_zero_predicted_positives_has_undefined_precision_and_zero_f1(self):
        rows = [
            {"label": "real", "score": 0.1},
            {"label": "ai", "score": 0.4},
        ]
        report = metrics(rows, 0.5)
        self.assertIsNone(report["precision_ai"])
        self.assertEqual(report["f1_ai"], 0.0)

    def test_auc_is_undefined_if_a_class_is_missing(self):
        self.assertIsNone(roc_auc([{"label": "ai", "score": 0.8}]))

    def test_csv_validation_and_optional_group_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            path.write_text("label,score\nreal,0.1\nai,0.9\n", encoding="utf-8")
            rows = parse_rows(path)
            self.assertEqual(rows[0]["generator"], "unspecified")
            self.assertEqual(rows[1]["transformation"], "clean")

    def test_invalid_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            path.write_text("label,score\nreal,0.1\nai,1.7\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                parse_rows(path)

    def test_requires_both_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            path.write_text("label,score\nreal,0.1\nreal,0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least one real and one AI"):
                parse_rows(path)

    def test_malformed_rows_report_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            path.write_text("label,score\nreal,0.1,unexpected\nai,0.9\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Line 2: malformed CSV row"):
                parse_rows(path)


if __name__ == "__main__":
    unittest.main()

