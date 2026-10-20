import unittest

from tools.calibrate_detector import calibrate_and_test, choose_threshold


def rows(real_scores, ai_scores):
    output = []
    output.extend({"label": "real", "score": score, "generator": "camera", "transformation": "clean"} for score in real_scores)
    output.extend({"label": "ai", "score": score, "generator": "gen-a", "transformation": "clean"} for score in ai_scores)
    return output


class DetectorCalibrationTests(unittest.TestCase):
    def test_perfect_separation_selects_threshold_with_perfect_metrics(self):
        calibration = rows([0.1, 0.2, 0.3], [0.7, 0.8, 0.9])
        selected = choose_threshold(calibration, "balanced_accuracy")
        # The documented tie-break favors the higher threshold. Since scores
        # equal to the threshold are classified as AI, 0.7 is a valid optimum.
        self.assertGreater(selected["threshold"], 0.3)
        self.assertLessEqual(selected["threshold"], 0.7)
        self.assertEqual(selected["calibration_metrics"]["balanced_accuracy"], 1.0)

    def test_reports_calibration_and_held_out_metrics_separately(self):
        calibration = rows([0.1, 0.3, 0.4], [0.6, 0.7, 0.9])
        test = rows([0.2, 0.8], [0.75, 0.95])
        report = calibrate_and_test(calibration, test, "f1")
        self.assertIn("threshold", report["calibration"])
        self.assertEqual(report["held_out_test"]["overall"]["samples"], 4)
        self.assertIn("warning", report)

    def test_rejects_unknown_objective(self):
        with self.assertRaisesRegex(ValueError, "objective"):
            choose_threshold(rows([0.1], [0.9]), "accuracy")


if __name__ == "__main__":
    unittest.main()
