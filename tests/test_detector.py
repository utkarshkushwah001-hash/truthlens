import unittest
from unittest.mock import patch

from PIL import Image

from backend import detector


class DetectorAdapterTests(unittest.TestCase):
    def tearDown(self):
        detector.reset_model_for_tests()

    def test_detector_is_opt_in(self):
        with patch.dict("os.environ", {"TRUTHLENS_DETECTOR": "off"}):
            self.assertFalse(detector.detector_enabled())
        with patch.dict("os.environ", {"TRUTHLENS_DETECTOR": "smaug"}):
            self.assertTrue(detector.detector_enabled())

    def test_returns_finite_probability_from_local_model(self):
        fake_model = lambda _image: 0.73
        with patch.object(detector, "_load_model", return_value=fake_model):
            score = detector.score_image(Image.new("RGB", (3, 3)))
        self.assertEqual(score, 0.73)

    def test_rejects_invalid_model_score(self):
        fake_model = lambda _image: float("nan")
        with patch.object(detector, "_load_model", return_value=fake_model):
            with self.assertRaisesRegex(detector.DetectorUnavailable, "invalid score"):
                detector.score_image(Image.new("RGB", (3, 3)))


if __name__ == "__main__":
    unittest.main()

