import io
import unittest

from PIL import Image

from backend.app import app


def png_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (8, 6), color=(120, 80, 160)).save(stream, format="PNG")
    return stream.getvalue()


class TruthLensApiTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_reports_detector_not_configured(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["detector"], "not_configured")

    def test_valid_image_is_inconclusive_without_fake_score(self):
        response = self.client.post(
            "/analyze",
            data={"image": (io.BytesIO(png_bytes()), "sample.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["status"], "inconclusive")
        self.assertNotIn("score", body)
        self.assertEqual(body["checks"]["ai_image_detection"], "not_configured")
        self.assertEqual(body["metadata"]["width"], 8)

    def test_missing_upload_is_rejected(self):
        response = self.client.post("/analyze", data={}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_unsupported_extension_is_rejected(self):
        response = self.client.post(
            "/analyze",
            data={"image": (io.BytesIO(b"not an image"), "payload.exe")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 415)

    def test_fake_png_bytes_are_rejected(self):
        response = self.client.post(
            "/analyze",
            data={"image": (io.BytesIO(b"not really a png"), "fake.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
