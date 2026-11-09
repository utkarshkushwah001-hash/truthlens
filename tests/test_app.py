import io
import unittest
from pathlib import Path

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
            data={"image": (io.BytesIO(png_bytes()), "sample.png", "image/png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["status"], "inconclusive")
        self.assertNotIn("score", body)
        self.assertEqual(body["checks"]["ai_image_detection"], "not_configured")
        self.assertEqual(body["metadata"]["width"], 8)

    def test_valid_upload_is_not_persisted(self):
        uploads = Path(__file__).resolve().parents[1] / "backend" / "uploads"
        before = set(uploads.iterdir()) if uploads.exists() else set()
        response = self.client.post(
            "/analyze",
            data={"image": (io.BytesIO(png_bytes()), "sample.png", "image/png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        after = set(uploads.iterdir()) if uploads.exists() else set()
        self.assertEqual(after, before)

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

    def test_mime_type_must_match_extension(self):
        response = self.client.post(
            "/analyze",
            data={"image": (io.BytesIO(png_bytes()), "sample.png", "image/jpeg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 415)

    def test_image_content_must_match_extension(self):
        response = self.client.post(
            "/analyze",
            data={"image": (io.BytesIO(png_bytes()), "sample.jpg", "image/jpeg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 415)

    def test_empty_upload_is_rejected(self):
        response = self.client.post(
            "/analyze",
            data={"image": (io.BytesIO(b""), "empty.png", "image/png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)

    def test_oversized_upload_is_rejected(self):
        from backend.app import MAX_UPLOAD_BYTES

        response = self.client.post(
            "/analyze",
            data={"image": (io.BytesIO(b"x" * (MAX_UPLOAD_BYTES + 1)), "large.png", "image/png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 413)

    def test_fake_png_bytes_are_rejected(self):
        response = self.client.post(
            "/analyze",
            data={"image": (io.BytesIO(b"not really a png"), "fake.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)

    def test_api_method_errors_are_json(self):
        response = self.client.get("/analyze")
        self.assertEqual(response.status_code, 405)
        self.assertIn("error", response.get_json())

    def test_cors_only_allows_configured_local_origins(self):
        trusted = self.client.options(
            "/analyze",
            headers={"Origin": "http://localhost:5500", "Access-Control-Request-Method": "POST"},
        )
        untrusted = self.client.options(
            "/analyze",
            headers={"Origin": "https://untrusted.example", "Access-Control-Request-Method": "POST"},
        )
        self.assertEqual(trusted.headers.get("Access-Control-Allow-Origin"), "http://localhost:5500")
        self.assertIsNone(untrusted.headers.get("Access-Control-Allow-Origin"))


if __name__ == "__main__":
    unittest.main()

