"""TruthLens upload validation API.

This version validates image uploads but deliberately does not claim AI-image
classification: a validated detector has not yet been integrated.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _allowed_origins() -> list[str]:
    configured = os.getenv("TRUTHLENS_ALLOWED_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
CORS(app, resources={r"/analyze": {"origins": _allowed_origins()}})


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_error):
    return jsonify({"error": "Image exceeds the 10 MB upload limit."}), 413


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "detector": "not_configured"})


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Use the 'image' form field."}), 400

    uploaded = request.files["image"]
    if not uploaded.filename or not uploaded.filename.strip():
        return jsonify({"error": "Please choose an image file."}), 400

    extension = Path(uploaded.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported file extension. Use PNG, JPG, JPEG or WEBP."}), 415

    raw = uploaded.stream.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        return jsonify({"error": "The uploaded file is empty."}), 400
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({"error": "Image exceeds the 10 MB upload limit."}), 413

    try:
        with Image.open(io.BytesIO(raw)) as image:
            actual_format = (image.format or "").upper()
            width, height = image.size
            if actual_format not in ALLOWED_FORMATS:
                return jsonify({"error": "File content is not a supported raster image."}), 415
            if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                return jsonify({"error": "Image dimensions are invalid or exceed the 40-megapixel limit."}), 413
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return jsonify({"error": "The file could not be decoded as a valid image."}), 400

    # Do not persist uploads. Metadata is not evidence that an image is real or AI-generated.
    return jsonify({
        "status": "inconclusive",
        "verdict": "Inconclusive",
        "message": "The image passed basic validation, but no validated AI-image detector is configured yet.",
        "metadata": {
            "format": actual_format,
            "width": width,
            "height": height,
            "file_size_kb": round(len(raw) / 1024, 2),
        },
        "checks": {
            "upload_validation": "passed",
            "ai_image_detection": "not_configured",
            "provenance_verification": "not_configured",
        },
    })


if __name__ == "__main__":
    # Debug mode stays off. Use a production WSGI server when deploying.
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
