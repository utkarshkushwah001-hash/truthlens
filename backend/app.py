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
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

if __package__:
    from .config import ALLOWED_FORMATS_BY_EXTENSION, MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES, allowed_origins
else:  # Support `python backend/app.py` from the project root on Windows.
    from config import ALLOWED_FORMATS_BY_EXTENSION, MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES, allowed_origins


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
CORS(app, resources={r"/analyze": {"origins": allowed_origins()}})


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_error):
    return jsonify({"error": "Image exceeds the 10 MB upload limit."}), 413


@app.errorhandler(HTTPException)
def handle_http_error(error: HTTPException):
    return jsonify({"error": error.description}), error.code or 500


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    app.logger.exception("Unhandled TruthLens API error")
    return jsonify({"error": "An unexpected server error occurred."}), 500


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
    expected = ALLOWED_FORMATS_BY_EXTENSION.get(extension)
    if expected is None:
        return jsonify({"error": "Unsupported file extension. Use PNG, JPG, JPEG or WEBP."}), 415

    expected_format, expected_mime = expected
    if uploaded.mimetype.lower() != expected_mime:
        return jsonify({"error": "File MIME type does not match its supported extension."}), 415

    raw = uploaded.stream.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        return jsonify({"error": "The uploaded file is empty."}), 400
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({"error": "Image exceeds the 10 MB upload limit."}), 413

    try:
        with Image.open(io.BytesIO(raw)) as image:
            actual_format = (image.format or "").upper()
            width, height = image.size
            if actual_format != expected_format:
                return jsonify({"error": "File content is not a supported raster image."}), 415
            if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                return jsonify({"error": "Image dimensions are invalid or exceed the 40-megapixel limit."}), 413
            image.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return jsonify({"error": "The file could not be decoded as a valid image."}), 400

    # Do not persist uploads. Metadata is not evidence that an image is real or AI-generated.
    return jsonify({
        "status": "inconclusive",
        "verdict": "Inconclusive",
        "message": "The image passed file validation. TruthLens cannot determine whether it is AI-generated or real because no validated detector is configured.",
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

