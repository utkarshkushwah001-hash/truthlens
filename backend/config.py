"""Runtime configuration for the TruthLens API."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_FORMATS_BY_EXTENSION = {
    ".jpg": ("JPEG", "image/jpeg"),
    ".jpeg": ("JPEG", "image/jpeg"),
    ".png": ("PNG", "image/png"),
    ".webp": ("WEBP", "image/webp"),
}


def allowed_origins() -> list[str]:
    """Return exact HTTP(S) frontend origins; wildcard CORS is rejected."""
    configured = os.getenv(
        "TRUTHLENS_ALLOWED_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500",
    )
    origins = [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    if not origins or "*" in origins:
        raise ValueError("TRUTHLENS_ALLOWED_ORIGINS must contain exact HTTP(S) origins, not '*'.")
    for origin in origins:
        parsed = urlsplit(origin)
        try:
            parsed.port
        except ValueError as error:
            raise ValueError(f"Invalid frontend origin in TRUTHLENS_ALLOWED_ORIGINS: {origin!r}") from error
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"Invalid frontend origin in TRUTHLENS_ALLOWED_ORIGINS: {origin!r}")
    return origins

