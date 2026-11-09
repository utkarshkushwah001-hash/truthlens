"""Optional local Smaug AI-image detector adapter."""
from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
import threading
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "abacusai/Smaug-AI-Image-Detector"
DEFAULT_MODEL_REVISION = "c7c9595"
_load_lock = threading.Lock()
_model: Any | None = None
_model_error: Exception | None = None


class DetectorUnavailable(RuntimeError):
    """Raised when the configured model cannot be loaded or run."""


def detector_enabled() -> bool:
    return os.getenv("TRUTHLENS_DETECTOR", "off").strip().lower() == "smaug"


def _friendly_error(error: Exception) -> str:
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        return (
            "The Smaug detector runtime is missing. Install backend/detector-requirements.txt, "
            "set TRUTHLENS_DETECTOR=smaug, and restart the API."
        )
    return (
        "The Smaug detector could not load its model. Check internet access for the first model "
        "download, available disk space, and the configured model revision, then restart the API."
    )


def _load_model() -> Any:
    global _model, _model_error
    if _model is not None:
        return _model
    if _model_error is not None:
        raise DetectorUnavailable(_friendly_error(_model_error)) from _model_error

    with _load_lock:
        if _model is not None:
            return _model
        if _model_error is not None:
            raise DetectorUnavailable(_friendly_error(_model_error)) from _model_error
        try:
            from huggingface_hub import snapshot_download

            if importlib.util.find_spec("sentence_transformers") is None:
                raise ModuleNotFoundError("sentence_transformers")

            revision = os.getenv("TRUTHLENS_DETECTOR_REVISION", DEFAULT_MODEL_REVISION)
            model_path = Path(snapshot_download(MODEL_ID, revision=revision))
            model_path_text = str(model_path)
            if model_path_text not in sys.path:
                sys.path.insert(0, model_path_text)
            module = importlib.import_module("smaug_image_detect")
            _model = module.SmaugImageDetect.from_pretrained(model_path_text)
            return _model
        except Exception as error:
            _model_error = error
            raise DetectorUnavailable(_friendly_error(error)) from error


def score_image(image: Image.Image) -> float:
    """Return the model's raw AI score for a decoded image."""
    model = _load_model()
    try:
        score = float(model(image.convert("RGB")))
    except Exception as error:
        raise DetectorUnavailable("The Smaug model failed while scoring this image.") from error
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise DetectorUnavailable("The Smaug model returned an invalid score.")
    return score


def reset_model_for_tests() -> None:
    """Reset cached state so model loading can be tested without downloaded weights."""
    global _model, _model_error
    with _load_lock:
        _model = None
        _model_error = None

