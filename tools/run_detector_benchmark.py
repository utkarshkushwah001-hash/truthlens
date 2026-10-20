#!/usr/bin/env python3
"""Run an optional aidetector backend over a labeled folder and export scores.

Expected layout:
  dataset/
    real/<source>/<transformation>/image.jpg
    ai/<generator>/<transformation>/image.jpg

The first directory under real/ is recorded as the source in the CSV's
'generator' column; use a descriptive source name such as 'camera' or 'stock'.
The first directory under ai/ should identify the generator. Missing grouping
folders are labeled 'unspecified' and 'clean'.

Install the optional detector separately (see docs/DETECTOR_SHORTLIST.md).
This script is an offline evaluation helper; it does not enable inference in
the Flask upload API.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Callable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def iter_labeled_images(root: Path):
    for label in ("real", "ai"):
        label_dir = root / label
        if not label_dir.is_dir():
            raise ValueError(f"Expected directory not found: {label_dir}")
        for path in sorted(label_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = path.relative_to(label_dir)
            generator = relative.parts[0] if len(relative.parts) >= 2 else "unspecified"
            transformation = relative.parts[1] if len(relative.parts) >= 3 else "clean"
            yield label, path, generator, transformation


def score_images(root: Path, detector: Any) -> list[dict[str, str | float]]:
    rows = []
    for label, path, generator, transformation in iter_labeled_images(root):
        result = detector.predict_path(str(path))
        payload = result.as_dict() if hasattr(result, "as_dict") else result
        if not isinstance(payload, dict) or "probability_ai" not in payload:
            raise ValueError(
                f"Detector result for {path} must contain 'probability_ai'; "
                "inspect the installed detector API rather than guessing a score field."
            )
        try:
            score = float(payload["probability_ai"])
        except (TypeError, ValueError):
            raise ValueError(f"Detector returned a non-numeric probability for {path}.") from None
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Detector returned a score outside [0, 1] for {path}.")
        rows.append({
            "label": label,
            "score": score,
            "generator": generator,
            "transformation": transformation,
        })
    if not rows:
        raise ValueError(f"No supported image files found under {root}/real and {root}/ai.")
    if not any(row["label"] == "real" for row in rows) or not any(row["label"] == "ai" for row in rows):
        raise ValueError("Benchmark input must contain at least one real and one AI image.")
    return rows


def write_scores(rows: list[dict[str, str | float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "score", "generator", "transformation"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None, detector_factory: Callable[[str], Any] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Folder containing real/ and ai/ subdirectories")
    parser.add_argument("--backend", default="univfd", help="aidetector backend name (default: univfd)")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/predictions.csv"))
    args = parser.parse_args(argv)

    if detector_factory is None:
        try:
            from aidetector import create_detector
        except ImportError:
            print(
                "Optional detector package is missing. Install a selected backend from "
                "https://github.com/lynote-ai/ai-image-detector and follow its install instructions.",
                file=sys.stderr,
            )
            return 2
        detector_factory = lambda backend: create_detector(backend, device="auto")

    try:
        detector = detector_factory(args.backend)
        rows = score_images(args.dataset_root, detector)
        write_scores(rows, args.output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {len(rows)} scored images to {args.output}")
    print("Scores are model outputs, not calibrated probabilities or proof of image origin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
