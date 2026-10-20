#!/usr/bin/env python3
"""Score images listed in a CIFAKE-style manifest without copying image files.

Expected CSV columns: path,label,group_id,official_split.
Labels must be 'real' or 'ai'; official_split values are 'train_pool' or
'official_test'. Scores are raw model outputs and are not calibrated probabilities.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Callable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def read_manifest(manifest: Path, split: str, limit: int | None = None) -> list[dict[str, str]]:
    required = {"path", "label", "group_id", "official_split"}
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Manifest must contain columns: path,label,group_id,official_split")
        for line, raw in enumerate(reader, start=2):
            if raw.get("official_split", "").strip() != split:
                continue
            row = {key: (raw.get(key) or "").strip() for key in required}
            if not row["path"] or not row["label"] or not row["group_id"]:
                raise ValueError(f"Missing path, label, or group_id on CSV line {line}.")
            row["label"] = row["label"].lower()
            if row["label"] not in {"real", "ai"}:
                raise ValueError(f"Invalid label on CSV line {line}: expected real or ai.")
            path = Path(row["path"])
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image extension on CSV line {line}: {path}")
            normalized = str(path.resolve())
            if normalized in seen:
                raise ValueError(f"Duplicate image path in selected split: {normalized}")
            seen.add(normalized)
            row["path"] = normalized
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found for split '{split}' in {manifest}.")
    if {row["label"] for row in rows} != {"real", "ai"}:
        raise ValueError(f"Selected split '{split}' must contain both real and ai labels.")
    if limit is not None and len(rows) > limit:
        # Balance smoke-test samples so a sorted manifest does not select one class only.
        per_label = limit // 2
        selected = [row for label in ("real", "ai") for row in [item for item in rows if item["label"] == label][:per_label]]
        if limit % 2:
            selected.append(next(row for row in rows if row["label"] == "real" and row not in selected))
        rows = selected
    missing = [row["path"] for row in rows if not Path(row["path"]).is_file()]
    if missing:
        raise ValueError(f"{len(missing)} image path(s) do not exist; first missing path: {missing[0]}")
    return rows


def score_rows(rows: list[dict[str, str]], detector: Any) -> list[dict[str, str | float]]:
    results: list[dict[str, str | float]] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        prediction = detector.predict_path(row["path"])
        payload = prediction.as_dict() if hasattr(prediction, "as_dict") else prediction
        if not isinstance(payload, dict) or "probability_ai" not in payload:
            raise ValueError(
                f"Detector result for {row['path']} must contain 'probability_ai'; "
                "inspect the installed detector API instead of guessing a score field."
            )
        try:
            score = float(payload["probability_ai"])
        except (TypeError, ValueError):
            raise ValueError(f"Detector returned a non-numeric probability for {row['path']}.") from None
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Detector returned probability_ai outside [0, 1] for {row['path']}.")
        results.append({
            "path": row["path"],
            "label": row["label"],
            "score": score,
            "group_id": row["group_id"],
            "official_split": row["official_split"],
        })
        if index == 1 or index % 100 == 0 or index == total:
            print(f"Scored {index}/{total}", file=sys.stderr)
    return results


def write_scores(rows: list[dict[str, str | float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "score", "group_id", "official_split"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None, detector_factory: Callable[[str], Any] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Manifest CSV produced by prepare_cifake_manifest.py")
    parser.add_argument("--split", choices=("train_pool", "official_test"), required=True)
    parser.add_argument("--backend", default="univfd", help="aidetector backend name (default: univfd)")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/manifest_predictions.csv"))
    parser.add_argument("--limit", type=int, help="Optional maximum number of selected rows, for a smoke test")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be at least 2 so both classes can be checked")

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
        rows = read_manifest(args.manifest, args.split, args.limit)
        detector = detector_factory(args.backend)
        predictions = score_rows(rows, detector)
        write_scores(predictions, args.output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {len(predictions)} scored images to {args.output}")
    print("Scores are model outputs, not calibrated probabilities or proof of image origin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
