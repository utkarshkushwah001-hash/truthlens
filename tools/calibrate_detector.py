#!/usr/bin/env python3
"""Choose a detector threshold on calibration data, then evaluate once on test data.

Both CSV files use label,score and optional generator,transformation columns.
Scores must be in [0, 1], with higher scores indicating more AI-like content.
The test CSV is used only after the threshold has been selected.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from tools.evaluate_detector import evaluate, parse_rows


def candidate_thresholds(rows: list[dict[str, str | float]]) -> list[float]:
    scores = sorted({float(row["score"]) for row in rows})
    candidates = {0.0, 1.0}
    candidates.update(scores)
    for left, right in zip(scores, scores[1:]):
        candidates.add((left + right) / 2.0)
    return sorted(candidates)


def choose_threshold(rows: list[dict[str, str | float]], objective: str = "balanced_accuracy") -> dict:
    if objective not in {"balanced_accuracy", "f1"}:
        raise ValueError("objective must be 'balanced_accuracy' or 'f1'.")
    best = None
    for threshold in candidate_thresholds(rows):
        report = evaluate(rows, threshold)["overall"]
        value = report["balanced_accuracy"] if objective == "balanced_accuracy" else report["precision_ai"]
        if objective == "f1":
            tp = report["true_positive"]
            fp = report["false_positive"]
            fn = report["false_negative"]
            value = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
        # Tie-break toward the higher threshold to avoid unnecessarily labeling
        # borderline scores as AI when calibration performance is identical.
        candidate = (float(value or 0.0), threshold, report)
        if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] > best[1]):
            best = candidate
    assert best is not None
    return {
        "objective": objective,
        "threshold": best[1],
        "calibration_objective_value": best[0],
        "calibration_metrics": best[2],
        "candidate_threshold_count": len(candidate_thresholds(rows)),
    }


def calibrate_and_test(
    calibration_rows: list[dict[str, str | float]],
    test_rows: list[dict[str, str | float]],
    objective: str = "balanced_accuracy",
) -> dict:
    selected = choose_threshold(calibration_rows, objective)
    test_report = evaluate(test_rows, selected["threshold"])
    return {
        "warning": (
            "The threshold was selected using calibration data and then evaluated on the supplied test set. "
            "This estimate is only as independent as the dataset split; keep source images and near-duplicates "
            "out of both splits, and do not repeatedly tune against the test set."
        ),
        "calibration": selected,
        "held_out_test": test_report,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_csv", type=Path)
    parser.add_argument("test_csv", type=Path)
    parser.add_argument("--objective", choices=("balanced_accuracy", "f1"), default="balanced_accuracy")
    parser.add_argument("--output", type=Path, help="Optional path for JSON report")
    args = parser.parse_args(argv)
    try:
        calibration_rows = parse_rows(args.calibration_csv)
        test_rows = parse_rows(args.test_csv)
        report = calibrate_and_test(calibration_rows, test_rows, args.objective)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
