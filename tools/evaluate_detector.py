#!/usr/bin/env python3
"""Evaluate saved AI-image detector scores against a labeled CSV.

CSV columns: label,score. Optional columns: generator,transformation.
Labels are 'real' or 'ai'; score is a numeric detector score where larger
values mean more likely AI-generated. This script evaluates a model; it does
not run inference or make a detector trustworthy by itself.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def parse_rows(path: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"label", "score"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("CSV must contain 'label' and 'score' columns.")
        for line, row in enumerate(reader, start=2):
            if None in row or row.get("label") is None or row.get("score") is None:
                raise ValueError(f"Line {line}: malformed CSV row; expected one label and score value.")
            label = (row.get("label") or "").strip().lower()
            if label not in {"real", "ai"}:
                raise ValueError(f"Line {line}: label must be 'real' or 'ai'.")
            try:
                score = float(row["score"])
            except (TypeError, ValueError):
                raise ValueError(f"Line {line}: score must be a number.") from None
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"Line {line}: score must be finite and between 0 and 1.")
            rows.append({
                "label": label,
                "score": score,
                "generator": (row.get("generator") or "unspecified").strip() or "unspecified",
                "transformation": (row.get("transformation") or "clean").strip() or "clean",
            })
    if not rows:
        raise ValueError("CSV contains no evaluation rows.")
    if not any(row["label"] == "real" for row in rows) or not any(row["label"] == "ai" for row in rows):
        raise ValueError("Evaluation requires at least one real and one AI-labeled sample.")
    return rows


def roc_auc(rows: Iterable[dict[str, str | float]]) -> float | None:
    items = list(rows)
    positives = [float(row["score"]) for row in items if row["label"] == "ai"]
    negatives = [float(row["score"]) for row in items if row["label"] == "real"]
    if not positives or not negatives:
        return None
    # Pairwise definition handles tied scores correctly and avoids dependencies.
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def metrics(rows: list[dict[str, str | float]], threshold: float) -> dict[str, float | int | None]:
    tp = fp = tn = fn = 0
    for row in rows:
        actual_ai = row["label"] == "ai"
        predicted_ai = float(row["score"]) >= threshold
        if actual_ai and predicted_ai:
            tp += 1
        elif actual_ai:
            fn += 1
        elif predicted_ai:
            fp += 1
        else:
            tn += 1
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    recall = sensitivity
    f1 = None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    balanced = (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None
    return {
        "samples": len(rows),
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": (tp + tn) / len(rows) if rows else None,
        "sensitivity_ai_recall": sensitivity,
        "specificity_real_recall": specificity,
        "precision_ai": precision,
        "recall_ai": recall,
        "f1_ai": f1,
        "false_positive_rate": 1 - specificity if specificity is not None else None,
        "balanced_accuracy": balanced,
        "roc_auc": roc_auc(rows),
        "confusion_matrix": {"actual_real": {"predicted_real": tn, "predicted_ai": fp}, "actual_ai": {"predicted_real": fn, "predicted_ai": tp}},
    }


def score_distributions(rows: list[dict[str, str | float]]) -> dict[str, dict[str, float | int | None]]:
    """Summarize raw score ranges by label; these are not calibrated confidence."""
    result: dict[str, dict[str, float | int | None]] = {}
    for label in ("real", "ai"):
        scores = sorted(float(row["score"]) for row in rows if row["label"] == label)
        if not scores:
            result[label] = {"samples": 0, "mean": None, "median": None, "min": None, "max": None}
            continue
        middle = len(scores) // 2
        median = scores[middle] if len(scores) % 2 else (scores[middle - 1] + scores[middle]) / 2
        result[label] = {"samples": len(scores), "mean": sum(scores) / len(scores), "median": median, "min": scores[0], "max": scores[-1]}
    return result


def evaluate(rows: list[dict[str, str | float]], threshold: float) -> dict:
    report: dict = {
        "warning": (
            "Metrics apply only to this labeled sample. They do not guarantee performance "
            "on unseen generators, image domains, or edited/compressed images."
        ),
        "overall": metrics(rows, threshold),
        "score_distributions": score_distributions(rows),
        "by_generator": {},
        "by_transformation": {},
    }
    for field, key in (("generator", "by_generator"), ("transformation", "by_transformation")):
        groups: dict[str, list[dict[str, str | float]]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        report[key] = {name: metrics(group, threshold) for name, group in sorted(groups.items())}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path, help="CSV with label,score and optional generator,transformation columns")
    parser.add_argument("--threshold", type=float, default=0.5, help="AI-score decision threshold (default: 0.5)")
    parser.add_argument("--backend", default="unspecified", help="Detector/backend label recorded in the report")
    parser.add_argument("--split", default="unspecified", help="Evaluation split label recorded in the report")
    parser.add_argument("--output", type=Path, help="Optional path for a reproducible JSON report")
    args = parser.parse_args(argv)
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    try:
        rows = parse_rows(args.csv_file)
        report = evaluate(rows, args.threshold)
        report["evaluation"] = {
            "dataset": str(args.csv_file),
            "backend": args.backend,
            "split": args.split,
            "samples": len(rows),
            "class_balance": {label: sum(row["label"] == label for row in rows) for label in ("real", "ai")},
        }
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

